"""Arc-length sampling of a chart (request C1) and the batched path solve
(request C2): :meth:`Chart.length`, :meth:`Chart.sample`,
:func:`ssik.chart.track_all` and :meth:`Manipulator.solve_path`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.chart import Chart, PathTrack, charts, track, track_all
from ssik.kinematics.poe_fk import poe_forward_kinematics

FIXTURES = Path(__file__).parent / "fixtures"

_ARMS = {
    "franka_panda": ("panda_link0", "panda_link8"),
    "kuka_iiwa14": ("base", "iiwa_link_ee_kuka"),
}


def _kb(name: str):
    return load_urdf_kinbody_normalized(FIXTURES / f"{name}.urdf", *_ARMS[name])


def _random_q(kb, rng):
    return np.array([rng.uniform(lo, hi) for lo, hi in (j.limits for j in kb.joints)])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _steps(qs, M=None):
    d = _wrap(np.diff(qs, axis=0))
    if M is None:
        return np.linalg.norm(d, axis=1)
    return np.sqrt(np.einsum("ni,ij,nj->n", d, M, d))


def _cv(x):
    return float(x.std() / x.mean())


def _panda_chart_with_folds(rng) -> Chart:
    """A Panda chart whose longest piece ends in folds, so ``|dq/dt|`` diverges
    at both ends and a ``t``-grid is far from uniform on it."""
    kb = _kb("franka_panda")
    for _ in range(20):
        fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng)))
        for c in fam.charts:
            lo, hi = max(c.domain, key=lambda d: d[1] - d[0])
            if hi - lo < 2 * np.pi - 1e-6 and hi - lo > 0.5:
                return c
    pytest.fail("no Panda chart with a folded piece in 20 poses")


def _longest(segments):
    return max(segments, key=lambda s: s[0].shape[0])


# --------------------------------------------------------------------------
# C1: arc-length sampling
# --------------------------------------------------------------------------


def test_sample_is_uniform_in_arc_length_where_t_is_not() -> None:
    chart = _panda_chart_with_folds(np.random.default_rng(1))
    ts, qs = _longest(chart.sample(200))
    lo, hi = max(chart.domain, key=lambda d: d[1] - d[0])
    qu = chart.q(np.linspace(lo, hi, ts.shape[0]))
    qu = qu[np.all(np.isfinite(qu), axis=1)]
    # The failing case first: a uniform t-grid is badly non-uniform on this piece,
    # so a small spread below is the sampler's doing, not the chart's.
    assert _cv(_steps(qu)) > 0.3
    assert _cv(_steps(qs)) < 1e-3
    assert np.all(np.diff(ts) > 0)


def test_samples_lie_on_the_chart() -> None:
    kb = _kb("franka_panda")
    rng = np.random.default_rng(2)
    T = poe_forward_kinematics(kb, _random_q(kb, rng))
    for c in charts(kb, T).charts:
        for ts, qs in c.sample(60):
            assert np.allclose(_wrap(qs - c.q(ts)), 0.0, atol=1e-12)
            for q in qs:
                # Near a fold q(t) closes FK only to ~1e-7 (see Chart.q).
                assert np.abs(poe_forward_kinematics(kb, q) - T).max() < 1e-7


def test_length_matches_the_integral_of_the_tangent_rate() -> None:
    """Independent check on a fold-free chart: the polyline length equals
    ``integral |dq/dt| dt`` computed from :meth:`Chart.tangent`."""
    kb = _kb("kuka_iiwa14")
    rng = np.random.default_rng(3)
    c = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng))).charts[0]
    t = np.linspace(-np.pi, np.pi, 20001)
    _, rate = c.tangent(t)
    integral = float(np.sum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t)))
    assert c.length() == pytest.approx(integral, rel=1e-6)


def test_length_converges_through_folds() -> None:
    """Independent check through folds, where ``|dq/dt|`` is not integrable in
    closed form: a dense chord sum on a cosine-clustered ``t``-grid. Near a fold
    ``q ~ sqrt(t - t_0)`` and the grid spacing goes like ``u^2``, so the chords
    are near-uniform in ``q`` there without knowing where the folds are."""
    chart = _panda_chart_with_folds(np.random.default_rng(4))
    ref = 0.0
    for lo, hi in chart.domain:
        u = np.linspace(0.0, 1.0, 400001)
        qs = chart.q(lo + (hi - lo) * 0.5 * (1.0 - np.cos(np.pi * u)))
        qs = qs[np.all(np.isfinite(qs), axis=1)]
        ref += float(_steps(qs).sum())
    assert chart.length() == pytest.approx(ref, rel=1e-6)
    assert chart.length(tol=1e-8) == pytest.approx(ref, rel=1e-6)


def test_metric_scales_and_weights_the_spacing() -> None:
    chart = _panda_chart_with_folds(np.random.default_rng(5))
    assert chart.length(4.0 * np.eye(7)) == pytest.approx(2.0 * chart.length(), rel=1e-12)

    A = np.diag(np.linspace(0.3, 3.0, 7))
    batched = lambda qs: np.broadcast_to(A, (qs.shape[0], 7, 7))  # noqa: E731
    assert chart.length(batched) == pytest.approx(chart.length(A), rel=1e-12)

    _, qs = _longest(chart.sample(150, A))
    assert _cv(_steps(qs, A)) < 1e-3
    # ...and it is the metric doing it: the same points are not uniform in the
    # Euclidean distance.
    assert _cv(_steps(qs)) > 10 * _cv(_steps(qs, A))

    with pytest.raises(ValueError, match="dof"):
        chart.length(lambda qs: np.eye(7))


def test_closed_chart_spaces_the_seam_like_the_rest() -> None:
    kb = _kb("kuka_iiwa14")
    c = charts(kb, poe_forward_kinematics(kb, _random_q(kb, np.random.default_rng(6)))).charts[0]
    ((ts, qs),) = c.sample(100)
    assert ts.shape[0] == 100
    steps = _steps(np.vstack([qs, qs[:1]]))
    assert _cv(steps) < 1e-3  # the seam step included
    # A chord sum falls short of the arc by O(h^2); dense enough, it meets it.
    ((_, dense),) = c.sample(4000)
    assert _steps(np.vstack([dense, dense[:1]])).sum() == pytest.approx(c.length(), rel=1e-6)


def test_limits_restrict_and_join_across_the_seam() -> None:
    kb = _kb("kuka_iiwa14")
    rng = np.random.default_rng(7)
    lims = np.array([j.limits for j in kb.joints])
    for _ in range(30):
        c = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng))).charts[0]
        arcs = c.in_limits()
        if len(arcs) >= 2 and arcs[0][0] == -np.pi and arcs[-1][1] == np.pi:
            break
    else:
        pytest.fail("no iiwa chart with an in-limits arc across the seam")
    segs = c.sample(200, limits=True)
    assert len(segs) == len(arcs) - 1  # the two seam pieces are one
    for _, qs in segs:
        # In limits modulo 2*pi, as Chart.in_limits defines it. The piece ends are
        # in_limits' own bisected boundaries, which overshoot by a few 1e-9 rad.
        k = np.round((lims.mean(axis=1) - qs) / (2 * np.pi))
        qk = qs + 2 * np.pi * k
        assert np.all(qk >= lims[:, 0] - 1e-8)
        assert np.all(qk <= lims[:, 1] + 1e-8)
    assert c.length(limits=True) < c.length()
    assert c.length(limits=lims) == pytest.approx(c.length(limits=True), rel=1e-12)


def test_zero_dimensional_chart() -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5e.urdf", "world", "tool0")
    rng = np.random.default_rng(8)
    fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng)))
    c = fam.charts[0]
    assert c.length() == 0.0
    ((_, qs),) = c.sample(10)
    assert qs.shape == (1, 6)
    assert np.allclose(qs[0], c.q(0.0))
    with pytest.raises(ValueError, match="zero-dimensional"):
        c.pullback_metric(0.0)


def _position_metric(qs):
    """An SPD metric that varies with the posture, like a mass matrix."""
    w = 1.0 + np.sin(qs) ** 2  # (N, 7), in [1, 2]
    B = np.eye(7) + 0.1 * np.ones((7, 7))
    return w[:, :, None] * B[None] * w[:, None, :]


def test_pullback_metric_integrates_to_the_length() -> None:
    """``integral sqrt(g(t)) dt`` is the length :meth:`Chart.length` sums by
    chords, on a fold-free chart and in a posture-dependent metric."""
    kb = _kb("kuka_iiwa14")
    c = charts(kb, poe_forward_kinematics(kb, _random_q(kb, np.random.default_rng(3)))).charts[0]
    t = np.linspace(-np.pi, np.pi, 20001)

    def integral(metric):
        s = np.sqrt(c.pullback_metric(t, metric))
        return float(np.sum(0.5 * (s[1:] + s[:-1]) * np.diff(t)))

    L = c.length(_position_metric)
    # The metric is doing something: the weighted length is not the Euclidean one.
    assert c.length() * 1.5 < L
    assert integral(_position_metric) == pytest.approx(L, rel=1e-6)
    assert integral(None) == pytest.approx(c.length(), rel=1e-6)


def test_pullback_metric_is_the_metric_of_dq_dt() -> None:
    """Pointwise: ``g(t) = q'(t)^T G(q(t)) q'(t)`` against a central difference
    of ``q``, and it diverges at a fold, where the rate does."""
    chart = _panda_chart_with_folds(np.random.default_rng(1))
    lo, hi = max(chart.domain, key=lambda d: d[1] - d[0])
    t = np.linspace(lo, hi, 41)[5:-5]  # clear of the folds
    h = 1e-6
    dq = _wrap(chart.q(t + h) - chart.q(t - h)) / (2 * h)
    G = _position_metric(chart.q(t))
    fd = np.einsum("ni,nij,nj->n", dq, G, dq)
    assert chart.pullback_metric(t, _position_metric) == pytest.approx(fd, rel=1e-5)
    _, rate = chart.tangent(t)
    assert chart.pullback_metric(t, None) == pytest.approx(rate**2, rel=1e-12)
    assert np.shape(chart.pullback_metric(float(t[0]))) == ()

    # At a fold q ~ sqrt(t - t_0), so g ~ 1/(t - t_0): a hundredfold per two
    # decades of approach, at both ends of the piece.
    far = chart.pullback_metric(np.array([lo + 1e-3, hi - 1e-3]))
    near = chart.pullback_metric(np.array([lo + 1e-5, hi - 1e-5]))
    assert near / far == pytest.approx([100.0, 100.0], rel=0.01)


# --------------------------------------------------------------------------
# C2: batched path solve
# --------------------------------------------------------------------------


def _loop(T, rng, radius=0.01, n=41):
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    v = np.cross(u, rng.normal(size=3))
    v /= np.linalg.norm(v)
    poses = np.repeat(T[None], n, axis=0)
    s = 2 * np.pi * np.arange(n) / (n - 1)
    poses[:, :3, 3] += radius * ((np.cos(s) - 1)[:, None] * u + np.sin(s)[:, None] * v)
    return poses


@pytest.mark.parametrize("name", list(_ARMS))
def test_solve_path_tracks_every_branch_around_a_loop(name: str) -> None:
    kb = _kb(name)
    arm = ssik.Manipulator(kb)
    rng = np.random.default_rng(11)
    for _ in range(8):
        q = _random_q(kb, rng)
        poses = _loop(poe_forward_kinematics(kb, q), rng)
        path = arm.solve_path(poses)
        if not path.events():
            break
    else:
        pytest.fail("no loop stayed on its branches in 8 attempts")
    assert isinstance(path, PathTrack)
    assert path.closed
    assert len(path.tracks) == len(charts(kb, poses[0]).charts)
    assert path.permutation == {label: label for label in path.tracks}
    assert path.is_bijection()
    for label, steps in path.tracks.items():
        assert len(steps) == poses.shape[0]
        assert steps[0].event == "start"
        qs = path.q(label)
        for qi, P in zip(qs, poses, strict=True):
            assert np.abs(poe_forward_kinematics(kb, qi) - P).max() < 1e-9
        assert np.max(np.abs(_wrap(qs[-1] - qs[0]))) < 1e-9


def test_solve_path_from_q0_is_track() -> None:
    kb = _kb("franka_panda")
    rng = np.random.default_rng(12)
    q = _random_q(kb, rng)
    poses = _loop(poe_forward_kinematics(kb, q), rng)
    path = ssik.Manipulator(kb).solve_path(poses, q0=q)
    assert len(path.tracks) == 1
    ref = track(kb, poses, q)
    (steps,) = path.tracks.values()
    assert [s.label for s in steps] == [s.label for s in ref]
    assert np.allclose(np.stack([s.q for s in steps]), np.stack([s.q for s in ref]))
    assert np.allclose(steps[0].q, q)


def test_tracking_stops_outside_the_workspace() -> None:
    """A pose with no manifold ends every track there; the reachable poses after
    it are not picked up again, and the closed path gets no permutation."""
    kb = _kb("kuka_iiwa14")
    q = _random_q(kb, np.random.default_rng(13))
    T = poe_forward_kinematics(kb, q)
    far = T.copy()
    far[:3, 3] *= 10.0  # far outside the workspace
    poses = np.stack([T, T, far, T, T])
    path = track_all(kb, poses)
    assert path.closed
    assert path.stopped_at == 2
    assert path.permutation is None
    assert not path.is_bijection()
    assert len(path.tracks) == 8
    for label, steps in path.tracks.items():
        assert [s.event for s in steps] == ["start", "label", "unreachable"]
        assert np.all(np.isnan(steps[-1].q))
        assert path.q(label).shape == (3, 7)
    single = track(kb, poses, q)
    assert [s.event for s in single] == ["start", "label", "unreachable"]


def test_open_path_has_no_permutation() -> None:
    kb = _kb("kuka_iiwa14")
    q = _random_q(kb, np.random.default_rng(14))
    poses = _loop(poe_forward_kinematics(kb, q), np.random.default_rng(15))[:20]
    path = track_all(kb, poses)
    assert not path.closed
    assert path.stopped_at is None
    assert path.permutation is None
    assert not path.is_bijection()
    with pytest.raises(ValueError, match="poses"):
        track_all(kb, poses[0])


# --------------------------------------------------------------------------
# In-box gap and drift_to_merge
# --------------------------------------------------------------------------


def _in_box(q, box, tol=1e-8) -> bool:
    return bool(np.all(q >= box[:, 0] - tol) and np.all(q <= box[:, 1] + tol))


def test_inbox_distance_follows_the_box_width() -> None:
    """Narrow box: only one winding is holdable, so points a turn apart are one
    posture and a point outside the box is no posture at all (wrap-to-pi says
    0.48). Wide box: every winding is holdable and the nearest pair counts."""
    from ssik.chart import _inbox_dist2

    narrow = np.array([[-2.9, 2.9]] * 7)
    wide = np.array([[-2 * np.pi, 2 * np.pi]] * 7)
    qa = np.zeros(7)
    qa[0] = 2.8
    qb = qa.copy()
    qb[0] = 2.8 - 2 * np.pi
    for box in (narrow, wide):
        d2, ra, rb = _inbox_dist2(qa[None], qb[None], box)
        assert d2[0] == pytest.approx(0.0, abs=1e-24)
        assert _in_box(ra, box, 0.0)
    qb[0] = -3.0  # holdable only as 3.283 -- outside the narrow box
    assert np.isinf(_inbox_dist2(qa[None], qb[None], narrow)[0][0])
    d2, ra, rb = _inbox_dist2(qa[None], qb[None], wide)
    assert np.sqrt(d2[0]) == pytest.approx(2 * np.pi - 5.8)
    assert np.linalg.norm(rb - ra) == pytest.approx(np.sqrt(d2[0]))


def test_gap_in_the_box_returns_holdable_representatives() -> None:
    kb = _kb("franka_panda")
    lims = np.array([j.limits for j in kb.joints])
    fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, np.random.default_rng(0))))
    cs = fam.charts
    seen_larger = False
    for a, b in [(cs[0], cs[1]), (cs[0], cs[2]), (cs[1], cs[3])]:
        g_torus, _, _ = fam.gap(a, b)
        g, ra, rb = fam.gap(a, b, limits=True)
        assert g >= g_torus - 1e-9  # the box only removes candidates
        seen_larger |= g > g_torus + 1e-3
        assert _in_box(ra, lims)
        assert _in_box(rb, lims)
        assert np.linalg.norm(rb - ra) == pytest.approx(g, abs=1e-12)
        assert np.abs(poe_forward_kinematics(kb, ra) - fam._T).max() < 1e-7
    # The failing case: at this pose the torus and the box disagree.
    assert seen_larger
    assert fam.gap(cs[0], cs[1], limits=lims)[0] == pytest.approx(
        fam.gap(cs[0], cs[1], limits=True)[0]
    )


def test_drift_to_merge_finds_the_touch_and_only_a_touch() -> None:
    """iiwa14, the sheet pair of example 02: the reported merge is a real touch
    (gap ~0 there, back near 0.1 rad a hundredth of drift away) at the drift an
    independent dense scan finds, and a pair that stays apart reports why."""
    from ssik.chart import drift_to_merge, se3_exp

    kb = _kb("kuka_iiwa14")
    rng = np.random.default_rng(1)
    cases: list[tuple[np.ndarray, Chart, Chart, np.ndarray]] = []
    while len(cases) < 3:
        T = poe_forward_kinematics(kb, _random_q(kb, rng))
        fam = charts(kb, T)
        sh = fam.sheets()
        if len(sh) >= 2:
            a, b = sh[0][0], sh[1][0]
            xi = fam.escape(a, b)
            d = np.asarray(xi[0] if isinstance(xi, tuple) else xi, dtype=np.float64)
            cases.append((T, a, b, d / np.linalg.norm(d)))

    T, a, b, d = cases[2]
    found, why = drift_to_merge(kb, T, a, b, d, explain=True)
    assert found is not None, why
    s_star = found[0]
    assert s_star == pytest.approx(0.32631364, abs=1e-6)  # dense scan + golden section

    def gap_at(s):
        f = charts(kb, se3_exp(s * d) @ T)
        xa, xb = f.by_label(a.label), f.by_label(b.label)
        assert xa is not None
        assert xb is not None
        return f.gap(xa, xb)[0]

    assert gap_at(s_star) < 1e-3
    assert gap_at(s_star - 0.01) > 0.1

    T, a, b, d = cases[1]
    found, why = drift_to_merge(kb, T, a, b, d, explain=True)
    assert found is None
    assert why.startswith("no merge")
    assert "0.91" in why  # the closest approach, reported


def test_feasible_arc_survives_a_flip_at_its_midpoint() -> None:
    """At a wrist gimbal lock a chart's coordinate flips by pi at one point. When
    that point is exactly the midpoint of a feasible arc, a midpoint test threw the
    whole arc away (an iiwa14 joint in range over 92% of the circle came back never
    in range). The grid samples inside the arc decide it instead."""
    from ssik.solvers.seven_r._feasible_param import PARAM_GRID, arcs_for_joint

    lo, hi = -2.9, 2.9
    bad = (-2.6, -2.4)  # a genuine out-of-range stretch: q = 3.0 there

    def smooth(t):
        return 3.0 if bad[0] < t < bad[1] else 0.1

    # The long feasible arc runs from -2.4 round to -2.6 + 2pi; its midpoint:
    mid = float(((bad[1] + bad[0] + 2 * np.pi) / 2 + np.pi) % (2 * np.pi) - np.pi)

    def flipped(t):
        # 1e-9 wide: far below the grid step, wide enough to hold the arc's
        # midpoint, which the bisected boundaries place within ~1e-12 of `mid`.
        return 0.1 + np.pi if abs(t - mid) < 1e-9 else smooth(t)  # -3.04: out of range

    q_col = np.array([smooth(t) for t in PARAM_GRID])
    for q_of in (smooth, flipped):
        arcs = arcs_for_joint(q_of, lo, hi, PARAM_GRID, q_col)
        covered = sum(b - a for a, b in arcs)
        assert covered == pytest.approx(2 * np.pi - 0.2, abs=1e-6), (q_of.__name__, arcs)


def test_track_all_from_several_starts() -> None:
    kb = _kb("kuka_iiwa14")
    q = _random_q(kb, np.random.default_rng(21))
    poses = _loop(poe_forward_kinematics(kb, q), np.random.default_rng(22))
    fam = charts(kb, poses[0])
    starts = np.stack([c.q(0.3) for c in fam.charts[:3]])
    path = track_all(kb, poses, q0=starts)
    assert list(path.tracks) == [c.label for c in fam.charts[:3]]
    for k, label in enumerate(path.tracks):
        single = track(kb, poses, starts[k])
        assert np.allclose(path.q(label), np.stack([st.q for st in single]))
    with pytest.raises(ValueError, match="one chart"):
        track_all(kb, poses, q0=np.stack([starts[0], starts[0]]))


def test_seeded_solve_capped_to_one_returns_the_nearest() -> None:
    """The tracking idiom `solve(T, q_seed=q, max_solutions=1, respect_limits=False)`
    on a solver that takes no seed (SRS): the cap used to reach the solver before
    the seed ranking, returning its first branch instead of the nearest."""
    kb = _kb("kuka_iiwa14")
    arm = ssik.Manipulator(kb)
    rng = np.random.default_rng(31)
    for _ in range(5):
        q = _random_q(kb, rng)
        T = poe_forward_kinematics(kb, q)
        seed = q + rng.uniform(-0.05, 0.05, 7)
        full = arm.solve(T, respect_limits=False)
        nearest = min(float(np.max(np.abs(_wrap(s.q - seed)))) for s in full)
        (top,) = arm.solve(T, q_seed=seed, max_solutions=1, respect_limits=False)
        assert float(np.max(np.abs(_wrap(top.q - seed)))) == pytest.approx(nearest, abs=1e-9)


def _box_margin(qs, lims):
    """Signed distance inside the joint box, modulo 2*pi (``>= 0`` in limits)."""
    box = np.asarray(lims, dtype=np.float64)
    c, h = box.mean(axis=1), 0.5 * (box[:, 1] - box[:, 0])
    m = np.min(h - np.abs(_wrap(qs - c)), axis=-1)
    return np.where(np.all(np.isfinite(qs), axis=-1), m, -np.inf)


def test_in_limits_catches_an_excursion_between_grid_points() -> None:
    """A joint that leaves its range and returns between two points of the
    bracketing grid. Checked in the shared arc finder (the chart's in_limits and
    the solver's in-limits resolver both call it) on a synthetic bump 0.24 past
    the stop and 0.004 wide, then on the Panda branch it was found on (0.243 rad
    past a stop over 0.01 of t)."""
    from ssik.solvers.seven_r._feasible_param import PARAM_GRID, feasible_arcs

    def q_scalar(t):
        bump = 1.24 * np.exp(-(((t - 0.3) / 0.002) ** 2))
        return np.array([0.5 * np.sin(t) + bump])

    q_grid = np.array([q_scalar(t) for t in PARAM_GRID])
    arcs = feasible_arcs(q_scalar, q_grid, (0,), [(-1.0, 1.0)], PARAM_GRID)
    t = np.linspace(-np.pi, np.pi, 400001, endpoint=False)
    inside = np.array([abs(q_scalar(x)[0]) <= 1.0 for x in t])
    covered = np.zeros_like(inside)
    for lo, hi in arcs:
        covered |= (t >= lo - 1e-9) & (t <= hi + 1e-9)
    assert not np.any(covered & ~inside), arcs  # nothing out of range is kept
    assert not np.any(inside & ~covered), arcs  # nothing in range is lost

    kb = _kb("franka_panda")
    lims_panda = tuple((float(lo), float(hi)) for lo, hi in (j.limits for j in kb.joints))
    rng = np.random.default_rng(0)
    for _ in range(26):
        _random_q(kb, rng)
    fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng)))
    chart = fam.by_label((0, 1, 1, 0))
    assert chart is not None
    for lo, hi in chart.in_limits():
        t = np.linspace(lo, hi, 20001)
        assert _box_margin(chart.q(t), lims_panda).min() > -1e-6


def test_drift_to_merge_follows_sheets_on_the_panda() -> None:
    """A Panda sheet is several charts glued at folds, and the nearest point of
    a gap sits on a fold, where either chart can claim it; the heads follow
    sheets, not charts. The touch at drift 0.19960583 is real -- a dense scan
    brings the two branches within 3e-4 rad of each other there, the gap a
    square-root cusp ~1e-8 wide in drift -- and must be found on every BLAS
    kernel: ``gap`` once refined from only the argmin of a grid with a mirror
    tie, so on some kernels it stalled at 0.109 and walked past this touch to
    one at 0.848 (py3.10-3.13 in CI, OPENBLAS_CORETYPE locally)."""
    from ssik.chart import drift_to_merge, se3_exp

    kb = _kb("franka_panda")
    rng = np.random.default_rng(2)
    cases: list[tuple[np.ndarray, Chart, Chart, np.ndarray]] = []
    while len(cases) < 4:
        T = poe_forward_kinematics(kb, _random_q(kb, rng))
        fam = charts(kb, T)
        sh = fam.sheets()
        if len(sh) >= 2:
            a, b = sh[0][0], sh[1][0]
            xi = fam.escape(a, b)
            d = np.asarray(xi[0] if isinstance(xi, tuple) else xi, dtype=np.float64)
            cases.append((T, a, b, d / np.linalg.norm(d)))

    T, a, b, d = cases[0]
    found, why = drift_to_merge(kb, T, a, b, d, explain=True)
    assert found is not None, why
    assert found[0] == pytest.approx(0.00878219, abs=1e-6)

    T, a, b, d = cases[3]
    found, why = drift_to_merge(kb, T, a, b, d, explain=True)
    assert found is not None, why
    assert found[0] == pytest.approx(0.19960583, abs=1e-6)

    def chart_gaps(s):
        f = charts(kb, se3_exp(s * d) @ T)
        xa, xb = f.by_label(a.label), f.by_label(b.label)
        assert xa is not None
        assert xb is not None
        return f.gap(xa, xb)[0], f.gap(xb, xa)[0]

    # Both argument orders: swapping them transposes gap's grid, which moves
    # a mirror tie to the other side of argmin.
    assert max(chart_gaps(found[0])) < 5e-3
    assert min(chart_gaps(found[0] - 0.01)) > 0.5
