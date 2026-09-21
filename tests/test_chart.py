"""Self-motion charts (:mod:`ssik.chart`): labels, ``q(t)``, domains, and the
inverse map, on the two closed-form 7R families.

The headline guarantee is the inverse map: for a configuration ``q`` the arm is
actually in, ``charts(fk(q)).locate(q)`` must return a chart that reproduces
``q`` exactly at ``t = param_of(q)``. That is the property a controller needs to
know where on the manifold it is without a nearest-neighbour search.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.chart import Chart, ChartFamily, charts
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.seven_r import spherical_shoulder, srs

FIXTURES = Path(__file__).parent / "fixtures"

_ARMS = {
    "franka_panda": ("panda_link0", "panda_link8", "seven_r.spherical_shoulder", "q6"),
    "kuka_iiwa14": ("base", "iiwa_link_ee_kuka", "seven_r.srs", "swivel"),
}


def _kb(name: str):
    base, ee, _solver, _param = _ARMS[name]
    return load_urdf_kinbody_normalized(FIXTURES / f"{name}.urdf", base, ee)


def _random_q(kb, rng):
    return np.array([rng.uniform(lo, hi) for lo, hi in (j.limits for j in kb.joints)])


def _wrap_dist(a, b) -> float:
    return float(np.max(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)))


def _interior_samples(chart: Chart, n: int = 12):
    """``t`` values strictly inside each domain interval (boundaries are folds)."""
    for lo, hi in chart.domain:
        if chart.periodic and hi - lo >= 2 * np.pi - 1e-9:
            yield from np.linspace(-np.pi, np.pi, n, endpoint=False)
        else:
            yield from np.linspace(lo, hi, n + 2)[1:-1]


@pytest.mark.parametrize("name", list(_ARMS))
def test_dispatch_and_parameter(name: str) -> None:
    kb = _kb(name)
    _base, _ee, solver, param = _ARMS[name]
    arm = ssik.Manipulator(kb)
    assert arm.solver_name == solver
    fam = arm.charts(arm.fk(np.zeros(7)))
    assert isinstance(fam, ChartFamily)
    assert fam.parameter == param
    assert all(c.parameter == param for c in fam)


@pytest.mark.parametrize("name", list(_ARMS))
def test_every_chart_fk_closes_along_its_domain(name: str) -> None:
    """``q(t)`` is an IK solution of the pose at every interior ``t``."""
    kb = _kb(name)
    solver = _ARMS[name][2]
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(25):
        T = poe_forward_kinematics(kb, _random_q(kb, rng))
        fam = charts(kb, T, solver_name=solver)
        assert len(fam) > 0
        for chart in fam:
            ts = np.array(list(_interior_samples(chart)))
            qs = chart.q(ts)
            assert qs.shape == (ts.shape[0], 7)
            assert np.all(np.isfinite(qs))
            for q in qs:
                worst = max(worst, float(np.linalg.norm(poe_forward_kinematics(kb, q) - T)))
    assert worst < 1e-9, f"{name}: worst FK closure along charts {worst:.2e}"


@pytest.mark.parametrize("name", list(_ARMS))
def test_locate_recovers_the_arms_own_configuration(name: str) -> None:
    """The inverse chart map: the configuration the arm is in lies on exactly
    one chart at ``t = param_of(q)``, and that chart reproduces it to machine
    precision. No nearest-neighbour search, no threshold."""
    kb = _kb(name)
    solver = _ARMS[name][2]
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(100):
        q = _random_q(kb, rng)
        fam = charts(kb, poe_forward_kinematics(kb, q), solver_name=solver)
        located = fam.locate(q)
        assert located is not None, f"{name}: own configuration not on any chart"
        chart, t = located
        assert chart.contains(t)
        assert abs(t - fam.param_of(q)) < 1e-12
        worst = max(worst, _wrap_dist(chart.q(t), q))
    assert worst < 1e-9, f"{name}: worst reconstruction of own q {worst:.2e}"


@pytest.mark.parametrize("name", list(_ARMS))
def test_every_solver_branch_lies_on_a_chart(name: str) -> None:
    """The charts cover the solver's own enumeration: each ``solve()`` branch
    locates onto some chart, so labels can be attached to solutions."""
    kb = _kb(name)
    solver = _ARMS[name][2]
    solve = spherical_shoulder.solve if solver.endswith("shoulder") else srs.solve
    rng = np.random.default_rng(13)
    n_sols = 0
    for _ in range(10):
        T = poe_forward_kinematics(kb, _random_q(kb, rng))
        fam = charts(kb, T, solver_name=solver)
        sols, _ = solve(kb, T)
        for sol in sols:
            n_sols += 1
            assert fam.locate(sol.q) is not None, f"{name}: a solve() branch is on no chart"
    assert n_sols > 0


@pytest.mark.parametrize("name", list(_ARMS))
def test_labels_are_unique_and_off_manifold_is_none(name: str) -> None:
    kb = _kb(name)
    solver = _ARMS[name][2]
    rng = np.random.default_rng(17)
    q = _random_q(kb, rng)
    fam = charts(kb, poe_forward_kinematics(kb, q), solver_name=solver)
    labels = fam.labels()
    assert len(set(labels)) == len(labels)
    # A configuration for a *different* pose is on none of these charts.
    assert fam.locate(_random_q(kb, rng)) is None


def test_panda_domains_are_reachable_intervals_of_q6() -> None:
    """Panda charts live on the reachable q6 sub-intervals: every interval is
    ordered and non-empty, ``q(t)`` is finite inside it, and NaN past an
    interior boundary. Shoulder/wrist boundaries are refined to ~1e-10 rad;
    elbow folds are exact but the subproblem gates carry a ~1e-9 feasibility
    slack, so validity extends ~1e-8 rad past them -- probe at 1e-6. Many
    poses, so single-grid-point runs and wrap-around intervals are exercised."""
    kb = _kb("franka_panda")
    rng = np.random.default_rng(19)
    checked = 0
    for _ in range(40):
        q = _random_q(kb, rng)
        fam = charts(kb, poe_forward_kinematics(kb, q), solver_name="seven_r.spherical_shoulder")
        assert not fam.periodic
        for chart in fam:
            assert len(chart.label) == 4
            for lo, hi in chart.domain:
                assert lo <= hi
                assert np.all(np.isfinite(chart.q(0.5 * (lo + hi))))
                if lo > -np.pi + 1e-6:
                    assert np.all(np.isnan(chart.q(lo - 1e-6)))
                    checked += 1
                if hi < np.pi - 1e-6:
                    assert np.all(np.isnan(chart.q(hi + 1e-6)))
                    checked += 1
    assert checked > 0


def test_iiwa_charts_are_eight_full_circles() -> None:
    kb = _kb("kuka_iiwa14")
    rng = np.random.default_rng(23)
    fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng)), solver_name="seven_r.srs")
    assert fam.periodic
    assert len(fam) == 8
    for chart in fam:
        assert chart.domain == ((-np.pi, np.pi),)
        assert chart.contains(3.0)
        assert chart.contains(-3.0)
        assert chart.contains(4.0)  # wrapped onto the circle
    assert {c.label for c in fam} == {(e, s, w) for e in (0, 1) for s in (1, -1) for w in (1, -1)}


def test_curve_is_continuous_within_each_segment() -> None:
    """``curve()`` unwraps along ``t``: within a segment no joint jumps by
    ``2*pi`` between neighbours, and there is one segment per domain interval."""
    for name in _ARMS:
        kb = _kb(name)
        rng = np.random.default_rng(29)
        fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng)), solver_name=_ARMS[name][2])
        for chart in fam:
            segments = chart.curve(360)
            assert len(segments) == len(chart.domain)
            for ts, qs in segments:
                assert ts.shape[0] == qs.shape[0] > 2
                assert np.all(np.isfinite(qs))
                assert np.all(np.diff(ts) > 0)
                # 1 deg spacing in t; a 2*pi representative jump would show as ~6.28.
                assert np.abs(np.diff(qs, axis=0)).max() < np.pi


def test_unsupported_families_are_refused() -> None:
    """Approximate-class (polished) and joint-lock 7R arms have no closed-form
    chart and are refused with a message; a bad pose shape is a ValueError."""
    import glob
    import xml.etree.ElementTree as ET

    fixture = sorted(
        glob.glob(str(FIXTURES / "*xarm7*.urdf")) + glob.glob(str(FIXTURES / "*rizon4*.urdf"))
    )[0]
    root = ET.parse(fixture).getroot()
    links = [str(ln.get("name")) for ln in root.findall("link")]
    base = "link_base" if "link_base" in links else links[0]
    ee = "link7" if "link7" in links else links[-1]
    kb = load_urdf_kinbody_normalized(fixture, base, ee)
    arm = ssik.Manipulator(kb)
    assert arm.solver_name not in (
        "seven_r.spherical_shoulder",
        "seven_r.srs",
        "ikgeo.three_parallel",
    )
    with pytest.raises(NotImplementedError, match="no closed-form chart"):
        arm.charts(arm.fk(np.zeros(7)))
    ur = load_urdf_kinbody_normalized(FIXTURES / "ur5e.urdf", "world", "tool0")
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        charts(ur, np.eye(3), solver_name="ikgeo.three_parallel")


def test_panda_sliver_branch_next_to_a_fold_is_charted() -> None:
    """Regression: a branch that exists only on an arc thinner than a domain
    grid cell (here ~5 mrad, ending at the fold where it meets its partner
    slot) is still charted, so the arm's own configuration on it locates. The
    grid scan alone misses it; the fold-seeded probing catches it."""
    kb = _kb("franka_panda")
    rng = np.random.default_rng(37)
    for _ in range(58):  # the same draw sequence as the native locate test
        _random_q(kb, rng)
        _random_q(kb, rng)
    q = _random_q(kb, rng)
    fam = charts(kb, poe_forward_kinematics(kb, q), solver_name="seven_r.spherical_shoulder")
    located = fam.locate(q)
    assert located is not None
    chart, t = located
    piece = next(d for d in chart.domain if d[0] - 1e-9 <= t <= d[1] + 1e-9)
    assert piece[1] - piece[0] < 0.02  # thinner than one grid cell of the domain scan
    assert _wrap_dist(chart.q(t), q) < 1e-9


@pytest.mark.parametrize("name", list(_ARMS))
def test_tangent_matches_finite_differences(name: str) -> None:
    """``tangent(t)`` (request B2) is ``dq/dt``: ``rate * direction`` agrees with
    an independent first-order central difference of ``q(t)`` to the
    difference's own error, ``direction`` is unit, and both are oriented by
    increasing ``t``. Points within a fold or gimbal lock (rate > 50 rad/rad)
    are excluded: there ``rate`` diverges by construction."""
    kb = _kb(name)
    solver = _ARMS[name][2]
    rng = np.random.default_rng(43)
    worst = 0.0
    n_pts = 0
    for _ in range(12):
        fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng)), solver_name=solver)
        for chart in fam:
            ts = np.array(list(_interior_samples(chart, 10)))
            direction, rate = chart.tangent(ts)
            assert direction.shape == (ts.shape[0], 7)
            assert rate.shape == (ts.shape[0],)
            ok = np.isfinite(rate) & (rate < 50.0)
            if not ok.any():
                continue
            assert np.allclose(np.linalg.norm(direction[ok], axis=1), 1.0, atol=1e-12)
            h = 1e-7
            fd = _wrap_pi_arr(chart.q(ts + h) - chart.q(ts - h)) / (2 * h)
            d = direction[ok] * rate[ok, None]
            worst = max(worst, float(np.max(np.abs(d - fd[ok]) / (1.0 + np.abs(fd[ok])))))
            n_pts += int(ok.sum())
    assert n_pts > 100
    assert worst < 1e-6, f"{name}: tangent vs finite difference {worst:.2e}"


def test_tangent_scalar_and_off_branch() -> None:
    kb = _kb("franka_panda")
    rng = np.random.default_rng(47)
    q = _random_q(kb, rng)
    fam = charts(kb, poe_forward_kinematics(kb, q), solver_name="seven_r.spherical_shoulder")
    located = fam.locate(q)
    assert located is not None
    chart, t = located
    direction, rate = chart.tangent(t)
    assert direction.shape == (7,)
    assert np.isfinite(rate)
    assert abs(np.linalg.norm(direction) - 1.0) < 1e-12
    # off the branch (past the whole reachable set) both are NaN
    lo = min(lo for c in fam for lo, _hi in c.domain)
    if lo > -np.pi + 0.01:
        d_off, r_off = chart.tangent(lo - 0.005)
        assert np.isnan(r_off)
        assert np.all(np.isnan(d_off))


def _wrap_pi_arr(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def test_gimbal_lock_configurations_are_charted_exactly() -> None:
    """Regression: at a gimbal lock of the wrist triple (Panda joint 2 at zero,
    the home keyframe among them) the two closing SP1 calls are 0/0 and the
    SP4 root is a double root that ``arccos`` returns as ~1e-8 instead of 0.
    Untreated, the chart marked a wrong configuration valid (FK residual ~3)
    and ``locate`` failed at the home pose. Now the tangent case is snapped to
    an exact double root and the lock gets one canonical split (q0 = 0)."""
    kb = _kb("franka_panda")
    home = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])
    T = poe_forward_kinematics(kb, home)
    fam = charts(kb, T, solver_name="seven_r.spherical_shoulder")
    located = fam.locate(home)
    assert located is not None
    chart, t = located
    assert _wrap_dist(chart.q(t), home) < 1e-12
    for c in fam:
        q = c.q(t)
        if np.all(np.isfinite(q)):
            assert np.linalg.norm(poe_forward_kinematics(kb, q) - T) < 1e-9
    rng = np.random.default_rng(53)
    for _ in range(20):
        q = _random_q(kb, rng)
        q[1] = 0.0  # lock: only q0 + q2 is determined; the chart carries q0 = 0
        T = poe_forward_kinematics(kb, q)
        canonical = q.copy()
        canonical[0], canonical[2] = 0.0, _wrap_pi_arr(q[0] + q[2])
        fam = charts(kb, T, solver_name="seven_r.spherical_shoulder")
        assert fam.locate(canonical) is not None
        for c in fam:
            qq = c.q(q[6])
            if np.all(np.isfinite(qq)):
                assert np.linalg.norm(poe_forward_kinematics(kb, qq) - T) < 1e-9


@pytest.mark.parametrize("name", list(_ARMS))
def test_in_limits_arcs(name: str) -> None:
    """``in_limits()`` (request A3): the arm's own configuration lies on an arc,
    every interior point of every arc is within limits, and an arc end that is
    not a domain end is a limit crossing (some joint leaves its range just past
    it). Custom limits are honoured."""
    kb = _kb(name)
    solver = _ARMS[name][2]
    lims = np.array([j.limits for j in kb.joints])
    rng = np.random.default_rng(59)
    checked_ends = 0
    for _ in range(20):
        q = _random_q(kb, rng)
        fam = charts(kb, poe_forward_kinematics(kb, q), solver_name=solver)
        located = fam.locate(q)
        assert located is not None
        chart, t = located
        arcs = chart.in_limits()
        assert any(lo - 1e-9 <= t <= hi + 1e-9 for lo, hi in arcs)
        dom_ends = {x for d in chart.domain for x in d}
        for lo, hi in arcs:
            for s in np.linspace(lo, hi, 9)[1:-1]:
                assert _in_range_mod_2pi(chart.q(s), lims)
            for end, side in ((lo, -1.0), (hi, 1.0)):
                if any(abs(end - x) < 1e-9 for x in dom_ends) or abs(abs(end) - np.pi) < 1e-9:
                    continue
                qq = chart.q(end + side * 1e-6)
                if np.all(np.isfinite(qq)):
                    assert not _in_range_mod_2pi(qq, lims)
                    checked_ends += 1
    assert checked_ends > 0
    # tighter custom limits shrink (or empty) the arcs, never grow them
    tight = np.column_stack([lims[:, 0] + 0.3, lims[:, 1] - 0.3])
    assert _total(chart.in_limits(tight)) <= _total(chart.in_limits()) + 1e-12


def _total(arcs) -> float:
    return float(sum(hi - lo for lo, hi in arcs))


def _in_range_mod_2pi(q, lims) -> bool:
    """A joint is in range when some ``q_i + 2*pi*k`` is (the arcs' convention)."""
    c = 0.5 * (lims[:, 0] + lims[:, 1])
    rep = q + 2 * np.pi * np.round((c - q) / (2 * np.pi))
    return bool(np.all(rep >= lims[:, 0] - 1e-9) and np.all(rep <= lims[:, 1] + 1e-9))


def test_frame_is_a_metric_orthogonal_split() -> None:
    """``frame(t, metric)`` (request D1): unit tangent in ``ker(J)``, complement
    orthonormal and ``M``-orthogonal to it, continuous along the arc."""
    from ssik.refinement import kinbody_jacobian

    kb = _kb("franka_panda")
    rng = np.random.default_rng(67)
    q = _random_q(kb, rng)
    fam = charts(kb, poe_forward_kinematics(kb, q), solver_name="seven_r.spherical_shoulder")
    located = fam.locate(q)
    assert located is not None
    chart, t = located
    d, V = chart.frame(t)
    assert np.linalg.norm(kinbody_jacobian(kb, chart.q(t)) @ d) < 1e-9
    assert np.allclose(V.T @ V, np.eye(6), atol=1e-12)
    assert np.abs(d @ V).max() < 1e-12
    M = np.diag([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])
    d2, V2 = chart.frame(t, metric=M)
    assert np.allclose(d2, d)
    assert np.abs(d2 @ M @ V2).max() < 1e-12
    assert np.allclose(V2.T @ V2, np.eye(6), atol=1e-12)
    lo, hi = next(dm for dm in chart.domain if dm[0] - 1e-9 <= t <= dm[1] + 1e-9)
    ts = np.linspace(lo + 0.01, hi - 0.01, 40)
    frames = np.array([chart.frame(s, metric=M)[1] for s in ts])
    assert np.abs(np.diff(frames, axis=0)).max() < 0.5  # no sign flips between neighbours


def test_frame_batches_like_tangent() -> None:
    """``frame`` follows :meth:`Chart.tangent`'s shape contract: scalar ``t`` ->
    ``((7,), (7, 6))``, ``(N,)`` -> ``((N, 7), (N, 7, 6))``, with the batched
    result identical to the scalar one and ``NaN`` rows off the branch."""
    kb = _kb("franka_panda")
    rng = np.random.default_rng(67)
    q = _random_q(kb, rng)
    fam = charts(kb, poe_forward_kinematics(kb, q), solver_name="seven_r.spherical_shoulder")
    located = fam.locate(q)
    assert located is not None
    chart, t = located
    M = np.diag([5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5])

    d, V = chart.frame(t, metric=M)
    assert d.shape == (7,)
    assert V.shape == (7, 6)

    lo, hi = next(dm for dm in chart.domain if dm[0] - 1e-9 <= t <= dm[1] + 1e-9)
    ts = np.linspace(lo + 0.01, hi - 0.01, 40)
    db, Vb = chart.frame(ts, metric=M)
    assert db.shape == (40, 7)
    assert Vb.shape == (40, 7, 6)
    assert np.array_equal(db, np.array([chart.frame(s, metric=M)[0] for s in ts]))
    assert np.array_equal(Vb, np.array([chart.frame(s, metric=M)[1] for s in ts]))

    # Off the branch the whole row is NaN, in batch as for a scalar.
    off = lo - 0.5
    if not chart.contains(off):
        do, Vo = chart.frame(off, metric=M)
        assert np.isnan(do).all()
        assert np.isnan(Vo).all()
        dm_, Vm_ = chart.frame(np.array([off, t]), metric=M)
        assert np.isnan(dm_[0]).all()
        assert np.isnan(Vm_[0]).all()
        assert np.isfinite(dm_[1]).all()
        assert np.isfinite(Vm_[1]).all()


@pytest.mark.parametrize("name", list(_ARMS))
def test_track_follows_a_branch_around_a_loop(name: str) -> None:
    """``track`` (request B4): along a small closed loop in position the branch
    is continued by label lookup and returns to itself. Loops that push the
    fixed coordinate off the reachable range are reported as collisions, not
    hidden; the test picks a pose where the loop stays on the branch."""
    from ssik.chart import track

    kb = _kb(name)
    rng = np.random.default_rng(71)
    for _attempt in range(8):
        q = _random_q(kb, rng)
        T = poe_forward_kinematics(kb, q)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = np.cross(u, rng.normal(size=3))
        v /= np.linalg.norm(v)
        poses = []
        for k in range(41):
            s = 2 * np.pi * k / 40
            P = T.copy()
            P[:3, 3] = T[:3, 3] + 0.01 * ((np.cos(s) - 1) * u + np.sin(s) * v)
            poses.append(P)
        steps = track(kb, poses, q, solver_name=_ARMS[name][2])
        assert len(steps) == 41
        assert steps[0].event == "start"
        if all(st.event == "label" for st in steps[1:]):
            break
    else:
        pytest.fail("no loop stayed on its branch in 8 attempts")
    assert steps[-1].label == steps[0].label
    assert _wrap_dist(steps[-1].q, steps[0].q) < 1e-9
    for st, P in zip(steps, poses, strict=True):
        assert np.linalg.norm(poe_forward_kinematics(kb, st.q) - P) < 1e-9


def test_cuspidality_report_panda_is_noncuspidal() -> None:
    """ssik's lock-7 slicing of the Panda is noncuspidal (Salunkhe et al. lock
    joint 5 and find the other slicing cuspidal): four principal aspects plus
    two thin ones, and no pose with two solutions in one aspect."""
    from ssik.chart import cuspidality_report

    kb = _kb("franka_panda")
    rep = cuspidality_report(kb, locked_angles=(0.5, -1.2), n_poses=40, grid=360)
    assert not rep.cuspidal
    assert all(n >= 4 for n in rep.aspects.values())
    assert all(v == 0 for v in rep.shared_pairs.values())


def test_ur5e_zero_dimensional_charts() -> None:
    """UR-class 6R arms (``ikgeo.three_parallel``): the manifold is a finite
    set of labelled points. Labels are the geometric (shoulder, elbow, wrist)
    signs, distinct across a pose's solutions, and change only across a
    singularity; ``locate`` identifies the branch; ``track`` follows it by label."""
    from ssik.chart import three_parallel_label, track

    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5e.urdf", "world", "tool0")
    arm = ssik.Manipulator(kb)
    assert arm.solver_name == "ikgeo.three_parallel"
    rng = np.random.default_rng(5)
    for _ in range(15):
        q = _random_q(kb, rng)
        T = arm.fk(q)
        fam = arm.charts(T)
        assert fam.dimension == 0
        labels = fam.labels()
        assert len(set(labels)) == len(labels)
        located = fam.locate(q)
        assert located is not None
        chart, t = located
        assert chart.dimension == 0
        assert chart.label == three_parallel_label(kb, q)
        assert _wrap_dist(chart.q(t), q) < 1e-9
        assert _wrap_dist(chart.q(1.234), q) < 1e-9  # t is ignored
        assert chart.in_limits() == ((0.0, 0.0),)
        assert np.isnan(chart.tangent(0.0)[1])
        assert fam.singularity_margin(q) > 0.0
    # label changes only across a singularity along random small steps
    for _ in range(100):
        q = _random_q(kb, rng)
        dq = rng.normal(size=6) * 0.02
        if three_parallel_label(kb, q) != three_parallel_label(kb, q + dq):
            margins = [fam.singularity_margin(q + a * dq) for a in np.linspace(0, 1, 11)]
            assert min(margins) < 0.02
    # track around a small loop: label lookup throughout, closes on itself
    q = _random_q(kb, rng)
    T = arm.fk(q)
    poses = []
    for k in range(31):
        s = 2 * np.pi * k / 30
        P = T.copy()
        P[:3, 3] = T[:3, 3] + 0.01 * np.array([np.cos(s) - 1, np.sin(s), 0.0])
        poses.append(P)
    steps = track(kb, poses, q)
    assert all(st.event == "label" for st in steps[1:])
    assert steps[-1].label == steps[0].label
    assert _wrap_dist(steps[-1].q, steps[0].q) < 1e-9


@pytest.mark.parametrize("fixture", ["franka_panda", "ur5e"])
def test_escape_direction_closes_the_gap(fixture: str) -> None:
    """``escape`` (request D2): along the returned twist the joint gap between
    the two branches' nearest points shrinks to first order, and faster than
    along random twists; ``drift_to_merge`` finds a merge along it when one
    happens within the search range."""
    from ssik.chart import drift_to_merge

    if fixture == "ur5e":
        kb = load_urdf_kinbody_normalized(FIXTURES / "ur5e.urdf", "world", "tool0")
    else:
        kb = _kb("franka_panda")
    rng = np.random.default_rng(83)
    for _ in range(6):
        q = _random_q(kb, rng)
        T = poe_forward_kinematics(kb, q)
        fam = charts(kb, T)
        located = fam.locate(q)
        assert located is not None
        a = located[0]
        sheets = fam.sheets()
        mine = next(sh for sh in sheets if a in sh)
        others = [c for sh in sheets if sh is not mine for c in sh]
        if not others:
            continue
        b = others[0]
        direction, mag, _qa, _qb = fam.escape(a, b)
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-12
        assert np.isfinite(mag)
        gap0 = fam.gap(a, b)[0]

        g_esc = _gap_along(kb, T, a, b, direction)
        if np.isnan(g_esc):
            continue
        assert g_esc < gap0
        g_rand = [_gap_along(kb, T, a, b, r / np.linalg.norm(r)) for r in rng.normal(size=(6, 6))]
        g_rand = [g for g in g_rand if not np.isnan(g)]
        assert g_esc <= np.median(g_rand)
        merged = drift_to_merge(kb, T, a, b, direction, max_drift=1.0, n_steps=40)
        if merged is not None:
            s_merge, T_merge = merged
            assert 0 < s_merge <= 1.5
            assert np.linalg.norm(T_merge[:3, 3] - T[:3, 3]) < 1.5
        break
    else:
        pytest.skip("no pose with two charts")


def _gap_along(kb, T, a, b, twist, s=1e-3):
    from ssik.chart import se3_exp

    f2 = charts(kb, se3_exp(s * twist) @ T)
    ca = f2.by_label(a.label)
    cb = f2.by_label(b.label)
    if ca is None or cb is None:
        return np.nan
    return f2.gap(ca, cb)[0]


def test_sheets_glue_partner_charts_at_folds() -> None:
    """``sheets()`` (request B3): on the Panda the eight slot charts of a pose
    glue into fewer sheets where partner slots meet at a fold; every chart is
    in exactly one sheet, and charts on different sheets have a positive gap."""
    kb = _kb("franka_panda")
    rng = np.random.default_rng(89)
    fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng)))
    sheets = fam.sheets()
    assert sum(len(sh) for sh in sheets) == len(fam)
    assert len(sheets) < len(fam)
    for i, sa in enumerate(sheets):
        for sb in sheets[i + 1 :]:
            assert fam.gap(sa[0], sb[0])[0] > 1e-3
