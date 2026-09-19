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
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    arm = ssik.Manipulator(kb)
    with pytest.raises(NotImplementedError, match="no closed-form chart"):
        arm.charts(arm.fk(np.zeros(6)))
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        charts(kb, np.eye(3), solver_name="seven_r.srs")


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
