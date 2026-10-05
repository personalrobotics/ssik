"""``Chart.margin``: the holdability margin of a branch under joint
limits -- minus the least worst-case violation along the branch, the contact
minimiser of ``in_limits`` as a signed value everywhere -- against a dense scan
of the branch, against ``in_limits`` (positive exactly where an arc of positive
width exists, within the error band of zero at a contact), attained at the ``t``
it reports, continuous in the pose, and the native C++ equal to the Python
reference (ADR-0001: Python is the oracle)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ssik
import ssik._native
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.chart import _chart_margin, charts
from ssik.postprocess import _BAND_CAP
from ssik.solvers.seven_r._minimax import SLIVER, limit_violation
from tests._cpp_backend import _load_ext, cpp_available

FIXTURES = Path(__file__).parent / "fixtures"
_ARMS = {
    "franka_panda": ("panda_link0", "panda_link8", "seven_r.spherical_shoulder"),
    "kuka_iiwa14": ("base", "iiwa_link_ee_kuka", "seven_r.srs"),
}


def _kb(name: str):
    base, ee, _solver = _ARMS[name]
    return load_urdf_kinbody_normalized(FIXTURES / f"{name}.urdf", base, ee)


def _limits(kb):
    return tuple((float(lo), float(hi)) for lo, hi in (j.limits for j in kb.joints))


def _random_q(kb, rng):
    return np.array([rng.uniform(lo, hi) for lo, hi in (j.limits for j in kb.joints)])


def _fk(kb, q):
    from ssik.kinematics.poe_fk import poe_forward_kinematics

    return poe_forward_kinematics(kb, q)


def _tightened(limits, scale: float):
    """``limits`` shrunk about their centres to ``scale`` of their half-widths."""
    lim = np.asarray(limits, dtype=np.float64)
    c, h = lim.mean(axis=1), 0.5 * scale * (lim[:, 1] - lim[:, 0])
    return tuple((float(a), float(b)) for a, b in np.stack([c - h, c + h], axis=1))


def _slack(q, limits) -> float:
    return -float(limit_violation(np.asarray(q), limits)[0])


def _independent_slack(q, limits) -> float:
    """Signed worst-joint slack, each joint wrapped to the turn nearest its box
    centre: written out here rather than through ``limit_violation``."""
    lim = np.asarray(limits, dtype=np.float64)
    c, h = lim.mean(axis=1), 0.5 * (lim[:, 1] - lim[:, 0])
    return float((h - np.abs((np.asarray(q) - c + np.pi) % (2 * np.pi) - np.pi)).min())


@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_margin_matches_a_dense_scan_and_in_limits(arm: str, scale: float) -> None:
    # At the arms' own limits nearly every branch has a minimum within SCAN of a
    # limit; at half the half-widths most are far from one, the regime where the
    # least minimum must still come from the refined grid.
    kb = _kb(arm)
    limits = _tightened(_limits(kb), scale)
    rng = np.random.default_rng(0)
    checked = contacts = 0
    for _ in range(8):
        family = charts(kb, _fk(kb, _random_q(kb, rng)), native=False)
        for chart in family:
            if not chart.domain:
                continue
            margin, t = chart.margin(limits)
            dense = -np.inf
            for lo, hi in chart.domain:
                ts = np.linspace(lo, hi, 4096) if hi > lo else np.array([lo])
                dense = max(dense, float(-limit_violation(chart.q(ts), limits).min()))
            assert margin >= dense - 1e-6, (chart.label, margin, dense)  # at a fold q ~ sqrt(dt)
            assert margin - dense < 2e-3, (chart.label, margin, dense)  # not far above a 4096 scan
            assert abs(_slack(chart.q(t), limits) - margin) < 1e-12  # attained at t
            arcs = chart.in_limits(limits)
            wide = any(b > a for a, b in arcs)
            if abs(margin) > 1e-6:
                assert (margin > 0) == wide, (chart.label, margin, arcs)
            if arcs and not wide:  # a contact: the margin is within the error band of zero
                assert -_BAND_CAP - 1e-9 <= margin <= SLIVER + 1e-9, (chart.label, margin, arcs)
                contacts += 1
            checked += 1
    assert checked > 20


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_margin_is_continuous_in_the_pose(arm: str) -> None:
    kb = _kb(arm)
    rng = np.random.default_rng(1)
    T = _fk(kb, _random_q(kb, rng))
    step = np.eye(4)
    step[0, 3] = 1e-6
    before = {c.label: c.margin()[0] for c in charts(kb, T, native=False)}
    after = {c.label: c.margin()[0] for c in charts(kb, T @ step, native=False)}
    for label in before.keys() & after.keys():
        if np.isfinite(before[label]) and np.isfinite(after[label]):
            assert abs(before[label] - after[label]) < 1e-4, label


def test_point_chart_margin_is_the_postures_slack() -> None:
    # Every UR5e joint spans two turns, which never limits (both sides +inf), so
    # the check uses a box of +-1.5 rad about the chain's own centres.
    m = ssik.Manipulator.from_prebuilt("ur5e")
    kb = m._kb
    centre = np.asarray(_limits(kb)).mean(axis=1)
    limits = np.stack([centre - 1.5, centre + 1.5], axis=1)
    rng = np.random.default_rng(2)
    q0 = centre + rng.uniform(-1.0, 1.0, size=centre.shape)  # inside the box
    margins = []
    for chart in m.self_motion(m.fk(q0)):
        margin, t = chart.margin(limits)
        assert np.isfinite(margin)
        assert abs(margin - _independent_slack(np.reshape(chart.q(t), -1), limits)) < 1e-12
        margins.append(margin)
    # q0's own branch is among them, with q0's slack as its margin
    assert min(abs(mg - _independent_slack(q0, limits)) for mg in margins) < 1e-9
    assert min(margins) < 0


@pytest.mark.parametrize("native", [False, True])
def test_margin_far_from_a_limit_is_the_least_minimum(native: bool) -> None:
    # Every refined grid point of this branch lies above SCAN, and near its
    # singular swivel a joint moves ~0.9 rad between raw grid points, so the
    # lowest *raw* point sits in the wrong basin (-0.3045 at t = -2.08).
    if native and not cpp_available():
        pytest.skip("ssik._ssik_native not built")
    m = ssik.Manipulator.from_prebuilt("openarm_left_ik")
    limits = _tightened(m.joint_limits, 0.5)
    q0 = np.array([1.0474958963304213, -1.3458241248812546, -0.1336426807855837,
                   0.9845989754256299, -0.38493339121963177, -0.5384205332527616,
                   0.20609679524668617])  # fmt: skip
    chart = next(c for c in m.self_motion(m.fk(q0), native=native) if c.label == (0, 1, 1))
    margin, t = chart.margin(limits)
    ts = np.linspace(-np.pi, np.pi, 200_001)
    dense = np.array([_independent_slack(q, limits) for q in chart.q(ts)]).max()
    assert dense - 1e-6 <= margin < dense + 1e-3, (margin, dense)
    assert abs(t - (-0.2952)) < 1e-3
    assert abs(_independent_slack(np.reshape(chart.q(t), -1), limits) - margin) < 1e-12


@pytest.mark.parametrize("native", [False, True])
def test_margin_is_unbounded_when_no_joint_is_limited(native: bool) -> None:
    if native and not cpp_available():
        pytest.skip("ssik._ssik_native not built")
    full_turn = np.array([[-4.0, 4.0]] * 7)
    for arm in ("panda", "iiwa14"):
        m = ssik.Manipulator.from_prebuilt(arm)
        T = m.fk(np.array([0.1, 0.3, -0.2, -1.5, 0.2, 1.6, 0.4]))
        for chart in m.self_motion(T, native=native):
            margin, t = chart.margin(full_turn)
            if not chart.domain:
                assert margin == -np.inf
                assert np.isnan(t)
                continue
            assert margin == np.inf, (arm, chart.label, margin)
            assert chart.contains(t)
            assert chart.in_limits(full_turn)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (np.array([[-1.0, 1.0]] * 6), r"shape \(7, 2\)"),
        (np.array([[-1.0, 1.0, 0.0]] * 7), r"shape \(7, 2\)"),
        (np.array([[0.0, np.nan]] + [[-1.0, 1.0]] * 6), "finite"),
        (np.array([[-np.inf, 1.0]] * 7), "finite"),
        (np.array([[1.0, -1.0]] * 7), "lower <= upper"),
    ],
)
def test_margin_rejects_malformed_limits(native: bool, bad, match: str) -> None:
    if native and not cpp_available():
        pytest.skip("ssik._ssik_native not built")
    m = ssik.Manipulator.from_prebuilt("franka_panda_ik")
    T = m.fk(np.array([0.1, 0.3, -0.2, -1.5, 0.2, 1.6, 0.4]))
    chart = next(c for c in m.self_motion(T, native=native) if c.domain)
    with pytest.raises(ValueError, match=match):
        chart.margin(bad)


def test_margin_rejects_non_real_limits() -> None:
    m = ssik.Manipulator.from_prebuilt("franka_panda_ik")
    T = m.fk(np.array([0.1, 0.3, -0.2, -1.5, 0.2, 1.6, 0.4]))
    chart = next(c for c in m.self_motion(T, native=False) if c.domain)
    with pytest.raises(TypeError):
        chart.margin(np.array([["a", "b"]] * 7))


@pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")
@pytest.mark.parametrize("scale", [1.0, 0.5])
@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_native_margin_matches_the_reference(
    arm: str, scale: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ssik._native, "_ext", _load_ext())
    monkeypatch.setattr(ssik._native, "_ext_tried", True)
    kb = _kb(arm)
    limits = _tightened(_limits(kb), scale)
    rng = np.random.default_rng(3)
    for _ in range(8):
        T = _fk(kb, _random_q(kb, rng))
        theirs = {c.label: c for c in charts(kb, T, native=False)}
        for chart in charts(kb, T, native=True):
            ref = theirs[chart.label]
            if not ref.domain:
                continue
            m_native, t_native = chart.margin(limits)
            m_ref, _t_ref = _chart_margin(ref.q, ref.domain, ref.periodic, limits)
            assert abs(m_native - m_ref) < 1e-9, (chart.label, m_native, m_ref)
            assert abs(_slack(chart.q(t_native), limits) - m_native) < 1e-9
