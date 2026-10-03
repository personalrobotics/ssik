"""``Chart.margin`` (request A4): the holdability margin of a branch under joint
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


def _slack(q, limits) -> float:
    return -float(limit_violation(np.asarray(q), limits)[0])


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_margin_matches_a_dense_scan_and_in_limits(arm: str) -> None:
    kb = _kb(arm)
    limits = _limits(kb)
    rng = np.random.default_rng(0)
    checked = contacts = 0
    for _ in range(8):
        family = charts(kb, _fk(kb, _random_q(kb, rng)), native=False)
        for chart in family:
            if not chart.domain:
                continue
            margin, t = chart.margin()
            dense = -np.inf
            for lo, hi in chart.domain:
                ts = np.linspace(lo, hi, 4096) if hi > lo else np.array([lo])
                dense = max(dense, float(-limit_violation(chart.q(ts), limits).min()))
            assert margin >= dense - 1e-6, (chart.label, margin, dense)  # at a fold q ~ sqrt(dt)
            assert margin - dense < 2e-3, (chart.label, margin, dense)  # not far above a 4096 scan
            assert abs(_slack(chart.q(t), limits) - margin) < 1e-12  # attained at t
            arcs = chart.in_limits()
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
    m = ssik.Manipulator.from_prebuilt("ur5e")
    kb = m._kb
    limits = _limits(kb)
    rng = np.random.default_rng(2)
    q0 = _random_q(kb, rng)
    for chart in m.self_motion(m.fk(q0)):
        margin, t = chart.margin()
        own = _slack(chart.q(t), limits)
        # every UR5e joint spans a full turn, which never limits: both are +inf
        assert margin == own or abs(margin - own) < 1e-12


@pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")
@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_native_margin_matches_the_reference(arm: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ssik._native, "_ext", _load_ext())
    monkeypatch.setattr(ssik._native, "_ext_tried", True)
    kb = _kb(arm)
    limits = _limits(kb)
    rng = np.random.default_rng(3)
    for _ in range(8):
        T = _fk(kb, _random_q(kb, rng))
        theirs = {c.label: c for c in charts(kb, T, native=False)}
        for chart in charts(kb, T, native=True):
            ref = theirs[chart.label]
            if not ref.domain:
                continue
            m_native, t_native = chart.margin()
            m_ref, _t_ref = _chart_margin(ref.q, ref.domain, ref.periodic, limits)
            assert abs(m_native - m_ref) < 1e-9, (chart.label, m_native, m_ref)
            assert abs(_slack(chart.q(t_native), limits) - m_native) < 1e-9
