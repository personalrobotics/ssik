"""The native (C++) self-motion charts match the Python reference (ADR-0001:
Python is the oracle; ``cpp/include/ssik_cpp/chart.hpp`` must agree with
``ssik/chart.py``): same labels in the same order, domains equal to the
bisection tolerance, ``q(t)`` equal to floating-point precision, and
``locate`` picking the same chart at the same coordinate.

Skips when the test-only extension isn't built (the cpp CI job builds it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ssik
import ssik._native
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.chart import charts
from ssik.kinematics.poe_fk import poe_forward_kinematics
from tests._cpp_backend import _load_ext, cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")

FIXTURES = Path(__file__).parent / "fixtures"
_ARMS = {
    "franka_panda": ("panda_link0", "panda_link8", "seven_r.spherical_shoulder"),
    "kuka_iiwa14": ("base", "iiwa_link_ee_kuka", "seven_r.srs"),
}


@pytest.fixture(autouse=True)
def _use_built_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``ssik._native`` at the test-only build under cpp/build when the
    wheel's extension is not installed (the same module either way)."""
    monkeypatch.setattr(ssik._native, "_ext", _load_ext())
    monkeypatch.setattr(ssik._native, "_ext_tried", True)


def _kb(name: str):
    base, ee, _solver = _ARMS[name]
    return load_urdf_kinbody_normalized(FIXTURES / f"{name}.urdf", base, ee)


def _random_q(kb, rng):
    return np.array([rng.uniform(lo, hi) for lo, hi in (j.limits for j in kb.joints)])


def _wrap_dist(a, b) -> float:
    return float(np.max(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)))


def _max_ulp(a, b) -> int:
    """Largest distance in representable doubles between two arrays, ignoring
    positions where both are NaN. 0 means bit-identical."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    ia, ib = a.view(np.int64).copy(), b.view(np.int64).copy()
    # Map the sign-magnitude layout onto a monotone integer line.
    ia[ia < 0] = np.int64(np.iinfo(np.int64).min) - ia[ia < 0]
    ib[ib < 0] = np.int64(np.iinfo(np.int64).min) - ib[ib < 0]
    diff = np.abs(ia - ib)
    diff[both_nan] = 0
    return int(diff.max()) if diff.size else 0


def _raw_tangents(chart, pad: float = 0.05, n: int = 120):
    """The chart's raw (un-normalized) ``dq/dt`` sampled over its first domain
    interval, reaching past both ends so the sweep carries NaN rows too."""
    lo, hi = chart.domain[0]
    assert chart._deriv is not None
    ts = np.linspace(lo - pad, hi + pad, n)
    return np.ascontiguousarray(chart._deriv(ts), dtype=np.float64)


@pytest.mark.parametrize("name", list(_ARMS))
def test_native_matches_python_reference(name: str) -> None:
    kb = _kb(name)
    solver = _ARMS[name][2]
    rng = np.random.default_rng(31)
    for _ in range(25):
        q = _random_q(kb, rng)
        T = poe_forward_kinematics(kb, q)
        nat = charts(kb, T, solver_name=solver, native=True)
        ref = charts(kb, T, solver_name=solver, native=False)
        assert nat.native
        assert not ref.native
        assert nat.parameter == ref.parameter
        assert nat.labels() == ref.labels()
        for cn, cr in zip(nat.charts, ref.charts, strict=True):
            assert len(cn.domain) == len(cr.domain)
            assert np.allclose(np.array(cn.domain), np.array(cr.domain), atol=1e-8)
            for lo, hi in cr.domain:
                ts = np.linspace(lo, hi, 9)[1:-1]
                qn, qr = cn.q(ts), cr.q(ts)
                assert np.all(np.isfinite(qn))
                assert _wrap_dist(qn, qr) < 1e-9
            direction_n, rate_n = cn.tangent(ts)
            direction_r, rate_r = cr.tangent(ts)
            ok = np.isfinite(rate_n) & np.isfinite(rate_r) & (rate_r < 50.0)
            if ok.any():
                dn, dr = direction_n[ok] * rate_n[ok, None], direction_r[ok] * rate_r[ok, None]
                assert np.max(np.abs(dn - dr) / (1.0 + np.abs(dr))) < 1e-7
            an, ar = np.array(cn.in_limits()), np.array(cr.in_limits())
            assert an.shape == ar.shape
            if an.size:
                assert np.max(np.abs(an - ar)) < 1e-8
        ln, lr = nat.locate(q), ref.locate(q)
        assert ln is not None
        assert lr is not None
        assert ln[0].label == lr[0].label
        assert abs(ln[1] - lr[1]) < 1e-9
        assert abs(nat.param_of(q) - ref.param_of(q)) < 1e-9


@pytest.mark.parametrize("name", list(_ARMS))
def test_native_locate_recovers_own_configuration(name: str) -> None:
    kb = _kb(name)
    arm = ssik.Manipulator(kb)
    rng = np.random.default_rng(37)
    worst = 0.0
    for _ in range(100):
        q = _random_q(kb, rng)
        fam = arm.charts(arm.fk(q))
        assert fam.native
        located = fam.locate(q)
        assert located is not None
        chart, t = located
        worst = max(worst, _wrap_dist(chart.q(t), q))
        assert fam.locate(_random_q(kb, rng)) is None  # a different pose's q
    assert worst < 1e-9


def test_native_off_manifold_and_outside_domain_are_nan() -> None:
    kb = _kb("franka_panda")
    rng = np.random.default_rng(41)
    fam = charts(kb, poe_forward_kinematics(kb, _random_q(kb, rng)), native=True)
    for chart in fam:
        for lo, hi in chart.domain:
            if lo > -np.pi + 1e-6:
                assert np.all(np.isnan(chart.q(lo - 1e-6)))
            if hi < np.pi - 1e-6:
                assert np.all(np.isnan(chart.q(hi + 1e-6)))


def test_native_and_python_agree_at_the_panda_home_keyframe() -> None:
    """Regression: the two backends used to pick different representatives at
    the home keyframe's gimbal lock (and Python's was not even on the manifold).
    Both must now return the canonical split and locate the keyframe."""
    kb = _kb("franka_panda")
    home = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, -0.7853])
    T = poe_forward_kinematics(kb, home)
    nat = charts(kb, T, native=True)
    ref = charts(kb, T, native=False)
    for fam in (nat, ref):
        located = fam.locate(home)
        assert located is not None
        assert _wrap_dist(located[0].q(located[1]), home) < 1e-12
    for cn, cr in zip(nat.charts, ref.charts, strict=True):
        qn, qr = cn.q(home[6]), cr.q(home[6])
        assert np.all(np.isfinite(qn)) == np.all(np.isfinite(qr))
        if np.all(np.isfinite(qn)):
            assert _wrap_dist(qn, qr) < 1e-9


@pytest.mark.parametrize("name", list(_ARMS))
def test_native_chart_frame_matches_the_python_tail(name: str) -> None:
    """``_ssik_native.chart_frame`` agrees with :func:`ssik.chart._frame_tail`
    (the reference) on both families, Euclidean and metric, including the NaN
    rows off the branch."""
    from ssik.chart import _frame_tail

    ext = _load_ext()
    assert hasattr(ext, "chart_frame"), "extension predates chart_frame; rebuild"
    kb = _kb(name)
    rng = np.random.default_rng(11)
    q = _random_q(kb, rng)
    fam = charts(kb, poe_forward_kinematics(kb, q), solver_name=_ARMS[name][2])
    located = fam.locate(q)
    assert located is not None
    chart = located[0]
    raw = _raw_tangents(chart)
    # SPD, and deliberately not diagonal, so M genuinely rotates the complement.
    a = rng.standard_normal((7, 7))
    metric = np.ascontiguousarray(a @ a.T + 7.0 * np.eye(7))
    for m in (None, metric):
        pd, pv = _frame_tail(raw, m)
        nd, nv = ext.chart_frame(raw, m)
        assert np.array_equal(np.isnan(pd), np.isnan(nd))
        assert np.array_equal(np.isnan(pv), np.isnan(nv))
        assert np.nanmax(np.abs(pd - nd)) < 1e-12
        assert np.nanmax(np.abs(pv - nv)) < 1e-12


def test_chart_frame_rejects_a_metric_of_the_wrong_shape() -> None:
    ext = _load_ext()
    tangent = np.ones((1, 7))
    with pytest.raises(ValueError, match=r"\(7, 7\)"):
        ext.chart_frame(tangent, np.eye(6))


@pytest.mark.parametrize("name", list(_ARMS))
def test_native_chart_tangent_matches_the_python_tail(name: str) -> None:
    """``_ssik_native.chart_tangent`` agrees with :func:`ssik.chart._tangent_tail`
    (the reference) on both families: the same directions, the same rates, and
    the same NaN pattern off the branch. Agreement is to a few ULP, not exact --
    Eigen's ``norm()`` and numpy's sum the seven squares in a different order."""
    from ssik.chart import _tangent_tail

    ext = _load_ext()
    assert hasattr(ext, "chart_tangent"), "extension predates chart_tangent; rebuild"
    kb = _kb(name)
    rng = np.random.default_rng(11)
    q = _random_q(kb, rng)
    fam = charts(kb, poe_forward_kinematics(kb, q), solver_name=_ARMS[name][2])
    located = fam.locate(q)
    assert located is not None
    raw = _raw_tangents(located[0])

    pd, pr = _tangent_tail(raw)
    nd, nr = ext.chart_tangent(raw)
    assert np.array_equal(np.isnan(pd), np.isnan(nd))
    assert np.array_equal(np.isnan(pr), np.isnan(nr))
    # A few ULP, which is reassociation of the same arithmetic and nothing more.
    # Asserted as ULP rather than a relative tolerance so a real divergence in
    # the formula cannot hide under a loose atol.
    assert _max_ulp(pd, nd) <= 4
    assert _max_ulp(pr, nr) <= 4


def test_chart_tangent_carries_the_degenerate_rows_like_numpy() -> None:
    """The rows :meth:`Chart.tangent` documents: a fold (rate diverges, the
    direction stays unit), a vanishing tangent (rate 0, direction 0/0 = NaN),
    and off the branch (both NaN). Native and numpy must agree on all of them."""
    from ssik.chart import _tangent_tail

    ext = _load_ext()
    raw = np.ascontiguousarray(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # already unit
                [0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],  # rate 5
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # vanishing: 0/0
                [np.nan] * 7,  # off the branch
                [1e300, 1e300, 0.0, 0.0, 0.0, 0.0, 0.0],  # near the overflow edge
            ]
        )
    )
    with np.errstate(over="ignore"):  # the 1e300 row overflows by design
        pd, pr = _tangent_tail(raw)
    nd, nr = ext.chart_tangent(raw)
    assert _max_ulp(pd, nd) <= 4
    assert _max_ulp(pr, nr) <= 4
    assert nr[1] == 5.0
    assert np.isnan(nd[2]).all()
    assert nr[2] == 0.0
    assert np.isnan(nd[3]).all()
    assert np.isnan(nr[3])
