"""Alternative ``Manipulator`` constructors: from_dh / from_axes / from_transforms
(#8, #9).

Each builds a KinBody from a non-URDF parameterisation (matching EAIK's DhRobot /
HPRobot / HomogeneousRobot) and must produce a dispatchable, solvable arm whose FK
equals the parameterisation's own reference FK to machine precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik.manipulator import Manipulator

_PI = np.pi


def _rot(axis: np.ndarray, theta: float) -> np.ndarray:
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    c, s = np.cos(theta), np.sin(theta)
    kk = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    m = np.eye(4)
    m[:3, :3] = np.eye(3) + s * kk + (1 - c) * kk @ kk
    return m


def _tr(v: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = v
    return m


def _hp_fk(h: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    t = _tr(p[0])
    for i in range(len(h)):
        t = t @ _rot(h[i], q[i]) @ _tr(p[i + 1])
    return t


def _dh_fk(alpha, a, d, q):
    def rz(th):
        return _rot(np.array([0.0, 0.0, 1.0]), th)

    def tx(x):
        m = np.eye(4)
        m[0, 3] = x
        return m

    def tz(z):
        m = np.eye(4)
        m[2, 3] = z
        return m

    def rx(al):
        return _rot(np.array([1.0, 0.0, 0.0]), al)

    t = np.eye(4)
    for i in range(len(q)):
        t = t @ rz(q[i]) @ tz(d[i]) @ tx(a[i]) @ rx(alpha[i])
    return t


def _tf_fk(trafos: np.ndarray, axis, q: np.ndarray) -> np.ndarray:
    t = trafos[0]
    for i in range(len(q)):
        t = t @ _rot(np.asarray(axis, dtype=np.float64), q[i]) @ trafos[i + 1]
    return np.asarray(t, dtype=np.float64)


def _roundtrip_worst(m: Manipulator, seed: int, n: int = 200) -> float:
    rng = np.random.default_rng(seed)
    worst = 0.0
    checked = 0
    for _ in range(n):
        q = rng.uniform(-2.0, 2.0, m.dof)
        target = m.fk(q)
        sols = m.solve(target, respect_limits=False)
        if sols:
            checked += 1
            worst = max(worst, min(float(np.max(np.abs(m.fk(s.q) - target))) for s in sols))
    assert checked > n // 2, "arm produced solutions on too few poses"
    return worst


# ---------------------------------------------------------------------------
# from_axes (ik-geo / EAIK HPRobot)
# ---------------------------------------------------------------------------

_H = np.array(
    [[0, 0, 1.0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float64
)
_P = np.array(
    [[0, 0, 0.1], [0.1, 0, 0.2], [0.3, 0, 0], [0.2, 0, 0], [0, 0, 0], [0.1, 0, 0], [0.05, 0, 0]],
    dtype=np.float64,
)


def test_from_axes_fk_matches_reference() -> None:
    m = Manipulator.from_axes(_H, _P)
    rng = np.random.default_rng(1)
    worst = max(
        float(np.max(np.abs(m.fk(q) - _hp_fk(_H, _P, q)))) for q in rng.uniform(-3, 3, (200, 6))
    )
    assert worst < 1e-12, f"from_axes FK vs H/P reference {worst:.2e}"


def test_from_axes_dispatches_and_solves() -> None:
    m = Manipulator.from_axes(_H, _P)
    assert m._plan.solver_name.startswith("ikgeo.")  # spherical-wrist arm
    assert _roundtrip_worst(m, seed=2) < 1e-9


def test_from_axes_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="joint_axes must be"):
        Manipulator.from_axes(np.zeros((6, 2)), _P)
    with pytest.raises(ValueError, match="offsets must be"):
        Manipulator.from_axes(_H, np.zeros((6, 3)))


# ---------------------------------------------------------------------------
# from_dh (EAIK DhRobot) -- Puma 560, the classic analytic benchmark
# ---------------------------------------------------------------------------

_PUMA_ALPHA = [_PI / 2, 0.0, -_PI / 2, _PI / 2, -_PI / 2, 0.0]
_PUMA_A = [0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0]
_PUMA_D = [0.0, 0.0, 0.15005, 0.4318, 0.0, 0.0]


def test_from_dh_puma560_fk_matches_reference() -> None:
    m = Manipulator.from_dh(_PUMA_ALPHA, _PUMA_A, _PUMA_D)
    rng = np.random.default_rng(3)
    worst = max(
        float(np.max(np.abs(m.fk(q) - _dh_fk(_PUMA_ALPHA, _PUMA_A, _PUMA_D, q))))
        for q in rng.uniform(-3, 3, (200, 6))
    )
    assert worst < 1e-12, f"from_dh FK vs DH reference {worst:.2e}"


def test_from_dh_puma560_is_spherical_and_solves() -> None:
    """Puma 560 has a parallel shoulder + spherical wrist -> Tier-0 spherical
    solver, and solve(fk(q)) recovers a machine-precision IK (the #8 benchmark)."""
    m = Manipulator.from_dh(_PUMA_ALPHA, _PUMA_A, _PUMA_D)
    assert m._plan.solver_name == "ikgeo.spherical_two_parallel", m._plan.solver_name
    assert _roundtrip_worst(m, seed=4) < 1e-9


def test_from_dh_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        Manipulator.from_dh([0.0, 0.0], [0.1], [0.1, 0.2])


# ---------------------------------------------------------------------------
# from_transforms (EAIK HomogeneousRobot)
# ---------------------------------------------------------------------------


def _sample_trafos() -> np.ndarray:
    t = np.stack([np.eye(4) for _ in range(7)])
    for i in range(7):
        t[i, :3, 3] = [0.1 * i, 0.0, 0.1]
    t[1, :3, :3] = _rot(np.array([1.0, 0.0, 0.0]), 0.5)[:3, :3]
    t[3, :3, :3] = _rot(np.array([0.0, 1.0, 0.0]), -0.4)[:3, :3]
    return t


def test_from_transforms_fk_matches_reference() -> None:
    trafos = _sample_trafos()
    axis = [0.0, 0.0, 1.0]
    m = Manipulator.from_transforms(trafos, joint_axis=axis)
    rng = np.random.default_rng(5)
    worst = max(
        float(np.max(np.abs(m.fk(q) - _tf_fk(trafos, axis, q))))
        for q in rng.uniform(-3, 3, (200, 6))
    )
    assert worst < 1e-12, f"from_transforms FK vs reference {worst:.2e}"


def test_from_transforms_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="joint_trafos must be"):
        Manipulator.from_transforms(np.zeros((7, 3, 4)))


# ---------------------------------------------------------------------------
# limits pass-through (shared across all three)
# ---------------------------------------------------------------------------


def test_constructor_limits_are_applied() -> None:
    limits = np.array([[-1.0, 1.0]] * 6)
    m = Manipulator.from_axes(_H, _P, limits=limits)
    assert [j.limits for j in m.kinbody.joints] == [(-1.0, 1.0)] * 6


# ---------------------------------------------------------------------------
# axis-length gauge: a joint axis denotes only a direction, so a non-unit axis
# must build the identical arm (the Rodrigues kernel + predicates assume unit
# length; the URDF loader normalizes for free, the direct constructors must too)
# ---------------------------------------------------------------------------


def test_non_unit_axis_is_normalized_and_fk_identical() -> None:
    """Scaling any ``from_axes`` H-row (a non-unit axis) must yield an arm that
    is FK-identical to the unit version -- not a silently different robot."""
    m_unit = Manipulator.from_axes(_H, _P)
    h_scaled = _H.copy()
    h_scaled[1] *= 2.0
    h_scaled[3] *= 0.5
    m_scaled = Manipulator.from_axes(h_scaled, _P)
    rng = np.random.default_rng(11)
    worst = max(
        float(np.max(np.abs(m_unit.fk(q) - m_scaled.fk(q)))) for q in rng.uniform(-3, 3, (200, 6))
    )
    assert worst == 0.0, f"non-unit axis changed FK by {worst:.2e} (not normalized)"
    # And the scaled-axis arm still solves against itself at machine precision.
    assert _roundtrip_worst(m_scaled, seed=12) < 1e-9


def test_degenerate_axis_is_rejected() -> None:
    """A near-zero axis carries no direction and is a construction error."""
    h_bad = _H.copy()
    h_bad[2] = 0.0
    with pytest.raises(ValueError, match="degenerate"):
        Manipulator.from_axes(h_bad, _P)
