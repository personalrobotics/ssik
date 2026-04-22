"""End-to-end IK tests for :mod:`ssik.solvers.ur_family_hawkins` on UR5.

Strategy: pick a non-singular ``q*``, compute ``T* = FK(q*)`` via the KinBody,
invert via :func:`ur_family_hawkins.solve`, and verify:

1. At least one returned solution reproduces ``T*`` under FK (the minimal
   correctness claim).
2. Every returned solution reproduces ``T*`` (stronger: the solver doesn't
   emit fabricated branches).
3. ``q*`` itself is close to at least one solution modulo ``2pi`` (the
   solver enumerates the configuration space, not just one branch).
4. UR5's classical eight-fold IK enumeration is respected at generic poses.

This module is a **correctness oracle** for the upcoming tier-1 generic
solver. Once tier-1 is cross-validated against these tests on UR5, both
the Hawkins solver and this test file get deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.solvers import ur_family_hawkins

FIXTURES = Path(__file__).parent / "fixtures"
URDF_PATH = FIXTURES / "ur5.urdf"


def _rodrigues(k: np.ndarray, t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R: np.ndarray = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R


def _axis_angle_4x4(k: np.ndarray, t: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rodrigues(k, t)
    return T


def _fk(kb: Any, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _axis_angle_4x4(j.axis, qi) @ j.T_right
    return T


@pytest.fixture(scope="module")
def ur5_kb() -> Any:
    return load_urdf_kinbody_normalized(URDF_PATH, "base_link", "ee_link")


def _q_matches(q_a: np.ndarray, q_b: np.ndarray, tol: float = 1e-4) -> bool:
    """Two joint vectors match modulo ``2pi`` on each joint."""

    def wrap(x: float) -> float:
        return float(((x + np.pi) % (2 * np.pi)) - np.pi)

    return all(abs(wrap(float(a - b))) < tol for a, b in zip(q_a, q_b, strict=True))


@pytest.mark.parametrize(
    "q_star",
    [
        np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2]),
        np.array([-1.2, -1.8, 2.1, -0.4, 0.7, -1.5]),
        np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
        np.array([1.5, -1.0, -0.5, 2.0, -0.8, 0.9]),
    ],
)
def test_solutions_all_satisfy_fk(ur5_kb: Any, q_star: np.ndarray) -> None:
    """Seeded q*: every returned solution reproduces T* under FK.

    UR5 returns up to 8 solutions (2 shoulder x 2 wrist-pitch x 2 elbow).
    Specific poses may have fewer if some branches leave the reachable
    workspace; the test only asserts >=1 and that every returned one is
    exact.
    """
    T_star = _fk(ur5_kb, q_star)

    solutions, is_ls = ur_family_hawkins.solve(ur5_kb, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8

    for i, q in enumerate(solutions):
        T_check = _fk(ur5_kb, q)
        assert np.allclose(T_check, T_star, atol=1e-9), (
            f"solution {i} q={q.tolist()} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


@pytest.mark.parametrize(
    "q_star",
    [
        np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2]),
        np.array([-1.2, -1.8, 2.1, -0.4, 0.7, -1.5]),
        np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
    ],
)
def test_q_star_recovered(ur5_kb: Any, q_star: np.ndarray) -> None:
    """The seeded q* should appear in the returned solution set (mod 2pi)."""
    T_star = _fk(ur5_kb, q_star)
    solutions, _ = ur_family_hawkins.solve(ur5_kb, T_star)
    assert any(_q_matches(q, q_star) for q in solutions), (
        f"q_star={q_star.tolist()} not among returned solutions:\n"
        + "\n".join(f"  {q.tolist()}" for q in solutions)
    )


def test_wrong_dof_raises(ur5_kb: Any) -> None:
    """Solver rejects non-6-DOF chains with a clear error before reaching
    internal geometry extraction."""
    kb = load_urdf_kinbody_normalized(URDF_PATH, "base_link", "wrist_2_link")
    T_dummy = np.eye(4)
    with pytest.raises(ValueError, match="6-DOF"):
        ur_family_hawkins.solve(kb, T_dummy)


def test_no_tool_offset_raises() -> None:
    """Loading UR5 to wrist_3_link (no tool flange) has d6=0, which the solver
    rejects with a clear error pointing at the ee_link workaround."""
    kb = load_urdf_kinbody_normalized(URDF_PATH, "base_link", "wrist_3_link")
    T_dummy = np.eye(4)
    with pytest.raises(ValueError, match="d6"):
        ur_family_hawkins.solve(kb, T_dummy)


# ---------------------------------------------------------------------------
# Property-based sweep: random generic poses
# ---------------------------------------------------------------------------

_ANGLE = st.floats(
    min_value=-np.pi + 0.3,
    max_value=np.pi - 0.3,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def _random_q(draw: st.DrawFn) -> np.ndarray:
    """Random joint config avoiding wrist-singular zone (s5 ~= 0)."""
    q = np.array(
        [draw(_ANGLE), draw(_ANGLE), draw(_ANGLE), draw(_ANGLE), draw(_ANGLE), draw(_ANGLE)]
    )
    # Keep wrist pitch well away from singular sin(theta5) = 0.
    assume(abs(np.sin(q[4])) > 0.2)
    # Keep elbow away from singular straight-arm configuration (cos(theta3) = +/-1).
    assume(abs(np.sin(q[2])) > 0.2)
    return q


@given(_random_q())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture],
)
def test_random_q_roundtrip(ur5_kb: Any, q_star: np.ndarray) -> None:
    """Random non-singular q*: at least one returned solution satisfies FK.

    Since q* comes from FK of an actual reachable configuration, the IK
    must find it. Some returned branches may differ (other postures reaching
    the same pose); the test only requires **one** solution to recover T_star.
    """
    T_star = _fk(ur5_kb, q_star)
    solutions, _ = ur_family_hawkins.solve(ur5_kb, T_star)
    assert len(solutions) >= 1
    assert any(np.allclose(_fk(ur5_kb, q), T_star, atol=1e-8) for q in solutions), (
        f"no solution recovers T_star for q_star={q_star.tolist()}"
    )
