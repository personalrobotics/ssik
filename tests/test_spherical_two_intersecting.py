"""End-to-end validation for :mod:`ssik.solvers.ikgeo.spherical_two_intersecting`.

Same validation structure as Phase F.1/F.2: FK is ground truth, hand-picked
generic poses, seeded q* recovery, near-singular coverage, synthetic non-Puma
arm, 500 hypothesis random poses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from ssik._kinbody import Joint, KinBody, Link
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.solvers.ikgeo import spherical_two_intersecting

FIXTURES = Path(__file__).parent / "fixtures"
PUMA_URDF = FIXTURES / "puma560.urdf"


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


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _q_matches(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    return all(abs(_wrap(float(ai - bi))) < tol for ai, bi in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Fixtures: Puma 560 + a synthetic spherical+intersecting-shoulder arm.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def puma_kb() -> Any:
    return load_urdf_kinbody_normalized(PUMA_URDF, "base_link", "wrist_3_link")


@pytest.fixture(scope="module")
def synthetic_spherical_two_intersecting_kb() -> KinBody:
    """A synthesised 6R arm with a spherical wrist at (3, 4, 5) and joints
    (0, 1) sharing an origin (p[1] = 0). Deliberately non-parallel
    shoulder (joints 1, 2 neither parallel nor coincident with Puma's
    geometry) to validate the 'generic, not Puma-specific' claim.
    """
    # d1 = 0 (joint 0/1 share world origin). Shoulder tilts differently.
    a2, a3, d4 = 0.45, 0.06, 0.38
    # Axes: joint1 z, joint2 tilted away from y to force non-parallel j1/j2.
    tilt_theta = np.deg2rad(15.0)  # 15 degrees off y toward x
    axes = [
        np.array([0.0, 0.0, 1.0]),
        np.array([np.sin(tilt_theta), -np.cos(tilt_theta), 0.0]),
        np.array([np.sin(tilt_theta), -np.cos(tilt_theta), 0.0]),  # parallel to j1 axis
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    t_lefts = [
        np.array([0.0, 0.0, 0.0]),  # joint 0 at world origin
        np.array([0.0, 0.0, 0.0]),  # joint 1 coincident with joint 0 (p[1] = 0)
        np.array([a2, 0.0, 0.0]),
        np.array([a3, 0.0, 0.0]),
        np.array([0.0, 0.0, d4]),
        np.array([0.0, 0.0, 0.0]),
    ]
    link_names = [
        "base_link",
        *(f"link_{i}" for i in range(1, 6)),
        "ee_link",
    ]
    links = [Link(name=n) for n in link_names]
    joints: list[Joint] = []
    for i in range(6):
        t_left_i = np.eye(4)
        t_left_i[:3, 3] = t_lefts[i]
        t_right_i = np.eye(4)
        joints.append(
            Joint(
                name=f"joint_{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=t_left_i,
                T_right=t_right_i,
                axis=axes[i],
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


# ---------------------------------------------------------------------------
# Hand-picked q*: exact FK roundtrip on generic poses.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_star",
    [
        np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2]),
        np.array([-1.2, -1.8, 2.1, -0.4, 0.7, -1.5]),
        np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
        np.array([1.5, -1.0, -0.5, 2.0, -0.8, 0.9]),
    ],
)
def test_generic_pose_all_solutions_fk_match(puma_kb: Any, q_star: np.ndarray) -> None:
    T_star = _fk(puma_kb, q_star)
    solutions, is_ls = spherical_two_intersecting.solve(puma_kb, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for i, q in enumerate(solutions):
        T_check = _fk(puma_kb, q)
        assert np.allclose(T_check, T_star, atol=1e-8), (
            f"solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


@pytest.mark.parametrize(
    "q_star",
    [
        np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2]),
        np.array([-1.2, -1.8, 2.1, -0.4, 0.7, -1.5]),
        np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
    ],
)
def test_seeded_q_star_is_recovered(puma_kb: Any, q_star: np.ndarray) -> None:
    T_star = _fk(puma_kb, q_star)
    solutions, _ = spherical_two_intersecting.solve(puma_kb, T_star)
    assert any(_q_matches(q, q_star) for q in solutions), (
        f"q_star={q_star.tolist()} not recovered in {len(solutions)} solutions"
    )


def test_generic_pose_returns_eight_solutions(puma_kb: Any) -> None:
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2])
    T_star = _fk(puma_kb, q_star)
    solutions, _ = spherical_two_intersecting.solve(puma_kb, T_star)
    assert len(solutions) == 8


# ---------------------------------------------------------------------------
# Near-singular coverage.
# ---------------------------------------------------------------------------


_NEAR_SINGULAR_Q = [
    # Wrist-pitch singularity: sin(q[4]) = 0 aligns joints 3 and 5.
    np.array([0.5, -0.8, 1.0, 0.3, 0.0, 0.4]),
    np.array([0.5, -0.8, 1.0, 0.3, np.pi, 0.4]),
    # Elbow singularity: sin(q[2]) = 0.
    np.array([0.5, -0.8, 0.0, 0.3, 0.6, 0.4]),
    # Shoulder-pan zero.
    np.array([0.0, -0.8, 1.0, 0.3, 0.6, 0.4]),
    np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.0]),
]


@pytest.mark.parametrize("q_star", _NEAR_SINGULAR_Q)
def test_near_singular_pose_returned_solutions_fk_match(puma_kb: Any, q_star: np.ndarray) -> None:
    T_star = _fk(puma_kb, q_star)
    solutions, _ = spherical_two_intersecting.solve(puma_kb, T_star)
    assert len(solutions) >= 1, "no solutions at near-singular pose"
    for i, q in enumerate(solutions):
        T_check = _fk(puma_kb, q)
        assert np.allclose(T_check, T_star, atol=1e-6), (
            f"singular-pose solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


# ---------------------------------------------------------------------------
# Non-Puma arm: same topology, different geometry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_star",
    [
        np.array([0.4, -0.6, 0.8, 1.0, -0.4, 0.3]),
        np.array([-1.1, -1.5, 1.9, -0.5, 0.6, -1.3]),
        np.array([0.2, -0.4, 0.5, -0.6, 0.7, -0.8]),
    ],
)
def test_synthetic_spherical_two_intersecting_fk_roundtrip(
    synthetic_spherical_two_intersecting_kb: KinBody, q_star: np.ndarray
) -> None:
    T_star = _fk(synthetic_spherical_two_intersecting_kb, q_star)
    solutions, is_ls = spherical_two_intersecting.solve(
        synthetic_spherical_two_intersecting_kb, T_star
    )
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for i, q in enumerate(solutions):
        T_check = _fk(synthetic_spherical_two_intersecting_kb, q)
        assert np.allclose(T_check, T_star, atol=1e-8), f"synthetic solution {i} fails FK"
    assert any(_q_matches(q, q_star) for q in solutions), "seeded q* not recovered"


# ---------------------------------------------------------------------------
# 500 random hypothesis poses on Puma 560.
# ---------------------------------------------------------------------------


_ANGLE = st.floats(min_value=-np.pi + 0.3, max_value=np.pi - 0.3, allow_nan=False, width=64)


@st.composite
def _random_q(draw: st.DrawFn) -> np.ndarray:
    q = np.array([draw(_ANGLE) for _ in range(6)])
    assume(abs(np.sin(q[1])) > 0.2)
    assume(abs(np.sin(q[2])) > 0.2)
    assume(abs(np.sin(q[4])) > 0.2)
    return q


@given(_random_q())
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture],
)
def test_random_q_roundtrip_fk(puma_kb: Any, q_star: np.ndarray) -> None:
    T_star = _fk(puma_kb, q_star)
    solutions, is_ls = spherical_two_intersecting.solve(puma_kb, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for q in solutions:
        assert np.allclose(_fk(puma_kb, q), T_star, atol=1e-8), f"FK mismatch at q={q.tolist()}"
    assert any(_q_matches(q, q_star, tol=1e-4) for q in solutions), (
        f"seeded q*={q_star.tolist()} not recovered"
    )


# ---------------------------------------------------------------------------
# Topology validation.
# ---------------------------------------------------------------------------


def test_wrong_dof_raises(puma_kb: Any) -> None:
    kb = load_urdf_kinbody_normalized(PUMA_URDF, "base_link", "wrist_2_link")
    with pytest.raises(ValueError, match="6-DOF"):
        spherical_two_intersecting.solve(kb, np.eye(4))


def test_wrong_topology_raises_three_parallel() -> None:
    """UR5 has three parallel axes at (1, 2, 3), not a spherical wrist at
    (3, 4, 5). The solver must refuse."""
    ur5_kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    with pytest.raises(ValueError, match=r"\(3, 4, 5\)"):
        spherical_two_intersecting.solve(ur5_kb, np.eye(4))


def test_wrong_topology_raises_nonzero_p1() -> None:
    """A spherical-wrist arm whose joint-1 translation is non-zero must
    be rejected (belongs to spherical_two_parallel or a more general
    solver instead)."""
    # Clone the Puma fixture but shift joint 1 to break p[1] = 0.
    kb = load_urdf_kinbody_normalized(PUMA_URDF, "base_link", "wrist_3_link")
    # Non-destructive: build a parallel list with a shifted joint-1.
    from dataclasses import replace

    shifted_T_left = kb.joints[1].T_left.copy()
    shifted_T_left[2, 3] = 0.05  # non-zero z offset
    shifted_joint = replace(kb.joints[1], T_left=shifted_T_left)
    shifted_kb = KinBody(
        links=kb.links,
        joints=[shifted_joint if i == 1 else kb.joints[i] for i in range(6)],
    )
    with pytest.raises(ValueError, match=r"p\[1\] = 0"):
        spherical_two_intersecting.solve(shifted_kb, np.eye(4))
