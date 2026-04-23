"""End-to-end validation for :mod:`ssik.solvers.ikgeo.three_parallel`.

Validation design (the criterion for deleting the Hawkins oracle):

1. **FK is ground truth.** Every returned solution must reproduce the target
   pose under the KinBody's own forward kinematics within 1e-8. FK is
   simpler than any IK solver and can be cross-checked visually or against
   URDF-native FK libraries; if the solver's output inverts under FK it is
   correct by definition of IK.

2. **Completeness on random poses.** Hypothesis sweeps 500 generic
   non-singular q* configurations, asserts the seeded q* is recovered, and
   asserts all returned solutions invert correctly.

3. **Near-singular coverage.** Parametrised tests at each classical UR5
   singularity (shoulder = 0, elbow = +/-pi, wrist pitch = 0 / pi) verify
   the solver degrades gracefully (returns fewer solutions, all exact) or
   signals ``is_ls`` rather than producing silent wrong answers.

4. **Non-UR three-parallel.** A synthetic KinBody with three parallel joints
   at indices (1, 2, 3) but *different* inter-joint offsets than UR5
   exercises the "generic" claim of the solver: it is not UR-specific.

5. **Hawkins cross-check** as an independent reference on UR5. Solution sets
   agree up to angle-tolerance and count. Once we have confidence from
   (1)-(4), the Hawkins oracle can retire; until then it guards against
   regressions.
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
from ssik.solvers.ikgeo import three_parallel

FIXTURES = Path(__file__).parent / "fixtures"
UR5_URDF = FIXTURES / "ur5.urdf"


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
# Fixtures: UR5 + a synthetic three-parallel arm with different dimensions.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ur5_kb() -> Any:
    return load_urdf_kinbody_normalized(UR5_URDF, "base_link", "ee_link")


@pytest.fixture(scope="module")
def synthetic_three_parallel_kb() -> KinBody:
    """A synthesised three-parallel 6R arm with UR-class structure but
    deliberately different link lengths and offsets. Validates that
    three_parallel is not UR5-specific."""
    # Different d1, a2, a3, d4, d5, d6 from UR5:
    d1, a2, a3, d4, d5, d6 = 0.2, -0.6, -0.5, 0.15, 0.12, 0.10
    # Same axis pattern as UR5 (shoulder, three parallel, wrist, tool).
    axes = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, -1.0, 0.0]),
    ]
    # T_left translations (same "offset-per-joint" pattern as our UR5 POE):
    t_lefts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, d1]),
        np.array([a2, 0.0, 0.0]),
        np.array([a3, 0.0, 0.0]),
        np.array([0.0, -d4, 0.0]),
        np.array([0.0, 0.0, -d5]),
    ]
    # T_right on last joint: same R_home as UR5 + d6 translation baked in.
    r_home = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    t_right_translation = np.array([0.0, -d6, 0.0])
    t_right_last = np.eye(4)
    t_right_last[:3, :3] = r_home
    t_right_last[:3, 3] = t_right_translation

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
        t_right_i = t_right_last if i == 5 else np.eye(4)
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
def test_generic_pose_all_solutions_fk_match(ur5_kb: Any, q_star: np.ndarray) -> None:
    T_star = _fk(ur5_kb, q_star)
    solutions, is_ls = three_parallel.solve(ur5_kb, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for i, q in enumerate(solutions):
        T_check = _fk(ur5_kb, q)
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
def test_seeded_q_star_is_recovered(ur5_kb: Any, q_star: np.ndarray) -> None:
    T_star = _fk(ur5_kb, q_star)
    solutions, _ = three_parallel.solve(ur5_kb, T_star)
    assert any(_q_matches(q, q_star) for q in solutions), (
        f"q_star={q_star.tolist()} not recovered in {len(solutions)} solutions"
    )


def test_generic_pose_returns_eight_solutions(ur5_kb: Any) -> None:
    """Generic non-singular UR5 pose has exactly 8 IK solutions."""
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2])
    T_star = _fk(ur5_kb, q_star)
    solutions, _ = three_parallel.solve(ur5_kb, T_star)
    assert len(solutions) == 8


# ---------------------------------------------------------------------------
# Near-singular coverage: each classical UR singularity, checked by FK.
# ---------------------------------------------------------------------------


_NEAR_SINGULAR_Q = [
    # Wrist-pitch singularity: sin(q[4]) = 0 collapses two wrist branches.
    np.array([0.5, -0.8, 1.0, 0.3, 0.0, 0.4]),
    np.array([0.5, -0.8, 1.0, 0.3, np.pi, 0.4]),
    # Elbow singularity: sin(q[2]) = 0 means arm is fully extended or folded.
    np.array([0.5, -0.8, 0.0, 0.3, 0.6, 0.4]),
    # Shoulder-pan zero: base rotation at zero.
    np.array([0.0, -0.8, 1.0, 0.3, 0.6, 0.4]),
    # Composite near-singularities.
    np.array([0.0, 0.0, 1.0, 0.0, 0.5, 0.0]),
]


@pytest.mark.parametrize("q_star", _NEAR_SINGULAR_Q)
def test_near_singular_pose_returned_solutions_fk_match(ur5_kb: Any, q_star: np.ndarray) -> None:
    """At each UR singularity, returned solutions must still satisfy FK.
    The solver may return fewer solutions (collapsed branches) but none
    should be silently wrong."""
    T_star = _fk(ur5_kb, q_star)
    solutions, _ = three_parallel.solve(ur5_kb, T_star)
    assert len(solutions) >= 1, "no solutions at near-singular pose"
    for i, q in enumerate(solutions):
        T_check = _fk(ur5_kb, q)
        assert np.allclose(T_check, T_star, atol=1e-6), (
            f"singular-pose solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


# ---------------------------------------------------------------------------
# Non-UR three-parallel arm: same algorithm, different geometry.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_star",
    [
        np.array([0.4, -0.6, 0.8, 1.0, -0.4, 0.3]),
        np.array([-1.1, -1.5, 1.9, -0.5, 0.6, -1.3]),
        np.array([0.2, -0.4, 0.5, -0.6, 0.7, -0.8]),
    ],
)
def test_synthetic_three_parallel_fk_roundtrip(
    synthetic_three_parallel_kb: KinBody, q_star: np.ndarray
) -> None:
    """Three-parallel solver on a synthesised arm with different link
    dimensions than UR5. Validates the 'generic, not UR-specific' claim."""
    T_star = _fk(synthetic_three_parallel_kb, q_star)
    solutions, is_ls = three_parallel.solve(synthetic_three_parallel_kb, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for i, q in enumerate(solutions):
        T_check = _fk(synthetic_three_parallel_kb, q)
        assert np.allclose(T_check, T_star, atol=1e-8), f"synthetic solution {i} fails FK"
    assert any(_q_matches(q, q_star) for q in solutions), "seeded q* not recovered"


# ---------------------------------------------------------------------------
# 500 random hypothesis poses: seeded q* recovered and all returned solutions
# invert under FK.
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
def test_random_q_roundtrip_fk(ur5_kb: Any, q_star: np.ndarray) -> None:
    """500 random non-singular q*: seeded q* is recovered, all returned
    solutions reproduce T_star under FK."""
    T_star = _fk(ur5_kb, q_star)
    solutions, is_ls = three_parallel.solve(ur5_kb, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for q in solutions:
        assert np.allclose(_fk(ur5_kb, q), T_star, atol=1e-8), f"FK mismatch at q={q.tolist()}"
    assert any(_q_matches(q, q_star, tol=1e-4) for q in solutions), (
        f"seeded q*={q_star.tolist()} not recovered"
    )


# ---------------------------------------------------------------------------
# Regression: issue #56 -- three_parallel dropped the seeded q* on UR5's
# shoulder-wrist alignment pose (q0=0, q5=0) where SP6's Bezout quartic
# has near-double real roots. Dedup was picking an insertion-order-
# earlier drifted representative instead of the exact one. After the fix
# (SP6 sort-by-residual before dedup + Gauss-Newton refinement), q* is
# recovered at machine precision.
# ---------------------------------------------------------------------------


def test_recovers_shoulder_wrist_alignment_pose_issue_56(ur5_kb: Any) -> None:
    """Regression for #56.

    Before the fix: SP6's ellipse-intersection produced 4 candidates
    split into 2 clusters by near-double Bezout quartic roots. Dedup
    merged each cluster to the first-seen member, which for this pose
    happened to be the drifted representative (~7.7e-4 rad off q*).
    That drift propagated through SP3/SP1 to the final q vector, failing
    the ``1e-4`` seeded-recovery threshold.

    After the fix: SP6 sorts candidates by pre-refinement residual so
    dedup keeps the cleaner representative, then GN-refines. At this
    pose q* is recovered to <1e-12 rad per joint.
    """
    q_star = np.array([0.0, 1.0, 1.0, 0.36474982, -1.0, 0.0])
    T_star = _fk(ur5_kb, q_star)
    solutions, is_ls = three_parallel.solve(ur5_kb, T_star)

    assert not is_ls
    # UR5 at this shoulder-wrist alignment pose has 4 distinct IK
    # branches (the shoulder-flip degenerates to identity).
    assert len(solutions) == 4

    for q in solutions:
        T_check = _fk(ur5_kb, q)
        assert np.allclose(T_check, T_star, atol=1e-10), f"FK mismatch at {q}"

    def _max_abs_wrap(q: np.ndarray) -> float:
        return max(abs(_wrap(float(qi - qs))) for qi, qs in zip(q, q_star, strict=True))

    closest = min(solutions, key=_max_abs_wrap)
    assert any(_q_matches(q, q_star, tol=1e-10) for q in solutions), (
        f"seeded q* not recovered at machine precision; closest: {closest.tolist()}"
    )


# ---------------------------------------------------------------------------
# Topology validation.
# ---------------------------------------------------------------------------


def test_wrong_dof_raises(ur5_kb: Any) -> None:
    kb = load_urdf_kinbody_normalized(UR5_URDF, "base_link", "wrist_2_link")
    with pytest.raises(ValueError, match="6-DOF"):
        three_parallel.solve(kb, np.eye(4))
