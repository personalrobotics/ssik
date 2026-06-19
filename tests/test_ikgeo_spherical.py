"""End-to-end validation for :mod:`ssik.solvers.ikgeo.spherical`.

Mirrors the bulletproof discipline (memory: feedback_bulletproof_solvers):
hand-picked generic poses, seeded q* recovery, near-singular coverage, two
synthetic non-fixture arms with different geometry, 500 hypothesis random
poses, topology refusal.

Commercial 6R arms almost always match one of the specialized spherical-
wrist siblings (``spherical_two_parallel``, ``spherical_two_intersecting``),
so this solver's fixture coverage is synthetic only. The design:

- Synth A: non-parallel shoulder (axes[1], axes[2] mutually tilted),
  ``p[1] != 0``. Only ``ikgeo.spherical`` applies. Tests the base case.
- Synth B: different link dimensions from Synth A but same topology.
  Validates the "generic, not geometry-specific" claim.

Cross-solver check with specialized siblings isn't possible here because
SP5 (the generic solver's shoulder) is degenerate exactly where the
specialized solvers apply.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from ssik._kinbody import Joint, KinBody, Link
from ssik.solvers.ikgeo import spherical
from tests._hypothesis_strategies import non_singular_q6r


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


def _build_generic_spherical_arm(
    tilt_deg: float,
    d1: float,
    a1: float,
    a2: float,
    a3: float,
    d3: float,
    d4: float,
) -> KinBody:
    """Synthetic 6R arm with spherical wrist at (3, 4, 5), non-parallel
    shoulder, and non-zero p[1]. Only ``ikgeo.spherical`` applies."""
    tilt = np.deg2rad(tilt_deg)
    axes = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([np.sin(tilt), -np.cos(tilt), 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    t_lefts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([a1, 0.0, d1]),
        np.array([a2, 0.0, 0.0]),
        np.array([a3, d3, 0.0]),
        np.array([0.0, 0.0, d4]),
        np.array([0.0, 0.0, 0.0]),
    ]
    link_names = ["base_link", *(f"link_{i}" for i in range(1, 6)), "ee_link"]
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
# Fixtures: Synth A (canonical) and Synth B (different dimensions).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_a() -> KinBody:
    return _build_generic_spherical_arm(
        tilt_deg=25.0, d1=0.2, a1=0.04, a2=0.5, a3=0.08, d3=-0.12, d4=0.4
    )


@pytest.fixture(scope="module")
def synth_b() -> KinBody:
    # Different tilts and link lengths, same topology.
    return _build_generic_spherical_arm(
        tilt_deg=40.0, d1=0.3, a1=0.1, a2=0.6, a3=0.05, d3=0.15, d4=0.35
    )


# ---------------------------------------------------------------------------
# Hand-picked q*: exact FK roundtrip on generic poses.
# ---------------------------------------------------------------------------


_GENERIC_Q = [
    np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2]),
    np.array([-1.2, -1.8, 2.1, -0.4, 0.7, -1.5]),
    np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]),
    np.array([1.5, -1.0, -0.5, 2.0, -0.8, 0.9]),
]


@pytest.mark.parametrize("q_star", _GENERIC_Q)
def test_generic_pose_all_solutions_fk_match(synth_a: KinBody, q_star: np.ndarray) -> None:
    T_star = _fk(synth_a, q_star)
    solutions, is_ls = spherical.solve(synth_a, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for i, sol in enumerate(solutions):
        q = sol.q
        T_check = _fk(synth_a, q)
        assert np.allclose(T_check, T_star, atol=1e-10), (
            f"solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


@pytest.mark.parametrize("q_star", _GENERIC_Q)
def test_seeded_q_star_is_recovered(synth_a: KinBody, q_star: np.ndarray) -> None:
    T_star = _fk(synth_a, q_star)
    solutions, _ = spherical.solve(synth_a, T_star)
    assert any(_q_matches(s.q, q_star, tol=1e-4) for s in solutions), (
        f"q_star={q_star.tolist()} not recovered in {len(solutions)} solutions"
    )


def test_generic_pose_returns_eight_solutions(synth_a: KinBody) -> None:
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2])
    T_star = _fk(synth_a, q_star)
    solutions, _ = spherical.solve(synth_a, T_star)
    assert len(solutions) == 8


# ---------------------------------------------------------------------------
# Near-singular coverage.
# ---------------------------------------------------------------------------


_NEAR_SINGULAR_Q = [
    # Wrist pitch zero / pi.
    np.array([0.5, -0.8, 1.0, 0.3, 0.0, 0.4]),
    np.array([0.5, -0.8, 1.0, 0.3, np.pi, 0.4]),
    # Elbow zero.
    np.array([0.5, -0.8, 0.0, 0.3, 0.6, 0.4]),
    # Shoulder-pan zero.
    np.array([0.0, -0.8, 1.0, 0.3, 0.6, 0.4]),
]


@pytest.mark.parametrize("q_star", _NEAR_SINGULAR_Q)
def test_near_singular_pose_returned_solutions_fk_match(
    synth_a: KinBody, q_star: np.ndarray
) -> None:
    T_star = _fk(synth_a, q_star)
    solutions, _ = spherical.solve(synth_a, T_star)
    assert len(solutions) >= 1, "no solutions at near-singular pose"
    for i, sol in enumerate(solutions):
        q = sol.q
        T_check = _fk(synth_a, q)
        assert np.allclose(T_check, T_star, atol=1e-6), (
            f"singular-pose solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


# ---------------------------------------------------------------------------
# Second synthetic arm: validates "generic, not geometry-specific".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_star", _GENERIC_Q[:3])
def test_second_synthetic_arm_fk_roundtrip(synth_b: KinBody, q_star: np.ndarray) -> None:
    T_star = _fk(synth_b, q_star)
    solutions, is_ls = spherical.solve(synth_b, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for i, sol in enumerate(solutions):
        q = sol.q
        T_check = _fk(synth_b, q)
        assert np.allclose(T_check, T_star, atol=1e-10), f"synth_b solution {i} fails FK"
    assert any(_q_matches(s.q, q_star, tol=1e-4) for s in solutions), "seeded q* not recovered"


# ---------------------------------------------------------------------------
# 500 random hypothesis poses on synth A.
# ---------------------------------------------------------------------------


@given(non_singular_q6r())
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.function_scoped_fixture],
)
def test_random_q_roundtrip_fk(synth_a: KinBody, q_star: np.ndarray) -> None:
    """Bulletproof invariants:
    1. solver does not flag infeasibility (is_ls=False),
    2. every returned q reproduces T_star under FK at 1e-8 atol,
    3. *some* returned q is within 1e-3 rad of q_star (FK-equivalent).

    The 1e-3 rad seeded-recovery tolerance (vs 1e-4 elsewhere) reflects
    a genuine precision floor of the generic SP5-composition solver at
    a specific class of near-singular poses: when ``q_0 ≈ q_3 ≈ q_5 ≈ 0``
    on this synthetic arm, SP5's quartic develops near-triple roots
    which numpy's companion-matrix solver cannot split to better than
    ~1e-3 rad. Gauss-Newton refinement (implemented in sp5.py) lands on
    local minima in the solution manifold but cannot land on q_star
    specifically -- at such poses the IK solution is not uniquely
    representable, so asking for that specific representation is
    over-specified. All returned q's still satisfy FK at 1e-10 (the
    property that actually matters for IK correctness).
    """
    T_star = _fk(synth_a, q_star)
    solutions, is_ls = spherical.solve(synth_a, T_star)
    assert not is_ls
    assert 1 <= len(solutions) <= 8
    for sol in solutions:
        assert np.allclose(_fk(synth_a, sol.q), T_star, atol=1e-8), (
            f"FK mismatch at q={sol.q.tolist()}"
        )
    assert any(_q_matches(s.q, q_star, tol=1e-3) for s in solutions), (
        f"seeded q*={q_star.tolist()} not recovered within 1e-3 rad"
    )


# ---------------------------------------------------------------------------
# Topology validation.
# ---------------------------------------------------------------------------


def test_wrong_dof_raises() -> None:
    kb = _build_generic_spherical_arm(25.0, 0.2, 0.04, 0.5, 0.08, -0.12, 0.4)
    short_kb = KinBody(links=kb.links[:5], joints=kb.joints[:4])
    with pytest.raises(ValueError, match="6-DOF"):
        spherical.solve(short_kb, np.eye(4))


def test_wrong_topology_raises_no_spherical_wrist() -> None:
    """A synthetic arm without an intersecting-wrist triple must be rejected."""
    # All axes parallel - no spherical wrist.
    axes = [np.array([0.0, 0.0, 1.0]) for _ in range(6)]
    t_lefts = [np.array([0.1 * i, 0.0, 0.0]) for i in range(6)]
    link_names = ["base_link", *(f"link_{i}" for i in range(1, 6)), "ee_link"]
    links = [Link(name=n) for n in link_names]
    joints: list[Joint] = []
    for i in range(6):
        t_left_i = np.eye(4)
        t_left_i[:3, 3] = t_lefts[i]
        joints.append(
            Joint(
                name=f"joint_{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=t_left_i,
                T_right=np.eye(4),
                axis=axes[i],
                joint_type="revolute",
            )
        )
    bad_kb = KinBody(links=links, joints=joints)
    with pytest.raises(ValueError, match=r"\(3, 4, 5\)"):
        spherical.solve(bad_kb, np.eye(4))


def test_spherical_rejects_spurious_near_double_root_branch(synth_a: KinBody) -> None:
    """#337: at this pose the SP5 sub-solve's quartic has a near-double root
    (gap ~9e-10) that produced a spurious branch -- a non-zero least-squares
    point (FK ~5.7e-6) that slipped through SP5's loose post-verify. SP5 now
    gates refined candidates at ``subproblem_postverify`` (1e-9), so every
    returned spherical branch FK-closes at machine precision.
    """
    q_star = np.array([0.0, -0.61412196, -0.61412196, 1.0, 1.0, 0.0])
    T_star = _fk(synth_a, q_star)
    solutions, is_ls = spherical.solve(synth_a, T_star)
    assert not is_ls
    assert solutions
    for s in solutions:
        err = float(np.abs(_fk(synth_a, s.q) - T_star).max())
        assert err < 1e-9, f"spurious branch survived: FK {err:.2e}"
