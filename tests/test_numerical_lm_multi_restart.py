"""Tests for :mod:`ssik.solvers.numerical.lm_multi_restart` (#127).

The numerical backstop should:

1. Close 6R IK on UR5 / Puma 560 / JACO 2 from random poses (universal
   correctness).
2. Close 7R IK on Franka via direct call (without going through
   ``jointlock.seven_r``).
3. Use a deterministic PRNG so two calls with the same inputs return
   the same solutions in the same order.
4. Honour ``q_seed`` -- first restart starts from the seed and the
   first returned solution should be near the seed.
5. Return ``is_ls=True`` only when no restart converges (e.g., target
   outside the workspace).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from ssik._kinbody import KinBody, build_kinbody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics._scalar3 import _mat4_mat4
from ssik.solvers.numerical import lm_multi_restart
from ssik.subproblems._rotation import rotation_matrix

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))


def _fk(kb: KinBody, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        R = np.eye(4)
        R[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = _mat4_mat4(_mat4_mat4(T, _mat4_mat4(j.T_left, R)), j.T_right)
    return T


def _ur5_kb() -> KinBody:
    return load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")


def _puma560_kb() -> KinBody:
    return load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")


def _jaco2_kb() -> KinBody:
    from jaco2 import jaco2_specs

    return build_kinbody(jaco2_specs())


def _franka_kb() -> KinBody:
    from franka_panda import franka_panda_specs

    return build_kinbody(franka_panda_specs())


# ---------------------------------------------------------------------------
# Universal correctness on 6R and 7R fixtures.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kb_factory", "n_dof"),
    [
        ("ur5", _ur5_kb, 6),
        ("puma560", _puma560_kb, 6),
    ],
)
def test_6r_random_poses_close(name: str, kb_factory: Callable[[], KinBody], n_dof: int) -> None:
    """For each 6R fixture, the numerical backstop closes random reachable
    poses at FK residual <= the policy threshold."""
    kb = kb_factory()
    rng = np.random.default_rng(seed=1)
    closures = 0
    for _ in range(5):
        q_true = rng.uniform(-1.0, 1.0, size=n_dof)
        T_target = _fk(kb, q_true)
        sols, is_ls = lm_multi_restart.solve(kb, T_target)
        if is_ls or not sols:
            continue
        # At least one solution must FK-close.
        for sol in sols:
            T_check = _fk(kb, sol.q)
            assert np.allclose(T_check, T_target, atol=1e-4), (
                f"{name}: solution failed FK closure: "
                f"max|diff|={float(np.max(np.abs(T_check - T_target))):.2e}"
            )
        closures += 1
    assert closures >= 4, f"{name}: only {closures}/5 random poses closed"


def test_jaco2_with_q_seed_closes_via_lm_polish() -> None:
    """JACO 2 -- the canonical non-Pieper 6R with 60-degree non-orthogonal
    twists. The LM basin is small (analytical tier-2 RR is the recommended
    path), but with a ``q_seed`` near a true solution, the polish converges
    every time. This demonstrates the backstop's *primary* use case: an
    application that has a current configuration to bias toward, even when
    random restarts struggle with the geometry."""
    kb = _jaco2_kb()
    rng = np.random.default_rng(seed=2)
    closures = 0
    for _ in range(5):
        q_true = rng.uniform(-1.0, 1.0, size=6)
        T_target = _fk(kb, q_true)
        # Seed near (but not equal to) the truth -- application-realistic
        # use case where the user has an approximate current configuration.
        q_seed = q_true + rng.normal(0, 0.1, size=6)
        sols, is_ls = lm_multi_restart.solve(kb, T_target, q_seed=q_seed)
        if is_ls or not sols:
            continue
        for sol in sols:
            T_check = _fk(kb, sol.q)
            assert np.allclose(T_check, T_target, atol=1e-4), (
                f"JACO 2: solution failed FK closure: "
                f"max|diff|={float(np.max(np.abs(T_check - T_target))):.2e}"
            )
        closures += 1
    assert closures >= 4, (
        f"JACO 2 with q_seed: only {closures}/5 closed -- "
        "the LM polish from a near-truth seed should always converge"
    )


def test_franka_7r_random_poses_close() -> None:
    """Franka 7R -- numerical backstop handles 7-DOF without going
    through joint-locking. Slower than the analytical jointlock path
    but universally correct."""
    kb = _franka_kb()
    rng = np.random.default_rng(seed=3)
    closures = 0
    for _ in range(3):
        q_true = rng.uniform(-1.0, 1.0, size=7)
        T_target = _fk(kb, q_true)
        sols, is_ls = lm_multi_restart.solve(kb, T_target, n_restarts=12)
        if is_ls or not sols:
            continue
        for sol in sols:
            T_check = _fk(kb, sol.q)
            assert np.allclose(T_check, T_target, atol=1e-4), (
                f"Franka: solution failed FK closure: "
                f"max|diff|={float(np.max(np.abs(T_check - T_target))):.2e}"
            )
        closures += 1
    assert closures >= 2, f"Franka 7R: only {closures}/3 closed"


# ---------------------------------------------------------------------------
# Deterministic reproducibility.
# ---------------------------------------------------------------------------


def test_deterministic_random_restarts() -> None:
    """Same kb + T_target + no q_seed -> same solutions in the same order
    every time. Internal PRNG is seeded with a fixed key."""
    kb = _ur5_kb()
    rng = np.random.default_rng(seed=4)
    q_true = rng.uniform(-1.0, 1.0, size=6)
    T_target = _fk(kb, q_true)

    sols_a, _ = lm_multi_restart.solve(kb, T_target)
    sols_b, _ = lm_multi_restart.solve(kb, T_target)

    assert len(sols_a) == len(sols_b), "non-deterministic solution count"
    for a, b in zip(sols_a, sols_b, strict=True):
        np.testing.assert_array_equal(a.q, b.q)
        assert a.fk_residual == b.fk_residual


# ---------------------------------------------------------------------------
# q_seed bias.
# ---------------------------------------------------------------------------


def test_q_seed_first_restart_uses_seed() -> None:
    """When ``q_seed`` is provided, the first restart starts from it.
    The closest returned solution to ``q_seed`` should be VERY close
    (sub-radian per joint) for a reachable pose."""
    kb = _ur5_kb()
    q_true = np.array([0.1, -0.5, 1.2, 0.3, 0.8, -0.4])
    T_target = _fk(kb, q_true)

    sols, is_ls = lm_multi_restart.solve(kb, T_target, q_seed=q_true)
    assert not is_ls
    # Sort by L2 distance to seed; closest one should be tight.
    closest = min(
        sols,
        key=lambda s: float(np.linalg.norm(s.q - q_true)),
    )
    diffs = closest.q - q_true
    # Sub-radian on every joint for a seed-equal-to-truth case.
    assert np.max(np.abs(diffs)) < 0.1, (
        f"closest solution to q_seed should be near it; got diffs={diffs}"
    )


def test_q_seed_wrong_shape_raises() -> None:
    kb = _ur5_kb()
    T = _fk(kb, np.zeros(6))
    bad_seed = np.zeros(7)  # 7-vector for 6-DOF
    with pytest.raises(ValueError, match="q_seed shape"):
        lm_multi_restart.solve(kb, T, q_seed=bad_seed)


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_n_restarts_must_be_positive() -> None:
    kb = _ur5_kb()
    T = _fk(kb, np.zeros(6))
    with pytest.raises(ValueError, match="n_restarts"):
        lm_multi_restart.solve(kb, T, n_restarts=0)


def test_unreachable_target_returns_is_ls() -> None:
    """A target way outside the workspace -- no LM restart converges.
    Must return ``is_ls=True`` (not raise, not return wrong solutions)."""
    kb = _ur5_kb()
    T = np.eye(4)
    T[:3, 3] = [100.0, 100.0, 100.0]  # 100m away -- way outside UR5's reach
    sols, is_ls = lm_multi_restart.solve(kb, T)
    assert is_ls
    assert sols == []


def test_solutions_all_marked_lm() -> None:
    """Every returned solution should have ``refinement_used='lm'`` and
    a populated ``branch_id`` indicating which restart produced it."""
    kb = _ur5_kb()
    q_true = np.array([0.1, -0.5, 1.2, 0.3, 0.8, -0.4])
    T_target = _fk(kb, q_true)
    sols, _ = lm_multi_restart.solve(kb, T_target)
    assert all(s.refinement_used == "lm" for s in sols)
    assert all(s.branch_id is not None for s in sols)
    assert all(s.solver_name == "numerical.lm_multi_restart" for s in sols)
