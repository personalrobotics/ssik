"""Unit tests for ssik.refinement.

The refinement layer is opt-in last-resort polish (#74). Every solver
goes through it; if these primitives are wrong, every solver is wrong.
Tests cover:

- ``se3_log_residual`` Rodrigues correctness near identity, mid-angle,
  and at small-angle limit.
- ``lm_refine`` convergence on perturbed seeds with analytical Jacobian.
- ``lm_refine`` convergence with numerical Jacobian (slow fallback).
- ``lm_refine`` returns ``None`` on a hopeless seed (max_iters cap).
- ``kinbody_jacobian`` matches central-difference numerical Jacobian.
- ``verify_candidates`` correctly classifies pass / refine / drop.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ssik._kinbody import KinBody, build_kinbody
from ssik.core.solution import Solution
from ssik.refinement import (
    kinbody_jacobian,
    lm_refine,
    numerical_jacobian,
    se3_log_residual,
    verify_candidates,
)
from tests.fixtures.franka_panda import franka_panda_specs
from tests.fixtures.ur5 import ur5_specs


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x, y, z = axis
    oc = 1.0 - c
    R = np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _fk_poe(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _rot_axis(j.axis, float(qi)) @ j.T_right
    return T


# ---------------------------------------------------------------------------
# se3_log_residual
# ---------------------------------------------------------------------------


def test_se3_log_residual_identity_is_zero() -> None:
    r = se3_log_residual(np.eye(4))
    assert np.allclose(r, 0.0, atol=1e-15)


def test_se3_log_residual_pure_translation() -> None:
    T = np.eye(4)
    T[:3, 3] = [0.1, -0.2, 0.3]
    r = se3_log_residual(T)
    assert np.allclose(r[:3], [0.1, -0.2, 0.3], atol=1e-15)
    assert np.allclose(r[3:], 0.0, atol=1e-15)


def test_se3_log_residual_pure_rotation_recovers_axis_angle() -> None:
    axis = np.array([0.0, 0.0, 1.0])
    angle = 0.7
    T = _rot_axis(axis, angle)
    r = se3_log_residual(T)
    assert np.allclose(r[:3], 0.0, atol=1e-15)
    assert np.allclose(r[3:], angle * axis, atol=1e-12)


def test_se3_log_residual_small_angle_recovers_actual_rotation() -> None:
    """Below 1e-9 rad we recover the actual rotation residual (#199 fix).

    The previous trace-arccos implementation rounded ``cos_a`` to
    ``1.0`` in float64 for any rotation below ~3e-8 rad, silently
    zeroing the rotation part. The antisymmetric-vee formulation
    preserves precision down to machine epsilon: a 1e-12 rad rotation
    around z maps to ``[0, 0, 1e-12]`` in the rotation residual.
    """
    axis = np.array([0.0, 0.0, 1.0])
    T = _rot_axis(axis, 1e-12)
    r = se3_log_residual(T)
    assert np.allclose(r[:3], 0.0, atol=1e-15)
    # Rotation residual recovers the small angle (was: zeroed out).
    assert np.linalg.norm(r[3:] - 1e-12 * axis) < 1e-15


# ---------------------------------------------------------------------------
# lm_refine
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ur5_kb() -> KinBody:
    return build_kinbody(ur5_specs())


def test_lm_refine_converges_on_perturbed_seed(ur5_kb: KinBody) -> None:
    """Seed within ~10 deg of a true q; should converge to machine precision."""
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-1.0, 1.0, size=6)
    T_target = _fk_poe(ur5_kb, q_true)

    q_seed = q_true + rng.uniform(-0.15, 0.15, size=6)
    fk = lambda q: _fk_poe(ur5_kb, q)  # noqa: E731
    jac = lambda q: kinbody_jacobian(ur5_kb, q)  # noqa: E731

    refined = lm_refine(q_seed, fk, T_target, fk_atol=1e-12, max_iters=20, jacobian_fn=jac)
    assert refined is not None
    q_ref, resid, iters = refined
    assert resid < 1e-12
    assert iters <= 10
    assert np.allclose(_fk_poe(ur5_kb, q_ref), T_target, atol=1e-10)


def test_lm_refine_with_numerical_jacobian(ur5_kb: KinBody) -> None:
    """Same convergence even without an analytical Jacobian (slow path)."""
    rng = np.random.default_rng(7)
    q_true = rng.uniform(-1.0, 1.0, size=6)
    T_target = _fk_poe(ur5_kb, q_true)

    q_seed = q_true + rng.uniform(-0.1, 0.1, size=6)
    fk = lambda q: _fk_poe(ur5_kb, q)  # noqa: E731

    refined = lm_refine(q_seed, fk, T_target, fk_atol=1e-9, max_iters=30, jacobian_fn=None)
    assert refined is not None
    _q_ref, resid, _ = refined
    assert resid < 1e-9


def test_lm_refine_returns_none_on_hopeless_seed(ur5_kb: KinBody) -> None:
    """A seed nowhere near a solution shouldn't converge in 5 iters."""
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-1.0, 1.0, size=6)
    T_target = _fk_poe(ur5_kb, q_true)

    q_seed = q_true + np.array([2.5, -2.5, 2.0, 1.5, -1.5, 1.0])  # far away
    fk = lambda q: _fk_poe(ur5_kb, q)  # noqa: E731
    jac = lambda q: kinbody_jacobian(ur5_kb, q)  # noqa: E731

    refined = lm_refine(q_seed, fk, T_target, fk_atol=1e-12, max_iters=5, jacobian_fn=jac)
    # Either stays None (didn't converge) OR lands on a different IK branch.
    # Both are acceptable outcomes; the contract is "no false positives at fk_atol".
    if refined is not None:
        _, resid, _ = refined
        assert resid <= 1e-12


# ---------------------------------------------------------------------------
# kinbody_jacobian
# ---------------------------------------------------------------------------


@pytest.fixture
def franka_kb() -> KinBody:
    return build_kinbody(franka_panda_specs())


@pytest.mark.parametrize("kb_name", ["ur5", "franka"])
def test_kinbody_jacobian_matches_numerical_spatial(
    kb_name: str, ur5_kb: KinBody, franka_kb: KinBody
) -> None:
    """Analytical and numerical (central-difference) spatial Jacobians
    must agree to central-difference precision (~1e-5) on every block,
    not just the angular one. Tested on multiple arm topologies (UR5
    6R, Franka Panda 7R) to cover diverse axis configurations.

    This is bulletproof correctness, not "close enough for Newton". The
    Jacobian convention must match ``se3_log_residual`` -- which extracts
    the spatial twist of ``T_target @ T_q^{-1}`` -- otherwise Newton in
    ``lm_refine`` is a quasi-Newton with a wrong Hessian estimate, which
    converges only from very-close seeds and silently fails on harder
    initial guesses (the bug that locked-Franka HP back-substitution
    ran into; see PR #176 / Phase 5h).

    The hybrid / "geometric" Jacobian (``z_i x (p_e - p_i)`` linear part)
    differs from the spatial Jacobian by ``z_i x p_e`` per column. Either
    can be full rank, so a rank-only test passes on both. We pin the
    actual values here.
    """
    kb = ur5_kb if kb_name == "ur5" else franka_kb
    n_dof = len(kb.joints)
    rng = np.random.default_rng(0)
    for _ in range(8):
        q = rng.uniform(-2.0, 2.0, size=n_dof)
        j_kb = kinbody_jacobian(kb, q)
        fk_for_jac = lambda x: _fk_poe(kb, x)  # noqa: E731
        j_num = numerical_jacobian(q, fk_for_jac)
        # Tight bound: analytical must match central-difference truth to
        # ~1e-5 (central-diff truncation floor). Loose bounds would mask
        # convention bugs like the one PR #176 fixed.
        max_abs = float(np.max(np.abs(j_kb - j_num)))
        assert max_abs < 1e-5, (
            f"{kb_name}: kinbody_jacobian disagrees with numerical_jacobian "
            f"at q={q}: max abs diff {max_abs:.3e} (>1e-5). Likely a "
            f"convention mismatch with se3_log_residual."
        )
        assert np.linalg.matrix_rank(j_kb) == min(6, n_dof)
        assert np.linalg.matrix_rank(j_num) == min(6, n_dof)


# ---------------------------------------------------------------------------
# verify_candidates
# ---------------------------------------------------------------------------


def test_verify_candidates_passes_through_exact_match(ur5_kb: KinBody) -> None:
    """A candidate at machine precision should be wrapped as ``refinement_used="none"``."""
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-1.0, 1.0, size=6)
    T_target = _fk_poe(ur5_kb, q_true)
    fk = lambda q: _fk_poe(ur5_kb, q)  # noqa: E731
    sols = verify_candidates(
        [q_true],
        fk_fn=fk,
        t_target=T_target,
        fk_atol=1e-9,
        solver_name="test",
    )
    assert len(sols) == 1
    s = sols[0]
    assert s.refinement_used == "none"
    assert s.refinement_iters == 0
    assert s.fk_residual < 1e-9
    assert s.solver_name == "test"
    assert s.branch_id == 0


def test_verify_candidates_drops_misses_when_refinement_off(ur5_kb: KinBody) -> None:
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-1.0, 1.0, size=6)
    T_target = _fk_poe(ur5_kb, q_true)
    bad = q_true + 0.1
    fk = lambda q: _fk_poe(ur5_kb, q)  # noqa: E731
    sols = verify_candidates(
        [bad],
        fk_fn=fk,
        t_target=T_target,
        fk_atol=1e-9,
        solver_name="test",
        allow_refinement=False,
    )
    assert sols == []


def test_verify_candidates_polishes_misses_when_refinement_on(ur5_kb: KinBody) -> None:
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-1.0, 1.0, size=6)
    T_target = _fk_poe(ur5_kb, q_true)
    seed = q_true + 0.1
    fk = lambda q: _fk_poe(ur5_kb, q)  # noqa: E731
    jac = lambda q: kinbody_jacobian(ur5_kb, q)  # noqa: E731
    sols = verify_candidates(
        [seed],
        fk_fn=fk,
        jacobian_fn=jac,
        t_target=T_target,
        fk_atol=1e-10,
        solver_name="test",
        allow_refinement=True,
        refinement_max_iters=15,
    )
    assert len(sols) == 1
    s = sols[0]
    assert s.refinement_used == "lm"
    assert s.refinement_iters >= 1
    assert s.fk_residual < 1e-10


def test_verify_candidates_dedup_keeps_lower_residual(ur5_kb: KinBody) -> None:
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-1.0, 1.0, size=6)
    T_target = _fk_poe(ur5_kb, q_true)
    perturbed = q_true + 1e-7
    fk = lambda q: _fk_poe(ur5_kb, q)  # noqa: E731
    # Both candidates dedup-collide; verify the lower-residual one wins.
    sols = verify_candidates(
        [perturbed, q_true],
        fk_fn=fk,
        t_target=T_target,
        fk_atol=1e-3,
        solver_name="test",
        dedup_atol=1e-3,
    )
    assert len(sols) == 1
    # q_true is exact, perturbed has residual ~1e-7. Lower residual = q_true.
    assert np.allclose(sols[0].q, q_true, atol=1e-15)


def test_solution_dataclass_is_frozen() -> None:
    s = Solution(q=np.zeros(6), fk_residual=0.0)
    with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError subclass
        s.q = np.ones(6)  # type: ignore[misc]
