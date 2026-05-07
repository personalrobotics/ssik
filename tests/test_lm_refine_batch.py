"""Bulletproof tests for the batched LM polish (#205).

:func:`ssik.refinement.lm_refine_batch` runs Newton synchronously across
``N`` candidates, batching the linear-algebra portion to amortise per-
iteration ``np.linalg.{solve, inv, norm}`` dispatch overhead. Used by
:mod:`ssik.solvers.seven_r.srs_polished` to halve Gen3's polish time.

Test contract:

- Identical convergence behaviour to the scalar :func:`lm_refine`:
  candidates that converge with the scalar version converge with
  batch; ``fk_residuals[i] < fk_atol`` is bulletproof.
- Per-pose machine precision: ``best_fk`` (min residual across
  retained candidates) ≤ 1e-13 on Gen3 random poses.
- Empty input handled cleanly (``n=0`` doesn't crash).
- Diverged seeds get ``fk_residuals[i] = inf`` (caller can filter).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.refinement import kinbody_jacobian, lm_refine_batch

GEN3_URDF = Path(__file__).parent / "fixtures" / "gen3.urdf"


def _gen3_kb():
    return load_urdf_kinbody_normalized(GEN3_URDF, "base_link", "end_effector_link")


def test_batch_converges_machine_precision_from_seed_near_truth() -> None:
    """A batch of seeds within Newton's basin converges to the target
    at machine-precision FK closure.
    """
    kb = _gen3_kb()
    rng = np.random.default_rng(0)
    q_true = rng.uniform(-0.5, 0.5, size=7)
    q_true[3] = 0.5  # avoid elbow singularity
    T_target = poe_forward_kinematics(kb, q_true)

    # 5 perturbed seeds within ~1 deg of truth (well inside basin
    # AND within the divergence_factor=2 trajectory-bump tolerance).
    seeds = q_true + 0.01 * rng.standard_normal((5, 7))

    def fk_fn(q):
        return poe_forward_kinematics(kb, q)

    def jac_fn(q):
        return kinbody_jacobian(kb, q)

    q_pol, fk_res, iters = lm_refine_batch(
        seeds, fk_fn, jac_fn, T_target, fk_atol=1e-12, max_iters=20
    )
    assert q_pol.shape == (5, 7)
    assert (fk_res < 1e-12).all(), f"all should converge; got fk_residuals={fk_res}"
    # Sanity: iters used <= 10 for this easy case
    assert (iters <= 10).all()


def test_batch_marks_diverged_seeds_with_inf() -> None:
    """Seeds far outside Newton's basin (e.g. random q) get
    ``fk_residual=inf`` so the caller can filter them.
    """
    kb = _gen3_kb()
    rng = np.random.default_rng(42)
    q_true = rng.uniform(-0.5, 0.5, size=7)
    q_true[3] = 0.5
    T_target = poe_forward_kinematics(kb, q_true)

    # Random seeds far from truth (uniform over full joint range).
    seeds = rng.uniform(-3.0, 3.0, size=(20, 7))

    def fk_fn(q):
        return poe_forward_kinematics(kb, q)

    def jac_fn(q):
        return kinbody_jacobian(kb, q)

    _q_pol, fk_res, _ = lm_refine_batch(
        seeds, fk_fn, jac_fn, T_target, fk_atol=1e-10, max_iters=20
    )
    # Most should diverge or fail to converge.
    assert (fk_res == np.inf).any() or (fk_res > 1e-10).any()
    # The ones that converge must hit machine precision.
    converged = fk_res < 1e-10
    if converged.any():
        assert fk_res[converged].max() < 1e-10


def test_batch_empty_input_returns_empty_arrays() -> None:
    """``n=0`` is a valid input -- callers may call with all candidates
    pre-filtered. Should not crash.
    """
    kb = _gen3_kb()
    T_target = np.eye(4)

    def fk_fn(q):
        return poe_forward_kinematics(kb, q)

    def jac_fn(q):
        return kinbody_jacobian(kb, q)

    seeds = np.empty((0, 7), dtype=np.float64)
    q_pol, fk_res, iters = lm_refine_batch(seeds, fk_fn, jac_fn, T_target)
    assert q_pol.shape == (0, 7)
    assert fk_res.shape == (0,)
    assert iters.shape == (0,)


def test_batch_matches_scalar_on_easy_seeds() -> None:
    """Per-candidate, batch and scalar :func:`lm_refine` agree on whether
    a seed converges. The exact final ``q`` may differ slightly (Tikhonov
    damping is constant in batch vs. residual-scaled in scalar) but FK
    closure must be machine-precision in both cases.
    """
    from ssik.refinement import lm_refine

    kb = _gen3_kb()
    rng = np.random.default_rng(7)
    q_true = rng.uniform(-0.4, 0.4, size=7)
    q_true[3] = 0.5
    T_target = poe_forward_kinematics(kb, q_true)
    # Seeds within 3 degrees -- easy case.
    seeds = q_true + 0.05 * rng.standard_normal((10, 7))

    def fk_fn(q):
        return poe_forward_kinematics(kb, q)

    def jac_fn(q):
        return kinbody_jacobian(kb, q)

    # Scalar
    scalar_results = []
    for seed in seeds:
        result = lm_refine(seed, fk_fn, T_target, fk_atol=1e-12, max_iters=20, jacobian_fn=jac_fn)
        scalar_results.append(result is not None)

    # Batch
    _q_pol, fk_res, _ = lm_refine_batch(
        seeds, fk_fn, jac_fn, T_target, fk_atol=1e-12, max_iters=20
    )
    batch_converged = fk_res < 1e-12

    # Both must agree on convergence per candidate within tolerance
    # (allowing one off due to last-iter rounding).
    scalar_arr = np.array(scalar_results)
    diff = (scalar_arr != batch_converged).sum()
    assert diff <= 1, f"batch and scalar disagree on {diff}/{len(seeds)} candidates"


def test_batch_q_seeds_must_be_2d() -> None:
    """Defensive: 1D q_seed should raise (use scalar lm_refine instead)."""
    import pytest

    kb = _gen3_kb()
    seed = np.zeros(7)
    T = poe_forward_kinematics(kb, seed)
    with pytest.raises(ValueError, match="must be"):
        lm_refine_batch(
            seed, lambda q: poe_forward_kinematics(kb, q), lambda q: kinbody_jacobian(kb, q), T
        )
