"""Damped Levenberg-Marquardt with multi-restart -- universal IK backstop.

For any 6R / 7R / NR chain, regardless of analytical decomposability:
run ``lm_refine`` from N seeded starting points and collect every
restart that converges. Restarts are deterministic (PRNG seeded with a
fixed key) so two calls with the same inputs return identical solutions.

API matches the analytical-solver protocol so the dispatcher can route
to it like any other solver. Two solver-specific kwargs:

  * ``q_seed`` -- if provided, use as the first restart and bias subsequent
    restarts as small perturbations around it. Useful when the caller has
    a current configuration to bias toward (mj_manipulator-style).
  * ``n_restarts`` -- number of random seeds to try. Default 8 covers the
    typical solution count for commercial 6R / 7R arms.

Phase 4 Cython compilation will collapse the per-iteration cost to
~50 us per LM step -> ~1.5 ms per restart -> ~12 ms total for 8 restarts.
That's competitive with mink (~100 ms via scipy) while remaining fully
analytical-IK-aligned (no scipy dependency, no symbolic precompute).

This is opt-in -- users invoke it directly or via a dispatcher hint;
it's not the default analytical-IK path.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.refinement import dedup_by_wrap_close, kinbody_jacobian, lm_refine
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["solve"]

_SOLVER_NAME = "numerical.lm_multi_restart"
_LOG = logging.getLogger(__name__)
_DEFAULT_N_RESTARTS = 16
_DEFAULT_MAX_ITERS = 50
_DEFAULT_RNG_SEED = 42  # deterministic restart seeds


def _kinbody_fk(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """POE forward kinematics for any KinBody. Used as ``fk_fn`` for ``lm_refine``."""
    T = np.eye(4, dtype=np.float64)
    for joint, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4, dtype=np.float64)
        rot[:3, :3] = rotation_matrix(joint.axis, float(qi))
        T = T @ joint.T_left @ rot @ joint.T_right
    return T


def _sample_random_q(kb: KinBody, rng: np.random.Generator) -> NDArray[np.float64]:
    """Sample a random q in each joint's reachable range.

    For revolute joints with ``limits=(lo, hi)``, sample uniformly in
    ``[lo, hi]``. Continuous joints (``limits=None``) sample in
    ``[-pi, pi]`` (one full revolution -- the unique sample space).
    Prismatic joints with ``limits=None`` sample in ``[-1, 1]`` (a
    bounded default; users with custom prismatic geometry should set
    explicit limits).
    """
    n = len(kb.joints)
    q = np.empty(n, dtype=np.float64)
    for i, joint in enumerate(kb.joints):
        if joint.limits is not None:
            lo, hi = joint.limits
        elif joint.joint_type == "revolute":
            lo, hi = -np.pi, np.pi
        else:  # prismatic with no limits
            lo, hi = -1.0, 1.0
        q[i] = rng.uniform(lo, hi)
    return q


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = True,
    refinement_max_iters: int = _DEFAULT_MAX_ITERS,
    n_restarts: int = _DEFAULT_N_RESTARTS,
    q_seed: NDArray[np.float64] | None = None,
) -> tuple[list[Solution], bool]:
    """Damped LM with multi-restart -- universal IK backstop.

    For any 6R / 7R chain (analytical or not), runs Newton-on-SE(3)-log-
    residual from N seeded starting points and collects every restart that
    converges. Returns deduplicated solutions sorted by FK residual
    (smallest first).

    :param kb: POE-normalised :class:`KinBody`.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerance policy. ``subproblem_numerical`` is the
        per-restart FK convergence threshold. ``subproblem_dedup`` is the
        wrap-to-pi distance below which two solutions collapse to one.
    :param allow_refinement: kept for solver-protocol consistency. The
        whole solver IS Newton refinement, so this kwarg has no effect;
        accepted to match :func:`ssik.solvers.ikgeo.spherical_two_parallel.solve`
        and friends.
    :param refinement_max_iters: per-restart LM iteration cap. Default 30
        is enough for the easy basin; raise if your fixture has tight
        Newton-method convergence.
    :param n_restarts: number of LM restarts. Default 8 is typical for
        commercial arms (6-8 IK branches). Raise for arms with more
        branches; lower for fast-but-rough.
    :param q_seed: optional reference configuration. When provided, used
        as the first restart and biases subsequent restarts via small
        gaussian perturbations -- favours solutions near ``q_seed``
        without losing global coverage.

    :returns: ``(solutions, is_ls)``. Each :class:`Solution` carries
        ``refinement_used="lm"`` (every solution came from refinement)
        and ``branch_id`` indexing the restart that produced it.
    """
    del allow_refinement  # retained for protocol compatibility; see docstring
    n_dof = len(kb.joints)
    if n_restarts <= 0:
        raise ValueError(f"n_restarts must be positive; got {n_restarts}")

    fk_fn = lambda q: _kinbody_fk(kb, q)  # noqa: E731
    jacobian_fn = lambda q: kinbody_jacobian(kb, q)  # noqa: E731

    # Build the restart list: q_seed (if given) + (n_restarts - 1) random.
    rng = np.random.default_rng(seed=_DEFAULT_RNG_SEED)
    starts: list[NDArray[np.float64]] = []
    if q_seed is not None:
        seed_arr = np.asarray(q_seed, dtype=np.float64)
        if seed_arr.shape != (n_dof,):
            raise ValueError(f"q_seed shape {seed_arr.shape} doesn't match chain DOF ({n_dof},)")
        starts.append(seed_arr.copy())
        # Perturbations around q_seed: small gaussian sigma=0.3 rad keeps
        # us in the seed's basin while testing nearby branches.
        for _ in range(n_restarts - 1):
            starts.append(seed_arr + 0.3 * rng.standard_normal(n_dof))
    else:
        for _ in range(n_restarts):
            starts.append(_sample_random_q(kb, rng))

    fk_atol = policy.subproblem_numerical
    candidates: list[Solution] = []
    for restart_idx, q0 in enumerate(starts):
        result = lm_refine(
            q0,
            fk_fn,
            T_target,
            fk_atol=fk_atol,
            max_iters=refinement_max_iters,
            jacobian_fn=jacobian_fn,
        )
        if result is None:
            continue
        q_conv, fk_resid, iters = result
        candidates.append(
            Solution(
                q=q_conv,
                fk_residual=fk_resid,
                refinement_used="lm",
                refinement_iters=iters,
                branch_id=restart_idx,
                solver_name=_SOLVER_NAME,
            )
        )

    # Sort by FK residual (smallest first) for stable downstream selection,
    # then dedup. dedup_by_wrap_close keeps the first occurrence on
    # collision -- with sort-by-residual that's the lowest-residual
    # representative.
    candidates.sort(key=lambda s: s.fk_residual)
    solutions = dedup_by_wrap_close(candidates, policy.subproblem_dedup)
    _LOG.info(
        "%s: %d/%d restarts converged -> %d unique solutions (is_ls=%s)",
        _SOLVER_NAME,
        len(candidates),
        n_restarts,
        len(solutions),
        len(solutions) == 0,
    )
    return solutions, len(solutions) == 0
