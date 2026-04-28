"""Universal 7R IK via joint-locking + sweep over a tier-0/1 6R sub-solver.

For any 7-DOF revolute arm, fixing one joint collapses the chain to a 6R
arm whose IK is analytical for most commercial topologies. This solver:

1. **Auto-selects the lock joint** (done once per KinBody) by trying each
   of the 7 joints and ranking the resulting 6R sub-chain by the
   strongest tier-0 / tier-1 solver it matches. Pure topology test, no
   pose involved.
2. Sweeps the lock joint over N samples (or a user-supplied list).
3. At each sample, dispatches the 6R sub-chain to the best-matching
   ikgeo solver and collects its IK solutions, padded back to 7D with
   the locked angle.
4. Deduplicates the full 7D output in angle space.

This one solver covers Franka Panda, FR3, KUKA iiwa, Flexiv Rizon,
Kinova Gen3, uFactory xArm7, and any other 7R arm -- all by reusing
the tier-0 speed of the inner 6R solvers.

## Completeness note

Sampling a single redundant joint returns a 1D slice of the 2D IK
manifold. With N=16 default samples we get ~16 * (up-to-8) = up to
128 candidate solutions per target; after dedup typically 16-32 are
distinct. Users who need exact redundancy parametrisation should
eventually use a specialist solver (``specialist.geofik`` for Franka,
``specialist.stereo_sew`` for iiwa / SRS arms). This generic wrapper
is the "works now, works everywhere" fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import Joint, KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.predicates import (
    axis_parallel,
    three_consecutive_intersecting,
    three_consecutive_parallel,
)
from ssik.refinement import dedup_by_wrap_close
from ssik.solvers.ikgeo import (
    gen_six_dof,
    spherical,
    spherical_two_intersecting,
    spherical_two_parallel,
    three_parallel,
    two_intersecting,
    two_parallel,
)
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["choose_lock_joint", "solve"]

_DEFAULT_SAMPLES = 16
_SOLVER_NAME = "jointlock.seven_r"


def _lock_joint(kb: KinBody, lock_idx: int, q_lock: float) -> KinBody:
    """Return a 6R KinBody with joint ``lock_idx`` folded out of the chain.

    The locked rotation ``R = R(axes[lock_idx], q_lock)`` is propagated
    through every subsequent joint's axis and offset via similarity
    transform, then absorbed into the last joint's ``T_right`` (composing
    with any existing ``R_home`` rotation). This preserves the POE
    invariant that ``T_left`` and (for non-final joints) ``T_right`` are
    pure translations — the assumption every IK-Geo solver relies on.

    Algebra: at the lock point the chain has
    ``... T_left[k] @ R @ T_right[k] @ T_left[k+1] @ R(axes[k+1], q[k+1]) @ ...``
    Pushing R to the right by similarity (``R @ M = M' @ R`` where ``M'``
    has axes/translations rotated by R), R eventually lands at the
    rightmost position and becomes part of ``T_right[-1]``.
    """
    if not (0 <= lock_idx < len(kb.joints)):
        raise IndexError(f"lock_idx {lock_idx} out of range")

    locked = kb.joints[lock_idx]
    last_idx = len(kb.joints) - 1
    R_lock = rotation_matrix(locked.axis, q_lock)

    new_joints: list[Joint] = []
    for i, j in enumerate(kb.joints):
        if i == lock_idx:
            continue
        if i < lock_idx:
            # Pre-lock joints unchanged.
            new_joints.append(replace(j, dof_index=len(new_joints)))
            continue

        # Post-lock joints: rotate axis and translations by R_lock as
        # the locked rotation propagates through them via similarity.
        # The rotation BLOCK of T_left and T_right stays identity for
        # intermediate joints; only the LAST joint's T_right absorbs
        # R_lock as its final rotation (composing with any R_home).
        new_axis = R_lock @ j.axis
        new_T_left = np.eye(4)
        new_T_left[:3, 3] = R_lock @ j.T_left[:3, 3]
        new_T_right = np.eye(4)
        new_T_right[:3, 3] = R_lock @ j.T_right[:3, 3]
        if i == last_idx:
            # End of chain: R_lock lands here, composed with any existing
            # R_home rotation in the original T_right.
            new_T_right[:3, :3] = R_lock @ j.T_right[:3, :3]

        if i == lock_idx + 1:
            # Absorb the locked-joint translation contribution: the chain
            # had ``T_left[k] @ R @ T_right[k] @ T_left[k+1]`` which (in
            # the typical POE case where T_left[k] and T_right[k] are pure
            # translations) factors as
            #   trans(t_left[k] + R(t_right[k] + t_left[k+1])) @ rot4(R)
            # The rot4(R) then propagates through subsequent joints via
            # similarity, eventually landing in T_right[-1].
            t_combined = locked.T_left[:3, 3] + R_lock @ (locked.T_right[:3, 3] + j.T_left[:3, 3])
            new_T_left[:3, 3] = t_combined
            new_T_left[:3, :3] = np.eye(3)

        new_joints.append(
            replace(
                j,
                axis=new_axis,
                T_left=new_T_left,
                T_right=new_T_right,
                dof_index=len(new_joints),
            )
        )

    # Drop one link to preserve N+1 links / N joints. Drop the link
    # immediately after the locked joint (or the locked joint's own link
    # if locking the last joint).
    drop_link_idx = lock_idx if lock_idx == last_idx else lock_idx + 1
    new_links = [link for i, link in enumerate(kb.links) if i != drop_link_idx]

    return KinBody(links=new_links, joints=new_joints)


# ---------------------------------------------------------------------------
# Lock-joint selection by topology rank.
# ---------------------------------------------------------------------------


def _topology_rank(sub_kb: KinBody, policy: TolerancePolicy) -> tuple[int, str]:
    """Score a 6R KinBody by its best matching solver family.

    Lower rank = better (faster, more specialized). Returns (rank,
    solver_name). Unmatched sub-chains fall through to the tier-2
    rank.
    """
    # Tier-0 closed-form: rank 0 = best.
    if three_consecutive_parallel(sub_kb.joints, policy) == (1, 2, 3):
        return (0, "three_parallel")
    if three_consecutive_intersecting(sub_kb.joints, policy) == (3, 4, 5):
        p1_norm = float(np.linalg.norm(sub_kb.joints[1].T_left[:3, 3]))
        p1_on_axis = p1_norm < policy.axis_intersect
        j12_parallel = axis_parallel(sub_kb.joints[1].axis, sub_kb.joints[2].axis, policy)
        if p1_on_axis and j12_parallel:
            # Both specializations match; pick parallel (smaller IK set typically).
            return (0, "spherical_two_parallel")
        if j12_parallel:
            return (0, "spherical_two_parallel")
        if p1_on_axis:
            return (0, "spherical_two_intersecting")
        return (1, "spherical")  # generic spherical wrist
    # Tier-1 univariate: rank 2.
    p5_norm = float(np.linalg.norm(sub_kb.joints[5].T_left[:3, 3]))
    if p5_norm < policy.axis_intersect:
        return (2, "two_intersecting")
    if axis_parallel(sub_kb.joints[1].axis, sub_kb.joints[2].axis, policy):
        return (2, "two_parallel")
    # Tier-2 fallback (slow but correct).
    return (3, "gen_six_dof")


def choose_lock_joint(kb: KinBody, policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY) -> int:
    """Return the joint index whose locked sub-chain (at the rest pose)
    matches the best-ranked solver.

    Pure function of kinematic topology, called once per arm. The
    *actual* inner-solver dispatch happens per-sample inside
    :func:`solve` because rotating downstream axes by ``R_lock`` can
    change which tier-0/1 specialization applies at each q_lock.
    """
    if len(kb.joints) != 7:
        raise ValueError(f"jointlock.seven_r requires 7 joints; got {len(kb.joints)}")

    best: tuple[int, int] | None = None  # (rank, lock_idx)
    for lock_idx in range(7):
        sub_kb = _lock_joint(kb, lock_idx, 0.0)
        rank, _ = _topology_rank(sub_kb, policy)
        if best is None or rank < best[0]:
            best = (rank, lock_idx)

    assert best is not None
    return best[1]


# ---------------------------------------------------------------------------
# Dispatch: map solver name -> solver function.
# ---------------------------------------------------------------------------


def _dispatch(
    solver_name: str,
    sub_kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy,
    *,
    allow_refinement: bool,
    refinement_max_iters: int,
) -> tuple[list[Solution], bool]:
    """Call the named ikgeo solver on ``sub_kb``. Returns ``(solutions, is_ls)``."""
    table = {
        "three_parallel": three_parallel.solve,
        "spherical_two_parallel": spherical_two_parallel.solve,
        "spherical_two_intersecting": spherical_two_intersecting.solve,
        "spherical": spherical.solve,
        "two_intersecting": two_intersecting.solve,
        "two_parallel": two_parallel.solve,
        "gen_six_dof": gen_six_dof.solve,
    }
    return table[solver_name](
        sub_kb,
        T_target,
        policy,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
    )


# ---------------------------------------------------------------------------
# Public solve.
# ---------------------------------------------------------------------------


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    *,
    lock_samples: int | Sequence[float] = _DEFAULT_SAMPLES,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> tuple[list[Solution], bool]:
    """Analytic IK for any 7R arm via joint-locking + inner 6R solver.

    :param kb: POE-normalized :class:`KinBody` with 7 revolute joints.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param lock_samples: either an ``int N`` (uniform sweep over
        ``[-pi, pi]`` with N samples), or an explicit sequence of
        lock-joint values. Default 16.
    :param policy: tolerances (forwarded to inner 6R solver).
    :param allow_refinement: opt into Newton polish on each inner-solver
        candidate (#74). Default off.
    :param refinement_max_iters: cap on Newton iterations per candidate.
    :returns: ``(solutions, is_ls)``. Each :class:`Solution.q` is a
        7-vector including the locked joint's value. ``branch_id``
        encodes the lock-sample index (in the order ``samples`` enumerates
        them). Solutions are deduplicated in wrap-to-pi joint-angle
        distance.
    """
    if len(kb.joints) != 7:
        raise ValueError(f"jointlock.seven_r requires a 7-DOF chain; got {len(kb.joints)}")

    lock_idx = choose_lock_joint(kb, policy)

    if isinstance(lock_samples, int):
        samples = np.linspace(-np.pi, np.pi, lock_samples, endpoint=False)
    else:
        samples = np.array(list(lock_samples), dtype=np.float64)

    dedup_tol = policy.subproblem_dedup
    candidates: list[Solution] = []

    for sample_idx, q_lock in enumerate(samples):
        sub_kb = _lock_joint(kb, lock_idx, float(q_lock))
        # Re-check topology per sample: rotating downstream axes by
        # R_lock can switch which tier-0/1 specialization matches.
        _, solver_name = _topology_rank(sub_kb, policy)
        try:
            sub_sols, is_ls = _dispatch(
                solver_name,
                sub_kb,
                T_target,
                policy,
                allow_refinement=allow_refinement,
                refinement_max_iters=refinement_max_iters,
            )
        except ValueError:
            # Topology may fail marginally on some lock values (e.g.
            # near-parallel becoming exactly parallel). Skip.
            continue
        if is_ls or not sub_sols:
            continue
        for inner in sub_sols:
            sub_q = inner.q
            full_q = np.empty(7, dtype=np.float64)
            full_q[:lock_idx] = sub_q[:lock_idx]
            full_q[lock_idx] = float(q_lock)
            full_q[lock_idx + 1 :] = sub_q[lock_idx:]
            candidates.append(
                Solution(
                    q=full_q,
                    fk_residual=inner.fk_residual,
                    refinement_used=inner.refinement_used,
                    refinement_iters=inner.refinement_iters,
                    branch_id=sample_idx,
                    solver_name=_SOLVER_NAME,
                )
            )

    solutions = dedup_by_wrap_close(candidates, dedup_tol)
    return solutions, len(solutions) == 0
