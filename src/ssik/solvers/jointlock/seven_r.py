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

import logging
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import Joint, KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics._scalar3 import _se3_inv
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.kinematics.predicates import (
    axis_parallel,
    three_consecutive_intersecting,
    three_consecutive_parallel,
)
from ssik.kinematics.reverse import map_reversed_q, reverse_kinematic_chain
from ssik.refinement import dedup_by_wrap_close
from ssik.solvers.husty_pfurner import general_6r as hp_general_6r
from ssik.solvers.ikgeo import (
    general_6r as rr_general_6r,
)
from ssik.solvers.ikgeo import (
    spherical,
    spherical_two_intersecting,
    spherical_two_parallel,
    three_parallel,
    two_intersecting,
    two_parallel,
)
from ssik.solvers.ikgeo._raghavan_roth import primed_linearity_for_dh
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["choose_lock_joint", "solve"]

_DEFAULT_SAMPLES = 16
_SOLVER_NAME = "jointlock.seven_r"
_LOG = logging.getLogger(__name__)


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


def _topology_rank_direct(sub_kb: KinBody, policy: TolerancePolicy) -> tuple[int, str]:
    """Score a 6R KinBody by its best matching solver family at canonical
    sub-chain positions: parallel triple at ``(1, 2, 3)`` and spherical
    wrist at ``(3, 4, 5)``. Sub-chains whose structure sits elsewhere are
    handled by :func:`_topology_rank` via chain reversal.
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
    # Tier-2 fallback (universal 6R analytical via Husty-Pfurner).
    # HP handles arms where no Pieper / parallel-axis predicate matches
    # the locked sub-chain (typical for KUKA iiwa / Rizon / non-Pieper 7R
    # arms). HP's perturbation path (#176) covers the measure-zero
    # singularities common in locked-7R DHs (Tv2 case [a_1=0, a_2=0]).
    # Faster than Raghavan-Roth for these symmetric DHs (RR's m_quad
    # conditioning blows up; empirically RR takes 25-60 s on iiwa14
    # locked sub-chains while HP takes 100-220 ms).
    return (3, "husty_pfurner.general_6r")


def _topology_rank(sub_kb: KinBody, policy: TolerancePolicy) -> tuple[int, str]:
    """Score a 6R KinBody by its best matching solver family, considering
    both the original chain ordering and the chain reversal.

    Lower rank = better. Returns ``(rank, solver_name)`` where
    ``solver_name`` is either a plain name (e.g. ``"three_parallel"``)
    for the original chain or a ``"reversed:..."`` prefixed name when
    reversal lands the kinematic structure at a canonical position.

    The reversal pre-pass closes the EAIK ``REVERSED`` decomposition
    family: arms whose post-lock 6R sub-chain has its spherical wrist or
    parallel triple at the BASE (not the END) of the chain. Franka Panda
    after locking joint 4 is the canonical example -- joints 0,1,2 of
    the sub-chain all pass through the shoulder origin, so reversing
    the chain places the spherical wrist at sub-chain positions
    ``(3, 4, 5)`` where :mod:`ssik.solvers.ikgeo.spherical_two_parallel`
    matches it directly.
    """
    orig_rank, orig_name = _topology_rank_direct(sub_kb, policy)
    if orig_rank == 0:
        # Already a tier-0 closed-form match on the original chain --
        # reversal can't beat that.
        return (orig_rank, orig_name)

    # Try the reversed chain. If it ranks better, use the reversed
    # dispatch path.
    sub_kb_rev = reverse_kinematic_chain(sub_kb)
    rev_rank, rev_name = _topology_rank_direct(sub_kb_rev, policy)
    if rev_rank < orig_rank:
        return (rev_rank, "reversed:" + rev_name)
    return (orig_rank, orig_name)


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
    max_solutions: int | None = None,
) -> tuple[list[Solution], bool]:
    """Call the named ikgeo solver on ``sub_kb``. Returns ``(solutions, is_ls)``.

    ``solver_name`` may be prefixed with ``"reversed:"`` to indicate that
    the inner solver should be called on the chain-reversed sub-chain
    with target ``T_target^{-1}``; the returned q-vectors are then mapped
    back to the original-chain ordering. This handles the EAIK
    ``REVERSED`` decomposition family (e.g. Franka post-lock-4).

    ``max_solutions`` (#198) is plumbed through to the inner solver so a
    capped-IK request stops branch enumeration once the cap is reached
    inside the sub-chain, avoiding wasted Newton polish on extra seeds.
    """
    if solver_name.startswith("reversed:"):
        inner_name = solver_name[len("reversed:") :]
        sub_kb_rev = reverse_kinematic_chain(sub_kb)
        T_target_rev = _se3_inv(np.asarray(T_target, dtype=np.float64))
        sub_sols_rev, is_ls = _dispatch(
            inner_name,
            sub_kb_rev,
            T_target_rev,
            policy,
            allow_refinement=allow_refinement,
            refinement_max_iters=refinement_max_iters,
            max_solutions=max_solutions,
        )
        # Map reversed-chain q's back to original-chain ordering.
        sub_sols_orig = [replace(sol, q=map_reversed_q(sol.q)) for sol in sub_sols_rev]
        return sub_sols_orig, is_ls

    # Cached-RR fast path (#210): for any non-strict-tier-0 inner solver,
    # try Raghavan-Roth first IF its symbolic derivation has been primed
    # at artifact-import time. Primed cache => ~1 ms warm solve; no prime
    # => the original solver (HP / two_parallel / etc.) runs as before.
    # This keeps the URDF-loaded path (tests, interactive use) on the
    # original solvers (no cold-cache cost) while letting production
    # artifacts opt in to the 15-30x speedup.
    if solver_name in _RR_ELIGIBLE_INNER_SOLVERS:
        rr_result = _try_cached_rr(
            sub_kb,
            T_target,
            policy,
            allow_refinement=allow_refinement,
            refinement_max_iters=refinement_max_iters,
            max_solutions=max_solutions,
        )
        if rr_result is not None:
            return rr_result

    table = {
        "three_parallel": three_parallel.solve,
        "spherical_two_parallel": spherical_two_parallel.solve,
        "spherical_two_intersecting": spherical_two_intersecting.solve,
        "spherical": spherical.solve,
        "two_intersecting": two_intersecting.solve,
        "two_parallel": two_parallel.solve,
        "husty_pfurner.general_6r": hp_general_6r.solve,
    }
    # ``Cython.Shadow``'s ``@cython.locals`` decorator widens the wrapped
    # function's signature for mypy, so the dispatch-table lookup returns
    # ``Any``. Reasserting the annotated return type here keeps the strict
    # ``no-any-return`` check happy without affecting runtime behaviour.
    result: tuple[list[Solution], bool] = table[solver_name](
        sub_kb,
        T_target,
        policy,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
        max_solutions=max_solutions,
    )
    return result


# Inner solvers eligible for cached-RR fast path (#210). The set excludes:
#   - Strict tier-0 specialisations (three_parallel, spherical_two_*): they
#     already run 1-2 ms per call and beat RR's per-call cost.
#   - rank-1 ``spherical`` (~7.5 ms): the prime cost (~14 s of cold RR
#     derivation per sub-chain DH) is not worth the ~6.5 ms per-call savings
#     for arms where this is the dominant inner solver (e.g. Franka's
#     ``reversed:spherical`` samples). Including it would add ~210 s to
#     module-import for ~100 ms warm-IK savings -- bad UX trade-off.
# Only the truly expensive inner solvers (tier-1 search-based, tier-2 HP)
# are eligible. Rizon 4 and Kassow KR810 -- whose dominant inner solvers
# are HP and two_parallel -- get the full 12-25x post-warmup speedup.
_RR_ELIGIBLE_INNER_SOLVERS: frozenset[str] = frozenset(
    {
        "two_intersecting",  # rank 2: tier-1 search, ~1.2 s
        "two_parallel",  # rank 2: tier-1 search, ~261 ms
        "husty_pfurner.general_6r",  # rank 3: ~13-35 ms
    }
)


def _try_cached_rr(
    sub_kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy,
    *,
    allow_refinement: bool,
    refinement_max_iters: int,
    max_solutions: int | None = None,
) -> tuple[list[Solution], bool] | None:
    """Attempt cached-RR solve on the sub-chain. Returns ``(sols, is_ls)``
    if RR's derivation cache is primed for this DH AND it produces a
    valid IK; ``None`` otherwise (caller falls back to original solver).

    Bulletproof gate: only accepts RR's output if (a) ``is_ls=False``,
    (b) at least one solution exists, and (c) max FK closure passes
    ``policy.subproblem_numerical``.
    """
    # poe_to_dh is cached on the kb object, so this is fast after first call.
    dh = poe_to_dh(sub_kb)
    alpha = tuple(float(x) for x in dh.alpha)
    a = tuple(float(x) for x in dh.a)
    d = tuple(float(x) for x in dh.d)

    # Look up the baked (linearity, apply_so3) for this DH. None means
    # the artifact hasn't primed RR for this sub-chain -- fall back to
    # the original solver to avoid the 30-150s cold-cache + leftvar
    # probe cost.
    primed = primed_linearity_for_dh(alpha, a, d)
    if primed is None:
        return None

    linearity, apply_so3 = primed
    try:
        # Pass the baked linearity to bypass the runtime AE-3 leftvar
        # probe inside ``solve_all_ik`` -- the leftvar selection at
        # codegen time is recorded in the prime; running it again at
        # runtime would cost ~42 s (3 derivations) per unique sub-chain.
        rr_sols, rr_is_ls = rr_general_6r.solve(
            sub_kb,
            T_target,
            policy,
            allow_refinement=allow_refinement,
            refinement_max_iters=refinement_max_iters,
            linearity_joint=linearity,
            apply_so3=apply_so3,
            max_solutions=max_solutions,
        )
    except Exception as exc:
        _LOG.debug("jointlock cached-RR raised %s; falling back", type(exc).__name__)
        return None

    if rr_is_ls or not rr_sols:
        return None
    if max(s.fk_residual for s in rr_sols) >= policy.subproblem_numerical:
        return None
    return rr_sols, rr_is_ls


# ---------------------------------------------------------------------------
# Public solve.
# ---------------------------------------------------------------------------


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    *,
    lock_samples: int | Sequence[float] | NDArray[np.float64] = _DEFAULT_SAMPLES,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    lock_idx: int | None = None,
    max_solutions: int | None = None,
    q_seed: NDArray[np.float64] | None = None,
    dispatch_cache: Sequence[str] | None = None,
) -> tuple[list[Solution], bool]:
    """Analytic IK for any 7R arm via joint-locking + inner 6R solver.

    :param kb: POE-normalized :class:`KinBody` with 7 revolute joints.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param lock_samples: either an ``int N`` (uniform sweep over the
        locked joint's reachable range with ``N`` samples), or an explicit
        sequence of lock-joint values. Default 16. The sweep range is
        ``kb.joints[lock_idx].limits`` if set, otherwise ``[-pi, pi]``
        (which is the full revolution and the right default for
        continuous / unspecified-limit joints). Explicit sequences
        bypass the range clamp -- the caller's values win.
    :param policy: tolerances (forwarded to inner 6R solver).
    :param allow_refinement: opt into Newton polish on each inner-solver
        candidate (#74). Default off.
    :param refinement_max_iters: cap on Newton iterations per candidate.
    :param lock_idx: pre-computed lock joint index. When provided, skips
        the per-call :func:`choose_lock_joint` topology-rank scan over
        all 7 lock candidates. Codegen artifacts pass the baked value
        here; runtime callers usually leave ``None``.
    :param max_solutions: optional early-exit cap. When set, stop the
        lock-sweep as soon as this many *deduplicated* solutions have
        been collected. ``None`` (default) sweeps every sample. Use
        ``max_solutions=1`` for the "give me any IK" case (typical
        speedup ~24x vs the full sweep) and ``max_solutions=N`` when you
        need a fixed-size handful (e.g. for a downstream postprocess
        ranker). Solutions returned under early-exit are a *subset* of
        what the full sweep would return, never different solutions.
    :param q_seed: optional length-7 seed configuration. When provided,
        the lock-joint samples are visited in order of wrap-to-pi
        distance to ``q_seed[lock_idx]``, nearest-first. Combined with
        ``max_solutions=1`` this turns the trajectory-tracking case
        ("track this current config") into a 1-2 sample lookup instead
        of a full 16/24 sweep.
    :param dispatch_cache: optional pre-computed inner-solver dispatch
        names, one per element of ``lock_samples`` and aligned by
        sample index. When provided, the per-sample
        :func:`_topology_rank` call (~70 us with chain reversal) is
        skipped -- saving ~1 ms per IK on a 16-sample sweep. Must
        match ``len(lock_samples)``; mismatch is a ``ValueError``.
        Codegen artifacts emitted for non-SRS 7R arms bake this from
        the codegen-time topology probe (#142 item 4); manual callers
        leave ``None``.
    :returns: ``(solutions, is_ls)``. Each :class:`Solution.q` is a
        7-vector including the locked joint's value. Solutions are
        deduplicated in wrap-to-pi joint-angle distance. Solutions are
        returned in the order their lock-samples were evaluated, so
        under ``q_seed=...`` ordering the first solution is the one
        with lock-sample closest to the seed.

    Common idioms::

        # Default: exhaustive search (~64 solutions, ~41 ms on Franka 7R).
        solutions, _ = seven_r.solve(kb, T_target)

        # "Just give me one IK" -- ~17x faster on Franka.
        solutions, _ = seven_r.solve(kb, T_target, max_solutions=1)

        # Trajectory tracking: find the IK closest to current config
        # (~37x faster on Franka).
        solutions, _ = seven_r.solve(
            kb, T_target, q_seed=q_current, max_solutions=1,
        )
    """
    if len(kb.joints) != 7:
        raise ValueError(f"jointlock.seven_r requires a 7-DOF chain; got {len(kb.joints)}")
    if max_solutions is not None and max_solutions < 1:
        raise ValueError(f"max_solutions must be >= 1 or None; got {max_solutions}")
    if q_seed is not None:
        q_seed_arr = np.asarray(q_seed, dtype=np.float64)
        if q_seed_arr.shape != (7,):
            raise ValueError(f"q_seed must have shape (7,); got {q_seed_arr.shape}")
    else:
        q_seed_arr = None

    if lock_idx is None:
        lock_idx = choose_lock_joint(kb, policy)

    if isinstance(lock_samples, int):
        # Clamp the uniform sweep to the locked joint's reachable range
        # when known (limits=None for continuous joints means free
        # rotation -> sweep [-pi, pi] = one full revolution, the unique
        # sample space). For limited joints this avoids wasting samples
        # in unreachable territory.
        joint_limits = kb.joints[lock_idx].limits
        if joint_limits is None:
            lo, hi = -np.pi, np.pi
        else:
            lo, hi = joint_limits
        samples = np.linspace(lo, hi, lock_samples, endpoint=False)
    else:
        samples = np.array(list(lock_samples), dtype=np.float64)

    cache_arr: list[str] | None
    if dispatch_cache is not None:
        cache_arr = list(dispatch_cache)
        if len(cache_arr) != len(samples):
            raise ValueError(
                f"dispatch_cache length {len(cache_arr)} must match "
                f"lock_samples length {len(samples)}"
            )
    else:
        cache_arr = None

    if q_seed_arr is not None:
        # Reorder samples by wrap-to-pi distance to seed[lock_idx], nearest
        # first. Combined with max_solutions=1, this is the trajectory-
        # tracking fast path: usually one or two dispatches before the
        # match emerges. The unstable sort + numerical tiebreak below keeps
        # ordering deterministic even when two samples land at equal
        # distance.
        seed_lock = float(q_seed_arr[lock_idx])
        diffs = np.abs((samples - seed_lock + np.pi) % (2 * np.pi) - np.pi)
        order = np.lexsort((samples, diffs))
        samples = samples[order]
        # Apply the same permutation to the dispatch cache so the
        # cache[i] still corresponds to samples[i].
        if cache_arr is not None:
            cache_arr = [cache_arr[i] for i in order]

    dedup_tol = policy.subproblem_dedup
    candidates: list[Solution] = []
    samples_evaluated = 0

    for sample_idx, q_lock in enumerate(samples):
        samples_evaluated = sample_idx + 1
        sub_kb = _lock_joint(kb, lock_idx, float(q_lock))
        if cache_arr is not None:
            # Codegen-time topology probe (#142 item 4); skip the per-IK
            # ``_topology_rank`` call (~70 us with chain-reversal).
            solver_name = cache_arr[sample_idx]
        else:
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
                max_solutions=max_solutions,
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
                )
            )
        # Incremental dedup so we know how many *unique* solutions we
        # actually have. Cheaper than waiting until the end and then
        # discovering we already had enough; the dedup primitive's
        # early-exit per-pair check (#141) keeps this affordable.
        if (
            max_solutions is not None
            and len(dedup_by_wrap_close(candidates, dedup_tol)) >= max_solutions
        ):
            break

    solutions = dedup_by_wrap_close(candidates, dedup_tol)
    if max_solutions is not None and len(solutions) > max_solutions:
        # Trim to exactly max_solutions. The incremental check above
        # exits the *outer* loop on the first sample that pushes the
        # deduped count past the cap, so the final dedup may yield
        # slightly more than the cap (the last sample contributed
        # multiple new unique solutions). Trimming preserves the
        # nearest-first order under q_seed.
        solutions = solutions[:max_solutions]
    _LOG.info(
        "%s: lock_idx=%d, %d/%d samples -> %d candidates -> %d unique solutions (is_ls=%s)",
        _SOLVER_NAME,
        lock_idx,
        samples_evaluated,
        len(samples),
        len(candidates),
        len(solutions),
        len(solutions) == 0,
    )
    return solutions, len(solutions) == 0
