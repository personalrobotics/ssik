"""Tier-1 univariate-search 6R solver: two intersecting wrist axes.

Handles any 6R kinematic chain where joints ``(4, 5)`` share a common
origin (``p[5] = 0`` in our POE convention) -- i.e., the final two wrist
axes intersect without needing the middle wrist axis to intersect with
them. This is weaker than a spherical wrist (which requires all three
wrist axes to intersect).

Algorithm: port of the BSD-3 [ik-geo Rust reference][ikgeo]'s
``two_intersecting`` (Elias & Wen, arXiv:2211.05737). The 6D IK problem
reduces to a 1D numerical search over ``theta_3`` (the wrist roll):

1. Consolidate POE offsets so ``p_16`` is the joint-6 origin minus the
   base-to-joint-0 and the tool-in-EE offsets.
2. For each candidate ``q4 in [-pi, pi]``, compute the "effective elbow
   offset" ``p_35_3 = p[3] + Rot(axes[3], q4) @ p[4]``. This depends on
   ``q4`` -- so we run SP5 on the position equation at each sampled
   ``q4`` to get up to 4 shoulder triples ``(q0, q1, q2)``.
3. For each SP5 branch, compute the remaining rotation error:
   ``axes[4] . (R_04^T R_06 axes[5]) - axes[4] . axes[5]``. This is zero
   iff ``q4`` is consistent with the wrist pitch constraint -- a 1D
   equation per branch.
4. :func:`ssik.solvers.ikgeo._univariate.search_1d` finds the zeros of
   this error function in each branch index. Each zero is a valid
   ``(q4, branch)``.
5. At each ``(q4, branch)``, solve ``(q0, q1, q2, q4)`` via SP5
   (recomputed at the refined ``q4``), then SP1 twice for ``q5, q6``.

Up to 8 IK solutions. Precision floor ~1e-5 rad on the refined ``q4``
(false-position convergence tolerance); downstream angles inherit.

**Topology** -- this solver assumes ``p[5] = 0`` (joints 4 and 5 share
an origin). If that's not satisfied the SP5 reduction is wrong. We
check at entry. Arms satisfying this: rare; typical use is custom
geometries. The dispatcher (Phase C) will reach for this when the
spherical-wrist siblings don't match.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/mod.rs
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.refinement import kinbody_jacobian, verify_candidates
from ssik.solvers.ikgeo._univariate import search_1d
from ssik.subproblems import sp1, sp5
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["solve"]

_SEARCH_SAMPLES = 200
_SOLVER_NAME = "ikgeo.two_intersecting"


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> tuple[list[Solution], bool]:
    """Analytic + univariate-search IK for 6R chains with ``p[5] = 0``.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints and
        joints ``(4, 5)`` sharing an origin (``|p[5]| < axis_intersect``).
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerances (forwarded to subproblems).
    :param allow_refinement: opt into Newton polish (#74). Default off.
    :param refinement_max_iters: cap on Newton iterations per candidate.
    :returns: ``(solutions, is_ls)``. Up to 8 :class:`Solution` candidates;
        ``is_ls=True`` iff none survived FK verification.
    """
    if len(kb.joints) != 6:
        raise ValueError(f"two_intersecting requires a 6-DOF chain; got {len(kb.joints)} joints")
    p5_translation = kb.joints[5].T_left[:3, 3]
    if float(np.linalg.norm(p5_translation)) > policy.axis_intersect:
        raise ValueError(
            f"two_intersecting requires joints (4, 5) to share an origin "
            f"(p[5] = 0); got ||p[5]|| = {float(np.linalg.norm(p5_translation)):.3e}."
        )

    axes = [j.axis for j in kb.joints]
    p = [kb.joints[i].T_left[:3, 3].copy() for i in range(6)]
    p.append(kb.joints[-1].T_right[:3, 3].copy())

    r_home = kb.joints[-1].T_right[:3, :3]
    t_target = np.asarray(T_target, dtype=np.float64)
    r_06 = t_target[:3, :3] @ r_home.T
    p_0t = t_target[:3, 3]

    p_16 = p_0t - p[0] - r_06 @ p[6]

    def _shoulder_triples(q4: float) -> list[tuple[float, float, float]]:
        p_35_3 = p[3] + rotation_matrix(axes[3], q4) @ p[4]
        sols, _ = sp5.solve(-p[1], p_16, p[2], p_35_3, -axes[0], axes[1], axes[2], policy)
        return sols

    def _alignment_error(q4: float) -> NDArray[np.float64]:
        """4-vector of ``axes[4] . R_04^T R_06 axes[5] - axes[4] . axes[5]``
        across the (up to 4) SP5 shoulder branches. Non-existent branches
        fill with ``inf`` (search_1d skips those)."""
        errors = np.full(4, np.inf, dtype=np.float64)
        target = float(axes[4] @ axes[5])
        triples = _shoulder_triples(q4)
        for i, (q1, q2, q3) in enumerate(triples):
            r_04 = (
                rotation_matrix(axes[0], q1)
                @ rotation_matrix(axes[1], q2)
                @ rotation_matrix(axes[2], q3)
                @ rotation_matrix(axes[3], q4)
            )
            errors[i] = float(axes[4] @ r_04.T @ r_06 @ axes[5]) - target
        return errors

    q4_branches = search_1d(_alignment_error, -np.pi, np.pi, _SEARCH_SAMPLES)

    candidates: list[NDArray[np.float64]] = []
    for q4, branch_idx in q4_branches:
        triples = _shoulder_triples(q4)
        if branch_idx >= len(triples):
            # Shoulder branching shifted during refinement; skip this zero.
            continue
        q1, q2, q3 = triples[branch_idx]

        r_04 = (
            rotation_matrix(axes[0], q1)
            @ rotation_matrix(axes[1], q2)
            @ rotation_matrix(axes[2], q3)
            @ rotation_matrix(axes[3], q4)
        )

        q5, _ = sp1.solve(axes[4], axes[5], r_04.T @ r_06 @ axes[5], policy)
        q6, _ = sp1.solve(-axes[5], axes[4], r_06.T @ r_04 @ axes[4], policy)

        candidates.append(np.array([q1, q2, q3, q4, q5, q6]))

    solutions = verify_candidates(
        candidates,
        fk_fn=lambda q: _forward_kinematics(kb, q),
        jacobian_fn=lambda q: kinbody_jacobian(kb, q),
        t_target=t_target,
        fk_atol=policy.subproblem_numerical,
        dedup_atol=policy.subproblem_dedup,
        solver_name=_SOLVER_NAME,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
    )
    return solutions, len(solutions) == 0


def _forward_kinematics(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """POE forward kinematics for the composed IK post-verification."""
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T
