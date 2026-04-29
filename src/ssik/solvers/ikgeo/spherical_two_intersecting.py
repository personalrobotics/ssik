"""Generic spherical-wrist + intersecting-shoulder 6R analytical IK solver.

Handles any 6R kinematic chain with both of these special structures:

- Three consecutive intersecting joint axes at the wrist (joint indices
  ``(3, 4, 5)``) -- a spherical wrist.
- Joints 0 and 1 share a common origin point -- the waist and shoulder
  pivots coincide (``p[1] = 0`` in our POE convention).

This is the second classical industrial-arm family: compact arms where the
shoulder-elbow assembly pivots at the base column (Puma 560, ABB IRB
smaller variants, many collaborative arms including uFactory lite6 and
xArm6 subfamilies).

Puma 560 actually satisfies both this solver's preconditions *and*
:mod:`ssik.solvers.ikgeo.spherical_two_parallel`'s. Both solvers return
the same 8-solution set; the dispatcher (Phase C) will pick by
specialization ranking.

**Implementation**: port of the BSD-3 [ik-geo Rust reference][ikgeo]'s
``spherical_two_intersecting`` (Elias & Wen, arXiv:2211.05737). The
algorithm:

1. Consolidate the POE per-joint offsets between joints 3 and the tool so
   the wrist center is reached by a single translation ``p[3]``.
2. Strip the POE home-pose rotation from the target so IK-Geo's
   identity-home formulas apply.
3. Since ``p[1] = 0``, the wrist-center equation reduces to
   ``p_16 = Rot(axes[0], q0) Rot(axes[1], q1) (p[2] + Rot(axes[2], q2) p[3])``.
   SP3 on the elbow-distance constraint isolates ``theta_2``.
4. For each ``theta_2`` branch, SP2 jointly solves for ``(theta_0, theta_1)``
   from the rotated shoulder equation.
5. For each ``(theta_0, theta_1, theta_2)`` branch, compute ``R_36`` and
   apply SP4 (wrist alignment) for ``theta_4``, then SP1 twice for
   ``theta_3`` and ``theta_5``.

Up to 8 IK solutions per target pose (2 elbow x 2 shoulder x 2 wrist).

**Convention** -- see
:mod:`ssik.solvers.ikgeo.spherical_two_parallel` module docstring for the
full POE + ``R_home`` + wrist-consolidation convention; this solver
uses the same.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/mod.rs
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.predicates import three_consecutive_intersecting
from ssik.refinement import kinbody_jacobian, verify_candidates
from ssik.subproblems import sp1, sp2, sp3, sp4
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["solve"]

_SOLVER_NAME = "ikgeo.spherical_two_intersecting"
_LOG = logging.getLogger(__name__)


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> tuple[list[Solution], bool]:
    """Analytic IK for spherical-wrist + intersecting-shoulder 6R chains.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints,
        three consecutive intersecting axes at positions ``(3, 4, 5)``,
        and ``p[1] = 0`` (joints 0 and 1 share a common origin).
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerances (forwarded to subproblems and topology
        predicates).
    :param allow_refinement: opt into Newton polish (#74). Default off.
    :param refinement_max_iters: cap on Newton iterations per candidate.
    :returns: ``(solutions, is_ls)``. Up to 8 :class:`Solution` candidates
        reproducing ``T_target`` under FK to within
        ``policy.subproblem_numerical``.
    """
    if len(kb.joints) != 6:
        raise ValueError(
            f"spherical_two_intersecting requires a 6-DOF chain; got {len(kb.joints)} joints"
        )
    triple = three_consecutive_intersecting(kb.joints, policy)
    if triple != (3, 4, 5):
        raise ValueError(
            f"spherical_two_intersecting requires the intersecting-axis triple at joints "
            f"(3, 4, 5); got {triple}. Check the chain's topology."
        )
    p1_translation = kb.joints[1].T_left[:3, 3]
    if float(np.linalg.norm(p1_translation)) > policy.axis_intersect:
        raise ValueError(
            f"spherical_two_intersecting requires joints (0, 1) to share an origin "
            f"(p[1] = 0); got ||p[1]|| = {float(np.linalg.norm(p1_translation)):.3e}."
        )

    axes = [j.axis for j in kb.joints]

    our_p = [kb.joints[i].T_left[:3, 3].copy() for i in range(6)]
    tool_p = kb.joints[-1].T_right[:3, 3].copy()

    p = [
        our_p[0],
        our_p[1],
        our_p[2],
        our_p[3] + our_p[4] + our_p[5],
        np.zeros(3),
        np.zeros(3),
        tool_p,
    ]

    r_home = kb.joints[-1].T_right[:3, :3]
    t_target = np.asarray(T_target, dtype=np.float64)
    r_06 = t_target[:3, :3] @ r_home.T
    p_0t = t_target[:3, 3]

    p_16 = p_0t - r_06 @ p[6] - p[0]

    # SP3: elbow distance constraint isolates theta_2. We need the rotated
    # elbow vector ``-p[3]`` placed on the shoulder arc so that its
    # distance from ``p[2]`` equals ||p_16|| (since p[1] = 0).
    t3_solutions, _ = sp3.solve(axes[2], p[3], -p[2], float(np.linalg.norm(p_16)), policy)

    candidates: list[NDArray[np.float64]] = []
    for q3 in t3_solutions:
        # SP2: jointly recover (theta_0, theta_1) from the shoulder
        # equation ``Rot(-axes[0], q1) @ p_16 = Rot(axes[1], q2) @
        # (p[2] + Rot(axes[2], q3) @ p[3])``.
        t12_solutions, _ = sp2.solve(
            -axes[0],
            axes[1],
            p_16,
            p[2] + rotation_matrix(axes[2], q3) @ p[3],
            policy,
        )

        for q1, q2 in t12_solutions:
            r_36 = (
                rotation_matrix(-axes[2], q3)
                @ rotation_matrix(-axes[1], q2)
                @ rotation_matrix(-axes[0], q1)
                @ r_06
            )

            t5_solutions, _ = sp4.solve(
                axes[3],
                axes[4],
                axes[5],
                float(axes[3] @ r_36 @ axes[5]),
                policy,
            )

            for q5 in t5_solutions:
                q4, _ = sp1.solve(
                    axes[3],
                    rotation_matrix(axes[4], q5) @ axes[5],
                    r_36 @ axes[5],
                    policy,
                )
                q6, _ = sp1.solve(
                    -axes[5],
                    rotation_matrix(-axes[4], q5) @ axes[3],
                    r_36.T @ axes[3],
                    policy,
                )
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
    _LOG.info(
        "%s: %d candidates -> %d solutions (is_ls=%s)",
        _SOLVER_NAME,
        len(candidates),
        len(solutions),
        len(solutions) == 0,
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
