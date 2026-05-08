"""Generic spherical-wrist + two-parallel-shoulder 6R analytical IK solver.

Handles any 6R kinematic chain with both of these special structures:

- Three consecutive intersecting joint axes at the wrist (joint indices
  ``(3, 4, 5)``) -- a spherical wrist.
- Two consecutive parallel joint axes at the shoulder/elbow (joint indices
  ``(1, 2)``).

This covers the classical industrial arm family: Puma 560, most Fanuc LR
and CR series, KUKA KR series, ABB IRB (with wrist offset cases falling
into a sibling solver).

**Implementation**: port of the BSD-3 [ik-geo Rust reference][ikgeo]'s
``spherical_two_parallel`` (Elias & Wen, arXiv:2211.05737). The algorithm:

1. Consolidate the POE per-joint offsets between joints 3 and the tool so
   the wrist center is reached by a single translation ``p[3]``. This is
   the shared intersection point of the wrist axes.
2. Strip the POE home-pose rotation from the target so IK-Geo's
   identity-home formulas apply.
3. Use SP4 on the shoulder projection ``axes[1] . (p_16 rotated by -q1)
   = axes[1] . (p[1] + p[2] + p[3])`` to solve for ``theta_0``. Works
   because ``axes[1] = axes[2]`` (parallel), so the projection is invariant
   under ``theta_1`` and ``theta_2``.
4. For each ``theta_0`` branch, use SP3 on the elbow triangle to solve for
   ``theta_2``.
5. Use SP1 to recover ``theta_1`` from the shoulder-plane constraint.
6. For each ``(theta_0, theta_1, theta_2)`` branch, compute ``R_36`` and
   apply SP4 (wrist alignment) for ``theta_4``, then SP1 twice for
   ``theta_3`` and ``theta_5``.

Up to 8 IK solutions per target pose (2 shoulder x 2 elbow x 2 wrist).

**Convention** -- we expect a POE-normalized KinBody (from
:func:`ssik._urdf.load_urdf_kinbody_normalized`):

- ``axes[i]`` is joint ``i``'s axis in the base frame at ``q = 0``.
- ``T_left[i][:3, 3]`` is the translational offset between joints ``i-1``
  and ``i``.
- ``T_right[5][:3, 3]`` is the tool-flange offset from joint 5 at ``q = 0``.
- ``T_right[5][:3, :3]`` carries any home-pose rotation baked into the
  URDF; we strip it from the target rotation before invoking IK-Geo.

For a spherical wrist the POE stores ``T_left[3..5]`` as sequential
offsets that together move from joint 3 to the wrist intersection (the
IK-Geo reference expects a single consolidated offset there). We sum them.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/mod.rs
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import axis_parallel, three_consecutive_intersecting
from ssik.refinement import kinbody_jacobian, verify_candidates
from ssik.subproblems import sp1, sp3, sp4
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["solve"]

_SOLVER_NAME = "ikgeo.spherical_two_parallel"
_LOG = logging.getLogger(__name__)


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    max_solutions: int | None = None,
) -> tuple[list[Solution], bool]:
    """Analytic IK for spherical-wrist + two-parallel-shoulder 6R chains.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints, three
        consecutive intersecting axes at positions ``(3, 4, 5)``, and two
        parallel axes at positions ``(1, 2)``.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerances (forwarded to the subproblems and topology
        predicates).
    :param allow_refinement: opt into Newton polish (#74). Default off.
    :param refinement_max_iters: cap on Newton iterations per candidate.
    :returns: ``(solutions, is_ls)``. Up to 8 :class:`Solution` candidates
        reproducing ``T_target`` under FK to within
        ``policy.subproblem_numerical``.
    """
    if len(kb.joints) != 6:
        raise ValueError(
            f"spherical_two_parallel requires a 6-DOF chain; got {len(kb.joints)} joints"
        )
    triple = three_consecutive_intersecting(kb.joints, policy)
    if triple != (3, 4, 5):
        raise ValueError(
            f"spherical_two_parallel requires the intersecting-axis triple at joints (3, 4, 5); "
            f"got {triple}. Check the chain's topology."
        )
    if not axis_parallel(kb.joints[1].axis, kb.joints[2].axis, policy):
        raise ValueError(
            "spherical_two_parallel requires joints (1, 2) to be parallel; "
            "they are not. Check the chain's topology."
        )

    axes = [j.axis for j in kb.joints]

    our_p = [kb.joints[i].T_left[:3, 3].copy() for i in range(6)]
    tool_p = kb.joints[-1].T_right[:3, 3].copy()

    # IK-Geo convention: a single consolidated shoulder-to-wrist offset at
    # p[3]. Our POE stores it split across T_left[3..5]. Sum them; p[4] and
    # p[5] are unused by this solver (wrist-collapse).
    p = [
        our_p[0],
        our_p[1],
        our_p[2],
        our_p[3] + our_p[4] + our_p[5],
        np.zeros(3),
        np.zeros(3),
        tool_p,
    ]

    # Strip the home-pose rotation from the target.
    r_home = kb.joints[-1].T_right[:3, :3]
    t_target = np.asarray(T_target, dtype=np.float64)
    r_06 = t_target[:3, :3] @ r_home.T
    p_0t = t_target[:3, 3]

    # SP4: project the base-to-wrist vector onto the shared parallel axis
    # axes[1] == axes[2]. That projection is invariant under q2 and q3, so
    # it isolates q1:
    #   axes[1] . (Rot(-axes[0], q1) @ (p_0t - r_06 @ p[6] - p[0])) =
    #       axes[1] . (p[1] + p[2] + p[3])
    t1_solutions, _ = sp4.solve(
        axes[1],
        -axes[0],
        p_0t - r_06 @ p[6] - p[0],
        float(axes[1] @ (p[1] + p[2] + p[3])),
        policy,
    )

    candidates: list[NDArray[np.float64]] = []
    for q1 in t1_solutions:
        # Shoulder-plane residual: what (p[2] + Rot(axes[2], q3) @ p[3])
        # must achieve, minus p[1], expressed in the joint-1 frame.
        shoulder = rotation_matrix(-axes[0], q1) @ (-p_0t + r_06 @ p[6] + p[0]) + p[1]

        # SP3: elbow distance constraint gives q3.
        t3_solutions, _ = sp3.solve(
            axes[2],
            -p[3],
            p[2],
            float(np.linalg.norm(shoulder)),
            policy,
        )

        for q3 in t3_solutions:
            # SP1: given elbow q3, recover q2 from the shoulder plane.
            q2, _ = sp1.solve(
                axes[1],
                -p[2] - rotation_matrix(axes[2], q3) @ p[3],
                shoulder,
                policy,
            )

            r_36 = (
                rotation_matrix(-axes[2], q3)
                @ rotation_matrix(-axes[1], q2)
                @ rotation_matrix(-axes[0], q1)
                @ r_06
            )

            # SP4 for the wrist pitch q5.
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
        fk_fn=lambda q: poe_forward_kinematics(kb, q),
        jacobian_fn=lambda q: kinbody_jacobian(kb, q),
        t_target=t_target,
        fk_atol=policy.subproblem_numerical,
        dedup_atol=policy.subproblem_dedup,
        solver_name=_SOLVER_NAME,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
        max_solutions=max_solutions,
    )
    _LOG.info(
        "%s: %d candidates -> %d solutions (is_ls=%s)",
        _SOLVER_NAME,
        len(candidates),
        len(solutions),
        len(solutions) == 0,
    )
    return solutions, len(solutions) == 0
