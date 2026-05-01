"""Generic spherical-wrist 6R analytical IK solver.

Handles any 6R kinematic chain whose only special structure is a spherical
wrist -- three consecutive intersecting joint axes at positions ``(3, 4, 5)``
-- with *no* additional parallel-shoulder or intersecting-shoulder
specialization.

This is the fallback member of the spherical-wrist family. Its
more-specialized siblings handle the common cases:

- :mod:`ssik.solvers.ikgeo.spherical_two_parallel` -- when joints ``(1, 2)``
  are parallel (Puma, Fanuc, KUKA KR, uFactory lite6/xArm6).
- :mod:`ssik.solvers.ikgeo.spherical_two_intersecting` -- when joints
  ``(0, 1)`` share a common origin (``p[1] = 0``) (Puma, IRB120-class
  compact arms).

This solver fires only when neither specialization applies. Commercial 6R
arms rarely sit in that gap -- essentially every industrial spherical-wrist
arm matches one of the two specializations. We ship it because the
dispatcher (Phase C) needs the common-ancestor fallback and because
IK-Geo's reference library includes it.

**Implementation**: port of the BSD-3 [ik-geo Rust reference][ikgeo]'s
``spherical`` (Elias & Wen, arXiv:2211.05737). Algorithm:

1. Consolidate the POE per-joint offsets between joints 3 and the tool
   so the wrist center is reached by a single translation ``p[3]``.
2. Strip the POE home-pose rotation from the target.
3. SP5 jointly solves for ``(theta_0, theta_1, theta_2)`` from the wrist-
   center position equation. Requires ``axes[1] || axes[2]`` to be false
   (otherwise the arm actually belongs to ``spherical_two_parallel``).
4. For each shoulder branch, compute ``R_36`` and apply SP4 (wrist
   alignment) for ``theta_4``, then SP1 twice for ``theta_3`` and
   ``theta_5``.

Up to 8 IK solutions per target pose (4 shoulder x 2 wrist).

**Robustness** -- SP5's quartic has cluster-root pathology near specific
geometries (issue #55). That was fixed in sp5.py via Gauss-Newton
refinement + scale-aware imaginary-root filter. This solver additionally
dedupes at the final q-vector level using full-FK residual as the
tie-break (platform-stable across LAPACK backends), mirroring the pattern
used in :mod:`ssik.solvers.ikgeo.three_parallel`.

**Convention** -- see :mod:`ssik.solvers.ikgeo.spherical_two_parallel`
docstring for the POE + ``R_home`` + wrist-consolidation convention.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/mod.rs
"""

from __future__ import annotations

import logging

import cython
import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import three_consecutive_intersecting
from ssik.refinement import kinbody_jacobian, verify_candidates
from ssik.subproblems import sp1, sp4, sp5
from ssik.subproblems._rotation import rotation_matrix

__all__ = ["solve"]

_SOLVER_NAME = "ikgeo.spherical"
_LOG = logging.getLogger(__name__)


@cython.locals(
    q1=cython.double,
    q2=cython.double,
    q3=cython.double,
    q4=cython.double,
    q5=cython.double,
    q6=cython.double,
)
def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> tuple[list[Solution], bool]:
    """Analytic IK for generic spherical-wrist 6R chains.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints and
        three consecutive intersecting axes at positions ``(3, 4, 5)``.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerances (forwarded to the subproblems).
    :param allow_refinement: opt into Newton polish (#74). Default off;
        closed-form spherical-wrist solves don't normally need it.
    :param refinement_max_iters: cap on Newton iterations per candidate.
    :returns: ``(solutions, is_ls)``. Up to 8 :class:`Solution` candidates
        reproducing ``T_target`` to within ``policy.subproblem_numerical``.
        ``is_ls=True`` iff no candidate survived (also happens when SP5's
        shoulder reduction is degenerate -- the arm actually belongs to
        ``spherical_two_parallel``).
    """
    if len(kb.joints) != 6:
        raise ValueError(f"spherical requires a 6-DOF chain; got {len(kb.joints)} joints")
    triple = three_consecutive_intersecting(kb.joints, policy)
    if triple != (3, 4, 5):
        raise ValueError(
            f"spherical requires the intersecting-axis triple at joints (3, 4, 5); "
            f"got {triple}. Check the chain's topology."
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

    p_16 = p_0t - p[0] - r_06 @ p[6]

    t123_solutions, _ = sp5.solve(
        -p[1],
        p_16,
        p[2],
        p[3],
        -axes[0],
        axes[1],
        axes[2],
        policy,
    )

    candidates: list[NDArray[np.float64]] = []
    for q1, q2, q3 in t123_solutions:
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

    # Post-verify and dedup at the q-vector level. SP5 has pre-sorted its
    # outputs by pre-GN residual (cluster-root clean-vs-drifted tiebreak);
    # verify_candidates preserves first-pass order then keeps the lower-
    # fk_residual representative on collision. See #56 for the analogous
    # three_parallel rationale.
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
    )
    _LOG.info(
        "%s: %d candidates -> %d solutions (is_ls=%s)",
        _SOLVER_NAME,
        len(candidates),
        len(solutions),
        len(solutions) == 0,
    )
    return solutions, len(solutions) == 0
