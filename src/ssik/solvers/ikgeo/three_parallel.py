"""Generic three-parallel 6R analytical IK solver.

Handles any 6R kinematic chain with three consecutive parallel joint axes
(the parallel-trio family) at joint indices ``(1, 2, 3)``. This covers the
entire UR family (UR3 / UR5 / UR10) plus any other arm with the same
three-parallel structure.

**Implementation**: port of the BSD-3 [ik-geo Rust reference][ikgeo]
(Elias & Wen, arXiv:2211.05737). The algorithm:

1. Compute the joint-0 origin and tool offset from the KinBody.
2. Apply SP6 to jointly solve for ``(theta_0, theta_4)`` using two
   scalar constraints derived from the parallel-trio axis and the
   target pose.
3. For each ``(theta_0, theta_4)`` branch, apply SP1 twice to recover
   ``(theta_0 + theta_1 + theta_2 + theta_3)`` (the parallel-trio
   total rotation) and ``theta_5``.
4. Apply SP3 to solve for ``theta_2`` via the elbow distance constraint.
5. Apply SP1 to recover ``theta_1``, then compute ``theta_3`` from the
   total-rotation constraint.

Up to 8 IK solutions per target pose (2 shoulder-pan x 2 wrist-pitch x
2 elbow branches).

**Convention** -- we expect a POE-normalized KinBody (from
:func:`ssik._urdf.load_urdf_kinbody_normalized`):

- ``axes[i]`` is joint ``i``'s axis in the base frame at ``q = 0``.
- ``T_left[i][:3, 3]`` is the translational offset between joints ``i-1``
  and ``i`` (``T_left[0]`` is the base-to-joint-0 offset).
- ``T_right[5][:3, 3]`` is the tool-flange offset from joint 5 at ``q = 0``.
- ``T_right[5][:3, :3]`` carries any home-pose rotation baked into the
  URDF. For IK we work with the combined ``T_target`` directly.

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
from ssik.kinematics.predicates import three_consecutive_parallel
from ssik.refinement import kinbody_jacobian, verify_candidates
from ssik.subproblems import sp1, sp3, sp6
from ssik.subproblems._rotation import rotate, rotation_matrix

__all__ = ["solve"]

_SOLVER_NAME = "ikgeo.three_parallel"

# FK-closure gate for accepting a composed candidate. Tighter than the generic
# ``subproblem_numerical`` (1e-5): at a near-singular pose an SP3/SP4 argument
# clipped to the reachability boundary yields a *spurious* branch that FK-closes
# to only ~1e-6 -- a least-squares near-miss with no exact IK nearby (#362).
# Genuine closed-form solutions close to <=~1e-9 (verified over thousands of
# random poses across the UR/Z1 roster), so 1e-7 -- the arms' declared precision
# (``fk_ceiling_fuzz``) -- cleanly drops the near-miss while keeping every real
# (incl. near-singular) solution.
_FK_VERIFY_ATOL = 1e-7
_LOG = logging.getLogger(__name__)


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)


def trio_reference_signs(axes: list[NDArray[np.float64]]) -> tuple[int, int]:
    """Sign of parallel-trio joints 2 and 3 relative to the reference ``axes[1]``.

    The elbow algebra collapses the parallel trio (joints 1, 2, 3) onto the
    single direction ``axes[1]`` and solves each trio angle *about axes[1]* (via
    the signed total ``theta14 = q1+q2+q3+q4``). A URDF may author a trio joint's
    axis anti-parallel to ``axes[1]`` -- still "parallel" to the dispatch
    predicate, since ``+/-a`` are geometrically parallel -- and its solved angle
    then comes out negated from the physical joint (gauge ``R(-a,q)=R(a,-q)``).
    The physical convention is recovered by negating those joints on the output
    candidate.

    Returns ``(s2, s3)``, each ``+1`` (aligned) or ``-1`` (anti-parallel).

    Single source of truth for the trio sign convention: imported by BOTH the
    live solver (below) and the codegen composer
    (:mod:`ssik.codegen._compose.three_parallel`), so the two representations
    cannot drift. Guarded by ``tests/test_artifact_live_parity.py`` and
    ``tests/test_axis_sign_robustness.py``.
    """
    return (
        1 if float(np.dot(axes[2], axes[1])) >= 0.0 else -1,
        1 if float(np.dot(axes[3], axes[1])) >= 0.0 else -1,
    )


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    max_solutions: int | None = None,
) -> tuple[list[Solution], bool]:
    """Analytic IK for three-parallel 6R chains.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints and
        three consecutive parallel axes at positions ``(1, 2, 3)``.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerances (forwarded to the subproblems).
    :param allow_refinement: opt into Newton polish (#74). Default off.
    :param refinement_max_iters: cap on Newton iterations per candidate.
    :returns: ``(solutions, is_ls)``. Up to 8 :class:`Solution` candidates
        reproducing ``T_target`` under FK to within
        ``policy.subproblem_numerical``. ``is_ls=True`` iff no candidate
        survived (post-verify is the source of truth for feasibility,
        not the inner subproblem ``is_ls`` flags).
    """
    if len(kb.joints) != 6:
        raise ValueError(f"three_parallel requires a 6-DOF chain; got {len(kb.joints)} joints")
    triple = three_consecutive_parallel(kb.joints, policy)
    if triple != (1, 2, 3):
        raise ValueError(
            f"three_parallel requires the parallel-axis triple at joints (1, 2, 3); "
            f"got {triple}. Check the chain's topology."
        )

    # Extract axes (all 6) and position offsets (7 entries: p[0] = base->j0,
    # p[1..5] = inter-joint offsets, p[6] = j5->tool). All in base frame at q=0.
    axes = [j.axis for j in kb.joints]
    p = [kb.joints[i].T_left[:3, 3].copy() for i in range(6)]
    p.append(kb.joints[-1].T_right[:3, 3].copy())

    # Parallel-trio sign normalization: negate any trio joint authored
    # anti-parallel to ``axes[1]`` back to its physical convention on the output
    # candidate (see :func:`trio_reference_signs`). Without this an anti-parallel
    # j2/j3 (e.g. Standard Bots core/spark) yields FK-failing candidates.
    s2, s3 = trio_reference_signs(axes)
    trio_flip = np.array([1.0, 1.0, float(s2), float(s3), 1.0, 1.0], dtype=np.float64)

    # Our POE's T_right[5] encodes a home-pose rotation after joint 5, so
    # FK(q) = [R_joints @ R_home, p; 0, 1]. IK-Geo's formulas assume the final
    # frame has identity home rotation (pure translation), so we strip
    # R_home from the target rotation before invoking them. Position is
    # unchanged because R_home acts after the full translation in our POE.
    r_home = kb.joints[-1].T_right[:3, :3]
    t_target = np.asarray(T_target, dtype=np.float64)
    r_06 = t_target[:3, :3] @ r_home.T
    p_0t = t_target[:3, 3]

    # Position of joint-6 origin (end of joint-5 rotation) in base at q=0.
    # IK-Geo: p_16 = p_0t - p[0] - r_06 * p[6]
    p_16 = p_0t - p[0] - r_06 @ p[6]

    # SP6 to solve for (theta_0, theta_4) jointly. The setup uses the common
    # parallel-trio axis h = axes[1] (= axes[2] = axes[3]) as the fixed
    # direction for all four h-vectors; the two k-axes are axes[0] and
    # axes[4] (the non-parallel ones). Follows the IK-Geo reference exactly.
    h_sp = [axes[1], axes[1], axes[1], axes[1]]
    k_sp = [-axes[0], axes[4], -axes[0], axes[4]]
    p_sp = [p_16, -p[5], r_06 @ axes[5], -axes[5]]
    d1 = float(axes[1] @ (p[2] + p[3] + p[4] + p[1]))
    d2 = 0.0

    theta15_solutions, _ = sp6.solve(h_sp, k_sp, p_sp, d1, d2, policy)

    candidates: list[NDArray[np.float64]] = []
    # Intermediate SP1 / SP3 calls may flag is_ls either on sub-microradian
    # numerical noise (SP1 on rotated unit axes) or on a single branch's
    # local infeasibility (SP3 when one elbow configuration can't reach).
    # Neither flag reflects the IK problem's overall feasibility. We
    # post-verify each composed candidate against the target pose and
    # derive is_ls purely from the post-verify outcome.

    for q1, q5 in theta15_solutions:
        r_01 = rotation_matrix(axes[0], q1)
        r_45 = rotation_matrix(axes[4], q5)

        theta14, _ = sp1.solve(
            axes[1],
            r_45 @ axes[5],
            r_01.T @ r_06 @ axes[5],
            policy,
        )
        q6, _ = sp1.solve(
            -axes[5],
            r_45.T @ axes[1],
            r_06.T @ r_01 @ axes[1],
            policy,
        )

        r_14 = rotation_matrix(axes[1], theta14)
        d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]
        d_elbow = float(np.linalg.norm(d_inner))

        theta3_solutions, _ = sp3.solve(axes[1], -p[3], p[2], d_elbow, policy)

        for q3 in theta3_solutions:
            p2_rotated = p[2] + rotate(axes[1], q3, p[3])
            q2, _ = sp1.solve(axes[1], p2_rotated, d_inner, policy)
            q4 = _wrap_to_pi(theta14 - q2 - q3)
            # Negate any anti-parallel trio joint back to its physical convention
            # (see trio_flip above) before FK-verify on the original chain.
            candidates.append(np.array([q1, q2, q3, q4, q5, q6]) * trio_flip)

    # Post-verify and dedup. SP6 has pre-sorted candidates by pre-GN
    # residual (cleanest Bezout-cluster representative first); the
    # verify_candidates helper preserves that insertion order then
    # tie-breaks collisions by lower fk_residual. See #56 for why this
    # specific ordering matters under cluster-root pathology.
    solutions = verify_candidates(
        candidates,
        fk_fn=lambda q: poe_forward_kinematics(kb, q),
        jacobian_fn=lambda q: kinbody_jacobian(kb, q),
        t_target=t_target,
        fk_atol=_FK_VERIFY_ATOL,
        dedup_atol=policy.subproblem_dedup,
        solver_name=_SOLVER_NAME,
        # The tight 1e-7 gate drops the spurious near-singular near-miss (#362,
        # UR ~7e-6) directly. Recovering a *genuine* near-singular solution
        # (#288, z1 q4=pi/2 ~1e-6 -> machine precision) needs Newton polish, so
        # the caller opts in via ``allow_refinement`` (the standalone-arm
        # artifacts force it on -- ``SolverSpec.force_refine``). We honour the
        # caller here so inner-solver users like jointlock keep their own
        # refinement policy (and their machine-precision-or-drop contract).
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
