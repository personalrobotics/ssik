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

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.predicates import three_consecutive_parallel
from ssik.subproblems import sp1, sp3, sp6
from ssik.subproblems._rotation import rotate

__all__ = ["solve"]


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return float(((angle + np.pi) % (2 * np.pi)) - np.pi)


def _rot_mat(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """3x3 rotation matrix around ``axis`` by ``angle`` (Rodrigues)."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[NDArray[np.float64]], bool]:
    """Analytic IK for three-parallel 6R chains.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints and
        three consecutive parallel axes at positions ``(1, 2, 3)``.
    :param T_target: 4x4 target end-effector pose in the base frame.
    :param policy: tolerances (forwarded to the subproblems).
    :returns: ``(solutions, is_ls)``. ``solutions`` is a list of up to 8
        length-6 joint vectors, each reproducing ``T_target`` under FK to
        within the subproblem-residual tolerance. ``is_ls=True`` iff at
        least one returned solution was produced via an LS branch inside
        a subproblem (e.g. wrist singularity); consumers should check the
        flag before treating results as exact.
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
        r_01 = _rot_mat(axes[0], q1)
        r_45 = _rot_mat(axes[4], q5)

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

        r_14 = _rot_mat(axes[1], theta14)
        d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]
        d_elbow = float(np.linalg.norm(d_inner))

        theta3_solutions, _ = sp3.solve(axes[1], -p[3], p[2], d_elbow, policy)

        for q3 in theta3_solutions:
            p2_rotated = p[2] + rotate(axes[1], q3, p[3])
            q2, _ = sp1.solve(axes[1], p2_rotated, d_inner, policy)
            q4 = _wrap_to_pi(theta14 - q2 - q3)
            candidates.append(np.array([q1, q2, q3, q4, q5, q6]))

    # Post-verify each candidate against the target pose. Only candidates
    # that actually recover T_target within subproblem_numerical survive.
    # Also dedup at the q-vector level (wrap-to-pi per joint, cleanest
    # FK-residual wins) so near-singular poses that split a physical
    # branch into numerically-distinct SP6 candidates collapse back to
    # the single physical solution. See issue #56.
    num_tol = policy.subproblem_numerical
    dedup_tol = policy.subproblem_dedup
    verified: list[tuple[NDArray[np.float64], float]] = []
    for q in candidates:
        t = _forward_kinematics(kb, q)
        fk_err = float(np.linalg.norm(t - t_target))
        if fk_err < num_tol:
            verified.append((q, fk_err))

    # Sort ascending by FK residual so when we merge clusters we keep the
    # cleanest representative deterministically (platform-independent).
    verified.sort(key=lambda pair: pair[1])
    solutions: list[NDArray[np.float64]] = []
    for q, _ in verified:
        if not any(_q_close(q, existing, dedup_tol) for existing in solutions):
            solutions.append(q)

    return solutions, len(solutions) == 0


def _q_close(a: NDArray[np.float64], b: NDArray[np.float64], tol: float) -> bool:
    """Element-wise wrap-to-pi closeness for joint-angle vectors."""
    for ai, bi in zip(a, b, strict=True):
        diff = float(((float(ai) - float(bi) + np.pi) % (2 * np.pi)) - np.pi)
        if abs(diff) > tol:
            return False
    return True


def _forward_kinematics(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """POE forward kinematics for the composed IK post-verification."""
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = _rot_mat(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T
