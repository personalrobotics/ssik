"""Analytical IK for UR-family 6R arms (UR3 / UR5 / UR10).

**Temporary correctness oracle.** This solver is intentionally narrow: it
encodes UR-specific DH frame conventions (``alpha1 = pi/2, alpha2 = alpha3 = 0,
alpha4 = pi/2, alpha5 = -pi/2, alpha6 = 0``) and will silently produce wrong
answers on any three-parallel 6R arm that does not match that structure
(Fanuc LR Mate, some KUKA variants, etc.). It is shipped as a correctness
oracle for the upcoming **tier-1 generic solver** (univariate polynomial
search over any chain with at least one parallel or intersecting axis pair),
which will subsume this module. Once the generic solver is cross-validated
against this one on UR5, this file is deleted. Do not build anything on top
of it. See umbrella #37.

Algorithm follows Hawkins' 2013 technical report "Analytic Inverse Kinematics
for the Universal Robots UR-5/UR-10 Arms" (public-domain). DH-equivalent
parameters ``(d1, a2, a3, d4, d5, d6)`` are extracted from the KinBody's
joint origins and axes so the same file works for every UR variant; the
UR-ness itself is what's hard-coded.

Up to 8 solutions per pose: 2 base-pan branches x 2 wrist-pitch branches x
2 elbow branches. Branches that leave the reachable workspace are pruned.
When every branch is infeasible the solver returns ``([], True)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.kinematics.predicates import joint_origins, three_consecutive_parallel

__all__ = ["solve"]


def _extract_ur_dh(kb: KinBody) -> tuple[float, float, float, float, float, float]:
    """Extract Hawkins DH parameters ``(d1, a2, a3, d4, d5, d6)`` from a
    POE-normalized UR-class KinBody.

    ``d`` parameters project onto the corresponding joint axis direction;
    ``a`` parameters project onto the ``x`` axis of the shoulder frame, which
    is ``+x`` in base after the ``alpha1 = pi/2`` rotation characteristic of
    UR-family arms.
    """
    origs = joint_origins(kb.joints)
    axes = [j.axis / float(np.linalg.norm(j.axis)) for j in kb.joints]
    t_right_last = kb.joints[-1].T_right

    d1 = float(np.dot(origs[1] - origs[0], axes[0]))
    arm_x = np.array([1.0, 0.0, 0.0])
    a2 = float(np.dot(origs[2] - origs[1], arm_x))
    a3 = float(np.dot(origs[3] - origs[2], arm_x))
    d4 = float(np.dot(origs[4] - origs[3], axes[3]))
    d5 = float(np.dot(origs[5] - origs[4], axes[4]))
    d6 = float(np.dot(t_right_last[:3, 3], axes[5]))

    return d1, a2, a3, d4, d5, d6


def _dh_matrix(theta: float, d: float, a: float, alpha: float) -> NDArray[np.float64]:
    """Classical DH transform ``Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha)``."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _fk_from_dh(
    thetas: NDArray[np.float64], dh: tuple[float, float, float, float, float, float]
) -> NDArray[np.float64]:
    """Forward kinematics via UR-class DH for sanity checks."""
    d1, a2, a3, d4, d5, d6 = dh
    dh_rows = [
        (d1, 0.0, np.pi / 2),
        (0.0, a2, 0.0),
        (0.0, a3, 0.0),
        (d4, 0.0, np.pi / 2),
        (d5, 0.0, -np.pi / 2),
        (d6, 0.0, 0.0),
    ]
    T = np.eye(4)
    for theta, (d, a, alpha) in zip(thetas, dh_rows, strict=True):
        T = T @ _dh_matrix(theta, d, a, alpha)
    return T


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], bool]:
    """Analytic IK for UR-class three-parallel 6R arms.

    :param kb: POE-normalized :class:`KinBody` (from
        :func:`ssik._urdf.load_urdf_kinbody_normalized`). Must have three
        consecutive parallel joints at indices ``(1, 2, 3)``.
    :param T_target: 4x4 target end-effector pose as an ndarray.
    :returns: ``(solutions, is_ls)`` where ``solutions`` is a list of length-6
        numpy arrays (joint values in radians). Up to 8 solutions are
        returned; an empty list indicates total infeasibility (rare; usually
        LS approximations are returned instead with ``is_ls=True``).
    """
    if len(kb.joints) != 6:
        raise ValueError(
            f"three_parallel 6R solver requires a 6-DOF chain; got {len(kb.joints)} joints."
        )
    triple = three_consecutive_parallel(kb.joints)
    if triple != (1, 2, 3):
        raise ValueError(
            f"three_parallel 6R solver requires the parallel-axis triple at "
            f"joints (1, 2, 3); got {triple}. Check the chain's topology."
        )

    d1, a2, a3, d4, d5, d6 = _extract_ur_dh(kb)
    if abs(d6) < 1e-12:
        raise ValueError(
            "UR-class solver requires a non-zero tool offset (d6). Load the "
            "URDF through the tool-flange link (e.g., 'ee_link' rather than "
            "'wrist_3_link') so T_right captures the d6 translation."
        )

    T = np.asarray(T_target, dtype=np.float64)
    solutions: list[NDArray[np.float64]] = []

    # --- Step 1: theta1 (shoulder pan) --------------------------------------
    # Wrist-2 origin in base frame: translate back by d6 along ee z-axis.
    p05 = T @ np.array([0.0, 0.0, -d6, 1.0])
    psi = float(np.arctan2(p05[1], p05[0]))
    r_xy = float(np.hypot(p05[0], p05[1]))
    if r_xy < abs(d4) - 1e-9:
        # Wrist-2 center inside the shoulder-pan radius: unreachable.
        return [], True
    phi = float(np.arccos(np.clip(d4 / r_xy, -1.0, 1.0)))
    theta1_options = [psi + phi + np.pi / 2, psi - phi + np.pi / 2]

    for theta1 in theta1_options:
        s1, c1 = float(np.sin(theta1)), float(np.cos(theta1))

        # --- Step 2: theta5 (wrist pitch) -----------------------------------
        # cos(theta5) = (px*s1 - py*c1 - d4) / d6
        rhs = T[0, 3] * s1 - T[1, 3] * c1 - d4
        c5 = rhs / d6
        if abs(c5) > 1.0 + 1e-9:
            # This theta1 branch cannot accommodate the wrist orientation.
            continue
        c5 = float(np.clip(c5, -1.0, 1.0))
        base_t5 = float(np.arccos(c5))
        theta5_options = [+base_t5, -base_t5]

        for theta5 in theta5_options:
            s5 = float(np.sin(theta5))

            # --- Step 3: theta6 (wrist yaw) ---------------------------------
            if abs(s5) < 1e-9:
                # Gimbal: theta6 indeterminate from position alone; pick 0.
                theta6 = 0.0
            else:
                n1 = -T[0, 1] * s1 + T[1, 1] * c1  # y-comp of "o" axis in frame 1
                n2 = T[0, 0] * s1 - T[1, 0] * c1  # -x-comp of "n" axis in frame 1
                theta6 = float(np.arctan2(n1 / s5, n2 / s5))

            # --- Step 4: theta2, theta3 (elbow) via planar 3R ---------------
            # Compute T14 = T01^{-1} T T56^{-1} T45^{-1}
            T01 = _dh_matrix(theta1, d1, 0.0, np.pi / 2)
            T45 = _dh_matrix(theta5, d5, 0.0, -np.pi / 2)
            T56 = _dh_matrix(theta6, d6, 0.0, 0.0)
            T14 = np.linalg.inv(T01) @ T @ np.linalg.inv(T56) @ np.linalg.inv(T45)

            # The middle three parallel joints (theta2, theta3, theta4) form a
            # planar mechanism that rotates around frame 1's z-axis. Under
            # alpha2 = alpha3 = 0, frame 3's z-axis coincides with frame 1's z,
            # so peeling d4 off along frame-1-z yields the frame-3 origin in
            # frame-1 coordinates.
            p14 = T14[:3, 3]
            p13 = np.array([p14[0], p14[1], p14[2] - d4])

            # Planar 2R inverse: frame-3 origin in frame 1 has the form
            #     (p13_x, p13_y, 0) = (a2 * c2 + a3 * c(2+3), a2 * s2 + a3 * s(2+3), 0)
            # since each A_i rotates around frame-1-z and contributes a pure
            # in-plane translation along the prior frame's x-axis.
            p13_x, p13_y = p13[0], p13[1]

            # Two-link planar: c3 = ((p13_x^2 + p13_y^2 - a2^2 - a3^2)) / (2*a2*a3)
            denom_sq = p13_x * p13_x + p13_y * p13_y
            c3 = (denom_sq - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            if abs(c3) > 1.0 + 1e-9:
                # This (theta1, theta5) branch doesn't reach the elbow target.
                continue
            c3 = float(np.clip(c3, -1.0, 1.0))
            base_t3 = float(np.arccos(c3))
            theta3_options = [+base_t3, -base_t3]

            for theta3 in theta3_options:
                s3 = float(np.sin(theta3))

                # theta2: solve (p13_x, p13_y) = Rot_z(θ2) * (a2 + a3*c3, a3*s3)
                # p13_x = (a2 + a3*c3)*c2 - a3*s3*s2
                # p13_y = (a2 + a3*c3)*s2 + a3*s3*c2
                k_c = a2 + a3 * c3
                k_s = a3 * s3
                theta2 = float(
                    np.arctan2(
                        k_c * p13_y - k_s * p13_x,
                        k_c * p13_x + k_s * p13_y,
                    )
                )

                # theta4 = (theta2 + theta3 + theta4)_total - theta2 - theta3
                # The total sum is encoded in T14's rotation: R14 = Rot_z(θ2+θ3+θ4)
                sum_234 = float(np.arctan2(T14[1, 0], T14[0, 0]))
                theta4 = sum_234 - theta2 - theta3

                sol = np.array([theta1, theta2, theta3, theta4, theta5, theta6])
                solutions.append(sol)

    # If every branch was infeasible we return an empty list with is_ls=True;
    # any successful branch counts as exact (the returned solutions are all
    # feasible even if some branches were pruned).
    return solutions, len(solutions) == 0
