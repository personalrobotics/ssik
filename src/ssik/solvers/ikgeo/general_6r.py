"""General 6R analytical IK via Raghavan-Roth + Manocha-Canny.

KinBody-input wrapper around the numeric Raghavan-Roth pipeline in
:mod:`ssik.solvers.ikgeo._raghavan_roth`. Closes the EAIK gap: any 6R chain,
including non-Pieper non-orthogonal arms (Kinova JACO 2 with 60-degree DH
twists, Agilex Piper, Flexiv Rizon 4) where subproblem-composition solvers
do not apply.

Pipeline:

1. Convert POE-normalized ``KinBody`` to standard distal DH form via
   :func:`~ssik.kinematics.poe_to_dh.poe_to_dh`. Returns
   ``(alpha, a, d, theta_offset, T_pre, T_post)`` such that
   ``FK_POE(q) = T_pre @ FK_DH(q + theta_offset) @ T_post``.
2. Bridge target: ``T_dh = T_pre^{-1} @ T_target @ T_post^{-1}``.
3. Run :func:`~ssik.solvers.ikgeo._raghavan_roth.solve_all_ik` (AE-3 leftvar
   selection cached per arm).
4. Convert the returned ``theta`` vectors back to POE-frame ``q`` by
   subtracting ``theta_offset``.
5. FK-validate each candidate against the POE chain (cross-check that
   the conversion + DH solve round-trips).

The DH-frame solver does the heavy lifting (algebraic-first then
Newton-on-spatial-Jacobian polish, hand-rolled to avoid scipy LM
overhead). Per-arm cold-cache cost is one-time sympy preprocessing
(~30-100s for AE-3 leftvar selection); subsequent IKs are warm-cache
single-digit milliseconds.

Naming note: the Tier-2 grid-search solver in
:mod:`ssik.solvers.ikgeo.gen_six_dof` is being kept as a reference / fallback
for now; this solver supersedes it on speed and precision (AE-3 + algebraic
back-substitute typically ~ms, vs. minutes for the 100x100 grid).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import solve_all_ik

__all__ = ["solve"]


def _rot_mat(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
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


def _forward_kinematics(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4, dtype=np.float64)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4, dtype=np.float64)
        rot[:3, :3] = _rot_mat(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[NDArray[np.float64]], bool]:
    """Analytic IK for any 6R chain via Raghavan-Roth + AE-3 leftvar selection.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints.
    :param T_target: 4x4 target end-effector pose in the POE base frame.
    :param policy: tolerance policy. ``subproblem_numerical`` is the
        FK-closure threshold; ``subproblem_dedup`` is the per-joint
        wrap-to-pi tolerance for collapsing equivalent solutions.
    :returns: ``(solutions, is_ls)``. Each solution is a length-6
        joint-angle vector in the POE frame (i.e. matches ``FK_POE(q)``,
        not ``FK_DH(theta)``). ``is_ls=True`` iff no solution closed
        within ``policy.subproblem_numerical``.
    """
    if len(kb.joints) != 6:
        raise ValueError(f"general_6r requires a 6-DOF chain; got {len(kb.joints)} joints")
    for joint in kb.joints:
        if joint.joint_type != "revolute":
            raise ValueError(
                f"general_6r requires all-revolute joints; got {joint.joint_type}"
            )

    dh = poe_to_dh(kb)
    t_target = np.asarray(T_target, dtype=np.float64)
    t_target_dh = np.linalg.solve(dh.t_pre, t_target) @ np.linalg.inv(dh.t_post)

    fk_atol = policy.subproblem_numerical
    dedup_atol = policy.subproblem_dedup
    theta_solutions, _is_ls = solve_all_ik(
        dh.to_dh_tuple(),
        t_target_dh,
        fk_atol=fk_atol,
        dedup_atol=dedup_atol,
        linearity_joint="auto",
    )

    solutions: list[NDArray[np.float64]] = []
    for theta in theta_solutions:
        q = np.asarray(theta, dtype=np.float64) - dh.theta_offset
        # Cross-check via POE FK (the DH solver validated against the
        # bridged DH target, but the bridge involves matrix inverses; verify
        # the round-trip closes in the user's frame too).
        if float(np.linalg.norm(_forward_kinematics(kb, q) - t_target)) > fk_atol:
            continue
        solutions.append(q)

    return solutions, len(solutions) == 0
