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
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import solve_all_ik

__all__ = ["solve"]

_SOLVER_NAME = "ikgeo.general_6r"


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
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> tuple[list[Solution], bool]:
    """Analytic IK for any 6R chain via Raghavan-Roth + AE-3 leftvar selection.

    :param kb: POE-normalized :class:`KinBody` with 6 revolute joints.
    :param T_target: 4x4 target end-effector pose in the POE base frame.
    :param policy: tolerance policy. ``subproblem_numerical`` is the
        FK-closure threshold; ``subproblem_dedup`` is the per-joint
        wrap-to-pi tolerance for collapsing equivalent solutions.
    :param allow_refinement: opt into Newton-on-spatial-Jacobian polish for
        algebraic candidates that don't meet ``policy.subproblem_numerical``
        on their own. Default off (#74); the algebraic path is exact for
        well-conditioned poses on most arms thanks to AE-3 leftvar choice.
    :param refinement_max_iters: cap on Newton iterations per candidate
        when ``allow_refinement=True``.
    :returns: ``(solutions, is_ls)``. Each :class:`Solution.q` is in the
        POE frame (matches ``FK_POE(q)``, not ``FK_DH(theta)``).
        ``Solution.fk_residual`` is measured against the user's POE chain.
        ``is_ls=True`` iff no candidate closed within
        ``policy.subproblem_numerical``.
    """
    if len(kb.joints) != 6:
        raise ValueError(f"general_6r requires a 6-DOF chain; got {len(kb.joints)} joints")
    for joint in kb.joints:
        if joint.joint_type != "revolute":
            raise ValueError(f"general_6r requires all-revolute joints; got {joint.joint_type}")

    dh = poe_to_dh(kb)
    t_target = np.asarray(T_target, dtype=np.float64)
    t_target_dh = np.linalg.solve(dh.t_pre, t_target) @ np.linalg.inv(dh.t_post)

    fk_atol = policy.subproblem_numerical
    dedup_atol = policy.subproblem_dedup
    inner_solutions, _is_ls = solve_all_ik(
        dh.to_dh_tuple(),
        t_target_dh,
        fk_atol=fk_atol,
        dedup_atol=dedup_atol,
        linearity_joint="auto",
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
        solver_name=_SOLVER_NAME,
    )

    solutions: list[Solution] = []
    for inner in inner_solutions:
        q = inner.q - dh.theta_offset
        # FK_residual measured against the user's POE chain (the inner
        # solver's residual was measured against the bridged DH target;
        # the bridge involves matrix inverses, so a fresh POE-chain
        # measurement is the contract we report to the caller).
        fk_resid_poe = float(np.linalg.norm(_forward_kinematics(kb, q) - t_target))
        if fk_resid_poe > fk_atol:
            continue
        solutions.append(
            Solution(
                q=q,
                fk_residual=fk_resid_poe,
                refinement_used=inner.refinement_used,
                refinement_iters=inner.refinement_iters,
                branch_id=inner.branch_id,
                solver_name=_SOLVER_NAME,
            )
        )

    return solutions, len(solutions) == 0
