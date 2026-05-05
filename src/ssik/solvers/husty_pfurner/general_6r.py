"""Universal 6R analytical IK via the Husty-Pfurner algorithm.

KinBody-input wrapper around the numeric Husty-Pfurner pipeline. Converts
a POE-normalised :class:`KinBody` to standard distal DH form, runs the
HP elimination + back-substitution to recover all up-to-16 IK solutions,
FK-closes each, and returns them as :class:`Solution` objects in the
caller's POE frame.

Pipeline:

1. Convert POE-normalised ``KinBody`` to standard distal DH form via
   :func:`~ssik.kinematics.poe_to_dh.poe_to_dh`. Returns
   ``(alpha, a, d, theta_offset, T_pre, T_post)`` such that
   ``FK_POE(q) = T_pre @ FK_DH(q + theta_offset) @ T_post``.
2. Bridge target into the DH frame:
   ``T_dh = T_pre^{-1} @ T_target @ T_post^{-1}``.
3. Bridge into HP convention (joint 6 has ``a_6 = d_6 = alpha_6 = 0``):
   ``T_HP = T_dh @ inverse(T_z(d_6) T_x(a_6) R_x(alpha_6))``.
4. Convert ``T_HP`` to a Study DQ ``sigma_E`` (8-vec).
5. Build the per-arm
   :class:`~ssik.solvers.husty_pfurner._eliminate.EliminatePrecompute`
   from DH alpha (via ``ls = tan(alpha/2)``), a, d.
6. Run :func:`~ssik.solvers.husty_pfurner._back_substitute.solve_ik`
   to recover all ``(v_1, ..., v_6)`` candidates with FK closure
   already filtered.
7. Convert each ``v_i = tan(theta_i/2)`` back to ``theta_i``, subtract
   ``theta_offset`` to land in the POE frame.
8. Wrap each candidate in a :class:`Solution` with FK-residual
   measured against the user's POE chain.

Targets the EAIK gap that even Raghavan-Roth doesn't fully close: HP's
universal degree-16 polynomial works on every 6R chain (Capco's RRR
case) and -- with future Phase 5c.4 dispatch -- on RRP/RPR/RPP/PRR/PPR
6R/P variants too.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.refinement import lm_refine
from ssik.solvers.husty_pfurner._back_substitute import solve_ik
from ssik.solvers.husty_pfurner._eliminate import precompute_rrr_chain
from ssik.solvers.husty_pfurner._study import dq_from_se3

_LOG = logging.getLogger(__name__)

__all__ = ["solve"]

_SOLVER_NAME = "husty_pfurner.general_6r"


def _se3_from_dh_offset(a: float, alpha: float, d: float) -> NDArray[np.float64]:
    """Build the SE(3) of ``T_z(d) T_x(a) R_x(alpha)`` (the joint-6 offset
    that HP convention puts inside ``sigma_E`` rather than inside
    ``sigma_6``)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [1.0, 0.0, 0.0, a],
            [0.0, ca, -sa, 0.0],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
) -> tuple[list[Solution], bool]:
    """Universal 6R analytical IK via Husty-Pfurner.

    :param kb: POE-normalised :class:`KinBody` with 6 revolute joints.
        (6R/P variants will land in Phase 5c.4 / future iterations.)
    :param T_target: 4x4 target end-effector pose in the POE base frame.
    :param policy: tolerance policy. ``subproblem_numerical`` is the
        FK-closure threshold passed through to
        :func:`~ssik.solvers.husty_pfurner._back_substitute.solve_ik`.
    :param allow_refinement: opt into Newton polish for candidates that
        don't meet ``policy.subproblem_numerical`` on their own. Newton
        is already part of the inner pipeline; this flag is currently
        a no-op (kept for API parity with other solvers; #74 follow-up
        will surface a tighter ``residue_tol`` override).
    :param refinement_max_iters: parity with other solvers' API.
        Currently unused.

    :returns: ``(solutions, is_ls)``. ``is_ls=True`` iff no candidate
        passed FK closure -- typically means the ``T_target`` is
        outside the workspace.
    """
    del allow_refinement  # always-on; the multi-root configs Phase 5g sees
    # routinely benefit (locked-Franka multiplicity-4 is unsolvable without
    # post-back-sub Newton on the POE FK residual).
    if len(kb.joints) != 6:
        raise ValueError(f"husty_pfurner.general_6r requires a 6-joint chain, got {len(kb.joints)}")

    dh = poe_to_dh(kb)

    # Capco eq.5 precondition: a_5 != 0 AND alpha_5 not in {0, pi}.
    # The RRR-variant T(v_1) hyperplane construction relies on this
    # (the 4-flat collapses at a_5 = 0 or sin(alpha_5) = 0). The
    # alternative T(v_2) / T(v_3) parametrisations cover the
    # degenerate cases (Phase 5c.4 / Capco's per-pattern files);
    # until they land we raise an informative error.
    a_5 = float(dh.a[4])
    alpha_5 = float(dh.alpha[4])
    if abs(a_5) < 1e-9 or abs(np.sin(alpha_5)) < 1e-9:
        raise ValueError(
            f"husty_pfurner.general_6r requires a_5 != 0 and alpha_5 "
            f"not in {{0, pi}} (Capco eq.5 precondition). Got "
            f"a_5={a_5}, alpha_5={alpha_5}. Pieper-class arms (Puma, "
            f"UR, JACO 2 -- chains where joint-5 has no x-axis offset) "
            f"don't satisfy this; use the IK-Geo solver instead. "
            f"Phase 5c.4 will add the V_2/V_3 fallbacks."
        )

    t_target = np.asarray(T_target, dtype=np.float64)
    t_target_dh = np.linalg.solve(dh.t_pre, t_target) @ np.linalg.inv(dh.t_post)

    # HP convention vs standard distal DH:
    # - joint 1 in HP: ``R_z(v_1) T_x(a_1) R_x(alpha_1)`` (no T_z(d_1)).
    #   Standard DH: ``R_z(theta_1) T_z(d_1) T_x(a_1) R_x(alpha_1)``.
    #   Absorb ``T_z(d_1)`` as a left prefix into the target.
    # - joint 6 in HP: ``R_z(v_6)`` only. Standard DH includes
    #   ``T_z(d_6) T_x(a_6) R_x(alpha_6)``. Absorb that as a right
    #   suffix (its inverse left-multiplied to the target).
    t_z_neg_d1 = np.eye(4, dtype=np.float64)
    t_z_neg_d1[2, 3] = -float(dh.d[0])
    t_joint6_offset = _se3_from_dh_offset(
        a=float(dh.a[5]), alpha=float(dh.alpha[5]), d=float(dh.d[5])
    )
    t_hp = t_z_neg_d1 @ t_target_dh @ np.linalg.inv(t_joint6_offset)
    sigma_E = dq_from_se3(t_hp)

    # Per-arm precompute. HP uses tan-half-angle for both joint and twist
    # variables; the (a_i, alpha_i, d_i) DH conversion is straightforward
    # for the 5 inner joints.
    ls = np.tan(0.5 * dh.alpha)
    pre = precompute_rrr_chain(
        a_1=float(dh.a[0]),
        l_1=float(ls[0]),
        d_2=float(dh.d[1]),
        a_2=float(dh.a[1]),
        l_2=float(ls[1]),
        d_3=float(dh.d[2]),
        a_3=float(dh.a[2]),
        l_3=float(ls[2]),
        d_4=float(dh.d[3]),
        a_4=float(dh.a[3]),
        l_4=float(ls[3]),
        d_5=float(dh.d[4]),
        a_5=float(dh.a[4]),
        l_5=float(ls[4]),
    )

    fk_atol = policy.subproblem_numerical
    # Pass a loose fk_tol so solve_ik returns ALL back-sub seeds, not only
    # those that algebraically close on the HP residue. The lm_refine pass
    # below polishes each seed against the POE FK and filters non-converging
    # spurious seeds.
    sols_v = solve_ik(
        pre,
        sigma_E,
        a_1=float(dh.a[0]),
        l_1=float(ls[0]),
        d_2=float(dh.d[1]),
        a_2=float(dh.a[1]),
        l_2=float(ls[1]),
        d_3=float(dh.d[2]),
        a_3=float(dh.a[2]),
        l_3=float(ls[2]),
        d_4=float(dh.d[3]),
        a_4=float(dh.a[3]),
        l_4=float(ls[3]),
        d_5=float(dh.d[4]),
        a_5=float(dh.a[4]),
        l_5=float(ls[4]),
        fk_tol=0.5,
    )

    if sols_v.shape[0] == 0:
        _LOG.info("husty_pfurner.general_6r: no IK seeds from elimination")
        return [], True

    # Newton polish: each algebraic seed -> machine-precision physical IK
    # via :func:`ssik.refinement.lm_refine` on the SE(3) log residual.
    # At multi-root configs (locked Franka, etc.) the algebraic seeds can
    # be 1e-1 away from the true IK; lm_refine's quadratic convergence
    # closes that gap in 3-5 iterations. Spurious seeds either fail to
    # converge or land on the same physical solution as a good seed
    # (collapsed by the dedup pass below).
    fk_fn = lambda q: _fk_poe_chain(kb, q)  # noqa: E731
    refined_solutions: list[tuple[NDArray[np.float64], float, int]] = []
    for v in sols_v:
        theta_dh = 2.0 * np.arctan(v)
        q_seed = theta_dh - dh.theta_offset
        result = lm_refine(
            q_seed,
            fk_fn,
            t_target,
            fk_atol=max(fk_atol, 1e-9),
            max_iters=refinement_max_iters,
        )
        if result is None:
            continue
        refined_solutions.append(result)

    if not refined_solutions:
        _LOG.info("husty_pfurner.general_6r: no IK seeds converged via lm_refine")
        return [], True

    # Dedup post-refinement: multiple seeds can converge to the same
    # physical IK. Cluster by joint-wrap-angle distance using the policy
    # tolerance.
    dedup_tol = policy.subproblem_dedup
    deduped: list[tuple[NDArray[np.float64], float, int]] = []
    for q_ref, fk_residual, iters in refined_solutions:
        is_dup = False
        for q_existing, _, _ in deduped:
            diffs = np.array(
                [(((q_ref[i] - q_existing[i] + np.pi) % (2 * np.pi)) - np.pi) for i in range(6)]
            )
            if np.max(np.abs(diffs)) < dedup_tol:
                is_dup = True
                break
        if not is_dup:
            deduped.append((q_ref, fk_residual, iters))

    solutions: list[Solution] = []
    for branch_id, (q_ref, fk_residual, iters) in enumerate(deduped):
        solutions.append(
            Solution(
                q=q_ref,
                fk_residual=fk_residual,
                refinement_used="lm",
                refinement_iters=iters,
                branch_id=branch_id,
                solver_name=_SOLVER_NAME,
            )
        )

    _LOG.info(
        "husty_pfurner.general_6r: returned %d IK solutions (max fk_residual=%.3e)",
        len(solutions),
        max(s.fk_residual for s in solutions),
    )
    return solutions, False


def _fk_poe_chain(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward kinematics of the POE chain: same convention as the FK
    used by :func:`ssik._kinbody.KinBody.fk_chain` (chain product of
    joint twist exponentials in the POE base frame).
    """
    # KinBody POE FK: T = prod_i T_left @ exp(qi * axis_i_hat) @ T_right
    # where axis_i is the screw axis. Use the helper if available; here
    # we inline the standard FK in a way that mirrors test fixtures.
    t_acc = np.eye(4, dtype=np.float64)
    for joint, qi in zip(kb.joints, q, strict=True):
        # Joint contribution: T_left @ axis-angle(qi) @ T_right
        axis = np.asarray(joint.axis, dtype=np.float64)
        c, s = float(np.cos(qi)), float(np.sin(qi))
        kx, ky, kz = float(axis[0]), float(axis[1]), float(axis[2])
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]], dtype=np.float64)
        R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
        joint_T = np.eye(4)
        joint_T[:3, :3] = R
        t_acc = t_acc @ joint.T_left @ joint_T @ joint.T_right
    return t_acc
