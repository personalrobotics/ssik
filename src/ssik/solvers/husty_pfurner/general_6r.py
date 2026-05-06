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

Coverage matrix (RRR pattern)
-----------------------------

Capco eq. (5)/(6) give DH-precondition rules for which left-/right-chain
parametrization is well-defined. Verified against Capco's reference
giac code at Zenodo 3157441 (``which_case.py:74-80``):

+------------+----------------------------------------------------+----------+
| Variant    | Applies when                                       | Status   |
+============+====================================================+==========+
| T(v_1)     | ``a_2 != 0 ∧ l_2 != 0``                            | done     |
| T(v_3)     | ``(a_2 = 0 OR l_2 = 0) ∧ a_1 != 0 ∧ l_1 != 0``      | #180     |
| T(v_2)     | ``(a_2 = 0 OR l_2 = 0) ∧ (a_1 = 0 OR l_1 = 0)``      | **#176** |
|            | -- 4 sub-cases: [a_1=0,a_2=0], [a_1=0,l_2=0],      |          |
|            | [l_1=0,a_2=0], [l_1=0,l_2=0]                       |          |
| T(v_6)     | ``a_4 != 0 ∧ l_4 != 0`` (right mirror)             | done     |
| T(v_4)     | ``(a_4 = 0 OR l_4 = 0) ∧ a_5 != 0 ∧ l_5 != 0``      | #177     |
| T(v_5)     | right-chain double-degenerate case                 | #177     |
+------------+----------------------------------------------------+----------+

The RRR-pattern eq. (5) simplified-form degeneracy is
``a_2 = 0 OR l_2 = 0`` (NOT ``|l_2| = ±1`` -- that's the RRP rule;
earlier ssik docstrings had this confused). DH audit shows every
locked-7R configuration on Franka, KUKA iiwa LBR, and xArm7 hits the
**RRR Tv2 sub-case [a_1=0, a_2=0]** specifically (a_1 = 0 universal on
industrial 7R; locked sub-chains inherit a_2 = 0 at most lock
positions). Implementing just that one sub-case unblocks 12/14 lock
configs per arm.

Until #176 lands, ``precompute_rrr_chain`` logs a one-time WARNING per
degenerate DH and proceeds with ``T(v_1)`` -- the partial IK set it
returns is still useful for many poses, but some branches are silently
missed. Callers needing complete coverage on these arms should fall
back to ``ssik.solvers.ikgeo.gen_six_dof`` (Raghavan-Roth, slower) or
wait for #176.

References:
- Capco, Loquias, Manongsong, Nemenzo (2019), arXiv 1906.07813
- Capco-Manongsong reference giac code, Zenodo 3157441 (MIT-licensed,
  the authoritative dispatch logic)
- Manongsong (2019), PhD thesis, UP Diliman -- T(v_2) derivation
- Noferini & Townsend (2015), arXiv 1507.00272 -- Sylvester pencil
  instability and rotation cure (#178)
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.refinement import kinbody_jacobian, verify_candidates
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
    fk_atol: float = 1e-12,
) -> tuple[list[Solution], bool]:
    """Universal 6R analytical IK via Husty-Pfurner.

    :param kb: POE-normalised :class:`KinBody` with 6 revolute joints.
        (6R/P variants will land in Phase 5c.4 / future iterations.)
    :param T_target: 4x4 target end-effector pose in the POE base frame.
    :param policy: tolerance policy. Used for structural predicates and
        deduplication; the FK-closure threshold is ``fk_atol`` (separate
        knob -- see #183 for the planned unified tolerance policy that
        derives both from a single primary).
    :param allow_refinement: opt into Newton polish for seeds that
        don't already meet ``fk_atol`` after algebraic elimination +
        back-sub. Defaults to ``False`` to match the other solvers'
        contract; Phase 5g multi-root configs (locked Franka, etc.)
        require ``True`` to recover machine-precision IK because pencil
        eigenvalues at multiplicity-k roots sit at ``O(eps^{1/k})`` from
        truth, AND singular-DH perturbed seeds (#176, e.g. locked-7R)
        sit at ``O(epsilon)`` and need LM polish to converge.
    :param refinement_max_iters: cap on Newton iterations per seed
        when ``allow_refinement=True``.
    :param fk_atol: target FK closure (Frobenius norm of ``FK(q) -
        T_target``). Default ``1e-12`` is tight enough to satisfy the
        bulletproof-validation contract (FK ≤ 1e-10) for HP, which can
        algebraically reach machine precision on non-degenerate chains
        and reaches ``epsilon`` precision on perturbed seeds via LM
        polish. Loosen for speed (looser ``fk_atol`` -> fewer LM
        iterations); tighten for precision (down to machine ~1e-15).
        See #183 for the unified policy that exposes this knob through
        :class:`TolerancePolicy.fk_closure`.

    :returns: ``(solutions, is_ls)``. ``is_ls=True`` iff no candidate
        passed FK closure -- typically means the ``T_target`` is
        outside the workspace.
    """
    if len(kb.joints) != 6:
        raise ValueError(f"husty_pfurner.general_6r requires a 6-joint chain, got {len(kb.joints)}")

    dh = poe_to_dh(kb)

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

    # Singular-DH perturbation (#176): when a_2 = 0 AND no Tv3 fallback
    # (Tv3 requires a_1 != 0 AND l_1 != 0), the Tv2 case applies and
    # V_L lies entirely in the Study quadric -- the algebraic variety
    # becomes bigger than the IK set, and the elimination produces
    # spurious roots. V_L ⊂ S is measure-zero in DH parameter space:
    # perturb a_2 (or l_2, whichever is zero) by epsilon = 1e-3 and the
    # singularity breaks. The standard Tv1 + Tv4 dispatch then applies.
    # Downstream verify_candidates polishes each O(epsilon) seed via 6-D
    # Newton on the *unperturbed* POE FK -> machine precision in 4-8
    # iters. Same mechanism for the right chain (a_4 = 0 -> Tv5 case;
    # perturb a_5 to enable Tv4 dispatch).
    _SINGULAR_TOL = 1e-9
    _EPS_PERTURB = 1e-3
    a_1, l_1 = float(dh.a[0]), float(ls[0])
    a_2, l_2 = float(dh.a[1]), float(ls[1])
    a_4, l_4 = float(dh.a[3]), float(ls[3])
    a_5, l_5 = float(dh.a[4]), float(ls[4])

    left_tv1_ok = abs(a_2) > _SINGULAR_TOL and abs(l_2) > _SINGULAR_TOL
    left_tv3_ok = abs(a_1) > _SINGULAR_TOL and abs(l_1) > _SINGULAR_TOL
    if not left_tv1_ok and not left_tv3_ok:
        if abs(a_2) < _SINGULAR_TOL:
            a_2 = _EPS_PERTURB
        elif abs(l_2) < _SINGULAR_TOL:
            l_2 = _EPS_PERTURB

    right_tv6_ok = abs(a_4) > _SINGULAR_TOL and abs(l_4) > _SINGULAR_TOL
    right_tv4_ok = (
        (abs(a_4) < _SINGULAR_TOL or abs(l_4) < _SINGULAR_TOL)
        and abs(a_5) > _SINGULAR_TOL
        and abs(l_5) > _SINGULAR_TOL
    )
    if not right_tv6_ok and not right_tv4_ok:
        if abs(a_5) < _SINGULAR_TOL:
            a_5 = _EPS_PERTURB
        elif abs(l_5) < _SINGULAR_TOL:
            l_5 = _EPS_PERTURB

    pre = precompute_rrr_chain(
        a_1=a_1,
        l_1=l_1,
        d_2=float(dh.d[1]),
        a_2=a_2,
        l_2=l_2,
        d_3=float(dh.d[2]),
        a_3=float(dh.a[2]),
        l_3=float(ls[2]),
        d_4=float(dh.d[3]),
        a_4=a_4,
        l_4=l_4,
        d_5=float(dh.d[4]),
        a_5=a_5,
        l_5=l_5,
    )

    # Pass loose tolerances so solve_ik returns ALL back-sub seeds, not
    # only those that algebraically close to full HP precision:
    #
    # - ``fk_tol=0.5``: skips the projective Study-DQ closure check
    #   inside solve_ik (downstream verify_candidates does the FK
    #   closure in POE space; the in-flight check is wasted overhead).
    # - ``accept_residue_tol=1e-3``: don't reject pass-1 (u, w)
    #   candidates whose 2-D Newton bottomed out above 1e-12. At
    #   multi-root degenerate poses (locked-Franka multiplicity-4
    #   cluster, IK at coincident axes, ...) Newton in 2-D plateaus at
    #   ~1e-6 to 1e-8, which used to silently reject real-but-imperfect
    #   IK candidates. The downstream lm_refine in 6-D joint space
    #   recovers them; pass-1 should refine, not reject.
    sols_v = solve_ik(
        pre,
        sigma_E,
        a_1=a_1,
        l_1=l_1,
        d_2=float(dh.d[1]),
        a_2=a_2,
        l_2=l_2,
        d_3=float(dh.d[2]),
        a_3=float(dh.a[2]),
        l_3=float(ls[2]),
        d_4=float(dh.d[3]),
        a_4=a_4,
        l_4=l_4,
        d_5=float(dh.d[4]),
        a_5=a_5,
        l_5=l_5,
        fk_tol=0.5,
        accept_residue_tol=1e-3,
    )

    if sols_v.shape[0] == 0:
        _LOG.info("husty_pfurner.general_6r: no IK seeds from elimination")
        return [], True

    # Tier-dispatch: convert each algebraic seed to q-space then hand
    # the whole batch to verify_candidates. Tier 0 = FK-check accepts
    # clean seeds without Newton; tier 1 = lm_refine polishes seeds
    # that fail the FK check (e.g. locked-Franka multiplicity-4
    # cluster). The analytical kinbody_jacobian + Cython
    # poe_forward_kinematics + lm_refine divergence-abort make the
    # per-spurious-seed cost ~150 us instead of 1 ms.
    fk_fn = lambda q: poe_forward_kinematics(kb, q)  # noqa: E731
    jac_fn = lambda q: kinbody_jacobian(kb, q)  # noqa: E731

    q_seeds = [2.0 * np.arctan(v) - dh.theta_offset for v in sols_v]

    solutions = verify_candidates(
        q_seeds,
        fk_fn=fk_fn,
        t_target=t_target,
        fk_atol=fk_atol,
        solver_name=_SOLVER_NAME,
        dedup_atol=policy.subproblem_dedup,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
        jacobian_fn=jac_fn,
    )

    if not solutions:
        _LOG.info("husty_pfurner.general_6r: no IK seeds passed FK closure or Newton polish")
        return [], True

    _LOG.info(
        "husty_pfurner.general_6r: returned %d IK solutions (max fk_residual=%.3e)",
        len(solutions),
        max(s.fk_residual for s in solutions),
    )
    return solutions, False
