"""Generated IK module for Puma 560.

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Solver: ``ikgeo.spherical_two_parallel`` (tier 0)
Expected median IK time: ~1.2 ms on commodity
single-thread hardware. FLOP budget: 1,316 per solve.

Usage:

    import puma560_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions, is_ls = puma560_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``(list[Solution], is_ls)``. ``is_ls=True``
signals that no solution closed within the solver's FK tolerance,
and the returned list is the best-LS approximation (or empty).
"""

from __future__ import annotations

import math

_DEG_SQ = 1e-16
_FEAS_TOL = 1e-08

import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.refinement import kinbody_jacobian as _kinbody_jacobian, lm_refine as _lm_refine
from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix

SOLVER_NAME = "ikgeo.spherical_two_parallel"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 1.2
FLOP_BUDGET = 1316
DISPATCH_REASON = 'Spherical wrist at joints (3, 4, 5) AND axes[1] parallel to axes[2] AND ||p[1]|| ~= 0.\nBoth Pieper specialisations apply (e.g. Puma 560); the parallel-shoulder solver is preferred for slightly tighter elbow conditioning.'

# --- baked KinBody constants ---

_LINK_NAMES = ['base_link', '_poe_link_1', '_poe_link_2', '_poe_link_3', '_poe_link_4', '_poe_link_5', 'wrist_3_link']

_JOINT_NAMES = [
    'joint_1',
    'joint_2',
    'joint_3',
    'joint_4',
    'joint_5',
    'joint_6',
]

_JOINT_AXES = [
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, -1.0, 6.123233995736766e-17], dtype=np.float64),
    np.array([0.0, -1.0, 6.123233995736766e-17], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, -1.0, 6.123233995736766e-17], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.4318], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.020299999999999985], [0.0, 1.0, 0.0, -0.15005], [0.0, 0.0, 1.0, 9.187912610603016e-18], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.4318], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_TYPES = [
    'revolute',
    'revolute',
    'revolute',
    'revolute',
    'revolute',
    'revolute',
]


def _build_kb() -> KinBody:
    """Reconstruct the baked KinBody. Run once at module import."""
    links = [Link(name=n) for n in _LINK_NAMES]
    joints = [
        Joint(
            name=_JOINT_NAMES[i],
            dof_index=i,
            parent_link=links[i],
            T_left=_JOINT_T_LEFTS[i],
            T_right=_JOINT_T_RIGHTS[i],
            axis=_JOINT_AXES[i],
            joint_type=_JOINT_TYPES[i],
        )
        for i in range(len(_JOINT_NAMES))
    ]
    return KinBody(links=links, joints=joints)


_KB = _build_kb()


def _solve_algebraic(T_target):
    """Algebraic IK candidates. Up to 8; verify + dedup in solve().
    """
    r_00 = T_target[0, 0]
    r_01 = T_target[0, 1]
    r_02 = T_target[0, 2]
    r_10 = T_target[1, 0]
    r_11 = T_target[1, 1]
    r_12 = T_target[1, 2]
    r_20 = T_target[2, 0]
    r_21 = T_target[2, 1]
    r_22 = T_target[2, 2]
    p_x = T_target[0, 3]
    p_y = T_target[1, 3]
    p_z = T_target[2, 3]
    candidates = []

    # SP4 for q1 (shoulder pan).
    _q1_R_sq = 1.0*p_x**2 + 1.0*p_y**2
    _q1_rhs = 0.15005 - 6.12323399573677e-17*p_z
    _q1_phi = math.atan2(1.0*p_x, -1.0*p_y)
    if _q1_R_sq < _DEG_SQ:
        theta_q1_plus = 0.0
        theta_q1_minus = 0.0  # degenerate; verify-step drops
    else:
        _q1_R = math.sqrt(_q1_R_sq)
        if abs(_q1_rhs) > _q1_R + _FEAS_TOL:
            # LS fallback: theta = phi (or phi + pi if rhs < 0)
            theta_q1_plus = (
                _q1_phi if _q1_rhs > 0 else _q1_phi + math.pi
            )
            theta_q1_minus = theta_q1_plus
        else:
            _q1_clipped = min(1.0, max(-1.0, _q1_rhs / _q1_R))
            _q1_delta = math.acos(_q1_clipped)
            theta_q1_plus = _q1_phi + _q1_delta
            theta_q1_minus = _q1_phi - _q1_delta

    for q1 in (theta_q1_plus, theta_q1_minus):
        s1 = math.sin(q1)
        c1 = math.cos(q1)
        # SP3 for q3 (elbow): reduces to SP4 with target shift.
        q3_x0 = 1.0*math.sin(q1)
        q3_x1 = math.cos(q1)
        _q3_R_sq = 0.0348408995890292
        _q3_rhs = -0.5*p_z**2 - 1/2*(p_x*q3_x0 - p_y*q3_x1)**2 - 1/2*(-p_x*q3_x1 - p_y*q3_x0)**2 + 0.19791478625
        _q3_phi = -1.52381841044681 + math.pi
        if _q3_R_sq < _DEG_SQ:
            theta_q3_plus = 0.0
            theta_q3_minus = 0.0  # degenerate; verify-step drops
        else:
            _q3_R = math.sqrt(_q3_R_sq)
            if abs(_q3_rhs) > _q3_R + _FEAS_TOL:
                # LS fallback: theta = phi (or phi + pi if rhs < 0)
                theta_q3_plus = (
                    _q3_phi if _q3_rhs > 0 else _q3_phi + math.pi
                )
                theta_q3_minus = theta_q3_plus
            else:
                _q3_clipped = min(1.0, max(-1.0, _q3_rhs / _q3_R))
                _q3_delta = math.acos(_q3_clipped)
                theta_q3_plus = _q3_phi + _q3_delta
                theta_q3_minus = _q3_phi - _q3_delta

        for q3 in (theta_q3_plus, theta_q3_minus):
            s3 = math.sin(q3)
            c3 = math.cos(q3)
            # SP1 for q2 (shoulder pitch): closed-form atan2.
            q2_x0 = math.sin(q3)
            q2_x1 = math.cos(q3)
            q2_x2 = 0.4318*q2_x0 - 0.0203*q2_x1 - 0.4318
            q2_x3 = 1.0*p_z
            q2_x4 = 0.0203*q2_x0 + 0.4318*q2_x1
            q2_x5 = math.cos(q1)
            q2_x6 = math.sin(q1)
            q2_x7 = 1.0*q2_x6
            q2_x8 = -p_x*q2_x5 - p_y*q2_x7
            q2_x9 = p_x*q2_x7 - p_y*q2_x5
            q2 = math.atan2(-q2_x2*q2_x3 + q2_x4*q2_x8 + q2_x9*(2.64401243935914e-17*q2_x0 - 1.24301650113456e-18*q2_x1 - 2.64401243935914e-17), -0.15005*p_x*q2_x6 + 0.15005*p_y*q2_x5 - 9.18791261060302e-18*p_z + q2_x2*q2_x8 - q2_x3*(-q2_x4 - 9.18791261060302e-18) + q2_x9*(-1.24301650113456e-18*q2_x0 - 2.64401243935914e-17*q2_x1 + 0.15005))
            s2 = math.sin(q2)
            c2 = math.cos(q2)
            # SP4 for q5 (wrist pitch).
            q5_x0 = math.sin(q2)
            q5_x1 = math.sin(q3)
            q5_x2 = math.cos(q2)
            q5_x3 = 6.12323399573677e-17*q5_x2 - 6.12323399573677e-17
            q5_x4 = math.cos(q3)
            q5_x5 = 6.12323399573677e-17*q5_x4 - 6.12323399573677e-17
            q5_x6 = 1.0*q5_x2
            q5_x7 = 1.0*q5_x4 + 3.74939945665464e-33
            q5_x8 = 1.0*q5_x7
            q5_x9 = 6.12323399573677e-17*q5_x0
            q5_x10 = -q5_x1*q5_x9 + q5_x3*q5_x7 + q5_x5
            q5_x11 = 1.0*math.sin(q1)
            q5_x12 = math.cos(q1)
            q5_x13 = -q5_x0*q5_x8 - q5_x1*q5_x6 - q5_x5*q5_x9
            _q5_R_sq = 1.00000000000000
            _q5_rhs = 1.0*r_02*(-q5_x10*q5_x11 + q5_x12*q5_x13) + 1.0*r_12*(q5_x10*q5_x12 + q5_x11*q5_x13) + 1.0*r_22*(-1.0*q5_x0*q5_x1 + 1.0*q5_x3*q5_x5 + q5_x8*(q5_x6 + 3.74939945665464e-33)) - 3.74939945665464e-33
            _q5_phi = 0
            if _q5_R_sq < _DEG_SQ:
                theta_q5_plus = 0.0
                theta_q5_minus = 0.0  # degenerate; verify-step drops
            else:
                _q5_R = math.sqrt(_q5_R_sq)
                if abs(_q5_rhs) > _q5_R + _FEAS_TOL:
                    # LS fallback: theta = phi (or phi + pi if rhs < 0)
                    theta_q5_plus = (
                        _q5_phi if _q5_rhs > 0 else _q5_phi + math.pi
                    )
                    theta_q5_minus = theta_q5_plus
                else:
                    _q5_clipped = min(1.0, max(-1.0, _q5_rhs / _q5_R))
                    _q5_delta = math.acos(_q5_clipped)
                    theta_q5_plus = _q5_phi + _q5_delta
                    theta_q5_minus = _q5_phi - _q5_delta

            for q5 in (theta_q5_plus, theta_q5_minus):
                s5 = math.sin(q5)
                c5 = math.cos(q5)
                # SP1 for q4 (wrist roll-1): closed-form atan2.
                q4_x0 = 6.12323399573677e-17*math.cos(q5) - 6.12323399573677e-17
                q4_x1 = math.cos(q3)
                q4_x2 = math.sin(q2)
                q4_x3 = 1.0*q4_x2
                q4_x4 = math.cos(q2)
                q4_x5 = 6.12323399573677e-17*q4_x4 - 6.12323399573677e-17
                q4_x6 = math.sin(q3)
                q4_x7 = 6.12323399573677e-17*q4_x6
                q4_x8 = 1.0*q4_x4 + 3.74939945665464e-33
                q4_x9 = 1.0*q4_x6
                q4_x10 = 1.0*r_22
                q4_x11 = math.cos(q1)
                q4_x12 = q4_x1*q4_x4 - q4_x2*q4_x9
                q4_x13 = 6.12323399573677e-17*q4_x2
                q4_x14 = q4_x1*q4_x13 + q4_x5*q4_x9 + q4_x7
                q4_x15 = 1.0*math.sin(q1)
                q4_x16 = 1.0*r_02
                q4_x17 = 1.0*r_12
                q4_x18 = q4_x10*(q4_x1*q4_x3 + q4_x5*q4_x7 + q4_x8*q4_x9) + q4_x16*(q4_x11*q4_x12 - q4_x14*q4_x15) + q4_x17*(q4_x11*q4_x14 + q4_x12*q4_x15)
                q4_x19 = 6.12323399573677e-17*q4_x1 - 6.12323399573677e-17
                q4_x20 = q4_x19*q4_x5 - 3.74939945665464e-33*q4_x2*q4_x6 + 1.0
                q4_x21 = -q4_x13 - q4_x19*q4_x3 - q4_x4*q4_x7
                q4_x22 = q4_x10*(1.0*q4_x19*q4_x8 - q4_x2*q4_x7 + q4_x5) + q4_x16*(q4_x11*q4_x21 - q4_x15*q4_x20) + q4_x17*(q4_x11*q4_x20 + q4_x15*q4_x21)
                q4_x23 = 1.0*math.sin(q5)
                q4 = math.atan2(-q4_x0*q4_x18 - q4_x22*q4_x23, q4_x0*q4_x22 - q4_x18*q4_x23)
                # SP1 for q6 (wrist roll-2): closed-form atan2.
                q6_x0 = math.sin(q2)
                q6_x1 = math.sin(q3)
                q6_x2 = math.cos(q2)
                q6_x3 = 6.12323399573677e-17*q6_x2 - 6.12323399573677e-17
                q6_x4 = math.cos(q3)
                q6_x5 = 6.12323399573677e-17*q6_x4 - 6.12323399573677e-17
                q6_x6 = 1.0*q6_x2
                q6_x7 = 1.0*q6_x4 + 3.74939945665464e-33
                q6_x8 = 1.0*q6_x7
                q6_x9 = -1.0*q6_x0*q6_x1 + 1.0*q6_x3*q6_x5 + 1.0*q6_x8*(q6_x6 + 3.74939945665464e-33)
                q6_x10 = 6.12323399573677e-17*q6_x0
                q6_x11 = -q6_x1*q6_x10 + q6_x3*q6_x7 + q6_x5
                q6_x12 = 1.0*math.sin(q1)
                q6_x13 = math.cos(q1)
                q6_x14 = -q6_x0*q6_x8 - q6_x1*q6_x6 - q6_x10*q6_x5
                q6_x15 = -1.0*q6_x11*q6_x12 + 1.0*q6_x13*q6_x14
                q6_x16 = 1.0*q6_x11*q6_x13 + 1.0*q6_x12*q6_x14
                q6_x17 = q6_x15*r_01 + q6_x16*r_11 + q6_x9*r_21
                q6_x18 = 1.0*math.sin(q5)
                q6_x19 = math.cos(q5)
                q6_x20 = 6.12323399573677e-17*q6_x19 - 6.12323399573677e-17
                q6_x21 = q6_x15*r_00 + q6_x16*r_10 + q6_x9*r_20
                q6_x22 = 1.0*q6_x19 + 3.74939945665464e-33
                q6_x23 = q6_x15*r_02 + q6_x16*r_12 + q6_x9*r_22
                q6 = math.atan2(-q6_x17*q6_x18 + q6_x20*q6_x21, q6_x17*q6_x20 + q6_x18*q6_x21)
                candidates.append([q1, q2, q3, q4, q5, q6])
    return candidates


def _fk(q):
    """POE forward kinematics using the baked KinBody."""
    T = np.eye(4)
    for j, qi in zip(_KB.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = _rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def _wrap_to_pi(a):
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def solve(
    T_target,
    *,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
):
    """Inverse kinematics. Returns ``(list[Solution], is_ls)``.

    :param T_target: 4x4 SE(3) target end-effector pose.
    :param policy: tolerance policy (FK closure + dedup tolerance).
    :param allow_refinement: opt into Newton-on-spatial-Jacobian
        polish for near-miss candidates (those whose algebraic q
        doesn't quite meet ``fk_atol``). Default off.
    :param refinement_max_iters: cap on Newton iterations per
        candidate when ``allow_refinement=True``.
    """
    T = np.asarray(T_target, dtype=np.float64)
    candidates = _solve_algebraic(T)

    fk_atol = policy.subproblem_numerical
    dedup_atol = policy.subproblem_dedup

    # Three-bucket sort: exact (closes within fk_atol), near-miss
    # (refinable when allow_refinement=True), or drop.
    verified: list[tuple[np.ndarray, float, str, int]] = []
    for cand_q in candidates:
        q = np.asarray(cand_q, dtype=np.float64)
        if not np.all(np.isfinite(q)):
            continue
        T_check = _fk(q)
        residual = float(np.linalg.norm(T_check - T))
        if residual <= fk_atol:
            verified.append((q, residual, "none", 0))
            continue
        if not allow_refinement:
            continue
        # Newton polish using the baked KinBody's spatial Jacobian.
        refined = _lm_refine(
            q,
            _fk,
            T,
            fk_atol=fk_atol,
            max_iters=refinement_max_iters,
            jacobian_fn=lambda qq: _kinbody_jacobian(_KB, qq),
        )
        if refined is None:
            continue
        q_ref, resid_ref, iters = refined
        verified.append((q_ref, resid_ref, "lm", iters))

    # Wrap-to-pi dedup; keep lowest fk_residual on collision.
    deduped: list[tuple[np.ndarray, float, str, int]] = []
    for cand_q, cand_res, ref_used, ref_iters in verified:
        dup_idx = None
        for j, (existing_q, _, _, _) in enumerate(deduped):
            diffs = np.array([_wrap_to_pi(a - b) for a, b in zip(cand_q, existing_q)])
            if np.all(np.abs(diffs) < dedup_atol):
                dup_idx = j
                break
        if dup_idx is None:
            deduped.append((cand_q, cand_res, ref_used, ref_iters))
        elif cand_res < deduped[dup_idx][1]:
            deduped[dup_idx] = (cand_q, cand_res, ref_used, ref_iters)

    solutions = [
        Solution(
            q=q,
            fk_residual=residual,
            refinement_used=ref_used,
            refinement_iters=ref_iters,
            branch_id=i,
            solver_name=SOLVER_NAME,
        )
        for i, (q, residual, ref_used, ref_iters) in enumerate(deduped)
    ]
    return solutions, len(solutions) == 0


__all__ = [
    "DISPATCH_REASON",
    "EXPECTED_MS_MEDIAN",
    "FLOP_BUDGET",
    "SOLVER_NAME",
    "SOLVER_TIER",
    "solve",
]
