"""Generated IK module for UR5.

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash a914c659b3a3 (sha256/12 of the input chain).

Solver: ``ikgeo.three_parallel`` (tier 0)
Expected median IK time: ~1.6 ms on commodity
single-thread hardware. FLOP budget: 2,519 per solve.

Usage:

    import ur5_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions, is_ls = ur5_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``(list[Solution], is_ls)``. ``is_ls=True``
signals that no solution closed within the solver's FK tolerance,
and the returned list is the best-LS approximation (or empty).
"""

from __future__ import annotations

import math
from ssik.subproblems import sp6 as _sp6_runtime

_DEG_SQ = 1e-16
_FEAS_TOL = 1e-08

import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.refinement import kinbody_jacobian as _kinbody_jacobian, lm_refine as _lm_refine
from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix

SOLVER_NAME = "ikgeo.three_parallel"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 1.6
FLOP_BUDGET = 2519
DISPATCH_REASON = 'Three consecutive parallel axes at joints (1, 2, 3) -- the UR-class structure (UR3 / UR5 / UR10).\nClosed-form via SP6 (joints 0+4) + SP1 + SP3.'

# --- baked KinBody constants ---

_LINK_NAMES = ['base_link', '_poe_link_1', '_poe_link_2', '_poe_link_3', '_poe_link_4', '_poe_link_5', 'ee_link']

_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

_JOINT_AXES = [
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, -1.0, 6.123233995736766e-17], dtype=np.float64),
    np.array([0.0, -1.0, 6.123233995736766e-17], dtype=np.float64),
    np.array([0.0, -1.0, 6.123233995736766e-17], dtype=np.float64),
    np.array([0.0, -1.2246467991473532e-16, -1.0], dtype=np.float64),
    np.array([0.0, -1.0, 6.123233995736766e-17], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.089159], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -0.425], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -0.39225000000000004], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.10915], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -1.3877787807814457e-17], [0.0, 0.0, 1.0, -0.09465], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 6.123233995736766e-17, -1.0, -0.0823], [0.0, 1.0, 6.123233995736766e-17, 5.204170427930421e-18], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
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
    """Algebraic IK candidates. Calls runtime SP6 for (q1, q5);
    inlines the post-SP6 SP1+SP3+SP1 chain.
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

    # T_target-derived SP6 inputs.
    p_16_x = p_x - 1.64748849439063e-19*r_01 - 0.0823*r_02
    p_16_y = p_y - 1.64748849439063e-19*r_11 - 0.0823*r_12
    p_16_z = p_z - 1.64748849439063e-19*r_21 - 0.0823*r_22
    r_06_axes5_x = 1.0*r_02
    r_06_axes5_y = 1.0*r_12
    r_06_axes5_z = 1.0*r_22
    _H_SP_0 = np.array([0.0, -1.0, 6.123233995736766e-17])
    _K_SP_0 = np.array([-0.0, -0.0, -1.0])
    _K_SP_1 = np.array([0.0, -1.2246467991473532e-16, -1.0])
    _NEG_P_5 = np.array([-0.0, 1.3877787807814457e-17, 0.09465])
    _NEG_AXES_5 = np.array([-0.0, 1.0, -6.123233995736766e-17])
    # Build SP6 input arrays. h_sp / k_sp constant per arm; p_sp[0],
    # p_sp[2] depend on T_target via the inlined components above.
    p_16 = np.array([p_16_x, p_16_y, p_16_z])
    r_06_axes5 = np.array([r_06_axes5_x, r_06_axes5_y, r_06_axes5_z])
    h_sp = (_H_SP_0, _H_SP_0, _H_SP_0, _H_SP_0)
    k_sp = (_K_SP_0, _K_SP_1, _K_SP_0, _K_SP_1)
    p_sp = (p_16, _NEG_P_5, r_06_axes5, _NEG_AXES_5)
    theta15_solutions, _ = _sp6_runtime.solve(h_sp, k_sp, p_sp, 0.10915, 0.0)

    for q1, q5 in theta15_solutions:
        s1 = math.sin(q1)
        c1 = math.cos(q1)
        s5 = math.sin(q5)
        c5 = math.cos(q5)
        # SP1 for theta14 = q1+q2+q3+q4 (sum of parallel-axis rotations).
        th14_x0 = math.sin(q5)
        th14_x1 = 1.0*r_22
        th14_x2 = math.sin(q1)
        th14_x3 = 6.12323399573677e-17*r_01 - 1.0*r_02
        th14_x4 = th14_x2*th14_x3
        th14_x5 = 1.0*r_01 + 6.12323399573677e-17*r_02
        th14_x6 = 6.12323399573677e-17*th14_x2*th14_x5
        th14_x7 = math.cos(q1)
        th14_x8 = 6.12323399573677e-17*r_11 - 1.0*r_12
        th14_x9 = th14_x7*th14_x8
        th14_x10 = 1.0*r_11 + 6.12323399573677e-17*r_12
        th14_x11 = 6.12323399573677e-17*th14_x10*th14_x7
        th14_x12 = th14_x11 + th14_x4 - th14_x6 - th14_x9
        th14_x13 = math.cos(q5)
        th14_x14 = 6.12323399573677e-17*th14_x10*th14_x2 - th14_x2*th14_x8 - th14_x3*th14_x7 + 6.12323399573677e-17*th14_x5*th14_x7
        th14_x15 = 1.0*th14_x13
        theta14 = math.atan2(-th14_x0*th14_x1 - 6.12323399573677e-17*th14_x0*th14_x12 + th14_x14*(6.12323399573677e-17 - 6.12323399573677e-17*th14_x13), -1.0*th14_x0*th14_x14 + th14_x1*(1.22464679914735e-16*th14_x13 - 6.12323399573677e-17) + th14_x12*(-th14_x15 - 7.49879891330929e-33) - (th14_x15 + 3.74939945665464e-33)*(6.12323399573677e-17*r_22 - th14_x11 - th14_x4 + th14_x6 + th14_x9))
        # SP1 for q6 (wrist roll-2): closed-form atan2.
        q6_x0 = math.cos(q5)
        q6_x1 = math.sin(q1)
        q6_x2 = 1.0*q6_x1
        q6_x3 = math.cos(q1)
        q6_x4 = 1.0*q6_x3
        q6_x5 = q6_x2*r_00 - q6_x4*r_10 + 6.12323399573677e-17*r_20
        q6_x6 = math.sin(q5)
        q6_x7 = 6.12323399573677e-17*r_22
        q6_x8 = 6.12323399573677e-17*r_01 - 1.0*r_02
        q6_x9 = q6_x4*(6.12323399573677e-17*r_11 - 1.0*r_12)
        q6_x10 = q6_x2*q6_x8 - q6_x7 - q6_x9 + 3.74939945665464e-33*r_21
        q6_x11 = 1.0*r_01 + 6.12323399573677e-17*r_02
        q6_x12 = 1.0*r_11 + 6.12323399573677e-17*r_12
        q6_x13 = q6_x11*q6_x2 - q6_x12*q6_x4 + 6.12323399573677e-17*r_21 + 3.74939945665464e-33*r_22
        q6_x14 = 1.0*q6_x6
        q6_x15 = 1.0*q6_x0
        q6 = math.atan2(-6.12323399573677e-17*q6_x10*q6_x6 - q6_x13*q6_x14 + q6_x5*(6.12323399573677e-17*q6_x0 - 6.12323399573677e-17), q6_x10*(-q6_x15 - 7.49879891330929e-33) + q6_x13*(1.22464679914735e-16*q6_x0 - 6.12323399573677e-17) + q6_x14*q6_x5 - (-q6_x15 - 3.74939945665464e-33)*(-6.12323399573677e-17*q6_x1*q6_x11 + 1.0*q6_x1*q6_x8 + 6.12323399573677e-17*q6_x12*q6_x3 - q6_x7 - q6_x9))
        s14 = math.sin(theta14)
        c14 = math.cos(theta14)
        # d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]
        dinr_x0 = math.sin(theta14)
        dinr_x1 = math.sin(q5)
        dinr_x2 = math.cos(theta14)
        dinr_x3 = math.cos(q5)
        dinr_x4 = 1.22464679914735e-16 - 1.22464679914735e-16*dinr_x3
        dinr_x5 = 1.0*dinr_x3 + 1.49975978266186e-32
        dinr_x6 = math.sin(q1)
        dinr_x7 = p_y - 1.64748849439063e-19*r_11 - 0.0823*r_12
        dinr_x8 = math.cos(q1)
        dinr_x9 = p_x - 1.64748849439063e-19*r_01 - 0.0823*r_02
        dinr_x10 = 6.12323399573677e-17*dinr_x2 - 6.12323399573677e-17
        dinr_x11 = 1.38777878078145e-17*dinr_x4
        d_inner_x = -1.96734287847793e-17*dinr_x0*dinr_x4 - 8.49769420904307e-34*dinr_x0*dinr_x5 - 0.09465*dinr_x0 + 2.28650585388476e-18*dinr_x1*dinr_x2 + 1.0*dinr_x6*dinr_x7 + dinr_x8*dinr_x9
        d_inner_y = 1.40008103759583e-34*dinr_x0*dinr_x1 + dinr_x10*dinr_x11 + 5.79564097696485e-18*dinr_x2 + 2.28650585388476e-18*dinr_x3 - 1.0*dinr_x6*dinr_x9 + dinr_x7*dinr_x8 + 0.10915
        d_inner_z = 2.28650585388476e-18*dinr_x0*dinr_x1 + 0.09465*dinr_x10*dinr_x4 + 1.38777878078145e-17*dinr_x10*dinr_x5 + dinr_x11*(1.0*dinr_x2 + 3.74939945665464e-33) + 0.09465*dinr_x2 + 1.0*p_z - 1.64748849439063e-19*r_21 - 0.0823*r_22 - 0.089159
        d_elbow = math.sqrt(d_inner_x*d_inner_x + d_inner_y*d_inner_y + d_inner_z*d_inner_z)
        # SP3 for q3 (elbow): reduces to SP4 with d_elbow target.
        _q3_R_sq = 0.0277909737890625
        _q3_rhs = 0.16724253125 - 1/2*d_elbow**2
        _q3_phi = math.pi
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
            q2_x1 = 0.39225*q2_x0
            q2_x2 = math.cos(q3)
            q2_x3 = -0.39225*q2_x2 - 0.425
            q2 = math.atan2(d_inner_x*q2_x1 + d_inner_y*(-2.40183853482775e-17*q2_x2 - 2.60237444818813e-17) + d_inner_z*q2_x3, d_inner_x*q2_x3 - 2.40183853482775e-17*d_inner_y*q2_x0 - d_inner_z*q2_x1)
            q4 = ((theta14 - q2 - q3 + math.pi) % (2.0 * math.pi)) - math.pi
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
