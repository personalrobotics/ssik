"""Generated IK module for KUKA KR 210 R2700 (Quantec).

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash 4b4f92341f49 (sha256/12 of the input chain).
``T_target`` is the pose of ``link_6`` (end-effector link) in
``base_link`` (base link). If your URDF differs (calibrated
geometry, custom tool past the flange, different link names),
run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
produce an artifact correct for your hardware.

DOF: 6    BASE_LINK: "base_link"    EE_LINK: "link_6"
Solver: ``ikgeo.spherical_two_parallel`` (tier 0)
Expected median IK time: ~1.2 ms on commodity
single-thread hardware. FLOP budget: 1,316 per solve.

Usage:

    import kr210_r2700_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose of link_6 in base_link
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions = kr210_r2700_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``list[Solution]``. Empty list iff no
candidate closed within the solver's FK tolerance -- check
``if not solutions:`` for the "unreachable" case.

Sanity-check the baked geometry: ``kr210_r2700_ik.T_HOME`` is the
4x4 home pose (FK at ``q = np.zeros(DOF)``). If it doesn't match
your robot's home pose, the artifact is for a different URDF.
"""

from __future__ import annotations

import math

_DEG_SQ = 1e-16
_FEAS_TOL = 1e-08

import cython
import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.refinement import lm_refine as _lm_refine
import functools as _functools
from ssik.refinement.rescue import rescue_via_T_perturbation as _rescue_via_T_perturbation
from ssik.postprocess import finalize_solutions as _ps_finalize
from ssik._native import try_native_solve as _try_native_solve
from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix

SOLVER_NAME = "ikgeo.spherical_two_parallel"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 1.2
FLOP_BUDGET = 1316
DISPATCH_REASON = 'Spherical wrist at joints (3, 4, 5) AND axes[1] parallel to axes[2].\nClosed-form via SP4 (shoulder) + SP3 (elbow) + SP1 (wrist). Covers most industrial 6R arms (Puma, Fanuc LR/CR, KUKA KR).'
BASE_LINK = "base_link"
EE_LINK = "link_6"
DOF = 6
# Home pose: FK at q = np.zeros(DOF). Sanity-check this against
# your robot's documented home pose to verify the baked geometry
# matches your URDF.
T_HOME = np.array([[2.83276944882399e-16, 2.83276944882399e-16, -1.0, 2.7], [1.6081226496766366e-16, 1.0, 2.8327694488239903e-16, -4.0609418083024977e-16], [1.0, -1.6081226496766373e-16, 2.8327694488239893e-16, 0.7599999999999997], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

# --- baked KinBody constants ---

_LINK_NAMES = ['base_link', '_poe_link_1', '_poe_link_2', '_poe_link_3', '_poe_link_4', '_poe_link_5', 'link_6']

_JOINT_NAMES = [
    'joint_1',
    'joint_2',
    'joint_3',
    'joint_4',
    'joint_5',
    'joint_6',
]

_JOINT_AXES = [
    np.array([0.0, 1.2246467991473532e-16, -1.0], dtype=np.float64),
    np.array([0.0, 1.0, -1.6081226496766366e-16], dtype=np.float64),
    np.array([0.0, 1.0, -1.6081226496766366e-16], dtype=np.float64),
    np.array([-1.0, 2.8327694488239903e-16, 2.8327694488239893e-16], dtype=np.float64),
    np.array([0.0, 1.0, -1.6081226496766366e-16], dtype=np.float64),
    np.array([-1.0, 2.8327694488239903e-16, 2.8327694488239893e-16], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.33], [0.0, 1.0, 0.0, -7.898971854500428e-17], [0.0, 0.0, 1.0, 0.645], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 1.15], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.8493410471281327e-17], [0.0, 0.0, 1.0, 0.11499999999999999], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 1.2200000000000002], [0.0, 1.0, 0.0, -3.4559787275652684e-16], [0.0, 0.0, 1.0, -3.3306690738754696e-16], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[2.83276944882399e-16, 2.83276944882399e-16, -1.0, 0.0], [1.6081226496766366e-16, 1.0, 2.8327694488239903e-16, 0.0], [1.0, -1.6081226496766373e-16, 2.8327694488239893e-16, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_TYPES = [
    'revolute',
    'revolute',
    'revolute',
    'revolute',
    'revolute',
    'revolute',
]

_JOINT_LIMITS = [
    (-3.2288591161895095, 3.2288591161895095),
    (-2.443460952792061, -0.08726646259971647),
    (-2.0943951023931953, 2.9321531433504737),
    (-6.1086523819801535, 6.1086523819801535),
    (-2.181661564992912, 2.181661564992912),
    (-6.1086523819801535, 6.1086523819801535),
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
            limits=_JOINT_LIMITS[i],
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
    q1_x0 = 1.0*p_y + 1.22464679914735e-16*p_z
    _q1_R_sq = 1.0*p_x**2 + q1_x0**2
    _q1_rhs = -3.46914203822471e-32*p_y + 2.83276944882399e-16*p_z - 5.28311502205674e-16
    _q1_phi = math.atan2(1.0*p_x, q1_x0)
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
        q3_x0 = math.sin(q1)
        q3_x1 = 1.0*q3_x0
        q3_x2 = 1.22464679914735e-16*q3_x0
        q3_x3 = math.cos(q1)
        q3_x4 = 1.22464679914735e-16*q3_x3 - 1.22464679914735e-16
        _q3_R_sq = 1.98589906250000
        _q3_rhs = -1/2*(-p_x*q3_x1 - p_y*(1.0*q3_x3 + 1.49975978266186e-32) - p_z*q3_x4 - 7.89897185450043e-17)**2 - 1/2*(-p_x*q3_x2 - p_y*q3_x4 - 1.0*p_z + 0.645)**2 - 1/2*(-p_x*q3_x3 + p_y*q3_x1 + p_z*q3_x2 + 0.33)**2 + 1.4120625
        _q3_phi = 0.093984588655882 - math.pi
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
            q2_x2 = 0.115*q2_x1
            q2_x3 = math.sin(q1)
            q2_x4 = 1.0*q2_x3
            q2_x5 = 1.22464679914735e-16*q2_x3
            q2_x6 = math.cos(q1)
            q2_x7 = -p_x*q2_x6 + p_y*q2_x4 + p_z*q2_x5 + 0.33
            q2_x8 = 0.115*q2_x0 + 1.22*q2_x1 + 1.15
            q2_x9 = 1.22464679914735e-16*q2_x6 - 1.22464679914735e-16
            q2_x10 = p_y*q2_x9
            q2_x11 = -p_x*q2_x5 - 1.0*p_z - q2_x10 + 0.645
            q2_x12 = p_y*(1.0*q2_x6 + 1.49975978266186e-32)
            q2_x13 = p_z*q2_x9
            q2_x14 = -p_x*q2_x4 - q2_x12 - q2_x13 - 7.89897185450043e-17
            q2 = math.atan2(q2_x11*q2_x8 + q2_x14*(1.84934104712813e-17*q2_x0 + 1.9619096326055e-16*q2_x1 + 1.84934104712813e-16) + q2_x7*(1.22*q2_x0 - q2_x2), 3.45597872756527e-16*p_x*q2_x3 - 5.55763766859835e-32*p_z - 5.55763766859835e-32*q2_x10 + q2_x11*(1.22*q2_x0 - q2_x2 - 5.55763766859835e-32) + 3.45597872756527e-16*q2_x12 + 3.45597872756527e-16*q2_x13 + q2_x14*(1.9619096326055e-16*q2_x0 - 1.84934104712813e-17*q2_x1 + 3.45597872756527e-16) - q2_x7*q2_x8 + 6.31454416612496e-32)
            s2 = math.sin(q2)
            c2 = math.cos(q2)
            # SP4 for q5 (wrist pitch).
            q5_x0 = 1.60812264967664e-16*r_20 + 1.0*r_21 + 2.83276944882399e-16*r_22
            q5_x1 = math.cos(q1)
            q5_x2 = 1.22464679914735e-16*q5_x1 - 1.22464679914735e-16
            q5_x3 = math.sin(q3)
            q5_x4 = 1.60812264967664e-16*q5_x3
            q5_x5 = math.sin(q2)
            q5_x6 = math.cos(q3)
            q5_x7 = 1.60812264967664e-16*q5_x6
            q5_x8 = math.cos(q2)
            q5_x9 = 1.60812264967664e-16*q5_x8 - 1.60812264967664e-16
            q5_x10 = 1.0*q5_x3
            q5_x11 = -q5_x10*q5_x9 - q5_x4 - q5_x5*q5_x7
            q5_x12 = q5_x10*q5_x5
            q5_x13 = -q5_x12 + q5_x6*q5_x8
            q5_x14 = math.sin(q1)
            q5_x15 = 1.22464679914735e-16*q5_x14
            q5_x16 = 1.0*q5_x6
            q5_x17 = 1.0*q5_x8
            q5_x18 = q5_x17 + 2.58605845640301e-32
            q5_x19 = q5_x10*q5_x18 + q5_x16*q5_x5 + q5_x4*q5_x9
            q5_x20 = q5_x11*q5_x2 - q5_x13*q5_x15 - q5_x19
            q5_x21 = 2.83276944882399e-16*r_20 + 2.83276944882399e-16*r_21 - 1.0*r_22
            q5_x22 = 1.0*r_20 - 1.60812264967664e-16*r_21 + 2.83276944882399e-16*r_22
            q5_x23 = 1.60812264967664e-16*r_00 + 1.0*r_01 + 2.83276944882399e-16*r_02
            q5_x24 = 1.0*q5_x14
            q5_x25 = -q5_x19
            q5_x26 = q5_x1*q5_x13 + q5_x11*q5_x24 + q5_x15*q5_x25
            q5_x27 = 2.83276944882399e-16*r_00 + 2.83276944882399e-16*r_01 - 1.0*r_02
            q5_x28 = 1.0*r_00 - 1.60812264967664e-16*r_01 + 2.83276944882399e-16*r_02
            q5_x29 = q5_x7 - 1.60812264967664e-16
            q5_x30 = q5_x29*q5_x9
            q5_x31 = -2.58605845640301e-32*q5_x3*q5_x5 + q5_x30 + 1.0
            q5_x32 = q5_x18*q5_x29
            q5_x33 = 1.60812264967664e-16*q5_x5
            q5_x34 = 1.0*q5_x5
            q5_x35 = q5_x29*q5_x34 + q5_x33 + q5_x4*q5_x8
            q5_x36 = -q5_x4*q5_x5
            q5_x37 = q5_x36 + q5_x9
            q5_x38 = -q5_x15*q5_x35 + q5_x2*q5_x31 + 1.0*q5_x32 + q5_x37
            q5_x39 = q5_x32 + q5_x37
            q5_x40 = q5_x1*q5_x35 + q5_x15*q5_x39 + q5_x24*q5_x31
            q5_x41 = 1.60812264967664e-16*r_10 + 1.0*r_11 + 2.83276944882399e-16*r_12
            q5_x42 = 1.0*q5_x1 + 1.49975978266186e-32
            q5_x43 = q5_x11*q5_x42 - q5_x13*q5_x24 + q5_x2*q5_x25
            q5_x44 = 2.83276944882399e-16*r_10 + 2.83276944882399e-16*r_11 - 1.0*r_12
            q5_x45 = 1.0*r_10 - 1.60812264967664e-16*r_11 + 2.83276944882399e-16*r_12
            q5_x46 = q5_x2*q5_x39 - q5_x24*q5_x35 + q5_x31*q5_x42
            q5_x47 = -q5_x12
            q5_x48 = q5_x16 + 2.58605845640301e-32
            q5_x49 = q5_x18*q5_x48
            q5_x50 = q5_x17*q5_x3 + q5_x29*q5_x33 + q5_x34*q5_x48
            q5_x51 = q5_x29 + q5_x36 + q5_x48*q5_x9
            q5_x52 = -q5_x15*q5_x50 + q5_x2*q5_x51 + 1.0*q5_x30 + q5_x47 + 1.0*q5_x49
            q5_x53 = q5_x30 + q5_x47 + q5_x49
            q5_x54 = q5_x1*q5_x50 + q5_x15*q5_x53 + q5_x24*q5_x51
            q5_x55 = q5_x2*q5_x53 - q5_x24*q5_x50 + q5_x42*q5_x51
            _q5_R_sq = 1.00000000000000
            _q5_rhs = -2.83276944882399e-16*q5_x0*q5_x20 + 8.02458275019057e-32*q5_x0*q5_x38 + 8.02458275019057e-32*q5_x0*q5_x52 + 1.0*q5_x20*q5_x21 - 2.83276944882399e-16*q5_x20*q5_x22 - 2.83276944882399e-16*q5_x21*q5_x38 - 2.83276944882399e-16*q5_x21*q5_x52 + 8.02458275019057e-32*q5_x22*q5_x38 + 8.02458275019057e-32*q5_x22*q5_x52 - 2.83276944882399e-16*q5_x23*q5_x26 + 8.02458275019057e-32*q5_x23*q5_x40 + 8.02458275019057e-32*q5_x23*q5_x54 + 1.0*q5_x26*q5_x27 - 2.83276944882399e-16*q5_x26*q5_x28 - 2.83276944882399e-16*q5_x27*q5_x40 - 2.83276944882399e-16*q5_x27*q5_x54 + 8.02458275019057e-32*q5_x28*q5_x40 + 8.02458275019057e-32*q5_x28*q5_x54 - 2.83276944882399e-16*q5_x41*q5_x43 + 8.02458275019057e-32*q5_x41*q5_x46 + 8.02458275019057e-32*q5_x41*q5_x55 + 1.0*q5_x43*q5_x44 - 2.83276944882399e-16*q5_x43*q5_x45 - 2.83276944882399e-16*q5_x44*q5_x46 - 2.83276944882399e-16*q5_x44*q5_x55 + 8.02458275019057e-32*q5_x45*q5_x46 + 8.02458275019057e-32*q5_x45*q5_x55 - 8.02458275019057e-32
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
                q4_x0 = math.sin(q5)
                q4_x1 = 2.83276944882399e-16*q4_x0
                q4_x2 = math.cos(q5)
                q4_x3 = math.cos(q1)
                q4_x4 = 1.22464679914735e-16*q4_x3 - 1.22464679914735e-16
                q4_x5 = math.sin(q3)
                q4_x6 = 1.60812264967664e-16*q4_x5
                q4_x7 = math.cos(q3)
                q4_x8 = math.sin(q2)
                q4_x9 = 1.60812264967664e-16*q4_x8
                q4_x10 = math.cos(q2)
                q4_x11 = 1.60812264967664e-16*q4_x10 - 1.60812264967664e-16
                q4_x12 = 1.0*q4_x5
                q4_x13 = -q4_x11*q4_x12 - q4_x6 - q4_x7*q4_x9
                q4_x14 = q4_x12*q4_x8
                q4_x15 = q4_x10*q4_x7 - q4_x14
                q4_x16 = math.sin(q1)
                q4_x17 = 1.22464679914735e-16*q4_x16
                q4_x18 = 1.0*q4_x7
                q4_x19 = 1.0*q4_x10
                q4_x20 = q4_x19 + 2.58605845640301e-32
                q4_x21 = q4_x11*q4_x6 + q4_x12*q4_x20 + q4_x18*q4_x8
                q4_x22 = q4_x13*q4_x4 - q4_x15*q4_x17 - q4_x21
                q4_x23 = 1.60812264967664e-16*r_20 + 1.0*r_21 + 2.83276944882399e-16*r_22
                q4_x24 = 2.83276944882399e-16*q4_x23
                q4_x25 = q4_x22*q4_x24
                q4_x26 = 2.83276944882399e-16*r_20 + 2.83276944882399e-16*r_21 - 1.0*r_22
                q4_x27 = 1.0*q4_x26
                q4_x28 = q4_x22*q4_x27
                q4_x29 = 1.0*r_20 - 1.60812264967664e-16*r_21 + 2.83276944882399e-16*r_22
                q4_x30 = 2.83276944882399e-16*q4_x29
                q4_x31 = q4_x22*q4_x30
                q4_x32 = 1.0*q4_x16
                q4_x33 = -q4_x21
                q4_x34 = q4_x13*q4_x32 + q4_x15*q4_x3 + q4_x17*q4_x33
                q4_x35 = 1.60812264967664e-16*r_00 + 1.0*r_01 + 2.83276944882399e-16*r_02
                q4_x36 = 2.83276944882399e-16*q4_x35
                q4_x37 = q4_x34*q4_x36
                q4_x38 = 2.83276944882399e-16*r_00 + 2.83276944882399e-16*r_01 - 1.0*r_02
                q4_x39 = 1.0*q4_x38
                q4_x40 = q4_x34*q4_x39
                q4_x41 = 1.0*r_00 - 1.60812264967664e-16*r_01 + 2.83276944882399e-16*r_02
                q4_x42 = 2.83276944882399e-16*q4_x41
                q4_x43 = q4_x34*q4_x42
                q4_x44 = 1.0*q4_x3 + 1.49975978266186e-32
                q4_x45 = q4_x13*q4_x44 - q4_x15*q4_x32 + q4_x33*q4_x4
                q4_x46 = 1.60812264967664e-16*r_10 + 1.0*r_11 + 2.83276944882399e-16*r_12
                q4_x47 = 2.83276944882399e-16*q4_x46
                q4_x48 = q4_x45*q4_x47
                q4_x49 = 2.83276944882399e-16*r_10 + 2.83276944882399e-16*r_11 - 1.0*r_12
                q4_x50 = 1.0*q4_x49
                q4_x51 = q4_x45*q4_x50
                q4_x52 = 1.0*r_10 - 1.60812264967664e-16*r_11 + 2.83276944882399e-16*r_12
                q4_x53 = 2.83276944882399e-16*q4_x52
                q4_x54 = q4_x45*q4_x53
                q4_x55 = q4_x25 - q4_x28 + q4_x31 + q4_x37 - q4_x40 + q4_x43 + q4_x48 - q4_x51 + q4_x54
                q4_x56 = 1.0*q4_x0 - 4.55544071196586e-32
                q4_x57 = 1.60812264967664e-16*q4_x7 - 1.60812264967664e-16
                q4_x58 = q4_x11*q4_x57
                q4_x59 = -2.58605845640301e-32*q4_x5*q4_x8 + q4_x58 + 1.0
                q4_x60 = q4_x20*q4_x57
                q4_x61 = 1.0*q4_x8
                q4_x62 = q4_x10*q4_x6 + q4_x57*q4_x61 + q4_x9
                q4_x63 = -q4_x6*q4_x8
                q4_x64 = q4_x11 + q4_x63
                q4_x65 = -q4_x17*q4_x62 + q4_x4*q4_x59 + 1.0*q4_x60 + q4_x64
                q4_x66 = q4_x60 + q4_x64
                q4_x67 = q4_x17*q4_x66 + q4_x3*q4_x62 + q4_x32*q4_x59
                q4_x68 = -q4_x32*q4_x62 + q4_x4*q4_x66 + q4_x44*q4_x59
                q4_x69 = q4_x24*q4_x65 - q4_x27*q4_x65 + q4_x30*q4_x65 + q4_x36*q4_x67 - q4_x39*q4_x67 + q4_x42*q4_x67 + q4_x47*q4_x68 - q4_x50*q4_x68 + q4_x53*q4_x68
                q4_x70 = -q4_x14
                q4_x71 = q4_x18 + 2.58605845640301e-32
                q4_x72 = q4_x20*q4_x71
                q4_x73 = q4_x19*q4_x5 + q4_x57*q4_x9 + q4_x61*q4_x71
                q4_x74 = q4_x11*q4_x71 + q4_x57 + q4_x63
                q4_x75 = -q4_x17*q4_x73 + q4_x4*q4_x74 + 1.0*q4_x58 + q4_x70 + 1.0*q4_x72
                q4_x76 = q4_x58 + q4_x70 + q4_x72
                q4_x77 = q4_x17*q4_x76 + q4_x3*q4_x73 + q4_x32*q4_x74
                q4_x78 = -q4_x32*q4_x73 + q4_x4*q4_x76 + q4_x44*q4_x74
                q4_x79 = q4_x24*q4_x75 - q4_x27*q4_x75 + q4_x30*q4_x75 + q4_x36*q4_x77 - q4_x39*q4_x77 + q4_x42*q4_x77 + q4_x47*q4_x78 - q4_x50*q4_x78 + q4_x53*q4_x78
                q4_x80 = 1.0*q4_x2
                q4 = math.atan2(q4_x55*(q4_x1 + 8.02458275019057e-32*q4_x2 - 8.02458275019057e-32) + q4_x69*(4.93038065763132e-32*q4_x2 + q4_x56) + q4_x79*(-1.60812264967664e-16*q4_x0 + 2.83276944882399e-16*q4_x2 - 2.83276944882399e-16), q4_x55*(q4_x1 - q4_x80) + q4_x69*(1.60812264967664e-16*q4_x0 + 4.55544071196586e-32*q4_x2 + 2.83276944882399e-16) + q4_x79*(2.83276944882399e-16*q4_x2 + q4_x56) - (q4_x80 + 8.02458275019057e-32)*(8.02458275019057e-32*q4_x23*q4_x65 + 8.02458275019057e-32*q4_x23*q4_x75 - q4_x25 - 2.83276944882399e-16*q4_x26*q4_x65 - 2.83276944882399e-16*q4_x26*q4_x75 + q4_x28 + 8.02458275019057e-32*q4_x29*q4_x65 + 8.02458275019057e-32*q4_x29*q4_x75 - q4_x31 + 8.02458275019057e-32*q4_x35*q4_x67 + 8.02458275019057e-32*q4_x35*q4_x77 - q4_x37 - 2.83276944882399e-16*q4_x38*q4_x67 - 2.83276944882399e-16*q4_x38*q4_x77 + q4_x40 + 8.02458275019057e-32*q4_x41*q4_x67 + 8.02458275019057e-32*q4_x41*q4_x77 - q4_x43 + 8.02458275019057e-32*q4_x46*q4_x68 + 8.02458275019057e-32*q4_x46*q4_x78 - q4_x48 - 2.83276944882399e-16*q4_x49*q4_x68 - 2.83276944882399e-16*q4_x49*q4_x78 + q4_x51 + 8.02458275019057e-32*q4_x52*q4_x68 + 8.02458275019057e-32*q4_x52*q4_x78 - q4_x54))
                # SP1 for q6 (wrist roll-2): closed-form atan2.
                q6_x0 = math.sin(q5)
                q6_x1 = math.cos(q5)
                q6_x2 = 2.83276944882399e-16*q6_x1
                q6_x3 = 1.0*r_20 - 1.60812264967664e-16*r_21 + 2.83276944882399e-16*r_22
                q6_x4 = math.cos(q1)
                q6_x5 = 1.22464679914735e-16*q6_x4 - 1.22464679914735e-16
                q6_x6 = math.sin(q3)
                q6_x7 = 1.60812264967664e-16*q6_x6
                q6_x8 = math.sin(q2)
                q6_x9 = math.cos(q3)
                q6_x10 = 1.60812264967664e-16*q6_x9
                q6_x11 = math.cos(q2)
                q6_x12 = 1.60812264967664e-16*q6_x11
                q6_x13 = q6_x12 - 1.60812264967664e-16
                q6_x14 = 1.0*q6_x6
                q6_x15 = -q6_x10*q6_x8 - q6_x13*q6_x14 - q6_x7
                q6_x16 = 1.0*q6_x8
                q6_x17 = q6_x16*q6_x6
                q6_x18 = q6_x11*q6_x9 - q6_x17
                q6_x19 = math.sin(q1)
                q6_x20 = 1.22464679914735e-16*q6_x19
                q6_x21 = 1.0*q6_x9
                q6_x22 = 1.0*q6_x11
                q6_x23 = q6_x22 + 2.58605845640301e-32
                q6_x24 = q6_x13*q6_x7 + q6_x14*q6_x23 + q6_x21*q6_x8
                q6_x25 = q6_x15*q6_x5 - q6_x18*q6_x20 - q6_x24
                q6_x26 = 1.0*q6_x25
                q6_x27 = 1.0*r_00 - 1.60812264967664e-16*r_01 + 2.83276944882399e-16*r_02
                q6_x28 = 1.0*q6_x19
                q6_x29 = -q6_x24
                q6_x30 = q6_x15*q6_x28 + q6_x18*q6_x4 + q6_x20*q6_x29
                q6_x31 = 1.0*q6_x30
                q6_x32 = q6_x10 - 1.60812264967664e-16
                q6_x33 = q6_x13*q6_x32
                q6_x34 = q6_x33 - 2.58605845640301e-32*q6_x6*q6_x8 + 1.0
                q6_x35 = q6_x23*q6_x32
                q6_x36 = 1.60812264967664e-16*q6_x8
                q6_x37 = q6_x12*q6_x6 + q6_x16*q6_x32 + q6_x36
                q6_x38 = -q6_x36*q6_x6
                q6_x39 = q6_x13 + q6_x38
                q6_x40 = -q6_x20*q6_x37 + q6_x34*q6_x5 + 1.0*q6_x35 + q6_x39
                q6_x41 = 2.83276944882399e-16*q6_x40
                q6_x42 = q6_x35 + q6_x39
                q6_x43 = q6_x20*q6_x42 + q6_x28*q6_x34 + q6_x37*q6_x4
                q6_x44 = 2.83276944882399e-16*q6_x43
                q6_x45 = 1.0*r_10 - 1.60812264967664e-16*r_11 + 2.83276944882399e-16*r_12
                q6_x46 = 1.0*q6_x4 + 1.49975978266186e-32
                q6_x47 = q6_x15*q6_x46 - q6_x18*q6_x28 + q6_x29*q6_x5
                q6_x48 = 1.0*q6_x47
                q6_x49 = -q6_x28*q6_x37 + q6_x34*q6_x46 + q6_x42*q6_x5
                q6_x50 = 2.83276944882399e-16*q6_x49
                q6_x51 = -q6_x17
                q6_x52 = q6_x21 + 2.58605845640301e-32
                q6_x53 = q6_x23*q6_x52
                q6_x54 = q6_x16*q6_x52 + q6_x22*q6_x6 + q6_x32*q6_x36
                q6_x55 = q6_x13*q6_x52 + q6_x32 + q6_x38
                q6_x56 = -q6_x20*q6_x54 + 1.0*q6_x33 + q6_x5*q6_x55 + q6_x51 + 1.0*q6_x53
                q6_x57 = 2.83276944882399e-16*q6_x56
                q6_x58 = q6_x33 + q6_x51 + q6_x53
                q6_x59 = q6_x20*q6_x58 + q6_x28*q6_x55 + q6_x4*q6_x54
                q6_x60 = 2.83276944882399e-16*q6_x59
                q6_x61 = -q6_x28*q6_x54 + q6_x46*q6_x55 + q6_x5*q6_x58
                q6_x62 = 2.83276944882399e-16*q6_x61
                q6_x63 = -q6_x26*q6_x3 - q6_x27*q6_x31 + q6_x27*q6_x44 + q6_x27*q6_x60 + q6_x3*q6_x41 + q6_x3*q6_x57 - q6_x45*q6_x48 + q6_x45*q6_x50 + q6_x45*q6_x62
                q6_x64 = 2.83276944882399e-16*q6_x0
                q6_x65 = 2.83276944882399e-16*r_20 + 2.83276944882399e-16*r_21 - 1.0*r_22
                q6_x66 = q6_x26*q6_x65
                q6_x67 = 2.83276944882399e-16*r_00 + 2.83276944882399e-16*r_01 - 1.0*r_02
                q6_x68 = q6_x31*q6_x67
                q6_x69 = 2.83276944882399e-16*r_10 + 2.83276944882399e-16*r_11 - 1.0*r_12
                q6_x70 = q6_x48*q6_x69
                q6_x71 = q6_x41*q6_x65 + q6_x44*q6_x67 + q6_x50*q6_x69 + q6_x57*q6_x65 + q6_x60*q6_x67 + q6_x62*q6_x69 - q6_x66 - q6_x68 - q6_x70
                q6_x72 = 1.0*q6_x0 + 4.55544071196586e-32
                q6_x73 = 1.60812264967664e-16*r_20 + 1.0*r_21 + 2.83276944882399e-16*r_22
                q6_x74 = 1.60812264967664e-16*r_00 + 1.0*r_01 + 2.83276944882399e-16*r_02
                q6_x75 = 1.60812264967664e-16*r_10 + 1.0*r_11 + 2.83276944882399e-16*r_12
                q6_x76 = -q6_x26*q6_x73 - q6_x31*q6_x74 + q6_x41*q6_x73 + q6_x44*q6_x74 - q6_x48*q6_x75 + q6_x50*q6_x75 + q6_x57*q6_x73 + q6_x60*q6_x74 + q6_x62*q6_x75
                q6_x77 = 1.0*q6_x1
                q6 = math.atan2(q6_x63*(-1.60812264967664e-16*q6_x0 - q6_x2 + 2.83276944882399e-16) + q6_x71*(-8.02458275019057e-32*q6_x1 + q6_x64 + 8.02458275019057e-32) + q6_x76*(-4.93038065763132e-32*q6_x1 + q6_x72), q6_x63*(q6_x2 - q6_x72) + q6_x71*(-q6_x64 - q6_x77) + q6_x76*(-1.60812264967664e-16*q6_x0 + 4.55544071196586e-32*q6_x1 + 2.83276944882399e-16) - (-q6_x77 - 8.02458275019057e-32)*(2.83276944882399e-16*q6_x25*q6_x3 + 2.83276944882399e-16*q6_x25*q6_x73 + 2.83276944882399e-16*q6_x27*q6_x30 - 8.02458275019057e-32*q6_x27*q6_x43 - 8.02458275019057e-32*q6_x27*q6_x59 - 8.02458275019057e-32*q6_x3*q6_x40 - 8.02458275019057e-32*q6_x3*q6_x56 + 2.83276944882399e-16*q6_x30*q6_x74 + 2.83276944882399e-16*q6_x40*q6_x65 - 8.02458275019057e-32*q6_x40*q6_x73 + 2.83276944882399e-16*q6_x43*q6_x67 - 8.02458275019057e-32*q6_x43*q6_x74 + 2.83276944882399e-16*q6_x45*q6_x47 - 8.02458275019057e-32*q6_x45*q6_x49 - 8.02458275019057e-32*q6_x45*q6_x61 + 2.83276944882399e-16*q6_x47*q6_x75 + 2.83276944882399e-16*q6_x49*q6_x69 - 8.02458275019057e-32*q6_x49*q6_x75 + 2.83276944882399e-16*q6_x56*q6_x65 - 8.02458275019057e-32*q6_x56*q6_x73 + 2.83276944882399e-16*q6_x59*q6_x67 - 8.02458275019057e-32*q6_x59*q6_x74 + 2.83276944882399e-16*q6_x61*q6_x69 - 8.02458275019057e-32*q6_x61*q6_x75 - q6_x66 - q6_x68 - q6_x70))
                candidates.append([q1, q2, q3, q4, q5, q6])
    return candidates


# Module-scope ``2*pi`` constant referenced inside the dedup hot
# loop (Cython compiles ``_TWO_PI`` to a typed C ``double``).
_TWO_PI: float = 2.0 * math.pi

# Cached 4x4 identity reused inside ``_fk`` / ``_spatial_jacobian``
# so each call avoids ``len(_JOINT_AXES)+1`` per-iteration ``np.eye(4)``
# allocations -- the orchestrator's #1 hotspot per Slice 4 profile
# (~22% of ``_fk`` cost on Puma 560).
_FK_EYE4 = np.eye(4, dtype=np.float64)
_FK_EYE4.flags.writeable = False


@cython.ccall
@cython.locals(
    i=cython.int,
    n=cython.int,
    ax=cython.double, ay=cython.double, az=cython.double,
    qi=cython.double, c=cython.double, s=cython.double, oc=cython.double,
    r00=cython.double, r01=cython.double, r02=cython.double,
    r10=cython.double, r11=cython.double, r12=cython.double,
    r20=cython.double, r21=cython.double, r22=cython.double,
    l00=cython.double, l01=cython.double, l02=cython.double, l03=cython.double,
    l10=cython.double, l11=cython.double, l12=cython.double, l13=cython.double,
    l20=cython.double, l21=cython.double, l22=cython.double, l23=cython.double,
    m00=cython.double, m01=cython.double, m02=cython.double, m03=cython.double,
    m10=cython.double, m11=cython.double, m12=cython.double, m13=cython.double,
    m20=cython.double, m21=cython.double, m22=cython.double, m23=cython.double,
    t00=cython.double, t01=cython.double, t02=cython.double, t03=cython.double,
    t10=cython.double, t11=cython.double, t12=cython.double, t13=cython.double,
    t20=cython.double, t21=cython.double, t22=cython.double, t23=cython.double,
    n00=cython.double, n01=cython.double, n02=cython.double, n03=cython.double,
    n10=cython.double, n11=cython.double, n12=cython.double, n13=cython.double,
    n20=cython.double, n21=cython.double, n22=cython.double, n23=cython.double,
    a00=cython.double, a01=cython.double, a02=cython.double, a03=cython.double,
    a10=cython.double, a11=cython.double, a12=cython.double, a13=cython.double,
    a20=cython.double, a21=cython.double, a22=cython.double, a23=cython.double,
    b00=cython.double, b01=cython.double, b02=cython.double, b03=cython.double,
    b10=cython.double, b11=cython.double, b12=cython.double, b13=cython.double,
    b20=cython.double, b21=cython.double, b22=cython.double, b23=cython.double,
)
def _fk(q):
    """POE forward kinematics using the baked chain constants.

    Hand-rolled scalar 4x4 matmul + inline Rodrigues -- no per-call
    ``np.eye(4)`` allocations and no per-joint numpy ``@`` dispatch.
    Each numpy ``@`` on a 4x4 has ~3 us of dispatch overhead;
    inlining the ~85 scalar ops per joint turns the inner loop into
    a single native-code chunk under Cython compile.

    Bottom row of the accumulator stays [0, 0, 0, 1] implicitly.
    """
    n = len(_JOINT_AXES)
    # Identity accumulator (the bottom row [0,0,0,1] is implicit).
    a00 = 1.0; a01 = 0.0; a02 = 0.0; a03 = 0.0
    a10 = 0.0; a11 = 1.0; a12 = 0.0; a13 = 0.0
    a20 = 0.0; a21 = 0.0; a22 = 1.0; a23 = 0.0
    for i in range(n):
        # Inline Rodrigues for this joint's axis.
        ax = float(_JOINT_AXES[i][0])
        ay = float(_JOINT_AXES[i][1])
        az = float(_JOINT_AXES[i][2])
        qi = float(q[i])
        c = math.cos(qi); s = math.sin(qi); oc = 1.0 - c
        r00 = c + ax*ax*oc;     r01 = ax*ay*oc - az*s; r02 = ax*az*oc + ay*s
        r10 = ay*ax*oc + az*s;  r11 = c + ay*ay*oc;    r12 = ay*az*oc - ax*s
        r20 = az*ax*oc - ay*s;  r21 = az*ay*oc + ax*s; r22 = c + az*az*oc
        # T_left[i] entries.
        Tl = _JOINT_T_LEFTS[i]
        l00 = float(Tl[0,0]); l01 = float(Tl[0,1])
        l02 = float(Tl[0,2]); l03 = float(Tl[0,3])
        l10 = float(Tl[1,0]); l11 = float(Tl[1,1])
        l12 = float(Tl[1,2]); l13 = float(Tl[1,3])
        l20 = float(Tl[2,0]); l21 = float(Tl[2,1])
        l22 = float(Tl[2,2]); l23 = float(Tl[2,3])
        # M = T_left[i] @ R (R is the homogeneous version of the 3x3
        # rotation above with column 3 = [0,0,0,1]^T).
        m00 = l00*r00 + l01*r10 + l02*r20
        m01 = l00*r01 + l01*r11 + l02*r21
        m02 = l00*r02 + l01*r12 + l02*r22
        m03 = l03
        m10 = l10*r00 + l11*r10 + l12*r20
        m11 = l10*r01 + l11*r11 + l12*r21
        m12 = l10*r02 + l11*r12 + l12*r22
        m13 = l13
        m20 = l20*r00 + l21*r10 + l22*r20
        m21 = l20*r01 + l21*r11 + l22*r21
        m22 = l20*r02 + l21*r12 + l22*r22
        m23 = l23
        # T_right[i] entries.
        Tr = _JOINT_T_RIGHTS[i]
        t00 = float(Tr[0,0]); t01 = float(Tr[0,1])
        t02 = float(Tr[0,2]); t03 = float(Tr[0,3])
        t10 = float(Tr[1,0]); t11 = float(Tr[1,1])
        t12 = float(Tr[1,2]); t13 = float(Tr[1,3])
        t20 = float(Tr[2,0]); t21 = float(Tr[2,1])
        t22 = float(Tr[2,2]); t23 = float(Tr[2,3])
        # N = M @ T_right[i]
        n00 = m00*t00 + m01*t10 + m02*t20
        n01 = m00*t01 + m01*t11 + m02*t21
        n02 = m00*t02 + m01*t12 + m02*t22
        n03 = m00*t03 + m01*t13 + m02*t23 + m03
        n10 = m10*t00 + m11*t10 + m12*t20
        n11 = m10*t01 + m11*t11 + m12*t21
        n12 = m10*t02 + m11*t12 + m12*t22
        n13 = m10*t03 + m11*t13 + m12*t23 + m13
        n20 = m20*t00 + m21*t10 + m22*t20
        n21 = m20*t01 + m21*t11 + m22*t21
        n22 = m20*t02 + m21*t12 + m22*t22
        n23 = m20*t03 + m21*t13 + m22*t23 + m23
        # T_acc = T_acc @ N
        b00 = a00*n00 + a01*n10 + a02*n20
        b01 = a00*n01 + a01*n11 + a02*n21
        b02 = a00*n02 + a01*n12 + a02*n22
        b03 = a00*n03 + a01*n13 + a02*n23 + a03
        b10 = a10*n00 + a11*n10 + a12*n20
        b11 = a10*n01 + a11*n11 + a12*n21
        b12 = a10*n02 + a11*n12 + a12*n22
        b13 = a10*n03 + a11*n13 + a12*n23 + a13
        b20 = a20*n00 + a21*n10 + a22*n20
        b21 = a20*n01 + a21*n11 + a22*n21
        b22 = a20*n02 + a21*n12 + a22*n22
        b23 = a20*n03 + a21*n13 + a22*n23 + a23
        a00, a01, a02, a03 = b00, b01, b02, b03
        a10, a11, a12, a13 = b10, b11, b12, b13
        a20, a21, a22, a23 = b20, b21, b22, b23
    return np.array(
        [[a00, a01, a02, a03],
         [a10, a11, a12, a13],
         [a20, a21, a22, a23],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


@cython.ccall
@cython.locals(i=cython.int, n=cython.int)
def _spatial_jacobian(q):
    """6 x n_dof spatial Jacobian using the baked chain constants.

    Math identical to ssik.refinement.kinbody_jacobian: column i
    is (p_i x z_i, z_i) where z_i is the i-th joint axis in the
    world frame at q and p_i is the i-th joint origin. This is
    the SPATIAL twist representation -- T(q+dq) @ T(q)^-1 ~
    exp([J @ dq]) -- matching the residual extracted by
    ssik.refinement.se3_log_residual. Per-arm version with
    baked _JOINT_AXES / _JOINT_T_LEFTS / _JOINT_T_RIGHTS so
    there's no KinBody walk at runtime.
    """
    n = len(_JOINT_AXES)
    cum = _FK_EYE4.copy()
    cums = [cum.copy()]
    rot = _FK_EYE4.copy()
    for i in range(n):
        rot[:3, :3] = _rotation_matrix(_JOINT_AXES[i], float(q[i]))
        cum = cum @ _JOINT_T_LEFTS[i] @ rot @ _JOINT_T_RIGHTS[i]
        cums.append(cum.copy())
    J = np.zeros((6, n), dtype=np.float64)
    for i in range(n):
        t_pre = cums[i] @ _JOINT_T_LEFTS[i]
        axis_unit = _JOINT_AXES[i] / np.linalg.norm(_JOINT_AXES[i])
        z_i = t_pre[:3, :3] @ axis_unit
        p_i = t_pre[:3, 3]
        # Scalar cross product p_i x z_i: bit-identical to np.cross but
        # ~11x faster for 3-vectors (np.cross's moveaxis / axis-normalize
        # overhead dominates the LM-refine loop). Mirrors the live
        # kinbody_jacobian, which is already scalar-inlined.
        J[0, i] = p_i[1] * z_i[2] - p_i[2] * z_i[1]
        J[1, i] = p_i[2] * z_i[0] - p_i[0] * z_i[2]
        J[2, i] = p_i[0] * z_i[1] - p_i[1] * z_i[0]
        J[3:, i] = z_i
    return J


@cython.ccall
def _wrap_to_pi(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``. Called inside the per-IK
    dedup hot loop (235k+ times on Franka 7R)."""
    return ((a + math.pi) % _TWO_PI) - math.pi


@cython.ccall
@cython.locals(
    i=cython.int,
    n=cython.int,
    diff=cython.double,
    ai=cython.double,
    bi=cython.double,
)
def _q_close_wrap(a, b, tol: float) -> bool:
    """Return ``True`` if joint vectors ``a`` and ``b`` agree (mod 2pi)
    within ``tol`` per element. Replaces the
    ``np.array([_wrap_to_pi(...)]) -> np.all(np.abs(...) < tol)``
    pipeline that allocated a numpy array per dedup-loop iteration --
    a per-element scalar loop avoids the array creation and the
    ``np.all`` reduction overhead, which together dominated the
    artifact's ``solve()`` body at the per-IK level."""
    n = len(a)
    for i in range(n):
        ai = float(a[i])
        bi = float(b[i])
        diff = ((ai - bi + math.pi) % _TWO_PI) - math.pi
        if abs(diff) > tol:
            return False
    return True


def solve(
    T_target,
    *,
    max_solutions: int | None = None,
    q_seed=None,
    respect_limits: bool = True,
    allow_refinement: bool = False,
    allow_rescue: bool = True,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    refinement_max_iters: int = 15,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    native: bool = False,
):
    """Inverse kinematics. Returns ``list[Solution]``.

    :param T_target: 4x4 SE(3) target end-effector pose.
    :param max_solutions: optional cap on returned IKs (post-dedup,
        post-limits filter). ``None`` = full redundancy enumeration.
        Combine with ``q_seed`` for the "give me the IK closest to
        where I am now" trajectory-tracking idiom.
    :param q_seed: optional joint configuration. When provided,
        returned solutions are sorted by distance from ``q_seed``
        (closest first, via ``seed_metric``); with ``max_solutions``
        this returns the nearest ``max_solutions`` to the seed -- the
        trajectory-tracking idiom.
    :param seed_metric: distance used to rank against ``q_seed``.
        ``"wrap_linf"`` (default) minimises the *largest* single-joint
        wrap-to-pi move, which holds the branch during tracking;
        ``"wrap_l2"`` minimises the summed move (can favour a flip
        "paid for" by smaller moves elsewhere). Ignored when
        ``q_seed`` is ``None``.
    :param seed_tolerance: optional max per-joint deviation from
        ``q_seed`` (radians, wrap-to-pi). When set, only solutions with
        *every* joint within ``seed_tolerance`` are returned -- a hard
        tracking guarantee that may return an empty list when no branch
        qualifies. ``None`` (default) keeps the best-effort behaviour.
        Requires ``q_seed``.
    :param respect_limits: when ``True`` (default), solutions
        outside URDF joint limits are dropped. Pass ``False`` for
        the raw geometric set (e.g. analysis / debugging).
    :param allow_refinement: opt into Newton polish for near-miss
        algebraic candidates that don't quite meet ``fk_atol``.
        Default ``False`` -- the algebraic path is already at
        machine precision on tier-0 / SRS arms. On tier-2 RR
        arms (JACO 2, Rizon 4, Kassow), polish can recover
        edge-case candidates whose algebraic FK drifts above
        ``fk_atol``, at ~100-300 us per polished branch.
    :param allow_rescue: when ``True`` (default), if the analytical
        path returns no solutions for a target within the arm's
        reach (a measure-zero rank-deficient RR ridge -- a
        reachable pose the algebraic path can't extract),
        ``solve()`` recovers the IK via the T-perturbation rescue
        (#319), returning machine-precision solutions tagged
        ``refinement_used="lm"``. Set ``False`` for a guaranteed
        purely-analytical result (returns ``[]`` at such ridges).
        Gated by a reach-sphere, so far-field unreachable targets
        stay cheap.
    :param policy: tolerance policy (FK closure + dedup tolerance).
        Rarely customised.
    :param refinement_max_iters: cap on Newton iterations per
        candidate when ``allow_refinement=True``.
    :param native: opt into the shipped native (C++) backend for this
        arm's solver family (~50x faster). Returns the same solution
        *set*; the *order* without a seed and the near-singular
        *representative* may differ (numpy vs Eigen). Silently falls back
        to the Python path when the native extension isn't available
        (Windows / source installs). Default ``False``.
    :returns: list of :class:`Solution`; empty list iff no IK
        closed within ``policy.subproblem_numerical`` (or all
        IKs were filtered by ``respect_limits=True``).
    """
    if seed_tolerance is not None and q_seed is None:
        raise ValueError("seed_tolerance requires q_seed")
    if native:
        _native_sols = _try_native_solve(
            SOLVER_NAME,
            _KB,
            T_target,
            respect_limits=respect_limits,
            q_seed=q_seed,
            seed_metric=seed_metric,
            seed_tolerance=seed_tolerance,
            max_solutions=max_solutions,
            allow_rescue=allow_rescue,
            refinement_max_iters=refinement_max_iters,
        )
        if _native_sols is not None:
            return _native_sols
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
        # Refine only NEAR-misses. An algebraic candidate whose FK is
        # already >0.1 off (Frobenius) is an eigensolve root that does not
        # correspond to a real IK solution here (genuine near-double-root
        # marginals sit at ~1e-4); polishing it just burns Newton iters
        # that stall out and add nothing (#490: force_refine was 5-10x on
        # degenerate arms doing exactly this). 0.1 is 100x looser than the
        # 1e-3 band that once dropped genuine piper solutions -- it only
        # skips candidates clearly not near any solution.
        if residual >= 0.1:
            continue
        # Newton polish using the per-arm spatial Jacobian.
        refined = _lm_refine(
            q,
            _fk,
            T,
            fk_atol=fk_atol,
            max_iters=refinement_max_iters,
            jacobian_fn=_spatial_jacobian,
        )
        if refined is None:
            continue
        q_ref, resid_ref, iters = refined
        verified.append((q_ref, resid_ref, "lm", iters))

    # Wrap-to-pi dedup; keep lowest fk_residual on collision.
    # Inner check via ``_q_close_wrap`` -- typed scalar loop, no per-
    # iteration numpy allocation (#137 Slice 3).
    deduped: list[tuple[np.ndarray, float, str, int]] = []
    for cand_q, cand_res, ref_used, ref_iters in verified:
        dup_idx = None
        for j, (existing_q, _, _, _) in enumerate(deduped):
            if _q_close_wrap(cand_q, existing_q, dedup_atol):
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
        )
        for q, residual, ref_used, _ref_iters in deduped
    ]

    # Bulletproof fallback (#319): the analytical path found nothing.
    # If the target is within the arm's max reach it may be a
    # measure-zero rank-deficient ridge (a reachable pose the algebraic
    # path can't extract) rather than an unreachable target -- recover
    # via the T-perturbation rescue. The reach-sphere (sum of link
    # lengths; an exact upper bound by the triangle inequality, so it
    # never rejects a reachable pose) is the gate: it is checked only
    # here in the rare empty branch and keeps genuinely far-field
    # targets cheap. (The RR real-root count is NOT used as a gate -- it
    # is an unreliable reachability signal: some reachable ridges, e.g.
    # Rizon 4's, yield only complex roots, so gating on it would
    # silently drop real solutions.) The perturbed re-solves run with
    # allow_rescue=False (recursion guard + analytical-only escape
    # hatch). Rescued sols carry refinement_used="lm", FK-gated to
    # machine precision.
    # Shared post-processing pipeline (limits -> seed -> truncate); the
    # one definition lives in ssik.postprocess.finalize_solutions.
    # Limit pass only (no seed/tolerance/truncate yet): the rescue gate is
    # "no in-limits solution exists", so it must not depend on the
    # seed-tolerance / max_solutions filters (a seed filter emptying a
    # non-empty in-limits set is a user preference, not a missing solution).
    _in_limits = _ps_finalize(solutions, _KB, respect_limits=respect_limits)
    # Rescue on LIMIT-empty (#524): fire when nothing survives the limit
    # filter (not just when the analytical count was zero), so a pose
    # whose only analytical candidates were out-of-limits still gets
    # rescued. Perturbed re-solves run with allow_rescue=False.
    if not _in_limits and allow_rescue:
        _reach_radius = sum(
            float(np.linalg.norm(np.asarray(_t)[:3, 3]))
            for _t in (*_JOINT_T_LEFTS, *_JOINT_T_RIGHTS)
        )
        if float(np.linalg.norm(T[:3, 3])) <= _reach_radius:
            _in_limits = _ps_finalize(
                _rescue_via_T_perturbation(
                    _fk,
                    _functools.partial(solve, allow_rescue=False),
                    T,
                    jacobian_fn=_spatial_jacobian,
                ),
                _KB,
                respect_limits=respect_limits,
            )
    # Seed tolerance / ranking / truncate over the in-limits set.
    return _ps_finalize(
        _in_limits,
        _KB,
        respect_limits=False,
        q_seed=q_seed,
        seed_metric=seed_metric,
        seed_tolerance=seed_tolerance,
        max_solutions=max_solutions,
    )

fk = _fk

__all__ = [
    "BASE_LINK",
    "DISPATCH_REASON",
    "DOF",
    "EE_LINK",
    "EXPECTED_MS_MEDIAN",
    "FLOP_BUDGET",
    "SOLVER_NAME",
    "SOLVER_TIER",
    "T_HOME",
    "fk",
    "solve",
]
