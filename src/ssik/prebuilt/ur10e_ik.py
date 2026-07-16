"""Generated IK module for Universal Robots UR10e.

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash 288770fb29a5 (sha256/12 of the input chain).
``T_target`` is the pose of ``tool0`` (end-effector link) in
``base_link`` (base link). If your URDF differs (calibrated
geometry, custom tool past the flange, different link names),
run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
produce an artifact correct for your hardware.

DOF: 6    BASE_LINK: "base_link"    EE_LINK: "tool0"
Solver: ``ikgeo.three_parallel`` (tier 0)
Expected median IK time: ~1.6 ms on commodity
single-thread hardware. FLOP budget: 2,519 per solve.

Usage:

    import ur10e_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose of tool0 in base_link
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions = ur10e_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``list[Solution]``. Empty list iff no
candidate closed within the solver's FK tolerance -- check
``if not solutions:`` for the "unreachable" case.

Sanity-check the baked geometry: ``ur10e_ik.T_HOME`` is the
4x4 home pose (FK at ``q = np.zeros(DOF)``). If it doesn't match
your robot's home pose, the artifact is for a different URDF.
"""

from __future__ import annotations

import math
from ssik.subproblems import sp6 as _sp6_runtime

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
from ssik.postprocess import (
    nearest_to_seed as _ps_nearest_to_seed,
    respect_limits as _ps_respect_limits,
    within_seed_tolerance as _ps_within_seed_tolerance,
    wrap_to_limits as _ps_wrap_to_limits,
)
from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix

SOLVER_NAME = "ikgeo.three_parallel"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 1.6
FLOP_BUDGET = 2519
DISPATCH_REASON = 'Three consecutive parallel axes at joints (1, 2, 3) -- the UR-class structure (UR3 / UR5 / UR10).\nClosed-form via SP6 (joints 0+4) + SP1 + SP3.'
BASE_LINK = "base_link"
EE_LINK = "tool0"
DOF = 6
# Home pose: FK at q = np.zeros(DOF). Sanity-check this against
# your robot's documented home pose to verify the baked geometry
# matches your URDF.
T_HOME = np.array([[-1.0, 1.8369701992233881e-16, 6.123233998248561e-17, 1.18425], [6.123234002016247e-17, 2.0510330605065122e-10, 1.0, 0.29069999995083656], [1.8369701990977985e-16, 1.0, -2.0510330605065122e-10, 0.06084999994037644], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

# --- baked KinBody constants ---

_LINK_NAMES = ['base_link', '_poe_link_1', '_poe_link_2', '_poe_link_3', '_poe_link_4', '_poe_link_5', 'tool0']

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
    np.array([1.2246467991473532e-16, 1.0, -2.0510342851533115e-10], dtype=np.float64),
    np.array([1.2246467991473532e-16, 1.0, -2.0510342851533115e-10], dtype=np.float64),
    np.array([1.2246467991473532e-16, 1.0, -2.0510342851533115e-10], dtype=np.float64),
    np.array([-5.023585144508965e-26, -4.102068570306623e-10, -1.0], dtype=np.float64),
    np.array([2.5117947530122772e-26, 1.0, -2.0510330605065122e-10], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1807], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.6127], [0.0, 1.0, 0.0, -7.503410938375834e-17], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.57155], [0.0, 1.0, 0.0, 0.17414999999999992], [0.0, 0.0, 1.0, -3.571876128205531e-11], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -4.91632845545098e-11], [0.0, 0.0, 1.0, -0.11985], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.11655000000000001], [0.0, 0.0, 1.0, -2.3904801749186078e-11], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[-1.0, 1.8369701992233881e-16, 6.123233998248561e-17, 0.0], [6.123234002016247e-17, 2.0510330605065122e-10, 1.0, 0.0], [1.8369701990977985e-16, 1.0, -2.0510330605065122e-10, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
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
    (-6.283185307179586, 6.283185307179586),
    (-6.283185307179586, 6.283185307179586),
    (-3.141592653589793, 3.141592653589793),
    (-6.283185307179586, 6.283185307179586),
    (-6.283185307179586, 6.283185307179586),
    (-6.283185307179586, 6.283185307179586),
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
    p_16_x = p_x
    p_16_y = p_y
    p_16_z = p_z - 0.1807
    r_06_axes5_x = 6.12323399573676e-17*r_00 + 1.0*r_02
    r_06_axes5_y = 6.12323399573676e-17*r_10 + 1.0*r_12
    r_06_axes5_z = 6.12323399573676e-17*r_20 + 1.0*r_22
    _H_SP_0 = np.array([1.2246467991473532e-16, 1.0, -2.0510342851533115e-10])
    _K_SP_0 = np.array([-0.0, -0.0, -1.0])
    _K_SP_1 = np.array([-5.023585144508965e-26, -4.102068570306623e-10, -1.0])
    _NEG_P_5 = np.array([-0.0, -0.11655000000000001, 2.3904801749186078e-11])
    _NEG_AXES_5 = np.array([-2.5117947530122772e-26, -1.0, 2.0510330605065122e-10])
    # Build SP6 input arrays. h_sp / k_sp constant per arm; p_sp[0],
    # p_sp[2] depend on T_target via the inlined components above.
    p_16 = np.array([p_16_x, p_16_y, p_16_z])
    r_06_axes5 = np.array([r_06_axes5_x, r_06_axes5_y, r_06_axes5_z])
    h_sp = (_H_SP_0, _H_SP_0, _H_SP_0, _H_SP_0)
    k_sp = (_K_SP_0, _K_SP_1, _K_SP_0, _K_SP_1)
    p_sp = (p_16, _NEG_P_5, r_06_axes5, _NEG_AXES_5)
    theta15_solutions, _ = _sp6_runtime.solve(h_sp, k_sp, p_sp, 0.17414999997541833, 0.0)

    for q1, q5 in theta15_solutions:
        s1 = math.sin(q1)
        c1 = math.cos(q1)
        s5 = math.sin(q5)
        c5 = math.cos(q5)
        # SP1 for theta14 = q1+q2+q3+q4 (sum of parallel-axis rotations).
        th14_x0 = 6.12323399573676e-17*r_20 + 1.0*r_22
        th14_x1 = math.sin(q5)
        th14_x2 = 1.0*th14_x1
        th14_x3 = math.cos(q5)
        th14_x4 = -1.0*r_00 + 1.83697019922339e-16*r_01 + 6.12323399824856e-17*r_02
        th14_x5 = math.sin(q1)
        th14_x6 = 2.51179475301228e-26*th14_x5
        th14_x7 = 6.12323400201625e-17*r_00 + 2.05103306050651e-10*r_01 + 1.0*r_02
        th14_x8 = 1.0*th14_x5
        th14_x9 = 1.8369701990978e-16*r_00 + 1.0*r_01 - 2.05103306050651e-10*r_02
        th14_x10 = math.cos(q1)
        th14_x11 = -1.0*r_10 + 1.83697019922339e-16*r_11 + 6.12323399824856e-17*r_12
        th14_x12 = 6.12323400201625e-17*r_10 + 2.05103306050651e-10*r_11 + 1.0*r_12
        th14_x13 = 1.8369701990978e-16*r_10 + 1.0*r_11 - 2.05103306050651e-10*r_12
        th14_x14 = 2.05103306050651e-10*th14_x13
        th14_x15 = -2.51179475301228e-26*th14_x10*th14_x11 - 1.0*th14_x10*th14_x12 + th14_x10*th14_x14 + th14_x4*th14_x6 - 2.05103306050651e-10*th14_x5*th14_x9 + th14_x7*th14_x8
        th14_x16 = -th14_x15
        th14_x17 = 2.51179475301228e-26*th14_x10*th14_x4 + 1.0*th14_x10*th14_x7 - 2.05103306050651e-10*th14_x10*th14_x9 + th14_x11*th14_x6 + th14_x12*th14_x8 - th14_x14*th14_x5
        th14_x18 = 1.0*th14_x3
        theta14 = math.atan2(th14_x0*(-th14_x2 + 1.22464679889617e-16*th14_x3 - 1.3363823550461e-51) + th14_x16*(-2.05103428515331e-10*th14_x1 + 5.02358514399379e-26*th14_x3 - 2.51179407201427e-26) + th14_x17*(-5.02358514399379e-26*th14_x1 - 2.05103428515331e-10*th14_x3 + 2.05103550980011e-10), th14_x0*(-5.02358514347861e-26*th14_x1 - 4.10206857030662e-10*th14_x3 + 2.05103550980011e-10) + th14_x16*(-2.51179475404263e-26*th14_x1 + th14_x18 + 8.41348830133386e-20) + th14_x17*(th14_x2 + 2.51179475198192e-26*th14_x3 + 1.03035515178922e-35) - (1.22464679889617e-16*th14_x1 + th14_x18 + 4.20674415066693e-20)*(-1.25589628612724e-26*r_20 - 2.05103428515331e-10*r_22 + 3.0760614043916e-42*th14_x10*th14_x4 + 1.22464679914735e-16*th14_x10*th14_x7 - 2.5117910724947e-26*th14_x10*th14_x9 + 3.0760614043916e-42*th14_x11*th14_x5 + 1.22464679914735e-16*th14_x12*th14_x5 - 2.5117910724947e-26*th14_x13*th14_x5 - th14_x15))
        # SP1 for q6 (wrist roll-2): closed-form atan2.
        q6_x0 = math.sin(q5)
        q6_x1 = -1.0*q6_x0
        q6_x2 = math.cos(q5)
        q6_x3 = 1.8369701990978e-16*r_00 + 1.0*r_01 - 2.05103306050651e-10*r_02
        q6_x4 = math.sin(q1)
        q6_x5 = 1.0*q6_x4
        q6_x6 = math.cos(q1)
        q6_x7 = 1.22464679914735e-16*q6_x6
        q6_x8 = 1.8369701990978e-16*r_10 + 1.0*r_11 - 2.05103306050651e-10*r_12
        q6_x9 = 1.22464679914735e-16*q6_x4
        q6_x10 = 1.0*q6_x6
        q6_x11 = q6_x10*q6_x8 - q6_x3*q6_x5 + q6_x3*q6_x7 + q6_x8*q6_x9 - 3.76768885915449e-26*r_20 - 2.05103428515331e-10*r_21 + 4.20673912708178e-20*r_22
        q6_x12 = 6.12323400201625e-17*r_00 + 2.05103306050651e-10*r_01 + 1.0*r_02
        q6_x13 = 6.12323400201625e-17*r_10 + 2.05103306050651e-10*r_11 + 1.0*r_12
        q6_x14 = -q6_x10*q6_x13 + q6_x12*q6_x5 - q6_x12*q6_x7 - q6_x13*q6_x9 + 2.05103428515331e-10*r_22
        q6_x15 = -q6_x14 - 1.25589628741518e-26*r_20 - 4.20673912708178e-20*r_21
        q6_x16 = -1.0*r_00 + 1.83697019922339e-16*r_01 + 6.12323399824856e-17*r_02
        q6_x17 = -1.0*r_10 + 1.83697019922339e-16*r_11 + 6.12323399824856e-17*r_12
        q6_x18 = q6_x10*q6_x17 - q6_x16*q6_x5 + q6_x16*q6_x7 + q6_x17*q6_x9 + 2.05103428515331e-10*r_20 - 3.76768885941208e-26*r_21 - 1.25589628664242e-26*r_22
        q6_x19 = 1.0*q6_x2
        q6 = math.atan2(q6_x11*(q6_x1 + 1.22464679889617e-16*q6_x2 + 1.03035453636614e-35) + q6_x15*(-2.05103306050651e-10*q6_x0 + 2.51179107146434e-26*q6_x2 + 5.15177715780967e-36) + q6_x18*(-2.5117910724947e-26*q6_x0 + 2.05103550980011e-10*q6_x2 - 2.05103428515331e-10), q6_x11*(2.05103428515331e-10 - 4.10206857030662e-10*q6_x2) + q6_x15*(1.22464679914735e-16*q6_x0 + q6_x19 + 8.41348327774871e-20) + q6_x18*(q6_x1 + 1.22464679914735e-16*q6_x2 + 1.03035453657747e-35) - (-1.22464679889617e-16*q6_x0 - q6_x19 - 4.20674415066693e-20)*(q6_x14 + 2.51179475301228e-26*q6_x16*q6_x4 - 3.0760614043916e-42*q6_x16*q6_x6 - 3.0760614043916e-42*q6_x17*q6_x4 - 2.51179475301228e-26*q6_x17*q6_x6 - 2.05103306050651e-10*q6_x3*q6_x4 + 2.5117910724947e-26*q6_x3*q6_x6 + 2.5117910724947e-26*q6_x4*q6_x8 + 2.05103306050651e-10*q6_x6*q6_x8 + 1.25589628612724e-26*r_20))
        s14 = math.sin(theta14)
        c14 = math.cos(theta14)
        # d_inner = r_01.T @ p_16 - p[1] - r_14 @ r_45 @ p[5] - r_14 @ p[4]
        dinr_x0 = math.sin(theta14)
        dinr_x1 = math.cos(theta14)
        dinr_x2 = math.cos(q1)
        dinr_x3 = 1.0*math.sin(q1)
        dinr_x4 = 2.05103428515331e-10*dinr_x0
        dinr_x5 = 1.22464679914735e-16*dinr_x1
        dinr_x6 = dinr_x4 - dinr_x5 + 1.22464679914735e-16
        dinr_x7 = math.cos(q5)
        dinr_x8 = 0.11655*dinr_x7 + 1.96118295204322e-20
        dinr_x9 = 1.0*dinr_x1
        dinr_x10 = dinr_x9 + 1.49975978266186e-32
        dinr_x11 = math.sin(q5)
        dinr_x12 = -4.10206857030662e-10*dinr_x11 - 5.02358514450897e-26*dinr_x7 + 5.02358514450897e-26
        dinr_x13 = 2.39048017491861e-11*dinr_x12
        dinr_x14 = 0.11655*dinr_x11 - 2.40175642476209e-36*dinr_x7 + 2.40175642476209e-36
        dinr_x15 = 1.0*dinr_x0
        dinr_x16 = 5.02358514450897e-26*dinr_x11
        dinr_x17 = 4.10206857030662e-10*dinr_x7
        dinr_x18 = -0.11655*dinr_x16 - 0.11655*dinr_x17 + 4.78096091869237e-11
        dinr_x19 = 2.39048017491861e-11*dinr_x16 - 2.39048017491861e-11*dinr_x17 + 9.8059135934747e-21
        dinr_x20 = -dinr_x4 - dinr_x5 + 1.22464679914735e-16
        dinr_x21 = 1.22464679914735e-16*dinr_x0
        dinr_x22 = 2.05103428515331e-10*dinr_x1 + dinr_x21 - 2.05103428515331e-10
        dinr_x23 = 2.51179257225448e-26*dinr_x1 - dinr_x15 - 2.51179257225448e-26
        d_inner_x = 0.119850000023905*dinr_x0 - 3.01038250807766e-27*dinr_x1 + dinr_x10*dinr_x13 - dinr_x10*dinr_x14 - dinr_x18*(2.51179257225448e-26*dinr_x1 + dinr_x15 - 2.51179257225448e-26) + dinr_x19*dinr_x6 + dinr_x2*p_x + dinr_x3*p_y - dinr_x6*dinr_x8 + 3.01038250807766e-27
        d_inner_y = -1.46773918907085e-17*dinr_x0 + 2.45816459124654e-11*dinr_x1 + 1.20087806949643e-36*dinr_x11 + 2.39048017491861e-11*dinr_x12*dinr_x20 - dinr_x14*dinr_x20 - dinr_x18*(2.05103428515331e-10*dinr_x1 - dinr_x21 - 2.05103428515331e-10) + dinr_x2*p_y - dinr_x3*p_x - 0.11655*dinr_x7 + 2.45816386322385e-11
        d_inner_z = 6.02076590652509e-27*dinr_x0 + 0.119850000023905*dinr_x1 + dinr_x13*dinr_x23 - dinr_x14*dinr_x23 - dinr_x18*(dinr_x9 + 4.20674163887436e-20) + dinr_x19*dinr_x22 - dinr_x22*dinr_x8 + 1.0*p_z - 0.1807
        d_elbow = math.sqrt(d_inner_x*d_inner_x + d_inner_y*d_inner_y + d_inner_z*d_inner_z)
        # SP3 for q3 (elbow): reduces to SP4 with d_elbow target.
        _q3_R_sq = 0.122632115102029
        _q3_rhs = 0.3661994575 - 1/2*d_elbow**2
        _q3_phi = 1.38901164514678e-18 - math.pi
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
            q2_x1 = 0.57155*q2_x0
            q2_x2 = math.cos(q3)
            q2_x3 = 0.57155*q2_x2 + 0.6127
            q2 = math.atan2(d_inner_x*(-q2_x1 + 7.93889589994533e-19*q2_x2 - 1.29246970711411e-26) + d_inner_y*(6.99946878051041e-17*q2_x0 - 1.17226864567938e-10*q2_x2 - 1.25666870651343e-10) + d_inner_z*(-7.93889605783639e-19*q2_x0 - q2_x3), d_inner_x*(7.93889591427489e-19*q2_x0 + q2_x3) + d_inner_y*(-1.17226864567938e-10*q2_x0 - 7.00020138456682e-17*q2_x2 + 0.17415) + d_inner_z*(-q2_x1 + 7.93889604352186e-19*q2_x2 - 3.57187620759449e-11) - (0.17415 - 7.32604056410151e-21*q2_x2)*(1.22464679914735e-16*d_inner_x + 1.0*d_inner_y - 2.05103428515331e-10*d_inner_z))
            q4 = ((theta14 - q2 - q3 + math.pi) % (2.0 * math.pi)) - math.pi
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
        J[:3, i] = np.cross(p_i, z_i)
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
    :returns: list of :class:`Solution`; empty list iff no IK
        closed within ``policy.subproblem_numerical`` (or all
        IKs were filtered by ``respect_limits=True``).
    """
    if seed_tolerance is not None and q_seed is None:
        raise ValueError("seed_tolerance requires q_seed")
    T = np.asarray(T_target, dtype=np.float64)
    candidates = _solve_algebraic(T)

    fk_atol = 1e-7
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
        if not (allow_refinement or True):
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
    if not solutions and allow_rescue:
        _reach_radius = sum(
            float(np.linalg.norm(np.asarray(_t)[:3, 3]))
            for _t in (*_JOINT_T_LEFTS, *_JOINT_T_RIGHTS)
        )
        if float(np.linalg.norm(T[:3, 3])) <= _reach_radius:
            solutions = _rescue_via_T_perturbation(
                _fk,
                _functools.partial(solve, allow_rescue=False),
                T,
                jacobian_fn=_spatial_jacobian,
            )

    # Post-processing pass (#238 item 4). Order matters:
    #   1. wrap_to_limits tries q +/- 2*pi per joint to bring
    #      candidates into the URDF's limit range
    #   2. respect_limits drops anything still outside
    #   3. nearest_to_seed sorts by distance to q_seed (if given)
    #   4. max_solutions truncates to the first k
    if respect_limits:
        solutions = _ps_wrap_to_limits(solutions, _KB)
        solutions = _ps_respect_limits(solutions, _KB)
    if q_seed is not None:
        if seed_tolerance is not None:
            solutions = _ps_within_seed_tolerance(solutions, q_seed, seed_tolerance)
        solutions = _ps_nearest_to_seed(solutions, q_seed, metric=seed_metric)
    if max_solutions is not None and len(solutions) > max_solutions:
        solutions = solutions[:max_solutions]
    return solutions

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
