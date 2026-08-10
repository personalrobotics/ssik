"""Generated IK module for Rokae xMate CR7.

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash dd97ecace834 (sha256/12 of the input chain).
``T_target`` is the pose of ``xMateCR7_link6`` (end-effector link) in
``xMateCR7_base`` (base link). If your URDF differs (calibrated
geometry, custom tool past the flange, different link names),
run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
produce an artifact correct for your hardware.

DOF: 6    BASE_LINK: "xMateCR7_base"    EE_LINK: "xMateCR7_link6"
Solver: ``ikgeo.general_6r`` (tier 2)
Expected median IK time: ~5.0 ms on commodity
single-thread hardware. FLOP budget: 30,000,000 per solve.

Usage:

    import xmatecr7_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose of xMateCR7_link6 in xMateCR7_base
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions = xmatecr7_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``list[Solution]``. Empty list iff no
candidate closed within the solver's FK tolerance -- check
``if not solutions:`` for the "unreachable" case.

Sanity-check the baked geometry: ``xmatecr7_ik.T_HOME`` is the
4x4 home pose (FK at ``q = np.zeros(DOF)``). If it doesn't match
your robot's home pose, the artifact is for a different URDF.
"""

from __future__ import annotations

import math
from ssik.solvers.ikgeo._raghavan_roth import (
    eliminate_q0_q1 as _ssik_eliminate_q0_q1,
    weierstrass_eliminate_trig as _ssik_weierstrass,
    build_m_matrix as _ssik_build_m_matrix,
    solve_x2_roots_mobius as _ssik_solve_x2_roots_mobius,
    _back_substitute_inner as _ssik_back_substitute_inner,
    _fk_dh as _ssik_fk_dh,
)

import cython
import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.refinement import lm_refine as _lm_refine
import functools as _functools
from ssik.refinement.rescue import rescue_via_T_perturbation as _rescue_via_T_perturbation
from ssik.postprocess import finalize_solutions as _ps_finalize
from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix

SOLVER_NAME = "ikgeo.general_6r"
SOLVER_TIER = 2
EXPECTED_MS_MEDIAN = 5.0
FLOP_BUDGET = 30000000
DISPATCH_REASON = 'No tier-0 (Pieper-class) match.\nTier-2 numeric Raghavan-Roth + Manocha-Canny pipeline with AE-3 leftvar selection. Closes the EAIK coverage gap (Kinova JACO 2 classical, Agilex Piper, custom non-Pieper 6R).\nWeaker structural matches (not used):\n  - axes[1] parallel to axes[2] (would match tier-1 `two_parallel`, but tier-2 RR is ~50x faster)'
BASE_LINK = "xMateCR7_base"
EE_LINK = "xMateCR7_link6"
DOF = 6
# Home pose: FK at q = np.zeros(DOF). Sanity-check this against
# your robot's documented home pose to verify the baked geometry
# matches your URDF.
T_HOME = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.151], [0.0, 0.0, 1.0, 1.2725], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

# --- baked KinBody constants ---

_LINK_NAMES = ['xMateCR7_base', '_poe_link_1', '_poe_link_2', '_poe_link_3', '_poe_link_4', '_poe_link_5', 'xMateCR7_link6']

_JOINT_NAMES = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
]

_JOINT_AXES = [
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.296], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.49000000000000005], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.3599999999999999], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.151], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.12650000000000006], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
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

_JOINT_LIMITS = [
    (-3.0543, 3.0543),
    (-3.0543, 3.0543),
    (-3.0543, 3.0543),
    (-3.0543, 3.0543),
    (-3.0543, 3.0543),
    (-3.0543, 3.0543),
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


# --- baked DH parameters (from poe_to_dh at build time) ---
_DH_ALPHA = np.array([1.5707963267948966, 3.141592653589793, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 0.0], dtype=np.float64)
_DH_A = np.array([0.0, 0.49000000000000005, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
_DH_D = np.array([0.296, 0.0, 0.0, 0.3599999999999999, 0.151, 0.12650000000000006], dtype=np.float64)
_DH_TUPLE = (_DH_ALPHA, _DH_A, _DH_D)
_DH_THETA_OFFSET = np.array([3.141592653589793, 1.5707963267948966, 1.5707963267948966, 3.141592653589793, 3.141592653589793, 0.0], dtype=np.float64)
_T_PRE = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
_T_POST = np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
_T_PRE_INV = np.linalg.inv(_T_PRE)
_T_POST_INV = np.linalg.inv(_T_POST)

_LINEARITY_JOINT = 1
_LEFT_BILINEAR = (2, 3)
_RIGHT_BILINEAR = (0, 5)
_DROP_JOINT = 4
_RR_META = {
    "linearity_joint": _LINEARITY_JOINT,
    "left_bilinear": _LEFT_BILINEAR,
    "right_bilinear": _RIGHT_BILINEAR,
    "drop_joint": _DROP_JOINT,
    "apply_so3": False,
}


# --- inlined per-arm matrix builders (sympy.cse'd) ---
# Per-arm DH constants combine with T_target here. JACO 2's
# 0.866 (cos 60-deg) and friends become explicit coefficients.
# Generic linear algebra (Q-rank, Weierstrass, eigvals,
# back-substitution) stays imported from ssik.solvers.ikgeo.

def _build_pq_matrices(T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, T_11):
    """Build (P_sin, P_cos, P_one, Q) for this T_target. CSE-shared."""
    _pq_x0 = 6.12323399573677e-17*T_10
    _pq_x1 = 3.74939945665464e-33*T_10
    _pq_x2 = 0.1265*T_10
    _pq_x3 = 1.0*T_11
    _pq_x4 = 6.12323399573677e-17*T_11
    _pq_x5 = 6.12323399573677e-17*T_2
    _pq_x6 = 3.74939945665464e-33*T_2
    _pq_x7 = 3.74939945665464e-33*T_6
    _pq_x8 = 0.074888*T_10
    _pq_x9 = T_0**2
    _pq_x10 = 0.022801*_pq_x9
    _pq_x11 = T_10**2
    _pq_x12 = T_11**2
    _pq_x13 = 1.0*_pq_x12
    _pq_x14 = T_2**2
    _pq_x15 = T_3**2
    _pq_x16 = 1.0*_pq_x15
    _pq_x17 = T_4**2
    _pq_x18 = 0.022801*_pq_x17
    _pq_x19 = T_6**2
    _pq_x20 = T_7**2
    _pq_x21 = 1.0*_pq_x20
    _pq_x22 = T_8**2
    _pq_x23 = 0.022801*_pq_x22
    _pq_x24 = 6.12323399573677e-17*T_6
    _pq_x25 = 7.70371977754894e-34*T_0
    _pq_x26 = T_10*T_11
    _pq_x27 = T_2*T_3
    _pq_x28 = T_6*T_7
    _pq_x29 = T_10**3
    _pq_x30 = 1.22464679914735e-16*T_11
    _pq_x31 = 0.038203*T_10
    _pq_x32 = 0.302*T_11
    _pq_x33 = 9.79855212082788e-19*T_10
    _pq_x34 = 1.5491782009214e-17*T_11
    _pq_x35 = 3.85185988877447e-34*T_8
    _pq_x36 = T_0*T_2
    _pq_x37 = 7.49879891330929e-33*T_11
    _pq_x38 = T_4*T_6
    _pq_x39 = 2.33925908339132e-18*T_10
    _pq_x40 = 1.8492166667125e-17*T_11
    _pq_x41 = 5.99988274552518e-35*T_10
    _pq_x42 = 9.48598062533625e-34*T_11
    _pq_x43 = 1.0*T_4
    _pq_x44 = 1.0*T_5
    _pq_x45 = 1.0*T_0
    _pq_x46 = 1.0*T_1
    _pq_x47 = 6.12323399573677e-17*T_0
    _pq_x48 = 6.12323399573677e-17*T_1
    _pq_x49 = 6.12323399573677e-17*T_4
    _pq_x50 = 6.12323399573677e-17*T_5
    _pq_x51 = 1.0*T_8
    _pq_x52 = 1.0*T_9
    _pq_x53 = 6.12323399573677e-17*T_8
    _pq_x54 = 6.12323399573677e-17*T_9
    _pq_x55 = 0.151*T_4
    _pq_x56 = 0.151*T_5
    _pq_x57 = -0.151*T_0
    _pq_x58 = -0.151*T_1
    _pq_x59 = 0.1265*T_6
    _pq_x60 = -1.0*T_7 + _pq_x59
    _pq_x61 = 0.1265*T_2
    _pq_x62 = 1.0*T_3 - _pq_x61
    _pq_x63 = 7.74589100460701e-18*T_2
    _pq_x64 = 7.74589100460701e-18*T_6
    _pq_x65 = T_0*_pq_x2 - T_0*_pq_x3 + 0.296*T_0 + T_3*_pq_x51 - T_8*_pq_x61
    _pq_x66 = T_1*_pq_x2 - T_1*_pq_x3 + 0.296*T_1 + T_3*_pq_x52 - T_9*_pq_x61
    _pq_x67 = 1.0*T_11*T_4 - T_4*_pq_x2 - 0.296*T_4 + 0.1265*T_6*T_8 - T_7*_pq_x51
    _pq_x68 = 1.0*T_11*T_5 - T_5*_pq_x2 - 0.296*T_5 + 0.1265*T_6*T_9 - T_7*_pq_x52
    _pq_x69 = -T_2*_pq_x4 + 1.81247726273808e-17*T_2 + T_3*_pq_x0
    _pq_x70 = 6.12323399573677e-17*T_11*T_6 - 1.81247726273808e-17*T_6 - T_7*_pq_x0
    _pq_x71 = 7.74589100460701e-18*T_10
    _pq_x72 = 0.038203*T_2
    _pq_x73 = 0.302*T_3
    _pq_x74 = 0.038203*T_6
    _pq_x75 = 0.302*T_7
    _pq_x76 = 0.592*T_11
    _pq_x77 = 0.074888*T_6
    _pq_x78 = T_4**3
    _pq_x79 = 0.0320045*T_6
    _pq_x80 = T_0*T_3
    _pq_x81 = 2.0*T_7
    _pq_x82 = 0.253*_pq_x26
    _pq_x83 = T_10*T_8
    _pq_x84 = T_11*T_8
    _pq_x85 = 0.253*_pq_x27
    _pq_x86 = -0.253*T_0*T_2*T_7 - 0.253*T_0*T_3*T_6 - 0.074888*T_10*T_4 - 0.253*T_10*T_7*T_8 - 0.253*T_11*T_6*T_8 - 0.253*T_4*T_6*T_7 + T_4*_pq_x10 - 0.01600225*T_4*_pq_x11 - 1.0*T_4*_pq_x12 - 0.01600225*T_4*_pq_x14 - 1.0*T_4*_pq_x15 + 0.01600225*T_4*_pq_x19 + T_4*_pq_x21 + T_4*_pq_x23 + T_4*_pq_x76 + T_4*_pq_x82 + T_4*_pq_x85 - 0.087616*T_4 - 0.592*T_7*T_8 + T_8*_pq_x77 + _pq_x36*_pq_x79 + 0.022801*_pq_x78 + _pq_x79*_pq_x83 + _pq_x80*_pq_x81 + _pq_x81*_pq_x84
    _pq_x87 = 0.045602*T_4
    _pq_x88 = T_0*T_1
    _pq_x89 = T_1*T_2
    _pq_x90 = T_1*T_3
    _pq_x91 = T_10*T_9
    _pq_x92 = T_11*T_9
    _pq_x93 = T_8*T_9
    _pq_x94 = T_5*_pq_x17
    _pq_x95 = -0.253*T_1*T_2*T_7 - 0.253*T_1*T_3*T_6 - 0.074888*T_10*T_5 - 0.253*T_10*T_7*T_9 - 0.253*T_11*T_6*T_9 - 0.253*T_5*T_6*T_7 + T_5*_pq_x10 - 0.01600225*T_5*_pq_x11 - 1.0*T_5*_pq_x12 - 0.01600225*T_5*_pq_x14 - 1.0*T_5*_pq_x15 + 0.01600225*T_5*_pq_x19 + T_5*_pq_x21 + T_5*_pq_x23 + T_5*_pq_x76 + T_5*_pq_x82 + T_5*_pq_x85 - 0.087616*T_5 - 0.592*T_7*T_9 + T_9*_pq_x77 + _pq_x79*_pq_x89 + _pq_x79*_pq_x91 + _pq_x81*_pq_x90 + _pq_x81*_pq_x92 + _pq_x87*_pq_x88 + _pq_x87*_pq_x93 + 0.068403*_pq_x94
    _pq_x96 = 0.074888*T_2
    _pq_x97 = T_0**3
    _pq_x98 = 0.253*_pq_x28
    _pq_x99 = 0.0320045*T_2
    _pq_x100 = 2.0*T_3
    _pq_x101 = T_3*_pq_x81
    _pq_x102 = 0.074888*T_0*T_10 + 0.253*T_0*T_2*T_3 + 0.01600225*T_0*_pq_x11 + 1.0*T_0*_pq_x12 - 0.01600225*T_0*_pq_x14 - T_0*_pq_x16 - T_0*_pq_x18 + 0.01600225*T_0*_pq_x19 + 1.0*T_0*_pq_x20 - T_0*_pq_x23 - T_0*_pq_x76 - T_0*_pq_x82 - T_0*_pq_x98 + 0.087616*T_0 + 0.253*T_10*T_3*T_8 + 0.253*T_11*T_2*T_8 + 0.253*T_2*T_4*T_7 + 0.253*T_3*T_4*T_6 + 0.592*T_3*T_8 - T_4*_pq_x101 - T_8*_pq_x96 - _pq_x100*_pq_x84 - _pq_x38*_pq_x99 - _pq_x83*_pq_x99 - 0.022801*_pq_x97
    _pq_x103 = T_0*T_5
    _pq_x104 = T_0*_pq_x93
    _pq_x105 = T_2*T_5
    _pq_x106 = T_1*_pq_x9
    _pq_x107 = 0.074888*T_1*T_10 + 0.253*T_1*T_2*T_3 + 0.01600225*T_1*_pq_x11 + 1.0*T_1*_pq_x12 - 0.01600225*T_1*_pq_x14 - T_1*_pq_x16 - T_1*_pq_x18 + 0.01600225*T_1*_pq_x19 + 1.0*T_1*_pq_x20 - T_1*_pq_x23 - T_1*_pq_x76 - T_1*_pq_x82 - T_1*_pq_x98 + 0.087616*T_1 + 0.253*T_10*T_3*T_9 + 0.253*T_11*T_2*T_9 + 0.253*T_2*T_5*T_7 + 0.253*T_3*T_5*T_6 + 0.592*T_3*T_9 - T_5*_pq_x101 - T_9*_pq_x96 - _pq_x100*_pq_x92 - _pq_x103*_pq_x87 - 0.045602*_pq_x104 - _pq_x105*_pq_x79 - 0.068403*_pq_x106 - _pq_x91*_pq_x99
    _pq_x108 = 3.62495452547617e-17*T_11
    _pq_x109 = T_6**3
    _pq_x110 = 1.22464679914735e-16*T_7
    _pq_x111 = 9.79855212082788e-19*T_6
    _pq_x112 = -3.62495452547617e-17*T_10*T_7 + T_6*_pq_x108 - 6.12323399573677e-17*T_6*_pq_x12 - 6.12323399573677e-17*T_6*_pq_x15 - 5.36493269770472e-18*T_6 - 1.5491782009214e-17*T_7*_pq_x11 - 1.5491782009214e-17*T_7*_pq_x14 - 0.302*T_7*_pq_x17 - 1.5491782009214e-17*T_7*_pq_x19 - 0.302*T_7*_pq_x22 - 0.302*T_7*_pq_x9 + 9.79855212082788e-19*_pq_x109 + _pq_x11*_pq_x111 + _pq_x110*_pq_x26 + _pq_x110*_pq_x27 + _pq_x111*_pq_x14 + _pq_x17*_pq_x74 + _pq_x20*_pq_x24 + _pq_x22*_pq_x74 + _pq_x74*_pq_x9
    _pq_x113 = T_2**3
    _pq_x114 = 1.22464679914735e-16*T_3
    _pq_x115 = 9.79855212082788e-19*T_2
    _pq_x116 = 3.62495452547617e-17*T_10*T_3 - T_2*_pq_x108 + 6.12323399573677e-17*T_2*_pq_x12 + 6.12323399573677e-17*T_2*_pq_x20 + 5.36493269770472e-18*T_2 + 1.5491782009214e-17*T_3*_pq_x11 + 1.5491782009214e-17*T_3*_pq_x14 + 0.302*T_3*_pq_x17 + 1.5491782009214e-17*T_3*_pq_x19 + 0.302*T_3*_pq_x22 + 0.302*T_3*_pq_x9 - _pq_x11*_pq_x115 - 9.79855212082788e-19*_pq_x113 - _pq_x114*_pq_x26 - _pq_x114*_pq_x28 - _pq_x115*_pq_x19 - _pq_x15*_pq_x5 - _pq_x17*_pq_x72 - _pq_x22*_pq_x72 - _pq_x72*_pq_x9
    _pq_x117 = 4.71716788355174e-50*T_6
    _pq_x118 = 4.58556747472735e-18*T_10
    _pq_x119 = 4.58556747472735e-18*T_2
    _pq_x120 = 3.62495452547617e-17*T_3
    _pq_x121 = 1.5491782009214e-17*T_0
    _pq_x122 = 1.95971042416558e-18*T_2
    _pq_x123 = 1.5491782009214e-17*T_3
    _pq_x124 = T_2*_pq_x34
    _pq_x125 = T_3*_pq_x30
    _pq_x126 = 1.5491782009214e-17*T_7
    _pq_x127 = 9.79855212082787e-19*T_0
    _pq_x128 = 9.79855212082788e-19*_pq_x14
    _pq_x129 = 1.39615858336794e-18*T_0
    _pq_x130 = 2.79231716673588e-18*T_4
    _pq_x131 = 1.5491782009214e-17*T_1
    _pq_x132 = 1.95971042416558e-18*T_6
    _pq_x133 = 9.79855212082787e-19*T_1
    _pq_x134 = 1.39615858336794e-18*T_1
    _pq_x135 = 4.58556747472735e-18*T_6
    _pq_x136 = 1.5491782009214e-17*T_4
    _pq_x137 = T_7*_pq_x30
    _pq_x138 = 1.39615858336794e-18*T_4
    _pq_x139 = 9.79855212082788e-19*_pq_x19
    _pq_x140 = 1.5491782009214e-17*T_5
    _pq_x141 = 1.39615858336794e-18*T_5
    _pq_x142 = 4.27642353614751e-50*T_10
    _pq_x143 = 2.21964447833955e-33*T_11
    _pq_x144 = T_10*_pq_x35
    _pq_x145 = 7.49879891330929e-33*T_3
    _pq_x146 = 1.8492166667125e-17*T_3
    _pq_x147 = 5.99988274552518e-35*T_2
    _pq_x148 = 9.48598062533625e-34*T_3
    _pq_x149 = 2.33925908339132e-18*T_2
    _pq_x150 = 7.49879891330929e-33*T_7
    _pq_x151 = 2.33925908339132e-18*T_6
    _pq_x152 = 5.99988274552518e-35*T_6
    _pq_x153 = T_8**3
    _pq_x154 = 0.0320045*T_10
    _pq_x155 = 2.0*T_11
    _pq_x156 = T_11*_pq_x81
    _pq_x157 = T_8*_pq_x88
    _pq_x158 = T_10*T_5
    _pq_x159 = T_5*T_8
    _pq_x160 = T_9*_pq_x22
    _pq_x161 = 1.95971042416558e-18*T_10
    _pq_x162 = 1.5491782009214e-17*T_8
    _pq_x163 = 1.39615858336794e-18*T_8
    _pq_x164 = 9.79855212082788e-19*_pq_x11
    _pq_x165 = 1.5491782009214e-17*T_9
    _pq_x166 = 1.39615858336794e-18*T_9

    p_sin = np.array([
        [1.00000000000000, 0, 0, -6.12323399573677e-17, 0, -6.12323399573677e-17, 0, -1.22464679914735e-16, 4.59169004331693e-49],
        [0, 6.12323399573677e-17, 1.00000000000000, 0, 6.12323399573677e-17, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.360000000000000, 0, 0, 2.69956760879134e-33],
        [0, 0, 0, 0, 0.360000000000000, 0, 0, 0, 0.490000000000000],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-6.00076931582203e-17, -0.360000000000000, -2.20436423846523e-17, 3.67441146752155e-33, 0, 3.67441146752155e-33, -4.40872847693047e-17, -0.490000000000000, 1.83720573376078e-33],
        [2.20436423846523e-17, 0, 0, -0.360000000000000, 0, 0, 0, 0, -2.10843958864610e-81],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.369700000000000, -4.32055390739186e-17, -2.64557625661552e-33, -2.26375960822388e-17, -1.89911354915196e-65, -6.76617356528913e-18, 0.352800000000000, -4.52751921644776e-17, 5.07381749786522e-50],
        [0, -6.76617356528913e-18, -0.110500000000000, 0, -2.26375960822388e-17, 3.79822709830392e-65, 0, 0, -2.16027695369593e-17],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)

    p_cos = np.array([
        [0, 6.12323399573677e-17, 1.00000000000000, 0, 6.12323399573677e-17, 0, 0, 0, 0],
        [-1.00000000000000, 0, 0, 6.12323399573677e-17, 0, 6.12323399573677e-17, 0, 1.22464679914735e-16, -4.59169004331693e-49],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.360000000000000, 0, 0, 0, 0.490000000000000],
        [0, 0, 0, 0, 0, 0.360000000000000, 0, 0, -2.69956760879134e-33],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.20436423846523e-17, 0, 0, -0.360000000000000, 0, 0, 0, 0, -2.10843958864610e-81],
        [6.00076931582203e-17, 0.360000000000000, 2.20436423846523e-17, -3.67441146752155e-33, 0, -3.67441146752155e-33, 4.40872847693047e-17, 0.490000000000000, -1.83720573376078e-33],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -6.76617356528913e-18, -0.110500000000000, 0, -2.26375960822388e-17, 3.79822709830392e-65, 0, 0, -2.16027695369593e-17],
        [-0.369700000000000, 4.32055390739186e-17, 2.64557625661552e-33, 2.26375960822388e-17, 1.89911354915196e-65, 6.76617356528913e-18, -0.352800000000000, 4.52751921644776e-17, -5.07381749786522e-50],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)

    p_one = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -_pq_x0],
        [1.22464679914735e-16, 0, 0, -7.49879891330929e-33, 0, -7.49879891330929e-33, 0, 1.00000000000000, -_pq_x1 - 3.74939945665464e-33],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, _pq_x2 - _pq_x3 + 0.296],
        [0, 0, 0, 0, 0, -4.40872847693047e-17, 0, 0, 7.74589100460701e-18*T_10 - _pq_x4 - 3.91886975727152e-18],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 6.12323399573677e-17*T_3*T_6 - T_7*_pq_x5],
        [0.490000000000000, -4.40872847693047e-17, -2.69956760879134e-33, -3.00038465791102e-17, 0, -3.00038465791102e-17, 0.360000000000000, -6.00076931582203e-17, T_3*_pq_x7 - T_7*_pq_x6 + 2.2499281212253e-49],
        [0, 0, 0, 0, 0.352800000000000, -3.42113882891801e-49, 0, 0, 0.253*T_10*T_11 + 0.592*T_11 + 0.253*T_2*T_3 + 0.253*T_6*T_7 - _pq_x10 - 0.01600225*_pq_x11 - _pq_x13 - 0.01600225*_pq_x14 - _pq_x16 - _pq_x18 - 0.01600225*_pq_x19 - _pq_x21 - _pq_x23 - _pq_x8 + 0.282084],
        [0, 3.00038465791102e-17, 0.490000000000000, 0, 3.00038465791102e-17, -3.79822709830392e-65, 0, 0, 1.81247726273808e-17*T_10 - T_11*_pq_x0 - T_3*_pq_x5 - T_7*_pq_x24 + 7.74589100460701e-18*_pq_x11 + 7.74589100460701e-18*_pq_x14 + 0.151*_pq_x17 + 7.74589100460701e-18*_pq_x19 + 0.151*_pq_x22 + 0.151*_pq_x9 + 2.20436423846523e-17],
        [0, 0, 0, 0, 0, 0, 0, 0, T_4*_pq_x25],
        [0, 0, 0, 0, 0, 0, 0, 0, 5.36493269770473e-18*T_10 + _pq_x0*_pq_x12 - _pq_x0*_pq_x15 - _pq_x0*_pq_x20 - _pq_x11*_pq_x34 + 4.58556747472735e-18*_pq_x11 + _pq_x14*_pq_x33 - _pq_x14*_pq_x34 + 4.58556747472735e-18*_pq_x14 + _pq_x17*_pq_x31 - _pq_x17*_pq_x32 + 0.089392*_pq_x17 + _pq_x19*_pq_x33 - _pq_x19*_pq_x34 + 4.58556747472735e-18*_pq_x19 + _pq_x22*_pq_x31 - _pq_x22*_pq_x32 + 0.089392*_pq_x22 - 3.62495452547617e-17*_pq_x26 + _pq_x27*_pq_x30 - 3.62495452547617e-17*_pq_x27 + _pq_x28*_pq_x30 - 3.62495452547617e-17*_pq_x28 + 9.79855212082788e-19*_pq_x29 + _pq_x31*_pq_x9 - _pq_x32*_pq_x9 + 0.089392*_pq_x9],
        [4.52751921644776e-17, 0.352800000000000, 2.16027695369593e-17, -2.77230595825044e-33, 0, -8.28617279920678e-34, 4.32055390739186e-17, 0.369700000000000, 2.33925908339132e-18*T_10*_pq_x22 + 3.28507382794253e-34*T_10 + _pq_x1*_pq_x12 - _pq_x1*_pq_x15 - _pq_x1*_pq_x20 - _pq_x11*_pq_x42 + 2.80785026509953e-34*_pq_x11 + _pq_x14*_pq_x41 - _pq_x14*_pq_x42 + 2.80785026509953e-34*_pq_x14 + _pq_x17*_pq_x39 - _pq_x17*_pq_x40 + 5.47368133346901e-18*_pq_x17 + _pq_x19*_pq_x41 - _pq_x19*_pq_x42 + 2.80785026509953e-34*_pq_x19 - _pq_x22*_pq_x40 + 5.47368133346901e-18*_pq_x22 - 2.21964447833955e-33*_pq_x26 + _pq_x27*_pq_x37 - 2.21964447833955e-33*_pq_x27 + _pq_x28*_pq_x37 - 2.21964447833955e-33*_pq_x28 + 5.99988274552518e-35*_pq_x29 + _pq_x35*_pq_x36 + _pq_x35*_pq_x38 + _pq_x39*_pq_x9 - _pq_x40*_pq_x9 + 5.47368133346901e-18*_pq_x9 - 4.14308639960339e-34],
    ], dtype=np.float64)

    q = np.array([
        [_pq_x43, _pq_x44, _pq_x45, _pq_x46, _pq_x24, _pq_x5, 0, 0],
        [-_pq_x47, -_pq_x48, _pq_x49, _pq_x50, -_pq_x6, _pq_x7, _pq_x51, _pq_x52],
        [_pq_x45, _pq_x46, -_pq_x43, -_pq_x44, _pq_x5, -_pq_x24, _pq_x53, _pq_x54],
        [-_pq_x55, -_pq_x56, _pq_x57, _pq_x58, -_pq_x60, _pq_x62, 0, 0],
        [9.24608333356252e-18*T_0, 9.24608333356252e-18*T_1, -9.24608333356252e-18*T_4, -9.24608333356252e-18*T_5, -6.12323399573677e-17*T_3 + _pq_x63, 6.12323399573677e-17*T_7 - _pq_x64, -0.151*T_8, -0.151*T_9],
        [_pq_x57, _pq_x58, _pq_x55, _pq_x56, _pq_x62, _pq_x60, -9.24608333356252e-18*T_8, -9.24608333356252e-18*T_9],
        [_pq_x65, _pq_x66, _pq_x67, _pq_x68, _pq_x69, _pq_x70, 0, 0],
        [-T_4*_pq_x4 + T_4*_pq_x71 + 1.81247726273808e-17*T_4 + T_7*_pq_x53 - T_8*_pq_x64, -T_5*_pq_x4 + T_5*_pq_x71 + 1.81247726273808e-17*T_5 + T_7*_pq_x54 - T_9*_pq_x64, -T_0*_pq_x4 + T_0*_pq_x71 + 1.81247726273808e-17*T_0 + T_3*_pq_x53 - T_8*_pq_x63, -T_1*_pq_x4 + T_1*_pq_x71 + 1.81247726273808e-17*T_1 + T_3*_pq_x54 - T_9*_pq_x63, -T_11*_pq_x7 + 1.10982223916977e-33*T_6 + T_7*_pq_x1, -T_11*_pq_x6 + 1.10982223916977e-33*T_2 + T_3*_pq_x1, 1.0*T_0*T_7 - T_0*_pq_x59 + 0.1265*T_2*T_4 - T_3*_pq_x43, 1.0*T_1*T_7 - T_1*_pq_x59 + 0.1265*T_2*T_5 - T_3*_pq_x44],
        [_pq_x67, _pq_x68, -_pq_x65, -_pq_x66, _pq_x70, -_pq_x69, 6.12323399573677e-17*T_0*T_7 - T_0*_pq_x64 + 7.74589100460701e-18*T_2*T_4 - T_3*_pq_x49, 6.12323399573677e-17*T_1*T_7 - T_1*_pq_x64 + 7.74589100460701e-18*T_2*T_5 - T_3*_pq_x50],
        [-_pq_x25, -7.70371977754894e-34*T_1, 7.70371977754894e-34*T_4, 7.70371977754894e-34*T_5, 0, 0, T_0*_pq_x72 - T_0*_pq_x73 + T_4*_pq_x74 - T_4*_pq_x75 + T_8*_pq_x31 - T_8*_pq_x32 + 0.089392*T_8, T_1*_pq_x72 - T_1*_pq_x73 + T_5*_pq_x74 - T_5*_pq_x75 + T_9*_pq_x31 - T_9*_pq_x32 + 0.089392*T_9],
        [0, 0, 0, 0, 0, 0, 1.0*T_0*T_3 - T_0*_pq_x61 + 1.0*T_11*T_8 + 1.0*T_4*T_7 - T_4*_pq_x59 - T_8*_pq_x2 - 0.296*T_8, 1.0*T_1*T_3 - T_1*_pq_x61 + 1.0*T_11*T_9 + 1.0*T_5*T_7 - T_5*_pq_x59 - T_9*_pq_x2 - 0.296*T_9],
        [-_pq_x86, -_pq_x95, _pq_x102, _pq_x107, -_pq_x112, _pq_x116, -T_0*_pq_x117, -T_1*_pq_x117],
        [T_0*_pq_x108 - T_0*_pq_x118 + T_0*_pq_x128 - 5.36493269770472e-18*T_0 - T_2*T_4*_pq_x126 + T_3*T_4*_pq_x110 + T_8*_pq_x119 - T_8*_pq_x120 - T_8*_pq_x124 + T_8*_pq_x125 - _pq_x11*_pq_x127 - _pq_x12*_pq_x47 + _pq_x121*_pq_x26 - _pq_x121*_pq_x27 + _pq_x121*_pq_x28 + _pq_x122*_pq_x38 + _pq_x122*_pq_x83 - _pq_x123*_pq_x38 - _pq_x123*_pq_x83 - _pq_x127*_pq_x19 + _pq_x129*_pq_x17 + _pq_x129*_pq_x22 + _pq_x15*_pq_x47 - _pq_x20*_pq_x47 + 1.39615858336794e-18*_pq_x97, T_1*_pq_x108 - T_1*_pq_x118 + T_1*_pq_x128 - 5.36493269770472e-18*T_1 + T_3*T_5*_pq_x110 - T_5*T_6*_pq_x123 + T_9*_pq_x119 - T_9*_pq_x120 - T_9*_pq_x124 + T_9*_pq_x125 + _pq_x103*_pq_x130 + 2.79231716673588e-18*_pq_x104 - _pq_x105*_pq_x126 + _pq_x105*_pq_x132 + 4.18847575010382e-18*_pq_x106 - _pq_x11*_pq_x133 - _pq_x12*_pq_x48 + _pq_x122*_pq_x91 - _pq_x123*_pq_x91 + _pq_x131*_pq_x26 - _pq_x131*_pq_x27 + _pq_x131*_pq_x28 - _pq_x133*_pq_x19 + _pq_x134*_pq_x17 + _pq_x134*_pq_x22 + _pq_x15*_pq_x48 - _pq_x20*_pq_x48, 1.5491782009214e-17*T_0*T_2*T_7 + 1.5491782009214e-17*T_0*T_3*T_6 + 4.58556747472735e-18*T_10*T_4 + 1.5491782009214e-17*T_10*T_7*T_8 + 1.5491782009214e-17*T_11*T_6*T_8 + 1.5491782009214e-17*T_4*T_6*T_7 - T_4*_pq_x108 + 9.79855212082787e-19*T_4*_pq_x11 + 6.12323399573677e-17*T_4*_pq_x12 - T_4*_pq_x139 + 9.79855212082787e-19*T_4*_pq_x14 + 6.12323399573677e-17*T_4*_pq_x15 + 5.36493269770472e-18*T_4 + 3.62495452547617e-17*T_7*T_8 - T_8*_pq_x135 - T_8*_pq_x137 - _pq_x110*_pq_x80 - _pq_x132*_pq_x36 - _pq_x132*_pq_x83 - _pq_x136*_pq_x26 - _pq_x136*_pq_x27 - _pq_x138*_pq_x22 - _pq_x138*_pq_x9 - _pq_x20*_pq_x49 - 1.39615858336794e-18*_pq_x78, 1.5491782009214e-17*T_1*T_2*T_7 + 1.5491782009214e-17*T_1*T_3*T_6 + 4.58556747472735e-18*T_10*T_5 + 1.5491782009214e-17*T_10*T_7*T_9 + 1.5491782009214e-17*T_11*T_6*T_9 + 1.5491782009214e-17*T_5*T_6*T_7 - T_5*_pq_x108 + 9.79855212082787e-19*T_5*_pq_x11 + 6.12323399573677e-17*T_5*_pq_x12 - T_5*_pq_x139 + 9.79855212082787e-19*T_5*_pq_x14 + 6.12323399573677e-17*T_5*_pq_x15 + 5.36493269770472e-18*T_5 + 3.62495452547617e-17*T_7*T_9 - T_9*_pq_x135 - T_9*_pq_x137 - _pq_x110*_pq_x90 - _pq_x130*_pq_x88 - _pq_x130*_pq_x93 - _pq_x132*_pq_x89 - _pq_x132*_pq_x91 - _pq_x140*_pq_x26 - _pq_x140*_pq_x27 - _pq_x141*_pq_x22 - _pq_x141*_pq_x9 - _pq_x20*_pq_x50 - 4.18847575010382e-18*_pq_x94, -1.54074395550979e-33*T_0*T_8 + T_0*_pq_x144 + 3.85185988877447e-34*T_0*_pq_x38 - 2.21964447833955e-33*T_10*T_3 + T_2*_pq_x142 + T_2*_pq_x143 + 2.33925908339132e-18*T_2*_pq_x9 - 3.28507382794253e-34*T_2 + _pq_x11*_pq_x147 - _pq_x11*_pq_x148 + 5.99988274552518e-35*_pq_x113 - _pq_x12*_pq_x6 - _pq_x14*_pq_x148 + _pq_x145*_pq_x26 + _pq_x145*_pq_x28 - _pq_x146*_pq_x17 - _pq_x146*_pq_x22 - _pq_x146*_pq_x9 + _pq_x147*_pq_x19 - _pq_x148*_pq_x19 + _pq_x149*_pq_x17 + _pq_x149*_pq_x22 + _pq_x15*_pq_x6 - _pq_x20*_pq_x6, 2.21964447833955e-33*T_10*T_7 + 1.54074395550979e-33*T_4*T_8 - T_4*_pq_x144 - 3.85185988877447e-34*T_4*_pq_x36 + 3.74939945665464e-33*T_6*_pq_x12 - T_6*_pq_x142 - T_6*_pq_x143 + 3.74939945665464e-33*T_6*_pq_x15 - 2.33925908339132e-18*T_6*_pq_x17 + 3.28507382794253e-34*T_6 + 9.48598062533625e-34*T_7*_pq_x11 + 9.48598062533625e-34*T_7*_pq_x14 + 1.8492166667125e-17*T_7*_pq_x17 + 9.48598062533625e-34*T_7*_pq_x19 + 1.8492166667125e-17*T_7*_pq_x22 + 1.8492166667125e-17*T_7*_pq_x9 - 5.99988274552518e-35*_pq_x109 - _pq_x11*_pq_x152 - _pq_x14*_pq_x152 - _pq_x150*_pq_x26 - _pq_x150*_pq_x27 - _pq_x151*_pq_x22 - _pq_x151*_pq_x9 - _pq_x20*_pq_x7, 0.253*T_0*T_10*T_3 + 0.253*T_0*T_11*T_2 + 0.592*T_0*T_3 - T_0*_pq_x96 + 0.253*T_10*T_11*T_8 + 0.253*T_10*T_4*T_7 + 0.253*T_11*T_4*T_6 + 0.592*T_11*T_8 + 0.592*T_4*T_7 - T_4*_pq_x156 - T_4*_pq_x77 - T_8*_pq_x10 - 0.01600225*T_8*_pq_x11 - T_8*_pq_x13 + 0.01600225*T_8*_pq_x14 + 1.0*T_8*_pq_x15 - T_8*_pq_x18 + 0.01600225*T_8*_pq_x19 + 1.0*T_8*_pq_x20 - T_8*_pq_x8 - T_8*_pq_x85 - T_8*_pq_x98 - 0.087616*T_8 - 0.022801*_pq_x153 - _pq_x154*_pq_x36 - _pq_x154*_pq_x38 - _pq_x155*_pq_x80, 0.253*T_1*T_10*T_3 + 0.253*T_1*T_11*T_2 + 0.592*T_1*T_3 - T_1*_pq_x96 + 0.253*T_10*T_11*T_9 + 0.253*T_10*T_5*T_7 + 0.253*T_11*T_5*T_6 + 0.592*T_11*T_9 + 0.592*T_5*T_7 - T_5*_pq_x156 - T_5*_pq_x77 - T_9*_pq_x10 - 0.01600225*T_9*_pq_x11 - T_9*_pq_x13 + 0.01600225*T_9*_pq_x14 + 1.0*T_9*_pq_x15 - T_9*_pq_x18 + 0.01600225*T_9*_pq_x19 + 1.0*T_9*_pq_x20 - T_9*_pq_x8 - T_9*_pq_x85 - T_9*_pq_x98 - 0.087616*T_9 - _pq_x154*_pq_x89 - _pq_x155*_pq_x90 - 0.045602*_pq_x157 - _pq_x158*_pq_x79 - _pq_x159*_pq_x87 - 0.068403*_pq_x160],
        [_pq_x102, _pq_x107, _pq_x86, _pq_x95, _pq_x116, _pq_x112, 1.5491782009214e-17*T_0*T_10*T_3 + 1.5491782009214e-17*T_0*T_11*T_2 + 3.62495452547617e-17*T_0*T_3 - T_0*_pq_x119 + 1.5491782009214e-17*T_10*T_11*T_8 + 1.5491782009214e-17*T_10*T_4*T_7 + 1.5491782009214e-17*T_11*T_4*T_6 + 3.62495452547617e-17*T_11*T_8 + 3.62495452547617e-17*T_4*T_7 - T_4*_pq_x135 - T_4*_pq_x137 - T_8*_pq_x118 + 9.79855212082787e-19*T_8*_pq_x14 + 6.12323399573677e-17*T_8*_pq_x15 - T_8*_pq_x164 + 9.79855212082787e-19*T_8*_pq_x19 + 6.12323399573677e-17*T_8*_pq_x20 - 5.36493269770473e-18*T_8 - _pq_x12*_pq_x53 - 1.39615858336794e-18*_pq_x153 - _pq_x161*_pq_x36 - _pq_x161*_pq_x38 - _pq_x162*_pq_x27 - _pq_x162*_pq_x28 - _pq_x163*_pq_x17 - _pq_x163*_pq_x9 - _pq_x30*_pq_x80, 1.5491782009214e-17*T_1*T_10*T_3 + 1.5491782009214e-17*T_1*T_11*T_2 + 3.62495452547617e-17*T_1*T_3 - T_1*_pq_x119 + 1.5491782009214e-17*T_10*T_11*T_9 + 1.5491782009214e-17*T_10*T_5*T_7 + 1.5491782009214e-17*T_11*T_5*T_6 + 3.62495452547617e-17*T_11*T_9 + 3.62495452547617e-17*T_5*T_7 - T_5*_pq_x135 - T_5*_pq_x137 - T_9*_pq_x118 + 9.79855212082787e-19*T_9*_pq_x14 + 6.12323399573677e-17*T_9*_pq_x15 - T_9*_pq_x164 + 9.79855212082787e-19*T_9*_pq_x19 + 6.12323399573677e-17*T_9*_pq_x20 - 5.36493269770473e-18*T_9 - _pq_x12*_pq_x54 - _pq_x130*_pq_x159 - _pq_x132*_pq_x158 - 2.79231716673588e-18*_pq_x157 - 4.18847575010382e-18*_pq_x160 - _pq_x161*_pq_x89 - _pq_x165*_pq_x27 - _pq_x165*_pq_x28 - _pq_x166*_pq_x17 - _pq_x166*_pq_x9 - _pq_x30*_pq_x90],
    ], dtype=np.float64)

    return p_sin, p_cos, p_one, q



def _solve_algebraic(T_target):
    """Tier-2 Raghavan-Roth IK candidates with INLINED P/Q matrices.

    Per-arm DH constants substituted into the 4 RR matrix builders
    (CSE'd above); generic linear algebra (eliminate_q0_q1,
    Weierstrass, eigvals, back-substitution) stays imported.
    """
    T = np.asarray(T_target, dtype=np.float64)
    T_dh = _T_PRE_INV @ T @ _T_POST_INV

    # Destructure T_dh's free entries (top 3 rows, 4 cols).
    T_0, T_1, T_2, T_3 = T_dh[0, 0], T_dh[0, 1], T_dh[0, 2], T_dh[0, 3]
    T_4, T_5, T_6, T_7 = T_dh[1, 0], T_dh[1, 1], T_dh[1, 2], T_dh[1, 3]
    T_8, T_9, T_10, T_11 = T_dh[2, 0], T_dh[2, 1], T_dh[2, 2], T_dh[2, 3]
    p_sin, p_cos, p_one, q_mat = _build_pq_matrices(
        T_0, T_1, T_2, T_3, T_4, T_5, T_6, T_7, T_8, T_9, T_10, T_11
    )

    # Generic linear algebra: arm-agnostic, runtime-imported. Phase 4
    # Cython links these to LAPACK directly.
    e_sin, e_cos, e_one = _ssik_eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = _ssik_weierstrass(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = _ssik_build_m_matrix(e_quad, e_lin, e_const)
    roots, eigvecs = _ssik_solve_x2_roots_mobius(m_quad, m_lin, m_const)

    q_pinv = np.linalg.pinv(q_mat).astype(np.float64)

    # Back-substitute per real root. ``_back_substitute_inner``
    # returns ``(q_dh, fk_err_alg)`` -- we drop fk_err here because
    # the outer ``solve()`` re-runs FK on each candidate (mirroring
    # the wrapper version's behavior, kept for parity).
    inner_qs = []
    for x_lin, eigvec in zip(roots, eigvecs):
        bs_result = _ssik_back_substitute_inner(
            x_lin, eigvec, p_sin, p_cos, p_one, q_pinv,
            _DH_TUPLE, T_dh, _RR_META,
        )
        if bs_result is None:
            continue
        q_dh, _fk_err = bs_result
        inner_qs.append(q_dh)

    # Map DH-frame q back to POE frame.
    return [
        list(np.asarray(q_dh, dtype=np.float64) - _DH_THETA_OFFSET)
        for q_dh in inner_qs
    ]


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
        if not (allow_refinement or True):
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
