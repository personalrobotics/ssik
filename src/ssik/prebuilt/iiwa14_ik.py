"""Generated IK module for KUKA iiwa LBR 14.

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash fd5922ca3dcc (sha256/12 of the input chain).
``T_target`` is the pose of ``ee_link`` (end-effector link) in
``base_link`` (base link). If your URDF differs (calibrated
geometry, custom tool past the flange, different link names),
run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
produce an artifact correct for your hardware.

DOF: 7    BASE_LINK: "base_link"    EE_LINK: "ee_link"
Solver: ``seven_r.srs`` (tier 0)
Expected median IK time: ~8.5 ms on commodity
single-thread hardware. FLOP budget: 1,900 per solve.

Usage:

    import iiwa14_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose of ee_link in base_link
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions = iiwa14_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``list[Solution]``. Empty list iff no
candidate closed within the solver's FK tolerance -- check
``if not solutions:`` for the "unreachable" case.

Sanity-check the baked geometry: ``iiwa14_ik.T_HOME`` is the
4x4 home pose (FK at ``q = np.zeros(DOF)``). If it doesn't match
your robot's home pose, the artifact is for a different URDF.
"""

from __future__ import annotations

import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.postprocess import (
    nearest_to_seed as _ps_nearest_to_seed,
    respect_limits as _ps_respect_limits,
    wrap_to_limits as _ps_wrap_to_limits,
)
from ssik.solvers.seven_r.srs import solve as _solver_solve

SOLVER_NAME = "seven_r.srs"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 8.5
FLOP_BUDGET = 1900
DISPATCH_REASON = 'SRS-class 7R: shoulder axes (joints 0, 1, 2) meet at\none point + wrist axes (joints 4, 5, 6) meet at one\npoint + joint 3 is the elbow. Closed-form Singh-Kreutz\n1989 algorithm, parameterised by elbow swivel angle.\nCovers KUKA iiwa LBR (canonical strict-SRS).'
BASE_LINK = "base_link"
EE_LINK = "ee_link"
DOF = 7
# Home pose: FK at q = np.zeros(DOF). Sanity-check this against
# your robot's documented home pose to verify the baked geometry
# matches your URDF.
T_HOME = np.array([[0.9999999999999982, 0.0, 0.0, 0.0], [0.0, 0.9999999999999987, 4.4408920985006217e-16, 3.976818874207309e-16], [0.0, 4.440892098500621e-16, 0.9999999999999987, 1.3059999999999994], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

# --- baked KinBody constants ---

_LINK_NAMES = ['base_link', 'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6', 'ee_link']

_JOINT_NAMES = [
    'iiwa_joint1',
    'iiwa_joint2',
    'iiwa_joint3',
    'iiwa_joint4',
    'iiwa_joint5',
    'iiwa_joint6',
    'iiwa_joint7',
]

_JOINT_AXES = [
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 0.9999999999999998, 2.220446049250313e-16], dtype=np.float64),
    np.array([0.0, 4.440892098500625e-16, 0.9999999999999996], dtype=np.float64),
    np.array([0.0, -0.9999999999999993, -2.220446049250312e-16], dtype=np.float64),
    np.array([0.0, 4.4408920985006237e-16, 0.9999999999999991], dtype=np.float64),
    np.array([0.0, 0.9999999999999989, 2.220446049250311e-16], dtype=np.float64),
    np.array([0.0, 4.4408920985006217e-16, 0.9999999999999987], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1575], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.20249999999999999], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 4.54081217071689e-17], [0.0, 0.0, 1.0, 0.2044999999999999], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 9.570122472268846e-17], [0.0, 0.0, 1.0, 0.2154999999999999], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.2290168882600478e-16], [0.0, 0.0, 1.0, 0.1844999999999999], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 9.570122472268846e-17], [0.0, 0.0, 1.0, 0.2154999999999998], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.7985612998927527e-17], [0.0, 0.0, 1.0, 0.08099999999999996], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[0.9999999999999982, 0.0, 0.0, 0.0], [0.0, 0.9999999999999987, 4.4408920985006217e-16, 1.9984014443252786e-17], [0.0, 4.440892098500621e-16, 0.9999999999999987, 0.04499999999999993], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_TYPES = [
    'revolute',
    'revolute',
    'revolute',
    'revolute',
    'revolute',
    'revolute',
    'revolute',
]

_JOINT_LIMITS = [
    (-2.96706, 2.96706),
    (-2.0944, 2.0944),
    (-2.96706, 2.96706),
    (-2.0944, 2.0944),
    (-2.96706, 2.96706),
    (-2.0944, 2.0944),
    (-3.05433, 3.05433),
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


def solve(
    T_target,
    *,
    max_solutions=None,
    q_seed=None,
    respect_limits: bool = True,
    allow_refinement: bool = False,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    refinement_max_iters: int = 15,
):
    """Inverse kinematics. Returns ``list[Solution]``.

    :param T_target: 4x4 SE(3) target end-effector pose, np.float64.
    :param max_solutions: optional cap on returned IKs (post-dedup,
        post-limits filter). ``None`` = full enumeration.
    :param q_seed: optional joint config. When provided, solutions
        are sorted by wrap-to-pi distance from ``q_seed`` (closest
        first). Combine with ``max_solutions=1`` for the
        trajectory-tracking idiom.
    :param respect_limits: when ``True`` (default), solutions
        outside URDF joint limits are dropped. ``False`` returns
        the raw geometric set.
    :param allow_refinement: when ``True`` (default), Newton polish
        fires on near-miss algebraic candidates. Tightens FK
        closure to machine precision.
    :param policy: tolerance policy. Rarely customised.
    :param refinement_max_iters: cap on Newton iterations per
        candidate when ``allow_refinement=True``.
    :returns: list of :class:`Solution`, one per analytical IK
        branch. Empty list iff no candidate met the FK tolerance
        -- check ``if not sols:`` for "unreachable target".

    Solver: srs.
    """
    sols, _is_ls = _solver_solve(
        _KB,
        T_target,
        policy=policy,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
    )
    if respect_limits:
        sols = _ps_wrap_to_limits(sols, _KB)
        sols = _ps_respect_limits(sols, _KB)
    if q_seed is not None:
        sols = _ps_nearest_to_seed(sols, q_seed)
    if max_solutions is not None and len(sols) > max_solutions:
        sols = sols[:max_solutions]
    return sols

from ssik.kinematics.poe_fk import poe_forward_kinematics as _poe_fk


def fk(q):
    """Forward kinematics: returns the 4x4 base->ee pose at ``q``."""
    return _poe_fk(_KB, np.asarray(q, dtype=np.float64))

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
