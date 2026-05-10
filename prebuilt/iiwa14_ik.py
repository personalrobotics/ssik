"""Generated IK module for KUKA iiwa LBR 14.

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash fd5922ca3dcc (sha256/12 of the input chain).

Solver: ``seven_r.srs`` (tier 0)
Expected median IK time: ~8.5 ms on commodity
single-thread hardware. FLOP budget: 1,900 per solve.

Usage:

    import iiwa14_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions, is_ls = iiwa14_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``(list[Solution], is_ls)``. ``is_ls=True``
signals that no solution closed within the solver's FK tolerance,
and the returned list is the best-LS approximation (or empty).
"""

from __future__ import annotations

import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.solvers.seven_r.srs import solve as _solver_solve

SOLVER_NAME = "seven_r.srs"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 8.5
FLOP_BUDGET = 1900
DISPATCH_REASON = 'SRS-class 7R: shoulder axes (joints 0, 1, 2) meet at\none point + wrist axes (joints 4, 5, 6) meet at one\npoint + joint 3 is the elbow. Closed-form Singh-Kreutz\n1989 algorithm, parameterised by elbow swivel angle.\nCovers KUKA iiwa LBR (canonical strict-SRS).'

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
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
):
    """Inverse kinematics. Returns ``(list[Solution], is_ls)``.

    :param T_target: 4x4 SE(3) target end-effector pose, np.float64.
    :param policy: tolerance policy. Pass a custom
        :class:`ssik.TolerancePolicy` to tighten or relax the
        FK-closure threshold (``subproblem_numerical``), the
        axis-parallel / axis-intersect predicates, etc. Defaults to
        :data:`ssik.DEFAULT_TOLERANCE_POLICY`.
    :param allow_refinement: opt into Newton-on-spatial-Jacobian
        polish for near-miss algebraic candidates. Default ``False``;
        turn on to recover candidates that don't quite meet
        ``policy.subproblem_numerical`` on their own (e.g. near
        kinematic singularities).
    :param refinement_max_iters: cap on Newton iterations per
        candidate when ``allow_refinement=True``.
    :returns: ``(solutions, is_ls)``. Each ``solution.q`` is a joint
        vector matching the source URDF's joint ordering;
        ``solution.fk_residual`` reports closure against
        ``T_target``. ``is_ls=True`` iff the algebraic path produced
        no candidate meeting the FK tolerance -- callers wanting
        only "exact" solutions check ``is_ls`` and discard.

    Solver: srs.
    """
    return _solver_solve(
        _KB,
        T_target,
        policy=policy,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
    )


__all__ = [
    "DISPATCH_REASON",
    "EXPECTED_MS_MEDIAN",
    "FLOP_BUDGET",
    "SOLVER_NAME",
    "SOLVER_TIER",
    "solve",
]
