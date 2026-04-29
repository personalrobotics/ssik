"""Generated IK module for Kinova JACO 2 (j2n6s200).

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Solver: ``ikgeo.general_6r`` (tier 2)
Expected median IK time: ~5.0 ms on commodity
single-thread hardware. FLOP budget: 30,000,000 per solve.

Usage:

    import jaco2_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions, is_ls = jaco2_ik.solve(T_target)
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
from ssik.solvers.ikgeo.general_6r import solve as _solver_solve

SOLVER_NAME = "ikgeo.general_6r"
SOLVER_TIER = 2
EXPECTED_MS_MEDIAN = 5.0
FLOP_BUDGET = 30000000
DISPATCH_REASON = 'No tier-0 (Pieper-class) match.\nTier-2 numeric Raghavan-Roth + Manocha-Canny pipeline with AE-3 leftvar selection. Closes the EAIK coverage gap (Kinova JACO 2 classical, Agilex Piper, custom non-Pieper 6R).\nWeaker structural matches (not used):\n  - axes[1] parallel to axes[2] (would match tier-1 `two_parallel`, but tier-2 RR is ~50x faster)'

# --- baked KinBody constants ---

_LINK_NAMES = ['base_link', 'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'ee_link']

_JOINT_NAMES = [
    'j2n6s200_joint_1',
    'j2n6s200_joint_2',
    'j2n6s200_joint_3',
    'j2n6s200_joint_4',
    'j2n6s200_joint_5',
    'j2n6s200_joint_6',
]

_JOINT_AXES = [
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.15675], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[-0.9999999999999996, -0.0, 0.0, 0.0], [0.0, 2.220446049250313e-16, -0.9999999999999998, 0.0016], [0.0, -0.9999999999999998, 2.220446049250313e-16, -0.11875], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.41], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[-0.9999999999999996, -0.0, 0.0, 0.0], [0.0, 2.220446049250313e-16, -0.9999999999999998, 0.2073], [0.0, -0.9999999999999998, 2.220446049250313e-16, -0.0114], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, -0.49999965031225546, 0.866025605676658, -0.03703], [0.0, 0.866025605676658, 0.49999965031225535, -0.06414], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[-1.0, 0.0, 0.0, 0.0], [0.0, -0.49999965031225546, 0.866025605676658, -0.03703], [0.0, 0.866025605676658, 0.49999965031225535, -0.06414], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[2.220446049250313e-16, 0.9999999999999998, 0.0, 0.0], [0.9999999999999998, 2.220446049250313e-16, 0.0, 0.0], [0.0, 0.0, -0.9999999999999996, -0.16], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
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

    Solver: general_6r.
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
