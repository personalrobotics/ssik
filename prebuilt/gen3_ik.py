"""Generated IK module for Kinova Gen3 (7-DOF).

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash 9a29063ad96b (sha256/12 of the input chain).

Solver: ``seven_r.srs_polished`` (tier 0)
Expected median IK time: ~56.0 ms on commodity
single-thread hardware. FLOP budget: 80,000 per solve.

Usage:

    import gen3_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions = gen3_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``list[Solution]``. Empty list iff no
candidate closed within the solver's FK tolerance -- check
``if not solutions:`` for the "unreachable" case.
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
from ssik.solvers.seven_r.srs_polished import solve as _solver_solve

SOLVER_NAME = "seven_r.srs_polished"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 56.0
FLOP_BUDGET = 80000
DISPATCH_REASON = 'Approximately-SRS 7R: shoulder axes meet within 11.8 mm, wrist axes meet within 0.4 mm.\nSingh-Kreutz on the relaxed pivots produces algebraic\ncandidates; LM polish recovers machine-precision FK\nagainst the original URDF. 16-30x faster than the\nuniversal jointlock+HP fallback on small-drift arms.\nCovers Kinova Gen3 (12 mm / 0.4 mm drift).'

# --- baked KinBody constants ---

_LINK_NAMES = ['base_link', '_poe_link_1', '_poe_link_2', '_poe_link_3', '_poe_link_4', '_poe_link_5', '_poe_link_6', 'end_effector_link']

_JOINT_NAMES = [
    'joint_1',
    'joint_2',
    'joint_3',
    'joint_4',
    'joint_5',
    'joint_6',
    'joint_7',
]

_JOINT_AXES = [
    np.array([-2.7628999999254435e-18, 7.346410206643587e-06, -0.9999999999730151], dtype=np.float64),
    np.array([-1.1102004795037241e-16, 0.999999999939284, 1.1019615309841479e-05], dtype=np.float64),
    np.array([3.098000999981099e-16, 7.346410206643586e-06, -0.999999999973015], dtype=np.float64),
    np.array([-2.775498020134744e-16, 0.9999999999392839, 1.1019615309841477e-05], dtype=np.float64),
    np.array([3.0657528439921877e-16, 7.3464102066435854e-06, -0.9999999999730149], dtype=np.float64),
    np.array([-8.271209801958048e-15, 0.9999999999392838, 1.1019615309841475e-05], dtype=np.float64),
    np.array([2.1017948830407832e-16, 7.3464102066435854e-06, -0.9999999999730148], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.15643], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 3.5470099289192083e-19], [0.0, 1.0, 0.0, -0.005375943131997285], [0.0, 0.0, 1.0, 0.12837996050958084], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -3.201042839519248e-18], [0.0, 1.0, 0.0, -0.00637731830628182], [0.0, 0.0, 1.0, 0.210379929737179], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -6.446799948640456e-17], [0.0, 1.0, 0.0, -0.006376545537607244], [0.0, 0.0, 1.0, 0.2103799531609578], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -4.884682014127912e-17], [0.0, 1.0, 0.0, -0.006377296818031966], [0.0, 0.0, 1.0, 0.20842992973729735], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -3.246580308269295e-17], [0.0, 1.0, 0.0, -0.00017582820522846557], [0.0, 0.0, 1.0, 0.10592999871115227], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -3.102442625151671e-17], [0.0, 1.0, 0.0, -0.0001762173078391424], [0.0, 0.0, 1.0, 0.1059299980645847], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, -8.326720029981853e-15, -2.1017948830410524e-16, -1.2931293017908443e-17], [8.326718485692419e-15, 0.9999999999730148, -7.346410203412496e-06, -4.5198788796441125e-07], [2.102406597994228e-16, 7.346410203412494e-06, 0.9999999999730148, 0.061524999998339824], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
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
    None,
    (-2.24, 2.24),
    None,
    (-2.57, 2.57),
    None,
    (-2.09, 2.09),
    None,
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
    allow_refinement: bool = True,
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

    Solver: srs_polished.
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
    "DISPATCH_REASON",
    "EXPECTED_MS_MEDIAN",
    "FLOP_BUDGET",
    "SOLVER_NAME",
    "SOLVER_TIER",
    "fk",
    "solve",
]
