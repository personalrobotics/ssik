"""Generated IK module for Enactic OpenArm v2.0 (left).

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash 90b1f58a603c (sha256/12 of the input chain).
``T_target`` is the pose of ``openarm_left_ee_base_link`` (end-effector link) in
``openarm_left_base_link`` (base link). If your URDF differs (calibrated
geometry, custom tool past the flange, different link names),
run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
produce an artifact correct for your hardware.

DOF: 7    BASE_LINK: "openarm_left_base_link"    EE_LINK: "openarm_left_ee_base_link"
Solver: ``seven_r.srs`` (tier 0)
Expected median IK time: ~8.5 ms on commodity
single-thread hardware. FLOP budget: 1,900 per solve.

Usage:

    import openarm_left_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose of openarm_left_ee_base_link in openarm_left_base_link
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions = openarm_left_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``list[Solution]``. Empty list iff no
candidate closed within the solver's FK tolerance -- check
``if not solutions:`` for the "unreachable" case.

Sanity-check the baked geometry: ``openarm_left_ik.T_HOME`` is the
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
    within_seed_tolerance as _ps_within_seed_tolerance,
    wrap_to_limits as _ps_wrap_to_limits,
)
import functools as _functools
from ssik.refinement import kinbody_jacobian as _kinbody_jacobian
from ssik.refinement.rescue import rescue_via_T_perturbation as _rescue_via_T_perturbation
from ssik.solvers.seven_r.srs import solve as _solver_solve

SOLVER_NAME = "seven_r.srs"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 8.5
FLOP_BUDGET = 1900
DISPATCH_REASON = 'SRS-class 7R: shoulder axes (joints 0, 1, 2) meet at\none point + wrist axes (joints 4, 5, 6) meet at one\npoint + joint 3 is the elbow. Closed-form Singh-Kreutz\n1989 algorithm, parameterised by elbow swivel angle.\nCovers KUKA iiwa LBR (canonical strict-SRS).'
BASE_LINK = "openarm_left_base_link"
EE_LINK = "openarm_left_ee_base_link"
DOF = 7
# Home pose: FK at q = np.zeros(DOF). Sanity-check this against
# your robot's documented home pose to verify the baked geometry
# matches your URDF.
T_HOME = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.12250000000000011], [0.0, 0.0, 1.0, -0.43599999999999933], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

# --- baked KinBody constants ---

_LINK_NAMES = ['openarm_left_base_link', '_poe_link_1', '_poe_link_2', '_poe_link_3', '_poe_link_4', '_poe_link_5', '_poe_link_6', 'openarm_left_ee_base_link']

_JOINT_NAMES = [
    'openarm_left_joint1',
    'openarm_left_joint2',
    'openarm_left_joint3',
    'openarm_left_joint4',
    'openarm_left_joint5',
    'openarm_left_joint6',
    'openarm_left_joint7',
]

_JOINT_AXES = [
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, -1.0], dtype=np.float64),
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, -1.0], dtype=np.float64),
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0625], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.06000000000000011], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.0662500000000008], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.15375], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.09550000000000053], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.120499999999998], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
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
    'revolute',
]

_JOINT_LIMITS = [
    (-3.4907, 1.3963),
    (-3.3161, 0.17453),
    (-1.5708, 1.5708),
    (0.0, 2.4435),
    (-1.5708, 1.5708),
    (-0.7854, 0.7854),
    (-0.7854, 0.7854),
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
    allow_rescue: bool = True,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    refinement_max_iters: int = 15,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
):
    """Inverse kinematics. Returns ``list[Solution]``.

    :param T_target: 4x4 SE(3) target end-effector pose, np.float64.
    :param max_solutions: optional cap on returned IKs (post-dedup,
        post-limits filter). ``None`` = full enumeration.
    :param q_seed: optional joint config. When provided, solutions
        are sorted by distance from ``q_seed`` (closest first, via
        ``seed_metric``). Combine with ``max_solutions=1`` for the
        trajectory-tracking idiom.
    :param seed_metric: distance used to rank against ``q_seed``.
        ``"wrap_linf"`` (default, largest single-joint move) holds
        the branch during tracking; ``"wrap_l2"`` uses the summed
        move. Ignored when ``q_seed`` is ``None``.
    :param seed_tolerance: optional max per-joint deviation from
        ``q_seed`` (radians, wrap-to-pi). When set, only solutions with
        *every* joint within ``seed_tolerance`` are returned -- a hard
        tracking guarantee that may return an empty list when no branch
        qualifies. ``None`` (default) keeps the best-effort behaviour.
        Requires ``q_seed``.
    :param respect_limits: when ``True`` (default), solutions
        outside URDF joint limits are dropped. ``False`` returns
        the raw geometric set.
    :param allow_refinement: when ``True`` (default), Newton polish
        fires on near-miss algebraic candidates. Tightens FK
        closure to machine precision.
    :param allow_rescue: when ``True`` (default), if the analytical
        path returns no solutions but the target is within the arm's
        reach-sphere, ``solve()`` recovers the IK via the
        T-perturbation rescue (#319) -- reachable-but-degenerate poses
        (near-singular / near-parallel-axis) return LM-polished
        solutions tagged ``refinement_used="lm"`` instead of ``[]``.
        Set ``False`` for a guaranteed-analytical-or-empty result.
        Gated by the reach-sphere, so far-field unreachable targets
        stay cheap (no rescue fired).
    :param policy: tolerance policy. Rarely customised.
    :param refinement_max_iters: cap on Newton iterations per
        candidate when ``allow_refinement=True``.
    :returns: list of :class:`Solution`, one per analytical IK
        branch (plus any rescued at a degenerate pose). Empty list
        iff the target is unreachable or ``allow_rescue=False`` and
        the analytical path found nothing.

    Solver: srs.
    """
    if seed_tolerance is not None and q_seed is None:
        raise ValueError("seed_tolerance requires q_seed")
    sols, _is_ls = _solver_solve(
        _KB,
        T_target,
        policy=policy,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
    )
    # Bulletproof fallback (#319 / #358): the analytical path found
    # nothing. If the target is within the arm's max reach it may be a
    # measure-zero degenerate pose (near-singular elbow/gimbal, or a
    # near-parallel-axis spherical joint) the algebraic extraction
    # can't resolve -- rather than an unreachable target. Recover via
    # the T-perturbation rescue. The reach-sphere (sum of link lengths;
    # an exact upper bound by the triangle inequality, so it never
    # rejects a reachable pose) is checked only in this rare empty
    # branch and keeps far-field targets cheap. Perturbed re-solves run
    # with allow_rescue=False (recursion guard + analytical escape
    # hatch); the rescue calls back with respect_limits=False, so the
    # rescued set flows through the same limit/seed postprocess below.
    if not sols and allow_rescue:
        _reach_radius = sum(
            float(np.linalg.norm(np.asarray(_t)[:3, 3]))
            for _t in (*_JOINT_T_LEFTS, *_JOINT_T_RIGHTS)
        )
        _T = np.asarray(T_target, dtype=np.float64)
        if float(np.linalg.norm(_T[:3, 3])) <= _reach_radius:
            sols = _rescue_via_T_perturbation(
                fk,
                _functools.partial(solve, allow_rescue=False),
                _T,
                jacobian_fn=lambda _q: _kinbody_jacobian(_KB, _q),
            )
    if respect_limits:
        sols = _ps_wrap_to_limits(sols, _KB)
        sols = _ps_respect_limits(sols, _KB)
    if q_seed is not None:
        if seed_tolerance is not None:
            sols = _ps_within_seed_tolerance(sols, q_seed, seed_tolerance)
        sols = _ps_nearest_to_seed(sols, q_seed, metric=seed_metric)
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
