"""Generated IK module for Kinova Gen3 (7-DOF).

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash e782ba121009 (sha256/12 of the input chain).
``T_target`` is the pose of ``end_effector_link`` (end-effector link) in
``base_link`` (base link). If your URDF differs (calibrated
geometry, custom tool past the flange, different link names),
run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
produce an artifact correct for your hardware.

DOF: 7    BASE_LINK: "base_link"    EE_LINK: "end_effector_link"
Solver: ``seven_r.srs_polished`` (tier 0)
Expected median IK time: ~56.0 ms on commodity
single-thread hardware. FLOP budget: 80,000 per solve.

Usage:

    import gen3_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose of end_effector_link in base_link
    T_target[:3, 3] = [0.5, 0.1, 0.3]
    solutions = gen3_ik.solve(T_target)
    for sol in solutions:
        print(sol.q, sol.fk_residual)

``solve(T)`` returns ``list[Solution]``. Empty list iff no
candidate closed within the solver's FK tolerance -- check
``if not solutions:`` for the "unreachable" case.

Sanity-check the baked geometry: ``gen3_ik.T_HOME`` is the
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
from ssik.refinement import seeded_track as _seeded_track
from ssik.refinement.rescue import rescue_via_T_perturbation as _rescue_via_T_perturbation
from ssik.solvers.seven_r._swivel_limits import resolve_in_limits as _resolve_in_limits
from ssik.solvers.seven_r.srs_polished import solve as _solver_solve

SOLVER_NAME = "seven_r.srs_polished"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 56.0
FLOP_BUDGET = 80000
DISPATCH_REASON = 'Approximately-SRS 7R: shoulder axes meet within 11.8 mm, wrist axes meet within 0.4 mm.\nSingh-Kreutz on the relaxed pivots produces algebraic\ncandidates; LM polish recovers machine-precision FK\nagainst the original URDF. 16-30x faster than the\nuniversal jointlock+HP fallback on small-drift arms.\nCovers Kinova Gen3 (12 mm / 0.4 mm drift).'
BASE_LINK = "base_link"
EE_LINK = "end_effector_link"
DOF = 7
# Home pose: FK at q = np.zeros(DOF). Sanity-check this against
# your robot's documented home pose to verify the baked geometry
# matches your URDF.
T_HOME = np.array([[1.0, -8.326720029981853e-15, -2.1017948830410524e-16, -1.925826838264291e-16], [8.326718485692419e-15, 0.9999999999730148, -7.346410203412496e-06, -0.024859601294873888], [2.102406597994228e-16, 7.3464102034124935e-06, 0.9999999999730148, 1.1873847699190918], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

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
    np.array([[1.0, -8.326720029981853e-15, -2.1017948830410524e-16, -1.2931293017908443e-17], [8.326718485692419e-15, 0.9999999999730148, -7.346410203412496e-06, -4.5198788796441125e-07], [2.102406597994228e-16, 7.3464102034124935e-06, 0.9999999999730148, 0.061524999998339824], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
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

    Solver: srs_polished.
    """
    if seed_tolerance is not None and q_seed is None:
        raise ValueError("seed_tolerance requires q_seed")
    # Seeded numerical-tracking fast path (#380): the caller gave a seed
    # and wants a single IK -- the trajectory-tracking idiom. Newton-
    # continue from the seed (~0.2 ms) instead of resolving the whole
    # redundancy (several ms). On a smooth trajectory the continuation is
    # exactly the seed-nearest solution the full solve would return; it
    # is run through the same limit/tolerance postprocess below so its
    # output is indistinguishable from the full path's. When the seed
    # doesn't continue cleanly (Newton jumped a branch, diverged, or the
    # result fails limits/seed_tolerance) ``_seeded_track`` returns
    # ``None`` / the postprocess empties and we fall through to the full
    # analytical solve -- correctness is never traded for speed.
    if q_seed is not None and max_solutions == 1:
        _tracked = _seeded_track(
            np.asarray(q_seed, dtype=np.float64),
            fk,
            lambda _q: _kinbody_jacobian(_KB, _q),
            np.asarray(T_target, dtype=np.float64),
        )
        if _tracked is not None:
            _fast = [_tracked]
            if respect_limits:
                _fast = _ps_respect_limits(_ps_wrap_to_limits(_fast, _KB), _KB)
            if _fast and seed_tolerance is not None:
                _fast = _ps_within_seed_tolerance(_fast, q_seed, seed_tolerance)
            if _fast:
                return _fast[:1]
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
        if not sols:
            # #359: the blind swivel sweep sampled no in-limits candidate
            # even though a reachable in-limits solution exists (the
            # in-limits swivel arc was narrower than the sampling). The
            # feasible-swivel resolver computes the in-limits arcs exactly
            # and returns solutions directly (no-op for non-SRS chains).
            sols = _resolve_in_limits(_KB, T_target, policy=policy)
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
