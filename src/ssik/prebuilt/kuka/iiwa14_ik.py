"""Generated IK module for KUKA iiwa LBR 14.

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance: KinBody hash 23855e20b39c (sha256/12 of the input chain).
``T_target`` is the pose of ``iiwa_link_ee_kuka`` (end-effector link) in
``base`` (base link). If your URDF differs (calibrated
geometry, custom tool past the flange, different link names),
run ``ssik build <your.urdf> --base <yours> --ee <yours>`` to
produce an artifact correct for your hardware.

DOF: 7    BASE_LINK: "base"    EE_LINK: "iiwa_link_ee_kuka"
Solver: ``seven_r.srs`` (tier 0)
Expected median IK time: ~8.5 ms on commodity
single-thread hardware. FLOP budget: 1,900 per solve.

Usage:

    import iiwa14_ik
    import numpy as np
    T_target = np.eye(4)  # 4x4 SE(3) pose of iiwa_link_ee_kuka in base
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
from ssik.postprocess import finalize_solutions as _ps_finalize
import functools as _functools
from ssik.refinement import kinbody_jacobian as _kinbody_jacobian
from ssik.refinement import seeded_track as _seeded_track
from ssik.refinement.rescue import rescue_via_T_perturbation as _rescue_via_T_perturbation
from ssik.solvers.seven_r._swivel_limits import resolve_in_limits as _resolve_in_limits
from ssik.solvers.seven_r.srs import solve as _solver_solve
from ssik._native import try_native_srs_solve as _try_native_srs_solve

SOLVER_NAME = "seven_r.srs"
SOLVER_TIER = 0
EXPECTED_MS_MEDIAN = 8.5
FLOP_BUDGET = 1900
DISPATCH_REASON = 'SRS-class 7R: shoulder axes (joints 0, 1, 2) meet at\none point + wrist axes (joints 4, 5, 6) meet at one\npoint + joint 3 is the elbow. Closed-form Singh-Kreutz\n1989 algorithm, parameterised by elbow swivel angle.\nCovers KUKA iiwa LBR (canonical strict-SRS).'
BASE_LINK = "base"
EE_LINK = "iiwa_link_ee_kuka"
DOF = 7
# Home pose: FK at q = np.zeros(DOF). Sanity-check this against
# your robot's documented home pose to verify the baked geometry
# matches your URDF.
T_HOME = np.array([[1.0, 4.898587196589412e-16, 1.4791141972893971e-31, -5.449678256205724e-17], [-4.898587196589413e-16, 1.0, 1.224646799147353e-16, -1.5884013222165315e-16], [-7.395570986446986e-32, -1.224646799147353e-16, 1.0, 1.3059999999999998], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

# --- baked KinBody constants ---

_LINK_NAMES = ['base', '_poe_link_1', '_poe_link_2', '_poe_link_3', '_poe_link_4', '_poe_link_5', '_poe_link_6', 'iiwa_link_ee_kuka']

_JOINT_NAMES = [
    'iiwa_joint_1',
    'iiwa_joint_2',
    'iiwa_joint_3',
    'iiwa_joint_4',
    'iiwa_joint_5',
    'iiwa_joint_6',
    'iiwa_joint_7',
]

_JOINT_AXES = [
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([1.2246467991473532e-16, 1.0, -3.8285686989269494e-16], dtype=np.float64),
    np.array([-1.2246467991473532e-16, 0.0, 1.0], dtype=np.float64),
    np.array([-1.2246467991473522e-16, -1.0, -3.8285686989269494e-16], dtype=np.float64),
    np.array([-1.0864713024671694e-31, -6.661338147750939e-16, 1.0], dtype=np.float64),
    np.array([2.4492935982947054e-16, 1.0, 2.83276944882399e-16], dtype=np.float64),
    np.array([-1.2246467991473525e-16, 4.930380657631324e-32, 1.0], dtype=np.float64),
]

_JOINT_T_LEFTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.1575], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.20249999999999999], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 9.588277803023819e-33], [0.0, 1.0, 0.0, 7.829422989305611e-17], [0.0, 0.0, 1.0, 0.20450000000000002], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -2.6391138521625462e-17], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.21550000000000002], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -2.2594733444268678e-17], [0.0, 1.0, 0.0, -7.063709249520222e-17], [0.0, 0.0, 1.0, 0.1845], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, -2.465190328815662e-32], [0.0, 1.0, 0.0, -1.435518370840327e-16], [0.0, 0.0, 1.0, 0.2154999999999999], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -2.2945432535474328e-17], [0.0, 0.0, 1.0, 0.08099999999999996], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
]

_JOINT_T_RIGHTS = [
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    np.array([[1.0, 4.898587196589412e-16, 1.4791141972893971e-31, -5.5109105961630834e-18], [-4.898587196589413e-16, 1.0, 1.224646799147353e-16, 0.0], [-7.395570986446986e-32, -1.224646799147353e-16, 1.0, 0.04499999999999993], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
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
    (-2.96705972839, 2.96705972839),
    (-2.09439510239, 2.09439510239),
    (-2.96705972839, 2.96705972839),
    (-2.09439510239, 2.09439510239),
    (-2.96705972839, 2.96705972839),
    (-2.09439510239, 2.09439510239),
    (-3.05432619099, 3.05432619099),
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
    native: bool = False,
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
    if native:
        _native_sols = _try_native_srs_solve(
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
            # Same post-processing as the full path (no in-limits
            # fallback: a tracked seed that fails limits/tolerance falls
            # through to the full analytical solve below).
            _fast = _ps_finalize(
                [_tracked],
                _KB,
                respect_limits=respect_limits,
                q_seed=q_seed,
                seed_metric=seed_metric,
                seed_tolerance=seed_tolerance,
                max_solutions=1,
            )
            if _fast:
                return _fast
    sols, _is_ls = _solver_solve(
        _KB,
        T_target,
        policy=policy,
        allow_refinement=allow_refinement,
        refinement_max_iters=refinement_max_iters,
    )
    # Limit pass + #359 in-limits fallback ONLY (no seed/tolerance/
    # truncate yet). The rescue gate below is "no in-limits solution
    # exists", so it must not depend on the seed-tolerance / max_solutions
    # filters -- a seed filter emptying a non-empty in-limits set is a
    # user preference, NOT a missing solution, and must not trigger a
    # rescue (which would diverge native vs Python by sampling a different
    # continuum). The in-limits fallback (#359) recovers a reachable
    # in-limits solution the coarse sweep missed (no-op for non-redundant
    # chains).
    _in_limits = _ps_finalize(
        sols,
        _KB,
        respect_limits=respect_limits,
        in_limits_fallback=lambda: _resolve_in_limits(_KB, T_target, policy=policy),
    )
    # Bulletproof fallback (#319 / #358 / #524): nothing survives the
    # limit filter. If the target is within the arm's max reach it may be
    # a measure-zero degenerate pose (near-singular elbow/gimbal) the
    # algebraic extraction can't resolve, NOT an unreachable target --
    # recover via the T-perturbation rescue. Gating on the LIMIT-filtered
    # empty (not the pre-limit analytical count) is what lets a pose whose
    # only analytical candidates were out-of-limits still get rescued
    # (#524). The reach-sphere (sum of link lengths; an exact triangle-
    # inequality upper bound, so it never rejects a reachable pose) keeps
    # far-field targets cheap; perturbed re-solves run allow_rescue=False.
    if not _in_limits and allow_rescue:
        _reach_radius = sum(
            float(np.linalg.norm(np.asarray(_t)[:3, 3]))
            for _t in (*_JOINT_T_LEFTS, *_JOINT_T_RIGHTS)
        )
        _T = np.asarray(T_target, dtype=np.float64)
        if float(np.linalg.norm(_T[:3, 3])) <= _reach_radius:
            _in_limits = _ps_finalize(
                _rescue_via_T_perturbation(
                    fk,
                    _functools.partial(solve, allow_rescue=False),
                    _T,
                    jacobian_fn=lambda _q: _kinbody_jacobian(_KB, _q),
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
