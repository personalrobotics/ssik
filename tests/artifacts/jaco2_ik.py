"""Generated IK module for Kinova JACO 2 (j2n6s200).

This file was emitted by ``ssik build`` and is the public artifact for
running analytical inverse kinematics on this specific arm. The
per-arm KinBody constants are baked in below; you do not need to
load a URDF or MJCF at runtime.

Provenance (stable across regens; deterministic by construction):
  ssik version : 0.1.dev68+gea952f27b
  KinBody hash : 677649eda2c4

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

import math
from ssik.solvers.ikgeo._raghavan_roth import (
    _cached_derivation as _ssik_cached_derivation,
    solve_all_ik as _ssik_solve_all_ik,
)

import numpy as np

from ssik._kinbody import Joint, KinBody, Link
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.refinement import kinbody_jacobian as _kinbody_jacobian, lm_refine as _lm_refine
from ssik.subproblems._rotation import rotation_matrix as _rotation_matrix

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


# --- baked DH parameters (from poe_to_dh at build time) ---
_DH_ALPHA = np.array([1.5707963267948963, 3.141592653589793, 1.5707963267948968, 1.0471979549811776, 1.0471979549811776, 3.141592653589793], dtype=np.float64)
_DH_A = np.array([0.0, 0.40999999999999986, 0.0, -0.0, 0.0, 0.0], dtype=np.float64)
_DH_D = np.array([-0.11874999999999997, -0.0015999999999996351, -0.011399999999999768, -0.250060739468094, -0.08551929043618828, -0.20275855096809398], dtype=np.float64)
_DH_THETA_OFFSET = np.array([3.141592653589793, 1.5707963267948966, 1.5707963267948966, -0.0, 3.141592653589793, 0.0], dtype=np.float64)
_T_PRE = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.15675], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
_T_POST = np.array([[2.220446049250311e-16, 0.9999999999999989, 0.0, 0.0], [-0.9999999999999984, -2.22044604925031e-16, 0.0, 0.0], [2.2204460492503106e-16, 4.9303806576313194e-32, 0.9999999999999982, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
_T_PRE_INV = np.linalg.inv(_T_PRE)
_T_POST_INV = np.linalg.inv(_T_POST)

# Eagerly trigger symbolic precompute at import time so first
# solve() call has no derivation latency. Returns immediately
# if already cached.
_ssik_cached_derivation(
    tuple(_DH_ALPHA.tolist()),
    tuple(_DH_A.tolist()),
    tuple(_DH_D.tolist()),
    linearity_joint=2,
    apply_so3=False,
)


def _solve_algebraic(T_target):
    """Tier-2 Raghavan-Roth IK candidates for this arm.

    Bakes the DH params; routes to ssik.solvers.ikgeo._raghavan_roth.
    solve_all_ik with linearity_joint='auto' (AE-3 picks per-pose).
    """
    T = np.asarray(T_target, dtype=np.float64)
    T_dh = _T_PRE_INV @ T @ _T_POST_INV
    inner_solutions, _is_ls = _ssik_solve_all_ik(
        (_DH_ALPHA, _DH_A, _DH_D),
        T_dh,
        fk_atol=1e-9,
        dedup_atol=1e-3,
        linearity_joint="auto",
        allow_refinement=False,
        refinement_max_iters=15,
        solver_name=SOLVER_NAME,
    )
    # Map DH-frame q back to POE frame.
    return [list(inner.q - _DH_THETA_OFFSET) for inner in inner_solutions]


def _fk(q):
    """POE forward kinematics using the baked KinBody."""
    T = np.eye(4)
    for j, qi in zip(_KB.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = _rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def _wrap_to_pi(a):
    return ((a + math.pi) % (2 * math.pi)) - math.pi


def solve(
    T_target,
    *,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
):
    """Inverse kinematics. Returns ``(list[Solution], is_ls)``.

    :param T_target: 4x4 SE(3) target end-effector pose.
    :param policy: tolerance policy (FK closure + dedup tolerance).
    :param allow_refinement: opt into Newton-on-spatial-Jacobian
        polish for near-miss candidates (those whose algebraic q
        doesn't quite meet ``fk_atol``). Default off.
    :param refinement_max_iters: cap on Newton iterations per
        candidate when ``allow_refinement=True``.
    """
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
        # Newton polish using the baked KinBody's spatial Jacobian.
        refined = _lm_refine(
            q,
            _fk,
            T,
            fk_atol=fk_atol,
            max_iters=refinement_max_iters,
            jacobian_fn=lambda qq: _kinbody_jacobian(_KB, qq),
        )
        if refined is None:
            continue
        q_ref, resid_ref, iters = refined
        verified.append((q_ref, resid_ref, "lm", iters))

    # Wrap-to-pi dedup; keep lowest fk_residual on collision.
    deduped: list[tuple[np.ndarray, float, str, int]] = []
    for cand_q, cand_res, ref_used, ref_iters in verified:
        dup_idx = None
        for j, (existing_q, _, _, _) in enumerate(deduped):
            diffs = np.array([_wrap_to_pi(a - b) for a, b in zip(cand_q, existing_q)])
            if np.all(np.abs(diffs) < dedup_atol):
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
            refinement_iters=ref_iters,
            branch_id=i,
            solver_name=SOLVER_NAME,
        )
        for i, (q, residual, ref_used, ref_iters) in enumerate(deduped)
    ]
    return solutions, len(solutions) == 0


__all__ = [
    "DISPATCH_REASON",
    "EXPECTED_MS_MEDIAN",
    "FLOP_BUDGET",
    "SOLVER_NAME",
    "SOLVER_TIER",
    "solve",
]
