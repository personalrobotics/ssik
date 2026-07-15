"""Approximately-spherical-shoulder 7R IK: closed-form seeds + LM polish (#373).

Some arms are *nearly* the exact spherical-shoulder + offset-wrist class of
:mod:`ssik.solvers.seven_r.spherical_shoulder` -- their reversed lock-6 wrist
triple is concurrent to within a small drift (uFactory xArm7: ~0, but with a
~1e-8 residual the exact 1e-10 gate rejects). The closed-form q_i(q6) recipe
still produces excellent seeds; a batched Levenberg-Marquardt polish against the
true URDF FK recovers machine precision.

Mirrors :mod:`ssik.solvers.seven_r.srs_polished` (approximate-SRS): run the
analytical machinery on the near-spherical geometry, LM-polish every candidate,
keep the ones that reach machine-precision FK (and stay in-limits for the
``respect_limits`` path), cluster-merge duplicates.

Refusal gate ``_APPROX_MAX_DRIFT_M``: arms with too large a wrist drift are
rejected -- the reachable-q6 interval computed on the approximate geometry no
longer matches the true reachability, so coverage (not just precision) degrades.
xArm7 passes; Flexiv Rizon 4 (32 mm drift) does not and stays on jointlock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics._scalar3 import _se3_inv
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.reverse import reverse_kinematic_chain
from ssik.refinement import dedup_by_wrap_close, kinbody_jacobian, lm_refine_batch
from ssik.solvers.jointlock.seven_r import _lock_joint
from ssik.solvers.seven_r.spherical_shoulder import (
    _LOCK,
    _SAMPLE_GRID,
    _bake,
    _closed_branches_grid,
    _joint_limits,
    _reachable_intervals,
    is_spherical_shoulder_7r,
)

if TYPE_CHECKING:  # pragma: no cover
    from ssik._kinbody import KinBody

_APPROX_MAX_DRIFT_M = 0.005  # reversed lock-6 wrist concurrency drift refusal gate
_POLISH_FK_ATOL = 1e-10  # accept a polished candidate at machine precision
_POLISH_MAX_ITERS = 30


def _wrist_drift(kb: KinBody) -> float | None:
    """Reversed lock-6 wrist-triple (3,4,5) concurrency residual (m), or None if
    not 7R. Small => the exact spherical recipe applies up to a polish."""
    if len(kb.joints) != 7:
        return None
    try:
        sub = reverse_kinematic_chain(_lock_joint(kb, _LOCK, 0.0))
    except (IndexError, ValueError):
        return None
    t = np.eye(4)
    pts, axs = [], []
    for j in sub.joints:
        t = t @ j.T_left
        pts.append(t[:3, 3].copy())
        axs.append(t[:3, :3] @ j.axis)
        t = t @ j.T_right
    axs = [a / float(np.linalg.norm(a)) for a in axs]
    mat = np.zeros((3, 3))
    vec = np.zeros(3)
    for i in (3, 4, 5):
        proj = np.eye(3) - np.outer(axs[i], axs[i])
        mat += proj
        vec += proj @ pts[i]
    common = np.linalg.solve(mat, vec)
    return max(float(np.linalg.norm(np.cross(common - pts[i], axs[i]))) for i in (3, 4, 5))


def is_approximately_spherical_shoulder_7r(
    kb: KinBody,
    max_drift_m: float = _APPROX_MAX_DRIFT_M,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> bool:
    """True iff ``kb`` is *approximately* the spherical-shoulder class -- close
    enough that the closed-form seeds + LM polish reach machine precision with no
    coverage loss. Excludes the exact class (handled by the un-polished solver)
    and arms whose wrist drift exceeds ``max_drift_m``."""
    if is_spherical_shoulder_7r(kb, policy):
        return False  # exact class: no polish needed
    drift = _wrist_drift(kb)
    return drift is not None and drift <= max_drift_m


def _polished(
    kb: KinBody,
    T: NDArray[np.float64],
    policy: TolerancePolicy,
    *,
    respect_limits: bool,
    limits: list[tuple[float, float]],
) -> list[Solution]:
    """Closed-form candidates over the reachable q6 intervals, LM-polished to
    machine precision, filtered (in-limits for ``respect_limits``), cluster-merged."""
    coef = _bake(kb)
    t_rev = _se3_inv(T)
    cand: list[NDArray[np.float64]] = []
    for a, b in _reachable_intervals(coef, t_rev, -np.pi, np.pi):
        grid = np.linspace(a, b, _SAMPLE_GRID)
        for branch_list in _closed_branches_grid(coef, t_rev, grid, policy):
            cand.extend(branch_list)
    if not cand:
        return []

    def _fk(q: NDArray[np.float64]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = poe_forward_kinematics(kb, q)
        return out

    def _jac(q: NDArray[np.float64]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = kinbody_jacobian(kb, q)
        return out

    q_pol, res, _iters = lm_refine_batch(
        np.asarray(cand), _fk, _jac, T, max_iters=_POLISH_MAX_ITERS
    )
    out: list[Solution] = []
    for i in range(q_pol.shape[0]):
        if res[i] > _POLISH_FK_ATOL:
            continue
        if respect_limits and not all(
            limits[j][0] - 1e-9 <= q_pol[i][j] <= limits[j][1] + 1e-9 for j in range(7)
        ):
            continue
        out.append(Solution(q=q_pol[i], fk_residual=float(res[i]), refinement_used="lm"))
    return dedup_by_wrap_close(out, policy.subproblem_dedup)


def solve(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    max_solutions: int | None = None,
    q_seed: NDArray[np.float64] | None = None,
    respect_limits: bool = False,
) -> tuple[list[Solution], bool]:
    """Analytic IK for the approximately-spherical-shoulder 7R class (closed-form
    seeds + LM polish). Same interface as
    :func:`ssik.solvers.seven_r.spherical_shoulder.solve`."""
    if len(kb.joints) != 7:
        return [], True
    T = np.asarray(T_target, dtype=np.float64)
    limits = _joint_limits(kb)
    out = _polished(kb, T, policy, respect_limits=respect_limits, limits=limits)
    if q_seed is not None:
        seed = np.asarray(q_seed, dtype=np.float64)
        out.sort(key=lambda s: float(np.max(np.abs((s.q - seed + np.pi) % (2 * np.pi) - np.pi))))
    if max_solutions is not None:
        out = out[:max_solutions]
    return out, len(out) == 0


def resolve_in_limits(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    max_solutions: int | None = None,
) -> list[Solution]:
    """Exact in-limits IK for the approximately-spherical class (polished)."""
    sols, _ = solve(kb, T_target, policy, max_solutions=max_solutions, respect_limits=True)
    return sols
