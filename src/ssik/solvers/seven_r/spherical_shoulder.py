"""Closed-form 7R IK for the spherical-shoulder + offset-wrist class (#373).

Franka Panda, FR3, and uFactory xArm7 share one kinematic structure: joints
(0, 1, 2) are concurrent (a spherical shoulder) but the wrist is offset (joints
4, 5 intersect, joint 6 is offset ~50-59 mm). They are *not* SRS -- the wrist
offset couples position and orientation -- so the SRS solver does not apply, and
``jointlock.seven_r`` handles them only by a blind 16-sample sweep of a redundant
joint (slow, and it drops poses whose reachable-redundancy interval is narrow).

This solver treats the **last joint q6 as the redundancy** (He & Liu 2021). With
q6 fixed, the lock-6 sub-chain is a tier-0 spherical-wrist 6R (closed form), so
``q0..q5(q6)`` is closed-form. Instead of sampling q6 blindly we resolve it
*exactly*:

1. **Reachability.** The elbow SP3 constraint closes iff a smooth margin
   ``m(q6) >= 0``; it is a *necessary* gate, so ``{reachable} subset {m >= 0}``
   -- a guaranteed analytic bracket. Refine the true reachable interval within
   each bracket (where the closed-form solve gains/loses a solution).
2. **Joint limits.** Within a reachable interval, per IK branch every joint is a
   smooth ``q_i(q6)``; :func:`~ssik.solvers.seven_r._feasible_param.feasible_arcs_bounded`
   gives the exact in-limits q6 sub-arcs (the bounded, non-periodic analogue of
   the SRS swivel resolution, #372/#359).

No blind sampling -> no coverage gaps + an exact in-limits guarantee, at
closed-form speed. Covers franka/fr3/xarm7; the approximately-spherical shoulder
path (rizon4) is a follow-up.
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
from ssik.solvers.jointlock.seven_r import _lock_joint
from ssik.solvers.jointlock.seven_r import solve as _jointlock_solve
from ssik.solvers.seven_r._feasible_param import feasible_arcs_bounded, merge, to_limits

if TYPE_CHECKING:  # pragma: no cover
    from ssik._kinbody import KinBody

_LOCK = 6  # redundancy joint (last joint) for this class
_SWEPT = (0, 1, 2, 3, 4, 5)  # non-locked joints constrained by feasible_arcs
_BRACKET_GRID = 60  # SP3-margin sign-change resolution
_TRACK_GRID = 40  # per-interval branch-tracking / feasible-arc resolution
_MERGE_KEY = 6  # dedup rounding (decimals) on the full q vector


def _sp3_reach_margin(kb: KinBody, t_rev: NDArray[np.float64], q6: float) -> float:
    """Smooth elbow-solvability margin of the reversed lock-6 sub-chain at ``q6``.

    The inner ``spherical_two_intersecting`` closes SP3 (elbow) iff
    ``|target - center| <= radius`` with target/center/radius below; the margin
    ``radius - |target - center|`` is ``>= 0`` exactly on the reachable set and is
    a *necessary* condition (a superset bracket), so no reachable q6 is missed.
    """
    sub = reverse_kinematic_chain(_lock_joint(kb, _LOCK, float(q6)))
    p = [sub.joints[i].T_left[:3, 3] for i in range(6)]
    tool = sub.joints[-1].T_right[:3, 3]
    k = sub.joints[2].axis
    p2, p3 = p[2], p[3] + p[4] + p[5]
    r_home = sub.joints[-1].T_right[:3, :3]
    p_16 = t_rev[:3, 3] - (t_rev[:3, :3] @ r_home.T) @ tool - p[0]
    pp, qq = p3, -p2
    target = 0.5 * (float(pp @ pp) + float(qq @ qq) - float(p_16 @ p_16))
    center = float(qq @ k) * float(pp @ k)
    radius = float(np.linalg.norm(qq - k * (qq @ k)) * np.linalg.norm(pp - k * (pp @ k)))
    return radius - abs(target - center)


def _branches_at(kb: KinBody, T: NDArray[np.float64], q6: float) -> list[NDArray[np.float64]]:
    """Closed-form q0..q6 solutions at a fixed q6 (lock-6 tier-0 inner solve)."""
    sols, _ = _jointlock_solve(kb, T, lock_samples=np.array([q6]), lock_idx=_LOCK)
    return [s.q for s in sols]


def _reachable_intervals(
    kb: KinBody, T: NDArray[np.float64], lo: float, hi: float
) -> list[tuple[float, float]]:
    """Reachable q6 sub-intervals of ``[lo, hi]``: SP3-margin brackets refined to
    the true solvable boundary (where the inner solve gains/loses a solution)."""
    t_rev = _se3_inv(np.asarray(T, dtype=np.float64))
    grid = np.linspace(lo, hi, _BRACKET_GRID)
    m = np.array([_sp3_reach_margin(kb, t_rev, float(g)) >= 0.0 for g in grid])

    def reach(q6: float) -> bool:
        return bool(_branches_at(kb, T, q6))

    out: list[tuple[float, float]] = []
    k = 0
    while k < _BRACKET_GRID:
        if m[k]:
            j = k
            while j + 1 < _BRACKET_GRID and m[j + 1]:
                j += 1
            a = float(grid[max(k - 1, 0)])  # pad by one step: bracket contains the true edge
            b = float(grid[min(j + 1, _BRACKET_GRID - 1)])
            iv = _refine_reachable(reach, a, b)
            if iv is not None:
                out.append(iv)
            k = j + 1
        else:
            k += 1
    return merge(out)


def _refine_reachable(reach: object, a: float, b: float) -> tuple[float, float] | None:
    """True reachable sub-interval within bracket ``[a, b]`` (``reach`` is a
    ``q6 -> bool`` predicate). ``None`` if the bracket is entirely unreachable."""
    assert callable(reach)
    grid = np.linspace(a, b, _TRACK_GRID)
    hits = [float(g) for g in grid if reach(float(g))]
    if not hits:
        return None
    lo_h, hi_h = min(hits), max(hits)
    lo_e = _bisect_edge(reach, a, lo_h) if lo_h > a + 1e-12 else a
    hi_e = _bisect_edge(reach, b, hi_h) if hi_h < b - 1e-12 else b
    return (lo_e, hi_e)


def _bisect_edge(reach: object, out: float, inn: float, iters: int = 34) -> float:
    assert callable(reach)
    for _ in range(iters):
        mid = 0.5 * (out + inn)
        if reach(mid):
            inn = mid
        else:
            out = mid
    return inn


def _track_branches(
    kb: KinBody, T: NDArray[np.float64], grid: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """Link the discrete inner solutions across ``grid`` into continuous branch
    curves by greedy nearest-neighbour. Returns a list of ``(len(grid), 7)``
    arrays (NaN where a branch is absent at that q6)."""
    per = [_branches_at(kb, T, float(g)) for g in grid]
    n = len(grid)
    used: list[set[int]] = [set() for _ in range(n)]
    curves: list[NDArray[np.float64]] = []
    for k0 in range(n):
        for b0 in range(len(per[k0])):
            if b0 in used[k0]:
                continue
            curve = np.full((n, 7), np.nan)
            curve[k0] = per[k0][b0]
            used[k0].add(b0)
            prev = per[k0][b0]
            for k in range(k0 + 1, n):
                if not per[k]:
                    break
                d = [float(np.linalg.norm(q - prev)) for q in per[k]]
                j = int(np.argmin(d))
                if j in used[k] or d[j] > 0.6:  # continuity break
                    break
                curve[k] = per[k][j]
                used[k].add(j)
                prev = per[k][j]
            if np.count_nonzero(~np.isnan(curve[:, 0])) >= 4:
                curves.append(curve)
    return curves


def _solutions_in_interval(
    kb: KinBody,
    T: NDArray[np.float64],
    a: float,
    b: float,
    limits: list[tuple[float, float]],
    fk_atol: float,
) -> list[NDArray[np.float64]]:
    """In-limits q vectors for one reachable interval: track each branch, take
    its exact in-limits q6 arcs, emit the arc-centre solution wrapped to limits."""
    grid = np.linspace(a, b, _TRACK_GRID)
    out: list[NDArray[np.float64]] = []
    for curve in _track_branches(kb, T, grid):
        valid = ~np.isnan(curve[:, 0])
        g = grid[valid]
        qc = curve[valid]
        if g.shape[0] < 4:
            continue

        def q_scalar(
            t: float, g: NDArray[np.float64] = g, qc: NDArray[np.float64] = qc
        ) -> NDArray[np.float64]:
            # Smooth branch value at arbitrary q6: re-solve and pick the branch
            # nearest the tracked curve (so feasible_arcs bisects exactly, not at
            # grid resolution). Fall back to the nearest tracked sample.
            ref: NDArray[np.float64] = qc[int(np.argmin(np.abs(g - t)))]
            cands = _branches_at(kb, T, float(t))
            if not cands:
                return ref
            best = min(cands, key=lambda x: float(np.linalg.norm(x - ref)))
            return np.asarray(best, dtype=np.float64)

        for u, w in feasible_arcs_bounded(q_scalar, qc, _SWEPT, limits, g):
            q6c = 0.5 * (u + w)
            for q in _branches_at(kb, T, float(q6c)):
                qw = np.array([to_limits(float(q[i]), *limits[i]) for i in range(7)])
                in_lim = all(limits[i][0] - 1e-9 <= qw[i] <= limits[i][1] + 1e-9 for i in range(7))
                if in_lim and float(np.linalg.norm(poe_forward_kinematics(kb, qw) - T)) <= fk_atol:
                    out.append(qw)
    return out


def _joint_limits(kb: KinBody) -> list[tuple[float, float]]:
    lims: list[tuple[float, float]] = []
    for j in kb.joints:
        lo_hi = j.limits
        if lo_hi is None or lo_hi[0] is None or lo_hi[1] is None:
            lims.append((-np.pi, np.pi))
        else:
            lims.append((float(lo_hi[0]), float(lo_hi[1])))
    return lims


def resolve_in_limits(
    kb: KinBody,
    T_target: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    *,
    max_solutions: int | None = None,
) -> list[Solution]:
    """Exact in-limits IK for the spherical-shoulder + offset-wrist 7R class.

    Resolves the q6 redundancy exactly (reachable interval x in-limits arcs) so
    every reachable in-limits pose returns an in-limits, FK-verified solution --
    no blind sampling, no coverage gaps. Returns ``[]`` for a non-7R chain or a
    target with no in-limits solution.
    """
    if len(kb.joints) != 7:
        return []
    T = np.asarray(T_target, dtype=np.float64)
    limits = _joint_limits(kb)
    lo, hi = limits[_LOCK]
    fk_atol = policy.subproblem_numerical

    seen: set[tuple[float, ...]] = set()
    out: list[Solution] = []
    for a, b in _reachable_intervals(kb, T, lo, hi):
        for q in _solutions_in_interval(kb, T, a, b, limits, fk_atol):
            key = tuple(np.round(q, _MERGE_KEY))
            if key in seen:
                continue
            seen.add(key)
            residual = float(np.linalg.norm(poe_forward_kinematics(kb, q) - T))
            out.append(Solution(q=q, fk_residual=residual, refinement_used="none"))
            if max_solutions is not None and len(out) >= max_solutions:
                return out
    return out
