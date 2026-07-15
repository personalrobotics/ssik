"""Closed-form 7R IK for the spherical-shoulder + offset-wrist class (#373).

Franka Panda, FR3, and uFactory xArm7 share one kinematic structure: joints
(0, 1, 2) are concurrent (a spherical shoulder) but the wrist is offset (joints
4, 5 intersect, joint 6 is offset ~50-59 mm). They are *not* SRS -- the wrist
offset couples position and orientation -- so the SRS solver does not apply, and
``jointlock.seven_r`` handles them only by a blind 16-sample sweep of a redundant
joint (slow, and it drops poses whose reachable-redundancy interval is narrow).

This solver treats the **last joint q6 as the redundancy** (He & Liu 2021). With
q6 fixed, the lock-6 sub-chain is a tier-0 spherical-wrist 6R, so ``q0..q5(q6)``
is closed-form. The reversed sub-chain geometry is affine in
``{cos q6, sin q6, 1}`` (exact), so we bake those coefficients once and evaluate
``q_i(q6)`` by the ``spherical_two_intersecting`` SP recipe (SP3->SP2->SP4->SP1x2)
at any q6 -- no KinBody rebuild, no verify. The redundancy is then resolved
*exactly*:

1. **Reachability.** The elbow SP3 constraint closes iff a smooth margin
   ``m(q6) >= 0``; it is a *necessary* gate, so ``{reachable} subset {m >= 0}``
   -- a guaranteed analytic bracket. Refine the true reachable interval within it.
2. **Joint limits.** Within a reachable interval, per IK branch every joint is a
   smooth ``q_i(q6)``; :func:`~ssik.solvers.seven_r._feasible_param.feasible_arcs_bounded`
   gives the exact in-limits q6 sub-arcs (bounded, non-periodic analogue of the
   SRS swivel resolution, #372/#359). The cheap closed-form eval affords a fine
   grid, so even razor-thin in-limits arcs are bracketed.

No blind sampling -> no coverage gaps + an exact in-limits guarantee. Covers
franka/fr3/xarm7; the approximately-spherical shoulder path (rizon4) is a
follow-up.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.kinematics._scalar3 import _se3_inv
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.reverse import map_reversed_q, reverse_kinematic_chain
from ssik.solvers.jointlock.seven_r import _lock_joint
from ssik.solvers.seven_r._feasible_param import feasible_arcs_bounded, merge, to_limits
from ssik.subproblems import sp1, sp2, sp3, sp4
from ssik.subproblems._rotation import rotation_matrix as _rot

if TYPE_CHECKING:  # pragma: no cover
    from ssik._kinbody import KinBody

_LOCK = 6  # redundancy joint (last joint) for this class
_SWEPT = (0, 1, 2, 3, 4, 5)  # non-locked joints constrained by feasible_arcs
_BRACKET_GRID = 90  # SP3-margin sign-change resolution
_TRACK_GRID = 180  # per-interval branch-tracking / feasible-arc resolution
_MERGE_KEY = 6  # dedup rounding (decimals) on the full q vector
_BAKE_Q6 = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])  # {cos,sin,1} basis samples
_Branches = Callable[[float], list[NDArray[np.float64]]]


# --- baked closed-form q_i(q6) ------------------------------------------------


def _bake(kb: KinBody) -> NDArray[np.float64]:
    """Coefficients (3, 48) of the reversed lock-6 sub-chain geometry as an affine
    function of ``[cos q6, sin q6, 1]`` -- axes (18) + offsets (18) + tool (3) +
    r_home (9). Exact: the geometry is affine in ``{cos q6, sin q6}`` by the
    ``R_lock`` similarity structure (verified to ~1e-15)."""

    def geom(q6: float) -> NDArray[np.float64]:
        sub = reverse_kinematic_chain(_lock_joint(kb, _LOCK, float(q6)))
        axes = np.array([j.axis for j in sub.joints])
        our_p = np.array([j.T_left[:3, 3] for j in sub.joints])
        tool = sub.joints[-1].T_right[:3, 3]
        r_home = sub.joints[-1].T_right[:3, :3]
        return np.concatenate([axes.ravel(), our_p.ravel(), tool, r_home.ravel()])

    basis = np.stack([np.cos(_BAKE_Q6), np.sin(_BAKE_Q6), np.ones(3)], axis=1)
    g = np.array([geom(q) for q in _BAKE_Q6])
    coef: NDArray[np.float64] = np.linalg.solve(basis, g)
    return coef


def _eval_geom(
    coef: NDArray[np.float64], q6: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    v = np.array([np.cos(q6), np.sin(q6), 1.0]) @ coef
    axes = v[0:18].reshape(6, 3)
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    return axes, v[18:36].reshape(6, 3), v[36:39], v[39:48].reshape(3, 3)


def _closed_branches(
    coef: NDArray[np.float64],
    t_rev: NDArray[np.float64],
    q6: float,
    policy: TolerancePolicy,
) -> list[NDArray[np.float64]]:
    """All q0..q6 IK branches at a fixed q6, closed-form on baked geometry
    (reversed spherical_two_intersecting recipe, mapped back + q6 appended)."""
    axes, our_p, tool, r_home = _eval_geom(coef, q6)
    p0, p2 = our_p[0], our_p[2]
    p3 = our_p[3] + our_p[4] + our_p[5]
    r_06 = t_rev[:3, :3] @ r_home.T
    p_16 = t_rev[:3, 3] - r_06 @ tool - p0
    out: list[NDArray[np.float64]] = []
    t3, _ = sp3.solve(axes[2], p3, -p2, float(np.linalg.norm(p_16)), policy)
    for q3 in t3:
        t12, _ = sp2.solve(-axes[0], axes[1], p_16, p2 + _rot(axes[2], q3) @ p3, policy)
        for q1, q2 in t12:
            r_36 = _rot(-axes[2], q3) @ _rot(-axes[1], q2) @ _rot(-axes[0], q1) @ r_06
            t5, _ = sp4.solve(axes[3], axes[4], axes[5], float(axes[3] @ r_36 @ axes[5]), policy)
            for q5 in t5:
                q4, _ = sp1.solve(axes[3], _rot(axes[4], q5) @ axes[5], r_36 @ axes[5], policy)
                q6i, _ = sp1.solve(-axes[5], _rot(-axes[4], q5) @ axes[3], r_36.T @ axes[3], policy)
                q_sub = map_reversed_q(np.array([q1, q2, q3, q4, q5, q6i]))
                out.append(np.concatenate([q_sub, [q6]]))
    return out


def _sp3_reach_margins(
    coef: NDArray[np.float64], t_rev: NDArray[np.float64], q6_grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Vectorised elbow-solvability margin over a q6 grid: ``>= 0`` on the
    reachable set (a *necessary* condition, so a superset bracket -- no reachable
    q6 is missed). Batches the ``{cos,sin,1} @ coef`` geometry so the reachability
    bracket costs one array pass, no per-sample sub-chain work."""
    n = q6_grid.shape[0]
    basis = np.stack([np.cos(q6_grid), np.sin(q6_grid), np.ones(n)], axis=1)  # (n,3)
    v = basis @ coef  # (n,48)
    axes = v[:, 0:18].reshape(n, 6, 3)
    axes = axes / np.linalg.norm(axes, axis=2, keepdims=True)
    our_p = v[:, 18:36].reshape(n, 6, 3)
    tool = v[:, 36:39]
    r_home = v[:, 39:48].reshape(n, 3, 3)
    p0, p2 = our_p[:, 0], our_p[:, 2]
    p3 = our_p[:, 3] + our_p[:, 4] + our_p[:, 5]
    r_06 = np.einsum("ij,nkj->nik", t_rev[:3, :3], r_home)  # t_rev_R @ r_home.T
    p_16 = t_rev[:3, 3] - np.einsum("nij,nj->ni", r_06, tool) - p0  # (n,3)
    k, pp, qq = axes[:, 2], p3, -p2
    target = 0.5 * ((pp * pp).sum(1) + (qq * qq).sum(1) - (p_16 * p_16).sum(1))
    center = (qq * k).sum(1) * (pp * k).sum(1)
    qperp = np.linalg.norm(qq - k * (qq * k).sum(1, keepdims=True), axis=1)
    pperp = np.linalg.norm(pp - k * (pp * k).sum(1, keepdims=True), axis=1)
    out: NDArray[np.float64] = qperp * pperp - np.abs(target - center)
    return out


# --- redundancy resolution ----------------------------------------------------


def _reachable_intervals(
    coef: NDArray[np.float64], t_rev: NDArray[np.float64], lo: float, hi: float
) -> list[tuple[float, float]]:
    """Reachable q6 sub-intervals of ``[lo, hi]`` -- the SP3-margin >= 0 brackets
    (a guaranteed superset of the true reachable set), padded by one grid step so
    the true boundary is contained. Branch-tracking + feasible_arcs restrict to
    the truly solvable, in-limits region within, so no full-solve boundary
    refinement is needed here."""
    grid = np.linspace(lo, hi, _BRACKET_GRID)
    m = _sp3_reach_margins(coef, t_rev, grid) >= 0.0
    out: list[tuple[float, float]] = []
    k = 0
    while k < _BRACKET_GRID:
        if m[k]:
            j = k
            while j + 1 < _BRACKET_GRID and m[j + 1]:
                j += 1
            a = float(grid[max(k - 1, 0)])  # pad by one step: bracket contains the true edge
            b = float(grid[min(j + 1, _BRACKET_GRID - 1)])
            out.append((a, b))
            k = j + 1
        else:
            k += 1
    return merge(out)


def _track_branches(branches: _Branches, grid: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """Link discrete branches across ``grid`` into continuous curves by greedy
    nearest-neighbour. Returns ``(len(grid), 7)`` arrays (NaN where absent)."""
    per = [branches(float(g)) for g in grid]
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
                if j in used[k] or d[j] > 0.4:  # continuity break
                    break
                curve[k] = per[k][j]
                used[k].add(j)
                prev = per[k][j]
            if np.count_nonzero(~np.isnan(curve[:, 0])) >= 4:
                curves.append(curve)
    return curves


def _solutions_in_interval(
    branches: _Branches,
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
    for curve in _track_branches(branches, grid):
        valid = ~np.isnan(curve[:, 0])
        g = grid[valid]
        qc = curve[valid]
        if g.shape[0] < 4:
            continue

        def q_scalar(
            t: float, g: NDArray[np.float64] = g, qc: NDArray[np.float64] = qc
        ) -> NDArray[np.float64]:
            # Smooth branch value at arbitrary q6 by interpolating the (dense,
            # continuous) tracked curve -- so feasible_arcs bisects sub-grid
            # without re-solving. The final arc-centre solution is FK-verified.
            return np.array([np.interp(t, g, qc[:, i]) for i in range(7)])

        for u, w in feasible_arcs_bounded(q_scalar, qc, _SWEPT, limits, g):
            q6c = 0.5 * (u + w)
            for q in branches(float(q6c)):
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

    coef = _bake(kb)
    t_rev = _se3_inv(T)

    def branches(q6: float) -> list[NDArray[np.float64]]:
        return _closed_branches(coef, t_rev, q6, policy)

    seen: set[tuple[float, ...]] = set()
    out: list[Solution] = []
    for a, b in _reachable_intervals(coef, t_rev, lo, hi):
        for q in _solutions_in_interval(branches, kb, T, a, b, limits, fk_atol):
            key = tuple(np.round(q, _MERGE_KEY))
            if key in seen:
                continue
            seen.add(key)
            residual = float(np.linalg.norm(poe_forward_kinematics(kb, q) - T))
            out.append(Solution(q=q, fk_residual=residual, refinement_used="none"))
            if max_solutions is not None and len(out) >= max_solutions:
                return out
    return out
