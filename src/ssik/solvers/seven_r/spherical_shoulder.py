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

No blind sampling -> no coverage gaps + an exact in-limits guarantee, at
machine precision (FK <= 1e-10). Exact-class membership is the reversed lock-6
sub-chain having an exact spherical wrist triple ``(3, 4, 5)`` -- Franka Panda
and FR3 qualify. Arms that are only *approximately* spherical (no exact triple,
e.g. xArm7, or an approximately-concurrent shoulder, e.g. rizon4) fail the
machine-precision gate and return ``[]``; an LM-polish path for them is a
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
from ssik.kinematics.predicates import three_consecutive_intersecting
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
_FK_ATOL = 1e-10  # bulletproof exact-class gate: only machine-precision solutions
# (feedback_bulletproof_solvers). Arms that are only *approximately* spherical
# (no exact wrist triple, e.g. xArm7) fail this and return [] -- they need the
# LM-polish path (follow-up), not silent 1e-6 solutions.
_Branches = Callable[[float], list[NDArray[np.float64]]]


def is_spherical_shoulder_7r(
    kb: KinBody, policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY
) -> bool:
    """True iff ``kb`` is an *exact* spherical-shoulder + offset-wrist 7R this
    solver handles at machine precision.

    Pure topology (no pose): lock the last joint and reverse the chain; the class
    is exactly the one whose reversed sub-chain is ``spherical_two_intersecting``
    -- an exact spherical wrist triple at ``(3, 4, 5)`` (the original spherical
    shoulder, invariant to the distal lock) with the shoulder pivot at the base
    (``p[1] = 0``). Franka Panda and FR3 qualify. Arms that are only
    *approximately* spherical (xArm7: non-concurrent wrist; rizon4: drifted
    shoulder) fail and route elsewhere -- they would not solve to machine
    precision here (an LM-polish variant is a follow-up).
    """
    if len(kb.joints) != 7:
        return False
    try:
        sub = reverse_kinematic_chain(_lock_joint(kb, _LOCK, 0.0))
    except (IndexError, ValueError):
        return False
    if three_consecutive_intersecting(sub.joints, policy) != (3, 4, 5):
        return False
    return bool(float(np.linalg.norm(sub.joints[1].T_left[:3, 3])) < policy.axis_intersect)


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


def _rot_batch(k: NDArray[np.float64], th: NDArray[np.float64]) -> NDArray[np.float64]:
    """Batched Rodrigues: per-row axis ``k`` (n,3) and angle ``th`` (n,) -> (n,3,3)."""
    n = k.shape[0]
    skew = np.zeros((n, 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -k[:, 2], k[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = k[:, 2], -k[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(th)[:, None, None]
    c = (1.0 - np.cos(th))[:, None, None]
    out: NDArray[np.float64] = np.eye(3)[None] + s * skew + c * (skew @ skew)
    return out


def _bdot(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = (a * b).sum(1)
    return out


def _sp1_batch(
    k: NDArray[np.float64], p: NDArray[np.float64], q: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.arctan2(_bdot(np.cross(k, p), q), _bdot(p, q) - _bdot(k, p) * _bdot(k, q))


def _sp4_batch(
    h: NDArray[np.float64],
    k: NDArray[np.float64],
    p: NDArray[np.float64],
    d: NDArray[np.float64],
    feas_tol: float,
    deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Batched SP4 (also serves SP3 via delegation). Returns both ``phi +- delta``
    branches (clipped, so no NaN when infeasible) and a feasibility mask; spurious
    branches at infeasible samples are filtered downstream by FK closure."""
    a = _bdot(h, p) - _bdot(k, p) * _bdot(h, k)
    b = _bdot(h, np.cross(k, p))
    cc = _bdot(k, p) * _bdot(h, k)
    r = np.hypot(a, b)
    rhs = d - cc
    ratio = np.clip(np.divide(rhs, r, out=np.zeros_like(r), where=r > 1e-12), -1.0, 1.0)
    delta = np.arccos(ratio)
    phi = np.arctan2(b, a)
    feas = (np.abs(rhs) - r <= feas_tol) & (r * r >= deg * deg)
    return np.stack([phi + delta, phi - delta], axis=1), feas


def _sp2_batch(
    k1: NDArray[np.float64],
    k2: NDArray[np.float64],
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    feas_tol: float,
    deg: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Batched SP2. Returns both ``(theta1, theta2)`` branches and feasibility."""
    c = _bdot(k1, k2)
    s_sq = 1.0 - c * c
    safe = np.where(s_sq > deg, s_sq, 1.0)
    d1, d2 = _bdot(k1, p), _bdot(k2, q)
    alpha = (d1 - c * d2) / safe
    beta = (d2 - c * d1) / safe
    kxk = np.cross(k1, k2)
    pp, qq = _bdot(p, p), _bdot(q, q)
    gss = 0.5 * (pp + qq) - alpha * alpha - beta * beta - 2.0 * alpha * beta * c
    feas = (np.abs(pp - qq) <= feas_tol) & (gss >= -feas_tol) & (s_sq >= deg)
    gamma = np.sqrt(np.maximum(gss, 0.0) / safe)
    base = alpha[:, None] * k1 + beta[:, None] * k2
    za = base + gamma[:, None] * kxk
    zb = base - gamma[:, None] * kxk
    t1 = np.stack([_sp1_batch(k1, p, za), _sp1_batch(k1, p, zb)], axis=1)
    t2 = np.stack([_sp1_batch(k2, q, za), _sp1_batch(k2, q, zb)], axis=1)
    return t1, t2, feas


def _closed_branches_grid(
    coef: NDArray[np.float64],
    t_rev: NDArray[np.float64],
    q6_grid: NDArray[np.float64],
    policy: TolerancePolicy,
) -> list[list[NDArray[np.float64]]]:
    """Vectorised :func:`_closed_branches` over a q6 grid: loop the <=8 branch
    slots (fixed), vectorise each SP stage over the grid. Returns, per grid point,
    the list of feasible q0..q6 branches. Validated to match the scalar oracle's
    FK-closing branch set exactly (0 missed) at ~7x the speed of the scalar loop."""
    n = q6_grid.shape[0]
    ft, dg = policy.subproblem_feasibility, policy.subproblem_degeneracy
    basis = np.stack([np.cos(q6_grid), np.sin(q6_grid), np.ones(n)], axis=1)
    v = basis @ coef
    axes = v[:, 0:18].reshape(n, 6, 3)
    axes = axes / np.linalg.norm(axes, axis=2, keepdims=True)
    a0, a1, a2, a3, a4, a5 = (axes[:, i] for i in range(6))
    our_p = v[:, 18:36].reshape(n, 6, 3)
    tool = v[:, 36:39]
    r_home = v[:, 39:48].reshape(n, 3, 3)
    p0, p2 = our_p[:, 0], our_p[:, 2]
    p3 = our_p[:, 3] + our_p[:, 4] + our_p[:, 5]
    r_06 = np.einsum("ij,nkj->nik", t_rev[:3, :3], r_home)
    p_16 = t_rev[:3, 3] - np.einsum("nij,nj->ni", r_06, tool) - p0

    d3 = 0.5 * (_bdot(p3, p3) + _bdot(p2, p2) - _bdot(p_16, p_16))
    q3_both, feas3 = _sp4_batch(-p2, a2, p3, d3, ft, dg)  # SP3 via SP4

    slots: list[list[NDArray[np.float64]]] = [[] for _ in range(n)]
    for e in range(2):
        q3 = q3_both[:, e]
        sp2_arg = p2 + np.einsum("nij,nj->ni", _rot_batch(a2, q3), p3)
        t1_both, t2_both, feas2 = _sp2_batch(-a0, a1, p_16, sp2_arg, ft, dg)
        for sh in range(2):
            q1, q2 = t1_both[:, sh], t2_both[:, sh]
            r36 = _rot_batch(-a2, q3) @ _rot_batch(-a1, q2) @ _rot_batch(-a0, q1) @ r_06
            d_sp4 = np.einsum("ni,nij,nj->n", a3, r36, a5)
            q5_both, feas4 = _sp4_batch(a3, a4, a5, d_sp4, ft, dg)
            valid = feas3 & feas2 & feas4
            for w in range(2):
                q5 = q5_both[:, w]
                q4 = _sp1_batch(
                    a3,
                    np.einsum("nij,nj->ni", _rot_batch(a4, q5), a5),
                    np.einsum("nij,nj->ni", r36, a5),
                )
                q6i = _sp1_batch(
                    -a5,
                    np.einsum("nij,nj->ni", _rot_batch(-a4, q5), a3),
                    np.einsum("nji,nj->ni", r36, a3),
                )
                # map_reversed_q(flip [q1..q6i]) = [q6i,q5,q4,q3,q2,q1] + q6
                full = np.concatenate(
                    [np.stack([q6i, q5, q4, q3, q2, q1], axis=1), q6_grid[:, None]], axis=1
                )
                for i in np.nonzero(valid)[0]:
                    slots[i].append(full[i])
    return slots


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


def _track_branches(
    per: list[list[NDArray[np.float64]]], grid: NDArray[np.float64]
) -> list[NDArray[np.float64]]:
    """Link discrete branches (``per[k]`` = branch list at ``grid[k]``) into
    continuous curves by greedy nearest-neighbour. Returns ``(len(grid), 7)``
    arrays (NaN where absent)."""
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
    coef: NDArray[np.float64],
    t_rev: NDArray[np.float64],
    kb: KinBody,
    T: NDArray[np.float64],
    a: float,
    b: float,
    limits: list[tuple[float, float]],
    policy: TolerancePolicy,
) -> list[NDArray[np.float64]]:
    """In-limits q vectors for one reachable interval: track each branch, take
    its exact in-limits q6 arcs, emit the arc-centre solution wrapped to limits."""
    fk_atol = _FK_ATOL
    grid = np.linspace(a, b, _TRACK_GRID)
    per = _closed_branches_grid(coef, t_rev, grid, policy)  # one vectorised pass
    out: list[NDArray[np.float64]] = []
    for curve in _track_branches(per, grid):
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
            for q in _closed_branches(coef, t_rev, float(q6c), policy):
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


_SAMPLE_GRID = 16  # default-path samples per reachable interval (like the sweep)


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
    """Analytic IK for the spherical-shoulder + offset-wrist 7R class.

    ``respect_limits=True`` returns the exact in-limits solution set (no coverage
    gaps -- see :func:`resolve_in_limits`). The default samples each reachable q6
    interval for a representative solution set. ``allow_refinement`` /
    ``refinement_max_iters`` are accepted for interface parity and unused (the
    closed-form solutions are already machine-precision). Returns
    ``(solutions, is_ls)`` with ``is_ls = True`` iff empty.
    """
    if len(kb.joints) != 7:
        return [], True
    if respect_limits:
        sols = resolve_in_limits(kb, T_target, policy, max_solutions=max_solutions)
        return sols, len(sols) == 0

    T = np.asarray(T_target, dtype=np.float64)
    coef = _bake(kb)
    t_rev = _se3_inv(T)
    lo, hi = _joint_limits(kb)[_LOCK]
    seen: set[tuple[float, ...]] = set()
    out: list[Solution] = []
    for a, b in _reachable_intervals(coef, t_rev, lo, hi):
        grid = np.linspace(a, b, _SAMPLE_GRID)
        per = _closed_branches_grid(coef, t_rev, grid, policy)
        for branch_list in per:
            for q in branch_list:
                residual = float(np.linalg.norm(poe_forward_kinematics(kb, q) - T))
                if residual > _FK_ATOL:
                    continue
                key = tuple(np.round(q, _MERGE_KEY))
                if key in seen:
                    continue
                seen.add(key)
                out.append(Solution(q=q, fk_residual=residual, refinement_used="none"))
                if max_solutions is not None and len(out) >= max_solutions:
                    return out, False
    return out, len(out) == 0


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

    coef = _bake(kb)
    t_rev = _se3_inv(T)

    seen: set[tuple[float, ...]] = set()
    out: list[Solution] = []
    for a, b in _reachable_intervals(coef, t_rev, lo, hi):
        for q in _solutions_in_interval(coef, t_rev, kb, T, a, b, limits, policy):
            key = tuple(np.round(q, _MERGE_KEY))
            if key in seen:
                continue
            seen.add(key)
            residual = float(np.linalg.norm(poe_forward_kinematics(kb, q) - T))
            out.append(Solution(q=q, fk_residual=residual, refinement_used="none"))
            if max_solutions is not None and len(out) >= max_solutions:
                return out
    return out
