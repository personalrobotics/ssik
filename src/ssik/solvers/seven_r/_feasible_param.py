"""Generic feasible-interval core for 1-parameter redundancy families (#148).

A 7R arm with one redundant DOF has, for each fixed IK branch, a smooth
1-parameter joint family ``q(t)`` (``t`` in ``[-pi, pi)``): the SRS elbow swivel
(:mod:`._swivel_limits`, #359), or -- the motivation for lifting this out -- a
locked-joint redundancy for non-SRS 7R (#148). This module computes the *exact*
set of ``t`` with every joint inside its limits, independent of how ``q(t)`` is
generated.

Per joint the feasibility test is

    phi_i(t) = cos(q_i(t) - c_i) - cos(h_i) >= 0

with ``c_i`` / ``h_i`` the limit centre / half-width. The ``cos`` removes the
``atan2`` branch-wrap discontinuity, so ``phi_i`` is smooth and its sign-zeros
are the exact arc boundaries -- bracketed on a precomputed grid, refined by
bisection. The per-branch feasible set is the intersection of the swept joints'
arcs (a fixed joint, constant in ``t``, is checked once by the caller).

Dependency-free on purpose: the SRS import chain stays scipy-free to keep
iiwa's import lean.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

_TWO_PI = 2.0 * np.pi
_EPS = 1e-9
ARC_GRID = 180  # bracketing resolution for phi sign-zeros (refined by bisection)
PARAM_GRID: NDArray[np.float64] = np.linspace(-np.pi, np.pi, ARC_GRID, endpoint=False)


def wrap(a: float) -> float:
    return float((a + np.pi) % _TWO_PI - np.pi)


def bisect(f: Callable[[float], float], a: float, b: float, tol: float = 1e-8) -> float:
    """Root of monotone-crossing ``f`` in ``[a, b]`` (dependency-free). ``tol``
    on the bracket width is ample: callers return arc *centres* and FK-verify."""
    fa = f(a)
    while b - a > tol:
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0.0:
            return m
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def merge(arcs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not arcs:
        return []
    arcs = sorted(arcs)
    out = [list(arcs[0])]
    for a, b in arcs[1:]:
        if a <= out[-1][1] + _EPS:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def intersect(
    a_arcs: list[tuple[float, float]], b_arcs: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for a0, a1 in a_arcs:
        for b0, b1 in b_arcs:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > _EPS:
                out.append((lo, hi))
    return merge(out)


def to_limits(v: float, lo: float, hi: float) -> float:
    """The 2*pi-equivalent of ``v`` nearest the limit centre."""
    k = round((0.5 * (lo + hi) - v) / _TWO_PI)
    return v + _TWO_PI * k


def _arc_feasible(
    u: float,
    w: float,
    grid: NDArray[np.float64],
    val: NDArray[np.float64],
    phi: Callable[[float], float],
    *,
    periodic: bool,
) -> bool:
    """Whether the arc ``(u, w)`` between two consecutive boundaries is feasible.

    Decided by the grid samples inside it, which all share one sign: boundaries
    are only placed where the grid changes sign. The midpoint is the fallback
    for an arc holding no grid sample. Testing the midpoint alone is wrong where
    ``q(t)`` is discontinuous -- at a wrist gimbal lock the chart's coordinate
    flips by ``pi`` -- and the midpoint lands on the discontinuity: an iiwa14
    joint in range over 92% of the circle came back never in range that way.
    """
    ts = grid if not periodic else np.where(grid < u, grid + _TWO_PI, grid)
    inside = val[(ts > u) & (ts < w)]
    if inside.size:
        return bool(np.count_nonzero(inside >= 0) * 2 > inside.size)
    mid = 0.5 * (u + w)
    return phi(wrap(mid) if periodic else mid) >= 0


def arcs_for_joint(
    q_of: Callable[[float], float],
    lo: float,
    hi: float,
    grid: NDArray[np.float64],
    q_col: NDArray[np.float64],
) -> list[tuple[float, float]]:
    """Feasible-``t`` arcs for a single joint ``q_of(t)`` in ``[lo, hi]``.

    ``q_col`` is ``q_of`` evaluated on ``grid`` (precomputed, so the batched joint
    family is evaluated once); sign changes of ``phi`` on it bracket the exact
    boundaries, refined by bisection on the scalar ``phi``.
    """
    c = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if half >= np.pi:  # unconstrained (continuous) joint
        return [(-np.pi, np.pi)]
    thr = float(np.cos(half))
    val = np.cos(q_col - c) - thr

    def phi(p: float) -> float:
        return float(np.cos(q_of(p) - c)) - thr

    n_grid = grid.shape[0]
    roots: list[float] = []
    for k in range(n_grid):
        a = float(grid[k])
        b = float(grid[k + 1]) if k + 1 < n_grid else np.pi
        if val[k] * val[(k + 1) % n_grid] < 0:
            roots.append(bisect(phi, a, b))
    if not roots:
        return [(-np.pi, np.pi)] if val[0] >= 0 else []
    roots.sort()
    arcs: list[tuple[float, float]] = []
    for u, w in pairwise([*roots, roots[0] + _TWO_PI]):
        if _arc_feasible(u, w, grid, val, phi, periodic=True):
            if w <= np.pi:
                arcs.append((u, w))
            else:  # arc straddles +pi: split
                arcs.append((u, np.pi))
                arcs.append((-np.pi, wrap(w)))
    return merge(arcs)


#: Largest move of any swept joint between two points of the bracketing grid.
#: A joint that leaves its range and comes back between two grid points is
#: invisible to the sign-change search, so the grid is refined until no step
#: moves a joint further than this -- measured 0.243 rad past a Panda stop over
#: 0.01 of ``t`` on the fixed 180-point grid, on 15 of 80 random Panda/iiwa14 poses.
MAX_JOINT_STEP = 0.02
#: Refinement floor in ``t``: where ``q(t)`` jumps (the coordinate flips by pi at
#: a gimbal lock) no step size is ever small enough, and the jump itself is the
#: boundary.
MIN_PARAM_STEP = 1e-12


def refine_grid(
    q_scalar: Callable[[float], NDArray[np.float64]],
    grid: NDArray[np.float64],
    q_grid: NDArray[np.float64],
    swept_joints: Sequence[int],
    *,
    periodic: bool,
    q_batch: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``grid`` and ``q_grid`` refined where the bracketing search could miss an
    excursion: a step is split while some swept joint moves more than
    :data:`MAX_JOINT_STEP` across it, *or* bends more than that away from the
    straight line between its ends at the step's midpoint. The first catches a
    joint racing past its stop near a fold; the second one that goes out and
    comes back between two samples, which moving endpoints alone never show.
    On the circle the step from the last point round to ``grid[0] + 2*pi``
    counts too. Steps that pass are not revisited. Sampling cannot see an
    excursion narrower than every sample it could land on; this makes that a
    sub-:data:`MAX_JOINT_STEP` event rather than a grid-step one.
    """
    cols = list(swept_joints)

    def dist(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        d = np.abs((a[cols] - b[cols] + np.pi) % _TWO_PI - np.pi)
        return float(np.max(np.where(np.isfinite(d), d, 0.0)))

    ts = [float(t) for t in grid]
    qs = [np.asarray(q, dtype=np.float64) for q in q_grid]
    n = len(ts)
    # Work list of steps (left t, left q, right t, right q), right t unwrapped.
    work = [
        (ts[k], qs[k], ts[k + 1] if k + 1 < n else ts[0] + _TWO_PI, qs[(k + 1) % n])
        for k in range(n if periodic else n - 1)
    ]
    added_t: list[float] = []
    added_q: list[NDArray[np.float64]] = []
    for _ in range(64):
        if not work:
            break
        mids = [0.5 * (a + b) for a, _qa, b, _qb in work]
        at = [float(wrap(m)) if periodic else m for m in mids]
        if q_batch is not None:
            mq = list(np.asarray(q_batch(np.asarray(at)), dtype=np.float64))
        else:
            mq = [np.asarray(q_scalar(t), dtype=np.float64) for t in at]
        nxt = []
        for (a, qa, b, qb), m, qm in zip(work, mids, mq, strict=True):
            if b - a <= MIN_PARAM_STEP:
                continue
            chord = qa + 0.5 * ((qb - qa + np.pi) % _TWO_PI - np.pi)
            if max(dist(qa, qb), dist(qm, chord)) <= MAX_JOINT_STEP:
                continue
            added_t.append(float(wrap(m)) if periodic else m)
            added_q.append(qm)
            nxt += [(a, qa, m, qm), (m, qm, b, qb)]
        work = nxt
    if not added_t:
        return np.asarray(grid, dtype=np.float64), np.asarray(q_grid, dtype=np.float64)
    all_t = np.concatenate((np.asarray(grid, dtype=np.float64), np.asarray(added_t)))
    all_q = np.vstack((np.asarray(q_grid, dtype=np.float64), np.asarray(added_q)))
    order = np.argsort(all_t, kind="stable")
    return all_t[order], all_q[order]


def feasible_arcs(
    q_scalar: Callable[[float], NDArray[np.float64]],
    q_grid: NDArray[np.float64],
    swept_joints: Sequence[int],
    limits: list[tuple[float, float]],
    grid: NDArray[np.float64] = PARAM_GRID,
    q_batch: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> list[tuple[float, float]]:
    """Exact feasible-``t`` set: the intersection of every swept joint's arcs.

    ``q_scalar(t) -> (K,)`` is the joint family; ``q_grid`` is it evaluated on
    ``grid`` (shape ``(N, K)``), so each joint's grid column is reused. Fixed
    (parameter-independent) joints are the caller's responsibility to pre-check.
    Empty iff no ``t`` keeps all swept joints in-limits.
    """

    def joint(i: int) -> Callable[[float], float]:
        return lambda t: float(q_scalar(t)[i])

    grid, q_grid = refine_grid(q_scalar, grid, q_grid, swept_joints, periodic=True, q_batch=q_batch)
    arcs: list[tuple[float, float]] = [(-np.pi, np.pi)]
    for i in swept_joints:
        arcs = intersect(
            arcs, arcs_for_joint(joint(i), limits[i][0], limits[i][1], grid, q_grid[:, i])
        )
        if not arcs:
            return []
    return arcs


# --- Bounded, non-periodic domain --------------------------------------------
# The SRS swivel psi lives on the circle [-pi, pi); a locked-joint redundancy
# (e.g. #373's q6) lives on the joint's own limits [a, b] with no wrap-around.
# Same phi_i = cos(q_i - c) - cos(h) >= 0 test, but sub-intervals are clipped to
# [a, b] and the domain endpoints are interval boundaries (no roots[0] + 2pi).


def arcs_for_joint_bounded(
    q_of: Callable[[float], float],
    lo: float,
    hi: float,
    grid: NDArray[np.float64],
    q_col: NDArray[np.float64],
) -> list[tuple[float, float]]:
    """Feasible sub-intervals of the bounded domain ``[grid[0], grid[-1]]`` for a
    single joint ``q_of(t)`` in ``[lo, hi]`` -- the non-periodic analogue of
    :func:`arcs_for_joint` (no wrap; endpoints are boundaries)."""
    a0, b0 = float(grid[0]), float(grid[-1])
    c = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if half >= np.pi:  # unconstrained (continuous) joint
        return [(a0, b0)]
    thr = float(np.cos(half))
    val = np.cos(q_col - c) - thr

    def phi(p: float) -> float:
        return float(np.cos(q_of(p) - c)) - thr

    roots: list[float] = []
    for k in range(grid.shape[0] - 1):
        if val[k] * val[k + 1] < 0:
            roots.append(bisect(phi, float(grid[k]), float(grid[k + 1])))
    pts = [a0, *sorted(roots), b0]
    arcs = [(u, w) for u, w in pairwise(pts) if _arc_feasible(u, w, grid, val, phi, periodic=False)]
    return merge(arcs)


def feasible_arcs_bounded(
    q_scalar: Callable[[float], NDArray[np.float64]],
    q_grid: NDArray[np.float64],
    swept_joints: Sequence[int],
    limits: list[tuple[float, float]],
    grid: NDArray[np.float64],
    q_batch: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
) -> list[tuple[float, float]]:
    """Bounded-domain analogue of :func:`feasible_arcs`: the in-limits sub-set of
    ``[grid[0], grid[-1]]`` (non-periodic) where every swept joint is in range."""

    def joint(i: int) -> Callable[[float], float]:
        return lambda t: float(q_scalar(t)[i])

    grid, q_grid = refine_grid(
        q_scalar, grid, q_grid, swept_joints, periodic=False, q_batch=q_batch
    )
    arcs: list[tuple[float, float]] = [(float(grid[0]), float(grid[-1]))]
    for i in swept_joints:
        arcs = intersect(
            arcs, arcs_for_joint_bounded(joint(i), limits[i][0], limits[i][1], grid, q_grid[:, i])
        )
        if not arcs:
            return []
    return arcs
