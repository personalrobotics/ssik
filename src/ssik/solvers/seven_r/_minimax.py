"""Limit contacts of a redundant 7R self-motion: the minimax-margin point (#662).

The in-limits resolvers find where every joint is inside its range by bracketing
the sign changes of each joint's margin along the self-motion chart
(:mod:`._feasible_param`). When the in-limit part of the chart is a single point
or a sliver narrower than the bisection tolerance -- two or more joints at their
limits, the usual case at a pose built from a configuration on its limits -- the
worst margin touches zero without changing sign, so nothing is bracketed and the
resolver returns ``[]`` although an in-limits solution exists (#652, class d1).

This module finds that point instead of bisecting for it. Along a chart ``q(t)``
the worst-case violation

    V(t) = max_i ( |wrap(q_i(t) - c_i)| - h_i )

(``c_i`` / ``h_i`` the limit centre and half-width; negative is a margin) is a
maximum of smooth curves, so its minimum is either a corner where two joints'
violations cross or the smooth extremum of one. :func:`chart_minima` locates
each local minimum on the resolver's grid and refines it by golden section;
on a closed-form chart that is the contact itself. On an approximate chart
(the ``*_polished`` families, whose closed form is of a best-fit arm) it is a
seed, and :func:`walk` minimises ``V`` along the true self-motion curve from
it: each step solves the 1-D linear minimax of the joints' linearised
violations along the null direction of the Jacobian, then projects back onto
``FK(q) = T``.

A minimum counts as in limits when its violation is within the solution's own
error band (#651): :func:`place` applies ``postprocess._onto_limits`` joint by
joint, which clamps a value outside its limit by at most that band onto the
limit and rejects anything farther, and re-measures ``fk_residual`` at the
clamped ``q`` (#649). So every contact returned lies within its limits exactly.

Nothing here runs on the default path: the resolvers call it only when their
own result is empty, and :meth:`ssik.chart.Chart.in_limits` only for a chart
with no in-limits arc. Mirrored in C++ by ``ssik_cpp/seven_r/minimax.hpp``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.postprocess import _BAND_CAP, _Band, _onto_limits
from ssik.refinement import kinbody_jacobian, se3_log_residual
from ssik.solvers.seven_r._feasible_param import refine_grid, to_limits, wrap

if TYPE_CHECKING:  # pragma: no cover
    from ssik._kinbody import KinBody

__all__ = ["chart_minima", "limit_violation", "place", "walk", "within_band"]

_TWO_PI = 2.0 * np.pi

#: Grid minima of ``V`` above this are not refined. After ``refine_grid`` no
#: joint moves or bends more than ``MAX_JOINT_STEP`` (0.02 rad) between two grid
#: points, so ``V`` cannot dip from above this to zero inside one step.
SCAN = 0.05
#: Deepest margin a contact may have (rad). This mechanism owns the in-limit
#: sets that are a point or a sliver: on the #648 targeted poses the deepest
#: such contact is 3.2e-5 (a GEN72 wrist 1e-9 from its stop near a singularity),
#: and a configuration 1e-6 inside its limits gives 1e-6. A deeper minimum with
#: no arc around it is an arc another gate lost -- the reach test at a singular
#: pose, an approximate chart's drift, the polished families' coarse sweep -- and
#: is left to the rescue, as before.
SLIVER = 1e-4
#: Golden-section stop: the bracket is this many ulps of its position wide.
_GOLDEN_REL = 4.0 * np.finfo(np.float64).eps
_GOLDEN_ITERS = 100
#: Walk tuning: initial and largest step along the self-motion curve (rad of
#: joint motion), iteration cap, and the Gauss-Newton projection's caps.
_WALK_STEP = 0.05
_WALK_ITERS = 100
_WALK_STOP = 1e-15
_PROJECT_ITERS = 30
_PROJECT_CLIP = 0.5


def limit_violation(
    qs: NDArray[np.float64],
    limits: Sequence[tuple[float, float]],
    joints: Sequence[int] | None = None,
) -> NDArray[np.float64]:
    """Worst angular limit violation per row of ``qs`` (``(N, 7)``): the largest
    ``|wrap(q_i - c_i)| - h_i`` over the limited joints, negative when every joint
    is inside (the smallest margin). A joint whose range spans a turn never
    counts, nor one outside ``joints`` when that is given; a row with a
    non-finite value is ``inf`` (off the chart)."""
    q = np.atleast_2d(np.asarray(qs, dtype=np.float64))
    lim = np.asarray(limits, dtype=np.float64)
    c = 0.5 * (lim[:, 0] + lim[:, 1])
    h = 0.5 * (lim[:, 1] - lim[:, 0])
    v = np.abs((q - c + np.pi) % _TWO_PI - np.pi) - h
    skip = h >= np.pi
    if joints is not None:
        skip = skip | ~np.isin(np.arange(lim.shape[0]), list(joints))
    v = np.where(skip, -np.inf, v)
    out: NDArray[np.float64] = np.max(v, axis=1)
    out[~np.all(np.isfinite(q), axis=1)] = np.inf
    return out


def _golden(f: Callable[[float], float], a: float, b: float) -> tuple[float, float]:
    """Minimum of ``f`` on ``[a, b]`` by golden section: the best point evaluated."""
    g = 0.5 * (np.sqrt(5.0) - 1.0)
    x1 = b - g * (b - a)
    x2 = a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(_GOLDEN_ITERS):
        if b - a <= _GOLDEN_REL * max(1.0, abs(a), abs(b)):
            break
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = f(x2)
    return (x1, f1) if f1 <= f2 else (x2, f2)


def chart_minima(
    q_batch: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    grid: NDArray[np.float64],
    limits: Sequence[tuple[float, float]],
    *,
    periodic: bool,
    joints: Sequence[int] | None = None,
) -> list[tuple[float, float]]:
    """Local minima ``(t, V)`` of the worst limit violation along a chart.

    ``q_batch(ts) -> (N, 7)`` evaluates the chart (rows ``NaN`` off it). The grid
    is refined as for the arcs (:func:`._feasible_param.refine_grid`, over all
    seven joints), every local minimum of ``V`` on it at or below :data:`SCAN`
    (and the lowest grid point, which a plateau can hide from the local test) is
    refined by golden section between its two neighbours, and the refined
    points come back sorted by ``V``, then ``t``.
    ``periodic`` charts live on the circle ``[-pi, pi)``. ``joints`` restricts
    ``V`` to those joints: an approximate SRS chart leaves out its elbow, which
    is constant along the chart and off by the pivot drift, so it would hide the
    other joints' minima under one plateau (the walk then sees the true elbow).
    """

    def q_scalar(t: float) -> NDArray[np.float64]:
        out: NDArray[np.float64] = q_batch(np.array([t]))[0]
        return out

    grid = np.asarray(grid, dtype=np.float64)
    ts, qs = refine_grid(
        q_scalar, grid, q_batch(grid), range(7), periodic=periodic, q_batch=q_batch
    )
    v = limit_violation(qs, limits, joints)
    n = ts.shape[0]
    if n == 0 or not np.isfinite(v).any():
        return []
    best = int(np.argmin(v))

    def f(t: float) -> float:
        q_t = q_batch(np.array([wrap(t) if periodic else t]))
        return float(limit_violation(q_t, limits, joints)[0])

    out: list[tuple[float, float]] = []
    for k in range(n):
        vk = float(v[k])
        if not vk <= SCAN:
            continue
        if periodic:
            left, right = (k - 1) % n, (k + 1) % n
        else:
            left, right = max(k - 1, 0), min(k + 1, n - 1)
        # The first point of a plateau: strictly below the left, not above the right.
        if k != best and ((left != k and not vk < v[left]) or (right != k and v[right] < vk)):
            continue
        a = float(ts[left]) - (_TWO_PI if periodic and left > k else 0.0)
        b = float(ts[right]) + (_TWO_PI if periodic and right < k else 0.0)
        t, ft = _golden(f, a, b)
        if not ft < vk:  # never worse than the grid point itself
            t, ft = float(ts[k]), vk
        out.append((wrap(t) if periodic else t, ft))
    out.sort()
    out.sort(key=lambda p: p[1])
    return out


def _null_direction(jac: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Unit null vector of a ``(6, 7)`` Jacobian by cofactors: component ``i``
    is ``(-1)^i det(J without column i)``. ``None`` where ``J`` drops rank."""
    n = np.array([(-1.0) ** i * np.linalg.det(np.delete(jac, i, axis=1)) for i in range(7)])
    norm = float(np.linalg.norm(n))
    if not norm > 0.0:
        return None
    out: NDArray[np.float64] = n / norm
    return out


def _project(
    kb: KinBody, q: NDArray[np.float64], t_target: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """``q`` moved onto ``FK(q) = T`` by minimum-norm Gauss-Newton steps (each
    clipped per joint), so a point near the self-motion curve lands next to
    where it was. ``None`` where the Jacobian drops rank."""
    q = np.asarray(q, dtype=np.float64).copy()
    for _ in range(_PROJECT_ITERS):
        e = se3_log_residual(t_target @ np.linalg.inv(poe_forward_kinematics(kb, q)))
        if float(np.max(np.abs(e))) <= _WALK_STOP:
            break
        jac = kinbody_jacobian(kb, q)
        try:
            dq = jac.T @ np.linalg.solve(jac @ jac.T, e)
        except np.linalg.LinAlgError:
            return None
        q = q + np.clip(dq, -_PROJECT_CLIP, _PROJECT_CLIP)
        if float(np.max(np.abs(dq))) <= _WALK_STOP:
            break
    return q


def _lp_step(v: NDArray[np.float64], g: NDArray[np.float64], rho: float) -> float:
    """The ``s`` in ``[-rho, rho]`` minimising ``max_i(v_i + g_i s)``: an end of
    the interval or a crossing of two lines; ties go to the shorter step."""
    cands = [-rho, rho]
    k = v.shape[0]
    for i in range(k):
        for j in range(i + 1, k):
            if g[i] != g[j]:
                s = float((v[j] - v[i]) / (g[i] - g[j]))
                if -rho < s < rho:
                    cands.append(s)
    best_s, best_v = 0.0, np.inf
    for s in cands:
        val = float(np.max(v + g * s))
        if val < best_v or (val == best_v and abs(s) < abs(best_s)):
            best_s, best_v = s, val
    return best_s


def walk(
    kb: KinBody,
    q: NDArray[np.float64],
    t_target: NDArray[np.float64],
    limits: Sequence[tuple[float, float]],
) -> NDArray[np.float64] | None:
    """Minimise the worst limit violation along the true self-motion curve
    through ``q``: project onto ``FK(q) = T``, then repeatedly take the step
    along the Jacobian's null direction that minimises the joints' linearised
    violations (:func:`_lp_step`, at most :data:`_WALK_STEP`), project back, and
    keep it only if the true ``V`` fell, else halve the step. Near a corner the
    linearisation is exact to second order, so the walk converges there in a
    few steps. ``None`` where the projection fails."""
    lim = np.asarray(limits, dtype=np.float64)
    c = 0.5 * (lim[:, 0] + lim[:, 1])
    h = 0.5 * (lim[:, 1] - lim[:, 0])
    con = h < np.pi
    start = _project(kb, q, t_target)
    if start is None:
        return None
    cur: NDArray[np.float64] = start
    v_cur = float(limit_violation(cur, limits)[0])
    rho = _WALK_STEP
    for _ in range(_WALK_ITERS):
        n = _null_direction(kinbody_jacobian(kb, cur))
        if n is None:
            break
        d = (cur - c + np.pi) % _TWO_PI - np.pi
        s = _lp_step((np.abs(d) - h)[con], (np.sign(d) * n)[con], rho)
        if abs(s) <= _WALK_STOP:
            break
        nxt = _project(kb, cur + s * n, t_target)
        v_nxt = float(limit_violation(nxt, limits)[0]) if nxt is not None else np.inf
        if nxt is not None and v_nxt < v_cur:
            cur, v_cur = nxt, v_nxt
        else:
            rho = 0.5 * abs(s)
            if rho <= _WALK_STOP:
                break
    return cur


def place(
    kb: KinBody,
    q: NDArray[np.float64],
    t_target: NDArray[np.float64],
    limits: Sequence[tuple[float, float]],
    fk_atol: float,
    refinement: Literal["none", "lm"],
) -> Solution | None:
    """A contact as a :class:`Solution`, or ``None``.

    Each joint is taken at its representative nearest its range's centre; the
    worst violation must lie between ``-SLIVER`` and the band's cap and the
    point must close FK within ``fk_atol`` there; then every limited joint goes
    through ``postprocess._onto_limits`` with the point's own error band (#651),
    which puts a value outside its limit by at most the band onto the limit and
    rejects anything farther. A point the clamp moved has its ``fk_residual``
    re-measured (#649), so the residual describes the ``q`` returned."""
    lim = list(limits)
    qw = np.array([to_limits(float(q[i]), *lim[i]) for i in range(len(lim))])
    if not -SLIVER <= float(limit_violation(qw, lim)[0]) <= _BAND_CAP:
        return None
    residual = float(np.linalg.norm(poe_forward_kinematics(kb, qw) - t_target))
    if not residual <= fk_atol:
        return None
    band = _Band(kb, qw, t_target)
    out = qw.copy()
    for i, joint in enumerate(kb.joints):
        if joint.limits is None or joint.limits[0] is None or joint.limits[1] is None:
            continue
        v = _onto_limits(float(qw[i]), lim[i][0], lim[i][1], band)
        if v is None:
            return None
        out[i] = v
    if np.any(out != qw):
        residual = float(np.linalg.norm(poe_forward_kinematics(kb, out) - t_target))
    return Solution(q=out, fk_residual=residual, refinement_used=refinement)


def within_band(
    kb: KinBody,
    q: NDArray[np.float64],
    t_target: NDArray[np.float64],
    limits: Sequence[tuple[float, float]],
) -> bool:
    """Whether every joint of ``q`` is in ``limits`` up to the point's own error
    band (the acceptance of :func:`place` without the FK gate), for a chart's
    in-limits contact. A cheap ``V`` check first: beyond the band's cap nothing
    is accepted, nor a minimum deeper than :data:`SLIVER`."""
    if not -SLIVER <= float(limit_violation(q, limits)[0]) <= _BAND_CAP:
        return False
    lim = list(limits)
    band = _Band(kb, np.asarray(q, dtype=np.float64), t_target)
    for i in range(len(lim)):
        lo, hi = lim[i]
        if hi - lo >= _TWO_PI:
            continue
        if _onto_limits(to_limits(float(q[i]), lo, hi), lo, hi, band) is None:
            return False
    return True
