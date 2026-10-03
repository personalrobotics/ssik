"""Exact reachability of every self-motion branch along a one-parameter family of
poses (request A5): the stretches of ``T0 exp(s xi)`` each branch can hold inside
the joint limits, their ends placed as critical points rather than by slicing.

For one branch, the holdable set in the plane of the family parameter ``s`` and
the chart coordinate ``t`` is where the posture exists and every joint is inside
its box. Its reachable stretches along ``s`` are the projection of that set, and
a projection ends only where a slice pinches off, at a point of the set's
boundary that is extremal in ``s``. The boundary is made of smooth curves, each
one equation, so every end is one of five kinds:

    peak       a joint on its bound, the limit curve tangent to the slice
    corner     two joints on their bounds (a kink of the margin)
    edge       a joint on its bound at a fold of the chart (``q6`` charts)
    birth      the branch appears, already holdable
    workspace  the pose leaves the workspace: every branch's chart is a point

The first three are square systems in the posture and ``s`` -- forward
kinematics on the family's pose, the joint on its bound, and the kind's own
equation, the fold being the determinant of the spatial Jacobian's first six
columns -- which Newton solves in a few steps from a seed. Seeds come from the
margin (:meth:`~ssik.chart.Chart.margin`) at a coarse scan, from stops dense in
``sqrt(s - s_w)`` next to each workspace boundary ``s_w``, where the manifold is
born as a point and everything grows like that square root, and on ``q6``
charts from the exact elbow-fold curves. A candidate is an end only if the
branch's own margin changes sign across it; a bracket no candidate settles is
bisected on the margin, so the result is complete either way, and each end says
how it was found.

Derived and checked against a per-pose reach in
``self-motion-charts/derivations/reach_critical_points.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy

if TYPE_CHECKING:
    from ssik.chart import Chart, SelfMotionManifold
    from ssik.internals import KinBody

Label = tuple[int, ...]
Interval = tuple[float, float]
Kind = tuple[str, int, str, "tuple[int, str] | None"]  # a critical-point system to try
Found = tuple[float, str, float]  # (s, kind, t) of an end

ABSENT = -np.pi  # the margin reported where a branch does not exist: any negative value
_ON = 1e-6  # a joint this close to its bound is on it
_CROSS = 2e-8  # the margin's sign must change within this of an accepted end
_NEWTON_TOL = 1e-9
_NEWTON_ITERS = 25


@dataclass(frozen=True)
class ReachEnd:
    """One end of a reachable stretch: the branch, where, how it was found, and
    the chart coordinate of the posture there (``nan`` for a workspace end)."""

    label: Label
    s: float
    kind: str  # peak, corner, edge, birth, workspace, bisected
    t: float


@dataclass
class Reach:
    """The reachable stretches of every branch along ``T0 exp(s xi)``.

    ``intervals[label]`` are the ``(s_lo, s_hi)`` stretches the branch holds
    inside the limits; ``workspace`` where any branch exists at all; ``ends``
    every interval end with the kind of critical point it is; ``stops`` the
    parameter values scanned and ``margins[label]`` the branch's margin at each
    (``ABSENT`` where it does not exist). :meth:`margin` evaluates a branch's
    margin anywhere."""

    s_range: Interval
    intervals: dict[Label, list[Interval]]
    workspace: list[Interval]
    ends: list[ReachEnd]
    stops: NDArray[np.float64]
    margins: dict[Label, NDArray[np.float64]]
    _family: _Family = field(repr=False)

    @property
    def labels(self) -> list[Label]:
        return sorted(self.intervals)

    def margin(self, label: Label, s: float) -> float:
        """The branch's holdability margin at ``s`` (:meth:`Chart.margin`), ``ABSENT``
        where it does not exist: a signed distance to the reachability boundary."""
        return self._family.margin(label, s)

    def pose(self, s: float) -> NDArray[np.float64]:
        return self._family.pose(s)


def reach_along(
    kb: KinBody,
    T0: ArrayLike,
    twist: ArrayLike,
    s_range: tuple[float, float],
    *,
    limits: ArrayLike | None = None,
    solver_name: str | None = None,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    native: bool = True,
    scan: int = 21,
    depth: float = 0.02,
    xtol: float = 1e-9,
) -> Reach:
    """Every branch's reachable stretches along ``T0 exp(s xi)``, ``s`` in ``s_range``.

    :param T0: the pose at ``s = 0``, 4x4 in the base frame.
    :param twist: the body-frame twist ``xi = (v, w)`` the pose moves along, per
        unit ``s``: a unit translation along the tool axis slides the tool along
        its own line; a rotation spins it.
    :param limits: ``(7, 2)`` per-joint ``(lower, upper)``; ``None`` uses the
        chain's own.
    :param scan: coarse stops over ``s_range`` that seed the search.
    :param depth: how far from a workspace boundary the stops are dense in
        ``sqrt(s - s_w)``, in units of ``s``.
    :param xtol: tolerance of the bisections and root brackets in ``s``.
    """
    from ssik.chart import _chain_limits

    lims = (
        _chain_limits(kb)
        if limits is None
        else tuple((float(lo), float(hi)) for lo, hi in np.asarray(limits, dtype=np.float64))
    )
    family = _Family(
        kb,
        np.asarray(T0, dtype=np.float64),
        np.asarray(twist, dtype=np.float64),
        lims,
        solver_name,
        policy,
        native,
    )
    return _Search(family, (float(s_range[0]), float(s_range[1])), scan, depth, xtol).run()


# -- the family of poses and its branches ------------------------------------------------------


class _Family:
    def __init__(
        self,
        kb: KinBody,
        T0: NDArray[np.float64],
        twist: NDArray[np.float64],
        limits: tuple[tuple[float, float], ...],
        solver_name: str | None,
        policy: TolerancePolicy,
        native: bool,
    ) -> None:
        self.kb, self.T0, self.twist, self.limits = kb, T0, twist, limits
        self.solver_name, self.policy, self.native = solver_name, policy, native
        self.lower = np.array([lo for lo, _ in limits], dtype=np.float64)
        self.upper = np.array([hi for _, hi in limits], dtype=np.float64)
        self._cache: dict[float, SelfMotionManifold] = {}

    def pose(self, s: float) -> NDArray[np.float64]:
        from ssik.chart import se3_exp

        return np.asarray(self.T0 @ se3_exp(s * self.twist), dtype=np.float64)

    def manifold(self, s: float) -> SelfMotionManifold:
        from ssik.chart import charts

        m = self._cache.get(s)
        if m is None:
            if len(self._cache) > 2048:
                self._cache.clear()
            m = self._cache[s] = charts(
                self.kb,
                self.pose(s),
                solver_name=self.solver_name,
                policy=self.policy,
                native=self.native,
            )
        return m

    def chart(self, label: Label, s: float) -> Chart | None:
        for c in self.manifold(s):
            if tuple(c.label) == label and c.domain:
                return c
        return None

    def exists(self, s: float) -> bool:
        return any(c.domain for c in self.manifold(s))

    def margins(self, s: float) -> dict[Label, tuple[float, float]]:
        out: dict[Label, tuple[float, float]] = {}
        for c in self.manifold(s):
            if not c.domain:
                continue
            m, t = c.margin(self.limits)
            if np.isfinite(m) and m > out.get(tuple(c.label), (-np.inf, 0.0))[0]:
                out[tuple(c.label)] = (float(m), float(t))
        return out

    def margin(self, label: Label, s: float) -> float:
        c = self.chart(label, s)
        if c is None:
            return ABSENT
        m, _ = c.margin(self.limits)
        return float(m) if np.isfinite(m) else ABSENT

    def q(self, label: Label, s: float, t: float) -> NDArray[np.float64] | None:
        c = self.chart(label, s)
        if c is None:
            return None
        q = np.asarray(c.q(t), dtype=np.float64)
        return None if np.isnan(q).any() else q

    def slacks(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Each joint's distance to its box, on the turn nearest the box centre."""
        c = 0.5 * (self.lower + self.upper)
        v = c + (q - c + np.pi) % (2.0 * np.pi) - np.pi
        return np.minimum(v - self.lower, self.upper - v)

    def crossing(self, label: Label, s: float, inward: float) -> bool:
        """Whether the branch's margin changes sign across ``s``: the definitive test
        of an end, whatever found it."""
        m_in, m_out = (
            self.margin(label, s + inward * _CROSS),
            self.margin(label, s - inward * _CROSS),
        )
        return m_in >= -1e-9 and m_out < 1e-9 and m_in > m_out

    def fold_ends(self, s: float) -> list[float]:
        """The exact elbow-fold values of ``t`` at ``s`` on a ``q6`` family, else none."""
        m = self.manifold(s)
        if m.parameter != "q6":
            return []
        from ssik.chart import _bake_cached
        from ssik.kinematics._scalar3 import _se3_inv
        from ssik.solvers.seven_r import spherical_shoulder as sh

        out: list[float] = []
        for lo, hi in sh.elbow_arcs(_bake_cached(self.kb), _se3_inv(self.pose(s))):
            out += [lo, hi]
        return out


# -- the critical-point systems ----------------------------------------------------------------


def _adjoint(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ad_T on ``(v, w)`` twists: the spatial twist of a motion given in T's frame."""
    R, p = T[:3, :3], T[:3, 3]
    P = np.array([[0.0, -p[2], p[1]], [p[2], 0.0, -p[0]], [-p[1], p[0], 0.0]])
    out = np.zeros((6, 6))
    out[:3, :3], out[:3, 3:], out[3:, 3:] = R, P @ R, R
    return out


class _System:
    """The square system ``F(q, s) = 0`` of one kind of critical point, with its
    Jacobian: the pose residual (6, analytic in ``q`` and ``s``), the active joint
    on its bound (1), and the kind's own equation (1, by central differences)."""

    def __init__(
        self,
        fam: _Family,
        kind: str,
        a: int,
        side: str,
        extra: tuple[int, str] | None,
        q_seed: NDArray[np.float64],
    ) -> None:
        from ssik.refinement import kinbody_jacobian
        from ssik.solvers.seven_r._minimax import _null_direction

        self.fam, self.kind, self.a, self.side, self.extra = fam, kind, a, side, extra
        self.bound = fam.lower[a] if side == "lo" else fam.upper[a]
        self.sign = 1.0 if side == "lo" else -1.0
        self._jac = kinbody_jacobian
        self._null = _null_direction
        n = self._null(kinbody_jacobian(fam.kb, q_seed))
        self.reference = n if n is not None else np.zeros(7)

    def residual(self, q: NDArray[np.float64], s: float) -> NDArray[np.float64]:
        from ssik.kinematics.poe_fk import poe_forward_kinematics
        from ssik.refinement import se3_log_residual

        return se3_log_residual(
            self.fam.pose(s) @ np.linalg.inv(poe_forward_kinematics(self.fam.kb, q))
        )

    def own(self, q: NDArray[np.float64]) -> float:
        jac = self._jac(self.fam.kb, q)
        if self.kind == "peak":
            n = self._null(jac)
            if n is None:
                return np.nan
            if n @ self.reference < 0:
                n = -n
            return float(n[self.a])
        if self.kind == "corner":
            assert self.extra is not None
            b, bside = self.extra
            v = self.fam.slacks(q)
            return (
                float(v[b])
                if bside is None
                else float(
                    self._wrapped(q)[b]
                    - (self.fam.lower[b] if bside == "lo" else self.fam.upper[b])
                )
            )
        return float(np.linalg.det(jac[:, :6]))  # edge: a fold of q6

    def _wrapped(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        c = 0.5 * (self.fam.lower + self.fam.upper)
        return np.asarray(c + (q - c + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)

    def F(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        q, s = x[:7], float(x[7])
        own = self.sign * (self._wrapped(q)[self.a] - self.bound)
        return np.asarray(np.r_[self.residual(q, s), own, self.own(q)], dtype=np.float64)

    def J(self, x: NDArray[np.float64], h: float = 1e-6) -> NDArray[np.float64]:
        q, s = x[:7], float(x[7])
        out = np.zeros((8, 8))
        out[:6, :7] = -self._jac(self.fam.kb, q)
        out[:6, 7] = _adjoint(self.fam.pose(s)) @ self.fam.twist
        out[6, self.a] = self.sign
        for j in range(7):
            e = np.zeros(7)
            e[j] = h
            out[7, j] = (self.own(q + e) - self.own(q - e)) / (2 * h)
        return out

    def solve(self, q0: NDArray[np.float64], s0: float) -> tuple[bool, NDArray[np.float64], float]:
        x = np.r_[q0, s0]
        last = np.inf
        for _ in range(_NEWTON_ITERS):
            f = self.F(x)
            if not np.all(np.isfinite(f)):
                return False, x[:7], float(x[7])
            if np.abs(f).max() < _NEWTON_TOL or (last < 1e-11 and np.abs(f).max() < 1e-6):
                return True, x[:7], float(x[7])
            try:
                step = np.linalg.lstsq(self.J(x), f, rcond=None)[0]
            except np.linalg.LinAlgError:
                return False, x[:7], float(x[7])
            x = x - step
            last = float(np.abs(step).max())
        return False, x[:7], float(x[7])


# -- the search --------------------------------------------------------------------------------


class _Search:
    def __init__(
        self, fam: _Family, s_range: Interval, scan: int, depth: float, xtol: float
    ) -> None:
        self.fam, self.s_range, self.scan, self.depth, self.xtol = (
            fam,
            s_range,
            max(int(scan), 3),
            depth,
            xtol,
        )

    def run(self) -> Reach:
        fam = self.fam
        lo, hi = self.s_range
        coarse = np.linspace(lo, hi, self.scan)
        present = [fam.exists(float(s)) for s in coarse]
        # the workspace boundaries inside the range: (s_w, inward)
        bounds: list[tuple[float, float]] = []
        for i in range(len(coarse) - 1):
            if present[i] != present[i + 1]:
                s_in, s_out = (
                    (coarse[i], coarse[i + 1]) if present[i] else (coarse[i + 1], coarse[i])
                )
                bounds.append(
                    (
                        self._bisect(lambda s: fam.exists(s), float(s_in), float(s_out)),
                        1.0 if s_out < s_in else -1.0,
                    )
                )
        workspace = self._stretches([float(s) for s in coarse], present, bounds)
        # the stops: coarse, plus dense in sqrt(s - s_w) from each boundary
        stops = {float(s) for s, p in zip(coarse, present, strict=True) if p}
        for s_w, inward in bounds:
            for r in np.linspace(0.0, np.sqrt(self.depth), 41)[1:]:
                s = s_w + inward * r * r
                if lo <= s <= hi:
                    stops.add(float(s))
        stops_sorted = sorted(stops)
        at = {s: fam.margins(s) for s in stops_sorted}
        labels = sorted({label for m in at.values() for label in m})
        margins = {
            label: np.array([at[s].get(label, (ABSENT, np.nan))[0] for s in stops_sorted])
            for label in labels
        }
        ends: list[ReachEnd] = []
        intervals: dict[Label, list[Interval]] = {}
        for label in labels:
            found = self._candidates(label, stops_sorted, at, bounds, ends)
            intervals[label] = self._assemble(label, stops_sorted, margins[label], found, ends)
        intervals = {label: iv for label, iv in intervals.items() if iv}
        return Reach(
            self.s_range,
            intervals,
            workspace,
            sorted(ends, key=lambda e: (e.label, e.s)),
            np.array(stops_sorted),
            margins,
            fam,
        )

    # -- helpers -----------------------------------------------------------------------------

    def _bisect(self, holds: Callable[[float], bool], s_in: float, s_out: float) -> float:
        while abs(s_out - s_in) > 1e-12:
            mid = 0.5 * (s_in + s_out)
            if holds(mid):
                s_in = mid
            else:
                s_out = mid
        return s_in

    def _stretches(
        self, stops: list[float], mask: list[bool], bounds: list[tuple[float, float]]
    ) -> list[Interval]:
        out: list[Interval] = []
        start: float | None = None
        for i, flag in enumerate(mask):
            if flag and start is None:
                start = (
                    stops[0]
                    if i == 0
                    else next(w for w, _ in bounds if stops[i - 1] < w <= stops[i])
                )
            if start is not None and (not flag or i == len(mask) - 1):
                end = (
                    stops[-1]
                    if flag
                    else next(w for w, _ in bounds if stops[i - 1] <= w < stops[i])
                )
                out.append((start, end))
                start = None
        return out

    def _classify(self, label: Label, s: float, t: float) -> list[Kind]:
        """Candidate systems at a seed, the likeliest first: the kind the active
        set suggests, then the alternatives the boundary might turn out to be."""
        fam = self.fam
        q = fam.q(label, s, t)
        if q is None:
            return []
        sl = fam.slacks(q)
        order = [int(i) for i in np.argsort(sl)]
        v = (
            0.5 * (fam.lower + fam.upper)
            + (q - 0.5 * (fam.lower + fam.upper) + np.pi) % (2 * np.pi)
            - np.pi
        )

        def side(i: int) -> str:
            return "lo" if v[i] - fam.lower[i] < fam.upper[i] - v[i] else "hi"

        a = order[0]
        out: list[Kind] = []
        if len(order) > 1 and sl[order[1]] - sl[a] < _ON:
            out.append(("corner", a, side(a), (order[1], side(order[1]))))
        out.append(("peak", a, side(a), None))
        if fam.manifold(s).parameter == "q6":
            out.append(("edge", a, side(a), None))
        for b in order[1:3]:
            out.append(("corner", a, side(a), (b, side(b))))
            out.append(("peak", b, side(b), None))
        return out

    def _candidates(
        self,
        label: Label,
        stops: list[float],
        at: dict[float, dict[Label, tuple[float, float]]],
        bounds: list[tuple[float, float]],
        ends: list[ReachEnd],
    ) -> list[Found]:
        """Ends found by Newton, by the fold curves and at the workspace boundaries,
        each verified by the branch's margin changing sign across it."""
        fam = self.fam
        found: list[Found] = []
        s_ws = [w for w, _ in bounds]

        # at a workspace boundary: holdable from birth
        for s_w, inward in bounds:
            c = fam.chart(label, s_w + inward * 1e-8)
            if c is not None and fam.margin(label, s_w + inward * 1e-8) >= 0:
                found.append((s_w, "workspace", float("nan")))

        # along the fold curves (q6 charts): sign changes of the least slack at an elbow-arc end
        prev: tuple[float, list[float]] | None = None
        for s in stops:
            cur = [
                min(fam.slacks(q)) if (q := fam.q(label, s, te)) is not None else np.nan
                for te in fam.fold_ends(s)
            ]
            if prev is not None and len(prev[1]) == len(cur):
                for k, (v0, v1) in enumerate(zip(prev[1], cur, strict=True)):
                    if np.isfinite(v0) and np.isfinite(v1) and (v0 >= 0) != (v1 >= 0):
                        s_in, s_out = (prev[0], s) if v0 >= 0 else (s, prev[0])
                        root = self._fold_root(label, k, s_in, s_out, s_ws)
                        inward = 1.0 if s_in > s_out else -1.0
                        if root is not None and fam.crossing(label, root, inward):
                            te = fam.fold_ends(root)
                            found.append((root, "edge", te[k] if k < len(te) else float("nan")))
            prev = (s, cur)

        # sign changes of the margin between stops: Newton on the critical-point systems
        for i in range(len(stops) - 1):
            s0, s1 = stops[i], stops[i + 1]
            if label not in at[s0] or label not in at[s1]:
                continue
            m0, m1 = at[s0][label][0], at[s1][label][0]
            if (m0 >= 0) == (m1 >= 0):
                continue
            s_in, s_out = (s0, s1) if m0 >= 0 else (s1, s0)
            inward = 1.0 if s_in > s_out else -1.0
            if any(min(s0, s1) - 1e-9 <= f <= max(s0, s1) + 1e-9 for f, _, _ in found):
                continue
            t_in = at[s_in][label][1]
            q_seed = fam.q(label, s_in, t_in)
            if q_seed is None:
                continue
            for kind, a, side, extra in self._classify(label, s_in, t_in):
                _ok, q, s = _System(fam, kind, a, side, extra, q_seed).solve(q_seed, s_in)
                if not np.isfinite(s) or not (min(s0, s1) - 1e-6 <= s <= max(s0, s1) + 1e-6):
                    continue
                if fam.crossing(label, s, inward):
                    found.append(
                        (
                            s,
                            kind,
                            float(q[6]) if fam.manifold(s).parameter == "q6" else float("nan"),
                        )
                    )
                    break
        return found

    def _fold_root(
        self, label: Label, k: int, s_in: float, s_out: float, s_ws: list[float]
    ) -> float | None:
        """The root in ``s`` of the least slack at fold ``k``, bracketed by the stops;
        in ``r = sqrt(s - s_w)`` within ``depth`` of a workspace boundary."""
        from scipy.optimize import brentq  # type: ignore[import-untyped]

        fam = self.fam

        def g(s: float) -> float:
            te = fam.fold_ends(s)
            if k >= len(te):
                return np.nan
            q = fam.q(label, s, te[k])
            return float(min(fam.slacks(q))) if q is not None else np.nan

        near = [w for w in s_ws if abs(min(s_in, s_out) - w) < self.depth and max(s_in, s_out) > w]
        try:
            if near:
                w = near[0]
                sign = 1.0 if s_in > w else -1.0

                def r(s: float) -> float:
                    return float(np.sqrt(max(sign * (s - w), 0.0)))

                def unr(r_: float) -> float:
                    return w + sign * r_ * r_

                root = brentq(lambda r_: g(unr(r_)), r(s_in), r(s_out), xtol=1e-12)
                return unr(float(root))
            return float(brentq(g, s_in, s_out, xtol=self.xtol))
        except ValueError:
            return None

    def _assemble(
        self,
        label: Label,
        stops: list[float],
        margin: NDArray[np.float64],
        found: list[Found],
        ends: list[ReachEnd],
    ) -> list[Interval]:
        """The branch's stretches: each run of holdable stops, its ends the found
        points in the bracketing gaps, else bisected on the margin (or on the
        branch's existence where it is absent at the next stop)."""
        fam = self.fam
        mask = [m >= 0 for m in margin]
        out: list[Interval] = []
        last = len(stops) - 1

        def bound(i_in: int, i_out: int) -> Found:
            a, b = sorted((stops[i_in], stops[i_out]))
            inward = 1.0 if stops[i_in] > stops[i_out] else -1.0
            hits = [f for f in found if a - 1e-9 <= f[0] <= b + 1e-9]
            if hits:
                return hits[0]
            # fallback: the branch vanishes in the gap, or its margin crosses zero
            if margin[i_out] == ABSENT:
                edge = self._bisect(
                    lambda s: fam.chart(label, s) is not None, stops[i_in], stops[i_out]
                )
                if (
                    fam.margin(label, edge) >= 0
                    or fam.margin(label, edge - inward * 2 * self.xtol) >= 0
                ):
                    return edge, "birth", float("nan")
                b_in = edge - inward * 2 * self.xtol
                return (
                    self._bisect(lambda s: fam.margin(label, s) >= 0, stops[i_in], b_in),
                    "bisected",
                    float("nan"),
                )
            return (
                self._bisect(lambda s: fam.margin(label, s) >= 0, stops[i_in], stops[i_out]),
                "bisected",
                float("nan"),
            )

        i = 0
        while i <= last:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 <= last and mask[j + 1]:
                j += 1
            start = (stops[0], "range", float("nan")) if i == 0 else bound(i, i - 1)
            end = (stops[last], "range", float("nan")) if j == last else bound(j, j + 1)
            for s, kind, t in (start, end):
                if kind != "range":
                    ends.append(ReachEnd(label, float(s), kind, float(t)))
            out.append((float(start[0]), float(end[0])))
            i = j + 1
        return out
