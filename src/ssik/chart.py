"""Charts of the self-motion manifold for redundant 7R arms.

A 7R arm holding a 6-DOF pose ``T`` still has a one-parameter family of
configurations: the *self-motion manifold* ``FK^-1(T)``. The closed-form 7R
solvers compute a chart of it internally, a discrete branch label plus a
redundancy coordinate ``t``, and until now discarded both at the ``Solution``
boundary, returning an unordered sample of ``q`` vectors. This module exposes
the chart itself.

A :class:`Chart` is one continuous branch ``q(t)`` of the manifold at one pose:

- ``label`` is a hashable branch identity, the sign bits of the closed-form
  subproblem roots the branch was built from. Two charts at two different poses
  with the same label are the same branch, which is what makes tracking a
  branch across poses a lookup instead of a nearest-neighbour search.
- ``q(t)`` evaluates the branch at any ``t``, scalar or batched; ``tangent(t)`` is
  its derivative as a unit direction and a rate (request B2).
- ``in_limits()`` is the domain under joint limits (request A3): sign-zeros of
  smooth per-joint margins, bracketed on a grid the shared resolver refines
  until no joint steps or bends more than 0.02 rad, then bisected.
- ``domain`` is the set of ``t`` where the branch exists (reachable), exact to
  bisection tolerance. Joint limits are *not* applied here: the chart is the
  geometric object, limits are a filter the caller composes on top.
- ``frame(t, metric)`` is the tangent plus a metric-orthogonal complement (D1);
  ``sample(n, metric)`` / ``length(metric)`` place points evenly in arc length
  in that metric, not in ``t`` (C1); ``restrict(fn)`` narrows the domain by any
  margin, such as a collision distance.

A :class:`SelfMotionManifold` is the fibre ``FK^-1(T)`` itself, presented as
its charts: every branch at one pose, plus the inverse map
:meth:`SelfMotionManifold.locate` -- given a configuration on the manifold, which
chart is it on and at what ``t``. Across charts it gives ``sheets()`` (the
components self-motion can reach), ``gap(a, b, limits=...)`` (distance between
holdable postures, minimised over the windings the joint box admits) and
``escape``; across poses, ``continue_from`` / :func:`track` / :func:`track_all`
(``Manipulator.solve_path``, request C2) follow branches by label and
:func:`drift_to_merge` finds where two sheets join or touch.

Two solver families are supported, each with its own redundancy coordinate:

``seven_r.spherical_shoulder`` (Franka Panda, FR3)
    ``t`` is the last joint ``q6`` (He & Liu 2021). Labels are
    ``(elbow, shoulder, wrist, interval)``: three subproblem sign bits plus the
    index of the reachable q6 interval the chart lives in. A branch slot is
    continuous within one reachable interval, so the interval index is part of
    the identity at a given pose; it is *not* stable across poses when the
    number of reachable intervals changes.

``seven_r.srs`` (KUKA iiwa and other exactly-concurrent SRS arms)
    ``t`` is the elbow swivel angle ``psi`` on the circle ``[-pi, pi)``,
    periodic. Labels are ``(elbow, shoulder_sign, wrist_sign)`` with ``elbow``
    the cosine-rule root index and the signs in ``{+1, -1}``. Every chart is
    defined on the whole circle.

``ikgeo.three_parallel`` (UR-class 6R: UR5e, Z1, CR5, ...)
    A 6R arm has no redundancy: the manifold at a regular pose is up to eight
    isolated points, each a zero-dimensional chart (``dimension == 0``,
    ``parameter == "point"``, ``q(t)`` ignores ``t``). Labels are the classic
    (shoulder, elbow, wrist) modes as the signs of the three factors of the
    singularity determinant, computed from the geometry
    (:func:`three_parallel_label`), so they change only across a singularity
    and :func:`track` follows a branch by label. UR arms are noncuspidal.

Approximate-class arms (``*_polished`` solvers) and the joint-lock family are
refused: their branches are not closed-form (LM polish) or carry no stable
discrete identity across samples (eigen-solver root ordering).

Cost model: ``charts(T)`` builds the family in microseconds (the Panda's
elbow-reachability arcs are closed form; nothing is scanned). ``locate`` and
``q(t)`` never need a domain. A chart's ``domain`` is computed on first access,
per reachable arc, and cached; enumerating ``family.charts`` computes every
domain, since a chart is listed only when its domain is non-empty.

When the native C++ extension is available (the Linux and macOS wheels), the
family is built and evaluated in C++ -- ``q(t)`` and ``locate(q)`` take about
2 microseconds, a Panda family builds in about 0.5 ms -- and ``native=False``
forces the pure-Python reference, which the parity tests pin the C++ path to.
:attr:`SelfMotionManifold.native` says which one you got.

Angles come back as the solver produces them, i.e. principal values from
``atan2``. Along a chart a joint may therefore jump by ``2*pi`` at a wrap; this
is a representative change, not a geometric one. :meth:`Chart.curve` unwraps
along ``t`` for callers who want a continuous curve.

Example::

    import ssik
    arm = ssik.Manipulator.from_urdf("panda.urdf", base="panda_link0", ee="panda_link8")
    T = arm.fk(q_now)
    family = arm.charts(T)
    chart, t = family.locate(q_now)       # the branch the arm is on, and where
    (ts, qs), *_ = chart.curve(200)       # the branch, continuous in t, per segment
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy

if TYPE_CHECKING:  # pragma: no cover
    from ssik._kinbody import KinBody

__all__ = [
    "Chart",
    "CuspidalityReport",
    "PathTrack",
    "SelfMotionManifold",
    "TrackStep",
    "charts",
    "cuspidality_report",
    "track",
    "track_all",
]

SUPPORTED_SOLVERS: frozenset[str] = frozenset(
    {"seven_r.spherical_shoulder", "seven_r.srs", "ikgeo.three_parallel"}
)

_TWO_PI = 2.0 * np.pi


def _wrap_pi(a: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = (a + np.pi) % _TWO_PI - np.pi
    return out


_FRAME_DEGENERATE = 1e-12  # chart.hpp kFrameDegenerate


def _frame_tail(
    raw: NDArray[np.float64], metric: NDArray[np.float64] | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The frame tail of :meth:`Chart.frame`, batched: raw tangents ``(N, 7)``
    -> unit directions ``(N, 7)`` and complements ``(N, 7, 6)``. The Python
    reference for ``_ssik_native.chart_frame``; parity in
    ``tests/test_chart_native.py``. ``NaN`` rows where the tangent is not
    finite or vanishes."""
    n = raw.shape[0]
    d: NDArray[np.float64] = np.full((n, 7), np.nan)
    v: NDArray[np.float64] = np.full((n, 7, 6), np.nan)
    with np.errstate(invalid="ignore"):
        rate: NDArray[np.float64] = np.linalg.norm(raw, axis=1)
        ok: NDArray[np.bool_] = np.isfinite(rate) & (rate > 0.0)
        if not ok.any():
            return d, v
        dk: NDArray[np.float64] = raw[ok] / rate[ok, None]
        # w = M d per row (M^T so a non-symmetric input still means M @ d).
        w: NDArray[np.float64] = dk if metric is None else dk @ metric.T
        nw: NDArray[np.float64] = np.linalg.norm(w, axis=1)
    good: NDArray[np.bool_] = np.isfinite(nw) & (nw > 0.0)
    if not good.all():  # a metric that annihilates the tangent: no frame there
        ok[np.flatnonzero(ok)[~good]] = False
        dk, w, nw = dk[good], w[good], nw[good]
    d[ok] = dk
    w = w / nw[:, None]
    # Householder H = I - 2 u u^T with H e0 = w  (u along e0 - w).
    u: NDArray[np.float64] = -w
    u[:, 0] += 1.0
    nu: NDArray[np.float64] = np.linalg.norm(u, axis=1)
    deg: NDArray[np.bool_] = nu < _FRAME_DEGENERATE  # w == e0: H = I, reached with u = 0
    u = np.where(deg[:, None], 0.0, u / np.where(deg, 1.0, nu)[:, None])
    h: NDArray[np.float64] = np.eye(7) - 2.0 * u[:, :, None] * u[:, None, :]
    v[ok] = h[:, :, 1:]
    return d, v


def _tangent_tail(
    raw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The normalizing tail of :meth:`Chart.tangent`, batched: raw tangents
    ``(N, 7)`` -> unit directions ``(N, 7)`` and rates ``(N,)``. The Python
    reference for ``_ssik_native.chart_tangent``; parity in
    ``tests/test_chart_native.py``."""
    rate: NDArray[np.float64] = np.linalg.norm(raw, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        direction: NDArray[np.float64] = raw / rate[:, None]
    return direction, rate


_native_fn_cache: dict[str, Any] = {}


def _native_fn(name: str) -> Any:
    """``_ssik_native.<name>`` if the extension ships it, else ``None``."""
    if name not in _native_fn_cache:
        ext = _native_ext()
        _native_fn_cache[name] = None if ext is None else getattr(ext, name, None)
    return _native_fn_cache[name]


def _wrap_dist(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Wrap-to-pi L-infinity distance between two joint vectors."""
    return float(np.max(np.abs(_wrap_pi(a - b))))


Limits = tuple[tuple[float, float], ...]


def _restrict_arcs(
    eval_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    domain: tuple[tuple[float, float], ...],
    fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    clearance: float,
    tol: float,
    samples: int,
    conservative: bool | None,
    periodic: bool,
) -> tuple[tuple[float, float], ...]:
    """Sub-intervals of ``domain`` where ``margin(q(t)) >= 0``.

    Brackets sign changes on a grid, then bisects each bracket to ``tol``. A
    boolean ``fn`` becomes a ``+/-1`` margin, which bisects to the same tolerance
    but only localises a step, hence the conservative shrink. ``NaN`` rows of
    ``q(t)`` (off-branch) are infeasible and never reach ``fn``.
    """

    def margin(ts: NDArray[np.float64]) -> NDArray[np.float64]:
        qs = np.atleast_2d(eval_fn(np.atleast_1d(ts)))
        live = np.all(np.isfinite(qs), axis=1)
        out = np.full(qs.shape[0], -np.inf)
        if not np.any(live):
            return out
        vals = np.asarray(fn(qs[live]))
        if vals.dtype == bool:
            out[live] = np.where(vals, 1.0, -1.0)
        else:
            out[live] = vals.astype(np.float64) - clearance
        return out

    is_bool = (
        bool(np.asarray(fn(np.atleast_2d(eval_fn(np.array([domain[0][0]]))))).dtype == bool)
        if domain
        else False
    )
    shrink = is_bool if conservative is None else conservative

    def bisect(t_in: float, t_out: float) -> float:
        """The boundary between a feasible ``t_in`` and an infeasible ``t_out``."""
        for _ in range(200):
            if abs(t_out - t_in) <= tol:
                break
            mid = 0.5 * (t_in + t_out)
            if margin(np.array([mid]))[0] >= 0.0:
                t_in = mid
            else:
                t_out = mid
        return t_in

    # One resolution for the whole chart: the conservative shrink has to mean the
    # same thing on every interval, and a narrow interval needs at least as fine a
    # grid as a wide one (sampling it at its own width would make the shrink
    # annihilate it, which is a resolution artefact rather than a finding).
    step = _TWO_PI / max(samples, 3)
    out: list[tuple[float, float]] = []
    for lo, hi in domain:
        n = max(int(np.ceil((hi - lo) / step)) + 1, 9)
        ts = np.linspace(lo, hi, n)
        ok = margin(ts) >= 0.0
        start: float | None = None
        for i, good in enumerate(ok):
            if good and start is None:
                start = lo if i == 0 else bisect(float(ts[i]), float(ts[i - 1]))
            elif not good and start is not None:
                out.append((start, bisect(float(ts[i - 1]), float(ts[i]))))
                start = None
        if start is not None:
            out.append((start, hi))
    if shrink:
        # Certify only what the grid resolves: an interval narrower than a step
        # cannot be vouched for by a step-function predicate, so it is dropped.
        out = [(a + step, b - step) for a, b in out]
    merged = [(a, b) for a, b in out if b - a > tol]
    if periodic and len(merged) >= 2:
        (a0, b0), (a1, b1) = merged[0], merged[-1]
        if abs(a0 - domain[0][0]) <= tol and abs(b1 - domain[-1][1]) <= tol:
            merged = [(a1 - _TWO_PI, b0), *merged[1:-1]]  # rejoin across the seam
    return tuple(merged)


class Chart:
    """One continuous branch ``q(t)`` of the self-motion manifold at one pose.

    ``label`` is a hashable branch identity (see the module docstring for the
    per-family meaning), ``parameter`` the redundancy coordinate's name
    (``"q6"`` or ``"swivel"``), ``periodic`` whether ``t`` lives on a circle,
    and :attr:`domain` the intervals on which the branch exists, computed on
    first access.
    """

    def __init__(
        self,
        *,
        label: tuple[int, ...],
        parameter: str,
        periodic: bool,
        eval_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        domain_fn: Callable[[], tuple[tuple[float, float], ...]],
        deriv_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        limit_arcs_fn: Callable[[Limits], tuple[tuple[float, float], ...]] | None = None,
        default_limits: Limits = (),
    ) -> None:
        self.label = label
        self.parameter = parameter
        self.periodic = periodic
        self.dimension = 0 if parameter == "point" else 1
        """0 for an isolated solution (6R arms), 1 for a one-parameter branch."""
        self._eval = eval_fn
        self._domain_fn = domain_fn
        self._deriv = deriv_fn
        self._limit_arcs = limit_arcs_fn
        self._default_limits = default_limits
        self._limit_cache: dict[Limits, tuple[tuple[float, float], ...]] = {}

    @cached_property
    def domain(self) -> tuple[tuple[float, float], ...]:
        """Intervals ``(t_lo, t_hi)`` on which the branch exists: the full circle
        for ``"swivel"``; for ``"q6"`` the exact elbow-fold arcs intersected with
        the shoulder and wrist gates, the latter refined to ~1e-10 rad.
        Validity of ``q(t)`` extends a tolerance-width (~1e-8 rad) past an
        elbow fold, since the subproblem gates carry a feasibility slack."""
        return self._domain_fn()

    def __repr__(self) -> str:
        return f"Chart(label={self.label}, parameter={self.parameter!r})"

    def q(self, t: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the branch. Scalar ``t`` -> ``(7,)``; ``(N,)`` -> ``(N, 7)``.

        Rows are ``NaN`` where the branch does not exist at ``t``.

        Accuracy: the closed form closes FK to ~1e-13 for ``t`` farther than
        1e-5 rad from a fold of the branch (a domain end). Within that distance
        the residual grows like the square root of the distance, to ~1e-7 at
        the fold itself (measured on the Panda: 2.5e-8 at 1e-6, 9e-8 at 1e-9),
        because a double root of a subproblem is only determined to
        ``sqrt(eps)``. A controller that must sit exactly on a fold should
        polish with one Newton step; the tangent's ``rate`` diverging is the
        same fold seen from the coordinate side.
        """
        ts = np.asarray(t, dtype=np.float64)
        scalar = ts.ndim == 0
        out = self._eval(np.atleast_1d(ts))
        return out[0] if scalar else out

    def in_limits(self, limits: ArrayLike | None = None) -> tuple[tuple[float, float], ...]:
        """Sub-intervals of :attr:`domain` on which every joint is inside its
        range (request A3): the chart's domain under joint limits, exact --
        boundaries are the sign-zeros of the smooth ``cos(q_i(t) - c_i) -
        cos(h_i)`` margins, bracketed on a grid and bisected, the same
        computation ssik's in-limits resolvers run internally.

        :param limits: ``(7, 2)`` per-joint ``(lower, upper)``; ``None`` uses the
            chain's own limits (a limitless joint counts as ``[-pi, pi]``).
        Cached per limits. Empty when no point of the branch is in limits.
        One connected self-motion sheet can come back as several arcs: the
        pieces the limits leave of it, which is what a controller can hold.

        Angles are compared modulo ``2*pi``: ``q(t)`` returns principal values,
        and a joint is in limits when *some* ``q_i + 2*pi*k`` is (the
        representative ``ssik.postprocess.wrap_to_limits`` would pick). A chart
        is a geometric branch; on a joint whose range exceeds one turn (UR
        arms, ``+-2*pi``) the windings ``q_i + 2*pi*k`` are distinct postures
        of that branch, and which one the arm can reach from where it is is
        the caller's question -- ``ssik.postprocess.rewrap_to_seed`` picks the
        winding nearest a seed.
        """
        if self._limit_arcs is None:
            return ()
        lims = (
            self._default_limits
            if limits is None
            else tuple((float(lo), float(hi)) for lo, hi in np.asarray(limits, dtype=np.float64))
        )
        arcs = self._limit_cache.get(lims)
        if arcs is None:
            arcs = self._limit_cache[lims] = self._limit_arcs(lims)
        return arcs

    def tangent(self, t: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The branch's tangent at ``t``: ``(direction, rate)`` with
        ``dq/dt = rate * direction``, ``direction`` a unit 7-vector oriented by
        increasing ``t`` and ``rate = |dq/dt|``. Scalar ``t`` -> ``((7,), ())``;
        ``(N,)`` -> ``((N, 7), (N,))``.

        The split matters at a fold of the joint coordinates (a gimbal lock of
        the shoulder or wrist decomposition): the manifold is smooth there but
        the coordinate ``t`` has a critical point, so ``rate`` diverges while
        ``direction`` stays well defined. Both are ``NaN`` where the branch does
        not exist.

        Exact on both families. SRS: the closed form (rotor derivative of the
        swivel orbit pushed through the two 3-axis decompositions).
        Spherical-shoulder: implicit differentiation of ``FK(q(t)) = T`` with
        ``q_6 = t`` -- ``J[:, :6] dq' = -J[:, 6]`` on the chain's spatial
        Jacobian, a 6x6 solve, singular exactly at a fold of ``t``.

        The tail past the raw tangent runs natively where the extension is
        present (``_ssik_native.chart_tangent``); :func:`_tangent_tail` is the
        reference.
        """
        ts = np.asarray(t, dtype=np.float64)
        scalar = ts.ndim == 0
        assert self._deriv is not None
        raw = np.ascontiguousarray(self._deriv(np.atleast_1d(ts)), dtype=np.float64)
        native = _native_fn("chart_tangent")
        if native is None:
            direction, rate = _tangent_tail(raw)
        else:
            out: tuple[NDArray[np.float64], NDArray[np.float64]] = native(raw)
            direction, rate = out
        return (direction[0], rate[0]) if scalar else (direction, rate)

    def frame(
        self, t: ArrayLike, metric: ArrayLike | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The chart frame at ``t`` (request D1): the unit tangent ``d`` and a
        ``(7, 6)`` basis ``V`` of its complement, orthogonal to ``d`` in
        ``metric``. Scalar ``t`` -> ``((7,), (7, 6))``; ``(N,)`` ->
        ``((N, 7), (N, 7, 6))``, as :meth:`tangent`.

        With ``metric=None`` the complement is ``ker(J)^perp = row(J)``, the
        kinematic split. With ``metric=M`` (SPD ``(7, 7)``, e.g. the mass matrix
        from gafro) it is the ``M``-orthogonal complement, ``d^T M v = 0``: the
        dynamically consistent split, so a task correction commanded in
        ``span(V)`` does no work along the self-motion. The columns of ``V`` are
        Euclidean-orthonormal (not ``M``-orthonormal) and are built by the
        Householder reflection carrying ``e_0`` onto ``w = M d / |M d|``, so
        ``V`` varies continuously along the arc except at the single direction
        ``w == +e_0``, where ``u = e_0 - w`` vanishes and the limit depends on
        the direction of approach (``w == -e_0`` is benign: ``u -> e_0`` from
        every side). That defect cannot be removed, only moved: a continuous
        ``V(w)`` on the whole sphere is a parallelisation of ``S^6``, which does
        not exist, and puncturing one point is the best possible -- an arc
        misses a codimension-6 point, where the ``sign(w_0)`` variant that fixes
        the conditioning would put a codimension-1 equator in its way. Unlike an
        SVD's per-evaluation sign flips, this is a measure-zero event. ``NaN``
        off the branch.

        The tail past the tangent runs natively where the extension is present
        (``_ssik_native.chart_frame``); :func:`_frame_tail` is the reference.
        """
        ts = np.asarray(t, dtype=np.float64)
        scalar = ts.ndim == 0
        assert self._deriv is not None
        raw = np.ascontiguousarray(self._deriv(np.atleast_1d(ts)), dtype=np.float64)
        m = None if metric is None else np.ascontiguousarray(metric, dtype=np.float64)
        native = _native_fn("chart_frame")
        if native is None:
            d, v = _frame_tail(raw, m)
        else:
            out: tuple[NDArray[np.float64], NDArray[np.float64]] = native(raw, m)
            d, v = out
        return (d[0], v[0]) if scalar else (d, v)

    def restrict(
        self,
        fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *,
        clearance: float = 0.0,
        tol: float = 1e-10,
        samples: int = 181,
        conservative: bool | None = None,
    ) -> Chart:
        """The same branch with its domain narrowed to where ``fn`` allows it.

        The chart is the geometric object; a constraint that is not a property of
        the kinematics -- self-collision, a task-space obstacle, a keep-out region
        -- is a restriction the caller composes on top. This is that composition,
        and it returns a :class:`Chart`, so ``q``, ``tangent``, ``contains``,
        ``curve`` and :meth:`in_limits` all respect it, and restrictions chain::

            reachable = chart.restrict(clearance_fn, clearance=0.02)
            usable = reachable.restrict(keep_out_fn)

        ``fn`` is batched: it takes ``(N, dof)`` and returns ``(N,)``. Two forms,
        with different accuracy:

        * **margin** (float) -- ``>= 0`` feasible. Boundaries are the zeros of
          ``fn(q(t)) - clearance``, bracketed on the sample grid and bisected to
          ``tol``, so they are exact to ``tol`` regardless of ``samples``. This is
          the form a collision backend already speaks (FCL signed distance,
          MuJoCo contact depth), and ``clearance`` buys a safety band for free.
        * **boolean** -- ``True`` feasible. Wrapped as a ``+/-1`` margin, so the
          bisection still converges, but on a step function: the boundary is only
          as good as the assumption that no forbidden region is narrower than the
          grid step. ``conservative`` therefore defaults to ``True`` here and
          shrinks every kept interval by one step; pass ``False`` to opt out.

        ``q(t)`` is ``NaN`` off the branch, and ``fn`` is never called there --
        those samples are infeasible by construction.

        A zero-dimensional chart (a 6R arm's isolated solution) is kept or dropped
        whole; an empty result means this branch is entirely forbidden, and
        :meth:`SelfMotionManifold.restrict` drops such charts from the manifold.

        :param fn: batched margin or predicate over configurations.
        :param clearance: required margin, in the margin's own units. Ignored for
            a boolean ``fn``.
        :param tol: bisection tolerance on the boundary, in ``t``.
        :param samples: grid points per ``2*pi`` of parameter used to bracket
            boundaries. Raise it when the forbidden regions are narrow.
        :param conservative: shrink each kept interval by one grid step on both
            sides. Defaults to ``True`` for a boolean ``fn`` and ``False`` for a
            margin, where the bisected boundary is already exact.
        :returns: a new :class:`Chart`, same ``label`` and same ``q(t)``.
        """
        arcs = _restrict_arcs(
            self._eval,
            self.domain,
            fn,
            clearance=clearance,
            tol=tol,
            samples=samples,
            conservative=conservative,
            periodic=self.periodic,
        )
        parent_limit_arcs = self._limit_arcs

        def limit_arcs(lims: Limits) -> tuple[tuple[float, float], ...]:
            if parent_limit_arcs is None:
                return ()
            from ssik.solvers.seven_r._feasible_param import intersect

            return tuple(intersect(list(parent_limit_arcs(lims)), list(arcs)))

        return Chart(
            label=self.label,
            parameter=self.parameter,
            periodic=self.periodic,
            eval_fn=self._eval,
            domain_fn=lambda: arcs,
            deriv_fn=self._deriv,
            limit_arcs_fn=None if parent_limit_arcs is None else limit_arcs,
            default_limits=self._default_limits,
        )

    def contains(self, t: float, tol: float = 1e-9) -> bool:
        """True iff ``t`` lies in :attr:`domain` (wrapped first when periodic)."""
        tt = float(_wrap_pi(np.asarray(t))) if self.periodic else float(t)
        return any(lo - tol <= tt <= hi + tol for lo, hi in self.domain)

    def curve(self, n: int = 181) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """Dense sample of the branch, continuous in ``t``, one segment per
        domain interval.

        Each segment is ``(ts, qs)``: ``ts`` of shape ``(N,)`` with about ``n``
        points per full ``2*pi`` of parameter, and ``qs`` of shape ``(N, 7)``
        unwrapped along ``t`` so no joint jumps by ``2*pi`` between consecutive
        rows. Segments are separate pieces of the branch; their end rows are
        not adjacent on the manifold.
        """
        segments: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for lo, hi in self.domain:
            m = max(round(n * (hi - lo) / _TWO_PI), 2)
            ts = np.linspace(lo, hi, m, endpoint=not (self.periodic and hi - lo >= _TWO_PI - 1e-9))
            qs = self._eval(ts)
            ok = np.all(np.isfinite(qs), axis=1)
            ts, qs = ts[ok], qs[ok]
            if ts.shape[0] == 0:
                continue
            segments.append((ts, np.unwrap(qs, axis=0)))
        return segments

    def length(
        self,
        metric: Metric | None = None,
        *,
        limits: ArrayLike | bool | None = None,
        tol: float = 1e-5,
    ) -> float:
        """Arc length of the branch, summed over its pieces (request C1).

        ``metric`` is ``None`` (Euclidean joint space), a constant SPD
        ``(dof, dof)`` matrix, or a batched callable ``(N, dof) -> (N, dof, dof)``
        -- the mass matrix from gafro gives kinetic-energy distance. ``limits``
        restricts the branch to :meth:`in_limits` first (``True`` for the chain's
        own limits, or a ``(dof, 2)`` array). See :meth:`sample` for ``tol``.
        Zero on a zero-dimensional chart.
        """
        return float(sum(s[-1] for _, _, s in self._polylines(metric, limits, tol)))

    def sample(
        self,
        n: int,
        metric: Metric | None = None,
        *,
        limits: ArrayLike | bool | None = None,
        tol: float = 1e-5,
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """About ``n`` points spaced uniformly in arc length (request C1).

        Uniform in ``t`` is not uniform on the manifold: at a fold of the
        coordinate the rate ``|dq/dt|`` diverges (see :meth:`tangent`), so a
        ``t``-grid bunches up in joint space in the middle of a branch and
        leaves gaps at its ends. This places points at equal arc length
        instead, in ``metric`` (see :meth:`length`), so a sum over the samples
        converges to the curve's value rather than to a sample-density artefact.

        Returns one ``(ts, qs)`` per piece, like :meth:`curve` (``qs`` unwrapped
        along the piece). The spacing ``h = L / n`` is shared by all pieces; an
        open piece gets ``round(L_i / h) + 1`` points (at least two) including
        both ends, a closed one ``round(L_i / h)`` (at least three) with the
        seam spaced like the rest. Every point is ``q(t)`` evaluated exactly on
        the chart; only its ``t`` comes from the arc-length inversion.

        The length is the midpoint-metric sum over a polyline of ``q(t)``,
        refined by bisection until no chord's midpoint strays more than ``tol``
        (rad) from the curve; near a fold ``q`` goes like ``sqrt(t - t_0)`` and
        the refinement concentrates there on its own. Spacing is uniform to
        about ``tol``. A position-dependent ``metric`` is evaluated once per
        chord, batched.
        """
        if n < 1:
            raise ValueError("sample: n must be >= 1")
        lines = self._polylines(metric, limits, tol)
        total = sum(s[-1] for _, _, s in lines)
        if not lines:
            return []
        if self.dimension == 0 or total <= 0.0:
            return [(ts[:1], qs[:1]) for ts, qs, _ in lines]
        h = total / n
        out: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
        for ts, qs, s in lines:
            L = s[-1]
            closed = _closes(qs)
            if closed:
                m = max(round(L / h), 3)
                targets = np.arange(m) * (L / m)
            else:
                m = max(round(L / h) + 1, 2)
                targets = np.linspace(0.0, L, m)
            k = np.clip(np.searchsorted(s, targets, side="right") - 1, 0, s.shape[0] - 2)
            ds = s[k + 1] - s[k]
            f = np.where(ds > 0.0, (targets - s[k]) / np.where(ds > 0.0, ds, 1.0), 0.0)
            t_new = ts[k] + f * (ts[k + 1] - ts[k])
            q_new = self._eval_t(t_new)
            ok = np.all(np.isfinite(q_new), axis=1)
            out.append((t_new[ok], np.unwrap(q_new[ok], axis=0)))
        return out

    def _eval_t(self, ts: NDArray[np.float64]) -> NDArray[np.float64]:
        """``q(t)`` accepting ``t`` past ``pi`` on a periodic chart (a piece
        joined across the seam)."""
        return self._eval(_wrap_pi(ts) if self.periodic else ts)

    def _pieces(self, limits: ArrayLike | bool | None) -> list[tuple[float, float]]:
        """The intervals to measure: the domain, or its in-limits part; on a
        periodic chart two pieces meeting at the ``+-pi`` seam are one piece."""
        if limits is None or limits is False:
            arcs = list(self.domain)
        else:
            arcs = list(self.in_limits(None if limits is True else limits))
        arcs.sort()
        if (
            self.periodic
            and len(arcs) >= 2
            and abs(arcs[0][0] + np.pi) < 1e-9
            and abs(arcs[-1][1] - np.pi) < 1e-9
        ):
            first = arcs.pop(0)
            lo, _ = arcs.pop()
            arcs.append((lo, first[1] + _TWO_PI))
        return arcs

    def _polylines(
        self, metric: Metric | None, limits: ArrayLike | bool | None, tol: float
    ) -> list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
        """Per piece ``(ts, qs, s)``: an adaptive polyline of the branch and the
        cumulative metric length along it."""
        if self.dimension == 0:
            q0 = self.q(0.0)
            if not np.all(np.isfinite(q0)):
                return []
            return [(np.zeros(1), q0[None, :], np.zeros(1))]
        mfn = _metric_fn(metric)
        lines = []
        for lo, hi in self._pieces(limits):
            ts, qs = _adaptive_polyline(self._eval_t, lo, hi, tol)
            if ts.shape[0] < 2:
                continue
            dq = _wrap_pi(qs[1:] - qs[:-1])
            if mfn is None:
                seg = np.linalg.norm(dq, axis=1)
            else:
                M = mfn(qs[:-1] + 0.5 * dq)
                seg = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", dq, M, dq), 0.0))
            lines.append((ts, np.unwrap(qs, axis=0), np.concatenate(([0.0], np.cumsum(seg)))))
        return lines


#: A joint-space metric for :meth:`Chart.length` / :meth:`Chart.sample`: a
#: constant SPD matrix or a batched ``(N, dof) -> (N, dof, dof)`` callable.
Metric = NDArray[np.float64] | Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _metric_fn(
    metric: Metric | None,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]] | None:
    if metric is None:
        return None
    if callable(metric):
        fn = metric

        def batched(qs: NDArray[np.float64]) -> NDArray[np.float64]:
            M = np.asarray(fn(qs), dtype=np.float64)
            if M.shape != (qs.shape[0], qs.shape[1], qs.shape[1]):
                raise ValueError(
                    f"metric callable must map (N, dof) -> (N, dof, dof); got {M.shape} "
                    f"for input {qs.shape}"
                )
            return M

        return batched
    M0 = np.asarray(metric, dtype=np.float64)
    return lambda qs: np.broadcast_to(M0, (qs.shape[0], *M0.shape))


def _closes(qs: NDArray[np.float64]) -> bool:
    """True when an (unwrapped) polyline ends where it began, modulo ``2*pi``
    per joint: a piece that is a loop on the manifold."""
    return qs.shape[0] > 2 and float(np.max(np.abs(_wrap_pi(qs[-1] - qs[0])))) < 1e-7


def _adaptive_polyline(
    eval_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    lo: float,
    hi: float,
    tol: float,
    n0: int = 65,
    max_levels: int = 48,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Polyline of ``q(t)`` on ``[lo, hi]``, every chord bisected until its
    midpoint lies within ``tol`` of the curve. Non-finite rows are dropped."""
    m = max(round(n0 * (hi - lo) / _TWO_PI), 9)
    ts = np.linspace(lo, hi, m)
    qs = eval_fn(ts)
    for _ in range(max_levels):
        ok = np.all(np.isfinite(qs), axis=1)
        ts, qs = ts[ok], qs[ok]
        if ts.shape[0] < 2:
            break
        mids = 0.5 * (ts[:-1] + ts[1:])
        qm = eval_fn(mids)
        dq = _wrap_pi(qs[1:] - qs[:-1])
        dev = np.linalg.norm(_wrap_pi(qm - (qs[:-1] + 0.5 * dq)), axis=1)
        bad = np.isfinite(dev) & (dev > tol)
        if not bad.any():
            break
        order = np.argsort(np.concatenate((ts, mids[bad])), kind="stable")
        ts = np.concatenate((ts, mids[bad]))[order]
        qs = np.concatenate((qs, qm[bad]))[order]
    return ts, qs


class SelfMotionManifold:
    """The self-motion manifold ``FK^-1(T)`` at one pose, presented as charts.

    The object is the fibre, not merely a container of charts: :meth:`gap`,
    :meth:`sheets`, :meth:`escape` and :meth:`singularity_margin` are questions
    about the manifold, and the charts are how it is coordinatised. Charts are
    created on demand and cached, so :meth:`locate` never pays for charts it does
    not return; :attr:`charts` enumerates the non-empty ones, which requires
    every domain.
    """

    def __init__(
        self,
        *,
        parameter: str,
        periodic: bool,
        param_of: Callable[[NDArray[np.float64]], float],
        make_chart: Callable[[int], Chart],
        nonempty: Callable[[], list[int]],
        locate: Callable[[NDArray[np.float64], float], tuple[int, float]] | None = None,
        native: bool = False,
        kb: KinBody | None = None,
        T: NDArray[np.float64] | None = None,
        solver_name: str | None = None,
    ) -> None:
        self.parameter = parameter
        self.periodic = periodic
        self.dimension = 0 if parameter == "point" else 1
        self.native = native
        """True when this manifold is backed by the native C++ extension."""
        self._param_of = param_of
        self._make_chart = make_chart
        self._nonempty = nonempty
        self._locate = locate
        self._kb = kb
        self._T = T
        self._solver_name = solver_name
        self._cache: dict[int, Chart] = {}

    def _chart(self, i: int) -> Chart:
        c = self._cache.get(i)
        if c is None:
            c = self._cache[i] = self._make_chart(i)
        return c

    @cached_property
    def charts(self) -> tuple[Chart, ...]:
        """The non-empty charts, in enumeration order (deterministic per pose)."""
        return tuple(self._chart(i) for i in self._nonempty())

    def __len__(self) -> int:
        return len(self.charts)

    def __iter__(self) -> Iterator[Chart]:
        return iter(self.charts)

    def restrict(
        self,
        fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        **kwargs: Any,
    ) -> SelfMotionManifold:
        """:meth:`Chart.restrict` applied to every chart, dropping those it empties.

        The manifold-level entry point for planning: hand it one predicate over
        configurations and get back the pose's manifold with the forbidden parts
        removed, ready for :meth:`locate` and :attr:`charts`. Keyword arguments are
        forwarded to :meth:`Chart.restrict` unchanged.

        :meth:`locate` on the result still answers for the *restricted* charts
        only, so a configuration in a forbidden region locates as ``None`` -- which
        is the question a controller is asking.
        """
        kept = [c.restrict(fn, **kwargs) for c in self.charts]
        kept = [c for c in kept if c.domain]
        return SelfMotionManifold(
            parameter=self.parameter,
            periodic=self.periodic,
            param_of=self._param_of,
            make_chart=lambda i: kept[i],
            nonempty=lambda: list(range(len(kept))),
            locate=None,  # the native locate knows nothing of the restriction
            native=False,
            # Carried through so the pose-level methods (gap, escape, drift_to_merge)
            # keep working on a restricted family; only the charts are narrower.
            kb=self._kb,
            T=self._T,
            solver_name=self._solver_name,
        )

    def labels(self) -> list[tuple[int, ...]]:
        return [c.label for c in self.charts]

    def by_label(self, label: tuple[int, ...]) -> Chart | None:
        """The chart with this label at this pose, or ``None``."""
        for c in self.charts:
            if c.label == label:
                return c
        return None

    def continue_from(
        self, chart: Chart, t: float, q: ArrayLike, max_step: float = 0.5
    ) -> tuple[Chart, float, str]:
        """Continuation (request B4): the point of this pose's manifold that a
        controller on ``chart`` at coordinate ``t`` (configuration ``q``) moves
        to. Returns ``(chart, t, event)`` with ``event`` one of

        - ``"label"``: the chart with the same label exists here and holds a
          point within ``max_step`` (wrap-Linf) of ``q`` at the same ``t`` -- a
          lookup, no search. Sound where the locked slice is noncuspidal (see
          :func:`cuspidality_report`), which the closed-form decomposition
          guarantees for the spherical-shoulder and SRS families.
        - ``"fold"``: that chart is gone or too far, but another chart holds a
          point within ``max_step``: the branch passed a fold and was relabelled.
        - ``"collision"``: nothing within ``max_step`` -- the branch collided
          with another and vanished (a singularity crossing); the returned chart
          is the nearest one and the caller must decide.

        ``t`` is kept, so the coordinate stays continuous along the path; on a
        periodic chart it is wrapped.
        """
        qa = np.asarray(q, dtype=np.float64)
        tt = float(_wrap_pi(np.asarray(t))) if self.periodic else float(t)
        same = self.by_label(chart.label)
        if same is not None:
            qc = same.q(tt)
            if np.all(np.isfinite(qc)) and _wrap_dist(qc, qa) <= max_step:
                return same, tt, "label"
        best: tuple[float, Chart] | None = None
        for c in self.charts:
            qc = c.q(tt)
            if not np.all(np.isfinite(qc)):
                continue
            dist = _wrap_dist(qc, qa)
            if best is None or dist < best[0]:
                best = (dist, c)
        if best is None:
            # No chart of this pose exists at t: the reachable range of the
            # coordinate moved past the branch (an elbow fold in t). Clamp to the
            # nearest domain end of the nearest chart and report the collision.
            cand: list[tuple[float, Chart, float]] = []
            for c in self.charts:
                for lo, hi in c.domain:
                    tc = min(max(tt, lo), hi)
                    qc = c.q(tc)
                    if np.all(np.isfinite(qc)):
                        cand.append((_wrap_dist(qc, qa), c, tc))
            if not cand:
                raise ValueError("continue_from: this pose has no chart at all")
            dist, c, tc = min(cand, key=lambda x: x[0])
            return c, tc, "collision"
        return best[1], tt, "fold" if best[0] <= max_step else "collision"

    def gap(
        self,
        a: Chart,
        b: Chart,
        n_samples: int = 91,
        *,
        limits: ArrayLike | bool | None = None,
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """The joint-space gap between two charts of this pose: the Euclidean
        distance between their nearest points, and those points.
        A grid of ``n_samples`` per piece brackets the pair; a bounded
        minimisation over the two coordinates then refines it.

        ``limits=None`` measures on the torus (every joint wrapped to
        ``[-pi, pi)``): right for the geometry, blind to the joint box. With
        ``limits`` (``True`` for the chain's own, or ``(dof, 2)``) only postures
        the arm can hold count: each chart is cut to :meth:`Chart.in_limits`,
        and the distance is the plain one between *in-box representatives*,
        minimised over the windings ``q_i + 2*pi*k`` that lie in the box. That
        one rule gets both cases right that wrap-to-pi gets half wrong: where a
        joint's range is under a turn only one winding is holdable and two
        points a turn apart are one posture; where it exceeds a turn (UR arms,
        ``+-2*pi``) both windings are holdable, distinct postures, and the
        nearer pair is what counts. The returned points are those in-box
        representatives, so ``|qb - qa|`` is the gap. ``inf`` (and ``NaN``
        points) when either chart has no posture in the box.
        """
        box = None if limits is None or limits is False else _box(a, limits)
        pa, ta = _sample_points(a, n_samples, with_t=True, limits=limits)
        pb, tb = _sample_points(b, n_samples, with_t=True, limits=limits)
        nan = np.full(a.q(0.0).shape[-1], np.nan)
        if pa.shape[0] == 0 or pb.shape[0] == 0:
            return float("inf"), nan, nan

        def dist2(qa: NDArray[np.float64], qb: NDArray[np.float64]) -> NDArray[np.float64]:
            if box is None:
                return np.asarray(np.sum(_wrap_pi(qb - qa) ** 2, axis=-1))
            return _inbox_dist2(qa, qb, box)[0]

        d2 = np.stack([dist2(qa[None, :], pb) for qa in pa])
        if not np.isfinite(d2).any():
            return float("inf"), nan, nan
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        qa, qb = pa[i], pb[j]
        free = [(a, float(ta[i])), (b, float(tb[j]))]
        free = [(c, t) for c, t in free if c.dimension == 1]
        if free:
            from scipy.optimize import minimize  # type: ignore[import-untyped]

            def bounds_of(c: Chart, t: float) -> tuple[float, float]:
                for lo, hi in c._pieces(limits):
                    if lo - 1e-9 <= t <= hi + 1e-9:
                        return lo, hi
                return t, t

            def points(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
                qs = []
                k = 0
                for c, q_fixed in ((a, qa), (b, qb)):
                    if c.dimension == 1:
                        qs.append(c._eval_t(np.array([float(x[k])]))[0])
                        k += 1
                    else:
                        qs.append(q_fixed)
                return qs[0], qs[1]

            def objective(x: NDArray[np.float64]) -> float:
                ua, ub = points(x)
                if not (np.all(np.isfinite(ua)) and np.all(np.isfinite(ub))):
                    return 1e6
                v = float(dist2(ua[None, :], ub[None, :])[0])
                return v if np.isfinite(v) else 1e6

            x0 = np.array([t for _c, t in free])
            # Tight tolerances: the gradient of a squared distance vanishes with the
            # distance, and at scipy's defaults the search stopped ~1e-3 rad short
            # of a touching pair -- too coarse for drift_to_merge to call a merge.
            res = minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=[bounds_of(c, t) for c, t in free],
                options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 500},
            )
            if res.fun < objective(x0):
                qa, qb = points(res.x)
        if box is None:
            return float(np.sqrt(np.sum(_wrap_pi(qb - qa) ** 2))), qa, qb
        d2v, ra, rb = _inbox_dist2(qa[None, :], qb[None, :], box)
        # The nearest pair can sit at a cusp in t narrow enough that the search
        # restricted to the in-limits pieces starts in the wrong cell and stops
        # short (0.085 where a pair 8.4e-4 apart was in the box, on the iiwa14).
        # The torus search's pair, measured in the box, is a second candidate.
        _, ta_q, tb_q = self.gap(a, b, n_samples)
        if np.all(np.isfinite(ta_q)) and np.all(np.isfinite(tb_q)):
            d2t, sa, sb = _inbox_dist2(ta_q[None, :], tb_q[None, :], box)
            if d2t[0] < d2v[0]:
                d2v, ra, rb = d2t, sa, sb
        return float(np.sqrt(d2v[0])), ra[0], rb[0]

    def sheets(self, tol: float = 1e-6) -> list[tuple[Chart, ...]]:
        """Connected components of the manifold (request B3): charts glued where
        they meet, i.e. whose :meth:`gap` is below ``tol`` (partner slots share a
        fold point at a domain end). Each sheet is a tuple of charts; the
        arm's self-motion can reach every point of the sheet it is on and no
        point of any other. Isolated solutions (6R arms) are one-chart sheets.
        """
        cs = list(self.charts)
        parent = list(range(len(cs)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if cs[i].dimension == 1 and self.gap(cs[i], cs[j])[0] <= tol:
                    parent[find(i)] = find(j)
        groups: dict[int, list[Chart]] = {}
        for i, c in enumerate(cs):
            groups.setdefault(find(i), []).append(c)
        return [tuple(g) for g in groups.values()]

    def escape(
        self, a: Chart, b: Chart, step: float = 1e-4
    ) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64]]:
        """Escape direction between two charts (request D2): the unit task-space
        twist along which the two branches approach each other fastest, and a
        first-order estimate of how far the pose must drift along it for them to
        merge.

        Meaningful between charts on different :meth:`sheets`: they can only be
        connected by moving the pose to a fold, where two branches coincide
        (partner charts on one sheet already touch, gap zero). The direction
        is the negative gradient of :meth:`gap` with
        respect to the pose, taken by central differences over the six spatial
        twist components (``se3_exp(dx) @ T``, the convention of
        ``ssik.refinement.kinbody_jacobian``), and the magnitude is
        ``gap / |grad gap|``, the distance to a zero gap if the gap kept closing
        at this rate.

        **The magnitude overshoots by about a factor of two, and systematically
        so.** Two branches annihilate at a fold, where they are the two roots of
        a quadratic, so the gap closes like ``C * sqrt(s_c - s)`` rather than
        linearly; a first-order step from ``s = 0`` then lands at ``2 * s_c``
        exactly. Measured over 204 sheet pairs the closest approach sits at a
        median ``0.50`` of this magnitude, with the fraction uncorrelated with
        the gap (r = -0.01) -- as it must be, the factor coming from the
        exponent alone. So use the direction as returned and line-search the
        distance, starting near ``0.5 * magnitude``; the minimum is sharp in
        ``s`` because ``d(gap)/ds`` diverges at the merge.
        :func:`drift_to_merge` walks it. Returns
        ``(direction, magnitude, q_a, q_b)`` with the nearest pair at ``T``.
        Model-free: twelve chart families of neighbouring poses.
        """
        if self._kb is None or self._T is None:
            raise ValueError("escape needs the manifold's chain and pose")
        gap0, qa, qb = self.gap(a, b)
        grad = np.zeros(6)
        for k in range(6):
            g = []
            for sgn in (1.0, -1.0):
                dx = np.zeros(6)
                dx[k] = sgn * step
                fam = charts(self._kb, se3_exp(dx) @ self._T, solver_name=self._solver_name)
                ca, cb = fam.by_label(a.label), fam.by_label(b.label)
                g.append(fam.gap(ca, cb)[0] if ca is not None and cb is not None else np.nan)
            grad[k] = (g[0] - g[1]) / (2 * step)
        if not np.all(np.isfinite(grad)):
            grad = np.where(np.isfinite(grad), grad, 0.0)
        n = float(np.linalg.norm(grad))
        if n == 0.0:
            return np.zeros(6), float("inf"), qa, qb
        return -grad / n, gap0 / n, qa, qb

    def singularity_margin(self, q: ArrayLike) -> float:
        """Smallest singular value of the chain's 6xN spatial Jacobian at ``q``:
        zero exactly at a kinematic singularity, where branches meet. For a 6R
        arm this is the margin a controller watches while it tracks a branch."""
        from ssik.refinement import kinbody_jacobian

        if self._kb is None:
            return float("nan")
        return float(
            np.linalg.svd(
                kinbody_jacobian(self._kb, np.asarray(q, dtype=np.float64)), compute_uv=False
            ).min()
        )

    def param_of(self, q: ArrayLike) -> float:
        """The redundancy coordinate of a configuration ``q``.

        Pure geometry, no branch search: ``q6`` itself for the spherical-shoulder
        family, the elbow's swivel angle on the shoulder-wrist circle for SRS.
        Meaningful only when ``FK(q)`` is this manifold's pose.
        """
        return self._param_of(np.asarray(q, dtype=np.float64))

    def locate(self, q: ArrayLike, tol: float = 1e-6) -> tuple[Chart, float] | None:
        """Inverse chart map: the chart ``q`` lies on and its coordinate there.

        Evaluates the branches at ``t = param_of(q)`` and returns the first
        whose value matches ``q`` to ``tol`` in wrap-to-pi L-infinity distance,
        or ``None`` when none does (``q`` is not on this pose's manifold to that
        tolerance). One closed-form evaluation; no domain is computed.
        """
        qa = np.asarray(q, dtype=np.float64)
        if self._locate is not None:
            idx, t = self._locate(qa, tol)
            return (self._chart(idx), t) if idx >= 0 else None
        t = self._param_of(qa)
        for chart in self.charts:
            if not chart.contains(t):
                continue
            qc = chart.q(t)
            if np.all(np.isfinite(qc)) and _wrap_dist(qc, qa) <= tol:
                return chart, t
        return None


_SOLVER_OF = {
    "q6": "seven_r.spherical_shoulder",
    "swivel": "seven_r.srs",
    "point": "ikgeo.three_parallel",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def charts(
    kb: KinBody,
    T_target: ArrayLike,
    *,
    solver_name: str | None = None,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    native: bool = True,
) -> SelfMotionManifold:
    """Enumerate the self-motion charts of ``kb`` at pose ``T_target``.

    :param kb: POE-normalised 7R :class:`~ssik.internals.KinBody`.
    :param T_target: 4x4 SE(3) pose in the base frame.
    :param solver_name: the dispatched solver (``Manipulator.solver_name`` or an
        artifact's ``SOLVER_NAME``). ``None`` runs the dispatcher.
    :param policy: tolerance policy forwarded to the subproblem feasibility
        gates.
    :param native: use the C++ extension when it is available (a performance
        hint, never an error when it is not); ``False`` forces the Python
        reference.
    :raises NotImplementedError: for a solver family without a closed-form
        chart (see the module docstring).
    """
    T = np.asarray(T_target, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"charts expected T_target of shape (4, 4), got {T.shape}")
    if solver_name is None:
        from ssik.core.dispatcher import dispatch

        solver_name = dispatch(kb, policy).solver_name
    ext = _native_ext() if native else None
    if solver_name == "seven_r.spherical_shoulder":
        if ext is not None:
            return _spherical_shoulder_family_native(ext, kb, T, policy)
        return _spherical_shoulder_family(kb, T, policy)
    if solver_name == "seven_r.srs":
        if ext is not None:
            fam = _srs_family_native(ext, kb, T)
            if fam is not None:
                return fam
        return _srs_family(kb, T, policy)
    if solver_name == "ikgeo.three_parallel":
        return _three_parallel_family(kb, T, policy, native=ext is not None)
    raise NotImplementedError(
        f"charts: no closed-form chart for solver {solver_name!r}. Supported: "
        f"{sorted(SUPPORTED_SOLVERS)}. Polished (approximate-class) solvers have no "
        "closed-form branch; jointlock branches carry no stable label across samples."
    )


# ---------------------------------------------------------------------------
# seven_r.spherical_shoulder: t = q6
# ---------------------------------------------------------------------------

_Q6_DOMAIN_GRID = 180  # per reachable arc, before refinement
_Q6_REFINE_POINTS = 15  # interior probes per bracket per round (16-ary search)
_Q6_REFINE_ROUNDS = 7  # (2*pi/180) / 16**7 ~ 1e-10 rad
_Q6_SEED_EPS = 1e-7  # probe offset past a refined boundary when seeding slivers
_N_SLOTS = 8
_bake_cache: dict[int, NDArray[np.float64]] = {}


def _chain_limits(kb: KinBody) -> Limits:
    out = []
    for j in kb.joints:
        lo_hi = j.limits
        if lo_hi is None or lo_hi[0] is None or lo_hi[1] is None:
            out.append((-np.pi, np.pi))
        else:
            out.append((float(lo_hi[0]), float(lo_hi[1])))
    return tuple(out)


def _jacobian_tangent(
    kb: KinBody,
    eval_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ts: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``dq/dt`` on a q6-chart by implicit differentiation: ``J dq/dt = 0`` with
    ``dq_6/dt = 1``, so ``J[:, :6] dq' = -J[:, 6]``. NaN off the branch and at
    an exact fold (singular 6x6)."""
    from ssik.refinement import kinbody_jacobian

    qs = eval_fn(ts)
    out = np.full_like(qs, np.nan)
    for i, q in enumerate(qs):
        if not np.all(np.isfinite(q)):
            continue
        jac = kinbody_jacobian(kb, q)
        try:
            out[i, :6] = np.linalg.solve(jac[:, :6], -jac[:, 6])
            out[i, 6] = 1.0
        except np.linalg.LinAlgError:
            pass
    return out


def _q6_limit_arcs(
    eval_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    domain: tuple[tuple[float, float], ...],
    limits: Limits,
) -> tuple[tuple[float, float], ...]:
    """In-limits arcs of a q6-chart: per domain interval, the exact feasible
    sub-intervals of joints 0..5 (``feasible_arcs_bounded``) intersected with
    joint 6's own range (``t`` is q6)."""
    from ssik.solvers.seven_r._feasible_param import feasible_arcs_bounded, intersect

    lo6, hi6 = limits[6]
    # t is q6 on [-pi, pi]; joint 6's range may sit on another turn of it.
    own = [(lo6 + k * _TWO_PI, hi6 + k * _TWO_PI) for k in (-1, 0, 1)]
    out: list[tuple[float, float]] = []
    for lo, hi in domain:
        grid = np.linspace(lo, hi, _Q6_DOMAIN_GRID)
        q_grid = eval_fn(grid)
        arcs = feasible_arcs_bounded(
            lambda t: eval_fn(np.array([t]))[0],
            q_grid,
            range(6),
            list(limits),
            grid,
            q_batch=eval_fn,
        )
        out += intersect(arcs, own)
    return tuple(out)


def _swivel_limit_arcs(
    eval_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]], limits: Limits
) -> tuple[tuple[float, float], ...]:
    """In-limits arcs of a swivel chart: the elbow q3 is constant along the
    swivel and is checked once; the other six joints give exact periodic arcs."""
    from ssik.solvers.seven_r._feasible_param import PARAM_GRID, feasible_arcs, to_limits

    q_grid = eval_fn(PARAM_GRID)
    q3 = to_limits(float(q_grid[0, 3]), *limits[3])  # the 2*pi-representative nearest the range
    if not (limits[3][0] <= q3 <= limits[3][1]):
        return ()
    return tuple(
        feasible_arcs(
            lambda t: eval_fn(np.array([t]))[0],
            q_grid,
            (0, 1, 2, 4, 5, 6),
            list(limits),
            PARAM_GRID,
            q_batch=eval_fn,
        )
    )


def _bake_cached(kb: KinBody) -> NDArray[np.float64]:
    from ssik.solvers.seven_r import spherical_shoulder as sh

    coef = _bake_cache.get(id(kb))
    if coef is None:
        coef = _bake_cache[id(kb)] = sh._bake(kb)
    return coef


def _refine_runs(
    runs: list[list[float]],
    pending: list[tuple[int, int, int, float, float]],
    eval_valid: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> None:
    """Refine every pending boundary at once by a batched 16-ary search -- each
    round is ONE slot evaluation over all (valid t_in, invalid t_out) pairs, 15
    interior points each, and shrinks every bracket 16x. Seven rounds take a
    2-degree grid cell to ~1e-10 rad. The batched evaluation has a fixed
    per-call cost far above its per-point cost, so few wide rounds beat many
    narrow bisection steps. Writes the refined value into ``runs[idx][end]``."""
    if not pending:
        return
    t_in = np.array([p[3] for p in pending])
    t_out = np.array([p[4] for p in pending])
    slots_arr = np.array([p[2] for p in pending])
    n_b = len(pending)
    m = _Q6_REFINE_POINTS
    frac = np.arange(1, m + 1) / (m + 1)
    rows = np.arange(n_b)
    for _ in range(_Q6_REFINE_ROUNDS):
        pts = t_in[:, None] + (t_out - t_in)[:, None] * frac[None, :]  # (B, m)
        ok = eval_valid(pts.ravel())[np.repeat(slots_arr, m), np.arange(n_b * m)].reshape(n_b, m)
        # Validity runs from t_in (True) to t_out (False); the boundary sits
        # between the last valid and the first invalid interior point.
        first_bad = np.where(ok.all(axis=1), m, np.argmin(ok, axis=1))
        new_in = np.where(first_bad == 0, t_in, pts[rows, np.maximum(first_bad - 1, 0)])
        new_out = np.where(first_bad == m, t_out, pts[rows, np.minimum(first_bad, m - 1)])
        t_in, t_out = new_in, new_out
    for (idx, end, _slot, _t0, _t1), t_ref in zip(pending, t_in, strict=True):
        runs[idx][end] = float(t_ref)


def _arc_domains(
    a: float,
    b: float,
    eval_valid: Callable[[NDArray[np.float64]], NDArray[np.bool_]],
) -> list[tuple[tuple[float, float], ...]]:
    """Per-slot domains within one reachable arc ``[a, b]`` of q6 (the elbow
    gate holds throughout; the shoulder and wrist gates cut it): grid scan,
    batched refinement of every interior boundary, fold-seeded slivers, merge.
    Returns eight tuples of ``(lo, hi)`` intervals, empty where the slot never
    exists on the arc."""
    grid = np.linspace(a, b, _Q6_DOMAIN_GRID)
    valid = eval_valid(grid)
    n = grid.shape[0]
    # Pass 1: grid-resolved runs per slot. A run touching the arc edge keeps the
    # edge: that is the exact elbow fold.
    runs: list[list[float]] = []
    run_slot: list[int] = []
    pending: list[tuple[int, int, int, float, float]] = []  # (run, end, slot, t_in, t_out)
    for slot in range(_N_SLOTS):
        v = valid[slot]
        i = 0
        while i < n:
            if not v[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and v[j + 1]:
                j += 1
            idx = len(runs)
            runs.append([float(grid[i]), float(grid[j])])
            run_slot.append(slot)
            if i > 0:
                pending.append((idx, 0, slot, float(grid[i]), float(grid[i - 1])))
            if j < n - 1:
                pending.append((idx, 1, slot, float(grid[j]), float(grid[j + 1])))
            i = j + 1
    _refine_runs(runs, pending, eval_valid)
    # Pass 3: slivers thinner than a grid cell. A branch can exist only on a
    # thin arc ending at a fold, where it meets its partner slot; that fold is a
    # refined boundary of the partner. Probe every slot just past every refined
    # boundary and seed a run wherever a slot is valid but uncovered, bracketed
    # by the (invalid) grid points around it, then refine those ends too.
    eps = _Q6_SEED_EPS
    bounds = sorted({x for lo, hi in runs for x in (lo, hi)})
    probes = np.array([x + d for x in bounds for d in (-eps, eps) if a < x + d < b])
    seeded: list[tuple[int, int, int, float, float]] = []
    if probes.shape[0]:
        pv = eval_valid(probes)
        step = (b - a) / (_Q6_DOMAIN_GRID - 1)
        for slot in range(_N_SLOTS):
            covered = [r for s, r in zip(run_slot, runs, strict=True) if s == slot]
            for pi in np.nonzero(pv[slot])[0]:
                t = float(probes[pi])
                if any(lo - 2 * eps <= t <= hi + 2 * eps for lo, hi in covered):
                    continue
                cell = min(int((t - a) / step), _Q6_DOMAIN_GRID - 2)
                if valid[slot, cell] or valid[slot, cell + 1]:
                    continue  # touches a grid run; that run's refinement owns it
                idx = len(runs)
                runs.append([t, t])
                run_slot.append(slot)
                covered.append(runs[idx])
                seeded.append((idx, 0, slot, t, float(grid[cell])))
                seeded.append((idx, 1, slot, t, float(grid[cell + 1])))
        if seeded:
            _refine_runs(runs, seeded, eval_valid)
    out: list[tuple[tuple[float, float], ...]] = []
    for slot in range(_N_SLOTS):
        pieces = sorted((lo, hi) for s, (lo, hi) in zip(run_slot, runs, strict=True) if s == slot)
        merged: list[tuple[float, float]] = []
        for lo, hi in pieces:
            if merged and lo <= merged[-1][1] + 2 * eps:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        out.append(tuple(merged))
    return out


def _spherical_shoulder_family(
    kb: KinBody, T: NDArray[np.float64], policy: TolerancePolicy
) -> SelfMotionManifold:
    from ssik.kinematics._scalar3 import _se3_inv
    from ssik.solvers.seven_r import spherical_shoulder as sh

    coef = _bake_cached(kb)
    t_rev = _se3_inv(T)
    arcs = sh.elbow_arcs(coef, t_rev)  # exact elbow folds, closed form
    limits = _chain_limits(kb)

    def eval_slots(ts: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        return sh._slot_grid(coef, t_rev, ts, policy)

    def eval_valid(ts: NDArray[np.float64]) -> NDArray[np.bool_]:
        return sh._slot_grid(coef, t_rev, ts, policy, valid_only=True)[1]

    domains_cache: dict[int, list[tuple[tuple[float, float], ...]]] = {}

    def arc_domains(k: int) -> list[tuple[tuple[float, float], ...]]:
        d = domains_cache.get(k)
        if d is None:
            d = domains_cache[k] = _arc_domains(arcs[k][0], arcs[k][1], eval_valid)
        return d

    def arc_of(t: float) -> int:
        for k, (lo, hi) in enumerate(arcs):
            if lo - 1e-9 <= t <= hi + 1e-9:
                return k
        return -1

    def make_eval(slot: int, k: int) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        lo, hi = arcs[k]

        def _eval(ts: NDArray[np.float64]) -> NDArray[np.float64]:
            q, valid = eval_slots(ts)
            out: NDArray[np.float64] = q[slot].copy()
            out[~(valid[slot] & (ts >= lo - 1e-9) & (ts <= hi + 1e-9))] = np.nan
            return out

        return _eval

    def make_chart(i: int) -> Chart:
        k, slot = divmod(i, _N_SLOTS)
        e, s, w = sh.slot_label(slot)
        ev = make_eval(slot, k)
        chart = Chart(
            label=(e, s, w, k),
            parameter="q6",
            periodic=False,
            eval_fn=ev,
            domain_fn=lambda: arc_domains(k)[slot],
            deriv_fn=lambda ts: _jacobian_tangent(kb, ev, ts),
            limit_arcs_fn=lambda lims: _q6_limit_arcs(ev, chart.domain, lims),
            default_limits=limits,
        )
        return chart

    def nonempty() -> list[int]:
        return [
            k * _N_SLOTS + s for k in range(len(arcs)) for s in range(_N_SLOTS) if arc_domains(k)[s]
        ]

    def param_of(q: NDArray[np.float64]) -> float:
        return float(_wrap_pi(np.asarray(q[sh._LOCK])))

    def locate(q: NDArray[np.float64], tol: float) -> tuple[int, float]:
        t = param_of(q)
        k = arc_of(t)
        if k < 0:
            return -1, t
        qs, valid = eval_slots(np.array([t]))
        for slot in range(_N_SLOTS):
            if valid[slot, 0] and _wrap_dist(qs[slot, 0], q) <= tol:
                return k * _N_SLOTS + slot, t
        return -1, t

    return SelfMotionManifold(
        parameter="q6",
        periodic=False,
        param_of=param_of,
        make_chart=make_chart,
        nonempty=nonempty,
        locate=locate,
        kb=kb,
        T=T,
        solver_name=_SOLVER_OF["q6"],
    )


# ---------------------------------------------------------------------------
# seven_r.srs: t = elbow swivel psi
# ---------------------------------------------------------------------------

_FULL_CIRCLE: tuple[tuple[float, float], ...] = ((-np.pi, np.pi),)


def _rates_3axis(
    n: list[NDArray[np.float64]], q: NDArray[np.float64], omega: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Joint rates of a 3-axis decomposition ``R = Rot(n0,q0) Rot(n1,q1) Rot(n2,q2)``
    with spatial angular velocity ``omega``: solve ``[n0, R0 n1, R0 R1 n2] rates = omega``.
    Singular (inf) at gimbal lock."""
    from ssik.kinematics._generalized_euler import _axis_angle_matrix as rot

    r0 = rot(n[0], float(q[0]))
    cols = np.stack([n[0], r0 @ n[1], r0 @ rot(n[1], float(q[1])) @ n[2]], axis=1)
    try:
        out: NDArray[np.float64] = np.linalg.solve(cols, omega).astype(np.float64, copy=False)
    except np.linalg.LinAlgError:
        out = np.full(3, np.nan)
    return out


def _srs_tangent(branch: Any, psis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Closed-form ``dq/dpsi`` along one SRS branch (request B2).

    ``R_sh(psi) = Rot(u_sw, psi) R_sh(0)``, so the shoulder's angular velocity
    with respect to psi is ``u_sw`` itself (the rotor derivative); ``q3`` is
    fixed; and with ``A = R_sh Rot(n3, q3)`` the wrist residual
    ``R_res = A^T R_t R_post^T`` has angular velocity ``-A^T u_sw``. Each triple
    of rates is then a 3x3 solve. Derived and checked against finite differences
    in research/self-motion-charts/derivations/srs_tangent.py."""
    from ssik.kinematics._generalized_euler import _axis_angle_matrix as rot

    n = branch.n
    u = branch.u_sw
    qs: NDArray[np.float64] = branch.q_grid(np.asarray(psis, dtype=np.float64))
    out: NDArray[np.float64] = np.zeros_like(qs)
    for i, psi in enumerate(psis):
        q = qs[i]
        out[i, :3] = _rates_3axis(n[:3], q[:3], u)
        a = rot(u, float(psi)) @ branch.R_sh0 @ rot(n[3], float(branch.q3))
        out[i, 4:] = _rates_3axis(n[4:], q[4:], -a.T @ u)
    return out


def _srs_family(kb: KinBody, T: NDArray[np.float64], policy: TolerancePolicy) -> SelfMotionManifold:
    from ssik.kinematics.predicates import _classify_srs_7r_geometric
    from ssik.solvers.seven_r import _swivel_limits as sw
    from ssik.solvers.seven_r.srs import _arm_constants, _frame_at_joint_batch, _swivel_basis

    cls = _classify_srs_7r_geometric(kb, policy)
    if cls is None:
        raise ValueError("charts: chain is not SRS-class (shoulder + wrist axes concurrent)")

    # Swivel-circle frame, the same construction srs.solve and _swivel_limits use:
    # psi = 0 puts the elbow on u_perp1, psi increases towards u_perp2 = u_sw x u_perp1.
    _l_se, _l_ew, ee_offset_local, _origins = _arm_constants(kb, cls)
    S = cls.shoulder_pivot
    W_t = T[:3, 3] - T[:3, :3] @ ee_offset_local
    SW = W_t - S
    d_sw = float(np.linalg.norm(SW))
    branches: list[sw._Branch] = sw._enumerate_branches(kb, cls, T)
    limits = _chain_limits(kb)
    u_sw = np.array([0.0, 0.0, 1.0]) if d_sw < 1e-12 else SW / d_sw
    u_p1, u_p2 = _swivel_basis(u_sw)
    elbow_index = cls.elbow_index

    def param_of(q: NDArray[np.float64]) -> float:
        _r, p = _frame_at_joint_batch(kb, q[None, :], elbow_index)
        e = p[0] - S
        return float(np.arctan2(e @ u_p2, e @ u_p1))

    # _enumerate_branches orders (elbow root) x (s_sgn) x (w_sgn), four per root.
    def make_chart(i: int) -> Chart:
        br = branches[i]

        def _eval(ts: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = br.q_grid(np.asarray(ts, dtype=np.float64))
            return out

        return Chart(
            label=(i // 4, int(br.s_sgn), int(br.w_sgn)),
            parameter="swivel",
            periodic=True,
            eval_fn=_eval,
            domain_fn=lambda: _FULL_CIRCLE,
            deriv_fn=lambda ts: _srs_tangent(br, ts),
            limit_arcs_fn=lambda lims: _swivel_limit_arcs(_eval, lims),
            default_limits=limits,
        )

    def locate(q: NDArray[np.float64], tol: float) -> tuple[int, float]:
        psi = param_of(q)
        for i, br in enumerate(branches):
            if _wrap_dist(br.q(psi), q) <= tol:
                return i, psi
        return -1, psi

    return SelfMotionManifold(
        parameter="swivel",
        periodic=True,
        param_of=param_of,
        make_chart=make_chart,
        nonempty=lambda: list(range(len(branches))),
        locate=locate,
        kb=kb,
        T=T,
        solver_name=_SOLVER_OF["swivel"],
    )


# ---------------------------------------------------------------------------
# ikgeo.three_parallel (UR-class 6R): zero-dimensional charts
# ---------------------------------------------------------------------------


def three_parallel_label(kb: KinBody, q: ArrayLike) -> tuple[int, int, int]:
    """The (shoulder, elbow, wrist) operation mode of a three-parallel 6R
    configuration as three signs, each the sign of one factor of the chain's
    singularity determinant, so a label can only change across a singularity:

    - shoulder: the wrist point (joint 5's origin) on either side of the plane
      spanned by axis 1 and the parallel-trio direction, ``(o4 - o0) . (a0 x h)``;
    - elbow: the trio's origins ``o1, o2, o3`` turning one way or the other,
      ``h . ((o2 - o1) x (o3 - o2))`` (zero with the arm stretched or folded);
    - wrist: axes 4 and 6 on either side of alignment, ``a4 . (a3 x a5)``.

    These are the classic left/right, up/down, flip/no-flip modes computed from
    the geometry, not from the order of the quartic roots the solver produces,
    which carries no identity. Convention-free: no DH offsets involved.
    """
    from ssik.solvers.seven_r.srs import _frame_at_joint_batch

    qa = np.asarray(q, dtype=np.float64)[None, :]
    axes = []
    origins = []
    for i in range(6):
        R, o = _frame_at_joint_batch(kb, qa, i)
        axes.append(R[0] @ np.asarray(kb.joints[i].axis, dtype=np.float64))
        origins.append(o[0])
    h = axes[1]
    shoulder = float((origins[4] - origins[0]) @ np.cross(axes[0], h))
    elbow = float(h @ np.cross(origins[2] - origins[1], origins[3] - origins[2]))
    wrist = float(axes[4] @ np.cross(axes[3], axes[5]))
    return (int(np.sign(shoulder)), int(np.sign(elbow)), int(np.sign(wrist)))


def _three_parallel_family(
    kb: KinBody, T: NDArray[np.float64], policy: TolerancePolicy, native: bool
) -> SelfMotionManifold:
    """A 6R arm's manifold at a regular pose is a finite set of points: every
    chart is zero-dimensional, ``q(t)`` ignores ``t``, and the label is
    :func:`three_parallel_label`. The solutions come from the native solver
    when it is available, else from the Python one; labels are computed here."""
    qs: list[NDArray[np.float64]] = []
    if native:
        from ssik._native import try_native_solve

        sols_n = try_native_solve(
            "ikgeo.three_parallel", kb, T, respect_limits=False, allow_rescue=False
        )
        if sols_n is not None:
            qs = [np.asarray(s.q, dtype=np.float64) for s in sols_n]
            native = True
        else:
            native = False
    if not native:
        from ssik.solvers.ikgeo import three_parallel

        sols_p, _ = three_parallel.solve(kb, T, policy)
        qs = [np.asarray(s.q, dtype=np.float64) for s in sols_p]
    labels = [three_parallel_label(kb, q) for q in qs]
    limits = _chain_limits(kb)

    def make_chart(i: int) -> Chart:
        q = qs[i]
        n_joints = q.shape[0]

        def _eval(ts: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.repeat(q[None, :], np.atleast_1d(ts).shape[0], axis=0)

        def _deriv(ts: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.full((np.atleast_1d(ts).shape[0], n_joints), np.nan)

        def _in_limits(lims: Limits) -> tuple[tuple[float, float], ...]:
            c = np.array([0.5 * (lo + hi) for lo, hi in lims])
            rep = q + _TWO_PI * np.round((c - q) / _TWO_PI)
            ok = all(lo - 1e-9 <= r <= hi + 1e-9 for r, (lo, hi) in zip(rep, lims, strict=True))
            return ((0.0, 0.0),) if ok else ()

        return Chart(
            label=labels[i],
            parameter="point",
            periodic=False,
            eval_fn=_eval,
            domain_fn=lambda: ((0.0, 0.0),),
            deriv_fn=_deriv,
            limit_arcs_fn=_in_limits,
            default_limits=limits,
        )

    def locate(q: NDArray[np.float64], tol: float) -> tuple[int, float]:
        for i, qi in enumerate(qs):
            if _wrap_dist(qi, q) <= tol:
                return i, 0.0
        return -1, 0.0

    return SelfMotionManifold(
        parameter="point",
        periodic=False,
        param_of=lambda q: 0.0,
        make_chart=make_chart,
        nonempty=lambda: list(range(len(qs))),
        locate=locate,
        native=native,
        kb=kb,
        T=T,
        solver_name=_SOLVER_OF["point"],
    )


# ---------------------------------------------------------------------------
# Tracking along a pose path, and the cuspidality report behind it
# ---------------------------------------------------------------------------


def _sample_points(
    chart: Chart, n: int, with_t: bool = False, limits: ArrayLike | bool | None = None
) -> Any:
    """Finite points of a chart: its solution (0-D) or ``n`` per piece (the
    domain, or its in-limits part), with their coordinates when ``with_t``."""
    if chart.dimension == 0:
        # Out-of-box is decided by the in-box distance itself (``inf``).
        pts, ts_all = chart.q(0.0)[None, :], np.zeros(1)
    else:
        rows, tss = [], []
        for lo, hi in chart._pieces(limits):
            ts = np.linspace(lo, hi, n)
            qs = chart._eval_t(ts)
            ok = np.all(np.isfinite(qs), axis=1)
            rows.append(qs[ok])
            tss.append(ts[ok])
        pts = np.concatenate(rows) if rows else np.zeros((0, 7))
        ts_all = np.concatenate(tss) if tss else np.zeros(0)
    return (pts, ts_all) if with_t else pts


def _box(chart: Chart, limits: ArrayLike | bool) -> NDArray[np.float64]:
    """``(dof, 2)`` joint box: the chain's own (``True``; what
    :meth:`Chart.in_limits` uses by default) or the one given."""
    if limits is True:
        if not chart._default_limits:
            raise ValueError("limits=True: this chart carries no joint limits; pass them")
        return np.asarray(chart._default_limits, dtype=np.float64)
    return np.asarray(limits, dtype=np.float64)


def _inbox_dist2(
    qa: NDArray[np.float64], qb: NDArray[np.float64], box: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Squared distance between in-box representatives, broadcast over leading
    axes: per joint, the nearest pair among the windings ``q + 2*pi*k`` of each
    side that lie in ``box``. The box is a product of intervals, so the
    minimum separates by joint. ``inf`` where a side has no in-box winding.
    Returns the distance and the chosen representatives."""
    ks = 2.0 * np.pi * np.arange(-2, 3, dtype=np.float64)
    lo, hi = box[:, 0], box[:, 1]
    tol = 1e-9
    wa = qa[..., None] + ks  # (..., dof, 5)
    wb = qb[..., None] + ks
    wa = np.where((wa >= lo[:, None] - tol) & (wa <= hi[:, None] + tol), wa, np.nan)
    wb = np.where((wb >= lo[:, None] - tol) & (wb <= hi[:, None] + tol), wb, np.nan)
    diff = np.abs(wa[..., :, None] - wb[..., None, :])  # (..., dof, 5, 5)
    flat = np.where(np.isnan(diff), np.inf, diff).reshape(*diff.shape[:-2], 25)
    best = np.argmin(flat, axis=-1)
    per_joint = np.take_along_axis(flat, best[..., None], axis=-1)[..., 0]
    d2 = np.sum(per_joint**2, axis=-1)
    ia, ib = np.divmod(best, 5)
    ra = np.take_along_axis(wa, ia[..., None], axis=-1)[..., 0]
    rb = np.take_along_axis(wb, ib[..., None], axis=-1)[..., 0]
    return d2, ra, rb


def se3_exp(xi: ArrayLike) -> NDArray[np.float64]:
    """4x4 exponential of a spatial twist ``xi = (v, w)`` (linear, angular), so
    that ``se3_exp(dx) @ T`` is the pose ``T`` moved by ``dx``, matching the
    spatial Jacobian convention ``T(q + dq) ~ exp([J dq]) T(q)``."""
    x = np.asarray(xi, dtype=np.float64)
    v, w = x[:3], x[3:]
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-12:
        R, V = np.eye(3), np.eye(3)
    else:
        R = np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K
        V = np.eye(3) + (1 - np.cos(th)) / th**2 * K + (th - np.sin(th)) / th**3 * K @ K
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = V @ v
    return out


#: :func:`drift_to_merge`'s answer, and the same paired with its reason.
_MergeFound = tuple[float, NDArray[np.float64]] | None
_MergeExplained = tuple[_MergeFound, str]


def _merge_outcome(found: _MergeFound, reason: str, explain: bool) -> _MergeFound | _MergeExplained:
    """:func:`drift_to_merge`'s return, with or without the reason."""
    return (found, reason) if explain else found


def _glued_sheet(
    family: SelfMotionManifold, chart: Chart, tol: float = 1e-6
) -> frozenset[tuple[int, ...]]:
    """Labels of the sheet ``chart`` lies on: charts glued where one's domain
    end coincides with another's (partner slots meet at a fold point). The
    endpoint test is what :meth:`SelfMotionManifold.sheets` decides by ``gap``,
    at a fraction of the cost; charts on a whole circle have no end and are a
    sheet of their own."""

    def ends(c: Chart) -> NDArray[np.float64]:
        if c.dimension == 0:
            return np.zeros((0, 7))
        ts = [e for lo, hi in c.domain if hi - lo < _TWO_PI - 1e-9 for e in (lo, hi)]
        if not ts:
            return np.zeros((0, 7))
        qs = c.q(np.asarray(ts))
        return np.asarray(qs[np.all(np.isfinite(qs), axis=1)])

    cs = list(family.charts)
    tips = {c.label: ends(c) for c in cs}
    sheet, frontier = {chart.label}, [chart.label]
    while frontier:
        mine = tips[frontier.pop()]
        for c in cs:
            if c.label in sheet or mine.shape[0] == 0 or tips[c.label].shape[0] == 0:
                continue
            d = np.abs(_wrap_pi(mine[:, None, :] - tips[c.label][None, :, :])).max(axis=2)
            if float(d.min()) <= tol:
                sheet.add(c.label)
                frontier.append(c.label)
    return frozenset(sheet)


@overload
def drift_to_merge(
    kb: KinBody,
    T: ArrayLike,
    a: Chart,
    b: Chart,
    direction: ArrayLike,
    *,
    max_drift: float = ...,
    n_steps: int = ...,
    merge_tol: float = ...,
    limits: ArrayLike | bool | None = ...,
    solver_name: str | None = ...,
    policy: TolerancePolicy = ...,
    native: bool = ...,
    explain: Literal[False] = False,
) -> _MergeFound: ...


@overload
def drift_to_merge(
    kb: KinBody,
    T: ArrayLike,
    a: Chart,
    b: Chart,
    direction: ArrayLike,
    *,
    max_drift: float = ...,
    n_steps: int = ...,
    merge_tol: float = ...,
    limits: ArrayLike | bool | None = ...,
    solver_name: str | None = ...,
    policy: TolerancePolicy = ...,
    native: bool = ...,
    explain: Literal[True],
) -> _MergeExplained: ...


def drift_to_merge(
    kb: KinBody,
    T: ArrayLike,
    a: Chart,
    b: Chart,
    direction: ArrayLike,
    *,
    max_drift: float = 1.0,
    n_steps: int = 200,
    merge_tol: float = 1e-3,
    limits: ArrayLike | bool | None = None,
    solver_name: str | None = None,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    native: bool = True,
    explain: bool = False,
) -> _MergeFound | _MergeExplained:
    """March the pose along a unit twist and report the drift at which the
    *sheets* of charts ``a`` and ``b`` merge (request D2, the verification of
    :meth:`SelfMotionManifold.escape`).

    A sheet is several charts glued at folds (the Panda) or one closed chart
    (an SRS arm), so what is followed is the sheet, not a chart: one point of
    each sheet is carried along by label continuation -- the same point at
    every step, so a nearest point sitting on a fold, where either of two
    charts can claim it, never changes which sheet is meant -- and the quantity
    watched is the gap between the two sheets, the least :meth:`~SelfMotionManifold.gap`
    over their chart pairs.

    Two ways to merge. The sheets can join, becoming one sheet (the Panda):
    the drift where that happens is bisected on "same sheet" to ~1e-12. Or
    they can touch without joining (the iiwa14, whose sheets are closed
    charts): the gap has a square-root cusp there, and each local minimum
    the grid brackets is refined by golden section. Either is a merge when
    the gap just before it is at most ``merge_tol``. Returns ``(drift, pose)``
    at the first merge, or ``None``.

    Resolution: the drift of a touch is found to ~1e-8, the gap at it to
    ~1e-4 rad (the nearest pair is itself cusp-shaped in ``t`` there), hence
    the default ``merge_tol``. The grid must see a touch before it can be
    refined; the gap grows like ``c*sqrt(|s - s*|)`` with ``c`` ~ 8 rad, so
    it is near 1 rad 0.015 away, and ``n_steps=200`` over unit drift resolves
    that. ``limits`` is passed to :meth:`~SelfMotionManifold.gap`: only
    postures inside the joint box count.

    ``explain=True`` returns ``(result, reason)``, the reason naming the
    outcome -- ``merged``, ``branch lost`` or ``no merge`` -- with the closest
    approach found; a bare ``None`` cannot tell "never met" from "the pose left
    the workspace".
    """
    T0 = np.asarray(T, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)

    def fam_at(s: float) -> SelfMotionManifold:
        pose = se3_exp(s * d) @ T0
        return charts(kb, pose, solver_name=solver_name, policy=policy, native=native)

    Head = tuple[Chart, float, NDArray[np.float64]]

    def advance(fam: SelfMotionManifold, h: Head) -> Head:
        c, t, _ = fam.continue_from(h[0], h[1], h[2], max_step=0.7)
        return c, t, c.q(t)

    def sheet_gap(fam: SelfMotionManifold, ha: Head, hb: Head) -> tuple[float, bool]:
        """(gap between the heads' sheets, whether they are one sheet)."""
        sa, sb = _glued_sheet(fam, ha[0]), _glued_sheet(fam, hb[0])
        if sa & sb:
            return 0.0, True
        best = float("inf")
        for x in fam.charts:
            if x.label not in sa:
                continue
            for y in fam.charts:
                if y.label in sb:
                    best = min(best, fam.gap(x, y, limits=limits)[0])
        return best, False

    fam0 = fam_at(0.0)
    _, qa, qb = fam0.gap(a, b, limits=limits)
    la, lb = fam0.locate(_wrap_pi(qa)), fam0.locate(_wrap_pi(qb))
    if la is None or lb is None:
        raise ValueError("drift_to_merge: the charts' nearest points do not locate")
    ha: Head = (la[0], la[1], _wrap_pi(qa))
    hb: Head = (lb[0], lb[1], _wrap_pi(qb))
    g_start, joined = sheet_gap(fam0, ha, hb)
    if joined:
        raise ValueError("drift_to_merge: a and b are on one sheet already")
    grid: list[tuple[float, Head, Head, float]] = [(0.0, ha, hb, g_start)]

    def state_at(k: int, s: float) -> tuple[float, bool] | None:
        """Sheet gap at drift ``s``, the heads continued from grid step ``k``."""
        fam = fam_at(s)
        if not fam.charts:
            return None
        _, ha_k, hb_k, _ = grid[k]
        return sheet_gap(fam, advance(fam, ha_k), advance(fam, hb_k))

    def gap_at(k: int, s: float) -> float:
        st = state_at(k, s)
        return float("inf") if st is None else st[0]

    def golden(k: int, lo: float, hi: float) -> tuple[float, float]:
        r = 0.5 * (np.sqrt(5.0) - 1.0)
        x1, x2 = hi - r * (hi - lo), lo + r * (hi - lo)
        f1, f2 = gap_at(k, x1), gap_at(k, x2)
        for _ in range(80):
            if hi - lo < 1e-14:
                break
            if f1 <= f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - r * (hi - lo)
                f1 = gap_at(k, x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + r * (hi - lo)
                f2 = gap_at(k, x2)
        return (x1, f1) if f1 <= f2 else (x2, f2)

    def joined_between(k: int, lo: float, hi: float) -> tuple[float, float]:
        """Bisect the drift where the sheets become one, from grid step ``k``;
        the gap just before it."""
        for _ in range(60):
            if hi - lo <= 1e-12:
                break
            mid = 0.5 * (lo + hi)
            st = state_at(k, mid)
            if st is not None and st[1]:
                hi = mid
            else:
                lo = mid
        return hi, gap_at(k, lo)

    def merged(s_min: float, g_min: float, how: str) -> _MergeFound | _MergeExplained:
        return _merge_outcome(
            (s_min, se3_exp(s_min * d) @ T0),
            f"merged at drift {s_min:.8f} ({how}), gap {g_min:.3e} rad",
            explain,
        )

    closest, closest_s = g_start, 0.0
    for k in range(1, n_steps + 1):
        s = max_drift * k / n_steps
        fam = fam_at(s)
        if not fam.charts:
            s_min, g_min = golden(k - 1, grid[max(k - 2, 0)][0], s)
            if g_min <= merge_tol:
                return merged(s_min, g_min, "touched before leaving the workspace")
            return _merge_outcome(
                None,
                f"branch lost at drift {s:.4f} (the pose left the workspace); closest "
                f"approach {min(closest, g_min):.4g} rad",
                explain,
            )
        ha, hb = advance(fam, grid[-1][1]), advance(fam, grid[-1][2])
        g, joined = sheet_gap(fam, ha, hb)
        if joined:
            s_star, g_before = joined_between(k - 1, grid[-1][0], s)
            if g_before <= merge_tol:
                return merged(s_star, g_before, "the sheets joined")
            return _merge_outcome(
                None,
                f"branch lost at drift {s_star:.6f}: the sheets became one while "
                f"{g_before:.4g} rad apart -- a continuation jumped, not a merge",
                explain,
            )
        grid.append((s, ha, hb, g))
        if k >= 2 and grid[k - 1][3] <= grid[k - 2][3] and grid[k - 1][3] < g:
            s_min, g_min = golden(k - 1, grid[k - 2][0], s)
            if g_min <= merge_tol:
                return merged(s_min, g_min, "the sheets touched")
            if g_min < closest:
                closest, closest_s = g_min, s_min
        if g < closest:
            closest, closest_s = g, s
    return _merge_outcome(
        None,
        f"no merge within max_drift={max_drift:g}; closest approach {closest:.4g} rad "
        f"at drift {closest_s:.6f}",
        explain,
    )


class TrackStep:
    """One step of :func:`track`: the pose index, the chart label, the
    coordinate, the configuration, and the continuation event."""

    __slots__ = ("event", "index", "label", "q", "t")

    def __init__(
        self, index: int, label: tuple[int, ...], t: float, q: NDArray[np.float64], event: str
    ) -> None:
        self.index = index
        self.label = label
        self.t = t
        self.q = q
        self.event = event

    def __repr__(self) -> str:
        return (
            f"TrackStep(index={self.index}, label={self.label}, "
            f"t={self.t:.4f}, event={self.event!r})"
        )


def track(
    kb: KinBody,
    poses: ArrayLike,
    q0: ArrayLike,
    *,
    solver_name: str | None = None,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    native: bool = True,
    max_step: float = 0.5,
) -> list[TrackStep]:
    """Follow one branch of the self-motion manifold along a pose path.

    ``poses[0]`` must be the pose of ``q0``. At each subsequent pose the branch is
    continued by :meth:`SelfMotionManifold.continue_from`: a label lookup while the
    label persists, a relabel at a fold, and a ``"collision"`` event where the
    branch vanished (the step then holds the nearest chart's point). At a pose
    with no manifold at all (outside the workspace) tracking stops: the list
    ends with one ``"unreachable"`` step (``q`` all ``NaN``) and is shorter than
    ``poses``. The
    coordinate ``t`` is held fixed step to step, so this is *pure* continuation
    of the redundancy: a controller would add its own motion along the chart.
    Closing a loop and comparing the first and last labels reads off the
    monodromy of the loop.
    """
    path = track_all(
        kb, poses, q0=q0, solver_name=solver_name, policy=policy, native=native, max_step=max_step
    )
    return next(iter(path.tracks.values()))


class PathTrack:
    """Result of :func:`track_all`: every tracked branch along a pose path.

    ``tracks`` maps each starting label (at ``poses[0]``) to its list of
    :class:`TrackStep`, one per pose up to where tracking stopped.
    ``stopped_at`` is the index of the first pose outside the workspace, where
    every track ends with an ``"unreachable"`` step, or ``None`` when the whole
    path was tracked. ``closed`` is ``True`` when the path returns to its
    starting pose, and then :attr:`permutation` reads off the monodromy.
    """

    __slots__ = ("closed", "stopped_at", "tracks")

    def __init__(
        self,
        tracks: dict[tuple[int, ...], list[TrackStep]],
        closed: bool,
        stopped_at: int | None = None,
    ) -> None:
        self.tracks = tracks
        self.closed = closed
        self.stopped_at = stopped_at

    @property
    def permutation(self) -> dict[tuple[int, ...], tuple[int, ...]] | None:
        """Start label -> end label, on a closed path tracked to its end
        (``None`` otherwise, including a path that left the workspace). The
        identity when every branch came back to itself. Not necessarily a
        bijection: after a ``"collision"`` two tracks can end on one chart --
        :meth:`is_bijection` says whether it is one."""
        if not self.closed or self.stopped_at is not None:
            return None
        return {label: steps[-1].label for label, steps in self.tracks.items()}

    def is_bijection(self) -> bool:
        perm = self.permutation
        return perm is not None and sorted(perm.values()) == sorted(perm)

    def events(self) -> list[TrackStep]:
        """Every step whose event is not a plain ``"label"`` lookup or the
        start: the folds, collisions and the stop outside the workspace, in
        pose order."""
        steps = [
            s for track in self.tracks.values() for s in track if s.event not in ("label", "start")
        ]
        return sorted(steps, key=lambda s: s.index)

    def q(self, label: tuple[int, ...]) -> NDArray[np.float64]:
        """The configurations of one track, ``(n_steps, dof)``; the last row is
        ``NaN`` when tracking stopped outside the workspace."""
        return np.stack([s.q for s in self.tracks[label]])

    def __repr__(self) -> str:
        return (
            f"PathTrack({len(self.tracks)} branches, "
            f"{len(next(iter(self.tracks.values()), []))} poses, closed={self.closed}, "
            f"{len(self.events())} events)"
        )


def _start_t(chart: Chart) -> float:
    """A representative coordinate on a chart: the middle of its longest
    domain interval (``0`` on a zero-dimensional chart)."""
    if chart.dimension == 0:
        return 0.0
    lo, hi = max(chart.domain, key=lambda d: d[1] - d[0])
    return 0.5 * (lo + hi)


def track_all(
    kb: KinBody,
    poses: ArrayLike,
    *,
    q0: ArrayLike | None = None,
    solver_name: str | None = None,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    native: bool = True,
    max_step: float = 0.5,
    close_tol: float = 1e-9,
) -> PathTrack:
    """Solve a pose path in one call, carrying chart labels along it (request C2).

    With ``q0=None`` every chart of ``poses[0]`` is tracked, each from the
    middle of its longest domain interval; with ``q0`` only the branch through
    ``q0``, from where ``q0`` sits (``poses[0]`` must be its pose), and with a
    ``(k, dof)`` ``q0`` the ``k`` branches through those starts, one per chart. The manifold
    is built once per pose and every branch is continued on it by
    :meth:`SelfMotionManifold.continue_from` -- a label lookup, no
    nearest-neighbour search, with the fold and collision events of
    :func:`track`. ``t`` is held fixed per branch, as in :func:`track`.
    Tracking stops at the first pose with no manifold at all (outside the
    workspace): nothing connects a branch across that gap, so picking a label
    up again on the far side would be a fresh lookup posing as continuation.
    Every track then ends with one ``"unreachable"`` step (``q`` all ``NaN``)
    and :attr:`PathTrack.stopped_at` names the pose.

    When ``poses[-1]`` equals ``poses[0]`` (to ``close_tol``, max abs entry)
    the path is a loop and :attr:`PathTrack.permutation` is its monodromy.
    """
    P = np.asarray(poses, dtype=np.float64)
    if P.ndim != 3 or P.shape[1:] != (4, 4) or P.shape[0] < 1:
        raise ValueError(f"track_all: poses must be (N, 4, 4) with N >= 1, got {P.shape}")
    fam = charts(kb, P[0], solver_name=solver_name, policy=policy, native=native)
    heads: list[tuple[Chart, float, NDArray[np.float64]]] = []
    if q0 is None:
        for c in fam.charts:
            t = _start_t(c)
            heads.append((c, t, c.q(t)))
    else:
        for q_start in np.atleast_2d(np.asarray(q0, dtype=np.float64)):
            located = fam.locate(q_start)
            if located is None:
                raise ValueError("track: q0 is not on the manifold of poses[0]")
            heads.append((located[0], located[1], q_start))
    starts = [c.label for c, _, _ in heads]
    if len(set(starts)) != len(starts):
        raise ValueError(
            f"track_all: two starts lie on one chart ({starts}); tracks are keyed by it"
        )
    tracks = {c.label: [TrackStep(0, c.label, t, q, "start")] for c, t, q in heads}
    stopped_at: int | None = None
    for i in range(1, P.shape[0]):
        fam = charts(kb, P[i], solver_name=solver_name, policy=policy, native=native)
        if not fam.charts:
            gone = np.full(len(kb.joints), np.nan)
            for k, (chart, t, _) in enumerate(heads):
                tracks[starts[k]].append(TrackStep(i, chart.label, t, gone, "unreachable"))
            stopped_at = i
            break
        for k, (chart, t, q) in enumerate(heads):
            chart, t, event = fam.continue_from(chart, t, q, max_step=max_step)
            q = chart.q(t)
            heads[k] = (chart, t, q)
            tracks[starts[k]].append(TrackStep(i, chart.label, t, q, event))
    closed = P.shape[0] > 1 and float(np.max(np.abs(P[-1] - P[0]))) <= close_tol
    return PathTrack(tracks, closed, stopped_at)


class CuspidalityReport:
    """Result of :func:`cuspidality_report`. ``cuspidal`` is ``True`` when, for
    some locked angle, two inverse-kinematic solutions of one pose fell into
    the same aspect of the positioning chain, so a task-space loop can return
    on another branch without crossing a singularity and label lookup is not a
    valid continuation there. ``aspects`` maps each locked angle to the number
    of aspects found; ``shared_pairs`` to the number of solution pairs sharing
    one."""

    __slots__ = ("aspects", "cuspidal", "shared_pairs", "solver_name")

    def __init__(
        self, solver_name: str, aspects: dict[float, int], shared_pairs: dict[float, int]
    ) -> None:
        self.solver_name = solver_name
        self.aspects = aspects
        self.shared_pairs = shared_pairs
        self.cuspidal = any(v > 0 for v in shared_pairs.values())

    def __repr__(self) -> str:
        verdict = "cuspidal" if self.cuspidal else "noncuspidal"
        return f"CuspidalityReport({self.solver_name}: {verdict}, aspects={self.aspects})"


def cuspidality_report(
    kb: KinBody,
    *,
    locked_angles: ArrayLike = (-2.0, -0.7, 0.5, 1.5, 2.5),
    n_poses: int = 100,
    grid: int = 480,
    seed: int = 0,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> CuspidalityReport:
    """Classify the chart slicing of a spherical-shoulder arm as cuspidal or
    not, by the criterion of Salunkhe, Gupta & Billard (RA-L 2025): lock the
    redundancy at each of ``locked_angles``, partition the positioning chain's
    ``(q1, q2)`` torus into aspects by the singular set of its positional
    Jacobian (signed components outside a dead band -- the determinant can touch
    zero without changing sign), and count solution pairs of one pose that share
    an aspect. A build-time diagnostic (seconds), not a runtime path.

    SRS-family charts are noncuspidal by construction (an exact spherical
    shoulder and wrist decouple, and the closed form relabels only at a critical
    point); this report covers the ``seven_r.spherical_shoulder`` family, where
    the answer depends on which joint is locked -- ssik locks the last one.
    """
    from scipy import ndimage  # type: ignore[import-untyped]

    from ssik.core.dispatcher import dispatch
    from ssik.kinematics._scalar3 import _se3_inv
    from ssik.kinematics.reverse import reverse_kinematic_chain
    from ssik.solvers.ikgeo import spherical_two_intersecting
    from ssik.solvers.jointlock.seven_r import _lock_joint
    from ssik.solvers.seven_r.srs import _frame_at_joint_batch

    name = dispatch(kb, policy).solver_name
    if name != "seven_r.spherical_shoulder":
        raise NotImplementedError(
            f"cuspidality_report: covers seven_r.spherical_shoulder, not {name!r}"
        )
    rng = np.random.default_rng(seed)
    limits = _chain_limits(kb)
    g = np.linspace(-np.pi, np.pi, grid, endpoint=False)
    step = g[1] - g[0]
    q1g, q2g = np.meshgrid(g, g, indexing="ij")
    aspects: dict[float, int] = {}
    shared: dict[float, int] = {}
    for t in np.asarray(locked_angles, dtype=np.float64):
        rev = reverse_kinematic_chain(_lock_joint(kb, 6, float(t)))
        det = _positioning_det(rev, q1g, q2g, _frame_at_joint_batch)
        band = 1e-3 * float(np.abs(det).max())
        pos, n_pos = _torus_components(det > band, ndimage)
        neg, n_neg = _torus_components(det < -band, ndimage)
        aspect = np.where(det > band, pos, np.where(det < -band, -neg, 0))
        aspects[float(t)] = n_pos + n_neg
        n_shared = 0
        for _ in range(n_poses):
            q = np.array([rng.uniform(lo, hi) for lo, hi in limits])
            from ssik.kinematics.poe_fk import poe_forward_kinematics

            sols, _ = spherical_two_intersecting.solve(
                rev, _se3_inv(poe_forward_kinematics(kb, q)), policy
            )
            pts = {(round(float(s.q[1]), 6), round(float(s.q[2]), 6)) for s in sols}
            ids: list[int] = []
            for a1, a2 in pts:
                i = int(np.round((a1 + np.pi) / step)) % grid
                j = int(np.round((a2 + np.pi) / step)) % grid
                a = int(aspect[i, j])
                if a == 0:
                    ids = []
                    break
                ids.append(a)
            n_shared += len(ids) - len(set(ids))
        shared[float(t)] = n_shared
    return CuspidalityReport(name, aspects, shared)


def _positioning_det(
    rev: Any, q1: NDArray[np.float64], q2: NDArray[np.float64], frame_at: Any
) -> NDArray[np.float64]:
    """det of the 3x3 positional Jacobian of the reversed chain's positioning
    triple (joints 0, 1, 2) moving the wrist centre -- the common point of the
    spherical triple (3, 4, 5), the point on axis 3 nearest axis 4 -- on a
    (q1, q2) grid with q0 = 0 (the determinant is q0-invariant)."""
    n = q1.size
    Q = np.zeros((n, 6))
    Q[:, 1], Q[:, 2] = q1.ravel(), q2.ravel()
    R3, o3 = frame_at(rev, Q, 3)
    R4, o4 = frame_at(rev, Q, 4)
    a3 = R3 @ np.asarray(rev.joints[3].axis, float)
    a4 = R4 @ np.asarray(rev.joints[4].axis, float)
    d = o4 - o3
    c = np.einsum("ni,ni->n", a3, a4)
    s3 = (np.einsum("ni,ni->n", d, a3) - c * np.einsum("ni,ni->n", d, a4)) / np.maximum(
        1.0 - c * c, 1e-12
    )
    p_w = o3 + s3[:, None] * a3
    cols = []
    for k in range(3):
        R, o = frame_at(rev, Q, k)
        ax = R @ np.asarray(rev.joints[k].axis, float)
        cols.append(np.cross(ax, p_w - o))
    out: NDArray[np.float64] = np.linalg.det(np.stack(cols, axis=2)).reshape(q1.shape)
    return out


def _torus_components(mask: NDArray[np.bool_], ndimage: Any) -> tuple[NDArray[np.int64], int]:
    """Connected components of a boolean (q1, q2) grid on the torus."""
    lab, n = ndimage.label(mask)
    parent = list(range(n + 1))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        if a and b:
            parent[find(a)] = find(b)

    for i in range(mask.shape[0]):
        union(int(lab[i, 0]), int(lab[i, -1]))
    for j in range(mask.shape[1]):
        union(int(lab[0, j]), int(lab[-1, j]))
    roots = {find(a) for a in range(1, n + 1)}
    remap = {r: k + 1 for k, r in enumerate(sorted(roots))}
    out = np.zeros_like(lab)
    nz = lab > 0
    out[nz] = [remap[find(int(a))] for a in lab[nz]]
    return out, len(roots)


# ---------------------------------------------------------------------------
# Native (C++) backend: cpp/include/ssik_cpp/chart.hpp via ssik._ssik_native
# ---------------------------------------------------------------------------


def _native_ext() -> Any:
    from ssik._native import _load_ext

    ext = _load_ext()
    return ext if ext is not None and hasattr(ext, "SphericalShoulderCharts") else None


def _spherical_shoulder_family_native(
    ext: Any, kb: KinBody, T: NDArray[np.float64], policy: TolerancePolicy
) -> SelfMotionManifold:
    from ssik.solvers.seven_r import spherical_shoulder as sh

    j = kb.joints
    nat = ext.SphericalShoulderCharts(
        np.array([jt.axis for jt in j], dtype=np.float64),
        np.array([jt.T_left for jt in j], dtype=np.float64),
        np.array([jt.T_right for jt in j], dtype=np.float64),
        np.array([0 if jt.joint_type == "revolute" else 1 for jt in j], dtype=np.int32),
        np.ascontiguousarray(_bake_cached(kb)),
        np.ascontiguousarray(T),
        policy.subproblem_feasibility,
        policy.subproblem_degeneracy,
    )
    limits = _chain_limits(kb)

    def limit_arcs(i: int, lims: Limits) -> tuple[tuple[float, float], ...]:
        lo = np.array([a for a, _b in lims], dtype=np.float64)
        hi = np.array([b for _a, b in lims], dtype=np.float64)
        return tuple((float(a), float(b)) for a, b in nat.in_limits(i, lo, hi))

    def make_chart(i: int) -> Chart:
        k, slot = divmod(i, _N_SLOTS)
        e, s, w = sh.slot_label(slot)

        def _eval(ts: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = nat.q(i, np.ascontiguousarray(ts, dtype=np.float64))
            return out

        def _deriv(ts: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = nat.tangent(i, np.ascontiguousarray(ts, dtype=np.float64))
            return out

        return Chart(
            label=(e, s, w, k),
            parameter="q6",
            periodic=False,
            eval_fn=_eval,
            domain_fn=lambda: tuple((float(lo), float(hi)) for lo, hi in nat.domain(i)),
            deriv_fn=_deriv,
            limit_arcs_fn=lambda lims: limit_arcs(i, lims),
            default_limits=limits,
        )

    def param_of(q: NDArray[np.float64]) -> float:
        return float(nat.param_of(np.ascontiguousarray(q, dtype=np.float64)))

    def locate(q: NDArray[np.float64], tol: float) -> tuple[int, float]:
        idx, t, _dist = nat.locate(np.ascontiguousarray(q, dtype=np.float64), tol)
        return int(idx), float(t)

    return SelfMotionManifold(
        parameter="q6",
        periodic=False,
        param_of=param_of,
        make_chart=make_chart,
        nonempty=lambda: [i for i in range(len(nat)) if nat.nonempty(i)],
        locate=locate,
        native=True,
        kb=kb,
        T=T,
        solver_name=_SOLVER_OF["q6"],
    )


_SRS_ARG_KEYS = (
    "axes",
    "t_left",
    "t_right",
    "types",
    "l_se",
    "l_ew",
    "ee_offset_local",
    "shoulder_pivot",
    "r_post_wrist",
    "elbow_index",
    "upper_home",
    "forearm_home",
    "general_path",
)


def _srs_family_native(ext: Any, kb: KinBody, T: NDArray[np.float64]) -> SelfMotionManifold | None:
    from ssik._native import _srs_native_args

    args = _srs_native_args(kb)
    if args is None:
        return None
    nat = ext.SrsCharts(*(args[k] for k in _SRS_ARG_KEYS), np.ascontiguousarray(T))
    labels = np.asarray(nat.labels)
    limits = _chain_limits(kb)

    def limit_arcs(i: int, lims: Limits) -> tuple[tuple[float, float], ...]:
        lo = np.array([a for a, _b in lims], dtype=np.float64)
        hi = np.array([b for _a, b in lims], dtype=np.float64)
        return tuple((float(a), float(b)) for a, b in nat.in_limits(i, lo, hi))

    def make_chart(i: int) -> Chart:
        def _eval(ts: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = nat.q(i, np.ascontiguousarray(ts, dtype=np.float64))
            return out

        def _deriv(ts: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = nat.tangent(i, np.ascontiguousarray(ts, dtype=np.float64))
            return out

        return Chart(
            label=tuple(int(x) for x in labels[i]),
            parameter="swivel",
            periodic=True,
            eval_fn=_eval,
            domain_fn=lambda: _FULL_CIRCLE,
            deriv_fn=_deriv,
            limit_arcs_fn=lambda lims: limit_arcs(i, lims),
            default_limits=limits,
        )

    def param_of(q: NDArray[np.float64]) -> float:
        return float(nat.param_of(np.ascontiguousarray(q, dtype=np.float64)))

    def locate(q: NDArray[np.float64], tol: float) -> tuple[int, float]:
        idx, psi, _dist = nat.locate(np.ascontiguousarray(q, dtype=np.float64), tol)
        return int(idx), float(psi)

    return SelfMotionManifold(
        parameter="swivel",
        periodic=True,
        param_of=param_of,
        make_chart=make_chart,
        nonempty=lambda: list(range(len(nat))),
        locate=locate,
        native=True,
        kb=kb,
        T=T,
        solver_name=_SOLVER_OF["swivel"],
    )
