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
- ``domain`` is the set of ``t`` where the branch exists (reachable), exact to
  bisection tolerance. Joint limits are *not* applied here: the chart is the
  geometric object, limits are a filter the caller composes on top.

A :class:`ChartFamily` is every chart at one pose plus the inverse map
:meth:`ChartFamily.locate`: given a configuration on the manifold, which chart
is it on and at what ``t``.

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
:attr:`ChartFamily.native` says which one you got.

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
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy

if TYPE_CHECKING:  # pragma: no cover
    from ssik._kinbody import KinBody

__all__ = ["Chart", "ChartFamily", "charts"]

SUPPORTED_SOLVERS: frozenset[str] = frozenset({"seven_r.spherical_shoulder", "seven_r.srs"})

_TWO_PI = 2.0 * np.pi


def _wrap_pi(a: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = (a + np.pi) % _TWO_PI - np.pi
    return out


def _wrap_dist(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Wrap-to-pi L-infinity distance between two joint vectors."""
    return float(np.max(np.abs(_wrap_pi(a - b))))


_STENCIL_H = 1e-4  # fourth-order central difference step for the numerical tangent


def _stencil(
    eval_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]], ts: NDArray[np.float64]
) -> NDArray[np.float64]:
    """dq/dt by the 5-point central difference on the closed form; representative
    jumps (2*pi) are removed before differencing. NaN where any stencil point is
    off the branch (within ~2h of a fold)."""
    h = _STENCIL_H
    q0 = eval_fn(ts)
    parts = [eval_fn(ts + k * h) for k in (-2, -1, 1, 2)]
    d = [(_wrap_pi(qk - q0)) for qk in parts]  # each relative to q(t), so wraps cancel
    out: NDArray[np.float64] = (d[0] - 8.0 * d[1] + 8.0 * d[2] - d[3]) / (12.0 * h)
    return out


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
    ) -> None:
        self.label = label
        self.parameter = parameter
        self.periodic = periodic
        self._eval = eval_fn
        self._domain_fn = domain_fn
        self._deriv = deriv_fn

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
        """
        ts = np.asarray(t, dtype=np.float64)
        scalar = ts.ndim == 0
        out = self._eval(np.atleast_1d(ts))
        return out[0] if scalar else out

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

        On the SRS family this is the closed form (rotor derivative of the
        swivel orbit pushed through the two 3-axis decompositions); on the
        spherical-shoulder family it is a fourth-order central difference of the
        closed-form ``q(t)`` with a 1e-4 step, exact to ~1e-12 away from folds.
        """
        ts = np.asarray(t, dtype=np.float64)
        scalar = ts.ndim == 0
        d = (
            self._deriv(np.atleast_1d(ts))
            if self._deriv is not None
            else _stencil(self._eval, np.atleast_1d(ts))
        )
        rate = np.linalg.norm(d, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = d / rate[:, None]
        return (direction[0], rate[0]) if scalar else (direction, rate)

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


class ChartFamily:
    """Every chart of the self-motion manifold at one pose, with the inverse map.

    Charts are created on demand and cached, so :meth:`locate` never pays for
    charts it does not return; :attr:`charts` enumerates the non-empty ones,
    which requires every domain.
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
    ) -> None:
        self.parameter = parameter
        self.periodic = periodic
        self.native = native
        """True when the family is backed by the native C++ extension."""
        self._param_of = param_of
        self._make_chart = make_chart
        self._nonempty = nonempty
        self._locate = locate
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

    def labels(self) -> list[tuple[int, ...]]:
        return [c.label for c in self.charts]

    def param_of(self, q: ArrayLike) -> float:
        """The redundancy coordinate of a configuration ``q``.

        Pure geometry, no branch search: ``q6`` itself for the spherical-shoulder
        family, the elbow's swivel angle on the shoulder-wrist circle for SRS.
        Meaningful only when ``FK(q)`` is the family's pose.
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
) -> ChartFamily:
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
) -> ChartFamily:
    from ssik.kinematics._scalar3 import _se3_inv
    from ssik.solvers.seven_r import spherical_shoulder as sh

    coef = _bake_cached(kb)
    t_rev = _se3_inv(T)
    arcs = sh.elbow_arcs(coef, t_rev)  # exact elbow folds, closed form

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
        return Chart(
            label=(e, s, w, k),
            parameter="q6",
            periodic=False,
            eval_fn=make_eval(slot, k),
            domain_fn=lambda: arc_domains(k)[slot],
        )

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

    return ChartFamily(
        parameter="q6",
        periodic=False,
        param_of=param_of,
        make_chart=make_chart,
        nonempty=nonempty,
        locate=locate,
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
        out: NDArray[np.float64] = np.linalg.solve(cols, omega)
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


def _srs_family(kb: KinBody, T: NDArray[np.float64], policy: TolerancePolicy) -> ChartFamily:
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
        )

    def locate(q: NDArray[np.float64], tol: float) -> tuple[int, float]:
        psi = param_of(q)
        for i, br in enumerate(branches):
            if _wrap_dist(br.q(psi), q) <= tol:
                return i, psi
        return -1, psi

    return ChartFamily(
        parameter="swivel",
        periodic=True,
        param_of=param_of,
        make_chart=make_chart,
        nonempty=lambda: list(range(len(branches))),
        locate=locate,
    )


# ---------------------------------------------------------------------------
# Native (C++) backend: cpp/include/ssik_cpp/chart.hpp via ssik._ssik_native
# ---------------------------------------------------------------------------


def _native_ext() -> Any:
    from ssik._native import _load_ext

    ext = _load_ext()
    return ext if ext is not None and hasattr(ext, "SphericalShoulderCharts") else None


def _spherical_shoulder_family_native(
    ext: Any, kb: KinBody, T: NDArray[np.float64], policy: TolerancePolicy
) -> ChartFamily:
    from ssik.solvers.seven_r import spherical_shoulder as sh

    nat = ext.SphericalShoulderCharts(
        np.ascontiguousarray(_bake_cached(kb)),
        np.ascontiguousarray(T),
        policy.subproblem_feasibility,
        policy.subproblem_degeneracy,
    )

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
        )

    def param_of(q: NDArray[np.float64]) -> float:
        return float(nat.param_of(np.ascontiguousarray(q, dtype=np.float64)))

    def locate(q: NDArray[np.float64], tol: float) -> tuple[int, float]:
        idx, t, _dist = nat.locate(np.ascontiguousarray(q, dtype=np.float64), tol)
        return int(idx), float(t)

    return ChartFamily(
        parameter="q6",
        periodic=False,
        param_of=param_of,
        make_chart=make_chart,
        nonempty=lambda: [i for i in range(len(nat)) if nat.nonempty(i)],
        locate=locate,
        native=True,
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


def _srs_family_native(ext: Any, kb: KinBody, T: NDArray[np.float64]) -> ChartFamily | None:
    from ssik._native import _srs_native_args

    args = _srs_native_args(kb)
    if args is None:
        return None
    nat = ext.SrsCharts(*(args[k] for k in _SRS_ARG_KEYS), np.ascontiguousarray(T))
    labels = np.asarray(nat.labels)

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
        )

    def param_of(q: NDArray[np.float64]) -> float:
        return float(nat.param_of(np.ascontiguousarray(q, dtype=np.float64)))

    def locate(q: NDArray[np.float64], tol: float) -> tuple[int, float]:
        idx, psi, _dist = nat.locate(np.ascontiguousarray(q, dtype=np.float64), tol)
        return int(idx), float(psi)

    return ChartFamily(
        parameter="swivel",
        periodic=True,
        param_of=param_of,
        make_chart=make_chart,
        nonempty=lambda: list(range(len(nat))),
        locate=locate,
        native=True,
    )
