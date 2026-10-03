"""Streaming IK: follow a stream of target poses one configuration at a time.

:meth:`Manipulator.solve_path` tracks a whole pose list offline. A teleoperated
arm sees the poses one at a time, from a VR controller, a SpaceMouse, a mocap
stream or a transform gizmo, and has to answer each before the next arrives. A
:class:`Tracker` holds the state that takes: the branch it is on, the
configuration it last commanded, and the time of the last pose.
:meth:`Tracker.update` takes one pose and returns one :class:`TrackerStep`.

Each update is a continuation of the branch the tracker is on, never a fresh
solve that might land on another one:

- On a redundant 7R arm with a closed-form chart (``seven_r.spherical_shoulder``,
  ``seven_r.srs``) the branch is continued on the charts as
  :func:`ssik.chart.track_all` continues it: the redundancy coordinate is held
  fixed and the chart point there nearest the followed one is taken, so the
  elbow does not flip and a tracked stretch returns what ``solve_path``
  returns for the same poses (``docs/api.md``, "Streaming IK", says where the
  two can differ).
- On every other arm it is the seeded solve,
  ``solve(T, q_seed=q, max_solutions=1, allow_rescue=False)``, ranked by ssik's seed metric
  (``wrap_linf``). At a singular pose ``solve()`` returns the point of a
  continuum nearest the seed (``docs/api.md``, "Singular continua"), so the
  continuation is well defined through a singularity.
- Where the chart continuation fails (the branch collided with another, or
  its point at the held coordinate is outside the joint limits) a 7R arm falls
  back to the seeded solve.

The candidate is then judged, and :class:`TrackStatus` reports the outcome.
On a 7R chart arm :meth:`Tracker.set_redundancy` is the other input: it slides
the followed point along its chart at the held target, so the elbow moves and
the hand does not. ``docs/api.md`` ("Streaming IK") is the normative statement
of the statuses and thresholds; this module is their implementation.
"""

from __future__ import annotations

import enum
import math
import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik.core.solution import Solution

if TYPE_CHECKING:
    from ssik.chart import Chart
    from ssik.manipulator import Manipulator

__all__ = ["Redundancy", "TrackStatus", "Tracker", "TrackerStep"]

# The redundant 7R families whose branches are continued on the chart, as
# solve_path does. The UR-class 6R arms also have charts, but their charts
# carry the solver's raw samples, without the continuum rule solve() applies
# at a singular pose, so 6R arms track with the seeded solve.
_CHART_FAMILIES = frozenset({"seven_r.spherical_shoulder", "seven_r.srs"})

HoldReason = Literal["unreachable", "limits", "jump"]

# set_redundancy samples the slide every this many radians of the coordinate,
# finer where a joint would step further than _SLIDE_MAX_STEP between samples,
# so the slide is followed continuously (windings kept) and a rate limit is
# applied along the chart rather than across it.
_SLIDE_DT = 0.01
_SLIDE_MAX_STEP = 0.05
_SLIDE_MAX_SAMPLES = 4096
# A rate-limited slide stops within (_SLIDE_DT / _NARROW_POINTS**_NARROW_ROUNDS)
# ~ 1e-8 of the coordinate where the first joint reaches its limit.
_NARROW_ROUNDS = 4
_NARROW_POINTS = 34
# Slack on coordinates compared against an arc: in_limits() places its ends to
# ~1e-10 by bisection, and a point held at a joint limit sits inside the limit
# band (docs/api.md, "Joint limits").
_COORD_TOL = 1e-8
_END_INSETS = (1e-8, 1e-7, 1e-6)
_TWO_PI = 2.0 * math.pi
# A chart point closes FK to ~1e-13 except within ~1e-5 rad of a fold, where a
# double root is determined only to sqrt(eps) (Chart.q). One Gauss-Newton step
# on FK restores closure there; it runs only above this residual.
_POLISH_ABOVE = 1e-11


class TrackStatus(enum.Enum):
    """What one :meth:`Tracker.update` did."""

    #: The configuration is the branch's continuation at the new target.
    OK = "ok"
    #: Same branch, but the move was clamped to ``max_joint_speed * dt``; the
    #: configuration lags the target and catches up over later updates.
    LIMITED = "limited"
    #: The last configuration was kept; :attr:`TrackerStep.reason` says why.
    HELD = "held"
    #: The tracker switched branch (``allow_jump=True``, or
    #: :meth:`Tracker.next_branch`).
    JUMPED = "jumped"


@dataclass(frozen=True, eq=False)
class TrackerStep:
    """One :meth:`Tracker.update` (or :meth:`Tracker.next_branch`) result.

    :param q: the configuration to command, ``(dof,)``, read-only. Under
        ``HELD`` it is the previous one, unchanged.
    :param status: the :class:`TrackStatus`.
    :param reason: under ``HELD``, why: ``"unreachable"`` (no configuration
        reaches the target), ``"limits"`` (the branch continues only outside
        the joint limits, or no in-limit configuration exists) or ``"jump"``
        (the nearest configuration is further than ``jump_threshold``: a branch
        switch). ``None`` otherwise.
    :param fk_residual: ``||FK(q) - T||_F`` against the target of this update.
        Within the solver's tolerance under ``OK`` and ``JUMPED`` without a rate
        limit; larger under ``LIMITED`` (the arm is still on its way) and
        ``HELD`` (the arm did not follow).
    :param moved: how far ``q`` moved from the previous step's ``q``, in ssik's
        seed metric (the largest single-joint move, continuous joints compared
        on the circle).
    :param branch_distance: the seed-metric distance from the branch point the
        tracker was following to the candidate at this target; ``nan`` when
        there was none. Compared against ``jump_threshold``.
    :param lag: the seed-metric distance from ``q`` to the branch point being
        followed; ``0`` unless a rate limit is holding the arm back.
    :param t: the timestamp passed to this update, or ``None``.
    """

    q: NDArray[np.float64]
    status: TrackStatus
    reason: HoldReason | None
    fk_residual: float
    moved: float
    branch_distance: float
    lag: float
    t: float | None

    @property
    def reachable(self) -> bool:
        """``False`` only when no configuration reaches the target."""
        return self.reason != "unreachable"


@dataclass(frozen=True, eq=False)
class Redundancy:
    """Where a :class:`Tracker` is on its arm's self-motion (:attr:`Tracker.redundancy`).

    :param chart: the followed chart, at :attr:`Tracker.target`.
    :param label: its branch label (``chart.label``).
    :param parameter: the redundancy coordinate's name: ``"q6"``
        (``seven_r.spherical_shoulder``) or ``"swivel"`` (``seven_r.srs``).
    :param periodic: whether the coordinate lives on a circle (the swivel).
    :param t: the coordinate of the followed point (the point the arm is at,
        unless a rate limit holds it back).
    :param arc: ``(lo, hi)`` with ``lo <= t <= hi``: the stretch of the chart
        :meth:`Tracker.set_redundancy` can slide along. It is the
        :meth:`~ssik.chart.Chart.in_limits` arc containing ``t`` (the chart's
        domain interval when the tracker does not respect limits); a fold or a
        branch junction is an end, and a single in-limit point is ``(t, t)``.
        On the periodic swivel the pieces that meet at ``+-pi`` are one arc,
        shifted by ``2*pi`` where needed to contain ``t``; ``(-pi, pi)`` when
        the whole circle is in limits.
    """

    chart: Chart
    label: tuple[int, ...]
    parameter: str
    periodic: bool
    t: float
    arc: tuple[float, float]


def _check_positive(x: Any, name: str, dof: int | None = None) -> float | NDArray[np.float64]:
    """A finite real ``> 0``, or with ``dof`` also a ``(dof,)`` vector of them."""
    if dof is not None and not isinstance(x, numbers.Real):
        from ssik._solve_inputs import as_real_array

        arr = as_real_array(x, name)
        if arr.shape != (dof,):
            raise ValueError(f"{name} must be a number or have shape ({dof},), got {arr.shape}")
        if not all(math.isfinite(v) and v > 0 for v in arr.tolist()):
            raise ValueError(f"{name} must be finite and > 0")
        return arr.copy()
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(x).__name__}")
    v = float(x)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and > 0, got {v}")
    return v


def _check_time(t: Any) -> float | None:
    if t is None:
        return None
    if isinstance(t, bool) or not isinstance(t, numbers.Real):
        raise TypeError(f"t must be a real number or None, got {type(t).__name__}")
    v = float(t)
    if not math.isfinite(v):
        raise ValueError(f"t must be finite, got {v}")
    return v


def _readonly(q: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.array(q, dtype=np.float64)
    out.flags.writeable = False
    return out


class Tracker:
    """Streaming, stateful IK on one :class:`~ssik.Manipulator`.

    ::

        tracker = arm.tracker(q_robot, max_joint_speed=2.0)
        for T, t in source.poses():
            step = tracker.update(T, t)
            robot.command(step.q)

    :param arm: the arm. A shipped arm is ``Manipulator.from_prebuilt(name)``,
        whose solve is the artifact's own.
    :param q0: the arm's current configuration, ``(dof,)``.
    :param max_joint_speed: optional joint-speed limit in rad/s (m/s for a
        prismatic joint), one number for every joint or one per joint. It
        applies only to updates that carry a timestamp ``t`` after an earlier
        one: the move is then scaled down, direction kept, until no joint moves
        more than ``max_joint_speed * dt``. ``None`` (default): no limit.
    :param jump_threshold: the largest seed-metric distance (radians) from the
        branch being followed to the candidate that still counts as the same
        branch. Default ``0.5``, ``solve_path``'s ``max_step``. It is a property
        of the branch geometry, not of time: a fast hand never triggers it,
        and a slow flip always does.
    :param allow_jump: when ``False`` (default) a candidate beyond
        ``jump_threshold`` is refused (``HELD``, reason ``"jump"``); when
        ``True`` the tracker switches to it and reports ``JUMPED``.
    :param respect_limits: keep every commanded configuration within the joint
        limits (default). ``False`` follows the geometric branch regardless.
    :param native: use the C++ extension when it is available (default);
        ``False`` forces the pure-Python reference.
    :param t0: optional timestamp of ``q0``, so the first update is rate
        limited too.

    :raises TypeError: if ``q0`` is not an array of real numbers, or a numeric
        option is not a number.
    :raises ValueError: if ``q0`` is not a finite ``(dof,)`` vector, or
        ``max_joint_speed`` / ``jump_threshold`` is not finite and positive.
    """

    __slots__ = (
        "_allow_jump",
        "_arm",
        "_chart_family",
        "_circular",
        "_goal",
        "_head",
        "_jump",
        "_kb",
        "_native",
        "_policy",
        "_q",
        "_respect",
        "_speed",
        "_t_last",
        "_target",
    )

    def __init__(
        self,
        arm: Manipulator,
        q0: ArrayLike,
        *,
        max_joint_speed: float | ArrayLike | None = None,
        jump_threshold: float = 0.5,
        allow_jump: bool = False,
        respect_limits: bool = True,
        native: bool = True,
        t0: float | None = None,
    ) -> None:
        from ssik.postprocess import _circular_mask

        self._arm = arm
        self._kb = arm.kinbody
        self._policy = arm._policy
        self._speed = (
            None
            if max_joint_speed is None
            else _check_positive(max_joint_speed, "max_joint_speed", arm.dof)
        )
        jt = _check_positive(jump_threshold, "jump_threshold")
        assert isinstance(jt, float)
        self._jump = jt
        self._allow_jump = bool(allow_jump)
        self._respect = bool(respect_limits)
        self._native = bool(native)
        self._chart_family = arm.solver_name in _CHART_FAMILIES
        self._circular = np.asarray(_circular_mask(self._kb, arm.dof), dtype=bool)
        self.reset(q0, t=t0)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def q(self) -> NDArray[np.float64]:
        """The configuration last returned (``q0`` before any update), read-only."""
        return _readonly(self._q)

    @property
    def target(self) -> NDArray[np.float64]:
        """The pose of the branch point being followed: the last target the
        tracker accepted (``FK(q0)`` before any update), read-only."""
        return _readonly(self._target)

    def reset(self, q: ArrayLike, *, t: float | None = None) -> None:
        """Restart from configuration ``q`` (the arm is there now), with
        optional timestamp ``t``. The branch is whichever ``q`` is on.

        :raises TypeError: if ``q`` is not an array of real numbers or ``t`` is
            not a number.
        :raises ValueError: if ``q`` is not a finite ``(dof,)`` vector or ``t``
            is not finite.
        """
        from ssik._solve_inputs import check_joints

        q_arr = check_joints(q, self._arm.dof, "q0").copy()
        self._t_last = _check_time(t)
        self._q = q_arr
        self._goal = q_arr.copy()
        self._target = self._arm.fk(q_arr)
        self._head: tuple[Chart, float] | None = None
        if self._chart_family:
            self._head = self._locate(self._target, q_arr)

    def __repr__(self) -> str:
        return (
            f"<Tracker on {self._arm.solver_name}: jump_threshold={self._jump}, "
            f"max_joint_speed={self._speed}, allow_jump={self._allow_jump}>"
        )

    # ------------------------------------------------------------------
    # Metric
    # ------------------------------------------------------------------

    def _delta(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
        """``b - a`` per joint, continuous joints the short way round."""
        d = b - a
        c = self._circular
        if c.any():
            d[c] = (d[c] + np.pi) % (2.0 * np.pi) - np.pi
        return d

    def _dist(self, a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        """ssik's seed metric (``wrap_linf``): the largest single-joint move."""
        return float(np.max(np.abs(self._delta(a, b))))

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def _charts(self, T: NDArray[np.float64]) -> Any:
        from ssik.chart import charts

        return charts(
            self._kb,
            T,
            solver_name=self._arm.solver_name,
            policy=self._policy,
            native=self._native,
        )

    def _locate(self, T: NDArray[np.float64], q: NDArray[np.float64]) -> tuple[Chart, float] | None:
        located: tuple[Chart, float] | None = self._charts(T).locate(q)
        return located

    def _seeded(self, T: NDArray[np.float64], respect_limits: bool) -> list[Solution]:
        kwargs: dict[str, Any] = {}
        if self._arm._prebuilt is not None:
            # Only an artifact's solve takes the backend switch; the live solve
            # is the Python one.
            kwargs["native"] = self._native
        # No T-perturbation rescue: it exists for a measure-zero ridge where the
        # analytical path finds nothing, and costs tens of milliseconds native
        # and seconds in Python on every update it runs -- which, with the
        # rescue on, is every update the target is out of limits or out of
        # reach. A tracker holds one update at such a ridge instead.
        sols: list[Solution] = self._arm.solve(
            T,
            q_seed=self._goal,
            max_solutions=1,
            respect_limits=respect_limits,
            allow_rescue=False,
            **kwargs,
        )
        return sols

    def _admit(
        self,
        q: NDArray[np.float64],
        T: NDArray[np.float64],
        seed: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64] | None:
        """A chart point as a commanded configuration: the representative
        nearest ``seed`` (default: the followed branch point), inside the
        limits when they are respected (``None`` if it cannot be), as
        ``solve()`` reports one."""
        from ssik.postprocess import respect_limits, rewrap_to_seed, wrap_to_limits

        res = float(np.linalg.norm(self._arm.fk(q) - T))
        sols = [Solution(q=np.asarray(q, dtype=np.float64), fk_residual=res)]
        if self._respect:
            sols = respect_limits(wrap_to_limits(sols, self._kb, T_target=T), self._kb, T_target=T)
            if not sols:
                return None
        sols = rewrap_to_seed(sols, self._kb, self._goal if seed is None else seed, T_target=T)
        return np.asarray(sols[0].q, dtype=np.float64)

    def _held_point(self, fam: Any) -> tuple[Chart, float, NDArray[np.float64]] | None:
        """The followed branch at a new pose: of every chart's point at the held
        redundancy coordinate, the one nearest the followed point (every joint
        compared on the circle), or ``None`` when none is within
        ``jump_threshold``.

        This is :meth:`~ssik.chart.SelfMotionManifold.continue_from` with the
        label lookup replaced by the nearest point it is meant to find. The two
        agree wherever the label contract holds; near a fold of the
        spherical-shoulder charts the old label can still name a point within
        ``max_step`` that is not the continuation (the relabelled chart is
        nearer), and the lookup would detour through it.
        """
        from ssik.chart import _wrap_dist, _wrap_pi

        assert self._head is not None
        t = self._head[1]
        if fam.periodic:
            t = float(_wrap_pi(np.asarray(t)))
        best: tuple[float, Chart, NDArray[np.float64]] | None = None
        for chart in fam.charts:
            qc = chart.q(t)
            if not np.all(np.isfinite(qc)):
                continue
            d = _wrap_dist(qc, self._goal)
            if best is None or d < best[0]:
                best = (d, chart, qc)
        if best is None or best[0] > self._jump:
            return None
        return best[1], t, best[2]

    def _candidate(
        self, T: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64] | None, tuple[Chart, float] | None, bool]:
        """The continuation of the followed branch at ``T``, its chart location
        (7R chart arms), and whether the branch's own continuation was found
        but lies outside the joint limits. ``(None, None, ...)`` when nothing
        reaches ``T`` within the limits."""
        fam = None
        blocked = False
        if self._chart_family:
            fam = self._charts(T)
            if self._head is not None:
                held = self._held_point(fam)
                if held is not None:
                    chart, t, qc = held
                    admitted = self._admit(qc, T)
                    if admitted is not None:
                        return admitted, (chart, t), False
                    blocked = True
        sols = self._seeded(T, self._respect)
        if not sols:
            return None, None, blocked
        q = np.asarray(sols[0].q, dtype=np.float64)
        head = fam.locate(q) if fam is not None else None
        return q, head, blocked

    def _hold_reason(self, T: NDArray[np.float64], best: float, blocked: bool) -> HoldReason:
        """Why the tracker holds, when its best in-limit candidate is ``best``
        away (``inf`` for none) and ``blocked`` says the branch's continuation
        was found outside the limits. Runs only on the hold path."""
        if blocked:
            return "limits"
        if not self._respect:
            return "unreachable" if math.isinf(best) else "jump"
        raw = self._seeded(T, respect_limits=False)
        if not raw:
            return "unreachable"
        if math.isinf(best) or self._dist(self._goal, np.asarray(raw[0].q)) <= self._jump:
            return "limits"
        return "jump"

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _toward(
        self, goal: NDArray[np.float64], dt: float | None
    ) -> tuple[NDArray[np.float64], bool]:
        """Move from the last commanded configuration toward ``goal``, scaled
        down (direction kept) to the rate limit; ``True`` when it clamped."""
        if self._speed is None or dt is None:
            return goal.copy(), False
        delta = self._delta(self._q, goal)
        cap = np.broadcast_to(np.asarray(self._speed, dtype=np.float64) * dt, delta.shape)
        mag = np.abs(delta)
        over = mag > cap
        if not over.any():
            return goal.copy(), False
        scale = float(np.min(np.where(over, cap / np.where(over, mag, 1.0), 1.0)))
        return self._q + scale * delta, True

    def _step(
        self,
        q_new: NDArray[np.float64],
        status: TrackStatus,
        T: NDArray[np.float64],
        t: float | None,
        reason: HoldReason | None = None,
        branch_distance: float = math.nan,
        lag_to: NDArray[np.float64] | None = None,
    ) -> TrackerStep:
        moved = self._dist(self._q, q_new)
        self._q = q_new
        return TrackerStep(
            q=_readonly(q_new),
            status=status,
            reason=reason,
            fk_residual=float(np.linalg.norm(self._arm.fk(q_new) - T)),
            moved=moved,
            branch_distance=branch_distance,
            lag=self._dist(q_new, self._goal if lag_to is None else lag_to),
            t=t,
        )

    def _advance_time(self, t: Any) -> tuple[float | None, float | None]:
        """Validate timestamp ``t`` against the last one; return it and the
        ``dt`` since the last one (``None`` without both), and record it."""
        tt = _check_time(t)
        if tt is not None and self._t_last is not None and tt < self._t_last:
            raise ValueError(f"t must not decrease: got {tt} after {self._t_last}")
        dt = None if tt is None or self._t_last is None else tt - self._t_last
        if tt is not None:
            self._t_last = tt
        return tt, dt

    def update(self, T: ArrayLike, t: float | None = None) -> TrackerStep:
        """Follow the branch to target ``T`` (timestamp ``t``, seconds).

        - ``OK``: the continuation is within ``jump_threshold`` and, under a
          rate limit, reachable this tick. ``q`` closes FK to ``T``.
        - ``LIMITED``: same branch, but the move was clamped to
          ``max_joint_speed * dt``. Later updates keep closing the gap.
        - ``HELD``: the previous ``q`` is returned unchanged; ``reason`` is
          ``"unreachable"``, ``"limits"`` or ``"jump"``. The tracker keeps
          following the branch it was on, so it resumes when the target
          comes back within reach.
        - ``JUMPED``: with ``allow_jump=True``, a candidate beyond
          ``jump_threshold`` was taken (rate-limited like any move).

        :param T: 4x4 target pose in the arm's base frame (the flange, as for
            ``solve()``).
        :param t: optional timestamp. Timestamps must not decrease.

        :raises TypeError: if ``T`` is not an array of real numbers or ``t`` is
            not a number.
        :raises ValueError: if ``T`` is not a finite rigid ``(4, 4)`` transform
            (``docs/api.md``, "Input validation"), or ``t`` is not finite or is
            earlier than the previous timestamp.
        """
        from ssik._solve_inputs import check_pose

        T_arr = check_pose(T, policy=self._policy, name="T")
        tt, dt = self._advance_time(t)

        cand, head, blocked = self._candidate(T_arr)
        dist = math.inf if cand is None else self._dist(self._goal, cand)
        if cand is None or (dist > self._jump and not self._allow_jump):
            reason = self._hold_reason(T_arr, dist, blocked)
            reported = math.nan if cand is None else dist
            return self._step(self._q.copy(), TrackStatus.HELD, T_arr, tt, reason, reported)

        assert cand is not None
        jumped = dist > self._jump
        self._goal = cand
        self._target = T_arr
        if self._chart_family:
            self._head = head
        q_new, limited = self._toward(cand, dt)
        if jumped:
            status = TrackStatus.JUMPED
        else:
            status = TrackStatus.LIMITED if limited else TrackStatus.OK
        return self._step(q_new, status, T_arr, tt, branch_distance=dist)

    # ------------------------------------------------------------------
    # Branches
    # ------------------------------------------------------------------

    def solutions(self, max_solutions: int | None = None) -> list[Solution]:
        """Every configuration at :attr:`target`, nearest the followed branch
        first: ``solve(target, q_seed=..., enumerate_windings=False)`` under
        the tracker's ``respect_limits``. On a redundant 7R arm these are
        ``solve()``'s samples of the self-motion. For rendering the other
        branches ("ghost arms"); :meth:`next_branch` switches to one.
        """
        kwargs: dict[str, Any] = {}
        if self._arm._prebuilt is not None:
            kwargs["native"] = self._native
        sols: list[Solution] = self._arm.solve(
            self._target,
            q_seed=self._goal,
            max_solutions=max_solutions,
            respect_limits=self._respect,
            enumerate_windings=False,
            **kwargs,
        )
        return sols

    def _branch_points(self) -> list[tuple[NDArray[np.float64], tuple[Chart, float] | None]]:
        """The branches at :attr:`target` in a fixed order, each as the
        configuration the tracker would follow on it."""
        if self._chart_family and self._head is not None:
            fam = self._charts(self._target)
            t = self._head[1]
            if fam.periodic:
                t = float((t + np.pi) % (2.0 * np.pi) - np.pi)
            out: list[tuple[NDArray[np.float64], tuple[Chart, float] | None]] = []
            for chart in sorted(fam.charts, key=lambda c: c.label):
                tc = t
                if not chart.contains(tc):
                    lo, hi = max(chart.domain, key=lambda d: d[1] - d[0])
                    tc = min(max(t, lo), hi)
                qc = chart.q(tc)
                if not np.all(np.isfinite(qc)):
                    continue
                admitted = self._admit(qc, self._target)
                if admitted is not None:
                    out.append((admitted, (chart, tc)))
            return out
        sols = self.solutions()
        qs = sorted(
            (np.asarray(s.q, dtype=np.float64) for s in sols),
            key=lambda q: tuple(round(float(v), 9) for v in q),
        )
        return [(q, None) for q in qs]

    def next_branch(self) -> TrackerStep | None:
        """Switch to the next branch at :attr:`target`, cycling through them
        in a fixed order (on a 7R chart arm: by chart label, at the held
        redundancy coordinate; otherwise: by configuration). Returns the
        ``JUMPED`` step, or ``None`` when there is no other branch.

        Without a rate limit the returned ``q`` is on the new branch. With
        ``max_joint_speed`` set the arm does not move here: the new branch
        becomes the one followed, and later timestamped updates carry the arm
        there at the joint-speed limit (``LIMITED`` until it arrives).
        """
        points = self._branch_points()
        if len(points) < 2:
            return None
        cur = min(range(len(points)), key=lambda i: self._dist(self._goal, points[i][0]))
        q_next, head = points[(cur + 1) % len(points)]
        if self._dist(self._goal, q_next) == 0.0:
            return None
        dist = self._dist(self._goal, q_next)
        self._goal = q_next
        if self._chart_family:
            self._head = head
        q_new = q_next.copy() if self._speed is None else self._q.copy()
        return self._step(
            q_new, TrackStatus.JUMPED, self._target, self._t_last, branch_distance=dist
        )

    # ------------------------------------------------------------------
    # Redundancy
    # ------------------------------------------------------------------

    def _arc(self, chart: Chart, t: float) -> tuple[float, float]:
        """The stretch of ``chart`` containing ``t`` that a slide may cover:
        its in-limits arc (its domain interval when limits are not respected),
        ``(t, t)`` when none contains ``t``. On a periodic chart the pieces
        meeting at ``+-pi`` are one arc, shifted by ``2*pi`` to contain ``t``."""
        pieces = list(chart.in_limits() if self._respect else chart.domain)
        if chart.periodic:
            low = [p for p in pieces if p[0] <= -math.pi + _COORD_TOL]
            high = [p for p in pieces if p[1] >= math.pi - _COORD_TOL]
            if low and high and low[0] is not high[0]:
                pieces = [p for p in pieces if p is not low[0] and p is not high[0]]
                pieces.append((high[0][0], low[0][1] + _TWO_PI))
            for lo, hi in pieces:
                if hi - lo >= _TWO_PI - _COORD_TOL:
                    return (-math.pi, math.pi)
                for shift in (0.0, -_TWO_PI):
                    if lo + shift - _COORD_TOL <= t <= hi + shift + _COORD_TOL:
                        return (lo + shift, hi + shift)
            return (t, t)
        for lo, hi in pieces:
            if lo - _COORD_TOL <= t <= hi + _COORD_TOL:
                return (lo, hi)
        return (t, t)

    @property
    def redundancy(self) -> Redundancy | None:
        """Where the tracker is on the self-motion: the followed chart, its
        coordinate, and the arc :meth:`set_redundancy` can slide along.

        ``None`` on an arm without a closed-form chart (6R arms, and 7R arms
        other than ``seven_r.spherical_shoulder`` / ``seven_r.srs``), and on a
        chart arm whose followed point no chart locates (a branch junction).
        Computed when read (the chart's :meth:`~ssik.chart.Chart.in_limits`),
        so :meth:`update` pays nothing for it.
        """
        if not self._chart_family or self._head is None:
            return None
        chart, t = self._head
        if chart.periodic:
            t = _wrap(t)
        return Redundancy(
            chart=chart,
            label=chart.label,
            parameter=chart.parameter,
            periodic=chart.periodic,
            t=float(t),
            arc=self._arc(chart, t),
        )

    def _on_circle(self, d: NDArray[np.float64]) -> NDArray[np.float64]:
        """Every joint difference wrapped to ``[-pi, pi)``: a chart returns one
        representative of each angle, so its values are compared on the circle
        before they are followed."""
        return np.asarray((d + np.pi) % _TWO_PI - np.pi, dtype=np.float64)

    @staticmethod
    def _snap(qs: NDArray[np.float64], near: NDArray[np.float64]) -> NDArray[np.float64]:
        """``qs`` moved by whole turns to the representative nearest ``near``."""
        return np.asarray(qs + _TWO_PI * np.round((near - qs) / _TWO_PI), dtype=np.float64)

    def _chart_q(self, chart: Chart, t: float, near: NDArray[np.float64]) -> NDArray[np.float64]:
        q = chart.q(_wrap(t) if chart.periodic else t)
        return self._snap(np.asarray(q, dtype=np.float64), near)

    def _slide_path(
        self, chart: Chart, t0: float, t1: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """``(ts, qs)``: the chart from ``t0`` to ``t1``, sampled finely enough
        that no joint steps more than ``_SLIDE_MAX_STEP``, each row in the
        winding continuous with the followed point. ``None`` if the chart does
        not exist along the way."""
        n = max(2, math.ceil(abs(t1 - t0) / _SLIDE_DT) + 1)
        while True:
            ts = np.linspace(t0, t1, n)
            qs = chart.q((ts + np.pi) % _TWO_PI - np.pi if chart.periodic else ts)
            if not np.all(np.isfinite(qs)):
                return None
            steps = self._on_circle(np.diff(qs, axis=0))
            if n > _SLIDE_MAX_SAMPLES or float(np.max(np.abs(steps))) <= _SLIDE_MAX_STEP:
                break
            n = 4 * n
        start = self._goal + self._on_circle(qs[0] - self._goal)
        cont = start + np.vstack([np.zeros((1, qs.shape[1])), np.cumsum(steps, axis=0)])
        return ts, self._snap(qs, cont)

    def _polish(self, q: NDArray[np.float64], T: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gauss-Newton steps on FK (two at most) for a chart point near a fold
        (``_POLISH_ABOVE``); ``q`` itself when it already closes."""
        from ssik.refinement import kinbody_jacobian, se3_log_residual

        best, best_res = q, float(np.linalg.norm(self._arm.fk(q) - T))
        for _ in range(2):
            if best_res <= _POLISH_ABOVE:
                break
            err = se3_log_residual(T @ np.linalg.inv(self._arm.fk(best)))
            dq = np.linalg.lstsq(kinbody_jacobian(self._kb, best), err, rcond=None)[0]
            cand = best + dq
            res = float(np.linalg.norm(self._arm.fk(cand) - T))
            if not res < best_res:
                break
            best, best_res = cand, res
        return best

    def _closes(self, q: NDArray[np.float64], T: NDArray[np.float64]) -> bool:
        return float(np.linalg.norm(self._arm.fk(q) - T)) <= _POLISH_ABOVE

    def _resolve(
        self, chart: Chart, t0: float, arc: tuple[float, float], value: float
    ) -> float | HoldReason:
        """The coordinate to slide to for a requested ``value``: the
        representative reached from ``t0`` without leaving ``arc``, or why
        there is none."""
        lo, hi = arc
        if chart.periodic:
            if hi - lo >= _TWO_PI - _COORD_TOL:  # the whole circle: the shorter way
                return t0 + float(self._on_circle(np.asarray(value - t0)))
            base = value + _TWO_PI * round((t0 - value) / _TWO_PI)
            inside = [
                r
                for r in (base - _TWO_PI, base, base + _TWO_PI)
                if lo - _COORD_TOL <= r <= hi + _COORD_TOL
            ]
            if inside:
                nearest: float = min(inside, key=lambda r: abs(r - t0))
                return min(max(nearest, lo), hi)
            return "limits" if chart.contains(value) else "jump"
        if lo - _COORD_TOL <= value <= hi + _COORD_TOL:
            return min(max(value, lo), hi)
        same_piece = any(
            lo_d - _COORD_TOL <= t0 <= hi_d + _COORD_TOL
            and lo_d - _COORD_TOL <= value <= hi_d + _COORD_TOL
            for lo_d, hi_d in chart.domain
        )
        return "limits" if same_piece else "jump"

    def set_redundancy(self, value: float, t: float | None = None) -> TrackerStep:
        """Slide along the self-motion at the held target: the hand stays at
        :attr:`target` while the followed chart's redundancy coordinate (``q6``
        or the swivel angle, :attr:`redundancy`) moves to ``value``.

        - ``OK``: ``value`` is on :attr:`redundancy`'s ``arc`` (on the swivel,
          some ``value + 2*pi*k`` is; with the whole circle in limits, the
          shorter way round is taken). ``q`` is the chart point there, followed
          along the arc from the current one, and closes FK to the target.
        - ``LIMITED`` (``max_joint_speed`` and timestamps): the slide goes
          along the chart only as far as no joint moves more than
          ``max_joint_speed * dt``, so the hand stays on the target. That
          point becomes the one followed, and ``lag`` is the distance still to
          go. The rest is not queued: sending ``value`` again on later ticks
          continues the slide, and the call that arrives is ``OK``.
        - ``HELD``: ``q`` unchanged. ``reason`` is ``"limits"`` when ``value``
          is on the chart but outside the arc, and ``"jump"`` when it is past a
          fold or a branch junction (going on would mean leaving the chart for
          another branch) or :attr:`redundancy` is ``None``.

        A slide stays on one branch however long it is, so ``moved`` can
        exceed ``jump_threshold`` without a rate limit; ``max_joint_speed``
        bounds the motion per tick. :meth:`update` then holds the coordinate
        set here as the target moves.

        :param value: the redundancy coordinate to slide to.
        :param t: optional timestamp, as for :meth:`update`.

        :raises NotImplementedError: on an arm without a closed-form chart, as
            ``Manipulator.self_motion`` raises (:attr:`redundancy` is always
            ``None`` there).
        :raises TypeError: if ``value`` or ``t`` is not a real number.
        :raises ValueError: if ``value`` or ``t`` is not finite, or ``t`` is
            earlier than the previous timestamp.
        """
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"value must be a real number, got {type(value).__name__}")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"value must be finite, got {v}")
        _check_time(t)
        if not self._chart_family:
            raise NotImplementedError(
                f"set_redundancy: {self._arm.solver_name} has no closed-form chart, so a "
                "Tracker on it has no redundancy coordinate"
            )
        tt, dt = self._advance_time(t)
        T = self._target
        red = self.redundancy
        if red is None:
            return self._step(self._q.copy(), TrackStatus.HELD, T, tt, "jump")
        chart, t0 = red.chart, red.t
        t1 = self._resolve(chart, t0, red.arc, v)
        if isinstance(t1, str):
            return self._step(self._q.copy(), TrackStatus.HELD, T, tt, t1)
        path = self._slide_path(chart, t0, t1)
        if path is None:
            return self._step(self._q.copy(), TrackStatus.HELD, T, tt, "jump")
        ts, qs = path
        requested = self._admit(self._polish(qs[-1], T), T, seed=qs[-1])
        lo, hi = red.arc
        inward = 1.0 if t1 - lo < hi - t1 else -1.0
        t_end = t1
        for inset in _END_INSETS:
            # An arc end found by in_limits() can sit a few 1e-8 rad past the
            # limit it marks, where the limits either refuse the point or
            # clamp it onto the limit (which costs FK closure ~1e-9). Step
            # in from it to the first point admitted as it is.
            if requested is not None and self._closes(requested, T):
                break
            t_in = t1 + inward * inset
            if not lo <= t_in <= hi:
                break
            q_in = self._chart_q(chart, t_in, qs[-1])
            inner = self._admit(self._polish(q_in, T), T, seed=q_in)
            if inner is not None and (requested is None or self._closes(inner, T)):
                requested, t_end = inner, t_in
        t1 = t_end
        if requested is None:
            return self._step(self._q.copy(), TrackStatus.HELD, T, tt, "limits")

        if self._speed is not None and dt is not None:
            cap = np.broadcast_to(np.asarray(self._speed, dtype=np.float64) * dt, qs[-1].shape)

            def within(q: NDArray[np.float64]) -> bool:
                return bool(np.all(np.abs(self._delta(self._q, q)) <= cap))

            first_out = next((i for i in range(len(ts)) if not within(qs[i])), None)
            if first_out == 0:
                # The arm is still behind the followed point (an earlier rate
                # limit): close that gap first, as update() does.
                q_new, _ = self._toward(self._goal, dt)
                return self._step(q_new, TrackStatus.LIMITED, T, tt, lag_to=requested)
            if first_out is not None:
                # Narrow down the furthest point of the slide the limit
                # reaches, one batched chart evaluation per round.
                a, b = float(ts[first_out - 1]), float(ts[first_out])
                q_a = qs[first_out - 1]
                for _ in range(_NARROW_ROUNDS):
                    sub = np.linspace(a, b, _NARROW_POINTS)[1:-1]
                    q_sub = chart.q((sub + np.pi) % _TWO_PI - np.pi if chart.periodic else sub)
                    for tm, qm in zip(sub, q_sub, strict=True):
                        qm = self._snap(qm, q_a)
                        if not within(qm):
                            b = float(tm)
                            break
                        a, q_a = float(tm), qm
                reached = self._admit(self._polish(q_a, T), T, seed=q_a)
                if reached is not None and within(reached):
                    self._goal = reached
                    self._head = (chart, _wrap(a) if chart.periodic else a)
                    return self._step(reached.copy(), TrackStatus.LIMITED, T, tt, lag_to=requested)
                return self._step(self._q.copy(), TrackStatus.LIMITED, T, tt, lag_to=requested)

        self._goal = requested
        self._head = (chart, _wrap(t1) if chart.periodic else t1)
        return self._step(requested.copy(), TrackStatus.OK, T, tt)


def _wrap(t: float) -> float:
    """A coordinate on the circle, in ``[-pi, pi)``."""
    return float((t + math.pi) % _TWO_PI - math.pi)
