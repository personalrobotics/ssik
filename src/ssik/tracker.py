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
``docs/api.md`` ("Streaming IK") is the normative statement of the statuses
and thresholds; this module is their implementation.
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

__all__ = ["TrackStatus", "Tracker", "TrackerStep"]

# The redundant 7R families whose branches are continued on the chart, as
# solve_path does. The UR-class 6R arms also have charts, but their charts
# carry the solver's raw samples, without the continuum rule solve() applies
# at a singular pose, so 6R arms track with the seeded solve.
_CHART_FAMILIES = frozenset({"seven_r.spherical_shoulder", "seven_r.srs"})

HoldReason = Literal["unreachable", "limits", "jump"]


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

    def _admit(self, q: NDArray[np.float64], T: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """A chart point as a commanded configuration: the representative
        nearest the followed branch point, inside the limits when they are
        respected (``None`` if it cannot be), as ``solve()`` reports one."""
        from ssik.postprocess import respect_limits, rewrap_to_seed, wrap_to_limits

        res = float(np.linalg.norm(self._arm.fk(q) - T))
        sols = [Solution(q=np.asarray(q, dtype=np.float64), fk_residual=res)]
        if self._respect:
            sols = respect_limits(wrap_to_limits(sols, self._kb, T_target=T), self._kb, T_target=T)
            if not sols:
                return None
        sols = rewrap_to_seed(sols, self._kb, self._goal, T_target=T)
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
            lag=self._dist(q_new, self._goal),
            t=t,
        )

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
        tt = _check_time(t)
        if tt is not None and self._t_last is not None and tt < self._t_last:
            raise ValueError(f"t must not decrease: got {tt} after {self._t_last}")
        dt = None if tt is None or self._t_last is None else tt - self._t_last
        if tt is not None:
            self._t_last = tt

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
