"""Streaming IK (``ssik.Tracker``) end to end, on both backends (#678).

Each scenario drives a tracker with a scripted pose stream built from a known
joint path ``q(s)``: the poses are ``FK(q(s))``, so wherever the path is
tracked the tracker must return the path itself. That is the oracle. The
streams pass through

- an exact wrist singularity (``q5 = 0`` on the IRB 120; the continuum rule of
  ``solve()`` makes the seeded continuation well defined there),
- a joint limit (the path drives a joint past its stop and back),
- an unreachable stretch (the target leaves the workspace and returns),
- a forced branch flip (the target teleports to another posture; and
  ``next_branch()``),
- a joint-speed limit with timestamps (``LIMITED``, then catch-up).

and assert the documented contract (``docs/api.md``, "Streaming IK"): no move
larger than ``jump_threshold`` unless reported ``JUMPED``; ``HELD`` returns the
previous configuration unchanged; ``OK`` closes FK to the target; ``LIMITED``
moves at most ``max_joint_speed * dt`` per joint; and on a redundant 7R arm the
tracker returns what ``solve_path`` returns for the same poses.

Artifact. Every scenario writes its per-update trace to
``<tmp_path>/tracker_trace_<scenario>.json``: one record per update with the
segment name, status, hold reason, ``q``, ``fk_residual``, ``moved``, ``lag``
and ``branch_distance``. To inspect it::

    uv run pytest tests/test_tracker.py --basetemp=/tmp/ssik-tracker
    ls /tmp/ssik-tracker/*/tracker_trace_*.json
    python -m json.tool /tmp/ssik-tracker/test_irb120_scenario_native_0/*.json

The streams are deterministic (no randomness), so the statuses and reasons
are identical run to run and across backends, and ``q`` agrees to round-off.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ssik
from ssik._native import native_available

BACKENDS = [
    pytest.param(
        True,
        id="native",
        marks=pytest.mark.skipif(not native_available(), reason="native extension not built"),
    ),
    pytest.param(False, id="python"),
]

FK_TOL = 1e-9  # FK closure under OK: the solvers close to ~1e-14 here
Q_TOL = 1e-9  # tracked q against the generating path


def _lerp(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    """``n`` evenly spaced joint vectors from ``a`` to ``b``, both included."""
    return [a + (b - a) * s for s in np.linspace(0.0, 1.0, n)]


def _wrapped(d: np.ndarray) -> float:
    return float(np.max(np.abs((d + np.pi) % (2 * np.pi) - np.pi)))


def _record(step: ssik.TrackerStep, segment: str) -> dict[str, Any]:
    def num(x: float) -> float | None:
        return None if math.isnan(x) else x

    return {
        "segment": segment,
        "status": step.status.name,
        "reason": step.reason,
        "q": step.q.tolist(),
        "fk_residual": step.fk_residual,
        "moved": step.moved,
        "lag": step.lag,
        "branch_distance": num(step.branch_distance),
        "t": step.t,
    }


def _write(tmp_path: Path, name: str, trace: list[dict[str, Any]]) -> Path:
    out = tmp_path / f"tracker_trace_{name}.json"
    out.write_text(json.dumps(trace, indent=1))
    # The artifact is the evidence: it must read back as the trace written.
    assert json.loads(out.read_text()) == trace
    return out


def _check_contract(steps: list[ssik.TrackerStep], jump: float, cap: float | None) -> None:
    """The invariants every stream satisfies, whatever happened in it."""
    for prev, st in itertools.pairwise(steps):
        assert not st.q.flags.writeable
        if st.status is ssik.TrackStatus.HELD:
            assert st.reason in ("unreachable", "limits", "jump")
            np.testing.assert_array_equal(st.q, prev.q)
            assert st.moved == 0.0
        else:
            assert st.reason is None
        if st.status is not ssik.TrackStatus.JUMPED:
            # No unreported branch switch: anything that is not JUMPED moved
            # the arm by at most the threshold (plus the rate-limit lag it
            # was already carrying, which is zero without a rate limit).
            assert st.moved <= jump + prev.lag + 1e-12
        if st.status is ssik.TrackStatus.OK:
            assert st.fk_residual <= FK_TOL
            assert st.lag <= 1e-12
        if cap is not None and st.t is not None and prev.t is not None:
            assert st.moved <= cap * (st.t - prev.t) + 1e-12


# ---------------------------------------------------------------------------
# 6R: IRB 120 (spherical wrist), the seeded-solve continuation
# ---------------------------------------------------------------------------

IRB_Q0 = np.array([0.3, 0.2, 0.1, 0.4, 0.6, 0.5])


def _irb120_stream(arm: ssik.Manipulator) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """``(segment, T, q_path or None)`` per update."""
    q_wrist = IRB_Q0.copy()
    q_wrist[4] = -0.6  # through the exact wrist singularity q5 = 0 (step 12)
    singular = _lerp(IRB_Q0, q_wrist, 25)
    assert abs(singular[12][4]) < 1e-15
    q_over = q_wrist.copy()
    q_over[1] = 2.2  # joint 2's upper limit is 1.92
    limit = _lerp(q_wrist, q_over, 30)[1:] + _lerp(q_over, q_wrist, 30)[1:]
    stream: list[tuple[str, np.ndarray, np.ndarray | None]] = []
    stream += [("singularity", arm.fk(q), q) for q in singular]
    stream += [("limit", arm.fk(q), q) for q in limit]
    far = arm.fk(q_wrist)
    far[:3, 3] *= 5.0  # five times the reach
    stream += [("unreachable", far, None)] * 3
    stream += [("return", arm.fk(q_wrist), q_wrist)]
    q_flip = q_wrist.copy()
    q_flip[3] += 1.2  # a posture 1.2 rad away: only a branch switch reaches it
    stream += [("flip", arm.fk(q_flip), q_flip)]
    return stream


@pytest.mark.parametrize("native", BACKENDS)
def test_irb120_scenario(native: bool, tmp_path: Path) -> None:
    arm = ssik.Manipulator.from_prebuilt("irb120")
    limits = arm.joint_limits[1]
    assert limits is not None
    upper = limits[1]
    stream = _irb120_stream(arm)
    tracker = arm.tracker(IRB_Q0, native=native)
    steps: list[ssik.TrackerStep] = []
    trace = []
    for i, (segment, T, _q_path) in enumerate(stream):
        st = tracker.update(T, t=0.01 * i)
        steps.append(st)
        trace.append(_record(st, segment))
    _write(tmp_path, f"irb120_{'native' if native else 'python'}", trace)
    _check_contract(steps, jump=0.5, cap=None)

    for (segment, _T, q_path), st in zip(stream, steps, strict=True):
        if segment in ("singularity", "return") or (
            segment == "limit" and q_path is not None and q_path[1] <= upper
        ):
            # Tracked: the path itself, through the singularity and back out
            # of the limit excursion.
            assert st.status is ssik.TrackStatus.OK, (segment, st)
            assert q_path is not None
            np.testing.assert_allclose(st.q, q_path, atol=Q_TOL)
        elif segment == "limit":
            assert (st.status, st.reason) == (ssik.TrackStatus.HELD, "limits")
            assert st.reachable
            assert st.q[1] <= upper
        elif segment == "unreachable":
            assert (st.status, st.reason) == (ssik.TrackStatus.HELD, "unreachable")
            assert not st.reachable
        else:  # flip: refused by default
            assert (st.status, st.reason) == (ssik.TrackStatus.HELD, "jump")
            assert st.branch_distance == pytest.approx(1.2, abs=1e-9)


@pytest.mark.parametrize("native", BACKENDS)
def test_irb120_allow_jump_and_next_branch(native: bool, tmp_path: Path) -> None:
    arm = ssik.Manipulator.from_prebuilt("irb120")
    q_flip = IRB_Q0.copy()
    q_flip[3] += 1.2
    T_flip = arm.fk(q_flip)
    tracker = arm.tracker(IRB_Q0, allow_jump=True, native=native)
    trace = []
    st = tracker.update(T_flip)
    trace.append(_record(st, "flip"))
    assert st.status is ssik.TrackStatus.JUMPED
    np.testing.assert_allclose(st.q, q_flip, atol=Q_TOL)
    assert st.fk_residual <= FK_TOL

    # next_branch cycles through every in-limit branch at the target, each a
    # real solution, and comes back to where it started.
    n = len(tracker.solutions())
    assert n >= 2
    seen = []
    for _ in range(n):
        step = tracker.next_branch()
        assert step is not None
        trace.append(_record(step, "next_branch"))
        assert step.status is ssik.TrackStatus.JUMPED
        assert step.fk_residual <= FK_TOL
        assert step.branch_distance > 0.0
        seen.append(step.q.copy())
    # Back on the starting branch, and every branch visited once. Compared on
    # the circle: joint 6 spans more than a turn, and after a wrist flip the
    # two representatives of the return are equally near (pi either way), so
    # which one a backend reports is round-off.
    assert _wrapped(seen[-1] - q_flip) <= Q_TOL
    assert len({tuple(np.round((q + np.pi) % (2 * np.pi) - np.pi, 6)) for q in seen}) == n
    lims = arm.joint_limits
    for q in seen:
        assert all(lim is None or lim[0] <= v <= lim[1] for v, lim in zip(q, lims, strict=True))
    _write(tmp_path, f"irb120_branches_{'native' if native else 'python'}", trace)


@pytest.mark.parametrize("native", BACKENDS)
def test_irb120_rate_limit_catches_up(native: bool, tmp_path: Path) -> None:
    arm = ssik.Manipulator.from_prebuilt("irb120")
    q_end = IRB_Q0.copy()
    q_end[0] += 0.5
    q_end[4] = -0.6  # through the wrist singularity again, now rate limited
    path = _lerp(IRB_Q0, q_end, 11)  # 0.05 rad per tick on joint 0
    dt, speed = 0.02, 1.0  # cap 0.02 rad per tick: the arm falls behind
    tracker = arm.tracker(IRB_Q0, max_joint_speed=speed, native=native, t0=0.0)
    stream = [("fast", arm.fk(q)) for q in path[1:]] + [("pause", arm.fk(q_end))] * 70
    steps = []
    trace = []
    for i, (segment, T) in enumerate(stream, start=1):
        st = tracker.update(T, t=dt * i)
        steps.append(st)
        trace.append(_record(st, segment))
    _write(tmp_path, f"irb120_rate_{'native' if native else 'python'}", trace)
    _check_contract(steps, jump=0.5, cap=speed)

    statuses = [s.status for s in steps]
    assert statuses[0] is ssik.TrackStatus.LIMITED
    # LIMITED until it has caught up, then OK for good, on the path's end.
    first_ok = statuses.index(ssik.TrackStatus.OK)
    assert all(s is ssik.TrackStatus.LIMITED for s in statuses[:first_ok])
    assert all(s is ssik.TrackStatus.OK for s in statuses[first_ok:])
    for st, (_segment, T) in zip(steps[:first_ok], stream, strict=False):
        # Honest residual: the arm is behind the target while LIMITED.
        assert st.fk_residual == float(np.linalg.norm(arm.fk(st.q) - T))
        assert st.fk_residual > 1e-3
        assert st.lag > 0.0
    np.testing.assert_allclose(steps[-1].q, q_end, atol=Q_TOL)
    # Caught up no sooner than the speed limit allows: joint 5 travels 1.2 rad
    # at 1 rad/s, arriving on update first_ok + 1.
    assert (first_ok + 1) * dt >= 1.2 / speed - 1e-9


# ---------------------------------------------------------------------------
# Redundant 7R: chart continuation, the same as solve_path
# ---------------------------------------------------------------------------

SEVEN_R = {
    # q0, q1 (a smooth in-limit stretch), and the joint-3 excursion past a limit.
    "panda": (
        np.array([0.1, -0.3, 0.2, -1.8, 0.3, 1.6, 0.4]),
        np.array([0.5, 0.2, -0.2, -1.2, -0.3, 2.0, 0.4]),
        0.3,  # joint 4's upper limit is -0.0698
    ),
    "iiwa14": (
        np.array([0.1, 0.5, 0.2, -1.2, 0.3, 0.9, 0.4]),
        np.array([0.5, 0.8, -0.2, -0.6, -0.3, 1.2, 0.2]),
        -2.4,  # joint 4's lower limit is -2.094
    ),
}


@pytest.mark.parametrize("native", BACKENDS)
@pytest.mark.parametrize("name", sorted(SEVEN_R))
def test_seven_r_matches_solve_path(name: str, native: bool, tmp_path: Path) -> None:
    arm = ssik.Manipulator.from_prebuilt(name)
    q0, q1, over = SEVEN_R[name]
    q_lim = q1.copy()
    q_lim[3] = over
    smooth = _lerp(q0, q1, 30)
    excursion = _lerp(q1, q_lim, 20)[1:] + _lerp(q_lim, q1, 20)[1:]
    poses = np.array([arm.fk(q) for q in smooth + excursion])
    limits = arm.joint_limits[3]
    assert limits is not None
    lo, hi = limits

    tracker = arm.tracker(q0, native=native)
    steps = [tracker.update(T) for T in poses]
    segments = ["smooth"] * len(smooth) + ["limit"] * len(excursion)
    _write(
        tmp_path,
        f"{name}_{'native' if native else 'python'}",
        [_record(s, seg) for s, seg in zip(steps, segments, strict=True)],
    )
    _check_contract(steps, jump=0.5, cap=None)

    # The in-limit stretch is what solve_path returns for the branch through
    # q0 (the redundancy coordinate held fixed), up to a 2*pi representative.
    path = arm.solve_path(poses[: len(smooth)], q0=q0, native=native)
    ref = path.q(next(iter(path.tracks)))
    for st, q_ref in zip(steps, ref, strict=False):
        assert st.status is ssik.TrackStatus.OK
        assert _wrapped(st.q - q_ref) <= Q_TOL

    # The excursion: tracked while joint 4's continuation is in limits, held
    # (never past the stop) while it is not, and the same branch again after.
    held = [s for s in steps[len(smooth) :] if s.status is ssik.TrackStatus.HELD]
    assert held
    assert all(s.reason == "limits" for s in held)
    for st in steps:
        assert lo <= st.q[3] <= hi
    assert steps[-1].status is ssik.TrackStatus.OK
    assert _wrapped(steps[-1].q - ref[-1]) <= Q_TOL


# ---------------------------------------------------------------------------
# Input validation: the solve() contract (docs/api.md, "Input validation")
# ---------------------------------------------------------------------------


def test_update_validates_like_solve() -> None:
    arm = ssik.Manipulator.from_prebuilt("irb120")
    tracker = arm.tracker(IRB_Q0)
    T = arm.fk(IRB_Q0)
    bad = T.copy()
    bad[0, 0] = 2.0  # not a rotation
    for target in (bad, np.eye(3), np.full((4, 4), np.nan)):
        # The same exception and message as solve(), naming the argument.
        try:
            arm.solve(target)
        except ValueError as e:
            expected = str(e).replace("T_target", "T")
        else:
            raise AssertionError("solve accepted a malformed target")
        with pytest.raises(ValueError, match=re.escape(expected)):
            tracker.update(target)
    with pytest.raises(TypeError):
        tracker.update(T.astype(complex))
    with pytest.raises(TypeError):
        tracker.update(T, t="now")  # type: ignore[arg-type]
    q_before = tracker.update(T, t=1.0).q
    with pytest.raises(ValueError, match="must not decrease"):
        tracker.update(T, t=0.5)
    with pytest.raises(ValueError, match="jump_threshold must be finite and > 0"):
        arm.tracker(IRB_Q0, jump_threshold=0.0)
    with pytest.raises(ValueError, match=r"max_joint_speed must be a number or have shape \(6,\)"):
        arm.tracker(IRB_Q0, max_joint_speed=np.ones(5))
    with pytest.raises(ValueError, match=r"q0 must have shape \(6,\)"):
        arm.tracker(np.zeros(5))
    # A rejected update leaves the tracker as it was.
    np.testing.assert_array_equal(tracker.q, q_before)
