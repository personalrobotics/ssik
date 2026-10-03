"""Example 02: trajectory tracking -- one configuration per pose, no jumps.

A controller following a Cartesian path wants one joint configuration per pose,
each close to the last. The idiom is a seeded, capped solve::

    q = arm.solve(T, q_seed=q_prev, max_solutions=1)[0].q

This example follows a smooth UR5 path that passes exactly through a wrist
singularity (``q5 = 0``), where the arm has a continuum of solutions instead of
isolated ones, and shows:

1. The seeded loop, with continuity checks along the whole path.
2. The singular-continuum contract of ssik 8.0 (``docs/api.md``, "Singular
   continua"): a seed on the continuum gets itself back; an unseeded solve
   returns one documented representative per continuum.
3. ``seed_metric`` (how "closest" is measured) and ``seed_tolerance`` (a hard
   bound that returns ``[]`` rather than jump).
4. The same path through ``Manipulator.solve_path``, which follows every
   branch's chart label in one call and reports where branches meet.

For a live stream (a teleoperation device, one pose at a time) the online
counterpart of this loop is ``ssik.Tracker``; see ``06_teleop.py``.

Headless, wheel only::

    python examples/02_trajectory_tracking.py

The script checks its own claims at the end and exits non-zero if one fails.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import ssik

N_POSES = 81
SINGULAR = N_POSES // 2  # the path's wrist angle q5 crosses exactly 0 here


def wrapped(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-joint difference between configurations, on the circle."""
    return np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi


def main() -> int:
    checks: list[tuple[str, bool]] = []
    arm = ssik.Manipulator.from_prebuilt("ur5")

    # A smooth joint motion (smoothstep in time) whose wrist q5 goes from +0.6
    # to -0.6, passing through 0 at the middle sample. Its FK is the Cartesian
    # path we track: smooth in position and orientation, singular in the middle.
    q_a = np.array([0.3, -1.2, 1.5, -1.0, 0.6, 0.4])
    q_b = np.array([0.9, -0.8, 1.1, -1.6, -0.6, 1.2])
    s = np.linspace(0.0, 1.0, N_POSES)
    path_q = q_a + np.outer(3 * s**2 - 2 * s**3, q_b - q_a)
    poses = np.stack([arm.fk(q) for q in path_q])
    path_step = float(np.abs(np.diff(path_q, axis=0)).max())
    print(f"UR5 path: {N_POSES} poses, wrist q5 = {path_q[SINGULAR, 4]:.1f} at pose {SINGULAR}")
    print(f"  largest joint step of the motion that drew it: {path_step:.4f} rad\n")

    # -----------------------------------------------------------------------
    # 1. The seeded loop.
    # -----------------------------------------------------------------------
    for _ in range(20):  # warm-up: one-time import and setup costs
        arm.solve(poses[1], q_seed=path_q[0], max_solutions=1)
    q = path_q[0]
    track = [q]
    residuals = []
    times = []
    for T in poses[1:]:
        t0 = time.perf_counter()
        (sol,) = arm.solve(T, q_seed=q, max_solutions=1)
        times.append(time.perf_counter() - t0)
        q = sol.q
        track.append(q)
        residuals.append(sol.fk_residual)
    track_q = np.array(track)
    step = float(np.abs(np.diff(track_q, axis=0)).max())
    dev = np.abs(track_q - path_q).max(axis=1)
    print("seeded loop: solve(T, q_seed=q_prev, max_solutions=1)")
    print(f"  median solve time        {np.median(times) * 1e6:.0f} us")
    print(f"  worst FK residual        {max(residuals):.1e}")
    print(f"  largest joint step       {step:.4f} rad")
    off = np.flatnonzero(dev > 1e-9).tolist()
    print(f"  poses off the drawn path {off}  (by {dev.max():.4f} rad)")
    print(f"  q at the singular pose   {(np.round(track_q[SINGULAR], 4) + 0.0).tolist()}")
    print(f"  q the path drew there    {np.round(path_q[SINGULAR], 4).tolist()}")
    seed = track_q[SINGULAR - 1]
    to_track = float(np.linalg.norm(wrapped(track_q[SINGULAR], seed)))
    to_drawn = float(np.linalg.norm(wrapped(path_q[SINGULAR], seed)))
    print(f"  distance from the seed   {to_track:.4f} (returned) vs {to_drawn:.4f} (drawn)")
    checks.append(("every tracked pose closes FK below 1e-9", max(residuals) < 1e-9))
    checks.append(("no joint step larger than 1.5x the path's", step < 1.5 * path_step))
    checks.append(("the track leaves the drawn path only at the singular pose", off == [SINGULAR]))
    checks.append(
        ("there it is no farther from the seed than the drawn q", to_track <= to_drawn + 1e-12)
    )
    print(
        "  At q5 = 0 only a combination of the wrist and arm joints is fixed, so the"
        "\n  drawn q is one point of a continuum. The seeded solve returns the point of"
        "\n  that continuum nearest the seed (the previous pose's q), and the next pose,"
        "\n  regular again, is back on the drawn path.\n"
    )

    # -----------------------------------------------------------------------
    # 2. The continuum contract, directly.
    # -----------------------------------------------------------------------
    q_sing = path_q[SINGULAR]
    T_sing = poses[SINGULAR]
    (own,) = arm.solve(T_sing, q_seed=q_sing, max_solutions=1)
    own_err = float(np.abs(own.q - q_sing).max())
    print("at the singular pose:")
    print(f"  seeded with its own q    returns it back, off by {own_err:.1e} rad")
    reps = arm.solve(T_sing, enumerate_windings=False)
    print(f"  unseeded                 {len(reps)} solutions; those on the q5 = 0 continuum:")
    on_lock = [r for r in reps if abs(np.sin(r.q[4])) < 1e-9]
    for r in on_lock:
        print(f"    {np.round(r.q, 3).tolist()}  (free joint q6 = {r.q[5]:.1f})")
    checks.append(("a seed on the continuum is returned as itself", own_err < 1e-9))
    checks.append(
        (
            "unseeded continuum representatives have q6 = 0",
            bool(on_lock) and all(abs(r.q[5]) < 1e-9 for r in on_lock),
        )
    )
    print()

    # -----------------------------------------------------------------------
    # 3. seed_metric and seed_tolerance.
    # -----------------------------------------------------------------------
    print("seed_metric ranks solutions by distance to the seed: 'wrap_linf' (default)")
    print("by the largest single-joint move, 'wrap_l2' by the Euclidean norm of the move.")
    print("Top three at the singular pose, seeded from the pose before it, as")
    print("(largest move, norm) in rad:")
    for metric in ("wrap_linf", "wrap_l2"):
        ranked = arm.solve(T_sing, q_seed=seed, max_solutions=3, seed_metric=metric)
        moves = [wrapped(r.q, seed) for r in ranked]
        pairs = [
            (round(float(np.abs(m).max()), 3), round(float(np.linalg.norm(m)), 3)) for m in moves
        ]
        print(f"  {metric:9s} {pairs}")

    k = 20
    near = arm.solve(poses[k + 1], q_seed=path_q[k], max_solutions=1, seed_tolerance=0.05)
    far = arm.solve(poses[k + 40], q_seed=path_q[k], max_solutions=1, seed_tolerance=0.05)
    print("seed_tolerance=0.05 (no joint may move more than 0.05 rad):")
    print(f"  next pose on the path      {len(near)} solution")
    print(f"  a pose 40 samples ahead    {far}  (a jump: refused, not taken)")
    checks.append(("seed_tolerance admits the next pose", len(near) == 1))
    checks.append(("seed_tolerance refuses a jump", far == []))
    print()

    # -----------------------------------------------------------------------
    # 4. The whole path in one call.
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    result = arm.solve_path(poses, q0=path_q[0])
    ms = (time.perf_counter() - t0) * 1e3
    (label,) = result.tracks
    chart_q = result.q(label)
    chart_fk = max(
        float(np.linalg.norm(arm.fk(q) - T)) for q, T in zip(chart_q, poses, strict=True)
    )
    differs = np.flatnonzero(np.abs(wrapped(chart_q, track_q)).max(axis=1) > 1e-9).tolist()
    events = sorted({e.index for e in result.events()})
    print(f"solve_path(poses, q0=...): {result} in {ms:.1f} ms")
    print(f"  worst FK residual         {chart_fk:.1e}")
    print(f"  events (where branches meet) at poses {events}")
    print(f"  differs from the seeded loop (on the circle) at poses {differs}")
    print(f"  q at the singular pose    {(np.round(chart_q[SINGULAR], 4) + 0.0).tolist()}")
    print(
        "  solve_path follows chart labels, not a seed. Where two branches meet it"
        "\n  reports an event, and its point on the continuum is the unseeded"
        "\n  representative (q6 = 0). Its angles are chart coordinates: compare them"
        "\n  on the circle, as here. For one configuration close to the last, use the"
        "\n  seeded loop above."
    )
    checks.append(("solve_path closes FK below 1e-9 everywhere", chart_fk < 1e-9))
    checks.append(("solve_path reports an event at the singular pose", SINGULAR in events))
    checks.append(
        ("solve_path agrees with the seeded loop away from events", set(differs) <= set(events))
    )
    print()

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
