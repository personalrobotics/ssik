"""Example 01: quickstart -- load an arm, solve it, and read the answers.

A tour of the calls most users need, on the Universal Robots UR5 that ships
with ssik:

1. ``ssik.list_arms()`` and ``Manipulator.from_prebuilt``.
2. ``fk`` and ``solve``: every IK branch, each with its FK residual.
3. Joint limits: ``respect_limits=True / False / "wrap"``, winding
   representatives and ``enumerate_windings=False``.
4. Input validation: what ``solve`` rejects, and ``max_solutions=0``.
5. ``explain=True``: why a result is empty.

Needs only the wheel (no extras, no source checkout)::

    pip install ssik
    python examples/01_quickstart.py

The script checks its own claims at the end and exits non-zero if one fails.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import ssik


def wrapped(a: np.ndarray, b: np.ndarray) -> float:
    """Largest per-joint difference between two configurations, on the circle."""
    d = np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi
    return float(np.max(np.abs(d)))


def main() -> int:
    checks: list[tuple[str, bool]] = []

    # -----------------------------------------------------------------------
    # 1. Pick an arm. list_arms() is metadata only: it imports no solver.
    # -----------------------------------------------------------------------
    arms = ssik.list_arms()
    ur = [a.name for a in arms if a.vendor == "universal_robots"]
    print(f"{len(arms)} prebuilt arms; Universal Robots: {', '.join(ur)}")

    arm = ssik.Manipulator.from_prebuilt("ur5")
    print(arm)
    print(f"  dof {arm.dof}, solver {arm.solver_name}")
    limits = [None if lim is None else np.round(lim, 3).tolist() for lim in arm.joint_limits]
    print(f"  joint limits: {limits}\n")

    # -----------------------------------------------------------------------
    # 2. FK, then IK back to it.
    # -----------------------------------------------------------------------
    q_star = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    T = arm.fk(q_star)
    print(f"fk({q_star.tolist()}) position: {np.round(T[:3, 3], 4).tolist()}")

    arm.solve(T)  # warm-up: the first call pays one-time import and setup costs
    t0 = time.perf_counter()
    branches = arm.solve(T, enumerate_windings=False)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"solve(T, enumerate_windings=False): {len(branches)} branches in {ms:.3f} ms")
    for i, s in enumerate(branches):
        print(f"  [{i}] q = {np.round(s.q, 3).tolist()}  fk_residual {s.fk_residual:.1e}")
    worst_fk = max(s.fk_residual for s in branches)
    checks.append(("8 geometric branches", len(branches) == 8))
    checks.append(("every branch closes FK below 1e-9", worst_fk < 1e-9))
    checks.append(("q* is among the branches", min(wrapped(s.q, q_star) for s in branches) < 1e-9))
    print()

    # -----------------------------------------------------------------------
    # 3. Joint limits and windings. Five UR5 joints span [-2*pi, 2*pi], so each
    #    branch is reachable at several joint coordinates: the same pose, but a
    #    different configuration of the robot (how far each joint has turned).
    #    solve() returns every in-limit coordinate by default.
    # -----------------------------------------------------------------------
    lifted = arm.solve(T)
    raw = arm.solve(T, respect_limits=False)
    wrap = arm.solve(T, respect_limits="wrap", enumerate_windings=False)
    print("solutions returned, by option:")
    print(f"  {len(lifted):3d}  default: 8 branches x 2^5 in-limit windings")
    print(f"  {len(branches):3d}  enumerate_windings=False: one coordinate per branch")
    print(f"  {len(raw):3d}  respect_limits=False: the solver's own coordinates, nothing dropped")
    print(f"  {len(wrap):3d}  respect_limits='wrap': wrapped into range, nothing dropped")
    checks.append(("256 winding representatives", len(lifted) == 256))

    # respect_limits=False reports each finite joint as the solver computed it,
    # which can be outside its limits by a full turn. The elbow (joint 3) is
    # limited to [-pi, pi]; compare its angles both ways.
    elbow_raw = np.round([s.q[2] for s in raw], 3).tolist()
    elbow_default = np.round([s.q[2] for s in branches], 3).tolist()
    print(f"  elbow angles, respect_limits=False: {elbow_raw}")
    print(f"  elbow angles, default:              {elbow_default}")
    print("  (same configurations; the default reports the in-limit representative,")
    print("   and a joint without limits is always reported in (-pi, pi])")
    checks.append(
        ("default results lie within the limits", all(_in_limits(arm, s.q) for s in lifted))
    )

    # A seed picks one of those coordinates: the nearest, joint by joint. A finite
    # joint is an interval, not a circle, so the seed's turn is what decides.
    q_seed = q_star.copy()
    q_seed[0] -= 2 * np.pi  # the base turned one full revolution the other way
    (nearest,) = arm.solve(T, q_seed=q_seed, max_solutions=1)
    print(f"  seed with joint 1 one turn back: {np.round(nearest.q, 3).tolist()}")
    checks.append(
        ("a seed one turn away gets that turn's representative", np.allclose(nearest.q, q_seed))
    )
    print()

    # -----------------------------------------------------------------------
    # 4. Input validation (docs/api.md, "Input validation"). A malformed call
    #    raises before any solver runs, identically on the native and pure-Python
    #    backends. It is never silently coerced or answered with [].
    # -----------------------------------------------------------------------
    print("input validation:")
    for label, bad in (
        ("3x4 target", T[:3]),
        ("scaled rotation", _scaled(T, 1.1)),
        ("NaN in the target", _with_nan(T)),
    ):
        try:
            arm.solve(bad)
        except ValueError as exc:
            print(f"  {label:18s} ValueError: {exc}")
            checks.append((f"{label} raises ValueError", True))
        else:
            print(f"  {label:18s} accepted (unexpected)")
            checks.append((f"{label} raises ValueError", False))
    capped = arm.solve(T, max_solutions=0)
    print(f"  max_solutions=0    {capped}  (a zero budget is valid, and solves nothing)")
    checks.append(("max_solutions=0 returns []", capped == []))
    print()

    # -----------------------------------------------------------------------
    # 5. explain=True. An empty list means "no certified solution", not a proof
    #    of unreachability; the Diagnostic says where the candidates went.
    # -----------------------------------------------------------------------
    far = T.copy()
    far[:3, 3] = [2.0, 0.0, 0.5]  # the UR5 reaches about 0.85 m
    sols, diag = arm.solve(far, explain=True)
    print(f"a target 2 m out: {len(sols)} solutions")
    print(f"  raw_candidates     {diag.raw_candidates}  (the solver found nothing to filter)")
    print(f"  dropped_by_limits  {diag.dropped_by_limits}")
    print(f"  dispatch_reason    {diag.dispatch_reason.splitlines()[0]}")
    checks.append(
        ("out-of-reach target: [] with no raw candidates", not sols and diag.raw_candidates == 0)
    )
    print()

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    return 1 if failed else 0


def _in_limits(arm: ssik.Manipulator, q: np.ndarray) -> bool:
    return all(
        lim is None or lim[0] <= qi <= lim[1] for qi, lim in zip(q, arm.joint_limits, strict=True)
    )


def _scaled(T: np.ndarray, k: float) -> np.ndarray:
    out = T.copy()
    out[:3, :3] *= k
    return out


def _with_nan(T: np.ndarray) -> np.ndarray:
    out = T.copy()
    out[0, 3] = np.nan
    return out


if __name__ == "__main__":
    sys.exit(main())
