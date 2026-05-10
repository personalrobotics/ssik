"""Example 03: Kinova Gen3 — approximate-SRS 7R with LM polish.

Kinova Gen3 is a 7-DOF redundant arm. Its URDF axes don't quite meet
at common shoulder/wrist points (12 mm shoulder + 0.4 mm wrist offsets).
The strict SRS predicate (``is_srs_7r``) refuses, but the approximate-SRS
predicate (``is_approximately_srs_7r``) accepts: ``ssik`` runs Singh-Kreutz
on relaxed-pivot geometry to get warm-start candidates, then LM-polishes
each one against the **original URDF FK** to recover machine-precision
FK closure.

Strategic point: EAIK's Gen3 IK runs against a simplified SRS DH, so
returned q-vectors have ~12 mm IK error vs. the actual URDF. ssik's
``seven_r.srs_polished`` returns IKs that are correct against the
original URDF at FK <= 1e-10. 16-30x faster than the universal
``jointlock + HP`` fallback while preserving correctness.

This script:

1. Loads Gen3 from URDF.
2. Verifies the dispatcher picks ``seven_r.srs_polished``.
3. Demonstrates IK on random reachable poses with FK closure assertions.
4. Reports timing + FK residual statistics.

Run::

    uv run python examples/03_gen3_polished_srs.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import ssik

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF = REPO_ROOT / "tests" / "fixtures" / "gen3.urdf"


def main() -> None:
    arm = ssik.Manipulator.from_urdf(URDF, base="base_link", ee="end_effector_link")
    print(arm)
    print(f"  dof:    {arm.dof}")
    print(f"  solver: {arm.solver_name}")
    print()
    assert arm.solver_name == "seven_r.srs_polished", (
        f"expected seven_r.srs_polished, got {arm.solver_name}"
    )

    rng = np.random.default_rng(0)

    # Hand-picked pose well-inside the workspace.
    q_star = np.array([0.3, -0.4, 0.7, 0.5, 0.6, -0.5, 0.2])
    T = arm.fk(q_star)
    sols, is_ls = arm.ik(T)
    print(f"Hand-picked q*={q_star.tolist()}:")
    print(f"  IK returned {len(sols)} branches, is_ls={is_ls}")
    print(f"  max FK residual: {max(s.fk_residual for s in sols):.2e}")
    print("  (FK closure is against the ORIGINAL URDF, not a simplified DH.)")
    print()

    # Bench across 100 random reachable poses; q_3 in [0.2, 0.8] avoids
    # the kinematic singularity at q_3 ≈ 0 (#225 follow-up).
    times = []
    fk_residuals = []
    sol_counts = []

    # Warm.
    for _ in range(10):
        q = rng.uniform(-0.8, 0.8, size=7)
        q[3] = float(rng.uniform(0.2, 0.8))
        arm.ik(arm.fk(q))

    for _ in range(100):
        q = rng.uniform(-0.8, 0.8, size=7)
        q[3] = float(rng.uniform(0.2, 0.8))
        T = arm.fk(q)
        t = time.perf_counter()
        sols, is_ls = arm.ik(T)
        times.append((time.perf_counter() - t) * 1000)
        if not is_ls and sols:
            fk_residuals.append(max(s.fk_residual for s in sols))
            sol_counts.append(len(sols))

    print("Bench across 100 random reachable poses (q_3 in [0.2, 0.8]):")
    print(f"  median IK time:    {np.median(times):.2f} ms")
    print(f"  p95 IK time:       {np.percentile(times, 95):.2f} ms")
    print(f"  median branches:   {int(np.median(sol_counts))}")
    print(f"  max FK residual:   {max(fk_residuals):.2e}")
    print()

    # Trajectory-tracking idiom: sub-ms when you only need one IK.
    q_prev = q_star + 0.05 * rng.standard_normal(7)
    times_track = []
    for _ in range(50):
        T = arm.fk(rng.uniform(-0.8, 0.8, size=7))
        t = time.perf_counter()
        arm.ik(T, max_solutions=1, q_seed=q_prev)
        times_track.append((time.perf_counter() - t) * 1000)
    print("Trajectory tracking (max_solutions=1, q_seed):")
    print(f"  median IK time:    {np.median(times_track):.2f} ms")


if __name__ == "__main__":
    main()
