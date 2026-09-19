"""Example 01: UR5 quickstart — the basic ssik API tour.

Universal Robots UR5 is a 6R Pieper-class arm with three consecutive
parallel axes (joints 1, 2, 3). ``ssik`` dispatches it to
``ikgeo.three_parallel`` — IK in ~1.6 ms returning all 8 branches at
machine-precision FK closure.

This script demonstrates:

1. Loading via ``Manipulator.from_urdf``.
2. Forward kinematics (``arm.fk``).
3. Inverse kinematics, full enumeration (``arm.solve``).
4. Trajectory-tracking with ``max_solutions=1`` and ``q_seed``.
5. Inspecting the dispatch plan.

Run::

    uv run python examples/01_ur5_quickstart.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import ssik

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF = REPO_ROOT / "tests" / "fixtures" / "ur5.urdf"


def main() -> None:
    # ---------------------------------------------------------------------
    # 1. Load.
    # ---------------------------------------------------------------------
    arm = ssik.Manipulator.from_urdf(URDF, base="base_link", ee="ee_link")
    print(arm)
    print(f"  dof:        {arm.dof}")
    print(f"  solver:     {arm.solver_name}")
    print(f"  expected:   {arm.dispatch_plan.expected_ms_median:.1f} ms median")
    print()

    # ---------------------------------------------------------------------
    # 2. Forward kinematics.
    # ---------------------------------------------------------------------
    q_star = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    T = arm.fk(q_star)
    print(f"FK at q={q_star.tolist()}:")
    print(f"  position:   {T[:3, 3].tolist()}")
    print()

    # ---------------------------------------------------------------------
    # 3. Inverse kinematics — full redundancy enumeration.
    # ---------------------------------------------------------------------
    t = time.perf_counter()
    sols = arm.solve(T, enumerate_windings=False)
    elapsed_ms = (time.perf_counter() - t) * 1000
    print(f"IK (geometric branches): {len(sols)} solutions in {elapsed_ms:.2f} ms")
    print(f"  max FK residual: {max(s.fk_residual for s in sols):.2e}")
    print("  branches:")
    for i, s in enumerate(sols):
        print(f"    [{i}] q = {[round(x, 3) for x in s.q]}")
    print()

    # ---------------------------------------------------------------------
    # 3b. Winding representatives (#562). Five of the UR5's joints have limits
    #     spanning more than a full turn, so each branch above is reachable at
    #     several different joint coordinates -- same pose, different
    #     configuration. solve() returns all of them by default.
    # ---------------------------------------------------------------------
    t = time.perf_counter()
    lifted = arm.solve(T)
    elapsed_ms = (time.perf_counter() - t) * 1000
    print(f"IK (default, with winding lifts): {len(lifted)} solutions in {elapsed_ms:.2f} ms")
    print(f"  {len(sols)} geometric branches x 2^5 wide joints = {len(lifted)}")
    print("  the same branch at three of its in-limit coordinates:")
    branch0 = [s for s in lifted if np.allclose(np.mod(s.q - sols[0].q, 2 * np.pi), 0, atol=1e-6)]
    for s in branch0[:3]:
        print(f"    q = {[round(x, 3) for x in s.q]}")
    print("  (identical FK; they differ in how far each joint has turned)")
    print()

    # ---------------------------------------------------------------------
    # 4. Trajectory tracking — give me ONE IK closest to a previous q.
    # ---------------------------------------------------------------------
    q_prev = q_star + 0.05 * np.random.default_rng(42).standard_normal(6)
    t = time.perf_counter()
    sols = arm.solve(T, max_solutions=1, q_seed=q_prev)
    elapsed_ms = (time.perf_counter() - t) * 1000
    print(f"IK (trajectory tracking, max_solutions=1): {elapsed_ms:.2f} ms")
    print(f"  q_prev:     {[round(x, 3) for x in q_prev]}")
    print(f"  q_chosen:   {[round(x, 3) for x in sols[0].q]}")
    print()

    # ---------------------------------------------------------------------
    # 5. Why this dispatch?
    # ---------------------------------------------------------------------
    print("Dispatch reason:")
    for line in arm.dispatch_plan.reason.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    main()
