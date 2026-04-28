"""Warm-cache per-IK bench for ssik.solvers.ikgeo.spherical_two_parallel on Puma 560.

Mirrors scripts/bench_three_parallel.py / bench_real_jaco2.py methodology.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _flop_budget import flop_budget, print_flop_summary

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.solvers.ikgeo import spherical_two_parallel


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s, 0],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s, 0],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc, 0],
            [0, 0, 0, 1],
        ]
    )


def fk_poe(kb, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _rot_axis(j.axis, float(qi)) @ j.T_right
    return T


FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
kb = load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")

print("warming caches...")
rng = np.random.default_rng(0)
q_warm = rng.uniform(-1.0, 1.0, size=6)
sols, _ = spherical_two_parallel.solve(kb, fk_poe(kb, q_warm))
print(f"warm-cache solve: {len(sols)} solutions\n")

n = 100
print(f"warm-cache: {n} random poses\n")

times = []
n_sols = []
fk_errs = []
fails = 0
for _ in range(n):
    q_star = rng.uniform(-1.0, 1.0, size=6)
    t_target = fk_poe(kb, q_star)
    t0 = time.perf_counter()
    sols, is_ls = spherical_two_parallel.solve(kb, t_target)
    dt = time.perf_counter() - t0
    times.append(dt)
    if is_ls or not sols:
        fails += 1
        continue
    n_sols.append(len(sols))
    for sol in sols:
        T_fk = fk_poe(kb, sol.q)
        fk_errs.append(float(np.linalg.norm(T_fk - t_target)))

times = np.array(times) * 1e3
print("per-IK time over 100 poses:")
print(f"  min      {times.min():7.3f} ms")
print(f"  median   {np.median(times):7.3f} ms")
print(f"  mean     {times.mean():7.3f} ms")
print(f"  p95      {np.percentile(times, 95):7.3f} ms")
print(f"  max      {times.max():7.3f} ms")

print(
    f"\nsolutions per pose: median={int(np.median(n_sols))}, min={min(n_sols)}, max={max(n_sols)}"
)
print(f"FK error: median={np.median(fk_errs):.2e}, max={max(fk_errs):.2e}")
print(f"failures (is_ls=True): {fails}/{n}")

# FLOP-budget pass: cProfile a fresh sweep for deterministic call counts;
# wall-clock under cProfile is not comparable to the unprofiled timing run.
rng_p = np.random.default_rng(1)
poses = [fk_poe(kb, rng_p.uniform(-1.0, 1.0, size=6)) for _ in range(n)]
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
for tt in poses:
    spherical_two_parallel.solve(kb, tt)
pr.disable()
total_flops, breakdown = flop_budget(pstats.Stats(pr))
unprofiled_total_s = float(times.sum()) / 1000.0
print_flop_summary(total_flops, n, unprofiled_total_s, breakdown)
