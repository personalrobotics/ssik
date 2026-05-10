"""Per-IK timing for ikgeo.three_parallel on UR5 (warm cache).

Baseline measurement for #93 — the broader speed pass over non-tier-2-RR
solver pathways. Mirrors scripts/bench_real_jaco2.py methodology:

  uv run python scripts/bench_three_parallel.py

Reports min / median / mean / p95 / max IK time, solutions/pose distribution,
FK error stats, and failure count over 100 random non-singular poses.

UR5 fixture loaded from tests/fixtures/ur5.urdf (in-tree URDF). The
three-parallel solver covers UR3 / UR5 / UR10 and any other arm with the
same parallel-trio axis structure at joints (1, 2, 3).
"""

from __future__ import annotations

import cProfile
import functools
import pstats
import sys
import time

import numpy as np

print = functools.partial(print, flush=True)

from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from _flop_budget import flop_budget, print_flop_summary  # noqa: E402

from ssik._urdf import load_urdf_kinbody_normalized  # noqa: E402
from ssik.solvers.ikgeo import three_parallel  # noqa: E402


def _rot_axis(axis, angle):
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


def fk_poe(kb, q):
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _rot_axis(j.axis, float(qi)) @ j.T_right
    return T


FIXTURES = _REPO_ROOT / "tests" / "fixtures"
kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")

print("warming caches...")
rng = np.random.default_rng(0)
q_warm = rng.uniform(-1.0, 1.0, size=6)
T_warm = fk_poe(kb, q_warm)
sols, _ = three_parallel.solve(kb, T_warm)
print(f"warm-cache solve: {len(sols)} solutions")

print("\nwarm-cache: 100 random poses")
times: list[float] = []
n_sols: list[int] = []
fk_errs: list[float] = []
n_fail = 0
for _ in range(100):
    q_star = rng.uniform(-1.0, 1.0, size=6)
    t_target = fk_poe(kb, q_star)
    t = time.perf_counter()
    sols, is_ls = three_parallel.solve(kb, t_target)
    times.append((time.perf_counter() - t) * 1000)
    n_sols.append(len(sols))
    if is_ls:
        n_fail += 1
        continue
    best_err = min(float(np.linalg.norm(fk_poe(kb, s.q) - t_target)) for s in sols)
    fk_errs.append(best_err)

ts = np.array(times)
print("\nper-IK time over 100 poses:")
print(f"  min     {ts.min():>8.3f} ms")
print(f"  median  {np.median(ts):>8.3f} ms")
print(f"  mean    {ts.mean():>8.3f} ms")
print(f"  p95     {np.percentile(ts, 95):>8.3f} ms")
print(f"  max     {ts.max():>8.3f} ms")
print(
    f"\nsolutions per pose: median={int(np.median(n_sols))}, min={min(n_sols)}, max={max(n_sols)}"
)
print(f"FK error: median={np.median(fk_errs):.2e}, max={max(fk_errs):.2e}")
print(f"failures (is_ls=True): {n_fail}/100")

# FLOP-budget pass: cProfile a fresh sweep so call counts are deterministic;
# wall-clock under cProfile is not comparable to the timing run above.
rng_p = np.random.default_rng(1)
poses = [fk_poe(kb, rng_p.uniform(-1.0, 1.0, size=6)) for _ in range(100)]
pr = cProfile.Profile()
t0 = time.perf_counter()
pr.enable()
for tt in poses:
    three_parallel.solve(kb, tt)
pr.disable()
profile_s = time.perf_counter() - t0
total_flops, breakdown = flop_budget(pstats.Stats(pr))
# Wall-clock for the FLOP/s rate uses the unprofiled timing-run total to
# match what users would see; cProfile adds ~30-50% overhead.
unprofiled_total_s = float(ts.sum()) / 1000.0
print_flop_summary(total_flops, len(poses), unprofiled_total_s, breakdown)
