"""Per-IK timing for ssik.solvers.seven_r.srs on KUKA iiwa LBR 14.

Mirrors scripts/bench_seven_r.py + bench_jaco2.py. Uses the real iiwa14
fixture from tests/fixtures/. Reports wall-clock (full sweep,
default 16 swivel samples x 8 branches) + FLOP budget + per-function
breakdown over 100 random reachable poses.

Run::

    uv run python scripts/bench_srs_iiwa14.py
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "fixtures"))

from _flop_budget import flop_budget, print_flop_summary
from kuka_iiwa14 import kuka_iiwa14_specs

from ssik._kinbody import build_kinbody
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.seven_r import srs

kb = build_kinbody(kuka_iiwa14_specs())

print("warming caches...")
rng = np.random.default_rng(0)
q_warm = rng.uniform(-0.8, 0.8, size=7)
sols, _ = srs.solve(kb, poe_forward_kinematics(kb, q_warm))
print(f"warm-cache solve: {len(sols)} solutions\n")

n = 100
print(f"warm-cache: {n} random poses (q ∈ [-0.8, 0.8] per joint)\n")

times: list[float] = []
n_sols: list[int] = []
fk_errs: list[float] = []
fails = 0
for _ in range(n):
    q_star = rng.uniform(-0.8, 0.8, size=7)
    t_target = poe_forward_kinematics(kb, q_star)
    t0 = time.perf_counter()
    sols, is_ls = srs.solve(kb, t_target)
    dt = time.perf_counter() - t0
    times.append(dt * 1e3)
    if is_ls or not sols:
        fails += 1
        continue
    n_sols.append(len(sols))
    best_err = min(float(np.linalg.norm(poe_forward_kinematics(kb, s.q) - t_target)) for s in sols)
    fk_errs.append(best_err)

ts = np.array(times)
print("per-IK time over 100 poses (full swivel sweep, 16 samples x 8 branches):")
print(f"  min      {ts.min():>8.3f} ms")
print(f"  median   {np.median(ts):>8.3f} ms")
print(f"  mean     {ts.mean():>8.3f} ms")
print(f"  p95      {np.percentile(ts, 95):>8.3f} ms")
print(f"  max      {ts.max():>8.3f} ms")

if n_sols:
    print(
        f"\nsolutions per pose: median={int(np.median(n_sols))}, "
        f"min={min(n_sols)}, max={max(n_sols)}"
    )
    print(f"FK error (best per pose): median={np.median(fk_errs):.2e}, max={max(fk_errs):.2e}")
print(f"failures (is_ls=True): {fails}/{n}")

# FLOP-budget pass.
rng_p = np.random.default_rng(1)
poses = [poe_forward_kinematics(kb, rng_p.uniform(-0.8, 0.8, size=7)) for _ in range(n)]
pr = cProfile.Profile()
pr.enable()
for tt in poses:
    srs.solve(kb, tt)
pr.disable()
total_flops, breakdown = flop_budget(pstats.Stats(pr))
unprofiled_total_s = float(ts.sum()) / 1000.0
print_flop_summary(total_flops, n, unprofiled_total_s, breakdown)

# Per-function cumulative time breakdown.
print("\n=== Per-function cumulative time (top 25) ===")
ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats(25)
