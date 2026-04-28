"""Per-IK timing for jointlock.seven_r on a synthetic SRS-with-spherical-wrist arm.

Mirrors scripts/bench_three_parallel.py + bench_spherical_two_parallel.py.
Uses the same synthetic 7R fixture as test_jointlock_seven_r.py
(``_build_srs_with_spherical_wrist``): shoulder pitch+pitch (parallel),
elbow roll (lock candidate), spherical wrist. Locking joint 3 yields a
6R with spherical_two_parallel topology.

Reports wall-clock + FLOP budget + transferability projections.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).parent))

from _flop_budget import flop_budget, print_flop_summary

from ssik._kinbody import Joint, KinBody, Link
from ssik.solvers.jointlock import seven_r


def _rodrigues(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    c, s = np.cos(t), np.sin(t)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R: NDArray[np.float64] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R


def _axis_angle_4x4(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = _rodrigues(k, t)
    return T


def _fk(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _axis_angle_4x4(j.axis, float(qi)) @ j.T_right
    return T


def _build_srs() -> KinBody:
    axes = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, -1.0, 0.0]),
    ]
    t_lefts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.2]),
        np.array([0.4, 0.0, 0.0]),
        np.array([0.05, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.4]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ]
    links = [Link(name=f"l{i}") for i in range(8)]
    joints = []
    for i in range(7):
        T_l = np.eye(4)
        T_l[:3, 3] = t_lefts[i]
        joints.append(
            Joint(
                name=f"j{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_l,
                T_right=np.eye(4),
                axis=axes[i],
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


kb = _build_srs()

print("warming caches...")
rng = np.random.default_rng(0)
q_warm = rng.uniform(-1.0, 1.0, size=7)
sols, _ = seven_r.solve(kb, _fk(kb, q_warm))
print(f"warm-cache solve: {len(sols)} solutions\n")

n = 100
print(f"warm-cache: {n} random poses\n")

times: list[float] = []
n_sols: list[int] = []
fk_errs: list[float] = []
fails = 0
for _ in range(n):
    q_star = rng.uniform(-1.0, 1.0, size=7)
    t_target = _fk(kb, q_star)
    t0 = time.perf_counter()
    sols, is_ls = seven_r.solve(kb, t_target)
    dt = time.perf_counter() - t0
    times.append(dt * 1e3)
    if is_ls or not sols:
        fails += 1
        continue
    n_sols.append(len(sols))
    best_err = min(float(np.linalg.norm(_fk(kb, s.q) - t_target)) for s in sols)
    fk_errs.append(best_err)

ts = np.array(times)
print("per-IK time over 100 poses:")
print(f"  min      {ts.min():>8.3f} ms")
print(f"  median   {np.median(ts):>8.3f} ms")
print(f"  mean     {ts.mean():>8.3f} ms")
print(f"  p95      {np.percentile(ts, 95):>8.3f} ms")
print(f"  max      {ts.max():>8.3f} ms")

print(
    f"\nsolutions per pose: median={int(np.median(n_sols))}, min={min(n_sols)}, max={max(n_sols)}"
)
print(f"FK error: median={np.median(fk_errs):.2e}, max={max(fk_errs):.2e}")
print(f"failures (is_ls=True): {fails}/{n}")

# FLOP-budget pass.
rng_p = np.random.default_rng(1)
poses = [_fk(kb, rng_p.uniform(-1.0, 1.0, size=7)) for _ in range(n)]
pr = cProfile.Profile()
pr.enable()
for tt in poses:
    seven_r.solve(kb, tt)
pr.disable()
total_flops, breakdown = flop_budget(pstats.Stats(pr))
unprofiled_total_s = float(ts.sum()) / 1000.0
print_flop_summary(total_flops, n, unprofiled_total_s, breakdown)
