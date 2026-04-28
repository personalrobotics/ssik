"""Shared bench harness: timing + FLOP-budget pass for one (kb, solver) pair.

Each per-solver bench script wires up its KinBody fixture and a solver
callable, then delegates the per-IK loop to :func:`run_bench` here so the
output format is uniform across the whole #93 priority list.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from _flop_budget import flop_budget, print_flop_summary
from numpy.typing import NDArray


def _rodrigues(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    c, s = np.cos(t), np.sin(t)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R: NDArray[np.float64] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R


def _aa4(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = _rodrigues(k, t)
    return T


def fk_poe(kb: Any, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """POE forward kinematics for any KinBody. Used as ground truth in the
    bench timing + FK-error reporting."""
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _aa4(j.axis, float(qi)) @ j.T_right
    return T


def run_bench(
    *,
    solver_label: str,
    solver_call: Callable[[Any, NDArray[np.float64]], tuple[Any, bool]],
    kb: Any,
    n_dof: int,
    n_poses: int = 100,
    seed: int = 0,
    profile_seed: int = 1,
) -> None:
    """Warm + time + FLOP-budget one solver+kb pair.

    :param solver_label: human-readable name (printed in the header).
    :param solver_call: callable ``(kb, T) -> (solutions, is_ls)``. Wrap your
        solver to apply any non-default kwargs.
    :param kb: a POE-normalised :class:`KinBody`.
    :param n_dof: number of joints (used for the random q sampler).
    :param n_poses: number of random poses to time.
    :param seed: RNG seed for the timing sweep (deterministic across runs).
    :param profile_seed: RNG seed for the cProfile pass (different from
        ``seed`` so the FLOP budget reflects fresh poses, not warm-cached).
    """
    print(f"=== {solver_label} ===")
    print("warming caches...")
    rng = np.random.default_rng(seed)
    q_warm = rng.uniform(-1.0, 1.0, size=n_dof)
    sols, _ = solver_call(kb, fk_poe(kb, q_warm))
    print(f"warm-cache solve: {len(sols)} solutions\n")

    print(f"warm-cache: {n_poses} random poses\n")
    times: list[float] = []
    n_sols: list[int] = []
    fk_errs: list[float] = []
    fails = 0
    for _ in range(n_poses):
        q_star = rng.uniform(-1.0, 1.0, size=n_dof)
        t_target = fk_poe(kb, q_star)
        t0 = time.perf_counter()
        sols, is_ls = solver_call(kb, t_target)
        times.append((time.perf_counter() - t0) * 1e3)
        if is_ls or not sols:
            fails += 1
            continue
        n_sols.append(len(sols))
        best_err = min(float(np.linalg.norm(fk_poe(kb, s.q) - t_target)) for s in sols)
        fk_errs.append(best_err)

    ts = np.array(times)
    print("per-IK time over poses:")
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
    if fk_errs:
        print(f"FK error: median={np.median(fk_errs):.2e}, max={max(fk_errs):.2e}")
    print(f"failures (is_ls=True): {fails}/{n_poses}")

    rng_p = np.random.default_rng(profile_seed)
    poses = [fk_poe(kb, rng_p.uniform(-1.0, 1.0, size=n_dof)) for _ in range(n_poses)]
    pr = cProfile.Profile()
    pr.enable()
    for tt in poses:
        solver_call(kb, tt)
    pr.disable()
    total_flops, breakdown = flop_budget(pstats.Stats(pr))
    unprofiled_total_s = float(ts.sum()) / 1000.0
    print_flop_summary(total_flops, n_poses, unprofiled_total_s, breakdown)
