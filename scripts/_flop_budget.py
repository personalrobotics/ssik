"""Deterministic FLOP-budget estimator for ssik bench scripts.

Wall-clock is system-load-dependent and platform-specific. FLOP count is not:
the same workload on the same input issues the same arithmetic ops on every
machine. So we report both:

- **wall-clock**: what this run took on this machine (here, now)
- **FLOP budget**: machine-invariant work performed
- **achieved rate**: FLOP budget / wall-clock = effective FLOP/s on this CPU
- **reference projection**: FLOP budget / target rate = predicted time on a
  machine with that effective rate (assumes similar Python+numpy overhead)

The FLOP budget is a *lower bound* on arithmetic — it counts operations on
dominant shapes only. Counts come from cProfile's per-function call counts,
which are deterministic for a fixed seed.

Per-call FLOP estimates use textbook formulas for the algorithm at our
workload's typical shapes (2x2 / 3x3 / 4x4 matrices, length-3 vectors).
Numbers like ``solve(NxN)`` use ``(2/3)N^3`` for LU, ``N^2`` for back-sub.
"""

from __future__ import annotations

import os
import pstats

# ssik primitives, matched by function name.
SSIK_FLOPS: dict[str, int] = {
    "_cross3": 9,  # 6 mul + 3 sub
    "_dot3": 5,  # 3 mul + 2 add
    "_norm3": 7,  # 3 mul + 2 add + 1 sqrt + 1 fudge
}

# numpy linalg + array ops, matched by (basename, funcname). FLOP estimates
# baked in for our workload's dominant shapes.
NUMPY_FLOPS: dict[tuple[str, str], int] = {
    # 3-vector cross/dot still hit numpy if any callsite was missed by the
    # _cross3/_dot3 propagation; keep these in case.
    ("numeric.py", "cross"): 9,
    ("numeric.py", "dot"): 5,
    # np.linalg.norm: 3-vector dominant (axis_*, joint_origins).
    ("_linalg.py", "norm"): 7,
    # np.linalg.solve: 2x2 dominant (sp6 _refine_sp6).
    ("_linalg.py", "solve"): 10,
    # np.linalg.qr complete on a 2x4 (sp6): R is 4x4, Q is 4x4. ~64 flops.
    ("_linalg.py", "qr"): 64,
    # np.linalg.lstsq: 3x2 (predicates.three_consecutive_intersecting).
    ("_linalg.py", "lstsq"): 32,
    # np.linalg.eig: 4x4 cubic + iteration. Only used in tier-2 RR.
    ("_linalg.py", "eig"): 200,
    # arctan2 / arcsin / cos / sin: count as 10 each (transcendentals).
    ("<frozen>", "atan2"): 10,
}


def flop_budget(stats: pstats.Stats) -> tuple[int, dict[str, int]]:
    """Compute FLOP budget from a :class:`pstats.Stats` snapshot.

    :returns: ``(total_flops, breakdown)`` where ``breakdown`` is keyed by
        ``"<module>:<funcname>"`` for traceability.
    """
    total = 0
    breakdown: dict[str, int] = {}
    for (filename, _lineno, funcname), (cc, _nc, _tt, _ct, _callers) in stats.stats.items():
        flops = SSIK_FLOPS.get(funcname)
        if flops is not None:
            cost = flops * cc
            breakdown[funcname] = breakdown.get(funcname, 0) + cost
            total += cost
            continue
        basename = os.path.basename(filename)
        key = (basename, funcname)
        flops = NUMPY_FLOPS.get(key)
        if flops is not None:
            cost = flops * cc
            label = f"{basename}:{funcname}"
            breakdown[label] = breakdown.get(label, 0) + cost
            total += cost
    return total, breakdown


def _fmt_time(seconds: float) -> str:
    """Format a time in human-readable units (ns / us / ms / s)."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:>9.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:>9.3f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:>9.3f} ms"
    return f"{seconds:>9.3f} s"


def print_flop_summary(
    total_flops: int,
    n_solves: int,
    wall_time_s: float,
    breakdown: dict[str, int],
) -> None:
    """Print a FLOP-budget report.

    Reports machine-invariant FLOP counts, machine-specific wall-clock, the
    achieved effective rate (FLOPs / wall_time), and the projected per-solve
    time at order-of-magnitude scaled rates. The scaled rows let a reader
    estimate cost on a different runtime (Cython port, Rust port) by picking
    the row whose achieved rate matches their target.
    """
    flops_per_solve = total_flops / n_solves
    achieved_gflops = total_flops / wall_time_s / 1e9 if wall_time_s > 0 else 0.0
    seconds_per_solve = wall_time_s / n_solves

    print("\n--- FLOP budget (machine-invariant) ---")
    print(f"  total       {total_flops:>12,d} FLOPs over {n_solves} solves")
    print(f"  per solve   {flops_per_solve:>12,.0f} FLOPs")
    print("  breakdown (per-solve):")
    sorted_ops = sorted(breakdown.items(), key=lambda kv: -kv[1])
    for label, cost in sorted_ops[:8]:
        print(f"    {label:<40s} {cost / n_solves:>10,.1f}")

    print("\n--- wall-clock (this run, this machine) ---")
    print(f"  total       {wall_time_s:>12.3f} s")
    print(f"  per solve   {_fmt_time(seconds_per_solve)}")
    print(f"  achieved    {achieved_gflops:>12.5f} GFLOP/s")

    # Scaled projections: what would this cost if effective rate were 10x,
    # 100x, 1000x, 10000x what we measured? Useful as a transferability
    # estimate -- a Cython port would typically hit 10-100x over Python+numpy
    # on small-op workloads, a native Rust/C port 100-1000x.
    print("\n--- projected per-solve time at scaled rates ---")
    if achieved_gflops > 0:
        for mult in (1.0, 10.0, 100.0, 1000.0, 10000.0):
            target_gflops = achieved_gflops * mult
            s = flops_per_solve / (target_gflops * 1e9)
            label = f"{mult:>5.0f}x measured ({target_gflops:>8.4f} GFLOP/s)"
            print(f"  {label}: {_fmt_time(s)}")
    print()
