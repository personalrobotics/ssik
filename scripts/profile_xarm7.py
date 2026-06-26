"""Profile xarm7 IK on the host (intended for the Linux/OpenBLAS runner) to
pin down the ~140x slowdown vs macOS (#350): is it import/build, per-solve, and
is the T-perturbation rescue firing because the analytical path returns empty?

Prints a clear split so we know whether to chase the rescue, the analytical
solver's Linux conditioning, or a one-time precompute.
"""

from __future__ import annotations

import os
import time

import numpy as np


def main() -> None:
    print("=== thread env ===")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        print(f"  {var}={os.environ.get(var, '(unset)')}")
    print("=== BLAS backend ===")
    try:
        np.show_config()
    except Exception as exc:  # noqa: BLE001
        print(f"(np.show_config failed: {exc})")

    t0 = time.perf_counter()
    from ssik.prebuilt import xarm7_ik as m

    print(
        f"\n=== import xarm7_ik (build/prime cost): {(time.perf_counter() - t0) * 1000:.0f} ms ==="
    )

    rng = np.random.default_rng(0)
    poses = []
    while len(poses) < 50:
        q = rng.uniform(-0.5, 0.5, size=7)
        q[3] = float(rng.uniform(0.3, 0.7))
        poses.append(m.fk(q))

    # First solve vs warm (catches a lazy one-time precompute on first solve).
    t1 = time.perf_counter()
    m.solve(poses[0])
    first_ms = (time.perf_counter() - t1) * 1000
    t2 = time.perf_counter()
    for T in poses[1:11]:
        m.solve(T)
    warm_ms = (time.perf_counter() - t2) * 100  # /10 *1000
    print(f"\n=== first solve: {first_ms:.0f} ms | warm solve avg: {warm_ms:.0f} ms ===")

    def _run(label: str, *, allow_rescue: bool) -> None:
        times, empty, rescue = [], 0, 0
        for T in poses:
            ts = time.perf_counter()
            sols = m.solve(T, allow_rescue=allow_rescue)
            times.append(time.perf_counter() - ts)
            if not sols:
                empty += 1
            elif any(getattr(s, "refinement_used", "none") == "lm" for s in sols):
                rescue += 1
        arr = np.array(times) * 1000
        print(
            f"\n=== {label} (allow_rescue={allow_rescue}) ===\n"
            f"  {len(poses)} solves: mean {arr.mean():.0f} ms, "
            f"median {np.median(arr):.0f} ms, max {arr.max():.0f} ms, total {arr.sum() / 1000:.1f} s\n"
            f"  empty (analytical found nothing): {empty}/{len(poses)} | "
            f"rescue-fired (lm): {rescue}/{len(poses)}"
        )

    # The decisive comparison: if allow_rescue=False is fast but True is slow,
    # the analytical path is returning empty on Linux and the rescue is the cost.
    _run("DEFAULT", allow_rescue=True)
    _run("ANALYTICAL-ONLY", allow_rescue=False)


if __name__ == "__main__":
    main()
