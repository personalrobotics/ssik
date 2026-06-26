"""Profile xarm7 IK on the host (intended for the Linux/OpenBLAS runner) to
pin down the ~230x slowdown vs macOS (#350).

Established so far (on Linux CI): it's the pure analytical solve (~5.3s/solve),
not import, not the T-perturbation rescue (fired 0/50; allow_rescue=False
identical), and not OpenBLAS thread count (=1 made no difference). So the solve
does vastly more *work* on Linux numerics -- this cProfiles a few solves to show
exactly which function that is.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import time

import numpy as np


def main() -> None:
    print("=== thread env ===")
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        print(f"  {var}={os.environ.get(var, '(unset)')}")

    from ssik.prebuilt import xarm7_ik as m

    rng = np.random.default_rng(0)
    poses = []
    while len(poses) < 10:
        q = rng.uniform(-0.5, 0.5, size=7)
        q[3] = float(rng.uniform(0.3, 0.7))
        poses.append(m.fk(q))

    m.solve(poses[0])  # warm
    t = time.perf_counter()
    for T in poses:
        m.solve(T)
    print(
        f"\n=== {len(poses)} solves: {(time.perf_counter() - t) / len(poses) * 1000:.0f} ms/solve ==="
    )

    print("\n=== cProfile of 3 solves (top 25 by cumulative time) ===")
    pr = cProfile.Profile()
    pr.enable()
    for T in poses[:3]:
        m.solve(T)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())

    print("=== same, sorted by total (self) time ===")
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(15)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
