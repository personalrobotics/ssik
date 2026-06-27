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

    # Instrument the cached-RR path: is it primed? does it return solutions or
    # None (→ slow two_intersecting fallback)? This distinguishes "prime didn't
    # load on Linux" from "primed but RR eig returns nothing on OpenBLAS".
    import ssik.solvers.jointlock.seven_r as jl
    from ssik.kinematics.poe_to_dh import poe_to_dh
    from ssik.solvers.ikgeo._raghavan_roth import primed_linearity_for_dh

    stats = {"calls": 0, "primed": 0, "returned_none": 0, "returned_sols": 0}
    _orig = jl._try_cached_rr

    def _wrapped(sub_kb, *a, **k):  # type: ignore[no-untyped-def]
        stats["calls"] += 1
        dh = poe_to_dh(sub_kb)
        if (
            primed_linearity_for_dh(
                tuple(float(x) for x in dh.alpha),
                tuple(float(x) for x in dh.a),
                tuple(float(x) for x in dh.d),
            )
            is not None
        ):
            stats["primed"] += 1
        r = _orig(sub_kb, *a, **k)
        stats["returned_none" if r is None else "returned_sols"] += 1
        return r

    jl._try_cached_rr = _wrapped
    m.solve(poses[0])  # warm
    t = time.perf_counter()
    for T in poses:
        m.solve(T)
    per = (time.perf_counter() - t) / len(poses) * 1000
    print(f"\n=== {len(poses)} solves: {per:.0f} ms/solve ===")
    print(f"=== _try_cached_rr: {stats} ===")  # expect primed==calls, none==0 after #350 fix
    jl._try_cached_rr = _orig

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
