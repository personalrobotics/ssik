#!/usr/bin/env python
"""Smoke-test the shipped native extension (#506).

Imports ``ssik._ssik_native`` and runs one three_parallel solve through it,
asserting the result matches the Python artifact. Used by the PR-CI wheel job
and runnable against an installed wheel. Exits 0 on success, non-zero on any
failure. On platforms where native isn't shipped (Windows, first cut) the
extension is absent and this script reports that and exits 0 -- the Python
fallback is the expected behaviour there.
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    try:
        from ssik import _ssik_native
    except ImportError:
        if sys.platform == "win32":
            print("[native_smoke] native not shipped on Windows (Python fallback) -- OK")
            return 0
        print(f"[native_smoke] FAIL: ssik._ssik_native missing on {sys.platform}", file=sys.stderr)
        return 1

    from ssik.prebuilt import ur5_ik

    kb = ur5_ik._KB
    axes = np.array([j.axis for j in kb.joints], dtype=np.float64)
    t_left = np.array([j.T_left for j in kb.joints], dtype=np.float64)
    t_right = np.array([j.T_right for j in kb.joints], dtype=np.float64)
    types = np.array([0 for _ in kb.joints], dtype=np.int32)
    lo = np.array([j.limits[0] for j in kb.joints], dtype=np.float64)
    hi = np.array([j.limits[1] for j in kb.joints], dtype=np.float64)
    has_limits = np.array([1 for _ in kb.joints], dtype=np.int32)

    q = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2])
    t_target = np.asarray(ur5_ik.fk(q), dtype=np.float64)

    qs, resids, _refine = _ssik_native.native_artifact_solve(
        "ikgeo.three_parallel",
        axes,
        t_left,
        t_right,
        types,
        lo,
        hi,
        has_limits,
        t_target,
        True,
        False,
        np.zeros(6),
        "wrap_linf",
        False,
        0.0,
        -1,
        True,
        15,
    )
    py = ur5_ik.solve(t_target)

    if len(qs) != len(py):
        print(f"[native_smoke] FAIL: native returned {len(qs)}, Python {len(py)}", file=sys.stderr)
        return 1
    worst = float(max(resids)) if len(resids) else 0.0
    if worst >= 1e-9:
        print(f"[native_smoke] FAIL: worst native FK residual {worst:.2e}", file=sys.stderr)
        return 1
    print(f"[native_smoke] OK: {len(qs)} solutions, worst FK residual {worst:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
