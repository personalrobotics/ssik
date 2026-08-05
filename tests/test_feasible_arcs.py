"""C++ feasible_arcs geometry vs the Python oracle (#515).

feasible_arcs / feasible_arcs_bounded are the SHARED redundant-7R in-limits
primitive: SRS swivel-limits (resolve_in_limits) and spherical_shoulder
q6-redundancy both build on them. This fuzzes a synthetic two-harmonic joint
family against ``ssik.solvers.seven_r._feasible_param`` -- the C++ port must
reproduce every arc boundary exactly.

Skips when the test-only extension isn't built (the cpp CI job builds it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ssik.solvers.seven_r._feasible_param import PARAM_GRID, feasible_arcs, feasible_arcs_bounded
from tests._cpp_backend import _load_ext, cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")

_K = 7
_N = 3000


def _q_scalar(coeffs: np.ndarray):  # type: ignore[no-untyped-def]
    def q(t: float) -> np.ndarray:
        tt = np.asarray(t, dtype=np.float64)
        out = (
            coeffs[:, 0]
            + coeffs[:, 1] * np.cos(tt)
            + coeffs[:, 2] * np.sin(tt)
            + coeffs[:, 3] * np.cos(2 * tt)
            + coeffs[:, 4] * np.sin(2 * tt)
        )
        return np.asarray(out, dtype=np.float64)

    return q


def _match(a: list[Any], b: list[Any], tol: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    return all(
        abs(x0 - y0) <= tol and abs(x1 - y1) <= tol
        for (x0, x1), (y0, y1) in zip(sorted(a), sorted(b), strict=True)
    )


def test_feasible_arcs_matches_oracle() -> None:
    ext = _load_ext()
    rng = np.random.default_rng(0)
    periodic_mismatch = bounded_mismatch = 0
    multi = empty = 0
    for _ in range(_N):
        coeffs = rng.uniform(-1.5, 1.5, (_K, 5))
        coeffs[:, 0] = rng.uniform(-np.pi, np.pi, _K)
        lo = rng.uniform(-np.pi, np.pi, _K)
        hi = lo + rng.uniform(0.1, 2.2 * np.pi, _K)  # mix of tight + unconstrained joints
        swept = sorted(rng.choice(_K, size=rng.integers(1, _K + 1), replace=False).tolist())
        limits = [(float(lo[i]), float(hi[i])) for i in range(_K)]
        qs = _q_scalar(coeffs)
        swept_arr = np.array(swept, dtype=np.int32)

        q_grid = np.array([qs(g) for g in PARAM_GRID])
        py_p = feasible_arcs(qs, q_grid, swept, limits, PARAM_GRID)
        cpp_p = ext.feasible_arcs_test(
            coeffs, swept_arr, lo, hi, PARAM_GRID.astype(np.float64), False
        )
        if not _match(py_p, cpp_p):
            periodic_mismatch += 1
        multi += len(py_p) >= 2
        empty += len(py_p) == 0

        a, b = sorted(rng.uniform(-np.pi, np.pi, 2).tolist())
        b = max(b, a + 0.2)
        bgrid = np.linspace(a, b, 120)
        bq_grid = np.array([qs(g) for g in bgrid])
        py_b = feasible_arcs_bounded(qs, bq_grid, swept, limits, bgrid)
        cpp_b = ext.feasible_arcs_test(coeffs, swept_arr, lo, hi, bgrid.astype(np.float64), True)
        if not _match(py_b, cpp_b):
            bounded_mismatch += 1

    assert periodic_mismatch == 0, f"{periodic_mismatch}/{_N} periodic feasible_arcs mismatches"
    assert bounded_mismatch == 0, f"{bounded_mismatch}/{_N} bounded feasible_arcs mismatches"
    # Sanity: the fuzz actually exercised the interesting branches.
    assert multi > 100, f"only {multi} multi-arc cases -- fuzz not exercising arc splits"
    assert empty > 100, f"only {empty} empty-arc cases -- fuzz not exercising infeasibility"
