"""2D grid + Nelder-Mead minimum finding for IK-Geo's tier-2 solver.

Private; consumed by ``gen_six_dof``. Ported clean-room from the BSD-3
[ik-geo Rust reference][ikgeo] ``auxiliary::search_2d``, with Nelder-Mead
implemented inline (to avoid a scipy dependency).

Algorithm:
1. Sample the error function on an n x n grid over
   ``[(min0, min1), (max0, max1)]``. The function returns a vector of
   per-branch errors; we mask entries above ``MIN_THRESHOLD`` to ``NaN``.
2. Iteratively find the global minimum across all cells and branches.
   After picking one, "clear the blob" -- NaN out the connected
   component (including wrap-around on the grid) so subsequent
   iterations find disjoint minima. Repeats until no finite minima
   remain.
3. Nelder-Mead refines each (x0, x1) to sub-grid precision on the
   selected branch's scalar error.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/auxiliary.rs
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["search_2d"]

_MIN_THRESHOLD = 1e-1
_N_MAX_MINIMA = 1000


def search_2d(
    f: Callable[[float, float], NDArray[np.float64]],
    low: tuple[float, float],
    high: tuple[float, float],
    n: int,
) -> list[tuple[float, float, int]]:
    """Return local minima of each component of a vector-valued 2D function.

    ``f(x0, x1)`` returns a length-N vector of errors per branch. We sample
    on an ``n x n`` grid, mask cells where the error exceeds
    ``_MIN_THRESHOLD`` (via NaN), iteratively pick the globally-minimum
    remaining cell, clear its connected blob, and refine via Nelder-Mead.

    :returns: list of ``(x0, x1, branch_index)`` triples at each refined
        minimum.
    """
    x0_vals = np.linspace(low[0], high[0], n, endpoint=False) + (high[0] - low[0]) / n
    x1_vals = np.linspace(low[1], high[1], n, endpoint=False) + (high[1] - low[1]) / n
    x0_vals = np.linspace(low[0], high[0], n, endpoint=False)
    x1_vals = np.linspace(low[1], high[1], n, endpoint=False)

    # Evaluate once per cell; probe N at each cell.
    mesh: NDArray[np.float64] | None = None
    for i, x0 in enumerate(x0_vals):
        for j, x1 in enumerate(x1_vals):
            v = np.asarray(f(float(x0), float(x1)), dtype=np.float64)
            if mesh is None:
                mesh = np.full((n, n, v.shape[0]), np.nan)
            for k, vk in enumerate(v):
                mesh[i, j, k] = vk if float(vk) <= _MIN_THRESHOLD else np.nan

    if mesh is None:
        return []

    minima: list[tuple[float, float, int]] = []
    for _ in range(_N_MAX_MINIMA):
        if np.all(np.isnan(mesh)):
            break
        flat_idx = int(np.nanargmin(mesh))
        i_arr, j_arr, k_arr = np.unravel_index(flat_idx, mesh.shape)
        i, j, k = int(i_arr), int(j_arr), int(k_arr)
        minima.append((float(x0_vals[i]), float(x1_vals[j]), k))
        _clear_blob(mesh, i, j, k, n)
    else:
        raise RuntimeError("search_2d: too many minima found (exceeded N_MAX_MINIMA)")

    delta0 = (high[0] - low[0]) / n
    delta1 = (high[1] - low[1]) / n

    refined: list[tuple[float, float, int]] = []
    for x0, x1, k in minima:
        fx0, fx1 = _nelder_mead(
            lambda x: float(f(x[0], x[1])[k]),  # noqa: B023 -- k is loop var, used eagerly per pass
            np.array([x0, x1]),
            initial_step=np.array([delta0 / 2.0, delta1 / 2.0]),
        )
        refined.append((fx0, fx1, k))

    return refined


def _clear_blob(mesh: NDArray[np.float64], i: int, j: int, k: int, n: int) -> None:
    """Iteratively NaN-out the connected blob of non-NaN cells in channel
    ``k`` reachable from ``(i, j)`` with wrap-around. Uses a stack to
    avoid Python recursion limits (the Rust version recurses).
    """
    stack: list[tuple[int, int]] = [(i, j)]
    while stack:
        ii, jj = stack.pop()
        ii_w = ii % n
        jj_w = jj % n
        if np.isnan(mesh[ii_w, jj_w, k]):
            continue
        mesh[ii_w, jj_w, k] = np.nan
        stack.append((ii_w + 1, jj_w))
        stack.append((ii_w - 1, jj_w))
        stack.append((ii_w, jj_w + 1))
        stack.append((ii_w, jj_w - 1))


# ---------------------------------------------------------------------------
# Nelder-Mead implementation (inline so we don't pull in scipy).
# ---------------------------------------------------------------------------


def _nelder_mead(
    f: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    initial_step: NDArray[np.float64],
    sd_tol: float = 1e-6,
    max_iters: int = 1_000_000,
) -> tuple[float, float]:
    """Classic Nelder-Mead simplex minimisation. Returns the refined
    ``(x0, x1)`` at convergence (or after ``max_iters``, whichever first).

    Constants match the Rust reference (argmin's NelderMead defaults):
    alpha=1.0 (reflection), gamma=2.0 (expansion), sigma=0.5 (contraction),
    rho=0.5 (shrinkage). ``sd_tol = 1e-6`` on the function values'
    standard deviation across the simplex.
    """
    alpha, gamma, sigma, rho = 1.0, 2.0, 0.5, 0.5

    simplex = np.array(
        [
            x0,
            x0 + np.array([initial_step[0], 0.0]),
            x0 + np.array([0.0, -initial_step[1]]),
        ],
        dtype=np.float64,
    )
    fvals = np.array([f(p) for p in simplex])

    for _ in range(max_iters):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        if float(np.std(fvals)) < sd_tol:
            break

        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
            continue

        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
            continue

        # fr >= fvals[-2] -- contract
        xc = centroid + sigma * (simplex[-1] - centroid)
        fc = f(xc)
        if fc < fvals[-1]:
            simplex[-1] = xc
            fvals[-1] = fc
            continue

        # Shrink toward simplex[0]
        simplex[1:] = simplex[0] + rho * (simplex[1:] - simplex[0])
        for idx in range(1, 3):
            fvals[idx] = f(simplex[idx])

    return float(simplex[0, 0]), float(simplex[0, 1])
