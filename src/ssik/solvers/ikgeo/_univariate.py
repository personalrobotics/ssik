"""1D zero-finding primitives for IK-Geo's univariate-search solvers.

Private; shared between ``two_intersecting`` and ``two_parallel`` (Round 2
of the solver roster in issue #53).

- :func:`search_1d` -- clean-room port of IK-Geo Rust's ``search_1d``:
  samples a vector-valued function on a grid and tracks sign changes
  per component index via false-position refinement. Works when the
  branch index is STABLE across samples.

- :func:`search_1d_matched` -- geometric-branch-matched variant:
  tracks branches by (cos, sin) proximity in branch-parameter space
  across adjacent samples rather than by index. Handles the case
  where the inner multi-valued solver returns branches whose order
  varies with the search variable (e.g. SP6's quartic-root index
  isn't stable across q1 samples). This gives substantially better
  completeness for ``two_parallel`` than plain ``search_1d``.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/auxiliary.rs
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = ["search_1d", "search_1d_matched"]

_CROSS_THRESHOLD = 0.1
_FZ_ITERATIONS = 100
_FZ_EPSILON = 1e-5


def search_1d(
    f: Callable[[float], NDArray[np.float64]],
    left: float,
    right: float,
    initial_samples: int,
) -> list[tuple[float, int]]:
    """Find zeros of each component of a vector-valued function ``f`` on
    ``[left, right]``.

    ``f(x)`` returns a length-N vector; ``search_1d`` detects sign changes
    in each component independently over a uniform sampling of
    ``initial_samples`` intervals and refines each bracketed zero via
    false-position iteration.

    :returns: list of ``(x_zero, component_index)`` pairs. Same API as
        the Rust reference: the caller uses ``component_index`` to pick
        which branch of the multi-valued inner solver (e.g. SP5 triple)
        produced the bracketed zero.
    """
    zeros: list[tuple[float, int]] = []
    delta = (right - left) / initial_samples

    last_v = f(left)
    x = left + delta

    for _ in range(initial_samples):
        v = f(x)
        for i, (y, last_y) in enumerate(zip(v, last_v, strict=True)):
            y_f = float(y)
            last_y_f = float(last_y)
            # Sign change AND both bracketed values are below
            # the cross-threshold (IK-Geo's heuristic to avoid chasing
            # ``NAN``/``INFINITY`` discontinuities from SP5 returning no
            # real solutions on a subinterval).
            if (y_f < 0.0) != (last_y_f < 0.0) and (
                abs(y_f) < _CROSS_THRESHOLD and abs(last_y_f) < _CROSS_THRESHOLD
            ):
                z = _find_zero(f, x - delta, x, i)
                if z is not None:
                    zeros.append((z, i))
        last_v = v
        x += delta

    return zeros


def _find_zero(
    f: Callable[[float], NDArray[np.float64]],
    left: float,
    right: float,
    i: int,
) -> float | None:
    """False-position (regula falsi) refinement of a bracketed zero of
    component ``i`` of ``f`` on ``[left, right]``. Returns ``None`` if a
    sampled value is non-finite (indicating the inner solver stopped
    producing real solutions mid-interval)."""
    x_left, x_right = left, right
    y_left = float(f(x_left)[i])
    y_right = float(f(x_right)[i])

    for _ in range(_FZ_ITERATIONS):
        delta = y_right - y_left
        if abs(delta) < _FZ_EPSILON:
            break
        x_0 = x_left - y_left * (x_right - x_left) / delta
        y_0 = float(f(x_0)[i])
        if not np.isfinite(y_0):
            return None
        if (y_left < 0.0) != (y_0 < 0.0):
            x_left = x_0
            y_left = y_0
        else:
            x_right = x_0
            y_right = y_0

    if left <= x_left <= right:
        return x_left
    return None


# ---------------------------------------------------------------------------
# Geometric-branch-matched variant for tier-1 solvers where the inner
# multi-valued subproblem's branch index isn't stable across samples.
# ---------------------------------------------------------------------------

_MATCH_ANGLE_TOL = np.pi / 6.0  # 30 degrees -- matches branches across a single
# grid step without merging geometrically distinct ones.


def search_1d_matched(
    f_branches: Callable[[float], list[tuple[tuple[float, float], float]]],
    left: float,
    right: float,
    initial_samples: int,
) -> list[tuple[float, tuple[float, float]]]:
    """Find zeros of a multi-branched 1D function using geometric branch
    tracking.

    ``f_branches(x)`` returns a list of ``((a1, a2), error)`` tuples: for
    each branch active at ``x``, its two characteristic angles ``(a1, a2)``
    and the scalar error at ``x``. The zero-finding tracks branches across
    adjacent grid samples by matching ``(a1, a2)`` in wrap-to-pi angular
    distance (threshold ``_MATCH_ANGLE_TOL``) rather than by list index.
    This handles the case where the underlying multi-valued solver
    reorders branches as the search variable changes -- which breaks the
    index-based :func:`search_1d`.

    :returns: list of ``(x_zero, (a1, a2))`` pairs. The ``(a1, a2)`` is
        the branch parameters at the detected zero (from evaluating
        ``f_branches`` at the refined ``x``, matched by proximity to the
        tracked branch). Caller uses these to avoid re-evaluating.
    """
    zeros: list[tuple[float, tuple[float, float]]] = []
    delta = (right - left) / initial_samples
    last_branches = f_branches(left)
    x_last = left
    x = left + delta

    for _ in range(initial_samples):
        cur_branches = f_branches(x)
        # For each current branch, find closest last branch; if found, check
        # for sign change.
        for cur_params, cur_err in cur_branches:
            if not np.isfinite(cur_err):
                continue
            matched = _closest_branch(cur_params, last_branches)
            if matched is None:
                continue
            _, last_err = matched
            if not np.isfinite(last_err):
                continue
            if (cur_err < 0.0) != (last_err < 0.0) and (
                abs(cur_err) < _CROSS_THRESHOLD and abs(last_err) < _CROSS_THRESHOLD
            ):
                z, z_params = _find_zero_matched(
                    f_branches, x_last, x, last_err, cur_err, cur_params
                )
                if z is not None and z_params is not None:
                    zeros.append((z, z_params))
        last_branches = cur_branches
        x_last = x
        x += delta

    return zeros


def _branch_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Wrap-to-pi angular distance between two (a1, a2) tuples."""

    def _w(x: float) -> float:
        return float(((x + np.pi) % (2 * np.pi)) - np.pi)

    d1 = _w(a[0] - b[0])
    d2 = _w(a[1] - b[1])
    return float(np.hypot(d1, d2))


def _closest_branch(
    params: tuple[float, float],
    candidates: list[tuple[tuple[float, float], float]],
) -> tuple[tuple[float, float], float] | None:
    """Return the candidate branch (params, err) closest to ``params`` in
    wrap-to-pi distance, or ``None`` if no candidate is within
    ``_MATCH_ANGLE_TOL``."""
    best: tuple[tuple[float, float], float] | None = None
    best_d = _MATCH_ANGLE_TOL
    for c_params, c_err in candidates:
        d = _branch_distance(params, c_params)
        if d < best_d:
            best_d = d
            best = (c_params, c_err)
    return best


def _find_zero_matched(
    f_branches: Callable[[float], list[tuple[tuple[float, float], float]]],
    left: float,
    right: float,
    y_left: float,
    y_right: float,
    branch_params_hint: tuple[float, float],
) -> tuple[float | None, tuple[float, float] | None]:
    """False-position refinement on a bracketed zero, tracking the
    matched geometric branch. At each iteration we re-evaluate
    ``f_branches`` and pick the branch closest to the previously-tracked
    ``branch_params``; if no branch is close enough the refinement
    bails out.
    """
    x_left, x_right = left, right
    tracked = branch_params_hint

    for _ in range(_FZ_ITERATIONS):
        delta = y_right - y_left
        if abs(delta) < _FZ_EPSILON:
            break
        x_0 = x_left - y_left * (x_right - x_left) / delta
        branches_0 = f_branches(x_0)
        match = _closest_branch(tracked, branches_0)
        if match is None:
            return None, None
        p_0, y_0 = match
        if not np.isfinite(y_0):
            return None, None
        tracked = p_0
        if (y_left < 0.0) != (y_0 < 0.0):
            x_right = x_0
            y_right = y_0
        else:
            x_left = x_0
            y_left = y_0

    if left <= x_left <= right:
        return x_left, tracked
    return None, None
