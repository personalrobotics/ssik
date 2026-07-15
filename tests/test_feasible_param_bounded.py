"""Bounded-domain feasible-interval core (#373).

The periodic ``feasible_arcs`` handles the SRS swivel on the circle; the bounded
variant handles a locked-joint redundancy on ``[a, b]`` with no wrap-around
(#373's q6). Validate the exact arcs against a dense brute-force feasibility mask.
"""

from __future__ import annotations

import numpy as np

from ssik.solvers.seven_r._feasible_param import (
    arcs_for_joint_bounded,
    feasible_arcs_bounded,
)


def _brute_mask(q_scalar, joints, limits, grid):
    return np.array(
        [all(limits[i][0] <= q_scalar(t)[i] <= limits[i][1] for i in joints) for t in grid]
    )


def _covers(arcs, grid, mask, tol=1e-6):
    """Every feasible grid point lies in some arc; every arc point is feasible."""
    for t, m in zip(grid, mask, strict=True):
        in_arc = any(a - tol <= t <= b + tol for a, b in arcs)
        if m and not in_arc:
            return False
    for a, b in arcs:
        mid = 0.5 * (a + b)
        j = int(np.argmin(np.abs(grid - mid)))
        if not mask[j]:
            return False
    return True


def test_single_joint_bounded_matches_brute() -> None:
    """A smooth q(t) with a limited joint: bounded arcs match the brute mask."""
    lo, hi = -2.5, 2.5
    grid = np.linspace(lo, hi, 240)

    # q0(t) = 1.2 sin(t) + 0.3 t  -- smooth, non-sinusoidal, a few crossings.
    def q_scalar(t):
        return np.array([1.2 * np.sin(t) + 0.3 * t])

    limits = [(-0.9, 0.9)]
    q_col = np.array([q_scalar(t)[0] for t in grid])
    arcs = arcs_for_joint_bounded(lambda t: q_scalar(t)[0], *limits[0], grid, q_col)
    mask = _brute_mask(q_scalar, [0], limits, grid)
    assert arcs, "expected some feasible sub-interval"
    assert _covers(arcs, grid, mask), f"arcs {arcs} disagree with brute feasibility"
    # arcs stay inside the bounded domain
    for a, b in arcs:
        assert lo - 1e-9 <= a <= b <= hi + 1e-9


def test_multi_joint_bounded_intersection() -> None:
    """Two joints: the feasible set is the intersection of their arcs, clipped
    to the bounded domain."""
    lo, hi = -3.0, 3.0
    grid = np.linspace(lo, hi, 300)

    def q_scalar(t):
        return np.array([np.sin(t), 0.8 * np.cos(0.7 * t) + 0.2 * t])

    limits = [(-0.6, 0.6), (-0.5, 0.7)]
    q_grid = np.array([q_scalar(t) for t in grid])
    arcs = feasible_arcs_bounded(q_scalar, q_grid, [0, 1], limits, grid)
    mask = _brute_mask(q_scalar, [0, 1], limits, grid)
    assert _covers(arcs, grid, mask), f"arcs {arcs} disagree with brute feasibility"


def test_bounded_unconstrained_and_empty() -> None:
    """A joint whose limit half-width >= pi is unconstrained (full domain); an
    unsatisfiable limit yields no arc."""
    lo, hi = -1.0, 1.0
    grid = np.linspace(lo, hi, 100)
    q_col = np.array([0.5 * t for t in grid])
    full = arcs_for_joint_bounded(lambda t: 0.5 * t, -np.pi, np.pi, grid, q_col)
    assert full == [(lo, hi)]
    # q(t) = 0.5 t in [-0.5, 0.5], demand [2.0, 3.0] -> empty
    empty = arcs_for_joint_bounded(lambda t: 0.5 * t, 2.0, 3.0, grid, q_col)
    assert empty == []
