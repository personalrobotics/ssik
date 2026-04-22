"""Subproblem 6: two coupled SP4-like equations in two unknowns.

Given four direction vectors ``h_i``, four rotation axes ``k_i``, four
position vectors ``p_i``, and two scalars ``d1, d2``, find ``(theta1, theta2)``
satisfying::

    h[0] . Rot(k[0], theta1) @ p[0] + h[1] . Rot(k[1], theta2) @ p[1] = d1
    h[2] . Rot(k[2], theta1) @ p[2] + h[3] . Rot(k[3], theta2) @ p[3] = d2

In practice callers set ``k[0] == k[2]`` (the shared axis of ``theta1``) and
``k[1] == k[3]`` (the shared axis of ``theta2``); the length-4 ``k`` argument
preserves the flexibility IK-Geo allows.

Up to 4 solutions.

**Implementation**: port of the BSD-3 [ik-geo Rust reference][ikgeo] (Elias &
Wen, arXiv:2211.05737). After Rodrigues expansion each equation becomes linear
in ``(cos theta1, sin theta1, cos theta2, sin theta2)``. Stacking the two
equations gives ``A @ x = b`` with ``A`` being 2x4. The general solution is
``x = x_min + xi_1 * x_null_1 + xi_2 * x_null_2`` where the null space comes
from the QR of ``A^T``. Enforcing the trigonometric constraints
``|x[:2]| = |x[2:]| = 1`` reduces to intersecting two conics in
``(xi_1, xi_2)``, solved via :func:`_aux.solve_two_ellipse_numeric`.

**Robustness beyond IK-Geo** (issue #48): upfront degeneracy rejection,
post-verification against the original equations, deduplication,
best-LS return on total infeasibility. See :mod:`ssik.subproblems.sp5`
docstring for the full list -- SP5 and SP6 share the same discipline.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/subproblems/mod.rs
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.subproblems._aux import (
    solve_lower_triangular_system_2x2,
    solve_two_ellipse_numeric,
)
from ssik.subproblems._rotation import rotate
from ssik.subproblems._validate import validate_vec3_iterable

__all__ = ["solve"]


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _close_pair(a: tuple[float, float], b: tuple[float, float], tol: float) -> bool:
    return abs(_wrap(a[0] - b[0])) < tol and abs(_wrap(a[1] - b[1])) < tol


def _dedup(pairs: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for p in pairs:
        if not any(_close_pair(p, u, tol) for u in unique):
            unique.append(p)
    return unique


def _all_p_collinear_with_k(
    k: Sequence[NDArray[np.float64]],
    p: Sequence[NDArray[np.float64]],
    deg_tol: float,
) -> bool:
    """Return True when *every* ``p_i`` is collinear with ``k_i``.

    When only one or two terms are collinear the overall 2D system can still
    be solvable -- the non-collinear terms provide the missing constraints.
    We only short-circuit when the entire system is trivially degenerate
    (every rotation is effectively identity). For other degeneracies we rely
    on the rank check of the stacked QR diagonals and on post-verification.
    """
    for k_i, p_i in zip(k, p, strict=True):
        p_perp_sq = float(np.dot(p_i, p_i)) - float(np.dot(k_i, p_i)) ** 2
        if p_perp_sq >= deg_tol:
            return False
    return True


def _residual(
    theta1: float,
    theta2: float,
    h: Sequence[NDArray[np.float64]],
    k: Sequence[NDArray[np.float64]],
    p: Sequence[NDArray[np.float64]],
    d1: float,
    d2: float,
) -> float:
    lhs1 = float(h[0] @ rotate(k[0], theta1, p[0])) + float(h[1] @ rotate(k[1], theta2, p[1]))
    lhs2 = float(h[2] @ rotate(k[2], theta1, p[2])) + float(h[3] @ rotate(k[3], theta2, p[3]))
    return max(abs(lhs1 - d1), abs(lhs2 - d2))


def solve(
    h: Sequence[NDArray[np.float64]],
    k: Sequence[NDArray[np.float64]],
    p: Sequence[NDArray[np.float64]],
    d1: float,
    d2: float,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[tuple[float, float]], bool]:
    """Solve SP6.

    See SP5's module docstring for the shared robustness guarantees.

    :returns: ``(solutions, is_ls)``. Exactly the same semantics as SP5:
        exact solutions (up to 4, deduplicated) with ``is_ls=False``, or a
        best-LS fallback with ``is_ls=True`` when no candidate satisfies
        both equations within ``subproblem_numerical``.
    """
    if len(h) != 4 or len(k) != 4 or len(p) != 4:
        raise ValueError("SP6 requires h, k, p to each be length-4 sequences")
    validate_vec3_iterable(h, "h")
    validate_vec3_iterable(k, "k")
    validate_vec3_iterable(p, "p")
    if not (np.isfinite(d1) and np.isfinite(d2)):
        raise ValueError(f"d1/d2 must be finite; got {d1}, {d2}")

    deg_tol = policy.subproblem_degeneracy
    num_tol = policy.subproblem_numerical

    if _all_p_collinear_with_k(k, p, deg_tol):
        return [], True

    a_cols = []
    for idx in range(4):
        kxp = np.cross(k[idx], p[idx])
        a_cols.append(np.column_stack([kxp, -np.cross(k[idx], kxp)]))

    h1_a1 = h[0] @ a_cols[0]
    h2_a2 = h[1] @ a_cols[1]
    h3_a3 = h[2] @ a_cols[2]
    h4_a4 = h[3] @ a_cols[3]

    a_mat = np.array(
        [
            [h1_a1[0], h1_a1[1], h2_a2[0], h2_a2[1]],
            [h3_a3[0], h3_a3[1], h4_a4[0], h4_a4[1]],
        ],
        dtype=np.float64,
    )  # 2x4

    b = np.array(
        [
            d1 - float(h[0] @ k[0]) * float(k[0] @ p[0]) - float(h[1] @ k[1]) * float(k[1] @ p[1]),
            d2 - float(h[2] @ k[2]) * float(k[2] @ p[2]) - float(h[3] @ k[3]) * float(k[3] @ p[3]),
        ],
        dtype=np.float64,
    )

    q_full, r_full = np.linalg.qr(a_mat.T, mode="complete")
    x_null_1 = q_full[:, 2]
    x_null_2 = q_full[:, 3]
    q_range = q_full[:, :2]
    r_upper = r_full[:2, :2]
    r_lower = r_upper.T

    if abs(float(r_upper[0, 0])) < deg_tol or abs(float(r_upper[1, 1])) < deg_tol:
        return [], True

    x_min_coefs = solve_lower_triangular_system_2x2(r_lower, b)
    x_min = q_range @ x_min_coefs

    xn1 = np.column_stack([x_null_1[:2], x_null_2[:2]])
    xn2 = np.column_stack([x_null_1[2:], x_null_2[2:]])

    xi_solutions = solve_two_ellipse_numeric(x_min[:2], xn1, x_min[2:], xn2, policy)
    if not xi_solutions:
        return [], True

    candidates: list[tuple[float, float]] = []
    for xi_0, xi_1 in xi_solutions:
        x = x_min + xi_0 * x_null_1 + xi_1 * x_null_2
        n1 = float(np.linalg.norm(x[:2]))
        n2 = float(np.linalg.norm(x[2:]))
        if abs(n1 - 1.0) > num_tol or abs(n2 - 1.0) > num_tol:
            continue
        theta1 = float(np.arctan2(x[0], x[1]))
        theta2 = float(np.arctan2(x[2], x[3]))
        candidates.append((theta1, theta2))

    def residual(cand: tuple[float, float]) -> float:
        return _residual(cand[0], cand[1], h, k, p, d1, d2)

    exact = [cand for cand in candidates if residual(cand) < num_tol]

    if exact:
        solutions = _dedup(exact, policy.subproblem_dedup)
        assert len(solutions) <= 4, f"SP6 returned >4 solutions: {len(solutions)}"
        return solutions, False

    if not candidates:
        return [], True

    best = min(candidates, key=residual)
    return [best], True
