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

__all__ = ["solve"]


def solve(
    h: Sequence[NDArray[np.float64]],
    k: Sequence[NDArray[np.float64]],
    p: Sequence[NDArray[np.float64]],
    d1: float,
    d2: float,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[tuple[float, float]], bool]:
    """Solve SP6.

    :param h: length-4 sequence of direction vectors (rows dotted into the
        rotated ``p_i``).
    :param k: length-4 sequence of rotation axes. Typically the caller sets
        ``k[0] == k[2]`` (axis of ``theta1``) and ``k[1] == k[3]`` (axis of
        ``theta2``).
    :param p: length-4 sequence of position vectors to rotate.
    :param d1: first scalar target.
    :param d2: second scalar target.
    :returns: ``(solutions, is_ls)`` where ``solutions`` is a list of up to
        4 ``(theta1, theta2)`` tuples. ``is_ls`` is ``True`` iff no real
        intersection was found (degenerate or infeasible inputs).
    """
    if len(h) != 4 or len(k) != 4 or len(p) != 4:
        raise ValueError("SP6 requires h, k, p to each be length-4 sequences")

    a_cols = []
    for idx in range(4):
        kxp = np.cross(k[idx], p[idx])
        a_cols.append(np.column_stack([kxp, -np.cross(k[idx], kxp)]))

    # a_i is 3x2. h[i] @ a_i is shape (2,), the row of the stacked system.
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

    # b = [d1 - (axial contributions from eq 1), d2 - (eq 2)]
    b = np.array(
        [
            d1 - float(h[0] @ k[0]) * float(k[0] @ p[0]) - float(h[1] @ k[1]) * float(k[1] @ p[1]),
            d2 - float(h[2] @ k[2]) * float(k[2] @ p[2]) - float(h[3] @ k[3]) * float(k[3] @ p[3]),
        ],
        dtype=np.float64,
    )

    # QR of A^T (4x2) in complete mode: Q is 4x4, R_full is 4x2.
    q_full, r_full = np.linalg.qr(a_mat.T, mode="complete")
    x_null_1 = q_full[:, 2]
    x_null_2 = q_full[:, 3]
    q_range = q_full[:, :2]
    r_upper = r_full[:2, :2]
    r_lower = r_upper.T  # lower triangular

    # Rank-deficient system (degenerate / collinear inputs): signal LS failure.
    deg = policy.subproblem_degeneracy
    if abs(float(r_upper[0, 0])) < deg or abs(float(r_upper[1, 1])) < deg:
        return [], True

    x_min_coefs = solve_lower_triangular_system_2x2(r_lower, b)
    x_min = q_range @ x_min_coefs  # length 4

    # Partition: (x[0], x[1]) = (cos t1, sin t1); (x[2], x[3]) = (cos t2, sin t2).
    xn1 = np.column_stack([x_null_1[:2], x_null_2[:2]])
    xn2 = np.column_stack([x_null_1[2:], x_null_2[2:]])

    xi_solutions = solve_two_ellipse_numeric(x_min[:2], xn1, x_min[2:], xn2, policy)

    if not xi_solutions:
        return [], True

    # Filter ``xi`` solutions that don't actually satisfy both unit-circle
    # constraints. ``solve_two_ellipse_numeric`` can return spurious quartic
    # roots near degenerate conic configurations; those show up here as
    # ``|x[:2]| != 1`` or ``|x[2:]| != 1`` and would otherwise produce invalid
    # ``(theta1, theta2)`` pairs. Gated by ``subproblem_numerical``.
    num_tol = policy.subproblem_numerical
    solutions: list[tuple[float, float]] = []
    for xi_0, xi_1 in xi_solutions:
        x = x_min + xi_0 * x_null_1 + xi_1 * x_null_2
        n1 = float(np.linalg.norm(x[:2]))
        n2 = float(np.linalg.norm(x[2:]))
        if abs(n1 - 1.0) > num_tol or abs(n2 - 1.0) > num_tol:
            continue
        theta1 = float(np.arctan2(x[0], x[1]))
        theta2 = float(np.arctan2(x[2], x[3]))
        solutions.append((theta1, theta2))

    if not solutions:
        return [], True
    return solutions, False
