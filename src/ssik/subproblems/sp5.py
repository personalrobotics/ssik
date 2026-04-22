"""Subproblem 5: three-rotation composition.

Given four position vectors ``p0, p1, p2, p3`` and three rotation axes
``k1, k2, k3``, find ``(theta1, theta2, theta3)`` satisfying::

    p0 + Rot(k1, theta1) @ p1 = Rot(k2, theta2) @ (p2 + Rot(k3, theta3) @ p3)

Up to 4 solutions (at most 8 intermediate candidates; in this port we return
whatever survives feasibility filtering rather than IK-Geo's top-4 reduction,
since Python lists have no fixed capacity).

**Implementation**: port of the BSD-3 [ik-geo Rust reference][ikgeo]
(Elias & Wen, arXiv:2211.05737). Each rotated vector traces a cone around
``k2`` parameterised by a scalar ``h``. Subtracting the two cone polynomials
yields a univariate quartic in ``h``; real roots give candidate ``h`` values,
from which ``(theta1, theta3)`` pairs are recovered via matching sign branches
of a 2x2 quadratic-circle system, and ``theta2`` via :func:`sp1.solve`.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/subproblems/mod.rs
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.subproblems import sp1
from ssik.subproblems._aux import (
    cone_polynomials,
    solve_quartic_roots,
    vec_convolve_3,
    vec_self_convolve_2,
    vec_self_convolve_3,
)

__all__ = ["solve"]


def solve(
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
    k1: NDArray[np.float64],
    k2: NDArray[np.float64],
    k3: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[tuple[float, float, float]], bool]:
    """Solve SP5.

    :param policy: tolerances. ``subproblem_numerical`` gates the imaginary-
        part filter on quartic roots and the sign-branch cone-closure check.
    :returns: ``(solutions, is_ls)`` where ``solutions`` is a list of up to
        4 ``(theta1, theta2, theta3)`` tuples. ``is_ls`` is ``True`` if no
        feasible solution was found.
    """
    num_tol = policy.subproblem_numerical

    p1_s = p0 + k1 * float(np.dot(k1, p1))
    p3_s = p2 + k3 * float(np.dot(k3, p3))

    delta1 = float(np.dot(k2, p1_s))
    delta3 = float(np.dot(k2, p3_s))

    p_1, r_1 = cone_polynomials(p0, k1, p1, p1_s, k2)
    p_3, r_3 = cone_polynomials(p2, k3, p3, p3_s, k2)

    p_13 = p_1 - p_3
    p_13_sq = vec_self_convolve_2(p_13)

    rhs = r_3 - r_1 - p_13_sq

    eqn = vec_self_convolve_3(rhs) - 4.0 * vec_convolve_3(p_13_sq, r_1)

    all_roots = solve_quartic_roots(eqn)
    h_vec = np.array([c.real for c in all_roots if abs(c.imag) < num_tol])

    kxp1 = np.cross(k1, p1)
    kxp3 = np.cross(k3, p3)
    a_1 = np.column_stack([kxp1, -np.cross(k1, kxp1)])  # 3x2
    a_3 = np.column_stack([kxp3, -np.cross(k3, kxp3)])

    signs_1 = [1.0, 1.0, -1.0, -1.0]
    signs_3 = [1.0, -1.0, 1.0, -1.0]

    j_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)

    solutions: list[tuple[float, float, float]] = []

    for h in h_vec:
        a1t_k2 = a_1.T @ k2  # shape (2,)
        a3t_k2 = a_3.T @ k2

        const_1 = a1t_k2 * (h - delta1)
        const_3 = a3t_k2 * (h - delta3)

        hd1 = h - delta1
        hd3 = h - delta3

        sq1 = float(np.dot(a1t_k2, a1t_k2)) - hd1 * hd1
        if sq1 < 0.0:
            continue
        sq3 = float(np.dot(a3t_k2, a3t_k2)) - hd3 * hd3
        if sq3 < 0.0:
            continue

        pm_1 = j_mat @ a1t_k2 * float(np.sqrt(sq1))
        pm_3 = j_mat @ a3t_k2 * float(np.sqrt(sq3))

        a1t_k2_ns = float(np.dot(a1t_k2, a1t_k2))
        a3t_k2_ns = float(np.dot(a3t_k2, a3t_k2))

        for s1, s3 in zip(signs_1, signs_3, strict=True):
            sc1 = (const_1 + s1 * pm_1) / a1t_k2_ns
            sc3 = (const_3 + s3 * pm_3) / a3t_k2_ns

            v1 = a_1 @ sc1 + p1_s
            v3 = a_3 @ sc3 + p3_s

            closure = abs(float(np.linalg.norm(v1 - h * k2)) - float(np.linalg.norm(v3 - h * k2)))
            if closure < num_tol:
                # Rot(k2, theta2) * v3 = v1  ->  SP1
                theta2, _ = sp1.solve(k2, v3, v1, policy)
                solutions.append(
                    (
                        float(np.arctan2(sc1[0], sc1[1])),
                        theta2,
                        float(np.arctan2(sc3[0], sc3[1])),
                    )
                )

    if not solutions:
        return [], True
    return solutions, False
