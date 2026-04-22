"""Polynomial and algebraic helpers used by SP5 and SP6.

Ported clean-room from the BSD-3 [ik-geo Rust reference][ikgeo] (Elias & Wen,
arXiv:2211.05737), retaining the algebra and variable names so the port is
auditable against the source.

Private; users should import :mod:`ssik.subproblems.sp5` and
:mod:`ssik.subproblems.sp6` instead of these helpers directly.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/subproblems/auxiliary.rs
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy

__all__ = [
    "cone_polynomials",
    "solve_lower_triangular_system_2x2",
    "solve_quartic_roots",
    "solve_two_ellipse_numeric",
    "vec_convolve_3",
    "vec_self_convolve_2",
    "vec_self_convolve_3",
]


def vec_self_convolve_2(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Coefficient vector of ``(a + b*x)^2``: returns length 3 from length 2."""
    a, b = float(v[0]), float(v[1])
    return np.array([a * a, 2.0 * a * b, b * b], dtype=np.float64)


def vec_self_convolve_3(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Coefficient vector of ``(a + b*x + c*x^2)^2``: length 5 from length 3."""
    a, b, c = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [a * a, 2.0 * a * b, 2.0 * a * c + b * b, 2.0 * b * c, c * c],
        dtype=np.float64,
    )


def vec_convolve_3(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convolution of two length-3 coefficient vectors (product of quadratics).

    Returns length 5: coefficients of ``(a + b*x + c*x^2) * (x + y*x + z*x^2)``.
    """
    a, b, c = float(v1[0]), float(v1[1]), float(v1[2])
    x, y, z = float(v2[0]), float(v2[1]), float(v2[2])
    return np.array(
        [a * x, b * x + a * y, a * z + b * y + c * x, b * z + c * y, c * z],
        dtype=np.float64,
    )


def cone_polynomials(
    p0_i: NDArray[np.float64],
    k_i: NDArray[np.float64],
    p_i: NDArray[np.float64],
    p_i_s: NDArray[np.float64],
    k2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(p, r)`` from the cone intersection step used by SP5.

    See [ik-geo subproblems/auxiliary.rs][src] for the derivation. The
    returned vectors are coefficient polynomials in the subproblem's search
    parameter; SP5 takes the difference of two such cones, squares, and
    root-finds.

    [src]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/subproblems/auxiliary.rs
    """
    ki_x_k2 = np.cross(k_i, k2)
    ki_x_ki_x_k2 = np.cross(k_i, ki_x_k2)
    norm_ki_x_k2_sq = float(np.dot(ki_x_k2, ki_x_k2))

    ki_x_pi = np.cross(k_i, p_i)
    norm_ki_x_pi_sq = float(np.dot(ki_x_pi, ki_x_pi))

    alpha = float(np.dot(p0_i, ki_x_ki_x_k2)) / norm_ki_x_k2_sq
    delta = float(np.dot(k2, p_i_s))
    beta = float(np.dot(p0_i, ki_x_k2)) / norm_ki_x_k2_sq

    p_const = norm_ki_x_pi_sq + float(np.dot(p_i_s, p_i_s)) + 2.0 * alpha * delta
    p = np.array([-2.0 * alpha, p_const], dtype=np.float64)

    r = np.array(
        [-1.0, 2.0 * delta, -delta * delta + norm_ki_x_pi_sq * norm_ki_x_k2_sq],
        dtype=np.float64,
    )
    r = (2.0 * beta) ** 2 * r

    return p, r


def solve_quartic_roots(coeffs: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return the 4 complex roots of ``a*x^4 + b*x^3 + c*x^2 + d*x + e = 0``.

    ``coeffs`` is ``[a, b, c, d, e]`` in decreasing-power order. Degrades
    gracefully to a cubic when the leading coefficient is near zero; delegates
    to :func:`numpy.roots` for numerical stability over the analytic quartic
    formula.
    """
    return np.roots(coeffs).astype(np.complex128)


def solve_lower_triangular_system_2x2(
    l_mat: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Solve ``L @ x = b`` where ``L`` is lower-triangular 2x2."""
    a = float(l_mat[0, 0])
    b0 = float(l_mat[1, 0])
    c = float(l_mat[1, 1])
    p = float(b[0])
    q = float(b[1])
    x0 = p / a
    return np.array([x0, (q - x0 * b0) / c], dtype=np.float64)


def solve_two_ellipse_numeric(
    xm1: NDArray[np.float64],
    xn1: NDArray[np.float64],
    xm2: NDArray[np.float64],
    xn2: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> list[tuple[float, float]]:
    """Return the real intersections of two conics parameterised as
    ``|xm_i + xn_i @ [xi, eta]^T| = 1``.

    Forms the Bezout resultant of the two conics (a quartic in ``y``), solves
    via :func:`solve_quartic_roots`, filters to real roots, and back-substitutes
    ``x`` linearly. Up to four real solution pairs.

    ``policy.subproblem_degeneracy`` gates the imaginary-part filter and the
    near-zero denominator check in the ``x`` back-substitution.

    See [ik-geo subproblems/auxiliary.rs][src] for the coefficient derivation.

    [src]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/subproblems/auxiliary.rs
    """
    epsilon = policy.subproblem_degeneracy

    a_1 = xn1.T @ xn1
    a = float(a_1[0, 0])
    b = 2.0 * float(a_1[1, 0])
    c = float(a_1[1, 1])
    b_1 = 2.0 * (xm1.T @ xn1)
    d = float(b_1[0])
    e = float(b_1[1])
    f = float(xm1 @ xm1) - 1.0

    a_2 = xn2.T @ xn2
    a1 = float(a_2[0, 0])
    b1 = 2.0 * float(a_2[1, 0])
    c1 = float(a_2[1, 1])
    b_2 = 2.0 * (xm2.T @ xn2)
    d1 = float(b_2[0])
    e1 = float(b_2[1])
    fq = float(xm2 @ xm2) - 1.0

    z0 = (
        f * a * d1 * d1
        + a * a * fq * fq
        - d * a * d1 * fq
        + a1 * a1 * f * f
        - 2.0 * a * fq * a1 * f
        - d * d1 * a1 * f
        + a1 * d * d * fq
    )

    z1 = (
        e1 * d * d * a1
        - fq * d1 * a * b
        - 2.0 * a * fq * a1 * e
        - f * a1 * b1 * d
        + 2.0 * d1 * b1 * a * f
        + 2.0 * e1 * fq * a * a
        + d1 * d1 * a * e
        - e1 * d1 * a * d
        - 2.0 * a * e1 * a1 * f
        - f * a1 * d1 * b
        + 2.0 * f * e * a1 * a1
        - fq * b1 * a * d
        - e * a1 * d1 * d
        + 2.0 * fq * b * a1 * d
    )

    z2 = (
        e1 * e1 * a * a
        + 2.0 * c1 * fq * a * a
        - e * a1 * d1 * b
        + fq * a1 * b * b
        - e * a1 * b1 * d
        - fq * b1 * a * b
        - 2.0 * a * e1 * a1 * e
        + 2.0 * d1 * b1 * a * e
        - c1 * d1 * a * d
        - 2.0 * a * c1 * a1 * f
        + b1 * b1 * a * f
        + 2.0 * e1 * b * a1 * d
        + e * e * a1 * a1
        - c * a1 * d1 * d
        - e1 * b1 * a * d
        + 2.0 * f * c * a1 * a1
        - f * a1 * b1 * b
        + c1 * d * d * a1
        + d1 * d1 * a * c
        - e1 * d1 * a * b
        - 2.0 * a * fq * a1 * c
    )

    z3 = (
        -2.0 * a * a1 * c * e1
        + e1 * a1 * b * b
        + 2.0 * c1 * b * a1 * d
        - c * a1 * b1 * d
        + b1 * b1 * a * e
        - e1 * b1 * a * b
        - 2.0 * a * c1 * a1 * e
        - e * a1 * b1 * b
        - c1 * b1 * a * d
        + 2.0 * e1 * c1 * a * a
        + 2.0 * e * c * a1 * a1
        - c * a1 * d1 * b
        + 2.0 * d1 * b1 * a * c
        - c1 * d1 * a * b
    )

    z4 = (
        a * a * c1 * c1
        - 2.0 * a * c1 * a1 * c
        + a1 * a1 * c * c
        - b * a * b1 * c1
        - b * b1 * a1 * c
        + b * b * a1 * c1
        + c * a * b1 * b1
    )

    y_all = solve_quartic_roots(np.array([z4, z3, z2, z1, z0], dtype=np.float64))
    real_mask = np.abs(y_all.imag) < epsilon
    y_real = y_all[real_mask].real

    # Back-substitute x linearly from the first conic equation.
    solutions: list[tuple[float, float]] = []
    for y in y_real:
        num = -((a * c1 * y * y + a * fq) - a1 * c * y * y + a * e1 * y - a1 * e * y - a1 * f)
        den = (a * b1 * y + a * d1) - a1 * b * y - a1 * d
        if abs(den) < epsilon:
            continue
        x = num / den
        solutions.append((float(x), float(y)))
    return solutions
