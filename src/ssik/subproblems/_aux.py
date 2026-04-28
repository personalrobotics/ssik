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
from ssik.subproblems._rotation import _cross3, _dot3

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
    ki_x_k2 = _cross3(k_i, k2)
    ki_x_ki_x_k2 = _cross3(k_i, ki_x_k2)
    norm_ki_x_k2_sq = _dot3(ki_x_k2, ki_x_k2)

    ki_x_pi = _cross3(k_i, p_i)
    norm_ki_x_pi_sq = _dot3(ki_x_pi, ki_x_pi)

    alpha = _dot3(p0_i, ki_x_ki_x_k2) / norm_ki_x_k2_sq
    delta = _dot3(k2, p_i_s)
    beta = _dot3(p0_i, ki_x_k2) / norm_ki_x_k2_sq

    p_const = norm_ki_x_pi_sq + _dot3(p_i_s, p_i_s) + 2.0 * alpha * delta
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
    to :func:`numpy.roots`. Returns an empty complex array if the coefficient
    magnitudes overflow the companion-matrix computation (callers treat this
    as "no real roots, signal LS").
    """
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.roots(coeffs).astype(np.complex128)
    except np.linalg.LinAlgError:
        return np.array([], dtype=np.complex128)


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


def _quadratic_real_roots(a: float, b: float, c: float, tol: float) -> list[float]:
    """Real roots of ``a x^2 + b x + c = 0``; linear or empty if degenerate."""
    if abs(a) < tol:
        if abs(b) < tol:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < -tol:
        return []
    root_disc = float(np.sqrt(max(0.0, disc)))
    return [(-b + root_disc) / (2.0 * a), (-b - root_disc) / (2.0 * a)]


def _solve_two_ellipse_degenerate(
    coef1: tuple[float, float, float, float, float, float],
    coef2: tuple[float, float, float, float, float, float],
    tol: float,
    *,
    primary: str,
) -> list[tuple[float, float]]:
    """Intersection when one conic decouples to a 1-variable quadratic.

    ``coef_i = (a, b, c, d, e, f)`` for ``a x^2 + b x y + c y^2 + d x + e y + f = 0``.
    ``primary='y'`` means ``coef1`` is a conic in y alone (a=b=d=0, solve for y);
    ``primary='x'`` means it is in x alone (b=c=e=0, solve for x).
    For each root of the 1-variable conic, substitute into ``coef2`` and solve
    the resulting quadratic in the other variable.
    """
    if primary == "y":
        _a, _b, c, _d, e, f = coef1
        primary_roots = _quadratic_real_roots(c, e, f, tol)
        a2, b2, c2, d2, e2, f2 = coef2
        solutions: list[tuple[float, float]] = []
        for y in primary_roots:
            # a2 x^2 + (b2 y + d2) x + (c2 y^2 + e2 y + f2) = 0
            other_roots = _quadratic_real_roots(a2, b2 * y + d2, c2 * y * y + e2 * y + f2, tol)
            for x in other_roots:
                solutions.append((float(x), float(y)))
        return solutions

    # primary == "x"
    a, _b, _c, d, _e, f = coef1
    primary_roots = _quadratic_real_roots(a, d, f, tol)
    a2, b2, c2, d2, e2, f2 = coef2
    solutions = []
    for x in primary_roots:
        other_roots = _quadratic_real_roots(c2, b2 * x + e2, a2 * x * x + d2 * x + f2, tol)
        for y in other_roots:
            solutions.append((float(x), float(y)))
    return solutions


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

    **Degenerate-conic fallback**: when one of the null-space matrices is
    rank-deficient (e.g. when a caller-level collinearity in SP6 makes one
    of the two unit-circle constraints decouple into a 1D equation), the
    standard ``x`` back-substitution divides 0 by 0. We detect that case
    via the conic coefficients and fall back to solving each quadratic
    separately, substituting between them. This is strictly beyond the
    upstream [ik-geo Rust reference][src], which silently produces NaN on
    this input class.

    ``policy.subproblem_degeneracy`` gates the imaginary-part filter and the
    near-zero denominator check in the ``x`` back-substitution.

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

    # Degenerate-ellipse fallbacks: when one conic decouples to a 1-variable
    # quadratic, the Bezout-resultant back-substitution divides 0/0. Solve
    # the decoupled constraint first and substitute into the full conic.
    coef1 = (a, b, c, d, e, f)
    coef2 = (a1, b1, c1, d1, e1, fq)
    if abs(a) < epsilon and abs(b) < epsilon and abs(d) < epsilon:
        return _solve_two_ellipse_degenerate(coef1, coef2, epsilon, primary="y")
    if abs(c) < epsilon and abs(b) < epsilon and abs(e) < epsilon:
        return _solve_two_ellipse_degenerate(coef1, coef2, epsilon, primary="x")
    if abs(a1) < epsilon and abs(b1) < epsilon and abs(d1) < epsilon:
        return _solve_two_ellipse_degenerate(coef2, coef1, epsilon, primary="y")
    if abs(c1) < epsilon and abs(b1) < epsilon and abs(e1) < epsilon:
        # Coef2 is x-only; solve it for x, substitute into coef1 for y.
        # We need the output to be (x, y), not (y, x), so map accordingly.
        swapped = _solve_two_ellipse_degenerate(coef2, coef1, epsilon, primary="x")
        return [(x, y) for x, y in swapped]

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
