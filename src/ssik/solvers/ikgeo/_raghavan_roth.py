"""Raghavan-Roth (P, Q) builder for the general 6R IK solver.

This module is private. It derives the 14x9 elimination matrix ``P`` and 14x8
elimination matrix ``Q`` symbolically (per arm, with numeric DH parameters
substituted) and emits a pure NumPy callable taking ``T_target`` to
``(P_sin, P_cos, P_one, Q)``. The eigenvalue route in :mod:`general_6r`
consumes those four matrices to construct ``M(x_2) = A x_2^2 + B x_2 + C``.

References (clean-room from open-access pubs only):

- Tsai, "Robot Analysis," Wiley, 1999. Appendix C reproduces the full
  Raghavan-Roth derivation: the 14-equation system, the elimination of
  ``(theta_0, theta_1)``, and the final 12x12 matrix.
- Manocha & Canny, "Efficient inverse kinematics for general 6R manipulators,"
  IEEE T-RA 10(5):648-657, October 1994. Eq. 4 specifies the monomial
  ordering used for ``P`` and ``Q``; Section IV gives the eigenvalue route.

LGPL note: this file is clean-room from Tsai App. C and Manocha-Canny 1994
only. The vendored ``ikfast.py:solveDialytically`` was read for algorithmic
existence proof only; do not copy structure or naming patterns from it.

Algorithmic specifics chosen here:

- Convention: standard DH ``A_i = R_z(theta_i) T_z(d_i) T_x(a_i) R_x(alpha_i)``,
  joints 0-indexed.
- Loop closure split: ``A_2 A_3 A_4 = A_1^{-1} A_0^{-1} T A_5^{-1}``.
  Cols 2 and 3 of both sides are q_5-free (``A_5^{-1}`` ends with
  ``R_z(-q_5)`` which postmultiplies, leaving cols 2,3 unchanged).
- We compute DH inverses in *closed form* rather than via ``sympy.inv()`` to
  avoid spurious ``1/(s^2 + c^2)`` factors that block polynomial reduction.
- Per-arm caching: the symbolic derivation is keyed on the (immutable) DH
  tuple. First call ~10-30 s; subsequent calls reuse the lambdified callable.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
import sympy as sp
from numpy.typing import NDArray

__all__ = [
    "back_substitute",
    "build_m_matrix",
    "build_pq",
    "eliminate_q0_q1",
    "solve_all_ik",
    "solve_x2_roots",
    "weierstrass_eliminate_trig",
]


DhParams = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]


# Monomial ordering -----------------------------------------------------------
#
# Left 9-vector (in q_3, q_4):
#   v_left(q_3, q_4) = (s_3 s_4, s_3 c_4, c_3 s_4, c_3 c_4, s_3, c_3, s_4, c_4, 1)
#
# Right 8-vector (in q_0, q_1):
#   v_right(q_0, q_1) = (s_0 s_1, s_0 c_1, c_0 s_1, c_0 c_1, s_0, c_0, s_1, c_1)
#
# (Manocha-Canny Eq. 4 ordering. We omit the trailing "1" from v_right: any
# constant term in the right side is moved into v_left's "1" column.)
#
# Each row of the 14-equation system is bilinear in
#   (s_2, c_2, 1) x v_left(q_3, q_4) - constant x v_right(q_0, q_1).


def _v_left(s3: float, c3: float, s4: float, c4: float) -> NDArray[np.float64]:
    return np.array([s3 * s4, s3 * c4, c3 * s4, c3 * c4, s3, c3, s4, c4, 1.0])


def _v_right(s0: float, c0: float, s1: float, c1: float) -> NDArray[np.float64]:
    return np.array([s0 * s1, s0 * c1, c0 * s1, c0 * c1, s0, c0, s1, c1])


# ---------------------------------------------------------------------------
# Symbolic DH transforms (with numeric DH params substituted at derive time).
# ---------------------------------------------------------------------------


def _dh_matrix_sym(s_q: sp.Symbol, c_q: sp.Symbol, alpha: float, a: float, d: float) -> sp.Matrix:
    """Standard DH: A = R_z(theta) T_z(d) T_x(a) R_x(alpha), numeric DH params."""
    ca = float(np.cos(alpha))
    sa = float(np.sin(alpha))
    return sp.Matrix(
        [
            [c_q, -s_q * ca, s_q * sa, a * c_q],
            [s_q, c_q * ca, -c_q * sa, a * s_q],
            [0, sa, ca, d],
            [0, 0, 0, 1],
        ]
    )


def _dh_matrix_inv_sym(s_q: sp.Symbol, c_q: sp.Symbol, alpha: float, a: float, d: float) -> sp.Matrix:
    """Closed-form A^{-1} for standard DH. Avoids sympy.inv()'s 1/(s^2+c^2) artifacts.

    Derivation: A = R_z T_z T_x R_x, so A^{-1} = R_x^T T_x^{-1} T_z^{-1} R_z^T.
    Using rigid-motion inverse [R^T, -R^T t; 0, 1] and the identity
    s_q^2 + c_q^2 = 1 simplifies the translation to (-a, -d sa, -d ca).
    """
    ca = float(np.cos(alpha))
    sa = float(np.sin(alpha))
    return sp.Matrix(
        [
            [c_q, s_q, 0, -a],
            [-s_q * ca, c_q * ca, sa, -d * sa],
            [s_q * sa, -c_q * sa, ca, -d * ca],
            [0, 0, 0, 1],
        ]
    )


# ---------------------------------------------------------------------------
# Per-arm symbolic derivation (cached by DH tuple).
# ---------------------------------------------------------------------------


def _reduce_trig(expr: sp.Expr, s_syms: tuple[sp.Symbol, ...], c_syms: tuple[sp.Symbol, ...]) -> sp.Expr:
    """Reduce ``expr`` modulo the trig ideal {s_i^2 + c_i^2 - 1 : i}.

    Sympy's ``sp.reduced`` does this canonically. We use it directly; no
    iterated substitution gymnastics. The result is multilinear in each
    (s_i, c_i) pair (degree <= 1 in each variable, after using s_i^2 +
    c_i^2 = 1 to eliminate squared terms).
    """
    basis = [s_syms[i] ** 2 + c_syms[i] ** 2 - 1 for i in range(len(s_syms))]
    all_gens = list(s_syms) + list(c_syms)
    _, remainder = sp.reduced(expr, basis, *all_gens)
    return sp.expand(remainder)


def _derive_pq_for_arm(
    alpha: tuple[float, ...],
    a: tuple[float, ...],
    d: tuple[float, ...],
) -> tuple[Callable[..., NDArray[np.float64]], Callable[..., NDArray[np.float64]], Callable[..., NDArray[np.float64]], Callable[..., NDArray[np.float64]]]:
    """Derive 14-row Raghavan-Roth (P, Q) callables for a specific arm.

    DH params are numeric; T_target stays symbolic. Output: four callables
    that each take 12 entries of T_target (rows 0-2) and return either a
    14x9 matrix (the three P factors) or a 14x8 matrix (Q).
    """
    if len(alpha) != 6 or len(a) != 6 or len(d) != 6:
        raise ValueError(f"DH must have 6 entries per array; got {len(alpha)}, {len(a)}, {len(d)}")

    # 6 joint angles -> sin/cos symbols. Joint 5 isn't in the polynomial we'll
    # extract from (cols 2/3 cancel q_5), but we need it to write A_5^{-1}.
    s = sp.symbols("s0:6", real=True)
    c = sp.symbols("c0:6", real=True)
    s_active = s[:5]  # s_0..s_4 (joint 5 trig is q_5-free in cols 2,3)
    c_active = c[:5]

    # Symbolic T_target entries (top 3 rows free; row 3 = [0, 0, 0, 1]).
    T_syms = sp.symbols("T_:12", real=True)
    T_mat = sp.Matrix(
        [
            [T_syms[0], T_syms[1], T_syms[2], T_syms[3]],
            [T_syms[4], T_syms[5], T_syms[6], T_syms[7]],
            [T_syms[8], T_syms[9], T_syms[10], T_syms[11]],
            [0, 0, 0, 1],
        ]
    )

    # Per-joint DH matrices (numeric DH).
    A_dh = [_dh_matrix_sym(s[i], c[i], alpha[i], a[i], d[i]) for i in range(6)]
    A_inv = [_dh_matrix_inv_sym(s[i], c[i], alpha[i], a[i], d[i]) for i in range(6)]

    # Loop closure split: A_2 A_3 A_4 = A_1^{-1} A_0^{-1} T A_5^{-1}.
    lhs_mat = A_dh[2] * A_dh[3] * A_dh[4]
    rhs_mat = A_inv[1] * A_inv[0] * T_mat * A_inv[5]

    # l = col 2 (z-axis after the chain), p = col 3 (translation).
    # Both sides' cols 2 and 3 are q_5-free by construction (A_5^{-1} ends in
    # R_z(-q_5) which only mixes cols 0,1 on postmultiplication).
    l_lhs = sp.Matrix([lhs_mat[r, 2] for r in range(3)])
    l_rhs = sp.Matrix([rhs_mat[r, 2] for r in range(3)])
    p_lhs = sp.Matrix([lhs_mat[r, 3] for r in range(3)])
    p_rhs = sp.Matrix([rhs_mat[r, 3] for r in range(3)])

    # 14 RR equations.
    eqs: list[sp.Expr] = []
    # 0-2: l match
    for i in range(3):
        eqs.append(sp.expand(l_lhs[i] - l_rhs[i]))
    # 3-5: p match
    for i in range(3):
        eqs.append(sp.expand(p_lhs[i] - p_rhs[i]))
    # 6-8: l x p moment identity
    lxp_lhs = l_lhs.cross(p_lhs)
    lxp_rhs = l_rhs.cross(p_rhs)
    for i in range(3):
        eqs.append(sp.expand(lxp_lhs[i] - lxp_rhs[i]))
    # 9: p . p
    pp_lhs = sp.expand(p_lhs.dot(p_lhs))
    pp_rhs = sp.expand(p_rhs.dot(p_rhs))
    eqs.append(sp.expand(pp_lhs - pp_rhs))
    # 10: l . p
    lp_lhs = sp.expand(l_lhs.dot(p_lhs))
    lp_rhs = sp.expand(l_rhs.dot(p_rhs))
    eqs.append(sp.expand(lp_lhs - lp_rhs))
    # 11-13: (p.p) l - 2 (l.p) p
    pplxx_lhs = pp_lhs * l_lhs - 2 * lp_lhs * p_lhs
    pplxx_rhs = pp_rhs * l_rhs - 2 * lp_rhs * p_rhs
    for i in range(3):
        eqs.append(sp.expand(pplxx_lhs[i] - pplxx_rhs[i]))

    # Reduce each modulo {s_i^2 + c_i^2 - 1 : i = 0..4}.
    # After reduction every equation is multilinear in (s_i, c_i) (degree <= 1
    # in each), so coeff_monomial against the (s_2, c_2, 1) x v_left and
    # v_right basis captures every term.
    reduced_eqs = [_reduce_trig(eq, s_active, c_active) for eq in eqs]

    # Extract coefficients ---------------------------------------------------
    left_9 = [
        s[3] * s[4],
        s[3] * c[4],
        c[3] * s[4],
        c[3] * c[4],
        s[3],
        c[3],
        s[4],
        c[4],
        sp.Integer(1),
    ]
    right_8 = [
        s[0] * s[1],
        s[0] * c[1],
        c[0] * s[1],
        c[0] * c[1],
        s[0],
        c[0],
        s[1],
        c[1],
    ]

    n_rows = len(reduced_eqs)  # 14
    p_sin_sym: list[list[sp.Expr]] = [[sp.Integer(0)] * 9 for _ in range(n_rows)]
    p_cos_sym: list[list[sp.Expr]] = [[sp.Integer(0)] * 9 for _ in range(n_rows)]
    p_one_sym: list[list[sp.Expr]] = [[sp.Integer(0)] * 9 for _ in range(n_rows)]
    q_sym: list[list[sp.Expr]] = [[sp.Integer(0)] * 8 for _ in range(n_rows)]

    poly_gens = (*s_active, *c_active)
    for r, eq in enumerate(reduced_eqs):
        poly = sp.Poly(eq, *poly_gens)
        for j, mon in enumerate(left_9):
            p_sin_sym[r][j] = poly.coeff_monomial(sp.expand(s[2] * mon))
            p_cos_sym[r][j] = poly.coeff_monomial(sp.expand(c[2] * mon))
            p_one_sym[r][j] = poly.coeff_monomial(mon)
        for j, mon in enumerate(right_8):
            # eq = LHS - RHS; right monomials live in -RHS.
            q_sym[r][j] = -poly.coeff_monomial(mon)

    # Lambdify with T_target entries as args.
    p_sin_fn = sp.lambdify(T_syms, sp.Matrix(p_sin_sym), "numpy")
    p_cos_fn = sp.lambdify(T_syms, sp.Matrix(p_cos_sym), "numpy")
    p_one_fn = sp.lambdify(T_syms, sp.Matrix(p_one_sym), "numpy")
    q_fn = sp.lambdify(T_syms, sp.Matrix(q_sym), "numpy")
    return p_sin_fn, p_cos_fn, p_one_fn, q_fn


# Cache the per-arm derivation. Keys are immutable DH tuples.
@lru_cache(maxsize=64)
def _cached_derivation(
    alpha: tuple[float, ...],
    a: tuple[float, ...],
    d: tuple[float, ...],
) -> tuple[Callable[..., NDArray[np.float64]], Callable[..., NDArray[np.float64]], Callable[..., NDArray[np.float64]], Callable[..., NDArray[np.float64]]]:
    return _derive_pq_for_arm(alpha, a, d)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def build_pq(
    dh: DhParams,
    t_target: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the (factored) Raghavan-Roth elimination matrices.

    :param dh: Tuple ``(alpha, a, d)`` of length-6 numpy arrays giving the
        standard DH parameters per joint. Convention:
        ``A_i = R_z(theta_i) T_z(d_i) T_x(a_i) R_x(alpha_i)``.
    :param t_target: 4x4 target end-effector pose in the base frame.
    :returns: ``(P_sin, P_cos, P_one, Q)`` -- each P matrix is 14x9, Q is
        14x8 (the full 14-row Raghavan-Roth system: 6 base column equations +
        8 vector-identity rows). At a valid IK solution ``(q_0, ..., q_4)``:

            (P_sin[i] s_2 + P_cos[i] c_2 + P_one[i]) . v_left(q_3, q_4)
            == Q[i] . v_right(q_0, q_1).

    First call for a given DH set takes 10-30 s (symbolic derivation +
    polynomial reduction); subsequent calls with the same DH reuse the cache.
    """
    alpha, a, d = dh
    if alpha.shape != (6,) or a.shape != (6,) or d.shape != (6,):
        raise ValueError(f"DH params must be length-6 arrays; got {alpha.shape}, {a.shape}, {d.shape}")
    t = np.asarray(t_target, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"t_target must be 4x4; got {t.shape}")

    p_sin_fn, p_cos_fn, p_one_fn, q_fn = _cached_derivation(
        tuple(alpha.tolist()),
        tuple(a.tolist()),
        tuple(d.tolist()),
    )

    args = (
        *t[0, :].tolist(),
        *t[1, :].tolist(),
        *t[2, :].tolist(),
    )
    p_sin = np.asarray(p_sin_fn(*args), dtype=np.float64)
    p_cos = np.asarray(p_cos_fn(*args), dtype=np.float64)
    p_one = np.asarray(p_one_fn(*args), dtype=np.float64)
    q = np.asarray(q_fn(*args), dtype=np.float64)
    return p_sin, p_cos, p_one, q


def eliminate_q0_q1(
    p_sin: NDArray[np.float64],
    p_cos: NDArray[np.float64],
    p_one: NDArray[np.float64],
    q_mat: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Eliminate the v_right(q_0, q_1) monomials via the left null space of Q.

    Q is 14x8. For a generic 6R chain it has full rank 8. Its left null space
    is 14 - 8 = 6 dimensional; let ``N`` be a 14x6 orthonormal basis. Then
    ``N^T Q = 0``, and multiplying the 14-row system from the left by ``N^T``
    annihilates v_right:

        (N^T P_sin s_2 + N^T P_cos c_2 + N^T P_one) . v_left = 0

    yielding 6 equations in (q_2, q_3, q_4) only -- Tsai's E (Eq. C.9).

    For arms where ``rank(Q) < 8`` (Pieper-class with collapsed monomials)
    the null space is larger; we still take the trailing 6 columns of the
    SVD's left singular matrix, which gives a well-conditioned 6-equation
    subset.

    :returns: ``(E_sin, E_cos, E_one)`` -- each 6x9. At a valid IK solution
        ``(q_2, q_3, q_4)``,

            (E_sin s_2 + E_cos c_2 + E_one) @ v_left(q_3, q_4) == 0.
    """
    if q_mat.shape != (14, 8):
        raise ValueError(f"Q must be 14x8; got {q_mat.shape}")

    u_full, _, _ = np.linalg.svd(q_mat, full_matrices=True)
    n_basis = u_full[:, -6:]  # 14x6 basis for the left null space

    e_sin = n_basis.T @ p_sin
    e_cos = n_basis.T @ p_cos
    e_one = n_basis.T @ p_one
    return e_sin, e_cos, e_one


# ---------------------------------------------------------------------------
# Weierstrass substitution for q_2, q_3, q_4.
# ---------------------------------------------------------------------------
#
# Polynomial basis for v_left after Weierstrass for (q_3, q_4) and clearing
# (1+x_3^2)(1+x_4^2):
#
#   v_left_x = (x_3^2 x_4^2, x_3^2 x_4, x_3^2, x_3 x_4^2, x_3 x_4, x_3,
#               x_4^2, x_4, 1)         -- 9 monomials, indices 0..8
#
# The 9x9 transform W maps v_left_trig (the original sin/cos basis) to v_left_x
# (after multiplying both sides by (1+x_3^2)(1+x_4^2)):
#
#   v_left_trig * (1+x_3^2)(1+x_4^2) = W @ v_left_x
#
# Each row of W lists the coefficients of one trig monomial in terms of v_left_x.
# Derived by hand from the Weierstrass identities (see comments per row).

_W_TRIG_TO_X = np.array(
    [
        # row k = expansion of v_left_trig[k] * (1+x_3^2)(1+x_4^2)
        # in basis (x_3^2 x_4^2, x_3^2 x_4, x_3^2, x_3 x_4^2, x_3 x_4, x_3,
        #           x_4^2,    x_4,    1)
        [0, 0, 0, 0, 4, 0, 0, 0, 0],  # s_3 s_4 -> 4 x_3 x_4
        [0, 0, 0, -2, 0, 2, 0, 0, 0],  # s_3 c_4 -> 2 x_3 - 2 x_3 x_4^2
        [0, -2, 0, 0, 0, 0, 0, 2, 0],  # c_3 s_4 -> 2 x_4 - 2 x_3^2 x_4
        [1, 0, -1, 0, 0, 0, -1, 0, 1],  # c_3 c_4 -> 1 - x_3^2 - x_4^2 + x_3^2 x_4^2
        [0, 0, 0, 2, 0, 2, 0, 0, 0],  # s_3 -> 2 x_3 (1+x_4^2) = 2 x_3 + 2 x_3 x_4^2
        [-1, 0, -1, 0, 0, 0, 1, 0, 1],  # c_3 -> 1 - x_3^2 + x_4^2 - x_3^2 x_4^2
        [0, 2, 0, 0, 0, 0, 0, 2, 0],  # s_4 -> 2 x_4 (1+x_3^2) = 2 x_4 + 2 x_3^2 x_4
        [-1, 0, 1, 0, 0, 0, -1, 0, 1],  # c_4 -> 1 + x_3^2 - x_4^2 - x_3^2 x_4^2
        [1, 0, 1, 0, 0, 0, 1, 0, 1],  # 1   -> 1 + x_3^2 + x_4^2 + x_3^2 x_4^2
    ],
    dtype=np.float64,
)


def _v_left_x(x3: float, x4: float) -> NDArray[np.float64]:
    """Polynomial basis (x_3^2 x_4^2, ..., 1) at numeric (x_3, x_4)."""
    return np.array(
        [
            x3 * x3 * x4 * x4,
            x3 * x3 * x4,
            x3 * x3,
            x3 * x4 * x4,
            x3 * x4,
            x3,
            x4 * x4,
            x4,
            1.0,
        ]
    )


def weierstrass_eliminate_trig(
    e_sin: NDArray[np.float64],
    e_cos: NDArray[np.float64],
    e_one: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Substitute Weierstrass half-angle for q_2, q_3, q_4 in the 6-equation E system.

    Input: ``E_sin, E_cos, E_one`` each 6x9 (Tsai's E in the v_left_trig basis,
    after eliminating q_0, q_1).

    Substitution:
      - For q_2: ``s_2 = 2 x_2 / (1+x_2^2)``, ``c_2 = (1-x_2^2) / (1+x_2^2)``.
        Multiply each row by ``(1+x_2^2)``. Coefficients re-group as a quadratic
        polynomial in x_2:
            E(x_2) = (E_one - E_cos) x_2^2 + (2 E_sin) x_2 + (E_one + E_cos)
      - For (q_3, q_4): apply the W transform to switch v_left_trig into the
        polynomial basis v_left_x. Multiplied by (1+x_3^2)(1+x_4^2) on both sides.

    :returns: ``(E_quad, E_lin, E_const)`` -- each 6x9 in the v_left_x basis.
        At a valid IK solution ``(q_2, q_3, q_4)`` with x_i = tan(q_i/2):

            (E_quad x_2^2 + E_lin x_2 + E_const) @ v_left_x(x_3, x_4) == 0.
    """
    e_quad_trig = e_one - e_cos
    e_lin_trig = 2.0 * e_sin
    e_const_trig = e_one + e_cos

    e_quad = e_quad_trig @ _W_TRIG_TO_X
    e_lin = e_lin_trig @ _W_TRIG_TO_X
    e_const = e_const_trig @ _W_TRIG_TO_X
    return e_quad, e_lin, e_const


# ---------------------------------------------------------------------------
# 12x12 matrix polynomial M(x_2).
# ---------------------------------------------------------------------------
#
# 12-vector for the doubled system:
#
#   v_12 = (v_left_x[0..8],
#           x_3^3 x_4^2, x_3^3 x_4, x_3^3)            -- 9 + 3 = 12
#
# The doubled system stacks the 6 "E @ v_left_x = 0" equations on top with
# 6 equations from "E @ (x_3 * v_left_x) = 0" on the bottom. Multiplying
# v_left_x by x_3 maps each entry to a 12-vec column:
#
#   x_3 * v_left_x[0] = x_3^3 x_4^2     -> 12-vec col 9 (NEW)
#   x_3 * v_left_x[1] = x_3^3 x_4       -> 12-vec col 10 (NEW)
#   x_3 * v_left_x[2] = x_3^3            -> 12-vec col 11 (NEW)
#   x_3 * v_left_x[3] = x_3^2 x_4^2     -> 12-vec col 0  (existing)
#   x_3 * v_left_x[4] = x_3^2 x_4       -> 12-vec col 1
#   x_3 * v_left_x[5] = x_3^2            -> 12-vec col 2
#   x_3 * v_left_x[6] = x_3 x_4^2       -> 12-vec col 3
#   x_3 * v_left_x[7] = x_3 x_4          -> 12-vec col 4
#   x_3 * v_left_x[8] = x_3              -> 12-vec col 5


def _v_12(x3: float, x4: float) -> NDArray[np.float64]:
    """The 12-monomial vector v_12 at numeric (x_3, x_4)."""
    return np.array(
        [
            x3 * x3 * x4 * x4,  # 0: x_3^2 x_4^2
            x3 * x3 * x4,  # 1: x_3^2 x_4
            x3 * x3,  # 2: x_3^2
            x3 * x4 * x4,  # 3: x_3 x_4^2
            x3 * x4,  # 4: x_3 x_4
            x3,  # 5: x_3
            x4 * x4,  # 6: x_4^2
            x4,  # 7: x_4
            1.0,  # 8: 1
            x3 * x3 * x3 * x4 * x4,  # 9: x_3^3 x_4^2 (NEW)
            x3 * x3 * x3 * x4,  # 10: x_3^3 x_4 (NEW)
            x3 * x3 * x3,  # 11: x_3^3 (NEW)
        ]
    )


def _embed_e_into_m(e: NDArray[np.float64]) -> NDArray[np.float64]:
    """Embed a 6x9 E into the 12x12 block structure for M.

    Top 6 rows: ``E @ v_left_x = 0`` -> [E | 0_{6x3}]  (cols 0-8 = E, cols 9-11 = 0)
    Bottom 6 rows: ``E @ (x_3 * v_left_x) = 0`` -> shifted columns per the
    monomial map above.
    """
    m = np.zeros((12, 12), dtype=np.float64)
    # Top: E in cols 0-8, zeros in cols 9-11
    m[:6, :9] = e
    # Bottom: cols 0-5 get E[:, 3:9]; cols 6-8 stay zero; cols 9-11 get E[:, 0:3]
    m[6:, 0:6] = e[:, 3:9]
    m[6:, 9:12] = e[:, 0:3]
    return m


def build_m_matrix(
    e_quad: NDArray[np.float64],
    e_lin: NDArray[np.float64],
    e_const: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build the 12x12 matrix polynomial ``M(x_2) = M_quad x_2^2 + M_lin x_2 + M_const``.

    Inputs are the three 6x9 coefficient matrices from
    :func:`weierstrass_eliminate_trig`. Outputs are the three 12x12 matrices
    obtained by stacking the base equations and their x_3-shifted versions.

    At a valid IK solution ``(q_2, q_3, q_4)`` with ``x_i = tan(q_i/2)``:

        (M_quad x_2^2 + M_lin x_2 + M_const) @ v_12(x_3, x_4) == 0.

    The 16 finite roots of ``det M(x_2) = 0`` are the 16 candidate
    ``tan(q_2/2)`` values for the IK problem. The eigenvalue route in
    :mod:`general_6r` operates on this matrix polynomial.
    """
    if e_quad.shape != (6, 9) or e_lin.shape != (6, 9) or e_const.shape != (6, 9):
        raise ValueError(
            f"E matrices must be 6x9; got {e_quad.shape}, {e_lin.shape}, {e_const.shape}"
        )
    return _embed_e_into_m(e_quad), _embed_e_into_m(e_lin), _embed_e_into_m(e_const)


# ---------------------------------------------------------------------------
# 24x24 companion eigenvalue route -> the 16 candidate tan(q_2/2) values.
# ---------------------------------------------------------------------------


def solve_x2_roots(
    m_quad: NDArray[np.float64],
    m_lin: NDArray[np.float64],
    m_const: NDArray[np.float64],
    *,
    spurious_tol: float = 0.1,
    imag_rel_tol: float = 1e-3,
    cond_threshold: float = 1e10,
) -> tuple[list[float], list[NDArray[np.complex128]]]:
    """Compute the real ``tan(q_2/2)`` roots of ``det M(x_2) = 0`` via
    24x24 companion eigenvalue (Manocha-Canny Theorem 1).

    Builds the companion matrix

        Sigma = [[ 0_{12},   I_{12}     ],
                 [ -A^{-1}C, -A^{-1}B    ]]   (24x24)

    where ``A = m_quad``, ``B = m_lin``, ``C = m_const``. Sigma's 24
    eigenvalues correspond to the roots of ``det(M(x_2)) = 0``. The construction
    introduces 8 spurious roots clustered near ``+/-i`` (multiplicity 4 each,
    from a ``(1 + x_2^2)^4`` factor in the determinant); we filter those out.

    The eigenvectors of Sigma corresponding to ``x_2 = lambda`` have block
    structure ``[v_12; lambda * v_12]``; we return the eigenvectors so the
    back-substitution stage can recover ``(x_3, x_4)``.

    :param spurious_tol: width of the near-i / near--i exclusion band.
    :param imag_rel_tol: scale-aware tolerance for accepting an eigenvalue as
        real-valued (analogous to SP5/SP6 imag filter).
    :param cond_threshold: above this ``cond(m_quad)``, raise -- caller must
        invoke the M\u00f6bius-reparameterization or generalized-eigenvalue path
        (Day 4 fallback).

    :returns: ``(real_roots, eigvecs)`` -- list of real x_2 values and matching
        24-component eigenvectors. ``len(real_roots) <= 16`` (fewer if some
        roots are complex-conjugate pairs).

    :raises numpy.linalg.LinAlgError: if ``m_quad`` is too ill-conditioned for
        the standard eigenvalue route.
    """
    cond = float(np.linalg.cond(m_quad))
    if cond > cond_threshold:
        raise np.linalg.LinAlgError(
            f"M_quad ill-conditioned (cond={cond:.3e}); generalized-eigenvalue fallback required"
        )

    a_inv_b = np.linalg.solve(m_quad, m_lin)
    a_inv_c = np.linalg.solve(m_quad, m_const)

    sigma = np.zeros((24, 24), dtype=np.float64)
    sigma[:12, 12:] = np.eye(12)
    sigma[12:, :12] = -a_inv_c
    sigma[12:, 12:] = -a_inv_b

    eigvals, eigvecs = np.linalg.eig(sigma)

    real_roots: list[float] = []
    real_eigvecs: list[NDArray[np.complex128]] = []
    for k in range(24):
        ev = eigvals[k]
        # Filter spurious roots near +/-i: |Re| small and |Im| ~ 1.
        if abs(abs(ev.imag) - 1.0) < spurious_tol and abs(ev.real) < spurious_tol:
            continue
        # Filter complex roots (no real IK solution): |Im| > tol * max(|Re|, 1).
        if abs(ev.imag) > imag_rel_tol * max(abs(ev.real), 1.0):
            continue
        real_roots.append(float(ev.real))
        real_eigvecs.append(eigvecs[:, k])
    return real_roots, real_eigvecs


# ---------------------------------------------------------------------------
# Back-substitution: eigenvector -> (x_3, x_4) -> (q_0, q_1, q_5).
# ---------------------------------------------------------------------------


def _dh_matrix_num(theta: float, alpha: float, a: float, d: float) -> NDArray[np.float64]:
    """Numeric DH transform matrix at a specific joint angle."""
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    ca, sa = float(np.cos(alpha)), float(np.sin(alpha))
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def back_substitute(
    x2: float,
    eigvec_24: NDArray[np.complex128],
    p_sin: NDArray[np.float64],
    p_cos: NDArray[np.float64],
    p_one: NDArray[np.float64],
    q_mat: NDArray[np.float64],
    dh: DhParams,
    t_target: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    """Recover ``(q_0, ..., q_5)`` from a real ``x_2`` root and its eigenvector.

    Algorithm (Manocha-Canny IV-C/IV-D):

    1. From x_2: ``q_2 = 2 atan(x_2)``.
    2. From eigenvector: V = ``[v_12; x_2 * v_12]`` (top + bottom 12). Pick the
       half with smaller relative error: top if |x_2| <= 1, else bottom.
    3. From v_12: ``x_3 = v_12[5] / v_12[8]`` and ``x_4 = v_12[7] / v_12[8]``
       (reading the canonical entries: v_12[5] = x_3, v_12[7] = x_4, v_12[8] = 1).
       If ``v_12[8]`` is small, fall back to higher-magnitude ratios.
    4. ``q_3 = 2 atan(x_3)``, ``q_4 = 2 atan(x_4)``.
    5. Solve the original 14x8 ``Q v_right = P_eff @ v_left`` for v_right via
       LSQ; recover ``(q_0, q_1)`` from ``v_right[5:8]`` via atan2 (with sign
       cross-checks against the bilinear entries v_right[0:4]).
    6. Recover ``q_5``: compute ``A_5_residual = FK(q_0..q_4)^{-1} @ T_target``,
       then ``q_5 = atan2(A_5_residual[1, 0], A_5_residual[0, 0])`` from the
       R_z(q_5) factor at the start of A_5.

    Returns ``None`` if any step fails (denominator collapse, sign mismatch,
    etc.). Caller filters those out.
    """
    alpha, a, d = dh

    q2 = 2.0 * np.arctan(x2)

    # Eigenvector structure: V = [v_12; x_2 * v_12]; pick the better-conditioned half.
    if abs(x2) <= 1.0:
        v_12 = eigvec_24[:12]
    else:
        v_12 = eigvec_24[12:] / x2
    # Real part (eigenvalue is real, so v_12 should be real up to global complex scale).
    v_12 = np.real(v_12)

    # Normalize so v_12[8] (the "1" monomial entry) has a definite scale.
    norm_idx = 8  # canonical "1" entry
    den = float(v_12[norm_idx])
    if abs(den) < 1e-9:
        # v_12[8] (the "1") collapsed -- try a larger entry as the normalizer.
        # Pick whichever has the largest magnitude.
        norm_idx = int(np.argmax(np.abs(v_12)))
        den = float(v_12[norm_idx])
        if abs(den) < 1e-9:
            return None

    v_12 = v_12 / den
    # Now v_12[8] should be 1 (or close); recover x_3 from v_12[5], x_4 from v_12[7].
    # Cross-checks: v_12[2] = x_3^2, v_12[6] = x_4^2.
    x3 = float(v_12[5])
    x4 = float(v_12[7])
    # Sign sanity: v_12[2] = x_3^2 should be >= 0 in exact arithmetic; v_12[6] = x_4^2.
    # If the cross-check ratios disagree by a lot, signal failure.
    cross_x3_sq = float(v_12[2])
    cross_x4_sq = float(v_12[6])
    if cross_x3_sq < -0.1 or cross_x4_sq < -0.1:
        return None
    if abs(cross_x3_sq - x3 * x3) > 1e-3 + 1e-3 * abs(cross_x3_sq):
        return None
    if abs(cross_x4_sq - x4 * x4) > 1e-3 + 1e-3 * abs(cross_x4_sq):
        return None

    q3 = 2.0 * np.arctan(x3)
    q4 = 2.0 * np.arctan(x4)

    # Step 5: solve Q v_right = P_eff @ v_left for v_right (14x8 LSQ).
    s2, c2 = float(np.sin(q2)), float(np.cos(q2))
    s3, c3 = float(np.sin(q3)), float(np.cos(q3))
    s4, c4 = float(np.sin(q4)), float(np.cos(q4))
    v_left = np.array([s3 * s4, s3 * c4, c3 * s4, c3 * c4, s3, c3, s4, c4, 1.0])
    p_eff = p_sin * s2 + p_cos * c2 + p_one  # 14x9
    rhs = p_eff @ v_left  # 14
    v_right, _, _, _ = np.linalg.lstsq(q_mat, rhs, rcond=None)

    # v_right = (s_0 s_1, s_0 c_1, c_0 s_1, c_0 c_1, s_0, c_0, s_1, c_1)
    s0, c0 = float(v_right[4]), float(v_right[5])
    s1, c1 = float(v_right[6]), float(v_right[7])
    q0 = float(np.arctan2(s0, c0))
    q1 = float(np.arctan2(s1, c1))

    # Step 6: recover q_5 from FK residual.
    # Forward kinematics through joints 0..4, then A_5_residual = FK_partial^{-1} @ T_target.
    fk_partial = np.eye(4)
    qs_known = [q0, q1, q2, q3, q4]
    for i in range(5):
        fk_partial = fk_partial @ _dh_matrix_num(qs_known[i], alpha[i], a[i], d[i])
    # A_5 residual: T_target = FK_partial @ A_5(q_5)
    # => A_5(q_5) = FK_partial^{-1} @ T_target
    a5_residual = np.linalg.solve(fk_partial, t_target)
    # A_5 = R_z(q_5) T_z(d_5) T_x(a_5) R_x(alpha_5).
    # Its rotation block = R_z(q_5) * (constant part). The (0,0) and (1,0)
    # entries are c_5 and s_5 respectively (the R_z(q_5) column 0).
    q5 = float(np.arctan2(a5_residual[1, 0], a5_residual[0, 0]))

    return np.array([q0, q1, q2, q3, q4, q5])


# ---------------------------------------------------------------------------
# End-to-end driver: enumerate all IK solutions for a (DH, T_target) problem.
# ---------------------------------------------------------------------------


def _fk_dh(q: NDArray[np.float64], dh: DhParams) -> NDArray[np.float64]:
    """Forward kinematics for a 6R standard-DH chain at joint vector ``q``."""
    alpha, a, d = dh
    t = np.eye(4)
    for i in range(6):
        t = t @ _dh_matrix_num(float(q[i]), float(alpha[i]), float(a[i]), float(d[i]))
    return t


def solve_all_ik(
    dh: DhParams,
    t_target: NDArray[np.float64],
    *,
    fk_atol: float = 1e-6,
    dedup_atol: float = 1e-3,
) -> tuple[list[NDArray[np.float64]], bool]:
    """Run the full Raghavan-Roth pipeline and return all valid IK solutions.

    Pipeline:
      1. Build the 14-row (P, Q) symbolic system (cached per arm).
      2. SVD-eliminate v_right to get 6x9 E.
      3. Weierstrass for q_2 + basis change for (q_3, q_4) to get the
         6x9 quadratic-in-x_2 system in the v_left_x basis.
      4. Build 12x12 M(x_2) via the x_3-shift block construction.
      5. Companion-matrix eigenvalue route -> up to 16 real x_2 roots
         (filtering 8 spurious near +/-i and complex-conjugate pairs).
      6. Back-substitute each root -> candidate (q_0, ..., q_5).
      7. FK-validate each candidate; drop those with residual > ``fk_atol``.
      8. Deduplicate via wrap-to-pi joint-distance threshold.

    :param dh: Tuple ``(alpha, a, d)`` of length-6 numpy arrays.
    :param t_target: 4x4 target end-effector pose.
    :param fk_atol: max allowed ``||FK(q) - t_target||_F`` for a solution to be kept.
    :param dedup_atol: per-joint wrap-to-pi tolerance below which two
        solutions collapse to one.
    :returns: ``(solutions, is_ls)`` where solutions is a list of 6-vectors
        and ``is_ls`` is True if no solution survived FK validation.
    """
    p_sin, p_cos, p_one, q_mat = build_pq(dh, t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)
    roots, eigvecs = solve_x2_roots(m_quad, m_lin, m_const)

    candidates: list[NDArray[np.float64]] = []
    for x2_root, eigvec in zip(roots, eigvecs, strict=True):
        q_cand = back_substitute(
            x2_root, eigvec, p_sin, p_cos, p_one, q_mat, dh, t_target,
        )
        if q_cand is None:
            continue
        # FK closure check.
        t_check = _fk_dh(q_cand, dh)
        if float(np.linalg.norm(t_check - t_target)) > fk_atol:
            continue
        candidates.append(q_cand)

    # Deduplicate with wrap-to-pi joint distance.
    solutions: list[NDArray[np.float64]] = []
    for q in candidates:
        is_dup = False
        for existing in solutions:
            diffs = [abs(((float(q[i] - existing[i]) + np.pi) % (2 * np.pi)) - np.pi) for i in range(6)]
            if max(diffs) < dedup_atol:
                is_dup = True
                break
        if not is_dup:
            solutions.append(q)

    return solutions, len(solutions) == 0
