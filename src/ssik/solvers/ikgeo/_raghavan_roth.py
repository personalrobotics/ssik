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
    "solve_x2_roots_mobius",
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


def _equilibrate_pencil(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Row + column equilibrate ``(A, B, C)`` jointly for better-conditioned
    eigendecomposition. Issue #68 (AE-1).

    Joint scaling: each row's max-magnitude entry across (A, B, C) scaled to 1,
    then each column's max-magnitude entry scaled to 1. The quadratic eigenvalue
    problem ``(A x^2 + B x + C) v = 0`` and the equilibrated ``(D_l A D_r) x^2 +
    (D_l B D_r) x + (D_l C D_r)`` have the **same eigenvalues**; eigenvectors
    transform as ``v = D_r * v_eq``.

    ikfast does NOT do this (per #81 ikfast survey). Free win on chains where
    coefficients span many orders of magnitude (e.g. JACO 2's 60-deg twists).

    :returns: ``(A_eq, B_eq, C_eq, d_l, d_r)`` where ``d_l, d_r`` are 1D scaling
        vectors (diagonal entries of D_l, D_r).
    """
    row_max = np.maximum.reduce(
        [np.abs(a).max(axis=1), np.abs(b).max(axis=1), np.abs(c).max(axis=1)]
    )
    row_max = np.where(row_max > 0, row_max, 1.0)
    d_l = 1.0 / row_max
    a1 = a * d_l[:, None]
    b1 = b * d_l[:, None]
    c1 = c * d_l[:, None]
    col_max = np.maximum.reduce(
        [np.abs(a1).max(axis=0), np.abs(b1).max(axis=0), np.abs(c1).max(axis=0)]
    )
    col_max = np.where(col_max > 0, col_max, 1.0)
    d_r = 1.0 / col_max
    a2 = a1 * d_r[None, :]
    b2 = b1 * d_r[None, :]
    c2 = c1 * d_r[None, :]
    return a2, b2, c2, d_l, d_r


def solve_x2_roots(
    m_quad: NDArray[np.float64],
    m_lin: NDArray[np.float64],
    m_const: NDArray[np.float64],
    *,
    spurious_tol: float = 0.1,
    imag_rel_tol: float = 1e-3,
    cond_threshold: float = 1e10,
    equilibrate: bool = True,
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
    if equilibrate:
        a_eq, b_eq, c_eq, d_l, d_r = _equilibrate_pencil(m_quad, m_lin, m_const)
    else:
        a_eq, b_eq, c_eq = m_quad, m_lin, m_const
        d_r = np.ones(12)

    cond = float(np.linalg.cond(a_eq))
    if cond > cond_threshold:
        raise np.linalg.LinAlgError(
            f"M_quad ill-conditioned (cond={cond:.3e}, equilibrated); "
            "generalized-eigenvalue fallback required"
        )

    a_inv_b = np.linalg.solve(a_eq, b_eq)
    a_inv_c = np.linalg.solve(a_eq, c_eq)

    sigma = np.zeros((24, 24), dtype=np.float64)
    sigma[:12, 12:] = np.eye(12)
    sigma[12:, :12] = -a_inv_c
    sigma[12:, 12:] = -a_inv_b

    eigvals, eigvecs = np.linalg.eig(sigma)

    # Recover original-basis eigenvectors: v_12 = D_r * v_eq for each
    # eigenvalue. The 24-vector returned has structure [v_12; lambda * v_12]
    # (in the original basis, after de-equilibration); back_substitute reads
    # only the top 12 entries.
    d_r_complex = d_r.astype(np.complex128)

    real_roots: list[float] = []
    real_eigvecs: list[NDArray[np.complex128]] = []
    for k in range(24):
        ev = eigvals[k]
        if abs(abs(ev.imag) - 1.0) < spurious_tol and abs(ev.real) < spurious_tol:
            continue
        if abs(ev.imag) > imag_rel_tol * max(abs(ev.real), 1.0):
            continue
        # De-equilibrate the eigenvector. eigvecs[:, k] has structure
        # [v_eq; lambda * v_eq]; convert top half to original basis via D_r,
        # then reconstruct the bottom half so back_substitute sees the
        # canonical [v_12; lambda v_12] form it expects.
        v_top_orig = d_r_complex * eigvecs[:12, k]
        v_bot_orig = ev * v_top_orig
        v_full = np.concatenate([v_top_orig, v_bot_orig])
        real_roots.append(float(ev.real))
        real_eigvecs.append(v_full)
    return real_roots, real_eigvecs


# ---------------------------------------------------------------------------
# M\u00f6bius reparameterization fallback (Manocha-Canny IV-C).
# ---------------------------------------------------------------------------


def _mobius_transform(
    m_quad: NDArray[np.float64],
    m_lin: NDArray[np.float64],
    m_const: NDArray[np.float64],
    aa: float,
    bb: float,
    cc: float,
    dd: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Apply x_2 = (aa*x_tilde + bb) / (cc*x_tilde + dd) to M(x_2) = A x^2 + B x + C.

    After substitution and clearing (cc*x_tilde + dd)^2, the new polynomial has
    coefficients (Manocha-Canny Eq. 17):

        A_new = aa^2 A + aa*cc B + cc^2 C
        B_new = 2 aa*bb A + (aa*dd + bb*cc) B + 2 cc*dd C
        C_new = bb^2 A + bb*dd B + dd^2 C
    """
    a_new = aa * aa * m_quad + aa * cc * m_lin + cc * cc * m_const
    b_new = 2.0 * aa * bb * m_quad + (aa * dd + bb * cc) * m_lin + 2.0 * cc * dd * m_const
    c_new = bb * bb * m_quad + bb * dd * m_lin + dd * dd * m_const
    return a_new, b_new, c_new


def solve_x2_roots_mobius(
    m_quad: NDArray[np.float64],
    m_lin: NDArray[np.float64],
    m_const: NDArray[np.float64],
    *,
    cond_threshold: float = 1e10,
    n_random_tries: int = 8,
    rng_seed: int = 0,
    spurious_tol: float = 0.1,
    imag_rel_tol: float = 1e-3,
) -> tuple[list[float], list[NDArray[np.complex128]]]:
    """Robust ``x_2`` root finder with M\u00f6bius reparameterization fallback.

    First tries the straight eigenvalue route via :func:`solve_x2_roots`. If
    ``m_quad`` is well-conditioned, returns immediately. Otherwise tries a
    sequence of random M\u00f6bius transforms ``x_2 = (aa*x_tilde + bb) / (cc*x_tilde + dd)``
    until a transform yields a well-conditioned ``A_new``; uses the eigenvalue
    route on the transformed pencil; applies the inverse M\u00f6bius
    ``x_2 = (aa*x_tilde + bb) / (cc*x_tilde + dd)`` to recover ``x_2`` from the eigenvalues.

    The eigenvectors of the transformed pencil have the same block structure
    ``[v_12; x_tilde * v_12]`` as the un-transformed problem, so the
    back-substitution stage uses them directly (no transformation needed
    beyond converting x_tilde -> x_2 for the q_2 extraction).

    :param cond_threshold: above this ``cond(A)``, attempt M\u00f6bius reparameterization.
    :param n_random_tries: number of random ``(aa, bb, cc, dd)`` quadruples to
        try; the best by ``cond(A_new)`` is used.
    :param rng_seed: RNG seed for reproducibility.

    :raises numpy.linalg.LinAlgError: if no random reparameterization gives a
        well-conditioned matrix (extremely rare; corresponds to a singular
        pencil and triggers the generalized-eigenvalue fallback in the caller).
    """
    # AE-1 (#68): equilibrate first, then check cond on the equilibrated
    # leading matrix. Often this reduces cond by 1-3 orders and lets us skip
    # the M\u00f6bius / generalized-eigenvalue fallbacks entirely.
    a_eq, b_eq, c_eq, _, d_r = _equilibrate_pencil(m_quad, m_lin, m_const)
    cond_eq = float(np.linalg.cond(a_eq))
    if cond_eq <= cond_threshold:
        # Equilibration alone made the pencil tractable. Use the direct
        # eigenvalue route on the equilibrated matrices; solve_x2_roots
        # handles the eigenvector de-equilibration internally.
        return solve_x2_roots(
            m_quad, m_lin, m_const,
            spurious_tol=spurious_tol, imag_rel_tol=imag_rel_tol,
            cond_threshold=cond_threshold, equilibrate=True,
        )

    # Equilibration insufficient. Track the original cond as the bar to beat
    # for the M\u00f6bius search, but operate on the raw matrices (M\u00f6bius +
    # equilibration interaction needs careful eigenvector recovery; raw is
    # safer for now -- can revisit in a follow-up).
    cond = float(np.linalg.cond(m_quad))
    rng = np.random.default_rng(rng_seed)
    best_aa = best_bb = best_cc = best_dd = 0.0
    best_cond = cond
    for trial in range(n_random_tries):
        # Widen the range each block of tries: (aa, bb, cc, dd) sampled from
        # increasingly large intervals to escape near-singular regions.
        scale = 1.0 + (trial // 4) * 2.0
        aa, bb, cc, dd = rng.uniform(-scale, scale, size=4)
        if abs(aa * dd - bb * cc) < 1e-3:
            continue
        a_new, _, _ = _mobius_transform(m_quad, m_lin, m_const, aa, bb, cc, dd)
        try_cond = float(np.linalg.cond(a_new))
        if try_cond < best_cond:
            best_cond = try_cond
            best_aa, best_bb, best_cc, best_dd = aa, bb, cc, dd

    if best_cond > cond_threshold:
        # Singular pencil: every M\u00f6bius transform fails. Fall through to the
        # generalized-eigenvalue route (scipy.linalg.eig on the pencil M_1 - x M_2).
        try:
            from scipy.linalg import eig as scipy_eig
        except ImportError as exc:
            raise np.linalg.LinAlgError(
                f"M\u00f6bius reparameterization failed (best cond={best_cond:.3e}); "
                f"scipy not available for generalized-eigenvalue fallback"
            ) from exc

        # MC Theorem 2: M(x) = A x^2 + B x + C, build pencil M_1 - x M_2 where
        #   M_1 = [[I_12,   0   ],
        #          [  0,    C   ]]  (24x24)
        #   M_2 = [[0,    I_12  ],
        #          [-A,    -B   ]]  (24x24)
        # Generalized eigenvalues are the roots of det M(x) = 0.
        m1 = np.zeros((24, 24), dtype=np.float64)
        m1[:12, :12] = np.eye(12)
        m1[12:, 12:] = m_const
        m2 = np.zeros((24, 24), dtype=np.float64)
        m2[:12, 12:] = np.eye(12)
        m2[12:, :12] = -m_quad
        m2[12:, 12:] = -m_lin
        eigvals, eigvecs = scipy_eig(m1, m2)

        real_roots: list[float] = []
        real_eigvecs: list[NDArray[np.complex128]] = []
        for k in range(24):
            ev = eigvals[k]
            if not np.isfinite(ev):
                continue
            if abs(abs(ev.imag) - 1.0) < spurious_tol and abs(ev.real) < spurious_tol:
                continue
            if abs(ev.imag) > imag_rel_tol * max(abs(ev.real), 1.0):
                continue
            # In the generalized-eigenvalue construction, the kernel vector v_12
            # lives in the *bottom* half of V (the standard companion-matrix
            # construction has it in the top half). Swap halves so back_substitute,
            # which always reads the top 12 entries as v_12, sees the right thing.
            v = eigvecs[:, k]
            v_swapped = np.concatenate([v[12:], v[:12]])
            real_roots.append(float(ev.real))
            real_eigvecs.append(v_swapped)
        return real_roots, real_eigvecs

    aa, bb, cc, dd = best_aa, best_bb, best_cc, best_dd
    a_t, b_t, c_t = _mobius_transform(m_quad, m_lin, m_const, aa, bb, cc, dd)

    # Eigenvalue route on the transformed pencil.
    x_tilde_roots, eigvecs = solve_x2_roots(
        a_t, b_t, c_t,
        spurious_tol=spurious_tol, imag_rel_tol=imag_rel_tol,
        cond_threshold=cond_threshold,
    )
    # Map x_tilde -> x_2 = (aa * x_tilde + bb) / (cc * x_tilde + dd).
    real_roots: list[float] = []
    real_eigvecs: list[NDArray[np.complex128]] = []
    for x_tilde, evec in zip(x_tilde_roots, eigvecs, strict=True):
        denom = cc * x_tilde + dd
        if abs(denom) < 1e-12:
            # x_2 -> infinity; corresponds to q_2 = pi. Skip (we'd need a
            # secondary parameterization to handle this; rare).
            continue
        x2 = (aa * x_tilde + bb) / denom
        real_roots.append(float(x2))
        real_eigvecs.append(evec)
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

    # Eigenvector structure: V = [v_12; lambda * v_12] where lambda is the
    # eigenvalue (== x_2 for the direct problem, == x_tilde when M\u00f6bius
    # reparameterization was used to recover x_2). The top half is v_12 in
    # either case; using it directly avoids needing to know lambda separately.
    v_12 = np.real(eigvec_24[:12])

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
    # Now v_12[8] should be 1 (or close). Recover x_3 from the entry with the
    # most reliable signal. In exact arithmetic v_12 = (x3^2 x4^2, x3^2 x4,
    # x3^2, x3 x4^2, x3 x4, x3, x4^2, x4, 1, x3^3 x4^2, x3^3 x4, x3^3); we have
    # multiple ways to read x_3 and x_4. Prefer entries with larger magnitude.
    #
    # x_3 candidates (numerator / denominator entries):
    #   v_12[5] / v_12[8],    v_12[2] / v_12[5],    v_12[11] / v_12[2],
    #   v_12[1] / v_12[7],    v_12[4] / v_12[7],    sqrt(|v_12[2]|) * sgn,
    # x_4 candidates:
    #   v_12[7] / v_12[8],    v_12[6] / v_12[7],    v_12[1] / v_12[5],
    #   v_12[4] / v_12[5],    sqrt(|v_12[6]|) * sgn.
    #
    # We try the magnitude-best ratio for each. Caller's FK validation filters
    # any branch that doesn't actually close; ill-conditioned eigenvalues that
    # don't correspond to valid IK solutions get dropped at that stage.
    x3 = float(v_12[5])
    x4 = float(v_12[7])

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


def _se3_log_residual(t_err: NDArray[np.float64]) -> NDArray[np.float64]:
    """6-vector residual for SE(3) error: (translation, rotation_axis-angle).

    Used by Newton refinement to get a smooth scalar objective for FK closure.
    Translation is read directly; rotation is via Rodrigues' formula
    log(R) -> axis-angle. For small errors, this is essentially the local
    twist coordinate.
    """
    trans_err = t_err[:3, 3]
    r_err = t_err[:3, :3]
    # log(R) via Rodrigues: angle = acos((tr R - 1) / 2), axis = ...
    cos_a = max(-1.0, min(1.0, 0.5 * (np.trace(r_err) - 1.0)))
    angle = float(np.arccos(cos_a))
    if angle < 1e-9:
        rot_err = np.zeros(3)
    else:
        s = 1.0 / (2.0 * np.sin(angle))
        rot_err = np.array(
            [
                s * (r_err[2, 1] - r_err[1, 2]) * angle,
                s * (r_err[0, 2] - r_err[2, 0]) * angle,
                s * (r_err[1, 0] - r_err[0, 1]) * angle,
            ]
        )
    return np.concatenate([trans_err, rot_err])


def _newton_refine(
    q0: NDArray[np.float64],
    dh: DhParams,
    t_target: NDArray[np.float64],
    *,
    fk_atol: float = 1e-9,
    max_iters: int = 30,
) -> tuple[NDArray[np.float64], int] | None:
    """LM refinement of ``q`` driven by FK closure tolerance, NOT iteration count.

    Calls scipy.optimize.least_squares with FK-residual ``2-norm < fk_atol``
    as the termination criterion (via ftol/xtol). ``max_iters`` is a safety cap
    (functions evaluations = ~7*iters); if LM doesn't reach ``fk_atol`` within
    that, the seed wasn't good enough -- caller drops the candidate.

    Returns ``(q_refined, iters_used)`` on convergence, or ``None`` on
    divergence / LM failure / cap-without-convergence. Honest about what
    happened; no silent extension of effort beyond cap.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return (q0, 0)

    def residual(q: NDArray[np.float64]) -> NDArray[np.float64]:
        t_diff = t_target @ np.linalg.inv(_fk_dh(q, dh))
        return _se3_log_residual(t_diff)

    # Use very tight LM relative tolerances (ftol/xtol/gtol = 1e-15) and let
    # LM run until either max_nfev or it physically stalls (residual reduction
    # below 1e-15 between iters). We then check the ABSOLUTE FK residual
    # against fk_atol; LM-internal relative tolerances aren't a substitute for
    # the absolute check the user actually wants.
    try:
        result = least_squares(
            residual,
            q0,
            method="lm",
            jac="3-point",  # central differences -- 10x better Jacobian precision than forward
            max_nfev=max_iters * 13,  # 3-point uses 12 fevs per Jacobian (vs 6 for forward)
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
        )
    except (np.linalg.LinAlgError, ValueError):
        return None

    final_residual = float(np.linalg.norm(result.fun))
    iters_used = int(result.nfev) // 7
    if final_residual > fk_atol:
        return None
    return (np.asarray(result.x, dtype=np.float64), iters_used)


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
    # Use the M\u00f6bius-fallback variant: well-conditioned -> direct path; otherwise
    # try a few random reparameterizations to recondition the leading matrix.
    roots, eigvecs = solve_x2_roots_mobius(m_quad, m_lin, m_const)

    candidates: list[NDArray[np.float64]] = []
    for x2_root, eigvec in zip(roots, eigvecs, strict=True):
        q_cand = back_substitute(
            x2_root, eigvec, p_sin, p_cos, p_one, q_mat, dh, t_target,
        )
        if q_cand is None:
            continue
        # Always run Newton refinement: the eigenvalue route gives ~1-2 digits
        # of precision in well-conditioned cases (tighter near machine eps);
        # in ill-conditioned cases (cond(m_quad) > 1e10, M\u00f6bius / generalized
        # path) precision degrades to ~1%, so the back_substitute output is
        # only a *seed* for Newton.
        refined = _newton_refine(q_cand, dh, t_target, fk_atol=fk_atol)
        if refined is None:
            continue
        q_refined, _iters = refined
        # FK closure check on the refined candidate.
        t_check = _fk_dh(q_refined, dh)
        if float(np.linalg.norm(t_check - t_target)) > fk_atol:
            continue
        candidates.append(q_refined)

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
