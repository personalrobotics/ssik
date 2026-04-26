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

__all__ = ["build_pq"]


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
