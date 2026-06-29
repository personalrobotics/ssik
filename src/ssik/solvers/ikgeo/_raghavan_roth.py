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

Provenance: this file is clean-room from Tsai App. C and Manocha-Canny 1994
only. No source-code lineage from any prior implementation.

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

import pickle
from collections.abc import Callable
from functools import lru_cache
from typing import Literal, cast, overload

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ssik.core.solution import Solution
from ssik.refinement import dedup_by_wrap_close, lm_refine

__all__ = [
    "back_substitute",
    "build_m_matrix",
    "build_pq",
    "eliminate_q0_q1",
    "pick_best_leftvar",
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


def _dh_matrix_inv_sym(
    s_q: sp.Symbol, c_q: sp.Symbol, alpha: float, a: float, d: float
) -> sp.Matrix:
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


def _reduce_trig(
    expr: sp.Expr, s_syms: tuple[sp.Symbol, ...], c_syms: tuple[sp.Symbol, ...]
) -> sp.Expr:
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


def _so3_basis(T_syms: tuple[sp.Symbol, ...]) -> list[sp.Expr]:
    """SO(3) constraint polynomials on the rotation block of T_target.

    T_syms is the flat 12-tuple ``(T_00 ... T_23)``; the rotation block is
    ``T_syms[i*4 + j]`` for i in 0..2, j in 0..2. Returns 9 quadratic
    constraints (col norms = 1, col dots = 0, col cross-products) -- only
    6 are algebraically independent but sympy's reduced() handles the
    redundancy. AE-4 (#71): adds these to the trig basis to symbolically
    remove rank-deficient combinations of T entries that bloat (P, Q)
    coefficients on chains where the rotation block's structure propagates
    into the eigenvalue conditioning (e.g. JACO 2's 60-deg twists).
    """
    R = [[T_syms[i * 4 + j] for j in range(3)] for i in range(3)]
    basis = []
    # Col norms = 1
    for j in range(3):
        basis.append(R[0][j] ** 2 + R[1][j] ** 2 + R[2][j] ** 2 - 1)
    # Col-pair dot products = 0
    basis.append(R[0][0] * R[0][1] + R[1][0] * R[1][1] + R[2][0] * R[2][1])
    basis.append(R[0][0] * R[0][2] + R[1][0] * R[1][2] + R[2][0] * R[2][2])
    basis.append(R[0][1] * R[0][2] + R[1][1] * R[1][2] + R[2][1] * R[2][2])
    # Cross product: col_0 x col_1 = col_2
    basis.append(R[1][0] * R[2][1] - R[2][0] * R[1][1] - R[0][2])
    basis.append(R[2][0] * R[0][1] - R[0][0] * R[2][1] - R[1][2])
    basis.append(R[0][0] * R[1][1] - R[1][0] * R[0][1] - R[2][2])
    return basis


def _reduce_trig_and_so3(
    expr: sp.Expr,
    s_syms: tuple[sp.Symbol, ...],
    c_syms: tuple[sp.Symbol, ...],
    T_syms: tuple[sp.Symbol, ...],
) -> sp.Expr:
    """Reduce ``expr`` modulo trig + SO(3) ideal. AE-4 (#71)."""
    trig = [s_syms[i] ** 2 + c_syms[i] ** 2 - 1 for i in range(len(s_syms))]
    so3 = _so3_basis(T_syms)
    basis = trig + so3
    all_gens = list(s_syms) + list(c_syms) + list(T_syms)
    _, remainder = sp.reduced(expr, basis, *all_gens)
    return sp.expand(remainder)


def _derive_pq_for_arm(
    alpha: tuple[float, ...],
    a: tuple[float, ...],
    d: tuple[float, ...],
    *,
    apply_so3: bool = False,
    linearity_joint: int = 2,
) -> tuple[
    Callable[..., NDArray[np.float64]],
    Callable[..., NDArray[np.float64]],
    Callable[..., NDArray[np.float64]],
    Callable[..., NDArray[np.float64]],
    dict[str, object],
]:
    """Derive 14-row Raghavan-Roth (P, Q) callables for a specific arm.

    DH params are numeric; T_target stays symbolic. Output: four callables
    that each take 12 entries of T_target (rows 0-2) and return either a
    14x9 matrix (the three P factors) or a 14x8 matrix (Q), plus a metadata
    dict describing the leftvar configuration.

    :param apply_so3: AE-4 (#71). If True, reduce (P, Q) coefficients modulo
        the SO(3) ideal on T_target's rotation block in addition to the trig
        ideal on joint angles. Adds ~9 quadratic constraints, slows derivation
        ~2-5x but may significantly reduce ``cond(m_quad)`` on chains where
        rank-deficient T-coefficient combinations bloat the leading matrix.
    :param linearity_joint: AE-3 (#70). The "leftvar" -- which joint's
        sin/cos appears linearly in the matrix entries. Valid values: 0, 1,
        or 2 (natural-orientation splits). Default 2 = original Manocha-Canny
        choice. Different leftvars yield different ``cond(m_quad)`` profiles
        for the same arm; pick-best-leftvar (#70) selects the lowest-cond
        choice per-arm.

        **Structural intuition (verified on JACO 2, 60-deg twists at
        joints 4, 5):** the singular pencil pathology arises when
        structurally-awkward joints sit in the v_left bilinear pair --
        v_left's monomial structure propagates directly into the polynomial
        ``M(x)`` and any rank deficiency there manifests as cond(A) > 1e15.
        Picking a leftvar that isolates pathological joints out of v_left
        (into "drop" or "v_right") avoids the singular pencil entirely.

        Per-leftvar role assignment for natural-orientation splits:

            linearity=0: v_left=(q_1,q_2), drop=q_3, v_right=(q_4,q_5)
            linearity=1: v_left=(q_2,q_3), drop=q_4, v_right=(q_0,q_5)
            linearity=2: v_left=(q_3,q_4), drop=q_5, v_right=(q_0,q_1)

        On JACO 2 geometry, linearity=q_1 gives cond(A)=127 vs
        cond(A)=3.75e16 at the default linearity=q_2 -- a 14-order
        reduction. Both pathological joints (q_4 dropped, q_5 in v_right)
        end up out of v_left, which holds standard pi/2 and pi twists.
    """
    if len(alpha) != 6 or len(a) != 6 or len(d) != 6:
        raise ValueError(f"DH must have 6 entries per array; got {len(alpha)}, {len(a)}, {len(d)}")
    if linearity_joint not in (0, 1, 2):
        raise ValueError(f"linearity_joint must be 0, 1, or 2; got {linearity_joint}")

    # 6 joint angles -> sin/cos symbols.
    s = sp.symbols("s0:6", real=True)
    c = sp.symbols("c0:6", real=True)

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

    # Loop-split configurations, parameterized on linearity_joint k (0, 1, 2).
    #   linearity = k:
    #     LHS = A_k A_{k+1} A_{k+2}              (3 LHS joints, R_z(q_k) at start)
    #     drop = q_{k+3}                          (postmultiplied via A_{k+3}^{-1} at far right)
    #     RHS = A_{k-1}^{-1} ... A_0^{-1} T A_{N-1}^{-1} ... A_{k+3}^{-1}
    #     left_bilinear = (k+1, k+2)              (joints absorbed into v_left)
    #     right_bilinear = remaining 2 joints     (in v_right)
    if linearity_joint == 2:
        # Current default. LHS=(q_2,q_3,q_4), drop=q_5, v_right=(q_0,q_1).
        lhs_mat = A_dh[2] * A_dh[3] * A_dh[4]
        rhs_mat = A_inv[1] * A_inv[0] * T_mat * A_inv[5]
        left_bilinear = (3, 4)
        right_bilinear = (0, 1)
    elif linearity_joint == 1:
        # LHS=(q_1,q_2,q_3), drop=q_4, v_right=(q_0,q_5).
        lhs_mat = A_dh[1] * A_dh[2] * A_dh[3]
        rhs_mat = A_inv[0] * T_mat * A_inv[5] * A_inv[4]
        left_bilinear = (2, 3)
        right_bilinear = (0, 5)
    else:  # linearity_joint == 0
        # LHS=(q_0,q_1,q_2), drop=q_3, v_right=(q_4,q_5).
        lhs_mat = A_dh[0] * A_dh[1] * A_dh[2]
        rhs_mat = T_mat * A_inv[5] * A_inv[4] * A_inv[3]
        left_bilinear = (1, 2)
        right_bilinear = (4, 5)

    # Active sin/cos symbols: 5 angles in the equation (drop joint excluded).
    drop_joint = (linearity_joint + 3) % 6 if linearity_joint > 0 else 3
    if linearity_joint == 2:
        drop_joint = 5
    elif linearity_joint == 1:
        drop_joint = 4
    else:
        drop_joint = 3
    s_active = tuple(s[i] for i in range(6) if i != drop_joint)
    c_active = tuple(c[i] for i in range(6) if i != drop_joint)

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
    if apply_so3:
        reduced_eqs = [_reduce_trig_and_so3(eq, s_active, c_active, T_syms) for eq in eqs]
    else:
        reduced_eqs = [_reduce_trig(eq, s_active, c_active) for eq in eqs]

    # Extract coefficients ---------------------------------------------------
    # v_left bilinear in joints (left_bilinear[0], left_bilinear[1]).
    lb0, lb1 = left_bilinear
    left_9 = [
        s[lb0] * s[lb1],
        s[lb0] * c[lb1],
        c[lb0] * s[lb1],
        c[lb0] * c[lb1],
        s[lb0],
        c[lb0],
        s[lb1],
        c[lb1],
        sp.Integer(1),
    ]
    # v_right bilinear in joints (right_bilinear[0], right_bilinear[1]).
    rb0, rb1 = right_bilinear
    right_8 = [
        s[rb0] * s[rb1],
        s[rb0] * c[rb1],
        c[rb0] * s[rb1],
        c[rb0] * c[rb1],
        s[rb0],
        c[rb0],
        s[rb1],
        c[rb1],
    ]

    n_rows = len(reduced_eqs)  # 14
    p_sin_sym: list[list[sp.Expr]] = [[sp.Integer(0)] * 9 for _ in range(n_rows)]
    p_cos_sym: list[list[sp.Expr]] = [[sp.Integer(0)] * 9 for _ in range(n_rows)]
    p_one_sym: list[list[sp.Expr]] = [[sp.Integer(0)] * 9 for _ in range(n_rows)]
    q_sym: list[list[sp.Expr]] = [[sp.Integer(0)] * 8 for _ in range(n_rows)]

    poly_gens = (*s_active, *c_active)
    s_lin = s[linearity_joint]
    c_lin = c[linearity_joint]
    for r, eq in enumerate(reduced_eqs):
        poly = sp.Poly(eq, *poly_gens)
        for j, mon in enumerate(left_9):
            p_sin_sym[r][j] = poly.coeff_monomial(sp.expand(s_lin * mon))
            p_cos_sym[r][j] = poly.coeff_monomial(sp.expand(c_lin * mon))
            p_one_sym[r][j] = poly.coeff_monomial(mon)
        for j, mon in enumerate(right_8):
            # eq = LHS - RHS; right monomials live in -RHS.
            q_sym[r][j] = -poly.coeff_monomial(mon)

    # Lambdify with T_target entries as args.
    p_sin_fn = sp.lambdify(T_syms, sp.Matrix(p_sin_sym), "numpy")
    p_cos_fn = sp.lambdify(T_syms, sp.Matrix(p_cos_sym), "numpy")
    p_one_fn = sp.lambdify(T_syms, sp.Matrix(p_one_sym), "numpy")
    q_fn = sp.lambdify(T_syms, sp.Matrix(q_sym), "numpy")
    metadata = {
        "linearity_joint": linearity_joint,
        "left_bilinear": left_bilinear,
        "right_bilinear": right_bilinear,
        "drop_joint": drop_joint,
        "apply_so3": apply_so3,
        # Symbolic matrices stashed for codegen consumers (#118): the
        # composer for general_6r needs the pre-lambdify expressions to
        # render explicit-trig source. Stored as ``sp.Matrix`` instances;
        # T_target enters as the 12 ``T_:12`` symbols.
        "_sym_p_sin": sp.Matrix(p_sin_sym),
        "_sym_p_cos": sp.Matrix(p_cos_sym),
        "_sym_p_one": sp.Matrix(p_one_sym),
        "_sym_q": sp.Matrix(q_sym),
        "_sym_t_target": T_syms,
    }
    return p_sin_fn, p_cos_fn, p_one_fn, q_fn, metadata


# Cache the per-arm derivation. Keyed on (DH, linearity, so3) tuples.
#
# Implemented as a module-level ``dict`` rather than ``functools.lru_cache``
# so build artifacts can populate it from a deserialised pickle without
# re-running the 7-50 s sympy derivation. (#210 Phase 2.)
_DhTuple = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
_DerivationKey = tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int, bool]
_DerivationValue = tuple[
    Callable[..., NDArray[np.float64]],
    Callable[..., NDArray[np.float64]],
    Callable[..., NDArray[np.float64]],
    Callable[..., NDArray[np.float64]],
    dict[str, object],
]
_DERIVATION_CACHE: dict[_DerivationKey, _DerivationValue] = {}

# Both caches are keyed on per-sample sub-chain DH (from ``poe_to_dh``). Those
# values differ in the last bits across BLAS backends (OpenBLAS vs Accelerate,
# ~1e-12); keying on exact floats then misses *across platforms* -- the
# macOS-baked key != the Linux-runtime key -- silently dropping the cached-RR
# fast path into the ~200x-slower ``search_1d`` fallback (#350: 55/176 keys
# missed on Linux, xarm7 5.3 s/solve vs 23 ms on macOS). Quantizing the key to
# 1e-6 absorbs the cross-backend jitter while keeping distinct sub-chains (which
# differ by >= 1e-2 between lock samples) separate. The full-precision DH is
# still used for the derivation itself; only the dict *key* is rounded.
_DH_KEY_DECIMALS = 6


def _dh_key(values: tuple[float, ...] | NDArray[np.float64]) -> tuple[float, ...]:
    return tuple(round(float(x), _DH_KEY_DECIMALS) for x in values)


def _cached_derivation(
    alpha: tuple[float, ...],
    a: tuple[float, ...],
    d: tuple[float, ...],
    linearity_joint: int = 2,
    apply_so3: bool = False,
) -> _DerivationValue:
    key = (_dh_key(alpha), _dh_key(a), _dh_key(d), int(linearity_joint), bool(apply_so3))
    cached = _DERIVATION_CACHE.get(key)
    if cached is not None:
        return cached
    value = _derive_pq_for_arm(alpha, a, d, linearity_joint=linearity_joint, apply_so3=apply_so3)
    _DERIVATION_CACHE[key] = value
    return value


# ---------------------------------------------------------------------------
# Build-time priming for jointlock cached-RR (#210).
# ---------------------------------------------------------------------------

# Map from (alpha, a, d) to (linearity_joint, apply_so3) for primed sub-chains.
# Lets jointlock dispatch look up the baked AE-3 leftvar choice without
# running the expensive ``_cached_best_leftvar`` probe (which would do
# 3 * (sympy derivation) = ~90 s cold per arm).
_PRIMED_LINEARITY_MAP: dict[_DhTuple, tuple[int, bool]] = {}


def prime_derivation(
    alpha: tuple[float, ...] | NDArray[np.float64],
    a: tuple[float, ...] | NDArray[np.float64],
    d: tuple[float, ...] | NDArray[np.float64],
    linearity_joint: int = 2,
    apply_so3: bool = False,
) -> None:
    """Pre-populate the RR derivation cache for a (DH, linearity) tuple.

    Runs the full sympy derivation (~7 s per call) and stores the result.
    Use :func:`prime_derivation_from_blob` instead when a pre-computed
    serialised derivation is available -- it's ~30x faster (~0.25 s).

    Also records the (DH -> linearity) mapping so the jointlock dispatch
    can look up the baked AE-3 leftvar choice without running the
    expensive runtime probe; :func:`primed_linearity_for_dh` returns the
    cached value after this call.

    :param alpha, a, d: DH parameters as 6-tuples (or 6-element arrays).
    :param linearity_joint: AE-3 leftvar choice. Pass the result of
        :func:`_cached_best_leftvar` for production-grade conditioning.
    :param apply_so3: SO(3)-ideal reduction flag (default False;
        sufficient for jointlock-inner sub-chains).
    """
    alpha_t = tuple(float(x) for x in alpha)
    a_t = tuple(float(x) for x in a)
    d_t = tuple(float(x) for x in d)
    _PRIMED_LINEARITY_MAP[(_dh_key(alpha_t), _dh_key(a_t), _dh_key(d_t))] = (
        int(linearity_joint),
        bool(apply_so3),
    )
    _cached_derivation(alpha_t, a_t, d_t, int(linearity_joint), bool(apply_so3))


def serialize_derivation(
    alpha: tuple[float, ...] | NDArray[np.float64],
    a: tuple[float, ...] | NDArray[np.float64],
    d: tuple[float, ...] | NDArray[np.float64],
    linearity_joint: int = 2,
    apply_so3: bool = False,
) -> bytes:
    """Run the symbolic RR derivation for ``(alpha, a, d, linearity_joint,
    apply_so3)`` and return a pickle-serialised blob containing the symbolic
    matrices needed to re-lambdify at load time.

    The serialised payload contains only the sympy ``Matrix`` expressions
    + the ``T_target`` symbol tuple, NOT the lambdified callables (which
    aren't picklable). At load time, :func:`prime_derivation_from_blob`
    deserialises the matrices and runs ``sp.lambdify`` to reconstruct the
    callables -- ~0.25 s on a 6-DOF chain vs ~7 s for the cold derivation.

    Build-time use (#210 Phase 2): the ``ssik build`` codegen composer for
    ``jointlock.seven_r`` calls this once per non-tier-0 inner sub-chain DH
    at codegen time and writes the bytes to a sidecar ``.pkl`` file.
    Module-init for the resulting artifact loads the pickle and primes the
    derivation cache via :func:`prime_derivation_from_blob`, paying the
    sympy cost once at build instead of on every ``import``.
    """
    alpha_t = tuple(float(x) for x in alpha)
    a_t = tuple(float(x) for x in a)
    d_t = tuple(float(x) for x in d)
    _, _, _, _, meta = _cached_derivation(alpha_t, a_t, d_t, int(linearity_joint), bool(apply_so3))
    # If the cached entry came from the AOT prime path (#320), the
    # build-time-only ``_sym_*`` matrices aren't present -- the AOT
    # artifact ships the lambdified callables directly, not the
    # symbolic forms. Run a fresh derivation in that case so the
    # legacy blob serialiser still produces a valid payload.
    if "_sym_p_sin" not in meta:
        _, _, _, _, meta = _derive_pq_for_arm(
            alpha_t, a_t, d_t, linearity_joint=int(linearity_joint), apply_so3=bool(apply_so3)
        )
    payload = {
        "version": 1,
        "alpha": alpha_t,
        "a": a_t,
        "d": d_t,
        "linearity_joint": int(linearity_joint),
        "apply_so3": bool(apply_so3),
        "sym_p_sin": meta["_sym_p_sin"],
        "sym_p_cos": meta["_sym_p_cos"],
        "sym_p_one": meta["_sym_p_one"],
        "sym_q": meta["_sym_q"],
        "sym_t_target": meta["_sym_t_target"],
        "left_bilinear": meta["left_bilinear"],
        "right_bilinear": meta["right_bilinear"],
        "drop_joint": meta["drop_joint"],
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def prime_derivation_from_blob(blob: bytes) -> None:
    """Deserialise a payload produced by :func:`serialize_derivation` and
    populate the derivation cache + linearity map without re-running the
    full sympy derivation.

    Faster than :func:`prime_derivation` by ~30x (re-lambdify-only vs
    full algebraic derivation). Used by ``ssik build`` artifacts at
    module-import time: the artifact loads a sidecar ``.pkl`` and calls
    this function once per primed sub-chain DH.

    :raises ValueError: on payload version mismatch.
    """
    payload = pickle.loads(blob)
    if payload.get("version") != 1:
        raise ValueError(f"unsupported derivation payload version: {payload.get('version')}")
    T_syms = payload["sym_t_target"]
    p_sin_fn = sp.lambdify(T_syms, payload["sym_p_sin"], "numpy")
    p_cos_fn = sp.lambdify(T_syms, payload["sym_p_cos"], "numpy")
    p_one_fn = sp.lambdify(T_syms, payload["sym_p_one"], "numpy")
    q_fn = sp.lambdify(T_syms, payload["sym_q"], "numpy")
    metadata: dict[str, object] = {
        "linearity_joint": payload["linearity_joint"],
        "left_bilinear": payload["left_bilinear"],
        "right_bilinear": payload["right_bilinear"],
        "drop_joint": payload["drop_joint"],
        "apply_so3": payload["apply_so3"],
        "_sym_p_sin": payload["sym_p_sin"],
        "_sym_p_cos": payload["sym_p_cos"],
        "_sym_p_one": payload["sym_p_one"],
        "_sym_q": payload["sym_q"],
        "_sym_t_target": T_syms,
    }
    key = (
        _dh_key(payload["alpha"]),
        _dh_key(payload["a"]),
        _dh_key(payload["d"]),
        payload["linearity_joint"],
        payload["apply_so3"],
    )
    _DERIVATION_CACHE[key] = (p_sin_fn, p_cos_fn, p_one_fn, q_fn, metadata)
    _PRIMED_LINEARITY_MAP[
        (_dh_key(payload["alpha"]), _dh_key(payload["a"]), _dh_key(payload["d"]))
    ] = (
        payload["linearity_joint"],
        payload["apply_so3"],
    )


def _prime_aot(
    alpha: tuple[float, ...],
    a: tuple[float, ...],
    d: tuple[float, ...],
    linearity_joint: int,
    apply_so3: bool,
    left_bilinear: tuple[int, int],
    right_bilinear: tuple[int, int],
    drop_joint: int,
    p_sin_fn: Callable[..., NDArray[np.float64]],
    p_cos_fn: Callable[..., NDArray[np.float64]],
    p_one_fn: Callable[..., NDArray[np.float64]],
    q_fn: Callable[..., NDArray[np.float64]],
) -> None:
    """Populate ``_DERIVATION_CACHE`` + ``_PRIMED_LINEARITY_MAP`` from
    AOT-baked callables embedded as Python source in the artifact (#320).

    Replacement for :func:`prime_derivation_from_blob`: no sympy work at
    import. The lambdified callables were emitted as Python source at
    build time via :func:`inspect.getsource`, so by the time this is
    called the functions are already live Python objects. Measured
    ~57x faster than the blob-prime path on Kassow KR810 (4.5 s -> 80 ms
    cold module-import), with bit-identical numerical output.

    The metadata stored is the lean runtime-only subset that
    :mod:`ssik.solvers.ikgeo._raghavan_roth.solve_all_ik` actually reads;
    the symbolic ``_sym_*`` matrices that the build-time
    :mod:`ssik.codegen._compose.general_6r` composer consumes are NOT
    shipped because no runtime path uses them.
    """
    metadata: dict[str, object] = {
        "linearity_joint": int(linearity_joint),
        "apply_so3": bool(apply_so3),
        "left_bilinear": left_bilinear,
        "right_bilinear": right_bilinear,
        "drop_joint": int(drop_joint),
    }
    key = (_dh_key(alpha), _dh_key(a), _dh_key(d), int(linearity_joint), bool(apply_so3))
    _DERIVATION_CACHE[key] = (p_sin_fn, p_cos_fn, p_one_fn, q_fn, metadata)
    _PRIMED_LINEARITY_MAP[(_dh_key(alpha), _dh_key(a), _dh_key(d))] = (
        int(linearity_joint),
        bool(apply_so3),
    )


def primed_linearity_for_dh(
    alpha: tuple[float, ...] | NDArray[np.float64],
    a: tuple[float, ...] | NDArray[np.float64],
    d: tuple[float, ...] | NDArray[np.float64],
) -> tuple[int, bool] | None:
    """Look up the baked (linearity_joint, apply_so3) for a sub-chain DH.

    Returns ``None`` if :func:`prime_derivation` hasn't been called for
    this DH (cache miss == cold-cache RR would fire == fall back to
    original solver).

    Used by ``jointlock.seven_r._dispatch`` to gate the cached-RR
    fast-path. The lookup is O(1) and free of sympy work, so it's safe
    to call on every per-sample inner dispatch.
    """
    return _PRIMED_LINEARITY_MAP.get((_dh_key(alpha), _dh_key(a), _dh_key(d)))


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


_PQ4 = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
_PQ5 = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, object],
]


@overload
def build_pq(
    dh: DhParams,
    t_target: NDArray[np.float64],
    *,
    linearity_joint: int = ...,
    apply_so3: bool = ...,
    return_metadata: Literal[False] = ...,
) -> _PQ4: ...


@overload
def build_pq(
    dh: DhParams,
    t_target: NDArray[np.float64],
    *,
    linearity_joint: int = ...,
    apply_so3: bool = ...,
    return_metadata: Literal[True],
) -> _PQ5: ...


def build_pq(
    dh: DhParams,
    t_target: NDArray[np.float64],
    *,
    linearity_joint: int = 2,
    apply_so3: bool = False,
    return_metadata: bool = False,
) -> _PQ4 | _PQ5:
    """Build the (factored) Raghavan-Roth elimination matrices.

    :param dh: Tuple ``(alpha, a, d)`` of length-6 numpy arrays giving the
        standard DH parameters per joint.
    :param t_target: 4x4 target end-effector pose in the base frame.
    :param linearity_joint: AE-3 (#70) leftvar choice. See
        :func:`_derive_pq_for_arm` docstring for the full intuition.
    :param apply_so3: AE-4 (#71) SO(3) identity reduction.
    :param return_metadata: If True, returns ``(P_sin, P_cos, P_one, Q, meta)``.
        ``meta`` carries the leftvar role assignment needed by
        :func:`back_substitute`.
    :returns: ``(P_sin, P_cos, P_one, Q)`` (14x9, 14x9, 14x9, 14x8) by default,
        or 5-tuple including metadata if ``return_metadata=True``.

    First call for a given (DH, linearity_joint, apply_so3) tuple takes
    30-100 s (symbolic derivation); subsequent calls hit the cache.
    """
    alpha, a, d = dh
    if alpha.shape != (6,) or a.shape != (6,) or d.shape != (6,):
        raise ValueError(
            f"DH params must be length-6 arrays; got {alpha.shape}, {a.shape}, {d.shape}"
        )
    t = np.asarray(t_target, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"t_target must be 4x4; got {t.shape}")

    p_sin_fn, p_cos_fn, p_one_fn, q_fn, meta = _cached_derivation(
        tuple(alpha.tolist()),
        tuple(a.tolist()),
        tuple(d.tolist()),
        linearity_joint,
        apply_so3,
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
    if return_metadata:
        return p_sin, p_cos, p_one, q, meta
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
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Row + column equilibrate ``(A, B, C)`` jointly for better-conditioned
    eigendecomposition. Issue #68 (AE-1).

    Thin adapter over the shared
    :func:`ssik._pencil.equilibrate_three_matrix_pencil`; the actual logic
    lives in :mod:`ssik._pencil` so the Husty-Pfurner pipeline (#162) can
    reuse it for arbitrary-degree polynomial matrices.

    The quadratic eigenvalue problem ``(A x^2 + B x + C) v = 0`` and the
    equilibrated ``(D_l A D_r) x^2 + (D_l B D_r) x + (D_l C D_r)`` have
    the **same eigenvalues**; eigenvectors transform as ``v = D_r * v_eq``.
    ikfast does NOT do this (per #81 ikfast survey).

    :returns: ``(A_eq, B_eq, C_eq, d_l, d_r)``.
    """
    from ssik._pencil import equilibrate_three_matrix_pencil

    return equilibrate_three_matrix_pencil(a, b, c)


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
        a_eq, b_eq, c_eq, _d_l, d_r = _equilibrate_pencil(m_quad, m_lin, m_const)
    else:
        a_eq, b_eq, c_eq = m_quad, m_lin, m_const
        d_r = np.ones(12)

    cond = float(np.linalg.cond(a_eq))
    if cond > cond_threshold:
        raise np.linalg.LinAlgError(
            f"M_quad ill-conditioned (cond={cond:.3e}, equilibrated); "
            "generalized-eigenvalue fallback required"
        )

    # Stack [B|C] into one 12x24 RHS so LU(A) is computed once instead of
    # twice (#86 Tier 2.3): a_inv_b and a_inv_c share the same factorisation.
    bc_stacked = np.empty((12, 24), dtype=np.float64)
    bc_stacked[:, :12] = b_eq
    bc_stacked[:, 12:] = c_eq
    a_inv_bc = np.linalg.solve(a_eq, bc_stacked)
    a_inv_b = a_inv_bc[:, :12]
    a_inv_c = a_inv_bc[:, 12:]

    sigma = np.zeros((24, 24), dtype=np.float64)
    sigma[:12, 12:] = np.eye(12)
    sigma[12:, :12] = -a_inv_c
    sigma[12:, 12:] = -a_inv_b

    eigvals, eigvecs = np.linalg.eig(sigma)

    # Vectorised filter + de-equilibration (#86 Tier 2.3): drop the per-
    # eigenvalue Python loop in favour of mask + array reshape.
    abs_imag = np.abs(eigvals.imag)
    abs_real = np.abs(eigvals.real)
    spurious = (np.abs(abs_imag - 1.0) < spurious_tol) & (abs_real < spurious_tol)
    too_complex = abs_imag > imag_rel_tol * np.maximum(abs_real, 1.0)
    keep = ~(spurious | too_complex)

    d_r_complex = d_r.astype(np.complex128)
    # ``eigvecs[:, k]`` has structure [v_eq; lambda * v_eq]; rebuild as
    # [D_r * v_eq; lambda * D_r * v_eq] (= [v_12; lambda * v_12] in the
    # original basis, which back_substitute expects).
    v_top = d_r_complex[:, None] * eigvecs[:12, keep]  # 12 x K
    v_bot = eigvals[keep][None, :] * v_top  # 12 x K
    v_full = np.concatenate([v_top, v_bot], axis=0)  # 24 x K

    real_roots = [float(r) for r in eigvals[keep].real]
    real_eigvecs = [v_full[:, k] for k in range(v_full.shape[1])]
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
    a_eq, _b_eq, _c_eq, _, _d_r = _equilibrate_pencil(m_quad, m_lin, m_const)
    cond_eq = float(np.linalg.cond(a_eq))
    if cond_eq <= cond_threshold:
        # Equilibration alone made the pencil tractable. Use the direct
        # eigenvalue route on the equilibrated matrices; solve_x2_roots
        # handles the eigenvector de-equilibration internally.
        return solve_x2_roots(
            m_quad,
            m_lin,
            m_const,
            spurious_tol=spurious_tol,
            imag_rel_tol=imag_rel_tol,
            cond_threshold=cond_threshold,
            equilibrate=True,
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
            from scipy.linalg import eig as scipy_eig  # type: ignore[import-untyped]
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
        a_t,
        b_t,
        c_t,
        spurious_tol=spurious_tol,
        imag_rel_tol=imag_rel_tol,
        cond_threshold=cond_threshold,
    )
    # Map x_tilde -> x_2 = (aa * x_tilde + bb) / (cc * x_tilde + dd).
    real_roots = []
    real_eigvecs = []
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


def _back_substitute_inner(
    x_lin: float,
    eigvec_24: NDArray[np.complex128],
    p_sin: NDArray[np.float64],
    p_cos: NDArray[np.float64],
    p_one: NDArray[np.float64],
    q_pinv: NDArray[np.float64],
    dh: DhParams,
    t_target: NDArray[np.float64],
    metadata: dict[str, object],
) -> tuple[NDArray[np.float64], float] | None:
    """Branch-level back-substitution worker.

    Same algorithm as :func:`back_substitute`, but takes the precomputed
    ``q_pinv = pinv(q_mat)`` (so that ``solve_all_ik`` can compute it once
    per pose and amortise across branches), and returns ``(q, fk_err)``
    where ``fk_err = ||FK(q) - T_target||_F`` so the caller doesn't pay
    for a redundant standalone FK call. This shape is internal; the
    public :func:`back_substitute` wraps this function.
    """
    linearity_joint = cast(int, metadata["linearity_joint"])
    lb0, lb1 = cast(tuple[int, int], metadata["left_bilinear"])
    rb0, rb1 = cast(tuple[int, int], metadata["right_bilinear"])
    drop_joint = cast(int, metadata["drop_joint"])

    alpha, a, d = dh

    q_lin = 2.0 * np.arctan(x_lin)

    # Eigenvector structure: V = [v_12; lambda * v_12] where lambda is the
    # eigenvalue (== x_lin for the direct problem, == x_tilde when M\u00f6bius
    # reparameterization was used). The top half is v_12 in either case.
    v_12 = np.real(eigvec_24[:12])

    # Robust ratio selection (Manocha-Canny IV-C, "use entries with largest
    # magnitude as denominators to minimize error"). v_12 entries:
    #   0:  x_lb0^2 x_lb1^2     6:  x_lb1^2
    #   1:  x_lb0^2 x_lb1       7:  x_lb1
    #   2:  x_lb0^2             8:  1
    #   3:  x_lb0 x_lb1^2       9:  x_lb0^3 x_lb1^2
    #   4:  x_lb0 x_lb1        10:  x_lb0^3 x_lb1
    #   5:  x_lb0              11:  x_lb0^3
    #
    # Each ratio (num, den) below has algebraic value = x_lb0 (or x_lb1).
    # Picking the pair where |v_12[den]| is largest minimizes amplified noise
    # (when x is large the canonical "1" entry is small in unit-normalized
    # eigenvectors, so v_12[5]/v_12[8] becomes ill-conditioned -- avoid).
    x0_ratio_candidates = [(5, 8), (2, 5), (11, 2), (4, 7), (10, 1), (3, 6), (9, 0)]
    x1_ratio_candidates = [(7, 8), (6, 7), (1, 2), (4, 5), (10, 11)]

    num_idx, den_idx = max(x0_ratio_candidates, key=lambda nd: abs(v_12[nd[1]]))
    if abs(v_12[den_idx]) < 1e-12:
        return None
    x_l0 = float(v_12[num_idx] / v_12[den_idx])

    num_idx, den_idx = max(x1_ratio_candidates, key=lambda nd: abs(v_12[nd[1]]))
    if abs(v_12[den_idx]) < 1e-12:
        return None
    x_l1 = float(v_12[num_idx] / v_12[den_idx])

    q_l0 = 2.0 * np.arctan(x_l0)
    q_l1 = 2.0 * np.arctan(x_l1)

    # Step 4: solve Q v_right = P_eff @ v_left for v_right via the
    # precomputed pseudoinverse (faster than per-branch lstsq).
    s_lin, c_lin = float(np.sin(q_lin)), float(np.cos(q_lin))
    s_l0, c_l0 = float(np.sin(q_l0)), float(np.cos(q_l0))
    s_l1, c_l1 = float(np.sin(q_l1)), float(np.cos(q_l1))
    v_left = np.array(
        [s_l0 * s_l1, s_l0 * c_l1, c_l0 * s_l1, c_l0 * c_l1, s_l0, c_l0, s_l1, c_l1, 1.0]
    )
    p_eff = p_sin * s_lin + p_cos * c_lin + p_one  # 14x9
    rhs = p_eff @ v_left  # 14
    v_right = q_pinv @ rhs

    # v_right = (s_rb0 s_rb1, s_rb0 c_rb1, c_rb0 s_rb1, c_rb0 c_rb1,
    #            s_rb0, c_rb0, s_rb1, c_rb1)
    s_r0, c_r0 = float(v_right[4]), float(v_right[5])
    s_r1, c_r1 = float(v_right[6]), float(v_right[7])
    q_r0 = float(np.arctan2(s_r0, c_r0))
    q_r1 = float(np.arctan2(s_r1, c_r1))

    # Assemble what we know.
    q_recovered = np.zeros(6, dtype=np.float64)
    q_recovered[linearity_joint] = q_lin
    q_recovered[lb0] = q_l0
    q_recovered[lb1] = q_l1
    q_recovered[rb0] = q_r0
    q_recovered[rb1] = q_r1

    # Step 5: recover drop joint from FK residual.
    # T_target = (chain joints 0..drop_joint-1) @ A_drop(q_drop) @ (chain joints drop_joint+1..5)
    # => A_drop(q_drop) = chain_before^{-1} @ T_target @ chain_after^{-1}
    # The R_z(q_drop) factor at the start of A_drop puts (c_drop, s_drop) in
    # column 0 rows 0,1 of A_drop_residual.
    chain_before = np.eye(4)
    for i in range(drop_joint):
        chain_before = chain_before @ _dh_matrix_num(
            float(q_recovered[i]), float(alpha[i]), float(a[i]), float(d[i])
        )
    chain_after = np.eye(4)
    for i in range(drop_joint + 1, 6):
        chain_after = chain_after @ _dh_matrix_num(
            float(q_recovered[i]), float(alpha[i]), float(a[i]), float(d[i])
        )
    a_drop_residual = np.linalg.solve(chain_before, t_target) @ np.linalg.inv(chain_after)
    q_drop = float(np.arctan2(a_drop_residual[1, 0], a_drop_residual[0, 0]))
    q_recovered[drop_joint] = q_drop

    # FK closure check using the chain matrices we already have. This
    # replaces the redundant ``_fk_dh(q, dh)`` call that used to live in
    # ``solve_all_ik`` -- re-using ``chain_before`` and ``chain_after``
    # plus a fresh ``A_drop`` matmul is one matmul instead of six.
    a_drop = _dh_matrix_num(
        q_drop, float(alpha[drop_joint]), float(a[drop_joint]), float(d[drop_joint])
    )
    fk = chain_before @ a_drop @ chain_after
    fk_err = float(np.linalg.norm(fk - t_target))
    return q_recovered, fk_err


def back_substitute(
    x_lin: float,
    eigvec_24: NDArray[np.complex128],
    p_sin: NDArray[np.float64],
    p_cos: NDArray[np.float64],
    p_one: NDArray[np.float64],
    q_mat: NDArray[np.float64],
    dh: DhParams,
    t_target: NDArray[np.float64],
    metadata: dict[str, object] | None = None,
) -> NDArray[np.float64] | None:
    """Recover ``(q_0, ..., q_5)`` from a real ``x_lin`` root and its eigenvector.

    Generalized for AE-3 (#70) leftvar choice via the metadata dict
    (defaults to the original Manocha-Canny linearity=q_2 case).

    Algorithm (per Manocha-Canny IV-C/IV-D, generalized over the leftvar):

    1. From x_lin: ``q_lin = 2 atan(x_lin)``  (lin = metadata["linearity_joint"]).
    2. From eigenvector: top half is v_12 (in canonical monomial ordering of
       the v_left bilinear pair).
    3. From v_12: ``x_lb0 = v_12[5] / v_12[8]`` and ``x_lb1 = v_12[7] / v_12[8]``
       where (lb0, lb1) = metadata["left_bilinear"]. ``q_lb0 = 2 atan(x_lb0)``, etc.
    4. Solve the 14x8 ``Q v_right = P_eff @ v_left`` for v_right via LSQ; recover
       ``(q_rb0, q_rb1)`` from ``v_right[4:8]`` via atan2 where (rb0, rb1) =
       metadata["right_bilinear"].
    5. Recover the drop joint from FK residual:
       ``A_drop(q_drop) = (FK_chain_before)^{-1} @ T_target @ (FK_chain_after)^{-1}``
       then ``q_drop = atan2(A_drop[1, 0], A_drop[0, 0])`` from the R_z(q_drop)
       factor at the start of A_drop.

    Returns ``None`` on numerical failure. Caller filters.

    Hot-path callers should prefer :func:`_back_substitute_inner` directly,
    passing the precomputed ``q_pinv = pinv(q_mat)`` so the SVD doesn't
    repeat per branch (``solve_all_ik`` already does this).
    """
    if metadata is None:
        # Default to original MC: linearity=q_2, v_left=(q_3,q_4), v_right=(q_0,q_1), drop=q_5.
        metadata = {
            "linearity_joint": 2,
            "left_bilinear": (3, 4),
            "right_bilinear": (0, 1),
            "drop_joint": 5,
        }
    q_pinv = np.linalg.pinv(q_mat).astype(np.float64)
    result = _back_substitute_inner(
        x_lin, eigvec_24, p_sin, p_cos, p_one, q_pinv, dh, t_target, metadata
    )
    if result is None:
        return None
    return result[0]


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


def _spatial_jacobian(q: NDArray[np.float64], dh: DhParams) -> NDArray[np.float64]:
    """6x6 spatial Jacobian for the standard-DH 6R chain at config q.

    Column i is joint i's screw axis in the world frame at q:
        J[:3, i] = z_i x (p_e - p_i)       (linear velocity component)
        J[3:, i] = z_i                       (angular velocity component)
    where ``z_i`` is the joint i axis in world frame at q, and ``p_i`` is
    the origin of frame i in world frame at q.

    For DH, frame i is reached by ``T_{0..i} = A_0 A_1 ... A_{i-1}`` (just
    before joint i acts). So z_i = T_{0..i}[:3, 2] and p_i = T_{0..i}[:3, 3].
    p_e = full FK at q = T_{0..6}[:3, 3].
    """
    alpha, a, d = dh
    Ts: list[NDArray[np.float64]] = [np.eye(4)]
    for i in range(6):
        Ts.append(Ts[-1] @ _dh_matrix_num(float(q[i]), float(alpha[i]), float(a[i]), float(d[i])))
    p_e = Ts[6][:3, 3]
    J = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        z_i = Ts[i][:3, 2]
        p_i = Ts[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_e - p_i)
        J[3:, i] = z_i
    return J


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
    max_iters: int = 15,
    step_clip: float = 0.5,
    return_trajectory: bool = False,
) -> tuple[NDArray[np.float64], int] | tuple[NDArray[np.float64], int, list[float]] | None:
    """Hand-rolled Newton refinement on FK closure with analytical Jacobian.

    For seeds within ~30\u00b0 of a solution, converges to machine precision in
    1-5 iters via Newton-Raphson on the SE(3) log residual:

        r(q) = se3_log(T_target @ T(q)^{-1})    -- 6-vec residual
        J_s(q) = spatial Jacobian               -- closed-form, 6x6
        dq = solve(J_s, r)                      -- single LAPACK call
        q  = q + dq                             -- step (clipped to ``step_clip``)

    Termination:
      - ||r|| < fk_atol  ->  return (q, iters)
      - max_iters hit without convergence  ->  return None

    No divergence-abort: Newton can have non-monotonic behavior near a
    saddle / step-clipped trajectory, and aggressive early termination
    misses recovery. Trust max_iters + final residual check instead.

    No scipy wrapper, no relative tolerance heuristics, no central-difference
    fallback. ~50x faster than the scipy LM path on cases where 1-5 iters
    suffice (which is virtually all reasonable seeds when J_s is exact).

    :param return_trajectory: if True, also return per-iter residual norms
        for diagnostic purposes.
    """
    q = q0.astype(np.float64).copy()
    norms: list[float] = []
    for it in range(max_iters):
        t_q = _fk_dh(q, dh)
        t_diff = t_target @ np.linalg.inv(t_q)
        r = _se3_log_residual(t_diff)
        norm = float(np.linalg.norm(r))
        norms.append(norm)
        if norm < fk_atol:
            if return_trajectory:
                return (q, it, norms)
            return (q, it)
        j_s = _spatial_jacobian(q, dh)
        try:
            dq = np.linalg.solve(j_s, r)
        except np.linalg.LinAlgError:
            # Singular Jacobian (kinematic singularity) -> Tikhonov-damped LSQ.
            damping = max(1e-9, 1e-6 * norm)
            jtj = j_s.T @ j_s + damping * np.eye(6)
            dq = np.linalg.solve(jtj, j_s.T @ r)
        dq = np.clip(dq, -step_clip, step_clip)
        q = q + dq
    # Final convergence check.
    t_check = _fk_dh(q, dh)
    final_r = float(np.linalg.norm(_se3_log_residual(t_target @ np.linalg.inv(t_check))))
    norms.append(final_r)
    if final_r > fk_atol:
        return None
    if return_trajectory:
        return (q, max_iters, norms)
    return (q, max_iters)


@lru_cache(maxsize=64)
def _cached_best_leftvar(
    alpha: tuple[float, ...],
    a: tuple[float, ...],
    d: tuple[float, ...],
    candidates: tuple[int, ...] = (0, 1, 2),
) -> int:
    """Cache the leftvar selection per arm. The pathology that determines the
    best leftvar is geometry-driven (DH-only), not pose-driven, so caching on
    the DH tuple alone is correct and gives constant-time lookup after the
    first call.
    """
    dh: DhParams = (np.asarray(alpha), np.asarray(a), np.asarray(d))
    best, _ = pick_best_leftvar(dh, candidates=candidates)
    return best


def pick_best_leftvar(
    dh: DhParams,
    *,
    test_pose: NDArray[np.float64] | None = None,
    candidates: tuple[int, ...] = (0, 1, 2),
    cond_threshold: float = 1e10,
) -> tuple[int, dict[int, float]]:
    """AE-3 (#70). Try each candidate leftvar, return the one with the smallest
    ``cond(m_quad)`` at a representative test pose.

    Per-arm cost: one symbolic derivation per candidate (~30-100 s each;
    cached). Per-IK cost after selection: same as fixed-leftvar.

    For non-Pieper 6R with structurally-awkward joints (60-deg twists, etc.)
    this can drop ``cond(m_quad)`` by 10+ orders of magnitude vs the default
    leftvar choice. See ``reference_ae3_leftvar_intuition`` memory entry for
    the structural intuition.

    :param dh: Tuple ``(alpha, a, d)`` of length-6 numpy arrays.
    :param test_pose: 4x4 pose at which to measure conditioning. If None,
        uses ``FK(q*=0.1*range)``: a generic non-trivial pose.
    :param candidates: Which linearity_joint values to try. Default ``(0,1,2)``
        covers the natural-orientation splits.
    :param cond_threshold: If the *best* candidate still has cond above this,
        emit a warning -- arm is genuinely hard, AE-4 / LM polish needed.
    :returns: ``(best_linearity_joint, {linearity_joint: cond})``.
    """
    _alpha, _a, _d = dh
    if test_pose is None:
        # Build a generic non-trivial pose via FK at small joint angles.
        q_test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        test_pose = _fk_dh(q_test, dh)

    conds: dict[int, float] = {}
    for lj in candidates:
        try:
            p_sin, p_cos, p_one, q_mat = build_pq(
                dh,
                test_pose,
                linearity_joint=lj,
                apply_so3=False,
            )
        except Exception:
            conds[lj] = float("inf")
            continue
        e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
        e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
        m_quad, _, _ = build_m_matrix(e_quad, e_lin, e_const)
        conds[lj] = float(np.linalg.cond(m_quad))
    best = min(conds, key=lambda k: conds[k])
    return best, conds


def solve_all_ik(
    dh: DhParams,
    t_target: NDArray[np.float64],
    *,
    fk_atol: float = 1e-6,
    dedup_atol: float = 1e-3,
    linearity_joint: int | str = "auto",
    apply_so3: bool = False,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    solver_name: str = "ikgeo._raghavan_roth",
    max_solutions: int | None = None,
) -> tuple[list[Solution], bool]:
    """Run the full Raghavan-Roth pipeline and return all valid IK solutions.

    Pipeline (per Manocha-Canny IV):
      1. Build the 14-row (P, Q) system at the given leftvar (cached per arm).
      2. SVD-eliminate v_right to get 6x9 E.
      3. Weierstrass for the linearity variable + basis change for the v_left
         bilinear pair -> 6x9 quadratic-in-x_lin system in v_left_x basis.
      4. Build 12x12 M(x_lin) via the x_lb0-shift block construction.
      5. Companion-matrix eigenvalue route -> up to 16 real x_lin roots
         (filtering 8 spurious near +/-i and complex-conjugate pairs).
      6. Back-substitute each root + leftvar metadata -> candidate (q_0..q_5).
      7. FK-validate each candidate; drop those with residual > ``fk_atol``.
         When ``allow_refinement=True``, candidates that miss ``fk_atol``
         algebraically get one :func:`~ssik.refinement.lm_refine` pass before
         the drop decision; the resulting :class:`Solution` records
         ``refinement_used="lm"`` and ``refinement_iters``.
      8. Deduplicate via wrap-to-pi joint-distance threshold.

    :param dh: Tuple ``(alpha, a, d)`` of length-6 numpy arrays.
    :param t_target: 4x4 target end-effector pose.
    :param fk_atol: max allowed ``||FK(q) - t_target||_F`` for a solution to be kept.
    :param dedup_atol: per-joint wrap-to-pi tolerance below which two
        solutions collapse to one.
    :param linearity_joint: AE-3 (#70) leftvar choice (0, 1, or 2). For
        non-Pieper arms with structurally-awkward joints, picking the right
        leftvar can drop ``cond(m_quad)`` by 14+ orders of magnitude (see
        JACO 2 case in #70 / `reference_ae3_leftvar_intuition`).
    :param apply_so3: AE-4 (#71) SO(3) identity reduction.
    :param allow_refinement: opt into Newton-on-spatial-Jacobian polish for
        candidates that don't meet ``fk_atol`` algebraically. Default off
        per #74 (refinement is a separate, transparent layer).
    :param refinement_max_iters: cap on Newton iterations per candidate
        when ``allow_refinement=True``.
    :param solver_name: tag stored on each returned :class:`Solution` for
        provenance when results pass through a dispatcher.
    :param max_solutions: optional early-exit cap (#198). When set, stop
        back-substituting roots once the post-dedup count reaches the cap.
        Default ``None`` enumerates all up-to-16 algebraic branches.
    :returns: ``(solutions, is_ls)`` where solutions is a list of
        :class:`~ssik.core.solution.Solution` and ``is_ls`` is True iff no
        candidate survived FK validation.
    """
    if max_solutions is not None and max_solutions < 1:
        raise ValueError(f"max_solutions must be >= 1 or None; got {max_solutions}")
    if linearity_joint == "auto":
        # AE-3 (#70): pick the best leftvar (cached per arm; the structural
        # pathology that determines the choice is geometry-driven, not
        # pose-driven, so we don't pass t_target here).
        alpha, a, d = dh
        linearity_joint = _cached_best_leftvar(
            tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist())
        )
    if not isinstance(linearity_joint, int):
        raise ValueError(f"linearity_joint must be int or 'auto'; got {linearity_joint!r}")

    p_sin, p_cos, p_one, q_mat, meta = build_pq(
        dh,
        t_target,
        linearity_joint=linearity_joint,
        apply_so3=apply_so3,
        return_metadata=True,
    )
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)
    # Use the M\u00f6bius-fallback variant: well-conditioned -> direct path; otherwise
    # try a few random reparameterizations to recondition the leading matrix.
    roots, eigvecs = solve_x2_roots_mobius(m_quad, m_lin, m_const)

    fk_fn = lambda q: _fk_dh(q, dh)  # noqa: E731
    jacobian_fn = lambda q: _spatial_jacobian(q, dh)  # noqa: E731

    # Precompute pinv(q_mat) once; reused across all back-substitution
    # branches. Saves the SVD per branch (#86 Tier 2).
    q_pinv = np.linalg.pinv(q_mat).astype(np.float64)

    candidates: list[Solution] = []
    for x_lin_root, eigvec in zip(roots, eigvecs, strict=True):
        bs_result = _back_substitute_inner(
            x_lin_root,
            eigvec,
            p_sin,
            p_cos,
            p_one,
            q_pinv,
            dh,
            t_target,
            meta,
        )
        if bs_result is None:
            continue
        # ``_back_substitute_inner`` already computed ``fk_err_alg`` from
        # the chain matrices it had to build for drop-joint recovery, so
        # we don't pay for a separate ``_fk_dh(q_cand, dh)`` here (#86).
        q_cand, fk_err_alg = bs_result
        appended = False
        if fk_err_alg <= fk_atol:
            candidates.append(
                Solution(
                    q=q_cand,
                    fk_residual=fk_err_alg,
                    refinement_used="none",
                )
            )
            appended = True
        elif allow_refinement:
            # Opt-in refinement: lm_refine polishes the algebraic seed.
            refined = lm_refine(
                q_cand,
                fk_fn,
                t_target,
                fk_atol=fk_atol,
                max_iters=refinement_max_iters,
                jacobian_fn=jacobian_fn,
            )
            if refined is not None:
                q_refined, fk_resid, _iters = refined
                candidates.append(
                    Solution(
                        q=q_refined,
                        fk_residual=fk_resid,
                        refinement_used="lm",
                    )
                )
                appended = True

        # Early-exit gate (#198): once we have enough unique solutions, stop
        # back-substituting remaining algebraic roots.
        if appended and max_solutions is not None and len(candidates) >= max_solutions:
            deduped_partial = dedup_by_wrap_close(candidates, dedup_atol)
            if len(deduped_partial) >= max_solutions:
                return deduped_partial[:max_solutions], False

    # Deduplicate with wrap-to-pi joint distance. Keep the lower-fk_residual
    # candidate when two collapse.
    solutions: list[Solution] = []
    for cand in candidates:
        dup_idx = None
        for j, existing in enumerate(solutions):
            diffs = [
                abs(((float(cand.q[i] - existing.q[i]) + np.pi) % (2 * np.pi)) - np.pi)
                for i in range(len(cand.q))
            ]
            if max(diffs) < dedup_atol:
                dup_idx = j
                break
        if dup_idx is None:
            solutions.append(cand)
        elif cand.fk_residual < solutions[dup_idx].fk_residual:
            solutions[dup_idx] = cand

    if max_solutions is not None and len(solutions) > max_solutions:
        solutions = solutions[:max_solutions]
    return solutions, len(solutions) == 0
