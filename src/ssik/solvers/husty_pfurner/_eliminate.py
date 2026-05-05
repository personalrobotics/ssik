"""Elimination pipeline for Husty-Pfurner: ``T(u) + T(w) -> r(u)``.

Phase 5d steps 2-5 of #162. Given the parametrised 3-spaces ``T(u)`` (left
chain) and ``T(w)`` (right chain) -- each a 4x8 matrix of hyperplanes whose
coefficients are linear in the parametrising joint variable -- this module
implements Capco et al. 2019 Section 5 to produce the candidate values of
the parametrising joint ``u = v_1`` that close the IK chain:

1. Stack ``T(u)`` and ``T(w)`` into an 8x8 system over ``C[u, w]``.
2. Pick 7 of the 8 hyperplanes; solve for the projective point
   ``P(u, w) in P^7`` via Cramer's rule (so the 8 components are
   polynomials, not rationals).
3. Substitute ``P(u, w)`` into the Study quadric to get a bivariate
   ``f(u, w) = P_0 P_4 + P_1 P_5 + P_2 P_6 + P_3 P_7``.
4. Substitute ``P(u, w)`` into the unused 8th hyperplane to get a
   bivariate ``g(u, w)``.
5. Find common roots of ``f(u, w) = g(u, w) = 0`` -- the candidate
   ``(u, w)`` values are IK solutions.

Step 5 is implemented via the **Sylvester matrix pencil eigenvalue**
trick (Manocha-Canny 1991/1994): the 10x10 Sylvester matrix
``S(u) = sum_k S_k u^k`` (degree <=8 in u) is linearised to an 80x80
generalised eigenvalue problem; finite real eigenvalues are the
candidate ``u`` values. This avoids the polynomial-coefficient
extraction that plagues monomial-basis polyfit when ``det S(u)`` spans
many orders of magnitude.

Algorithmic references:

- Capco, Loquias, Manongsong, Nemenzo (2019), arXiv 1906.07813,
  Section 5 (Parts 5.1, 5.2, 5.3) -- the algebraic derivation.
- Manocha & Canny, "Multipolynomial resultants and linear algebra"
  (J. Symbolic Computation, 1994) -- the matrix pencil trick that
  makes the algebra numerically tractable.

Runtime architecture
--------------------

Two implementations live in this module:

- :func:`eliminate_uw` -- slow sympy reference. Subresultant PRS over
  ``Q[u, w]`` -- exact arithmetic but multi-second. Used by tests as
  the ground-truth oracle on toy/symbolic instances only.

- :func:`eliminate_uw_numeric` -- the production hot path. Pure numpy +
  scipy.linalg + scipy.signal arithmetic, no sympy. Per-IK cost:

  * ~1 ms Cramer cofactors via 5x4 evaluation-interpolation grid
    (the bivariate degree of each cofactor is (<=4, <=3), so 20
    numeric 7x7 LU solves fix the polynomial exactly).
  * ~0.5 ms ``f = P*P`` Study quadric and ``g = c.P`` dropped row,
    via :func:`scipy.signal.convolve2d`.
  * ~8 ms generalised eigenvalue problem on the 80x80 pencil.

  Total ~10 ms per IK pose, comfortably under the 100 ms abort gate.

Both paths consume the same ``EliminatePrecompute`` per-arm cache
(DH params baked numerically; ``sigma_E`` plugged in at IK call time).
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
import sympy as sp
from numpy.typing import NDArray
from scipy.signal import convolve2d  # type: ignore[import-untyped]

__all__ = [
    "EliminatePrecompute",
    "build_pencil_tensor",
    "compute_fg_numeric",
    "eliminate_uw",
    "eliminate_uw_numeric",
    "eliminate_uw_pairs",
    "evaluate_poly",
    "extract_uv_linear_tensor",
    "polynomial_residual",
    "precompute_from_sympy",
    "precompute_rrr_chain",
    "solve_pencil_eigenvalues",
]


# =============================================================================
# Sympy reference path (slow, used as oracle in tests).
# =============================================================================


def eliminate_uw(
    T_u_sym: sp.Matrix,
    T_w_sym: sp.Matrix,
    u_sym: sp.Symbol,
    w_sym: sp.Symbol,
    *,
    drop_idx: int = 7,
) -> sp.Poly:
    """Sympy reference implementation of the HP elimination pipeline.

    Slow (>60s on real DH); use :func:`eliminate_uw_numeric` in production.
    Kept as the ground-truth oracle for tests on small/symbolic instances.

    See module docstring for the algorithm. ``drop_idx`` selects which of
    the 8 stacked hyperplanes becomes ``g(u, w)``; the other 7 are used
    for Cramer's rule. Default ``drop_idx=7`` drops the last ``T(w)`` row.

    :raises ValueError: input shape mismatch or singular Cramer system.
    """
    if T_u_sym.shape != (4, 8):
        raise ValueError(f"T_u_sym must be 4x8, got {T_u_sym.shape}")
    if T_w_sym.shape != (4, 8):
        raise ValueError(f"T_w_sym must be 4x8, got {T_w_sym.shape}")
    if not 0 <= drop_idx < 8:
        raise ValueError(f"drop_idx must be in [0, 8), got {drop_idx}")

    M_8 = sp.Matrix.vstack(T_u_sym, T_w_sym)

    keep_rows = [i for i in range(8) if i != drop_idx]
    M_7 = M_8[keep_rows, :]

    A = M_7[:, 1:]
    rhs = -M_7[:, 0]

    det_A = A.det()
    if det_A == 0:
        raise ValueError(
            f"Cramer 7x7 system is singular for drop_idx={drop_idx}; "
            f"retry with a different drop choice."
        )
    P_components: list[sp.Expr] = [det_A]
    for i in range(7):
        A_i = A.copy()
        A_i[:, i] = rhs
        P_components.append(A_i.det())
    P = sp.Matrix(P_components)

    f_uw = sp.expand(P[0] * P[4] + P[1] * P[5] + P[2] * P[6] + P[3] * P[7])
    c_dropped = M_8[drop_idx, :]
    g_uw = sp.expand(sum(c_dropped[i] * P[i] for i in range(8)))

    f_poly_w = sp.Poly(f_uw, w_sym)
    g_poly_w = sp.Poly(g_uw, w_sym)
    r_uw = sp.resultant(f_poly_w, g_poly_w, w_sym)
    return sp.Poly(r_uw, u_sym)


# =============================================================================
# Numeric hot path: per-arm precompute and bivariate-polynomial primitives.
# =============================================================================


# Fixed evaluation grid for the Cramer-det interpolation. 5 u-points and
# 4 w-points are exactly enough to recover a degree-(4, 3) polynomial.
# Symmetric-around-zero choices keep the Vandermonde well-conditioned.
_CRAMER_U_GRID = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
_CRAMER_W_GRID = np.array([-1.5, -0.5, 0.5, 1.5])
_CRAMER_VU_INV = np.linalg.inv(np.vander(_CRAMER_U_GRID, increasing=True))  # (5, 5)
_CRAMER_VW_INV = np.linalg.inv(np.vander(_CRAMER_W_GRID, increasing=True))  # (4, 4)


class EliminatePrecompute:
    """Per-arm precomputed tensors for the numeric HP elimination.

    DH parameters are baked numerically; only ``sigma_E`` (the 8-vec
    Study DQ of the target pose) is plugged in at IK call time.

    :ivar T_u: shape ``(4, 8, 2)``; last axis is ``[const, u-coeff]``.
    :ivar T_w_pre: shape ``(4, 8, 2)``; last axis is ``[const, w-coeff]``.
        This is ``T(v_6)`` BEFORE the ``sigma_E^*`` left-multiplication.
        At IK call time, the runtime multiplies by an 8x8 matrix derived
        from ``sigma_E`` to get the final ``T_w``.
    """

    __slots__ = ("T_u", "T_w_pre")

    def __init__(self, T_u: NDArray[np.float64], T_w_pre: NDArray[np.float64]) -> None:
        if T_u.shape != (4, 8, 2):
            raise ValueError(f"T_u must be (4, 8, 2), got {T_u.shape}")
        if T_w_pre.shape != (4, 8, 2):
            raise ValueError(f"T_w_pre must be (4, 8, 2), got {T_w_pre.shape}")
        self.T_u = T_u.astype(np.float64, copy=True)
        self.T_w_pre = T_w_pre.astype(np.float64, copy=True)


def extract_uv_linear_tensor(M_sym: sp.Matrix, var: sp.Symbol) -> NDArray[np.float64]:
    """Extract the ``(rows, cols, 2)`` tensor of [const, linear] coefficients
    from a sympy matrix whose entries are linear in ``var`` with all other
    symbols already substituted numerically.

    :raises ValueError: any entry has degree >1 in ``var`` or contains
        unsubstituted free symbols other than ``var``.
    """
    rows, cols = M_sym.shape
    out = np.zeros((rows, cols, 2), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            entry = sp.expand(M_sym[i, j])
            free = entry.free_symbols - {var}
            if free:
                raise ValueError(
                    f"entry [{i},{j}] has unsubstituted free symbols {free}; "
                    f"all DH/sigma must be numeric"
                )
            poly = sp.Poly(entry, var) if entry != 0 else None
            if poly is not None and poly.degree() > 1:
                raise ValueError(f"entry [{i},{j}] has degree {poly.degree()} > 1 in {var}")
            if poly is None:
                continue
            coeffs = poly.all_coeffs()
            if len(coeffs) == 1:
                out[i, j, 0] = float(coeffs[0])
            else:
                out[i, j, 0] = float(coeffs[1])
                out[i, j, 1] = float(coeffs[0])
    return out


def precompute_from_sympy(
    T_u_sym: sp.Matrix,
    u_sym: sp.Symbol,
    T_w_pre_sym: sp.Matrix,
    w_sym: sp.Symbol,
) -> EliminatePrecompute:
    """Build per-arm :class:`EliminatePrecompute` from sympy matrices.

    :param T_u_sym: 4x8 sympy matrix; entries linear in ``u_sym``, DH numeric.
    :param T_w_pre_sym: 4x8 sympy matrix; entries linear in ``w_sym``, DH
        numeric, but BEFORE ``sigma_E^*`` left-multiplication. (``T_w`` then
        equals ``T_w_pre @ M(sigma_E)`` at IK time -- linear in ``sigma_E``.)
    """
    T_u = extract_uv_linear_tensor(T_u_sym, u_sym)
    T_w_pre = extract_uv_linear_tensor(T_w_pre_sym, w_sym)
    return EliminatePrecompute(T_u, T_w_pre)


_TV1_DEGEN_WARNED: set[tuple[float, ...]] = set()
_HP_DEGEN_TOL = 1e-9
_LOG = __import__("logging").getLogger(__name__)


def _check_hp_preconditions(a_1: float, l_1: float, a_2: float, l_2: float) -> None:
    """Detect Capco eq. (5) degenerate cases for the RRR pattern and warn.

    Verified by reading Capco's ``which_case.py:74-80`` directly (see
    ``reference_capco_hp_dispatch`` memory entry). The RRR dispatch is:

    * ``T(v_1)`` applies when ``a_2 != 0 AND l_2 != 0``
      (eq. 5 simplified-form non-degenerate).
    * ``T(v_3)`` applies when ``(a_2 = 0 OR l_2 = 0) AND a_1 != 0 AND l_1 != 0``.
    * ``T(v_2)`` (sub-case-keyed) applies when both are degenerate:
      ``(a_2 = 0 OR l_2 = 0) AND (a_1 = 0 OR l_1 = 0)``.

    The previously-shipped check used the RRP-specific ``|l_2| = ±1``
    condition by mistake. Corrected here.

    This function logs a one-time WARNING per degenerate DH-tuple so
    callers can audit which arms/configurations are affected without
    spamming on every IK call. It deliberately does NOT raise -- the
    partial IK set ``T(v_1)`` returns on these arms is still useful;
    full IK coverage requires the missing parametrizations.
    """
    a2_zero = abs(a_2) < _HP_DEGEN_TOL
    l2_zero = abs(l_2) < _HP_DEGEN_TOL
    a1_zero = abs(a_1) < _HP_DEGEN_TOL
    l1_zero = abs(l_1) < _HP_DEGEN_TOL
    if not (a2_zero or l2_zero):
        return  # T(v_1) precondition holds; no warning needed.
    key = (round(a_1, 9), round(l_1, 9), round(a_2, 9), round(l_2, 9))
    if key in _TV1_DEGEN_WARNED:
        return
    _TV1_DEGEN_WARNED.add(key)
    if a1_zero or l1_zero:
        _LOG.warning(
            "husty_pfurner: HP T(v_1) parametrization degenerate for this DH "
            "(a_1=%.3e, l_1=%.3e, a_2=%.3e, l_2=%.3e): both eq.5 simplified "
            "preconditions violated -- (a_1=0 OR l_1=0) AND (a_2=0 OR l_2=0). "
            "Capco's RRR Tv2 sub-case keyed by which DH params are zero is "
            "required (see #176). Locked-7R configurations on Franka / KUKA "
            "iiwa LBR / xArm7 hit the case [a_1=0, a_2=0]. Proceeding with "
            "T(v_1) anyway -- partial IK set is still useful but some "
            "branches are silently missed.",
            a_1, l_1, a_2, l_2,
        )
    else:
        _LOG.warning(
            "husty_pfurner: HP T(v_1) parametrization degenerate for this DH "
            "(a_1=%.3e, l_1=%.3e, a_2=%.3e, l_2=%.3e): a_2=0 OR l_2=0 with "
            "a_1, l_1 != 0 -- T(v_3) (already coded in _constraints.py but not "
            "yet wired into back-substitution, see #180) is the well-conditioned "
            "alternative. Proceeding with T(v_1) -- some IK branches may be "
            "silently missed.",
            a_1, l_1, a_2, l_2,
        )


@functools.lru_cache(maxsize=128)
def precompute_rrr_chain(
    a_1: float,
    l_1: float,
    d_2: float,
    a_2: float,
    l_2: float,
    d_3: float,
    a_3: float,
    l_3: float,
    d_4: float,
    a_4: float,
    l_4: float,
    d_5: float,
    a_5: float,
    l_5: float,
) -> EliminatePrecompute:
    """One-shot per-arm precompute for the 6R/RRR case.

    Wraps the sympy boilerplate that builds ``T(v_1)`` (left chain) and
    ``T(v_6)`` (right chain BEFORE ``sigma_E^*`` left-multiplication) from
    DH parameters, extracts the ``(4, 8, 2)`` numeric tensors, and packs
    them into an :class:`EliminatePrecompute`.

    Joints are numbered 1..6. ``v = tan(theta/2)`` and ``l = tan(alpha/2)``.
    Joints 1 and 6 are the parametrising joints (``u = v_1``, ``w = v_6``);
    joint 6 is assumed to have ``a_6 = d_6 = l_6 = 0`` (Capco convention --
    EE offset is absorbed into ``sigma_E`` at IK call time).

    Wrapped in :func:`functools.lru_cache`: the DH parameters are arm
    constants, so a typical end-user IK loop calls ``precompute_rrr_chain``
    with the same 14 floats every IK and only pays the ~50 ms sympy
    boilerplate once per arm. The cache key is the 14-tuple of Python
    floats; downstream code never mutates the returned tensors so sharing
    the same instance across calls is safe.

    HP coverage matrix (RRR pattern; Capco eq. 5 simplified-form
    degeneracy criterion verified against ``which_case.py:74-80``):

    +----------+----------------------------------------------------+--------+
    | Variant  | Applies when                                       | Status |
    +==========+====================================================+========+
    | T(v_1)   | ``a_2 != 0 ∧ l_2 != 0``                            | OK     |
    | T(v_3)   | ``(a_2 = 0 ∨ l_2 = 0) ∧ a_1 != 0 ∧ l_1 != 0``      | #180   |
    | T(v_2)   | ``(a_2 = 0 ∨ l_2 = 0) ∧ (a_1 = 0 ∨ l_1 = 0)``,     | #176   |
    |          | sub-case keyed by which DH param(s) are zero       |        |
    |          | -- 4 RRR sub-cases: ``[a_1=0,a_2=0]``,             |        |
    |          | ``[a_1=0,l_2=0]``, ``[l_1=0,a_2=0]``,              |        |
    |          | ``[l_1=0,l_2=0]``                                  |        |
    +----------+----------------------------------------------------+--------+

    The RRR dispatch is on ``a_2 = 0 ∨ l_2 = 0`` (eq. 5 simplified-form
    degeneracy). The ``|l_2| = ±1`` rule applies to RRP, not RRR --
    earlier ssik docstrings had this wrong.

    Currently this function calls ``T(v_1)`` unconditionally. When the
    DH-precondition for T(v_1) is violated (i.e. T(v_1) lies in S), the
    Sylvester pencil constructed downstream is rank-deficient and **some
    IK branches are silently missed**. The current implementation logs
    a warning at WARNING level the first time each degenerate DH-tuple
    is seen so callers can audit their fixtures, but proceeds with T(v_1)
    anyway -- the partial IK set it returns is still useful.

    Issues #176 (T(v_2) for RRR, all 4 sub-cases) and #177
    (T(v_4)/T(v_5) right-mirrors) close this gap. Every locked-7R
    configuration on Franka / KUKA iiwa LBR / xArm7 hits **RRR Tv2
    sub-case [a_1=0, a_2=0]** -- implementing just that one sub-case
    unblocks 12/14 lock configs per arm.

    :raises ValueError: if any DH-derived T(v_i) entry has degree > 1 in
        the parametrising symbol (indicates a degenerate DH where the
        pure-tan-half-angle parametrisation breaks down).
    """
    _check_hp_preconditions(a_1, l_1, a_2, l_2)

    from ssik.solvers.husty_pfurner._constraints import (
        _V1_SYM,
        _V6_SYM,
        tv1_symbolic_in_v1,
    )

    T_u_sym = tv1_symbolic_in_v1(a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3).subs(_V1_SYM, _V1_SYM)
    T_w_pre_sym = tv1_symbolic_in_v1(
        a_1=-a_5,
        l_1=-l_5,
        d_2=-d_5,
        a_2=-a_4,
        l_2=-l_4,
        d_3=-d_4,
        a_3=0.0,
        l_3=0.0,
    ).subs(_V1_SYM, -_V6_SYM)
    return precompute_from_sympy(T_u_sym, _V1_SYM, T_w_pre_sym, _V6_SYM)


def _apply_sigma_e_to_tw_pre(
    T_w_pre: NDArray[np.float64], sigma_E: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply the ``sigma_E^*`` left-multiplication to the pre-multiplied
    ``T(v_6)`` tensor, returning the final ``T_w`` of shape ``(4, 8, 2)``.

    The HP construction is ``T_w = T_w_pre @ M(sigma_E^*)`` where the
    matrix ``M`` acts on the 8-vec column index. Linearity in ``w`` is
    preserved.
    """
    # Lazy import to avoid circular dependency.
    from ssik.solvers.husty_pfurner._constraints import _dq_left_mult_matrix

    if sigma_E.shape != (8,):
        raise ValueError(f"sigma_E must be 8-vec, got {sigma_E.shape}")
    sigma_E_conj = np.array(
        [
            sigma_E[0],
            -sigma_E[1],
            -sigma_E[2],
            -sigma_E[3],
            sigma_E[4],
            -sigma_E[5],
            -sigma_E[6],
            -sigma_E[7],
        ],
        dtype=np.float64,
    )
    M_e = _dq_left_mult_matrix(sigma_E_conj)
    out: NDArray[np.float64] = np.einsum("ijk,jl->ilk", T_w_pre, M_e)
    return out


def _build_full_8x8(
    T_u: NDArray[np.float64], T_w: NDArray[np.float64], u: float, w: float
) -> NDArray[np.float64]:
    """Materialise the 8x8 numeric system at one ``(u, w)`` grid point."""
    out = np.empty((8, 8), dtype=np.float64)
    out[:4] = T_u[..., 0] + u * T_u[..., 1]
    out[4:] = T_w[..., 0] + w * T_w[..., 1]
    return out


def _cramer_8vec_via_interp(
    T_u: NDArray[np.float64],
    T_w: NDArray[np.float64],
    drop_idx: int,
) -> NDArray[np.float64]:
    """Compute the 8-vector ``P(u, w)`` of Cramer cofactors as a
    ``(8, 5, 4)`` tensor of bivariate-polynomial coefficients (axes:
    component, u-power, w-power).

    Algorithm: evaluate the 7x7 system at every (u, w) on the 5x4 grid;
    at each point compute ``det(A)`` and ``A^{-1} rhs`` via one LU; that
    gives the 8 cofactor values. Interpolate back via the precomputed
    Vandermonde inverses.

    :raises np.linalg.LinAlgError: if the system is rank-deficient at some
        grid point (caller should pick a different ``drop_idx``).
    """
    keep = [i for i in range(8) if i != drop_idx]
    P_grid = np.zeros((8, 5, 4), dtype=np.float64)
    for i, ui in enumerate(_CRAMER_U_GRID):
        for j, wj in enumerate(_CRAMER_W_GRID):
            M_full = _build_full_8x8(T_u, T_w, ui, wj)
            M_7 = M_full[keep]
            A = M_7[:, 1:]
            rhs = -M_7[:, 0]
            x = np.linalg.solve(A, rhs)
            d = float(np.linalg.det(A))
            P_grid[0, i, j] = d
            P_grid[1:, i, j] = x * d
    P_u: NDArray[np.float64] = np.einsum("pi,kij->kpj", _CRAMER_VU_INV, P_grid)
    P_coef: NDArray[np.float64] = np.einsum("qj,kpj->kpq", _CRAMER_VW_INV, P_u)
    return P_coef


def _study_quadric_f(P_coef: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute ``f(u, w) = sum_{i=0..3} P_i * P_{i+4}``, returning a
    bivariate-polynomial coefficient tensor of shape ``(9, 7)``.
    """
    out = np.zeros((9, 7), dtype=np.float64)
    for i in range(4):
        out += convolve2d(P_coef[i], P_coef[i + 4])
    return out


def _dropped_row_g(
    T_u: NDArray[np.float64],
    T_w: NDArray[np.float64],
    P_coef: NDArray[np.float64],
    drop_idx: int,
) -> NDArray[np.float64]:
    """Compute ``g(u, w) = c_dropped(u, w) . P(u, w)`` as a bivariate-
    polynomial coefficient tensor of shape ``(6, 5)``.

    ``c_dropped`` is the 8-vector polynomial coefficients of one row of
    the stacked 8x8 system. For ``drop_idx in [0..3]``, c is a ``T(u)``
    row (degree (1, 0) per entry); for ``drop_idx in [4..7]``, c is a
    ``T(w)`` row (degree (0, 1) per entry). The product is therefore
    degree at most ``(5, 3)`` or ``(4, 4)`` respectively; the output
    shape ``(6, 5)`` covers both cases with zeros in unused monomials.
    """
    if drop_idx < 4:
        c = np.zeros((8, 2, 1), dtype=np.float64)
        c[:, 0, 0] = T_u[drop_idx, :, 0]
        c[:, 1, 0] = T_u[drop_idx, :, 1]
    else:
        c = np.zeros((8, 1, 2), dtype=np.float64)
        c[:, 0, 0] = T_w[drop_idx - 4, :, 0]
        c[:, 0, 1] = T_w[drop_idx - 4, :, 1]
    out = np.zeros((6, 5), dtype=np.float64)
    for i in range(8):
        prod = convolve2d(c[i], P_coef[i])
        out[: prod.shape[0], : prod.shape[1]] += prod
    return out


def _compute_fg_with_tw(
    T_u: NDArray[np.float64],
    T_w: NDArray[np.float64],
    drop_idx: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cramer + Study-quadric + dropped-row substitution given the
    pre-computed ``T_u`` and ``T_w`` tensors. Internal hot-path helper:
    ``T_w = _apply_sigma_e_to_tw_pre(pre.T_w_pre, sigma_E)`` is
    drop-independent, so multi-drop callers compute it once and reuse.
    """
    if not 0 <= drop_idx < 8:
        raise ValueError(f"drop_idx must be in [0, 8), got {drop_idx}")
    P_coef = _cramer_8vec_via_interp(T_u, T_w, drop_idx)
    f = _study_quadric_f(P_coef)
    g = _dropped_row_g(T_u, T_w, P_coef, drop_idx)
    return f, g


def compute_fg_numeric(
    pre: EliminatePrecompute,
    sigma_E: NDArray[np.float64],
    *,
    drop_idx: int = 7,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run the Cramer + Study-quadric + dropped-row substitution, returning
    the bivariate-polynomial coefficient tensors of ``f(u, w)`` and
    ``g(u, w)``.

    :returns: ``(f, g)`` where ``f`` is ``(9, 7)`` and ``g`` is ``(6, 5)``.
        Indexing is ``f[i, j] = coeff of u^i w^j``.
    """
    sigma_E_arr = np.asarray(sigma_E, dtype=np.float64)
    T_w = _apply_sigma_e_to_tw_pre(pre.T_w_pre, sigma_E_arr)
    return _compute_fg_with_tw(pre.T_u, T_w, drop_idx)


# =============================================================================
# Sylvester matrix pencil + generalised eigenvalue solve.
# =============================================================================


def build_pencil_tensor(f: NDArray[np.float64], g: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build the Sylvester matrix pencil ``S(u) = sum_d S_d * u^d`` from
    bivariate-polynomial coefficient tensors ``f(u, w)`` and ``g(u, w)``.

    :returns: tensor of shape ``(max_deg_u + 1, n, n)`` where
        ``n = deg_w(f) + deg_w(g)`` and ``S_d`` are the ``u^d`` Sylvester
        coefficient matrices. Sylvester convention: top ``deg_w(g)``
        rows are shifted ``f``-rows (descending in w); bottom ``deg_w(f)``
        rows are shifted ``g``-rows.

    The matrix ``S(u)`` is singular at every common root of f and g, so
    its generalised eigenvalues (over u) are the candidate ``u`` values
    of the IK pipeline.
    """
    deg_w_f = f.shape[1] - 1
    deg_w_g = g.shape[1] - 1
    deg_u_f = f.shape[0] - 1
    deg_u_g = g.shape[0] - 1
    n = deg_w_f + deg_w_g
    max_d = max(deg_u_f, deg_u_g)
    S = np.zeros((max_d + 1, n, n), dtype=np.float64)
    for shift in range(deg_w_g):
        for d in range(f.shape[0]):
            for k in range(f.shape[1]):
                col = shift + (deg_w_f - k)
                S[d, shift, col] += f[d, k]
    for shift in range(deg_w_f):
        for d in range(g.shape[0]):
            for k in range(g.shape[1]):
                col = shift + (deg_w_g - k)
                S[d, deg_w_g + shift, col] += g[d, k]
    return S


def solve_pencil_eigenvalues(
    f: NDArray[np.float64],
    g: NDArray[np.float64],
    *,
    real_tol: float = 1e-3,
    max_magnitude: float = 1e10,
) -> NDArray[np.float64]:
    """Compute approximate candidate ``u`` values as finite real
    generalised eigenvalues of the Sylvester matrix pencil built from
    ``f(u, w)`` and ``g(u, w)``.

    Thin wrapper over the shared
    :func:`ssik._pencil.solve_polynomial_matrix_eigenvalues`. The
    default tolerances are loose; downstream Newton refinement in
    :func:`eliminate_uw_numeric` filters spurious candidates by
    residue, not by tightening these knobs.

    :returns: sorted 1-D array of finite real candidate ``u`` values.
    """
    from ssik._pencil import solve_polynomial_matrix_eigenvalues

    S = build_pencil_tensor(f, g)
    cands, _leakage = solve_polynomial_matrix_eigenvalues(
        S, real_tol=real_tol, max_magnitude=max_magnitude, rescale_variable=True
    )
    return cands


# =============================================================================
# Newton refinement of (u, w) on the bivariate system (f, g) = 0.
#
# The matrix pencil delivers approximate (u_i, w_i) starting points whose
# accuracy is bounded by ``cond(S(u_i)) * machine_eps``. For benign IK poses
# this is ~1e-13; for multiplicity-2+ kinematic singularities or large-
# alpha DH it degrades to ~1e-3. Newton's method on the residue
# ``[f(u, w); g(u, w)] = 0`` polishes any starting guess to machine
# precision in 2-3 iterations (quadratic convergence). This is the
# textbook "tracking + polishing" pattern used in numerical algebraic
# geometry (Bertini, HomotopyContinuation.jl, Manocha-Canny 1994 sec V).
#
# Refinement also gives a free filter on spurious eigenvalues: a true root
# converges to residue ~ machine_eps; a numerical-infinity eigenvalue from
# the pencil's null space fails to converge.
# =============================================================================


# Newton convergence tolerance (relative residue). Roots that converge below
# this are accepted as IK candidates; any starting guess that doesn't reach
# this in ``_NEWTON_MAX_ITER`` iterations is rejected as spurious.
_NEWTON_RESIDUE_TOL = 1e-12

# Quadratic convergence: 1e-3 → 1e-6 → 1e-12 → 1e-24. Three iterations
# clear all benign starting points; five gives slack for poorly-conditioned
# Jacobians without unbounded cost.
_NEWTON_MAX_ITER = 5


def _refine_uw_inline(
    f: NDArray[np.float64],
    g: NDArray[np.float64],
    u0: float,
    w0: float,
    max_iter: int,
    tol: float,
) -> tuple[float, float, float]:
    """Inline 2x2 Newton on ``[f(u, w), g(u, w)] = 0`` with monotone-best
    tracking. Faster equivalent of
    :func:`ssik._pencil.newton_refine_system` for HP-shaped (small,
    fixed-degree) polynomial systems: avoids closure dispatch, pre-builds
    derivatives, and evaluates ``p @ w_pow @ u_pow`` via two
    explicit matvecs per derivative instead of the
    ``np.polynomial.polynomial.polyval2d`` ufunc path.

    Per Newton iter cost in this inlined version:
    ~0.05 ms (1 ms in the closure path), so ~10x cheaper on the inner
    Newton loop. The (u, w) trajectory and convergence behaviour match
    :func:`newton_refine_system` exactly: same monotone-best invariant,
    same residue scaling.

    :returns: ``(u_refined, w_refined, residue)``. Caller must reject
        ``residue > tol`` candidates as spurious.
    """
    np_p, nq_p = f.shape
    np_g, nq_g = g.shape
    f_du = f[1:, :] * np.arange(1, np_p, dtype=np.float64)[:, None]
    f_dw = f[:, 1:] * np.arange(1, nq_p, dtype=np.float64)[None, :]
    g_du = g[1:, :] * np.arange(1, np_g, dtype=np.float64)[:, None]
    g_dw = g[:, 1:] * np.arange(1, nq_g, dtype=np.float64)[None, :]
    abs_f = np.abs(f)
    abs_g = np.abs(g)
    max_f = float(np.max(abs_f))
    max_g = float(np.max(abs_g))
    max_deg_u = max(np_p, np_g) - 1
    max_deg_w = max(nq_p, nq_g) - 1

    def _eval(u: float, w: float) -> tuple[float, float, float, float, float, float, float, float]:
        # Powers of u and w up to the max degree across f, g, and their
        # derivatives. One numpy.power call per axis amortises across
        # the 6 polynomial evaluations (f, g, fdu, fdw, gdu, gdw) plus
        # the 2 abs_* evaluations the scale needs.
        u_pow = u ** np.arange(max_deg_u + 1)
        w_pow = w ** np.arange(max_deg_w + 1)
        u_abs_pow = abs(u) ** np.arange(max_deg_u + 1)
        w_abs_pow = abs(w) ** np.arange(max_deg_w + 1)
        # The slicing here is the only place shape-dependent indexing
        # touches the inner loop. Each ``A @ B @ C`` reduces to
        # one matvec + one dot, fixed cost.
        f_val = float(u_pow[:np_p] @ f @ w_pow[:nq_p])
        g_val = float(u_pow[:np_g] @ g @ w_pow[:nq_g])
        fdu_val = float(u_pow[: np_p - 1] @ f_du @ w_pow[:nq_p])
        fdw_val = float(u_pow[:np_p] @ f_dw @ w_pow[: nq_p - 1])
        gdu_val = float(u_pow[: np_g - 1] @ g_du @ w_pow[:nq_g])
        gdw_val = float(u_pow[:np_g] @ g_dw @ w_pow[: nq_g - 1])
        # natural scales: max of global max-coeff and the pointwise
        # absolute-coeff polyval. See _build_fg_closures.scale for the
        # rationale.
        scale_f = max(float(u_abs_pow[:np_p] @ abs_f @ w_abs_pow[:nq_p]), max_f)
        scale_g = max(float(u_abs_pow[:np_g] @ abs_g @ w_abs_pow[:nq_g]), max_g)
        return f_val, g_val, fdu_val, fdw_val, gdu_val, gdw_val, scale_f, scale_g

    u, w = float(u0), float(w0)
    f_val, g_val, _, _, _, _, sf, sg = _eval(u, w)
    best_residue = max(abs(f_val) / max(sf, 1e-300), abs(g_val) / max(sg, 1e-300))
    best_u, best_w = u, w
    for _ in range(max_iter):
        if best_residue < tol:
            break
        f_val, g_val, fdu, fdw, gdu, gdw, _sf, _sg = _eval(u, w)
        det = fdu * gdw - fdw * gdu
        if det == 0.0 or not np.isfinite(det):
            break
        inv_det = 1.0 / det
        # 2x2 inverse times -[f_val; g_val]. Exact (no LU dispatch).
        du = -inv_det * (gdw * f_val - fdw * g_val)
        dw = -inv_det * (-gdu * f_val + fdu * g_val)
        if not (np.isfinite(du) and np.isfinite(dw)):
            break
        u_new = u + du
        w_new = w + dw
        f_new, g_new, _, _, _, _, sf_new, sg_new = _eval(u_new, w_new)
        residue_new = max(
            abs(f_new) / max(sf_new, 1e-300), abs(g_new) / max(sg_new, 1e-300)
        )
        u, w = u_new, w_new
        if residue_new < best_residue:
            best_u, best_w, best_residue = u_new, w_new, residue_new
    return best_u, best_w, best_residue


def _initial_w_for(f: NDArray[np.float64], g: NDArray[np.float64], u0: float) -> float | None:
    """Recover an initial ``w`` value at ``u = u0`` for Newton refinement.

    Strategy: at the true ``(u_0, w_0)`` pair, **both** ``f(u_0, w) = 0``
    and ``g(u_0, w) = 0`` hold. Find the (w_f, w_g) pair from the two
    univariate root sets that minimises ``|w_f - w_g|``; their midpoint
    is the most accurate seed for Newton (cancels first-order error).

    Picking only the f-root with smallest ``|g(u_0, .)|`` (an earlier
    formulation) fails when several f-roots have comparable
    ``|g(u_0, .)|`` -- we'd choose by accidental sign and Newton drifts
    to a different basin. Cross-validating against g-roots fixes this.

    Returns ``None`` when neither polynomial has a real root at
    ``u = u_0``.
    """
    f_at_u0 = (u0 ** np.arange(f.shape[0])) @ f
    g_at_u0 = (u0 ** np.arange(g.shape[0])) @ g
    if float(np.max(np.abs(f_at_u0))) == 0.0 or float(np.max(np.abs(g_at_u0))) == 0.0:
        return None
    f_roots = np.roots(f_at_u0[::-1])
    g_roots = np.roots(g_at_u0[::-1])
    real_f = [float(r.real) for r in f_roots if abs(r.imag) <= 1e-6 * (1.0 + abs(r.real))]
    real_g = [float(r.real) for r in g_roots if abs(r.imag) <= 1e-6 * (1.0 + abs(r.real))]
    if not real_f or not real_g:
        return None
    best_w: float | None = None
    best_gap = float("inf")
    for wf in real_f:
        for wg in real_g:
            gap = abs(wf - wg)
            if gap < best_gap:
                best_gap = gap
                best_w = 0.5 * (wf + wg)
    return best_w


def _build_fg_closures(
    f: NDArray[np.float64], g: NDArray[np.float64]
) -> tuple[
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
]:
    """Build the (residual, jacobian, scale) closures over ``(f, g)``
    expected by :func:`ssik._pencil.newton_refine_system`.

    The HP-specific math: ``r(x) = [f(u, w); g(u, w)]``, Jacobian is
    the 2x2 matrix of partial derivatives, scale is the natural
    relative-residue normalisation ``sum |f[p, q]| |u|^p |w|^q``.
    """
    polyval2d = np.polynomial.polynomial.polyval2d
    f_du = np.polynomial.polynomial.polyder(f, axis=0).astype(np.float64)
    f_dw = np.polynomial.polynomial.polyder(f, axis=1).astype(np.float64)
    g_du = np.polynomial.polynomial.polyder(g, axis=0).astype(np.float64)
    g_dw = np.polynomial.polynomial.polyder(g, axis=1).astype(np.float64)
    abs_f = np.abs(f)
    abs_g = np.abs(g)
    max_f = float(np.max(abs_f))
    max_g = float(np.max(abs_g))

    def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        u, w = float(x[0]), float(x[1])
        return np.array([polyval2d(u, w, f), polyval2d(u, w, g)], dtype=np.float64)

    def jacobian(x: NDArray[np.float64]) -> NDArray[np.float64]:
        u, w = float(x[0]), float(x[1])
        return np.array(
            [
                [polyval2d(u, w, f_du), polyval2d(u, w, f_dw)],
                [polyval2d(u, w, g_du), polyval2d(u, w, g_dw)],
            ],
            dtype=np.float64,
        )

    def scale(x: NDArray[np.float64]) -> NDArray[np.float64]:
        # Two regimes contribute to the residue floor:
        # - global ``max|f|``: bounds polynomial value on the compact
        #   region containing all roots. Correct for "small (u, w)"
        #   candidates where pointwise scale collapses to ``|f[0,0]|``
        #   and the self-ratio loses meaning.
        # - pointwise ``Sigma|f[p,q]||u|^p|w|^q``: bounds float64 eval
        #   roundoff at this specific (u, w). Correct for "large
        #   (u, w)" candidates where the pointwise sum vastly exceeds
        #   ``max|f|``.
        # Taking max preserves both bounds: at a true root,
        # ``|f(u, w)| / max(global, pointwise)`` is always machine eps.
        u_abs, w_abs = abs(float(x[0])), abs(float(x[1]))
        return np.array(
            [
                max(polyval2d(u_abs, w_abs, abs_f), max_f),
                max(polyval2d(u_abs, w_abs, abs_g), max_g),
            ],
            dtype=np.float64,
        )

    return residual, jacobian, scale


# Cluster-merge tolerance: ~sqrt(float64 eps) covers multiplicity-2 root
# splits per Wilkinson 1965 / Stewart-Sun 1990 ch. 4. Larger multiplicities
# split wider but the centroid still converges as ``eps^((k-1)/k)``.
_HP_CLUSTER_TOL = 1e-7


def eliminate_uw_pairs(
    pre: EliminatePrecompute,
    sigma_E: NDArray[np.float64],
    *,
    drop_indices: tuple[int, ...] = (7, 4, 0),
    residue_tol: float = _NEWTON_RESIDUE_TOL,
    accept_residue_tol: float | None = None,
) -> NDArray[np.float64]:
    """Run the HP elimination pipeline; return refined ``(u, w)`` pairs.

    Identical pipeline to :func:`eliminate_uw_numeric` but returns the
    full bivariate candidates, not just the ``u`` projection. Phase 5f
    back-substitution consumes ``(u, w)`` pairs; the public
    :func:`eliminate_uw_numeric` wraps this and projects to ``u`` only
    for callers that need just the ``v_1`` candidates.

    :param residue_tol: Newton's early-exit tolerance. Iterations stop
        as soon as the best-so-far ``[f, g]`` residue is below this.
        Default ``1e-12`` (full algebraic precision).
    :param accept_residue_tol: post-Newton acceptance threshold. A
        candidate's ``(u, w)`` is kept iff its best-so-far residue is
        below this. Default ``residue_tol`` (i.e. only fully-converged
        candidates -- the strict semantics ``eliminate_uw_numeric``
        callers expect). HP's ``general_6r`` solver passes a much
        looser threshold here because the downstream ``lm_refine`` in
        ``verify_candidates`` operates on 6-D joint-space and recovers
        IK candidates that pass-1 (this Newton, in 2-D ``(u, w)``)
        couldn't push to ``residue_tol`` due to multi-root degeneracy.
        Filtering at ``residue_tol`` here was rejecting real-but-multi-
        root candidates that were physically valid IK solutions.

    :returns: 2-D array of shape ``(n, 2)`` with rows
        ``[u_i, w_i]`` sorted lexicographically by ``u`` then ``w``.
        Cluster-merging operates in 2-D Euclidean distance, so two
        candidates at the same ``u`` but different ``w`` are kept
        separate -- they are physically distinct IK solutions.
    """
    if not drop_indices:
        raise ValueError("drop_indices must be non-empty")
    accept_tol = accept_residue_tol if accept_residue_tol is not None else residue_tol

    # Hoist the drop-independent ``T_w`` precompute out of the per-drop
    # loop. ``_apply_sigma_e_to_tw_pre`` only depends on ``sigma_E`` and
    # the per-arm ``T_w_pre``, not on ``drop_idx``; calling
    # ``compute_fg_numeric`` per drop was redoing it 3x.
    sigma_E_arr = np.asarray(sigma_E, dtype=np.float64)
    T_w = _apply_sigma_e_to_tw_pre(pre.T_w_pre, sigma_E_arr)

    refined: list[tuple[float, float]] = []
    last_error: Exception | None = None
    for di in drop_indices:
        try:
            f, g = _compute_fg_with_tw(pre.T_u, T_w, di)
        except np.linalg.LinAlgError as e:
            last_error = e
            continue
        try:
            cands = solve_pencil_eigenvalues(f, g)
        except np.linalg.LinAlgError as e:
            last_error = e
            continue
        if cands.size == 0:
            continue
        for u0 in cands:
            w0 = _initial_w_for(f, g, float(u0))
            if w0 is None:
                continue
            u_ref, w_ref, residue = _refine_uw_inline(
                f, g, float(u0), float(w0), _NEWTON_MAX_ITER, residue_tol
            )
            if residue < accept_tol:
                refined.append((u_ref, w_ref))
    if not refined and last_error is not None:
        raise last_error
    if not refined:
        return np.empty((0, 2), dtype=np.float64)
    # Cluster-merge in (u, w) Euclidean space. Multiplicity-k splits and
    # duplicates from different drops cluster within sqrt(eps); two
    # genuinely distinct IK solutions stay separate.
    refined.sort()
    deduped: list[list[float]] = [list(refined[0])]
    for uw in refined[1:]:
        ref = deduped[-1]
        d2 = (uw[0] - ref[0]) ** 2 + (uw[1] - ref[1]) ** 2
        scale = 1.0 + abs(ref[0]) + abs(ref[1])
        if d2 <= (_HP_CLUSTER_TOL * scale) ** 2:
            # Merge: average into the cluster centroid.
            ref[0] = 0.5 * (ref[0] + uw[0])
            ref[1] = 0.5 * (ref[1] + uw[1])
        else:
            deduped.append(list(uw))
    return np.asarray(deduped, dtype=np.float64)


def eliminate_uw_numeric(
    pre: EliminatePrecompute,
    sigma_E: NDArray[np.float64],
    *,
    drop_indices: tuple[int, ...] = (7, 4, 0),
    residue_tol: float = _NEWTON_RESIDUE_TOL,
) -> NDArray[np.float64]:
    """Run the full HP elimination pipeline; return refined ``v_1`` candidates.

    Three-stage architecture, no tunable heuristics in the hot path:

    1. **Algebraic coverage** (matrix pencil, multi-drop). Run the
       Sylvester pencil eigsolve for each ``drop_idx`` in
       ``drop_indices``. Each drop produces ``O(eps * cond)``
       approximate ``(u_i, w_i)`` pairs; different drops cover
       different parts of the V_L cap V_R variety (left-chain
       rows ``{0..3}`` vs right-chain rows ``{4..7}``). The default
       ``(7, 4, 0)`` -- two right-chain + one left-chain -- guarantees
       coverage of every IK candidate.
    2. **Newton refinement** of every candidate against the
       bivariate residual ``[f(u, w); g(u, w)] = 0`` via the
       shared :func:`ssik._pencil.newton_refine_system`. Quadratic
       convergence at simple roots.
    3. **Residue filter + cluster merge**. Drop candidates whose
       post-Newton residue stays above ``residue_tol`` (spurious
       eigenvalues). Cluster-merge near-duplicates within
       ``sqrt(float64 eps)``.

    :param pre: per-arm precomputed tensors (DH baked).
    :param sigma_E: 8-vec Study DQ of the target end-effector pose.
    :param drop_indices: tuple of drop-row indices. Default ``(7, 4, 0)``.
    :param residue_tol: maximum post-Newton relative residue.

    :returns: sorted 1-D array of real candidate ``u = v_1`` values.
    """
    pairs = eliminate_uw_pairs(
        pre, sigma_E, drop_indices=drop_indices, residue_tol=residue_tol
    )
    if pairs.size == 0:
        return np.asarray([], dtype=np.float64)
    # Project to u only; re-cluster in 1-D since two pairs at the same u
    # but different w (genuinely distinct IK solutions) collapse to a
    # single u value.
    from ssik._pencil import cluster_merge_1d

    return np.asarray(
        cluster_merge_1d(pairs[:, 0].tolist(), tol=_HP_CLUSTER_TOL),
        dtype=np.float64,
    )


# =============================================================================
# Polynomial evaluation helpers (used by tests).
# =============================================================================


def evaluate_poly(poly: sp.Poly, val: float) -> float:
    """Evaluate a sympy ``Poly`` at a numeric value with float64 output."""
    return float(poly.eval(sp.Float(val)))


def polynomial_residual(poly: sp.Poly, val: float) -> float:
    """Return the **relative** residual ``|poly(val)| / max_coeff_scale``,
    where ``max_coeff_scale = max_i |c_i| * max(1, |val|)^deg(poly)``.

    The natural normalisation for ``poly(val) = 0`` claims on a polynomial
    with arbitrary coefficient magnitudes. Returns 0 when poly is the zero
    polynomial.
    """
    coeffs = [float(c) for c in poly.all_coeffs()]
    if not coeffs:
        return 0.0
    deg = len(coeffs) - 1
    max_coeff = max(abs(c) for c in coeffs)
    if max_coeff == 0.0:
        return 0.0
    val_scale = max(1.0, abs(val)) ** deg
    expected_scale = max_coeff * val_scale
    abs_residual = abs(evaluate_poly(poly, val))
    return abs_residual / expected_scale


# =============================================================================
# FK helper for integration-style tests.
# =============================================================================


def _full_6r_chain_dq_numpy(
    v_1: float,
    a_1: float,
    l_1: float,
    d_2: float,
    v_2: float,
    a_2: float,
    l_2: float,
    v_3: float,
    d_3: float,
    a_3: float,
    l_3: float,
    v_4: float,
    d_4: float,
    a_4: float,
    l_4: float,
    v_5: float,
    d_5: float,
    a_5: float,
    l_5: float,
    v_6: float,
) -> np.ndarray:
    """Compute the projective Study DQ of a full 6R chain in tan-half-angle
    parametrisation, with ``a_6 = d_6 = l_6 = 0``. Helper used by tests
    and by integration-style elimination validators in this module.

    Convention matches Capco et al.: ``v = tan(theta/2)`` and
    ``l = tan(alpha/2)``. Joint DQs are projective (no unit-norm
    scaling); ``dq_mul`` from ``_study`` composes correctly.
    """
    from ssik.solvers.husty_pfurner._study import dq_mul

    def _rz(v: float) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _tx(a: float) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5 * a, 0.0, 0.0], dtype=np.float64)

    def _tz(d: float) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * d], dtype=np.float64)

    def _rx(t: float) -> np.ndarray:
        return np.array([1.0, t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    sigma_1 = dq_mul(_rz(v_1), dq_mul(_tx(a_1), _rx(l_1)))
    sigma_2 = dq_mul(_rz(v_2), dq_mul(_tz(d_2), dq_mul(_tx(a_2), _rx(l_2))))
    sigma_3 = dq_mul(_rz(v_3), dq_mul(_tz(d_3), dq_mul(_tx(a_3), _rx(l_3))))
    sigma_4 = dq_mul(_rz(v_4), dq_mul(_tz(d_4), dq_mul(_tx(a_4), _rx(l_4))))
    sigma_5 = dq_mul(_rz(v_5), dq_mul(_tz(d_5), dq_mul(_tx(a_5), _rx(l_5))))
    sigma_6 = _rz(v_6)
    return dq_mul(
        sigma_1,
        dq_mul(sigma_2, dq_mul(sigma_3, dq_mul(sigma_4, dq_mul(sigma_5, sigma_6)))),
    )


__all__.append("_full_6r_chain_dq_numpy")
