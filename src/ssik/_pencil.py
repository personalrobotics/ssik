"""Shared matrix-pencil + Newton-refinement primitives for analytical-IK
eigenvalue solvers.

Several IK pipelines reduce to "find roots of ``det A(x) = 0``" where
``A(x)`` is a polynomial matrix, then refine each approximate root
against an underlying nonlinear residual system:

- IK-Geo Raghavan-Roth (``ssik.solvers.ikgeo._raghavan_roth``): degree
  2 in ``x_2`` -- ``(M_quad x_2^2 + M_lin x_2 + M_const) v = 0``,
  linearised to a 24x24 generalised eigenvalue problem; 14-equation
  loop-closure residual for refinement.
- Husty-Pfurner elimination (``ssik.solvers.husty_pfurner._eliminate``):
  degree up to 8 in ``u`` after Sylvester construction, linearised to
  an 80x80 generalised eigenvalue problem; bivariate ``[f; g] = 0``
  residual for refinement.

The numerical primitives are universal:

1. **Equilibration** (row + column scaling + variable rescaling) --
   recovers ~12 accurate digits when raw ``cond(A_k)`` reaches 1e16+
   (e.g. 60-deg twists on JACO 2, or large alpha on Husty-Pfurner).
2. **Frobenius companion linearisation** -- maps the polynomial matrix
   pencil to a generalised standard pencil ``(A, B)`` whose finite
   eigenvalues are roots of ``det A(x) = 0``.
3. **Generalised eigsolve + filter** -- via :func:`scipy.linalg.eig`,
   keep finite, near-real, magnitude-bounded eigenvalues.
4. **Newton refinement** -- starting from each approximate root,
   polish to machine precision against the underlying nonlinear
   residual. Quadratic convergence at simple roots.
5. **Cluster merge** -- multi-roots split into clusters of size
   ~sqrt(machine_eps) per Wilkinson 1965; their centroid is the
   best float64-precision estimate of the true root.

The HP and RR pipelines call these primitives with their own
problem-specific residual / Jacobian / scale closures. No tunable
heuristics live in the shared layer.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.linalg as la  # type: ignore[import-untyped]
from numpy.typing import NDArray

__all__ = [
    "build_frobenius_pencil_pair",
    "cluster_merge_1d",
    "equilibrate_polynomial_matrix",
    "equilibrate_three_matrix_pencil",
    "newton_refine_system",
    "solve_polynomial_matrix_eigenvalues",
]


def equilibrate_polynomial_matrix(
    S: NDArray[np.float64],
    *,
    rescale_variable: bool = False,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Row + column equilibrate the polynomial matrix
    ``A(x) = sum_k S[k] x^k`` (shape ``(d+1, n, n)``).

    :param S: tensor of shape ``(d+1, n, n)`` of coefficient matrices.
    :param rescale_variable: if True, also rescale ``x = c * x_internal``
        with ``c`` chosen so the leading and constant coefficients have
        comparable max norms. The eigenvalues of ``A_eq(x_internal) = 0``
        multiplied by ``c`` give those of ``A(x) = 0``. Default False
        (preserves drop-in compatibility with code that didn't expect
        an eigenvalue rescaling).

    :returns: ``(S_eq, d_l, d_r, scale_c)`` where:

        - ``S_eq`` is the equilibrated tensor ``D_l S_k D_r * c^k``.
        - ``d_l`` is the 1-D row-scaling vector (length ``n``).
        - ``d_r`` is the 1-D column-scaling vector (length ``n``).
        - ``scale_c`` is the variable-axis rescaling factor (1.0 if
          ``rescale_variable=False``).

    Eigenvalue relation: if ``A_eq(x_internal) v_eq = 0`` then
    ``A(scale_c * x_internal) (D_r * v_eq) = 0``.
    """
    if S.ndim != 3 or S.shape[1] != S.shape[2]:
        raise ValueError(f"S must be (d+1, n, n), got shape {S.shape}")
    d_plus_1, n, _ = S.shape
    d = d_plus_1 - 1

    if rescale_variable and d >= 1:
        norm_0 = float(np.max(np.abs(S[0])))
        norm_d = float(np.max(np.abs(S[d])))
        scale_c = (norm_0 / norm_d) ** (1.0 / d) if norm_0 > 0.0 and norm_d > 0.0 else 1.0
    else:
        scale_c = 1.0

    if scale_c != 1.0:
        S_scaled = np.empty_like(S)
        for k in range(d_plus_1):
            S_scaled[k] = S[k] * (scale_c**k)
    else:
        S_scaled = S

    row_max = np.zeros(n, dtype=np.float64)
    for k in range(d_plus_1):
        row_max = np.maximum(row_max, np.abs(S_scaled[k]).max(axis=1))
    row_max = np.where(row_max > 0.0, row_max, 1.0)
    d_l = 1.0 / row_max

    S_after_row = S_scaled * d_l[None, :, None]

    col_max = np.zeros(n, dtype=np.float64)
    for k in range(d_plus_1):
        col_max = np.maximum(col_max, np.abs(S_after_row[k]).max(axis=0))
    col_max = np.where(col_max > 0.0, col_max, 1.0)
    d_r = 1.0 / col_max

    S_eq = S_after_row * d_r[None, None, :]
    return S_eq, d_l, d_r, scale_c


def build_frobenius_pencil_pair(
    S: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Frobenius companion linearisation of the polynomial matrix
    ``S(x) = sum_k S[k] x^k`` (shape ``(d+1, n, n)``) into a generalised
    eigenvalue problem ``A v = x B v``.

    For ``d = 0`` (constant matrix), returns ``(-S_0, I)`` so the
    "eigenvalues" are roots of ``det S_0 = 0`` (degenerate; no
    finite x-dependence).

    For ``d >= 1``:

    - ``B = block_diag(I, I, ..., I, S_d)`` of size ``nd x nd``.
    - ``A`` has identity superdiagonal blocks and the bottom block-row
      is ``[-S_0, -S_1, ..., -S_{d-1}]``.

    The finite generalised eigenvalues of ``(A, B)`` are exactly the
    roots of ``det S(x) = 0``. The right eigenvector ``v`` has the
    block structure ``[v_0; x v_0; x^2 v_0; ...; x^{d-1} v_0]`` where
    ``v_0`` is the null vector of ``S(x)``.
    """
    if S.ndim != 3 or S.shape[1] != S.shape[2]:
        raise ValueError(f"S must be (d+1, n, n), got shape {S.shape}")
    d_plus_1, n, _ = S.shape
    d = d_plus_1 - 1
    if d == 0:
        return -S[0].astype(np.float64, copy=True), np.eye(n, dtype=np.float64)
    A = np.zeros((n * d, n * d), dtype=np.float64)
    B = np.eye(n * d, dtype=np.float64)
    for k in range(d - 1):
        A[k * n : (k + 1) * n, (k + 1) * n : (k + 2) * n] = np.eye(n)
    for k in range(d):
        A[(d - 1) * n : d * n, k * n : (k + 1) * n] = -S[k]
    B[(d - 1) * n : d * n, (d - 1) * n : d * n] = S[d]
    return A, B


# Imaginary-leakage threshold for the "near-real" classifier. Eigenvalues
# with ``|Im|/(1+|Re|) < _LEAKAGE_BAND`` are inspected; the max leakage
# across them is the multiplicity-cluster diagnostic returned alongside
# the candidate set.
_LEAKAGE_BAND = 1e-3


def solve_polynomial_matrix_eigenvalues(
    S: NDArray[np.float64],
    *,
    real_tol: float = 1e-3,
    max_magnitude: float = 1e10,
    rescale_variable: bool = True,
) -> tuple[NDArray[np.float64], float]:
    """End-to-end solve of ``det S(x) = 0`` via equilibration +
    Frobenius linearisation + generalised eigsolve + filter.

    :param S: tensor of shape ``(d+1, n, n)`` -- polynomial matrix
        coefficients.
    :param real_tol: an eigenvalue ``e`` is treated as real iff
        ``|Im(e)| <= real_tol * (1 + |Re(e)|)``. Default 1e-3 is
        loose; downstream Newton refinement filters spurious results
        on residue, not on imaginary tolerance.
    :param max_magnitude: discard eigenvalues larger than this.
        Default 1e10 is permissive; downstream residue filter
        catches numerical-infinity spurious roots.
    :param rescale_variable: pass through to
        :func:`equilibrate_polynomial_matrix`. Default True for
        polynomial degrees > 2 where coefficient magnitudes spread.

    :returns: ``(candidates, leakage)``. ``candidates`` is a sorted
        1-D array of finite real x values; ``leakage`` is the maximum
        ``|Im|/(1+|Re|)`` over near-real eigenvalues -- a diagnostic
        for multiplicity-2+ clustering.
    """
    S_eq, _, _, scale_c = equilibrate_polynomial_matrix(S, rescale_variable=rescale_variable)
    A, B = build_frobenius_pencil_pair(S_eq)
    eigvals = la.eig(A, B, left=False, right=False)
    with np.errstate(invalid="ignore"):
        eigvals = eigvals * scale_c
    out: list[float] = []
    leakage = 0.0
    for e in eigvals:
        if not np.isfinite(e):
            continue
        re, im = float(np.real(e)), float(np.imag(e))
        if abs(re) > max_magnitude:
            continue
        rel_im = abs(im) / (1.0 + abs(re))
        if rel_im < _LEAKAGE_BAND and rel_im > leakage:
            leakage = rel_im
        if rel_im > real_tol:
            continue
        out.append(re)
    out.sort()
    return np.asarray(out, dtype=np.float64), leakage


def newton_refine_system(
    residual_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    *,
    natural_scale_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    max_iter: int = 5,
    tol: float = 1e-12,
) -> tuple[NDArray[np.float64], float]:
    """Newton refinement of a nonlinear system ``r(x) = 0`` with
    monotone-best tracking.

    For square Jacobians, solves ``J dx = -r`` exactly. For
    overdetermined Jacobians (more equations than unknowns, common
    in IK loop-closure), solves the least-squares problem -- still
    converges quadratically near simple roots when the underlying
    system has rank equal to dim(x).

    Returns the BEST ``(x, residue)`` along the Newton trajectory --
    not the final iterate. At multiplicity-k roots the Jacobian is
    near-singular and a naive step can jump to a far-away simple
    root; this guard ensures Newton never makes things worse than
    the starting point.

    :param residual_fn: ``x -> r`` of shape ``(m,)``.
    :param jacobian_fn: ``x -> J`` of shape ``(m, n)`` where
        ``len(x) == n``.
    :param x0: starting point of shape ``(n,)``.
    :param natural_scale_fn: ``x -> s`` of shape ``(m,)`` giving the
        natural absolute scale of each residual component at ``x``.
        Used to compute ``residue = max(|r| / s)``. Defaults to
        ``max(|r|, 1)`` which is fine when residuals are O(1) but
        loses precision when polynomial coefficients span many
        orders of magnitude. Pass a real ``scale_fn`` for
        polynomial residuals.
    :param max_iter: maximum Newton iterations.
    :param tol: relative-residue early-exit threshold. Iteration
        stops as soon as the best-so-far residue drops below this.
        Default 1e-12 (10 digits under float64 machine epsilon).

    :returns: ``(x_refined, residue)``. Caller is responsible for
        rejecting candidates with ``residue > tol`` -- this is how
        spurious / non-converged candidates get filtered.
    """

    def _residue_at(x_at: NDArray[np.float64]) -> float:
        r_at = residual_fn(x_at)
        s_at = (
            natural_scale_fn(x_at)
            if natural_scale_fn is not None
            else np.maximum(np.abs(r_at), 1.0)
        )
        s_at = np.maximum(s_at, 1e-300)
        return float(np.max(np.abs(r_at) / s_at))

    x = x0.astype(np.float64, copy=True)
    # Track best (x, residue) along the Newton trajectory. At
    # multiplicity-k roots the Jacobian is near-singular and a single
    # naive step can jump to a far-away simple root; "best so far"
    # tracking ensures Newton never returns worse than the start.
    best_x = x.copy()
    best_residue = _residue_at(x)
    for _ in range(max_iter):
        if best_residue < tol:
            break
        r = residual_fn(x)
        J = jacobian_fn(x)
        if J.shape[0] == J.shape[1]:
            try:
                delta = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                break
        else:
            delta_lstsq, *_ = np.linalg.lstsq(J, -r, rcond=None)
            delta = delta_lstsq
        if not np.all(np.isfinite(delta)):
            break
        x = x + delta
        residue = _residue_at(x)
        if residue < best_residue:
            best_x = x.copy()
            best_residue = residue
    return best_x, best_residue


def cluster_merge_1d(values: list[float], tol: float = 1e-7) -> list[float]:
    """Merge sorted 1-D values into clusters; return the per-cluster
    centroid.

    Two values cluster if ``|v_i - v_j| <= tol * (1 + |v_j|)``.
    Multiplicity-k roots of polynomial systems split into k
    eigenvalue candidates clustered within ``sqrt(machine_eps)``
    (Wilkinson 1965, Stewart-Sun 1990 ch. 4); their centroid is
    closer to the true root than any individual member, by an
    additional factor of ``sqrt(k)``.

    :param values: input list (need not be sorted; sorted internally).
    :param tol: relative cluster-merge tolerance. Default 1e-7
        (~sqrt(float64 eps)). Pass ``0.0`` to disable merging.
    """
    if not values:
        return []
    sorted_values = sorted(values)
    clusters: list[list[float]] = [[sorted_values[0]]]
    for v in sorted_values[1:]:
        ref = clusters[-1][-1]
        if abs(v - ref) <= tol * (1.0 + abs(ref)):
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def equilibrate_three_matrix_pencil(
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
    """Legacy 3-matrix API for the Raghavan-Roth quadratic eigenvalue
    problem ``(A x^2 + B x + C) v = 0``. Wraps
    :func:`equilibrate_polynomial_matrix` and unpacks back to the
    historical signature.

    :returns: ``(A_eq, B_eq, C_eq, d_l, d_r)``.
    """
    if a.shape != b.shape or b.shape != c.shape:
        raise ValueError(f"a, b, c must have the same shape; got {a.shape}, {b.shape}, {c.shape}")
    S = np.stack([a, b, c], axis=0)
    S_eq, d_l, d_r, _ = equilibrate_polynomial_matrix(S, rescale_variable=False)
    return S_eq[0], S_eq[1], S_eq[2], d_l, d_r
