"""Subproblem 5: three-rotation composition.

Given four position vectors ``p0, p1, p2, p3`` and three rotation axes
``k1, k2, k3``, find ``(theta1, theta2, theta3)`` satisfying::

    p0 + Rot(k1, theta1) @ p1 = Rot(k2, theta2) @ (p2 + Rot(k3, theta3) @ p3)

Up to 4 solutions.

**Implementation**: port of the BSD-3 [ik-geo Rust reference][ikgeo]
(Elias & Wen, arXiv:2211.05737). Each rotated vector traces a cone around
``k2`` parameterised by a scalar ``h``. Subtracting the two cone polynomials
yields a univariate quartic in ``h``; real roots give candidate ``h`` values,
from which ``(theta1, theta3)`` pairs are recovered via matching sign branches
of a 2x2 quadratic-circle system, and ``theta2`` via :func:`sp1.solve`.

**Robustness beyond IK-Geo** (issue #48):

1. *Upfront degeneracy detection*. Axes too close to parallel, or ``p``
   vectors too close to collinear with their axes, return ``([], True)``
   immediately -- the algorithm's reduction is ill-defined there.
2. *All-branch enumeration*. For each real quartic root, enumerate all 4
   sign combinations without intermediate filtering. Post-verification is
   the single correctness gate; no valid solution can be dropped by a
   heuristic filter.
3. *Post-verification against the original equation*. Every candidate
   residual is checked; candidates above ``subproblem_postverify`` are
   dropped.
4. *Empty return on infeasibility* (issue #324). If no candidate satisfies
   the defining equation, return ``([], True)``. The equation is genuinely
   infeasible for some inputs (cone ranges that don't overlap), so unlike
   SP1-SP4 -- where the closest rotation always exists -- there is no useful
   "nearest" triple to hand back; post-verification is the only gate.
5. *Deduplication*. Near-duplicate solutions (angle-wise within
   ``subproblem_numerical``) are collapsed.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/subproblems/mod.rs
"""

from __future__ import annotations

import math

import cython
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
from ssik.subproblems._rotation import _cross3, _dot3, _norm3, rotate
from ssik.subproblems._validate import validate_vec3

__all__ = ["solve"]

# 2*pi as a typed constant -- referenced inside the hot _refine_sp5 loop.
# `float` is enough for mypy; Cython treats module-level float as `double`.
_TWO_PI: float = 2.0 * math.pi


@cython.ccall
def _wrap(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``. Inlined inside _refine_sp5's hot loop;
    every iteration does three of these."""
    return ((a + math.pi) % _TWO_PI) - math.pi


def _close_triple(a: tuple[float, float, float], b: tuple[float, float, float], tol: float) -> bool:
    # ``float(...)`` reasserts the boundary type: Cython.Shadow's ``@cython.ccall``
    # decorator widens ``_wrap``'s return to ``Any`` for mypy, so without this
    # cast the bool-and chain is inferred as ``Any``.
    return (
        abs(float(_wrap(a[0] - b[0]))) < tol
        and abs(float(_wrap(a[1] - b[1]))) < tol
        and abs(float(_wrap(a[2] - b[2]))) < tol
    )


def _dedup(
    triples: list[tuple[float, float, float]], tol: float
) -> list[tuple[float, float, float]]:
    unique: list[tuple[float, float, float]] = []
    for t in triples:
        if not any(_close_triple(t, u, tol) for u in unique):
            unique.append(t)
    return unique


def _degenerate(
    p1: NDArray[np.float64],
    p3: NDArray[np.float64],
    k1: NDArray[np.float64],
    k2: NDArray[np.float64],
    k3: NDArray[np.float64],
    deg_tol: float,
) -> bool:
    """Return True if input falls in a configuration where the cone-polynomial
    reduction is ill-defined (``k_i`` parallel to ``k_2``, or ``p_i`` collinear
    with its rotation axis)."""
    # ``float(...)`` reasserts the boundary type: ``_dot3`` is ``@cython.ccall``,
    # which widens to ``Any`` for mypy.
    k1xk2 = _cross3(k1, k2)
    k3xk2 = _cross3(k3, k2)
    k1xk2_sq = float(_dot3(k1xk2, k1xk2))
    k3xk2_sq = float(_dot3(k3xk2, k3xk2))
    if k1xk2_sq < deg_tol or k3xk2_sq < deg_tol:
        return True
    p1_perp_sq = float(_dot3(p1, p1)) - float(_dot3(k1, p1)) ** 2
    p3_perp_sq = float(_dot3(p3, p3)) - float(_dot3(k3, p3)) ** 2
    return p1_perp_sq < deg_tol or p3_perp_sq < deg_tol


def _residual(
    theta1: float,
    theta2: float,
    theta3: float,
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
    k1: NDArray[np.float64],
    k2: NDArray[np.float64],
    k3: NDArray[np.float64],
) -> float:
    lhs = p0 + rotate(k1, theta1, p1)
    rhs = rotate(k2, theta2, p2 + rotate(k3, theta3, p3))
    # ``float(...)`` reasserts the boundary: ``_norm3`` is ``@cython.ccall``.
    return float(_norm3(lhs - rhs))


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

    :param policy: tolerances. See module docstring for which field gates
        which stage.
    :returns: ``(solutions, is_ls)``. On exact feasibility, ``solutions``
        has 1 to 4 deduplicated triples that satisfy the defining equation
        within ``subproblem_postverify`` and ``is_ls`` is ``False``. On
        infeasibility or degeneracy, ``solutions`` is empty and ``is_ls`` is
        ``True`` -- every returned triple is guaranteed to satisfy the
        equation (issue #324).
    """
    for name, v in (
        ("p0", p0),
        ("p1", p1),
        ("p2", p2),
        ("p3", p3),
        ("k1", k1),
        ("k2", k2),
        ("k3", k3),
    ):
        validate_vec3(v, name)

    deg_tol = policy.subproblem_degeneracy

    if _degenerate(p1, p3, k1, k2, k3, deg_tol):
        return [], True

    p1_s = p0 + k1 * _dot3(k1, p1)
    p3_s = p2 + k3 * _dot3(k3, p3)

    delta1 = _dot3(k2, p1_s)
    delta3 = _dot3(k2, p3_s)

    p_1, r_1 = cone_polynomials(p0, k1, p1, p1_s, k2)
    p_3, r_3 = cone_polynomials(p2, k3, p3, p3_s, k2)

    p_13 = p_1 - p_3
    p_13_sq = vec_self_convolve_2(p_13)

    rhs = r_3 - r_1 - p_13_sq

    eqn = vec_self_convolve_3(rhs) - 4.0 * vec_convolve_3(p_13_sq, r_1)

    all_roots = solve_quartic_roots(eqn)

    # Scale-aware real-root filter: admit roots whose imaginary part is a
    # small fraction of the root magnitude (sqrt(cond)*eps noise from
    # cluster-root instability) while rejecting genuinely-complex roots
    # (|imag| ~ |real|). A strict absolute ``|imag| < num_tol`` filter
    # drops real roots whose cluster-noise imaginary part inflates above
    # ``num_tol``; that was the original completeness bug (issue #55).
    def _is_real_root(c: complex) -> bool:
        return abs(c.imag) < max(abs(c.real), 1.0) * 1e-3

    h_vec = np.array([c.real for c in all_roots if _is_real_root(c)])

    kxp1 = _cross3(k1, p1)
    kxp3 = _cross3(k3, p3)
    a_1 = np.column_stack([kxp1, -_cross3(k1, kxp1)])  # 3x2
    a_3 = np.column_stack([kxp3, -_cross3(k3, kxp3)])

    signs_1 = [1.0, 1.0, -1.0, -1.0]
    signs_3 = [1.0, -1.0, 1.0, -1.0]

    j_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)

    candidates: list[tuple[float, float, float]] = []

    for h in h_vec:
        a1t_k2 = a_1.T @ k2
        a3t_k2 = a_3.T @ k2

        const_1 = a1t_k2 * (h - delta1)
        const_3 = a3t_k2 * (h - delta3)

        hd1 = h - delta1
        hd3 = h - delta3

        a1t_k2_ns = float(a1t_k2[0] * a1t_k2[0] + a1t_k2[1] * a1t_k2[1])
        a3t_k2_ns = float(a3t_k2[0] * a3t_k2[0] + a3t_k2[1] * a3t_k2[1])
        sq1 = a1t_k2_ns - hd1 * hd1
        if sq1 < 0.0:
            continue
        sq3 = a3t_k2_ns - hd3 * hd3
        if sq3 < 0.0:
            continue

        pm_1 = j_mat @ a1t_k2 * float(np.sqrt(sq1))
        pm_3 = j_mat @ a3t_k2 * float(np.sqrt(sq3))

        for s1, s3 in zip(signs_1, signs_3, strict=True):
            sc1 = (const_1 + s1 * pm_1) / a1t_k2_ns
            sc3 = (const_3 + s3 * pm_3) / a3t_k2_ns

            v1 = a_1 @ sc1 + p1_s
            v3 = a_3 @ sc3 + p3_s

            theta2, _ = sp1.solve(k2, v3, v1, policy)
            candidates.append(
                (
                    float(np.arctan2(sc1[0], sc1[1])),
                    theta2,
                    float(np.arctan2(sc3[0], sc3[1])),
                )
            )

    def residual(cand: tuple[float, float, float]) -> float:
        return _residual(cand[0], cand[1], cand[2], p0, p1, p2, p3, k1, k2, k3)

    # Refine each candidate via Gauss-Newton on the full SP5 equation.
    # The quartic-derived angles have residuals O(num_tol) or worse when
    # the quartic has near-double roots (cond ~ 1/gap^2 blows up in the
    # companion-matrix root-finder). A few GN steps drop residuals to
    # O(eps) from a good initial guess. See issue #55 for the original
    # failure mode.
    #
    # Return order: original generation order (quartic roots x sign
    # branches). Callers that want lowest-residual-first can sort the
    # returned list. We keep the generation order because tier-1
    # univariate-search solvers (e.g. two_intersecting) rely on branch
    # indices being stable as the caller varies one input dimension --
    # sorting by residual can flip the index of a given geometric
    # solution when another branch's residual crosses it.
    refined: list[tuple[float, float, float]] = []
    for cand in candidates:
        t1, t2, t3 = _refine_sp5(cand[0], cand[1], cand[2], p0, p1, p2, p3, k1, k2, k3)
        refined.append((t1, t2, t3))

    # Post-verify against the SP5 equation. The gate is the *tight*
    # subproblem_postverify (not subproblem_numerical): Gauss-Newton refinement
    # drives every genuine solution to machine precision, so a candidate still
    # above this is a spurious near-double-quartic-root least-squares point
    # (FK ~1e-6, e.g. #337 / #159), not an IK solution -- drop it.
    exact = [cand for cand in refined if residual(cand) < policy.subproblem_postverify]

    if exact:
        solutions = _dedup(exact, policy.subproblem_dedup)
        # SP5 is a <=4-solution subproblem (quartic in h). A near-double
        # quartic root can leave two refined candidates just outside
        # ``subproblem_dedup`` -- a spurious 5th that violates the documented
        # contract and overflows fixed-width consumers (e.g. two_intersecting's
        # 4-branch univariate search, which crashed on a length-5 return).
        # Keep the <=4 lowest-residual: the genuine roots FK-close tightest,
        # the spurious near-duplicate carries the largest residual.
        if len(solutions) > 4:
            solutions = sorted(solutions, key=residual)[:4]
        return solutions, False

    # No candidate satisfies the defining equation. Post-verification is the
    # single correctness gate (issue #324): return no solution with
    # ``is_ls=True`` rather than a best-LS triple that fails the equation.
    # SP5's defining equation is genuinely infeasible for some inputs (e.g.
    # ``|p0 + Rot(k1,t1)p1| = |p1|`` can never equal ``|p2 + Rot(k3,t3)p3|``
    # when their ranges don't overlap), and a far-off "nearest" triple is not
    # a usable partial IK -- both callers fill non-existent branches with inf /
    # drop them, and the outer FK-closure gate would discard it anyway. Valid
    # solutions reach machine precision via the Gauss-Newton refinement above,
    # so this branch no longer drops near-valid solutions (the original #55
    # completeness concern that motivated the best-LS fallback).
    return [], True


@cython.ccall
@cython.locals(
    max_step=cython.double,
    j00=cython.double,
    j01=cython.double,
    j02=cython.double,
    j10=cython.double,
    j11=cython.double,
    j12=cython.double,
    j20=cython.double,
    j21=cython.double,
    j22=cython.double,
    c00=cython.double,
    c01=cython.double,
    c02=cython.double,
    c10=cython.double,
    c11=cython.double,
    c12=cython.double,
    c20=cython.double,
    c21=cython.double,
    c22=cython.double,
    det=cython.double,
    inv_det=cython.double,
    b0=cython.double,
    b1=cython.double,
    b2=cython.double,
    delta0=cython.double,
    delta1=cython.double,
    delta2=cython.double,
    step_norm=cython.double,
    scale=cython.double,
    _iter_idx=cython.int,
)
def _refine_sp5(
    t1: cython.double,
    t2: cython.double,
    t3: cython.double,
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
    k1: NDArray[np.float64],
    k2: NDArray[np.float64],
    k3: NDArray[np.float64],
    max_iter: cython.int = 20,
    step_tol: cython.double = 1e-15,
) -> tuple[float, float, float]:
    """Gauss-Newton refinement of an SP5 angle triple.

    Starts from the quartic-derived ``(t1, t2, t3)`` and iterates
    ``t <- t - J^{-1} F`` on the 3D residual ``F = p0 + Rot(k1,t1)p1 -
    Rot(k2,t2)(p2 + Rot(k3,t3)p3)``. Converges quadratically from a
    good initial guess; 2-3 iterations drop residuals to O(eps).

    Returns the (possibly refined) triple. On ill-conditioned Jacobian
    (e.g. degenerate geometry that escaped the upfront filter) falls
    back to the input without raising, leaving the caller's post-verify
    as the authoritative filter.
    """
    max_step = math.pi / 4.0
    for _iter_idx in range(max_iter):
        rotated_p1 = rotate(k1, t1, p1)
        rotated_p3_inner = rotate(k3, t3, p3)
        p2_plus_rotp3 = p2 + rotated_p3_inner
        rotated_p2 = rotate(k2, t2, p2_plus_rotp3)

        f = p0 + rotated_p1 - rotated_p2

        # Jacobian columns: dF/dt_i = axis_i x (rotated vector).
        col1 = _cross3(k1, rotated_p1)
        col2 = -_cross3(k2, rotated_p2)
        col3 = -rotate(k2, t2, _cross3(k3, rotated_p3_inner))

        # 3x3 closed-form solve via cofactor expansion -- avoids
        # np.linalg.solve dispatch (~5us) and the column_stack + np.array
        # construction. delta = inv(J) @ -f.
        j00 = float(col1[0])
        j10 = float(col1[1])
        j20 = float(col1[2])
        j01 = float(col2[0])
        j11 = float(col2[1])
        j21 = float(col2[2])
        j02 = float(col3[0])
        j12 = float(col3[1])
        j22 = float(col3[2])
        c00 = j11 * j22 - j12 * j21
        c01 = j12 * j20 - j10 * j22
        c02 = j10 * j21 - j11 * j20
        det = j00 * c00 + j01 * c01 + j02 * c02
        if abs(det) < 1e-15:
            break
        c10 = j02 * j21 - j01 * j22
        c11 = j00 * j22 - j02 * j20
        c12 = j01 * j20 - j00 * j21
        c20 = j01 * j12 - j02 * j11
        c21 = j02 * j10 - j00 * j12
        c22 = j00 * j11 - j01 * j10
        b0 = -float(f[0])
        b1 = -float(f[1])
        b2 = -float(f[2])
        inv_det = 1.0 / det
        delta0 = (c00 * b0 + c10 * b1 + c20 * b2) * inv_det
        delta1 = (c01 * b0 + c11 * b1 + c21 * b2) * inv_det
        delta2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det

        # Clip per-iteration step to pi/4 so an ill-conditioned Jacobian
        # doesn't launch us to a far-away minimum; quadratic convergence is
        # preserved near the true solution where |delta| is already small.
        step_norm = math.sqrt(delta0 * delta0 + delta1 * delta1 + delta2 * delta2)
        if step_norm > max_step:
            scale = max_step / step_norm
            delta0 *= scale
            delta1 *= scale
            delta2 *= scale

        t1 = _wrap(t1 + delta0)
        t2 = _wrap(t2 + delta1)
        t3 = _wrap(t3 + delta2)

        if step_norm < step_tol:
            break

    return t1, t2, t3
