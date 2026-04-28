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
   residual is checked; candidates above ``subproblem_numerical`` are
   dropped.
4. *Best-LS return on infeasibility*. If no candidate passes post-verify
   but the algorithm produced candidates, return the minimum-residual one
   with ``is_ls=True`` (consistent with SP1-SP4's LS semantics).
5. *Deduplication*. Near-duplicate solutions (angle-wise within
   ``subproblem_numerical``) are collapsed.

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
from ssik.subproblems._rotation import _cross3, _dot3, _norm3, rotate
from ssik.subproblems._validate import validate_vec3

__all__ = ["solve"]


def _wrap(a: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _close_triple(a: tuple[float, float, float], b: tuple[float, float, float], tol: float) -> bool:
    return (
        abs(_wrap(a[0] - b[0])) < tol
        and abs(_wrap(a[1] - b[1])) < tol
        and abs(_wrap(a[2] - b[2])) < tol
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
    k1xk2 = _cross3(k1, k2)
    k3xk2 = _cross3(k3, k2)
    k1xk2_sq = _dot3(k1xk2, k1xk2)
    k3xk2_sq = _dot3(k3xk2, k3xk2)
    if k1xk2_sq < deg_tol or k3xk2_sq < deg_tol:
        return True
    p1_perp_sq = _dot3(p1, p1) - _dot3(k1, p1) ** 2
    p3_perp_sq = _dot3(p3, p3) - _dot3(k3, p3) ** 2
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
    return _norm3(lhs - rhs)


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
        within ``subproblem_numerical`` and ``is_ls`` is ``False``. On
        infeasibility or degeneracy, ``solutions`` has at most 1 best-LS
        triple (or is empty if no candidate was even generated) and
        ``is_ls`` is ``True``.
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

    num_tol = policy.subproblem_numerical
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

    # Post-verify against the SP5 equation.
    exact = [cand for cand in refined if residual(cand) < num_tol]

    if exact:
        solutions = _dedup(exact, policy.subproblem_dedup)
        return solutions, False

    # No exact solution survived. Return best-LS if we have any candidate
    # at all; otherwise signal total infeasibility.
    if not refined:
        return [], True

    best = min(refined, key=residual)
    return [best], True


def _refine_sp5(
    t1: float,
    t2: float,
    t3: float,
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
    k1: NDArray[np.float64],
    k2: NDArray[np.float64],
    k3: NDArray[np.float64],
    max_iter: int = 20,
    step_tol: float = 1e-15,
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
    for _ in range(max_iter):
        rotated_p1 = rotate(k1, t1, p1)
        rotated_p3_inner = rotate(k3, t3, p3)
        p2_plus_rotp3 = p2 + rotated_p3_inner
        rotated_p2 = rotate(k2, t2, p2_plus_rotp3)

        f = p0 + rotated_p1 - rotated_p2

        # Jacobian columns: dF/dt_i = axis_i x (rotated vector).
        col1 = _cross3(k1, rotated_p1)
        col2 = -_cross3(k2, rotated_p2)
        col3 = -rotate(k2, t2, _cross3(k3, rotated_p3_inner))
        j_mat_3x3 = np.column_stack([col1, col2, col3])

        try:
            delta = np.linalg.solve(j_mat_3x3, -f)
        except np.linalg.LinAlgError:
            break

        # Clip per-iteration step to pi/4 so an ill-conditioned Jacobian
        # doesn't launch us to a far-away minimum; quadratic convergence is
        # preserved near the true solution where |delta| is already small.
        step_norm = _norm3(delta)
        max_step = np.pi / 4.0
        if step_norm > max_step:
            delta = delta * (max_step / step_norm)

        t1 = _wrap(t1 + float(delta[0]))
        t2 = _wrap(t2 + float(delta[1]))
        t3 = _wrap(t3 + float(delta[2]))

        if step_norm < step_tol:
            break

    return t1, t2, t3
