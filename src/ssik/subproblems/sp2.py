"""Subproblem 2: two coupled rotations that produce the same target vector.

Given two unit axes ``k1``, ``k2``, a source vector ``p``, and a target vector
``q``, find ``(theta1, theta2)`` such that::

    Rot(k1, theta1) @ p == Rot(k2, theta2) @ q

Equivalent under a sign flip to Paden-Kahan 2's ``Rot(k1, t1) Rot(k2, t2) p = q``.
Up to 2 solutions in the generic non-parallel, feasible case.

**Derivation.** Let ``z = Rot(k1, theta1) p = Rot(k2, theta2) q``. Then
``z`` satisfies:

    k1 . z = k1 . p      (axial component preserved by Rot(k1, .))
    k2 . z = k2 . q      (axial component preserved by Rot(k2, .))
    |z|    = |p| = |q|   (rotations preserve magnitude)

Expressing ``z = alpha k1 + beta k2 + gamma (k1 x k2)`` and letting
``c = k1 . k2``, the first two constraints form a 2x2 linear system::

    alpha + beta * c     = k1 . p
    alpha * c + beta     = k2 . q

Solved (when ``c^2 != 1``):

    alpha = (k1.p - c * k2.q) / (1 - c^2)
    beta  = (k2.q - c * k1.p) / (1 - c^2)

The sphere constraint gives two values for ``gamma`` (hence two ``z`` and
two ``(theta1, theta2)`` pairs), one value (tangent), or none (infeasible,
use LS approximation):

    gamma^2 * (1 - c^2) = |p|^2 - alpha^2 - beta^2 - 2 * alpha * beta * c

Each resulting ``z`` is used to recover ``theta1 = SP1(k1, p, z)`` and
``theta2 = SP1(k2, q, z)``.

**Degenerate case (parallel axes, k1 || k2).** ``c^2 == 1`` makes the 2x2
system singular; infinitely many ``(theta1, theta2)`` pairs satisfy the
equation (only their sum or difference is fixed). The implementation flags
this as LS (via ``subproblem_degeneracy``) and returns a canonical choice
with ``theta2 = 0``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.subproblems import sp1
from ssik.subproblems._rotation import _cross3, _dot3

__all__ = ["solve"]


def solve(
    k1: NDArray[np.float64],
    k2: NDArray[np.float64],
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[tuple[float, float]], bool]:
    """Solve SP2.

    :param k1: first rotation axis, shape ``(3,)``.
    :param k2: second rotation axis, shape ``(3,)``.
    :param p: source vector rotated by ``(k1, theta1)``.
    :param q: target vector rotated by ``(k2, theta2)``.
    :param policy: tolerances. ``subproblem_degeneracy`` gates the
        parallel-axis fallback; ``subproblem_feasibility`` gates is_ls
        on magnitude and sphere mismatches.
    :returns: ``(solutions, is_ls)``.
    """
    c = _dot3(k1, k2)
    s_sq = 1.0 - c * c

    if s_sq < policy.subproblem_degeneracy:
        # Parallel-axis case: infinitely many solutions; canonical choice.
        theta1, _ = sp1.solve(k1, p, q, policy)
        return [(theta1, 0.0)], True

    d1 = _dot3(k1, p)
    d2 = _dot3(k2, q)
    alpha = (d1 - c * d2) / s_sq
    beta = (d2 - c * d1) / s_sq
    kxk = _cross3(k1, k2)  # |kxk|^2 = s_sq

    pp = _dot3(p, p)
    qq = _dot3(q, q)
    z_sq_target = 0.5 * (pp + qq)
    gamma_sq_scaled = z_sq_target - alpha * alpha - beta * beta - 2 * alpha * beta * c

    feas_tol = policy.subproblem_feasibility
    mag_mismatch = abs(pp - qq) > feas_tol
    infeasible_sphere = gamma_sq_scaled < -feas_tol
    is_ls = mag_mismatch or infeasible_sphere

    if gamma_sq_scaled <= 0.0:
        # Tangent or LS: single representative z with gamma = 0.
        z = alpha * k1 + beta * k2
        theta1, _ = sp1.solve(k1, p, z, policy)
        theta2, _ = sp1.solve(k2, q, z, policy)
        return [(theta1, theta2)], is_ls

    gamma = float(np.sqrt(gamma_sq_scaled / s_sq))
    z_a = alpha * k1 + beta * k2 + gamma * kxk
    z_b = alpha * k1 + beta * k2 - gamma * kxk

    theta1_a, _ = sp1.solve(k1, p, z_a, policy)
    theta2_a, _ = sp1.solve(k2, q, z_a, policy)
    theta1_b, _ = sp1.solve(k1, p, z_b, policy)
    theta2_b, _ = sp1.solve(k2, q, z_b, policy)

    return [(theta1_a, theta2_a), (theta1_b, theta2_b)], is_ls
