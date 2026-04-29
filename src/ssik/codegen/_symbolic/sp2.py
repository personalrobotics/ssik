"""Symbolic SP2: rotate two vectors to share a common point.

SP2 has a 4-quantity output: two ``(theta1, theta2)`` branches. Each
branch reuses SP1 with a constructed "meeting point" ``z_a`` or ``z_b``
of the two cones around k1 and k2.

Closed form (see :mod:`ssik.subproblems.sp2`):

    c = k1.k2;  s_sq = 1 - c^2
    d1 = k1.p;  d2 = k2.q
    alpha = (d1 - c*d2) / s_sq
    beta  = (d2 - c*d1) / s_sq
    kxk = k1 x k2          ;  |kxk|^2 = s_sq
    z_sq_target = (|p|^2 + |q|^2) / 2
    gamma_sq_scaled = z_sq_target - alpha^2 - beta^2 - 2*alpha*beta*c
    gamma = sqrt(gamma_sq_scaled / s_sq)
    z_a = alpha*k1 + beta*k2 + gamma*kxk
    z_b = alpha*k1 + beta*k2 - gamma*kxk
    (theta1_a, theta2_a) = (SP1(k1, p, z_a), SP1(k2, q, z_a))
    (theta1_b, theta2_b) = (SP1(k1, p, z_b), SP1(k2, q, z_b))

This module returns the four angle expressions + the LS / degeneracy
guards (``s_sq``, ``gamma_sq_scaled``) so the codegen can emit:

  - ``s_sq < deg_tol`` -> parallel axes (LS, single branch)
  - ``gamma_sq_scaled <= 0`` -> tangent or LS (single branch with gamma = 0)
  - else -> two branches as above

Used by the spherical_two_intersecting composer (Puma's secondary path).
"""

from __future__ import annotations

import sympy as sp

from ssik.codegen._symbolic.sp1 import sp1_theta_sym

__all__ = ["sp2_branches_sym"]


def sp2_branches_sym(
    k1: sp.Matrix, k2: sp.Matrix, p: sp.Matrix, q: sp.Matrix
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return the symbolic SP2 outputs.

    :returns: 6-tuple ``(theta1_a, theta2_a, theta1_b, theta2_b,
        s_sq, gamma_sq_scaled)``:

        - ``theta1_a, theta2_a``: branch-a angles (gamma > 0).
        - ``theta1_b, theta2_b``: branch-b angles (gamma < 0).
        - ``s_sq = 1 - (k1.k2)^2``: degenerate when below ``deg_tol``
          (axes parallel).
        - ``gamma_sq_scaled``: tangent / LS condition checked at runtime.

        The codegen emits all six and branches in Python on the guards.
    """
    c = k1.dot(k2)
    s_sq = 1 - c * c
    d1 = k1.dot(p)
    d2 = k2.dot(q)
    alpha = (d1 - c * d2) / s_sq
    beta = (d2 - c * d1) / s_sq
    kxk = k1.cross(k2)
    pp = p.dot(p)
    qq = q.dot(q)
    z_sq_target = (pp + qq) / 2
    gamma_sq_scaled = z_sq_target - alpha**2 - beta**2 - 2 * alpha * beta * c
    gamma = sp.sqrt(gamma_sq_scaled / s_sq)

    # Two meeting points.
    z_a = alpha * k1 + beta * k2 + gamma * kxk
    z_b = alpha * k1 + beta * k2 - gamma * kxk

    # Each branch: SP1 to recover theta1 and theta2.
    theta1_a = sp1_theta_sym(k1, p, z_a)
    theta2_a = sp1_theta_sym(k2, q, z_a)
    theta1_b = sp1_theta_sym(k1, p, z_b)
    theta2_b = sp1_theta_sym(k2, q, z_b)

    return theta1_a, theta2_a, theta1_b, theta2_b, s_sq, gamma_sq_scaled
