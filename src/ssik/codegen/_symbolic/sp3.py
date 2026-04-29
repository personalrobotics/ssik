"""Symbolic SP3: rotate p around k so distance from q equals d.

SP3 reduces algebraically to SP4 with a target shift:

    | Rot(k, theta) p - q |^2 = d^2
    |p|^2 - 2 q . Rot(k, theta) p + |q|^2 = d^2
    q . Rot(k, theta) p = (|p|^2 + |q|^2 - d^2) / 2

i.e. SP4 with ``h = q``, ``p = p``, ``k = k``, target = (|p|^2 + |q|^2 - d^2) / 2.

This module just provides the target-shift; downstream composers feed
it into :func:`ssik.codegen._symbolic.sp4.sp4_branches_sym`.
"""

from __future__ import annotations

import sympy as sp

from ssik.codegen._symbolic.sp4 import sp4_branches_sym

__all__ = ["sp3_branches_sym"]


def sp3_branches_sym(
    k: sp.Matrix, p: sp.Matrix, q: sp.Matrix, d: sp.Expr
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return the symbolic SP3 outputs by reducing to SP4.

    :param k: 3x1 sympy Matrix, rotation axis.
    :param p: 3x1 sympy Matrix, vector to rotate.
    :param q: 3x1 sympy Matrix, target point.
    :param d: sympy expression, target distance.
    :returns: identical tuple to :func:`sp4_branches_sym`:
        ``(theta_plus, theta_minus, R_sq, rhs, phi)``.
    """
    target = (p.dot(p) + q.dot(q) - d**2) / 2
    return sp4_branches_sym(q, k, p, target)
