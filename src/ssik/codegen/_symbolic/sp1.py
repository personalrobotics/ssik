"""Symbolic SP1: rotate p to match q.

Mirrors :func:`ssik.subproblems.sp1.solve` but takes sympy 3x1 Matrix
inputs and returns a sympy expression for ``theta``.

SP1 closed form (see :mod:`ssik.subproblems.sp1` docstring for the
derivation):

    theta = atan2((k x p) . q, p . q - (k.p)(k.q))

The expression is exact for the unique solution. The runtime solver also
reports an ``is_ls`` flag based on feasibility tolerances; the symbolic
version drops it because the codegen emits a single ``atan2`` whose
output is correct in both exact and LS regimes (LS is the continuous
extension; the geometric meaning of LS is captured by the FK-residual
check the artifact's ``solve()`` does after composing the candidates).
"""

from __future__ import annotations

import sympy as sp

__all__ = ["sp1_theta_sym"]


def sp1_theta_sym(k: sp.Matrix, p: sp.Matrix, q: sp.Matrix) -> sp.Expr:
    """Return the sympy expression for ``theta`` solving SP1.

    :param k: 3x1 sympy Matrix, unit rotation axis (typically a
        ``sp.Matrix`` of constants for the user's arm; can also be
        symbolic if the axis itself depends on an upstream IK angle).
    :param p: 3x1 sympy Matrix, vector to rotate.
    :param q: 3x1 sympy Matrix, target vector. May contain symbolic
        components from ``T_target`` after upstream substitution.
    :returns: sympy expression ``theta`` in radians.

    The caller is responsible for substituting the arm's constants and
    running ``sympy.cse`` over the final composed expression.
    """
    kxp = k.cross(p)
    kp = k.dot(p)
    kq = k.dot(q)
    return sp.atan2(kxp.dot(q), p.dot(q) - kp * kq)
