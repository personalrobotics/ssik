"""Symbolic SP4: rotate p around k so h.(R p) = d.

Mirrors :func:`ssik.subproblems.sp4.solve` but takes sympy 3x1 Matrix
inputs and returns sympy expressions for ``theta`` candidates.

SP4 closed form (see :mod:`ssik.subproblems.sp4` docstring for the
derivation):

    A = h.p - (k.p)(h.k)
    B = h.(k x p)
    C = (k.p)(h.k)
    R = sqrt(A^2 + B^2)
    phi = atan2(B, A)
    delta = acos((d - C) / R)
    theta = phi +/- delta              # 2 branches in the generic case

Returned: ``(theta_plus, theta_minus)`` -- both branches as sympy
expressions. The codegen emits both into the artifact; runtime selects
based on which one closes FK (the standard SP4 branching).

LS / degeneracy handling at runtime:

- The artifact's ``solve()`` always evaluates both branches; if one
  produces NaN (because ``(d - C) / R`` is outside [-1, 1]), the
  artifact catches and falls back to the LS extension ``theta = phi``
  or ``theta = phi + pi`` based on ``sign(d - C)``.
- Degenerate ``R ~ 0`` (p collinear with k): the artifact detects
  via ``A**2 + B**2 < deg_tol`` and returns ``theta = 0`` with
  ``is_ls = (C != d)`` -- mirrors the runtime SP4 fallback.

Both fallbacks are emitted as small Python ``if`` blocks alongside
the main branches; the FAST path stays straight-line trig.
"""

from __future__ import annotations

import sympy as sp

__all__ = ["sp4_branches_sym"]


def sp4_branches_sym(
    h: sp.Matrix, k: sp.Matrix, p: sp.Matrix, d: sp.Expr
) -> tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]:
    """Return the symbolic SP4 outputs.

    :param h: 3x1 sympy Matrix, target direction.
    :param k: 3x1 sympy Matrix, rotation axis (unit).
    :param p: 3x1 sympy Matrix, source vector.
    :param d: sympy expression, target scalar.
    :returns: 5-tuple ``(theta_plus, theta_minus, R_sq, rhs, phi)``:

        - ``theta_plus = phi + delta``
        - ``theta_minus = phi - delta``
        - ``R_sq = A**2 + B**2`` (used for degeneracy and LS-feasibility checks at runtime)
        - ``rhs = d - C`` (used for LS sign branch)
        - ``phi = atan2(B, A)`` (used for LS fallback theta)

        The codegen emits all five so the runtime can branch correctly
        on degenerate / LS regimes.
    """
    hp = h.dot(p)
    kp = k.dot(p)
    hk = h.dot(k)
    A = hp - kp * hk
    B = h.dot(k.cross(p))
    C = kp * hk
    R_sq = A**2 + B**2
    rhs = d - C
    phi = sp.atan2(B, A)
    R = sp.sqrt(R_sq)
    delta = sp.acos(rhs / R)
    theta_plus = phi + delta
    theta_minus = phi - delta
    return theta_plus, theta_minus, R_sq, rhs, phi
