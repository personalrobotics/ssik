"""Subproblem 4: rotate a vector so its projection on ``h`` hits a scalar.

Given a unit axis ``k``, a vector ``p``, a direction ``h``, and a scalar ``d``,
find ``theta`` such that::

    h . (Rot(k, theta) @ p) == d

Up to 2 solutions in the generic feasible case.

**Derivation.** Expand ``Rot(k, theta) p`` via Rodrigues::

    Rot(k, theta) p = cos(theta) p + sin(theta) (k x p)
                   + (1 - cos(theta)) (k . p) k

Taking the dot product with ``h`` and grouping terms in ``cos(theta)`` and
``sin(theta)``::

    h . Rot(k, theta) p = A cos(theta) + B sin(theta) + C

where

    A = h . p - (k . p)(h . k)
    B = h . (k x p)
    C = (k . p)(h . k)

Setting this equal to ``d``::

    A cos(theta) + B sin(theta) = d - C

This is equivalent to ``R cos(theta - phi) = d - C`` with ``R = sqrt(A^2 + B^2)``
and ``phi = atan2(B, A)``:

    theta = phi +/- acos((d - C) / R)

Exact solutions exist when ``|d - C| <= R``; otherwise the LS extension
projects onto the feasible range (``cos(theta - phi) = sign(d - C)``).

**Degenerate case (``p`` along ``k``).** ``R = 0`` means ``p`` is collinear
with ``k``, so ``h . Rot(k, theta) p = C`` for all ``theta``. Returns
``theta = 0`` with ``is_ls`` reflecting whether ``C == d``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["solve"]

_FEASIBILITY_TOL = 1e-9


def solve(
    h: NDArray[np.float64],
    k: NDArray[np.float64],
    p: NDArray[np.float64],
    d: float,
) -> tuple[list[float], bool]:
    """Solve SP4.

    :param h: fixed unit direction the rotated vector's projection aligns with.
    :param k: unit rotation axis.
    :param p: source vector to rotate.
    :param d: target scalar projection value.
    :returns: ``(solutions, is_ls)`` where ``solutions`` is a list of
        ``theta`` values (0, 1, or 2 entries in the feasible case; exactly 1
        in the LS / degenerate cases) and ``is_ls`` is ``True`` when the
        problem is infeasible (the ``theta`` returned is the LS projection).
    """
    hp = float(np.dot(h, p))
    kp = float(np.dot(k, p))
    hk = float(np.dot(h, k))
    coef_a = hp - kp * hk
    coef_b = float(np.dot(h, np.cross(k, p)))
    coef_c = kp * hk
    r_sq = coef_a * coef_a + coef_b * coef_b

    if r_sq < 1e-20:
        # p is collinear with k: rotation has no effect on h . Rot(k, .) p.
        # The projection is always C; exact iff C == d.
        return [0.0], abs(coef_c - d) > _FEASIBILITY_TOL

    r = float(np.sqrt(r_sq))
    rhs = d - coef_c
    phi = float(np.arctan2(coef_b, coef_a))

    # Feasibility: |rhs| must not exceed r beyond floating-point noise. Use
    # an absolute tolerance on the excess (|rhs| - r) so tiny r values do not
    # trigger false infeasibility from ratio inflation.
    if abs(rhs) - r > _FEASIBILITY_TOL:
        # Infeasible. LS projection: cos(theta - phi) = +/- 1 matching sign.
        theta = phi if rhs > 0 else phi + float(np.pi)
        return [theta], True

    # Exact. Clip the ratio to [-1, 1] before acos for numerical safety.
    rhs_over_r = rhs / r
    rhs_clipped = max(-1.0, min(1.0, rhs_over_r))
    delta = float(np.arccos(rhs_clipped))
    if delta < 1e-9:
        # Tangent: single solution.
        return [phi], False
    return [phi + delta, phi - delta], False
