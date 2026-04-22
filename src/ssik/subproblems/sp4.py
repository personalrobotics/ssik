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

Exact solutions exist when ``|d - C| <= R + subproblem_feasibility``;
otherwise the LS extension projects onto the feasible range
(``cos(theta - phi) = sign(d - C)``).

**Degenerate case (``p`` along ``k``).** ``R ~ 0`` (below
``subproblem_degeneracy``) means ``p`` is collinear with ``k``, so
``h . Rot(k, theta) p = C`` for all ``theta``. Returns ``theta = 0`` with
``is_ls`` reflecting whether ``C == d``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy

__all__ = ["solve"]


def solve(
    h: NDArray[np.float64],
    k: NDArray[np.float64],
    p: NDArray[np.float64],
    d: float,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[float], bool]:
    """Solve SP4.

    :param h: fixed unit direction the rotated vector's projection aligns with.
    :param k: unit rotation axis.
    :param p: source vector to rotate.
    :param d: target scalar projection value.
    :param policy: tolerances. ``subproblem_feasibility`` gates is_ls vs
        exact; ``subproblem_degeneracy`` gates the ``p`` collinear with
        ``k`` fallback.
    :returns: ``(solutions, is_ls)``.
    """
    hp = float(np.dot(h, p))
    kp = float(np.dot(k, p))
    hk = float(np.dot(h, k))
    coef_a = hp - kp * hk
    coef_b = float(np.dot(h, np.cross(k, p)))
    coef_c = kp * hk
    r_sq = coef_a * coef_a + coef_b * coef_b

    # Scale-aware degeneracy threshold: r_sq is squared magnitude.
    deg_sq = policy.subproblem_degeneracy * policy.subproblem_degeneracy
    if r_sq < deg_sq:
        # p collinear with k: rotation doesn't change the projection.
        return [0.0], abs(coef_c - d) > policy.subproblem_feasibility

    r = float(np.sqrt(r_sq))
    rhs = d - coef_c
    phi = float(np.arctan2(coef_b, coef_a))

    # Absolute-tolerance feasibility: |rhs| must not exceed R by more than
    # subproblem_feasibility. Ratio tolerance is unstable for small r.
    if abs(rhs) - r > policy.subproblem_feasibility:
        theta = phi if rhs > 0 else phi + float(np.pi)
        return [theta], True

    rhs_over_r = rhs / r
    rhs_clipped = max(-1.0, min(1.0, rhs_over_r))
    delta = float(np.arccos(rhs_clipped))
    if delta < policy.subproblem_feasibility:
        # Tangent: single solution.
        return [phi], False
    return [phi + delta, phi - delta], False
