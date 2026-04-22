"""Subproblem 3: rotate a vector so its distance from a point equals a target.

Given a unit axis ``k``, a source vector ``p``, a target point ``q``, and a
scalar distance ``d``, find ``theta`` such that::

    |Rot(k, theta) @ p - q| == d

Up to 2 solutions in the generic feasible case.

**Derivation.** Expand the squared distance::

    |Rot(k, theta) p - q|^2 = |p|^2 - 2 q . Rot(k, theta) p + |q|^2

Setting this equal to ``d^2``::

    q . Rot(k, theta) p = (|p|^2 + |q|^2 - d^2) / 2

which is exactly :func:`ssik.subproblems.sp4.solve` with ``h = q``,
``p = p``, and a derived scalar target. This implementation delegates to SP4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.subproblems import sp4

__all__ = ["solve"]


def solve(
    k: NDArray[np.float64],
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    d: float,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[list[float], bool]:
    """Solve SP3.

    :param k: unit rotation axis.
    :param p: source vector rotated by ``(k, theta)``.
    :param q: target point whose distance to the rotated ``p`` must equal ``d``.
    :param d: target distance (non-negative).
    :param policy: tolerances (forwarded to :func:`sp4.solve`).
    :returns: ``(solutions, is_ls)`` with 0, 1, or 2 ``theta`` entries.
    """
    target = 0.5 * (float(np.dot(p, p)) + float(np.dot(q, q)) - d * d)
    return sp4.solve(q, k, p, target, policy)
