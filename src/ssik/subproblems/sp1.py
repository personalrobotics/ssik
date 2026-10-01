"""Subproblem 1: rotate a vector to match another vector.

Given a unit axis ``k``, a vector ``p``, and a target vector ``q``, find the
angle ``theta`` such that::

    Rot(k, theta) @ p == q

Exact solution exists iff ``|p| == |q|`` and ``k . p == k . q``. Otherwise the
function returns the least-squares optimum: the angle that minimises
``|Rot(k, theta) p - q|^2``. The LS form continuously extends the exact one.

**Solution count:** always exactly 1 (exact or LS), so the return type is a
scalar ``theta`` plus an ``is_ls`` flag.

**Derivation.** Decompose ``p = (k.p)k + p_perp`` where ``p_perp`` is the
component of ``p`` perpendicular to ``k``; similarly for ``q``. Rotating ``p``
around ``k`` leaves ``(k.p)k`` unchanged and rotates ``p_perp`` in the plane
spanned by ``p_perp`` and ``k x p_perp``. Matching ``q_perp``::

    cos(theta) = (p_perp . q_perp) / |p_perp|^2
    sin(theta) = ((k x p_perp) . q_perp) / |p_perp|^2

Both terms are evaluated from ``k x p`` and ``k x q`` (:func:`angle`), which
keeps them accurate when ``p`` and ``q`` lie near the axis.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.subproblems._rotation import _dot3

__all__ = ["angle", "angle_rows", "cross_rows", "solve"]


# Both atan2 terms are built from k x p and k x q, the perpendicular parts
# turned a quarter about k. These are computed to eps * |p| absolute and are of
# size |p_perp|, |q_perp|. For unit k, Lagrange's identity and
# (a x b) x (a x c) = (a . (b x c)) a give
#   (k x p) . (k x q)          = p . q - (k.p)(k.q)   (|p_perp||q_perp| cos)
#   k . ((k x p) x (k x q))    = (k x p) . q          (|p_perp||q_perp| sin)
# The right-hand forms subtract or absorb O(|p||q|) axial terms, so near a
# singularity, where p_perp and q_perp shrink to delta while p and q do not,
# they lose the angle to an error of eps / delta^2; the left-hand forms keep it
# to eps / delta (#661). (k x p) . q is accurate only for an axis-aligned k,
# where k x p carries no rounding to meet q's axial part. angle / angle_rows are
# the one rule for every SP1 site, mirrored by ssik_cpp::sp1_angle and by the
# generated artifacts (ssik.codegen._symbolic.sp1).
def angle(k: NDArray[np.float64], p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """The SP1 angle rotating ``p`` about unit axis ``k`` toward ``q``."""
    # Scalar arithmetic: a 3-vector numpy op costs more than these few flops.
    k0, k1, k2 = float(k[0]), float(k[1]), float(k[2])
    p0, p1, p2 = float(p[0]), float(p[1]), float(p[2])
    q0, q1, q2 = float(q[0]), float(q[1]), float(q[2])
    a0, a1, a2 = k1 * p2 - k2 * p1, k2 * p0 - k0 * p2, k0 * p1 - k1 * p0  # k x p
    b0, b1, b2 = k1 * q2 - k2 * q1, k2 * q0 - k0 * q2, k0 * q1 - k1 * q0  # k x q
    sin_term = k0 * (a1 * b2 - a2 * b1) + k1 * (a2 * b0 - a0 * b2) + k2 * (a0 * b1 - a1 * b0)
    return math.atan2(sin_term, a0 * b0 + a1 * b1 + a2 * b2)


def cross_rows(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise ``a x b`` of broadcastable ``(..., 3)`` arrays, written out by
    component: ``np.cross`` spends most of its time on axis bookkeeping here."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


def angle_rows(
    k: NDArray[np.float64],
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    kxp: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """:func:`angle` row by row over ``(N, 3)`` arrays (``k`` may be one axis).
    ``kxp`` is ``cross_rows(k, p)`` when the caller has it already."""
    if kxp is None:
        kxp = cross_rows(k, p)
    kxq = cross_rows(k, q)
    out: NDArray[np.float64] = np.arctan2((k * cross_rows(kxp, kxq)).sum(-1), (kxp * kxq).sum(-1))
    return out


def solve(
    k: NDArray[np.float64],
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[float, bool]:
    """Solve SP1.

    :param k: unit rotation axis, shape ``(3,)``.
    :param p: vector to rotate, shape ``(3,)``.
    :param q: target vector, shape ``(3,)``.
    :param policy: tolerances. ``subproblem_feasibility`` gates the
        ``is_ls`` boundary between exact and LS regimes.
    :returns: ``(theta, is_ls)`` where ``theta`` is the solution angle in
        radians and ``is_ls`` is ``True`` when the exact feasibility
        conditions do not hold (so ``theta`` is the LS optimum).
    """
    kp = _dot3(k, p)
    kq = _dot3(k, q)
    theta = angle(k, p, q)

    # Feasibility: |p_perp| = |q_perp| and k.p = k.q.
    p_perp_sq = _dot3(p, p) - kp * kp
    q_perp_sq = _dot3(q, q) - kq * kq
    tol = policy.subproblem_feasibility
    is_ls = abs(p_perp_sq - q_perp_sq) > tol or abs(kp - kq) > tol
    return theta, is_ls
