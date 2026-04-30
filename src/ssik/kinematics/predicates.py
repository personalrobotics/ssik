"""Kinematic-structure predicates: axis parallelism, axis intersection,
three-consecutive-intersecting, three-consecutive-parallel.

These are the primitives the solver dispatcher uses to classify a chain into
a closed-form kinematic family ("spherical wrist? three parallel? both?").
Every predicate takes a :class:`~ssik.core.tolerances.TolerancePolicy` so
behaviour is reproducible and user-tunable.

Preconditions
-------------
All functions assume a **POE-normalized** :class:`KinBody` (axes expressed in
the base frame at ``q = 0``, ``T_left`` as pure translation). Non-normalized
inputs produce undefined results. See
:func:`ssik._urdf.load_urdf_kinbody_normalized`.

Axes are assumed to be unit-length up to numerical noise. The URDF loader
(urchin) normalizes them on load; POE normalization preserves unit length
because cumulative rpy products stay orthonormal. If a user hand-constructs
a KinBody with non-unit axes, the parallel/intersect predicates will
misbehave -- silent failures, not raised errors. A future
:class:`TolerancePolicy` field can gate this with a runtime check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.subproblems._rotation import _cross3, _dot3, _norm3

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from numpy.typing import NDArray

    from ssik._kinbody import Joint

__all__ = [
    "axis_intersect",
    "axis_parallel",
    "joint_origins",
    "three_consecutive_intersecting",
    "three_consecutive_parallel",
]


def joint_origins(joints: list[Joint]) -> list[NDArray[np.float64]]:
    """Return each joint's axis-origin point in the base frame at ``q = 0``.

    On a POE-normalized chain, ``T_left`` for every joint is pure translation,
    so the cumulative position after joint ``i`` is just the sum of the first
    ``i + 1`` translations. The result is the point each joint's axis passes
    through, which the intersection predicates need.
    """
    cum = np.zeros(3, dtype=np.float64)
    origins: list[NDArray[np.float64]] = []
    for j in joints:
        cum = cum + j.T_left[:3, 3]
        origins.append(cum.copy())
    return origins


def axis_parallel(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> bool:
    """Return ``True`` if unit-vector axes ``a`` and ``b`` are parallel or
    anti-parallel within ``policy.axis_parallel``.

    Implementation: ``||a x b||`` is ``sin(theta)`` for unit vectors, which
    small-angle approximates to the angular misalignment in radians.
    """
    # ``float(...)`` reasserts the boundary type: ``_norm3`` is decorated
    # ``@cython.ccall``, which widens to ``Any`` for mypy.
    return float(_norm3(_cross3(a, b))) < policy.axis_parallel


def axis_intersect(
    a: NDArray[np.float64],
    oa: NDArray[np.float64],
    b: NDArray[np.float64],
    ob: NDArray[np.float64],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> bool:
    """Return ``True`` if lines ``(oa + t*a)`` and ``(ob + s*b)`` share a
    common point within ``policy.axis_intersect``.

    For skew lines the shortest distance is
    ``|dot(cross(a, b), (ob - oa))| / ||cross(a, b)||``. For parallel lines
    (``||cross(a, b)|| < policy.axis_parallel``) the two lines are either
    coincident or disjoint; the distance is then the perpendicular component
    of ``(ob - oa)`` after projecting out ``a``.
    """
    cross = _cross3(a, b)
    cross_norm = float(_norm3(cross))
    delta = ob - oa
    if cross_norm < policy.axis_parallel:
        # Parallel case: shortest distance is the perpendicular component
        # of delta relative to a (equivalently b -- they're parallel).
        perp = delta - _dot3(delta, a) * a
        return float(_norm3(perp)) < policy.axis_intersect
    distance = abs(float(_dot3(cross, delta))) / cross_norm
    return distance < policy.axis_intersect


def three_consecutive_parallel(
    joints: list[Joint],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[int, int, int] | None:
    """Find the first triple of consecutive joints whose axes are all pairwise
    parallel within ``policy.axis_parallel``.

    Returns the ``(i, i+1, i+2)`` indices of the first matching triple, or
    ``None`` if no such triple exists. This is the structural condition for
    the UR-class (three-inner-parallel) kinematic family.
    """
    if len(joints) < 3:
        return None
    for i in range(len(joints) - 2):
        a = joints[i].axis
        b = joints[i + 1].axis
        c = joints[i + 2].axis
        # All three pairwise parallel. Transitivity for unit vectors within
        # epsilon gives (a || c) from (a || b) and (b || c), but within tol
        # it can drift; check explicitly.
        if (
            axis_parallel(a, b, policy)
            and axis_parallel(b, c, policy)
            and axis_parallel(a, c, policy)
        ):
            return (i, i + 1, i + 2)
    return None


def three_consecutive_intersecting(
    joints: list[Joint],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[int, int, int] | None:
    """Find the first triple of consecutive joints whose axes share a common
    point (Pieper's spherical-wrist condition).

    Three lines in 3D that are pairwise intersecting do **not** necessarily
    share a common point -- they can form a triangle in space. This function
    requires strict coincidence: computes the intersection point of axes 0
    and 1, then verifies axis 2 passes through that same point within
    ``policy.axis_intersect``.

    Parallel-axis triples are rejected (axis_intersect short-circuits on
    parallel, which is the degenerate case where "intersection" is
    ill-defined).
    """
    if len(joints) < 3:
        return None
    origins = joint_origins(joints)
    for i in range(len(joints) - 2):
        a, b, c = joints[i].axis, joints[i + 1].axis, joints[i + 2].axis
        oa, ob, oc = origins[i], origins[i + 1], origins[i + 2]

        # Skip triples where any pair is parallel -- the spherical-wrist
        # condition requires intersecting non-parallel axes. Parallel
        # triples are the three_consecutive_parallel family, a separate case.
        if axis_parallel(a, b, policy) or axis_parallel(b, c, policy):
            continue

        # All three pairwise intersecting?
        if not (
            axis_intersect(a, oa, b, ob, policy)
            and axis_intersect(b, ob, c, oc, policy)
            and axis_intersect(a, oa, c, oc, policy)
        ):
            continue

        # Common-point check: compute the (a, b) intersection via least-squares
        # on the 3x2 linear system [a, -b] @ [t, s]^T = (ob - oa), then verify
        # c passes through that point.
        M = np.column_stack([a, -b])
        sol, *_ = np.linalg.lstsq(M, ob - oa, rcond=None)
        t = float(sol[0])
        p = oa + t * a

        delta = p - oc
        perp = delta - _dot3(delta, c) * c
        if _norm3(perp) < policy.axis_intersect:
            return (i, i + 1, i + 2)
    return None
