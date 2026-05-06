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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.subproblems._rotation import _cross3, _dot3, _norm3

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from ssik._kinbody import Joint, KinBody

__all__ = [
    "axes_meet_at_common_point",
    "axis_intersect",
    "axis_parallel",
    "is_srs_7r",
    "joint_origins",
    "three_consecutive_intersecting",
    "three_consecutive_parallel",
]


@dataclass(frozen=True)
class SrsClassification:
    """Result of :func:`is_srs_7r` -- the SRS-class topology evidence.

    Carries the geometric pivots a Singh-Kreutz solver needs (shoulder
    point, elbow joint, wrist point) along with the joint-index split.
    Returned by the predicate; consumed by
    :mod:`ssik.solvers.seven_r.srs`.
    """

    shoulder_indices: tuple[int, int, int]
    elbow_index: int
    wrist_indices: tuple[int, int, int]
    shoulder_pivot: NDArray[np.float64]
    wrist_pivot: NDArray[np.float64]


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
    """Find the first triple of consecutive joints whose **origins** all lie
    at a common axes-intersection point (the IK-Geo spherical-wrist condition).

    Three lines in 3D that are pairwise intersecting do **not** necessarily
    share a common point -- they can form a triangle in space. And even
    when they do share a common point, the IK-Geo ``spherical`` family
    requires more: the joint origins must *coincide at* that intersection
    point (within ``policy.axis_intersect``), because the inner solvers
    consolidate the wrist offset as ``p[3] = T_left[3] + T_left[4] +
    T_left[5]`` and assume the wrist rotations leave that consolidation
    invariant -- which holds iff ``T_left[i+1]`` and ``T_left[i+2]``
    contribute zero translation to the wrist intersection.

    Parallel-axis triples are rejected (axis_intersect short-circuits on
    parallel, which is the degenerate case where "intersection" is
    ill-defined).

    See #155 for the prior loose-predicate behaviour and the iiwa silent-
    failure repro.
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
        if _norm3(perp) >= policy.axis_intersect:
            continue

        # Strict consolidation check (#155): for the IK-Geo ``spherical``
        # family setup ``p[3] = T_left[i] + T_left[i+1] + T_left[i+2]`` to
        # correctly represent the offset from joint ``i-1`` to the wrist
        # intersection, the **last two** wrist joint origins (``i+1`` and
        # ``i+2``) must lie at the intersection point. Equivalently:
        # ``T_left[i+1]`` and ``T_left[i+2]`` must contribute zero net
        # displacement *off the axis intersection*, which is satisfied when
        # joint ``i+1`` and ``i+2`` origins coincide with ``p``. Joint
        # ``i``'s origin can be anywhere on its axis (the rotation
        # ``R(axes[i], q_i)`` applies *before* consolidating).
        #
        # iiwa14 fails this check: wrist axes meet at z=1.18, but the
        # MJCF places joint 5/6/7 origins at z=1.18, 1.261, 1.342 -- the
        # 2nd and 3rd wrist origins drift off the intersection. Predicate
        # used to silently mis-classify iiwa as spherical, sending it
        # to a solver that hard-fails on its geometry.
        if (
            float(_norm3(ob - p)) >= policy.axis_intersect
            or float(_norm3(oc - p)) >= policy.axis_intersect
        ):
            continue

        return (i, i + 1, i + 2)
    return None


def axes_meet_at_common_point(
    joints: list[Joint],
    indices: tuple[int, ...],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> NDArray[np.float64] | None:
    """Return the common point shared by ``len(indices)`` joint axes, or
    ``None`` if they don't all pass through a single point within
    ``policy.axis_intersect`` drift.

    *Relaxed* concurrence -- requires the axis LINES to meet at a common
    point but does NOT require joint origins to coincide with that
    point. This is sufficient for kinematic structure detection (e.g.
    Singh-Kreutz SRS-class solvers, which work with axis pivots, not
    joint origins) but NOT sufficient for IK-Geo ``spherical`` family
    solvers (which need origin coincidence -- see
    :func:`three_consecutive_intersecting` and #155).

    The pairwise non-parallel + axis-intersect predicate must hold for
    every consecutive pair; the common-point check is then performed by
    intersecting the first two axes and verifying every other axis
    passes through that point.
    """
    if len(indices) < 2:
        return None
    origins = joint_origins(joints)
    axes = [joints[i].axis for i in indices]
    pts = [origins[i] for i in indices]

    # Pairwise non-parallel guard: a degenerate set with parallel axes
    # has ill-defined "intersection point".
    for k in range(len(indices) - 1):
        if axis_parallel(axes[k], axes[k + 1], policy):
            return None

    # Solve for the common point of axes[0] and axes[1].
    M = np.column_stack([axes[0], -axes[1]])
    sol, *_ = np.linalg.lstsq(M, pts[1] - pts[0], rcond=None)
    common = pts[0] + float(sol[0]) * axes[0]

    # Verify every axis passes through that point (perpendicular component
    # of (common - origin) relative to axis must be < tol).
    for k in range(len(indices)):
        delta = common - pts[k]
        perp = delta - _dot3(delta, axes[k]) * axes[k]
        if float(_norm3(perp)) >= policy.axis_intersect:
            return None

    return common


def is_srs_7r(
    kb: KinBody,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> SrsClassification | None:
    """Detect SRS-class 7R topology: shoulder-roll-spherical with shoulder
    axes (joints 0, 1, 2) meeting at one point and wrist axes (joints
    4, 5, 6) meeting at one point.

    Returns the :class:`SrsClassification` evidence (shoulder pivot, elbow
    index, wrist pivot, joint splits) when the topology matches; ``None``
    otherwise. The predicate is a pure function of the chain's geometry;
    no per-arm hardcoding.

    Real arms that pass this predicate (verified against published
    URDF/MJCF):

    * KUKA iiwa LBR (7 / 14 / R820 / R14 / ...)
    * Flexiv Rizon 4 / 10
    * Kinova Gen3 (7-DOF)
    * Sawyer (Rethink)
    * Baxter (per-arm)
    * Kassow KR810 / KR1410

    Arms that FAIL (different topology, different solver families):

    * Franka Panda / FR3 -- anthropomorphic 7R (shoulder spherical but
      wrist axes don't meet at one common point in the home configuration).
    * xArm7 -- mixed structure with non-canonical wrist pivot.

    Distinguishing axis concurrence (this predicate) vs. axis +
    origin coincidence (the IK-Geo ``spherical`` predicate, #155): a
    common point of axes is sufficient for the Singh-Kreutz
    parameterization to apply -- the algorithm operates on the axis
    pivots in 3-space, not on joint origins. The IK-Geo ``spherical``
    family additionally needs joint origins to coincide with the
    intersection point because its inner consolidation
    ``p[3] = T_left[3] + T_left[4] + T_left[5]`` only represents the
    correct offset when origins lie at the intersection.
    """
    if len(kb.joints) != 7:
        return None
    shoulder = (0, 1, 2)
    wrist = (4, 5, 6)
    s_pivot = axes_meet_at_common_point(kb.joints, shoulder, policy)
    if s_pivot is None:
        return None
    w_pivot = axes_meet_at_common_point(kb.joints, wrist, policy)
    if w_pivot is None:
        return None
    return SrsClassification(
        shoulder_indices=shoulder,
        elbow_index=3,
        wrist_indices=wrist,
        shoulder_pivot=s_pivot,
        wrist_pivot=w_pivot,
    )
