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
    "ApproxSrsClassification",
    "axes_meet_at_common_point",
    "axis_intersect",
    "axis_parallel",
    "is_approximately_srs_7r",
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


@dataclass(frozen=True)
class ApproxSrsClassification:
    """Result of :func:`is_approximately_srs_7r` -- approximate SRS evidence
    plus the measured drift magnitudes that disqualified strict
    classification.

    Consumed by :mod:`ssik.solvers.seven_r.srs_polished`, which uses the
    Singh-Kreutz solver as a warm-start factory and LM-polishes each
    candidate against the original (non-snapped) URDF FK.
    """

    base: SrsClassification
    shoulder_drift_m: float
    wrist_drift_m: float

    @property
    def max_drift_m(self) -> float:
        return max(self.shoulder_drift_m, self.wrist_drift_m)


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
        # Relaxed concurrence: the three axes meet at a common point p (delegated
        # -- pairwise-non-parallel + lstsq intersection + all-axes-through-p live
        # once in axes_meet_at_common_point, scale-relative per #388).
        p = axes_meet_at_common_point(joints, (i, i + 1, i + 2), policy)
        if p is None:
            continue

        # Strict consolidation check (#155): for the IK-Geo ``spherical`` family
        # setup ``p[3] = T_left[i] + T_left[i+1] + T_left[i+2]`` to correctly
        # represent the offset from joint ``i-1`` to the wrist intersection, the
        # **last two** wrist joint origins must lie at ``p``. Joint ``i``'s origin
        # can be anywhere on its axis (its rotation applies *before* consolidating).
        # iiwa14 fails this: its wrist axes meet but joints 5/6/7 origins drift
        # off the intersection (z=1.18, 1.261, 1.342) -- so it is correctly
        # rejected here rather than mis-routed to a solver its geometry breaks.
        ob, oc = origins[i + 1], origins[i + 2]
        tol = policy.axis_intersect * _char_length([origins[i], ob, oc])
        if float(_norm3(ob - p)) >= tol or float(_norm3(oc - p)) >= tol:
            continue

        return (i, i + 1, i + 2)
    return None


def _char_length(pts: list[NDArray[np.float64]]) -> float:
    """Characteristic length of a joint cluster: the largest origin distance
    from the base (floored at 1 m). Used to make the axis-intersection tolerance
    scale-relative -- an absolute ``policy.axis_intersect`` (10 nm) is scale-blind
    and mis-rejects a large arm whose axes are concurrent by design but drift by
    the URDF's coordinate rounding, which grows with coordinate magnitude (#388).
    """
    return max(1.0, max((float(_norm3(p)) for p in pts), default=1.0))


def _common_point_and_max_drift(
    axes: list[NDArray[np.float64]],
    pts: list[NDArray[np.float64]],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[NDArray[np.float64] | None, float]:
    """Common point of the first two axis *lines* + the max perpendicular drift
    of every axis from it. ``(None, inf)`` if any consecutive pair is parallel
    (the intersection point is then ill-defined).

    The one geometry kernel behind :func:`axes_meet_at_common_point`,
    :func:`three_consecutive_intersecting`, and :func:`_max_axis_drift` (which
    used to each carry a near-identical copy -- one with a hand-coded ``1e-9``
    parallel guard that diverged from ``policy.axis_parallel``).
    """
    for k in range(len(axes) - 1):
        if axis_parallel(axes[k], axes[k + 1], policy):
            return None, float("inf")
    m = np.column_stack([axes[0], -axes[1]])
    sol, *_ = np.linalg.lstsq(m, pts[1] - pts[0], rcond=None)
    common = pts[0] + float(sol[0]) * axes[0]
    max_perp = 0.0
    for k in range(len(axes)):
        delta = common - pts[k]
        perp = delta - _dot3(delta, axes[k]) * axes[k]
        max_perp = max(max_perp, float(_norm3(perp)))
    return common, max_perp


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

    common, max_drift = _common_point_and_max_drift(axes, pts, policy)
    if common is None:  # a consecutive pair is parallel -- no intersection point
        return None
    # Scale-relative tolerance (#388): the axes are concurrent iff every axis
    # passes through the common point within the URDF's coordinate-rounding
    # floor, which grows with arm size.
    if max_drift >= policy.axis_intersect * _char_length(pts):
        return None
    return common


def _classify_srs_7r_geometric(
    kb: KinBody,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> SrsClassification | None:
    """Geometric-only SRS classification: shoulder axes (0, 1, 2) and wrist
    axes (4, 5, 6) each meet at a common point. Does NOT check the
    Z*Z Euler structural requirement that the strict Singh-Kreutz solver
    needs for FK closure.

    Used as the internal guard inside :func:`ssik.solvers.seven_r.srs.solve`
    and by :func:`ssik.solvers.seven_r.srs_polished.solve` (which tolerates
    the wrong-q-vector candidates produced when Z*Z fails because its LM
    polish pass rescues them). Public dispatch / tier-0 gating goes through
    the strict :func:`is_srs_7r` instead.
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


def is_srs_7r(
    kb: KinBody,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> SrsClassification | None:
    """Detect SRS-class 7R topology: shoulder-roll-spherical with shoulder
    axes (joints 0, 1, 2) meeting at one point and wrist axes (joints
    4, 5, 6) meeting at one point (joint 3 is the elbow).

    Returns the :class:`SrsClassification` evidence (shoulder pivot, elbow
    index, wrist pivot, joint splits) when the topology matches; ``None``
    otherwise. The predicate is a pure function of the chain's geometry;
    no per-arm hardcoding.

    Real arms that pass this predicate (verified against published
    URDF/MJCF):

    * KUKA iiwa LBR (7 / 14 / R820 / R14 / ...) -- canonical Z*Z
    * Flexiv Rizon 4 / 10
    * Kinova Gen3 (7-DOF)
    * Sawyer (Rethink)
    * Baxter (per-arm)
    * Kassow KR810 / KR1410
    * Galaxea R1 Pro -- y-x-z shoulder / z-y-x wrist (non-Z*Z; #354)

    Arms that FAIL (different topology, different solver families):

    * Franka Panda / FR3 -- anthropomorphic 7R (shoulder spherical but
      wrist axes don't meet at one common point in the home configuration).
    * xArm7 -- mixed structure with non-canonical wrist pivot.

    No Z*Z (parallel first/third axis) requirement (#354): the solver in
    :mod:`ssik.solvers.seven_r.srs` now decomposes both shoulder and wrist
    triples via the general Davenport (arbitrary three-axis) decomposition,
    which recovers an arbitrary target rotation for *any* concurrent triple,
    not just ``R_z R_y R_z``. The earlier Z*Z gate (#307) guarded the old
    ZYZ-Euler extraction that silently mis-solved non-Z*Z chains (OpenArm
    v2.0, R1 Pro); with the general extraction the gate is no longer needed
    and concurrent-axis non-Z*Z arms route straight to the native solver.

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
    return _classify_srs_7r_geometric(kb, policy)


def _max_axis_drift(
    joints: list[Joint],
    indices: tuple[int, ...],
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> tuple[float, NDArray[np.float64] | None]:
    """Max perpendicular drift of the triple's axes from their best-fit common
    point (and that point). Always returns a numerical drift, ``inf`` if a pair
    is parallel; used by :func:`is_approximately_srs_7r` to gate on a
    user-supplied drift budget. Shares the geometry kernel with the concurrence
    predicates (so its parallel guard now uses ``policy.axis_parallel``, not a
    divergent hand-coded ``1e-9``).
    """
    if len(indices) < 2:
        return 0.0, None
    origins = joint_origins(joints)
    axes = [joints[i].axis for i in indices]
    pts = [origins[i] for i in indices]
    common, max_perp = _common_point_and_max_drift(axes, pts, policy)
    return max_perp, common


def is_approximately_srs_7r(
    kb: KinBody,
    max_drift_m: float = 0.04,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
) -> ApproxSrsClassification | None:
    """Detect approximate SRS-class 7R topology with a user-supplied drift gate.

    Strict :func:`is_srs_7r` rejects arms whose shoulder/wrist axes only
    *approximately* meet at a common point (Kinova Gen3: 12 mm shoulder
    drift, 0.4 mm wrist drift; well above the default ``axis_intersect =
    1e-8``). This relaxed variant accepts any arm whose maximum axis
    drift is at most ``max_drift_m`` -- typically picked to keep the
    snap-and-polish trajectory inside Newton's basin (~3-5 cm task
    space empirically).

    Returns :class:`ApproxSrsClassification` carrying the best-fit
    pivots + the measured per-triple drifts. Caller is expected to
    polish the algebraic candidates via LM (see
    :mod:`ssik.solvers.seven_r.srs_polished`).

    The drift gate refuses arms whose offsets exceed the basin (Flexiv
    Rizon 4: 151 mm wrist drift; Kassow KR810: 111 mm wrist drift).
    Those arms continue to dispatch to ``jointlock + HP``.

    Parallel-axis triples are still rejected (the SRS algorithm is
    ill-defined when axes are parallel, regardless of drift).

    Z*Z structural requirement (#307): in addition to drift, the
    approximate-SRS solver path also requires the same
    Z*Z (parallel first/third axis) structure as :func:`is_srs_7r`,
    because ``srs_polished`` internally uses the strict ``srs.solve``
    to produce warm-start candidates and the LM polish needs those
    seeds to be close to a real solution. On non-Z*Z arms (Enactic
    OpenArm v2.0), the warm starts land 1-3 m off target and LM either
    fails to converge or converges glacially -- ``jointlock.seven_r``
    is 5x faster and reaches machine precision instead.
    """
    if len(kb.joints) != 7:
        return None
    shoulder = (0, 1, 2)
    wrist = (4, 5, 6)
    s_drift, s_pivot = _max_axis_drift(kb.joints, shoulder)
    w_drift, w_pivot = _max_axis_drift(kb.joints, wrist)
    if s_pivot is None or w_pivot is None:
        return None
    if s_drift > max_drift_m or w_drift > max_drift_m:
        return None
    # Z*Z gate (#307): mirror of the check in ``is_srs_7r`` -- the
    # approximate-SRS solver path inherits the strict solver's hidden
    # ZYZ Euler assumption via its warm-start dependency. Reject
    # non-Z*Z chains here so the dispatcher falls through to
    # ``jointlock.seven_r`` instead.
    if not axis_parallel(kb.joints[shoulder[0]].axis, kb.joints[shoulder[2]].axis, policy):
        return None
    if not axis_parallel(kb.joints[wrist[0]].axis, kb.joints[wrist[2]].axis, policy):
        return None
    base = SrsClassification(
        shoulder_indices=shoulder,
        elbow_index=3,
        wrist_indices=wrist,
        shoulder_pivot=s_pivot,
        wrist_pivot=w_pivot,
    )
    return ApproxSrsClassification(
        base=base,
        shoulder_drift_m=s_drift,
        wrist_drift_m=w_drift,
    )
