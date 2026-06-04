"""Tests for the kinematic-structure predicates.

Coverage split in three:

1. **Low-level primitives** (:func:`axis_parallel`, :func:`axis_intersect`)
   exercised on hand-built axes with deliberate geometric properties.
2. **Three-consecutive classifiers** on real-robot fixtures: Puma 560 should
   classify as spherical-wrist (three-consecutive-intersecting at joints
   ``(3, 4, 5)``); UR5 should classify as three-consecutive-parallel at
   joints ``(1, 2, 3)``.
3. **Tolerance sensitivity** with synthetic chains that live at deliberate
   near-misses, verifying that ``TolerancePolicy`` behaves correctly at the
   boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ssik import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik._kinbody import Joint, KinBody, Link
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.internals import describe_topology
from ssik.kinematics import (
    axis_intersect,
    axis_parallel,
    joint_origins,
    three_consecutive_intersecting,
    three_consecutive_parallel,
)

FIXTURES = Path(__file__).parent / "fixtures"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_chain(
    positions: list[Any],
    axes: list[Any],
) -> KinBody:
    """Hand-build a POE-normalized KinBody from offsets and axes.

    ``positions[i]`` is the base-frame offset from joint ``i-1`` to joint ``i``
    (``positions[0]`` is from base to joint 0). ``axes[i]`` is joint ``i``'s
    axis in the base frame, unit-length. Both lists must have the same length
    ``N``; the result has ``N`` joints and ``N + 1`` links.
    """
    assert len(positions) == len(axes)
    n = len(positions)
    link_names = ["base_link", *[f"link_{i}" for i in range(1, n)], "ee_link"]
    links = [Link(name=name) for name in link_names]
    joints: list[Joint] = []
    for i, (p, a) in enumerate(zip(positions, axes, strict=True)):
        T_left = np.eye(4, dtype=np.float64)
        T_left[:3, 3] = p
        joints.append(
            Joint(
                name=f"j{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_left,
                T_right=np.eye(4, dtype=np.float64),
                axis=np.asarray(a, dtype=np.float64),
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


def _unit(v: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr / float(np.linalg.norm(arr))


# --------------------------------------------------------------------------- #
# axis_parallel
# --------------------------------------------------------------------------- #


def test_axis_parallel_identical_vectors() -> None:
    a = np.array([1.0, 0.0, 0.0])
    assert axis_parallel(a, a)


def test_axis_parallel_anti_parallel_accepted() -> None:
    """Anti-parallel unit vectors should count as parallel -- the cross
    product is zero-magnitude in both the parallel and anti-parallel case,
    and downstream solvers don't care about sign of rotation axes."""
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, -1.0])
    assert axis_parallel(a, b)


def test_axis_parallel_orthogonal_rejected() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert not axis_parallel(a, b)


def test_axis_parallel_near_miss_rejected_by_default() -> None:
    """Angular misalignment of 1 milliradian is rejected at default 1e-8."""
    a = np.array([1.0, 0.0, 0.0])
    b = _unit((np.cos(1e-3), np.sin(1e-3), 0.0))
    assert not axis_parallel(a, b)


def test_axis_parallel_near_miss_accepted_with_loose_tol() -> None:
    """Same misalignment with a loose policy passes."""
    a = np.array([1.0, 0.0, 0.0])
    b = _unit((np.cos(1e-3), np.sin(1e-3), 0.0))
    assert axis_parallel(a, b, TolerancePolicy(axis_parallel=1e-2))


# --------------------------------------------------------------------------- #
# axis_intersect
# --------------------------------------------------------------------------- #


def test_axis_intersect_two_axes_through_same_point() -> None:
    """Two orthogonal axes both passing through the origin intersect there."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    origin = np.zeros(3)
    assert axis_intersect(a, origin, b, origin)


def test_axis_intersect_two_axes_through_same_non_origin_point() -> None:
    p = np.array([0.3, -0.5, 1.2])
    # Pick two non-parallel axes; parameterize lines so they both pass through p.
    a = np.array([1.0, 0.0, 0.0])
    b = _unit((0.2, 1.0, 0.3))
    oa = p - 4.0 * a  # shift along a
    ob = p + 2.5 * b  # shift along b
    assert axis_intersect(a, oa, b, ob)


def test_axis_intersect_skew_lines_rejected() -> None:
    """Two non-intersecting skew lines are rejected."""
    a = np.array([1.0, 0.0, 0.0])  # along +x
    b = np.array([0.0, 1.0, 0.0])  # along +y, but offset in z
    oa = np.array([0.0, 0.0, 0.0])
    ob = np.array([0.0, 0.0, 0.5])  # 0.5 above a's line
    assert not axis_intersect(a, oa, b, ob)


def test_axis_intersect_parallel_coincident_accepted() -> None:
    """Two coincident parallel lines (same direction, both through origin)
    satisfy intersection -- their shortest distance is 0."""
    a = np.array([1.0, 0.0, 0.0])
    assert axis_intersect(a, np.zeros(3), a, np.zeros(3))


def test_axis_intersect_parallel_disjoint_rejected() -> None:
    """Parallel but non-coincident lines don't intersect."""
    a = np.array([1.0, 0.0, 0.0])
    oa = np.zeros(3)
    ob = np.array([0.0, 1.0, 0.0])
    assert not axis_intersect(a, oa, a, ob)


def test_axis_intersect_near_miss_boundary() -> None:
    """A 1e-7 offset is rejected at default 1e-8 tol, accepted at 1e-6."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    oa = np.zeros(3)
    ob = np.array([0.0, 0.0, 1e-7])
    assert not axis_intersect(a, oa, b, ob)
    assert axis_intersect(a, oa, b, ob, TolerancePolicy(axis_intersect=1e-6))


# --------------------------------------------------------------------------- #
# joint_origins
# --------------------------------------------------------------------------- #


def test_joint_origins_cumulative_translation() -> None:
    kb = _make_chain(
        positions=[(1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)],
        axes=[(0, 0, 1)] * 3,
    )
    origs = joint_origins(kb.joints)
    assert np.allclose(origs[0], [1.0, 0.0, 0.0])
    assert np.allclose(origs[1], [1.0, 2.0, 0.0])
    assert np.allclose(origs[2], [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# three_consecutive_parallel -- UR5 fixture + synthetic
# --------------------------------------------------------------------------- #


def test_three_consecutive_parallel_found_on_synthetic() -> None:
    kb = _make_chain(
        positions=[(0, 0, 0.1)] * 4,
        axes=[(0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 1, 0)],
    )
    # Joints 1, 2, 3 all share axis (0, 1, 0).
    assert three_consecutive_parallel(kb.joints) == (1, 2, 3)


def test_three_consecutive_parallel_none_when_no_triple() -> None:
    kb = _make_chain(
        positions=[(0, 0, 0.1)] * 3,
        axes=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    assert three_consecutive_parallel(kb.joints) is None


def test_three_consecutive_parallel_finds_anti_parallel() -> None:
    """Anti-parallel axes should still count -- rotations are symmetric
    under axis reversal for Pieper-family classification."""
    kb = _make_chain(
        positions=[(0, 0, 0.1)] * 3,
        axes=[(0, 1, 0), (0, -1, 0), (0, 1, 0)],
    )
    assert three_consecutive_parallel(kb.joints) == (0, 1, 2)


def test_three_consecutive_parallel_ur5() -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "wrist_3_link")
    # Documented in test_urdf_normalize.py: UR5 joints 1, 2, 3 (and 5) are
    # all along (0, -1, 0) in base frame. The classifier finds the first
    # triple, which is (1, 2, 3).
    assert three_consecutive_parallel(kb.joints) == (1, 2, 3)


# --------------------------------------------------------------------------- #
# three_consecutive_intersecting -- Puma fixture + synthetic
# --------------------------------------------------------------------------- #


def test_three_consecutive_intersecting_found_on_synthetic() -> None:
    """Three axes through the same point (origin), all non-parallel."""
    kb = _make_chain(
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        axes=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    assert three_consecutive_intersecting(kb.joints) == (0, 1, 2)


def test_three_consecutive_intersecting_rejects_triangle() -> None:
    """Three axes that pairwise intersect at three *different* points should
    be rejected -- they form a triangle, not a common-point pencil."""
    # Axes along +x, +y, +z, originating at three different non-collinear points.
    # Pair (0,1) intersects at (0,0,0). Pair (1,2) intersects at (1,0,0).
    # Pair (0,2) does not intersect at (0,0,0) or (1,0,0) because axes are
    # skew relative to each other in this configuration.
    kb = _make_chain(
        positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        axes=[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    # Origins: j0 at (0,0,0), j1 at (1,0,0), j2 at (1,1,0).
    # j0-j1: line x=free,y=0,z=0 vs line x=1,y=free,z=0 -> intersect at (1,0,0).
    # j1-j2: line x=1,y=free,z=0 vs line x=1,y=1,z=free -> intersect at (1,1,0).
    # j0-j2: line x=free,y=0,z=0 vs line x=1,y=1,z=free -> skew, no intersection.
    assert three_consecutive_intersecting(kb.joints) is None


def test_three_consecutive_intersecting_rejects_parallel_triple() -> None:
    """Parallel axes are never classified as intersecting, even if they
    happen to be coincident (parallel triples go through the
    three_consecutive_parallel path instead)."""
    kb = _make_chain(
        positions=[(0, 0, 0)] * 3,
        axes=[(0, 0, 1), (0, 0, 1), (0, 0, 1)],
    )
    assert three_consecutive_intersecting(kb.joints) is None
    assert three_consecutive_parallel(kb.joints) == (0, 1, 2)


def test_three_consecutive_intersecting_puma560() -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")
    # Puma 560 has a classic spherical wrist at joints 3, 4, 5 -- the
    # textbook proof-of-concept for Pieper's decomposition.
    assert three_consecutive_intersecting(kb.joints) == (3, 4, 5)


def test_three_consecutive_intersecting_short_chain_returns_none() -> None:
    kb = _make_chain(
        positions=[(0, 0, 0), (0, 0, 0)],
        axes=[(1, 0, 0), (0, 1, 0)],
    )
    assert three_consecutive_intersecting(kb.joints) is None


# --------------------------------------------------------------------------- #
# describe_topology
# --------------------------------------------------------------------------- #


def test_describe_topology_ur5() -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "wrist_3_link")
    report = describe_topology(kb)
    assert report.dof == 6
    assert report.three_consecutive_parallel == (1, 2, 3)
    # UR5 does not have three consecutive intersecting axes -- that's why
    # this arm is not Pieper-compatible. But normalized UR5 does have
    # joints (1, 2, 3) all parallel; report surfaces that.
    assert report.three_consecutive_intersecting is None


def test_describe_topology_puma560() -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")
    report = describe_topology(kb)
    assert report.dof == 6
    assert report.three_consecutive_intersecting == (3, 4, 5)


def test_describe_topology_is_frozen() -> None:
    """TopologyReport is a frozen dataclass -- callers cannot mutate it."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "wrist_3_link")
    report = describe_topology(kb)
    with pytest.raises(Exception, match=r"cannot assign|is not writable|FrozenInstance"):
        report.dof = 7  # type: ignore[misc]


def test_describe_topology_honors_policy_loose() -> None:
    """A chain 1e-5 away from three-parallel: default policy rejects, loose
    policy accepts."""
    kb = _make_chain(
        positions=[(0, 0, 0.1)] * 3,
        axes=[(0, 1, 0), _unit((1e-5, 1.0, 0.0)), _unit((0, 1.0, 1e-5))],
    )
    # Default 1e-8 policy: no three-parallel triple.
    assert three_consecutive_parallel(kb.joints) is None
    # Loose policy absorbs the misalignment.
    loose = TolerancePolicy(axis_parallel=1e-3, axis_intersect=1e-3)
    assert three_consecutive_parallel(kb.joints, loose) == (0, 1, 2)


def test_default_tolerance_policy_is_singleton_like() -> None:
    """DEFAULT_TOLERANCE_POLICY is a concrete value, not just a type alias."""
    assert isinstance(DEFAULT_TOLERANCE_POLICY, TolerancePolicy)
    assert DEFAULT_TOLERANCE_POLICY.axis_parallel > 0
    assert DEFAULT_TOLERANCE_POLICY.axis_intersect > 0


# ============================================================================
# axes_meet_at_common_point + is_srs_7r (#187)
# ============================================================================


def _load_fixture_kb(spec_module: str) -> KinBody:
    """Build a KinBody from a fixture module name (e.g. 'kuka_iiwa14')."""
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from ssik._kinbody import build_kinbody as _build

    mod = importlib.import_module(spec_module)
    specs_fn = getattr(mod, f"{spec_module}_specs")
    return _build(specs_fn())


def test_axes_meet_at_common_point_iiwa14_shoulder() -> None:
    """iiwa14 shoulder (joints 0, 1, 2) axes meet at z = 0.36."""
    from ssik.kinematics.predicates import axes_meet_at_common_point

    kb = _load_fixture_kb("kuka_iiwa14")
    pivot = axes_meet_at_common_point(kb.joints, (0, 1, 2))
    assert pivot is not None, "iiwa14 shoulder must concur"
    assert np.allclose(pivot, [0.0, 0.0, 0.36], atol=1e-6)


def test_axes_meet_at_common_point_iiwa14_wrist() -> None:
    """iiwa14 wrist (joints 4, 5, 6) axes meet at z = 1.18."""
    from ssik.kinematics.predicates import axes_meet_at_common_point

    kb = _load_fixture_kb("kuka_iiwa14")
    pivot = axes_meet_at_common_point(kb.joints, (4, 5, 6))
    assert pivot is not None, "iiwa14 wrist must concur"
    assert np.allclose(pivot, [0.0, 0.0, 1.18], atol=1e-6)


def test_axes_meet_at_common_point_xarm7_wrist_does_not_meet() -> None:
    """xarm7's nominal wrist (joints 4, 5, 6) does not concur at one point
    in the home pose -- structure is non-canonical.
    """
    from ssik.kinematics.predicates import axes_meet_at_common_point

    kb = _load_fixture_kb("xarm7")
    pivot = axes_meet_at_common_point(kb.joints, (4, 5, 6))
    assert pivot is None, "xarm7 wrist (4, 5, 6) does not all meet at one point"


def test_axes_meet_at_common_point_returns_none_when_parallel() -> None:
    """Two parallel axes have ill-defined intersection point."""
    from ssik.kinematics.predicates import axes_meet_at_common_point

    kb = _make_chain(
        positions=[(0, 0, 0.1), (0, 0, 0.2), (0, 0, 0.3)],
        axes=[(0, 0, 1), (0, 0, 1), (1, 0, 0)],
    )
    # Joints 0 and 1 are parallel z-axes -- the predicate must return None.
    assert axes_meet_at_common_point(kb.joints, (0, 1, 2)) is None


def test_axes_meet_at_common_point_drift_rejection() -> None:
    """When axes meet pairwise but not at a common point (drift > tol),
    the predicate returns None.
    """
    from ssik.kinematics.predicates import axes_meet_at_common_point

    # Axis 0 (z) at origin; axis 1 (y) at z=0.1 -- they meet at (0,0,0.1).
    # Axis 2 (z) at (0.5, 0, 0.1) -- the z-line at x=0.5 doesn't pass
    # through (0, 0, 0.1).
    kb = _make_chain(
        positions=[(0, 0, 0), (0, 0, 0.1), (0.5, 0, 0.1)],
        axes=[(0, 0, 1), (0, 1, 0), (0, 0, 1)],
    )
    assert axes_meet_at_common_point(kb.joints, (0, 1, 2)) is None


def test_is_srs_7r_iiwa14_classified_as_srs() -> None:
    """iiwa14 is the canonical SRS-class 7R; the predicate must accept."""
    from ssik.kinematics.predicates import is_srs_7r

    kb = _load_fixture_kb("kuka_iiwa14")
    cls = is_srs_7r(kb)
    assert cls is not None
    assert cls.shoulder_indices == (0, 1, 2)
    assert cls.elbow_index == 3
    assert cls.wrist_indices == (4, 5, 6)
    assert np.allclose(cls.shoulder_pivot, [0.0, 0.0, 0.36], atol=1e-6)
    assert np.allclose(cls.wrist_pivot, [0.0, 0.0, 1.18], atol=1e-6)


def test_is_srs_7r_franka_rejected() -> None:
    """Franka Panda is anthropomorphic, not SRS -- the predicate must reject."""
    from ssik.kinematics.predicates import is_srs_7r

    kb = _load_fixture_kb("franka_panda")
    assert is_srs_7r(kb) is None


def test_is_srs_7r_xarm7_rejected() -> None:
    """xArm7 has non-canonical wrist (axes don't all meet at one point)."""
    from ssik.kinematics.predicates import is_srs_7r

    kb = _load_fixture_kb("xarm7")
    assert is_srs_7r(kb) is None


def test_is_srs_7r_rejects_non_7r() -> None:
    """The predicate is 7R-specific. 6R and 5R chains must return None."""
    from ssik.kinematics.predicates import is_srs_7r

    kb = _load_fixture_kb("ur5")  # 6R
    assert is_srs_7r(kb) is None


def test_is_srs_7r_rejects_non_zyz_wrist() -> None:
    """Geometrically-SRS chains with non-Z*Z wrist (first/third wrist axes
    NOT parallel) must be rejected by the strict predicate -- the Singh-
    Kreutz solver's ZYZ Euler decomposition silently produces wrong
    q-vectors on these chains (#307; first surfaced on Enactic OpenArm v2.0).

    The geometric-only helper still accepts the chain so ``srs_polished``
    can use it as a warm-start path with LM polish.
    """
    from ssik.kinematics.predicates import _classify_srs_7r_geometric, is_srs_7r

    # 7R chain with strict SRS shoulder (z, y, z meeting at origin),
    # axis-concurrent but non-Z*Z wrist (z, y, x meeting at (0, 0, -1.0)).
    kb = _make_chain(
        positions=[
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, -0.5),  # elbow
            (0, 0, -0.5),  # wrist start
            (0, 0, 0),
            (0, 0, 0),
        ],
        axes=[
            (0, 0, 1),  # j0 z (shoulder)
            (0, 1, 0),  # j1 y
            (0, 0, 1),  # j2 z -- Z*Z shoulder
            (0, 1, 0),  # j3 elbow y
            (0, 0, 1),  # j4 z (wrist)
            (0, 1, 0),  # j5 y
            (1, 0, 0),  # j6 x -- breaks ZYZ (j4 z, j6 x are perpendicular)
        ],
    )

    # Strict predicate: must reject (Z*Z wrist fails: z not parallel x).
    assert is_srs_7r(kb) is None
    # Geometric helper: still accepts (axes meet at a point).
    geom = _classify_srs_7r_geometric(kb)
    assert geom is not None
    assert geom.shoulder_indices == (0, 1, 2)
    assert geom.wrist_indices == (4, 5, 6)


def test_is_srs_7r_rejects_non_zyz_shoulder() -> None:
    """Same gate, applied at the shoulder triple (#307)."""
    from ssik.kinematics.predicates import _classify_srs_7r_geometric, is_srs_7r

    # Non-Z*Z shoulder (z, y, x) but Z*Z wrist (z, y, z).
    kb = _make_chain(
        positions=[
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, -0.5),
            (0, 0, -0.5),
            (0, 0, 0),
            (0, 0, 0),
        ],
        axes=[
            (0, 0, 1),  # j0 z
            (0, 1, 0),  # j1 y
            (1, 0, 0),  # j2 x -- breaks Z*Z shoulder
            (0, 1, 0),  # elbow
            (0, 0, 1),  # j4 z
            (0, 1, 0),  # j5 y
            (0, 0, 1),  # j6 z -- Z*Z wrist
        ],
    )

    assert is_srs_7r(kb) is None
    geom = _classify_srs_7r_geometric(kb)
    assert geom is not None
