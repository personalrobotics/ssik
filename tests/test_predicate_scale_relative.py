"""Scale-relative axis-intersection tolerance (#388, U5).

An absolute ``policy.axis_intersect`` (10 nm) is scale-blind: a large arm whose
wrist axes are concurrent *by design* but drift by the URDF's coordinate rounding
(which grows with coordinate magnitude) was mis-rejected as non-spherical. The
concurrence tolerance is now relative to the cluster's characteristic length.

These tests pin the intended behaviour directly on the predicate: the *same*
geometric drift is rejected on a small arm and accepted on a large one, while a
genuinely non-concurrent wrist is rejected at either scale.
"""

from __future__ import annotations

import numpy as np

from ssik._kinbody import JointSpec, build_kinbody
from ssik.kinematics.predicates import axes_meet_at_common_point

_Z = np.array([0.0, 0.0, 1.0])
_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])


def _trans(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _wrist_arm(reach: float, drift: float):
    """A 3-joint spherical-ish wrist whose center sits ``reach`` metres from the
    base. The first two axes meet exactly at the center; the third axis line
    misses it by ``drift`` (a perpendicular offset of its origin)."""
    return build_kinbody(
        [
            # Position the wrist center at (reach, 0, 0).
            JointSpec(parent_link_T=_trans(reach, 0, 0), axis=_Z, joint_type="revolute"),
            JointSpec(parent_link_T=_trans(0, 0, 0), axis=_X, joint_type="revolute"),
            # Third axis (Z) offset by `drift` along X from the center -> its line
            # misses the center by exactly `drift` (X perpendicular to Z).
            JointSpec(parent_link_T=_trans(drift, 0, 0), axis=_Z, joint_type="revolute"),
        ]
    )


def test_same_drift_scale_relative_accept_vs_reject() -> None:
    """A wrist drift of 3e-7 m: rejected on a ~1 m arm (3e-7 > 1e-8), accepted on
    a ~100 m arm (3e-7 < 1e-8 * 100). Same geometry, tolerance scales with size."""
    drift = 3e-7
    small = _wrist_arm(reach=1.0, drift=drift)
    large = _wrist_arm(reach=100.0, drift=drift)

    assert axes_meet_at_common_point(small.joints, (0, 1, 2)) is None
    assert axes_meet_at_common_point(large.joints, (0, 1, 2)) is not None


def test_genuine_offset_rejected_at_any_scale() -> None:
    """A real 1 cm wrist offset is non-concurrent and must be rejected even on a
    huge arm -- the scale-relative tolerance doesn't wave through real offsets."""
    for reach in (1.0, 100.0):
        arm = _wrist_arm(reach=reach, drift=1e-2)
        assert axes_meet_at_common_point(arm.joints, (0, 1, 2)) is None


def test_exact_concurrence_accepted_at_any_scale() -> None:
    """Zero drift is always concurrent."""
    for reach in (1.0, 1000.0):
        arm = _wrist_arm(reach=reach, drift=0.0)
        assert axes_meet_at_common_point(arm.joints, (0, 1, 2)) is not None
