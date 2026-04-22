"""Kinematic-structure utilities: predicates, POE helpers.

Everything here operates on POE-normalized kinematic chains -- axes in the
base frame at ``q = 0``, ``T_left`` as pure translation, ``T_right`` as
identity except on the final joint. Passing a non-normalized KinBody to these
predicates yields undefined results; use :func:`ssik._urdf.load_urdf_kinbody_normalized`
first.
"""

from ssik.kinematics.predicates import (
    axis_intersect,
    axis_parallel,
    joint_origins,
    three_consecutive_intersecting,
    three_consecutive_parallel,
)

__all__ = [
    "axis_intersect",
    "axis_parallel",
    "joint_origins",
    "three_consecutive_intersecting",
    "three_consecutive_parallel",
]
