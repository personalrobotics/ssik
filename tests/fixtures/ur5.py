"""UR5 kinematics fixture.

Classical (distal) DH parameters published by Universal Robots:
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/

The joint transform is decomposed as
    A_i(theta) = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
so for the kinbody shim we set ``parent_link_T = I`` (the joint axis is aligned
with the parent-link +Z) and lump the post-rotation DH factors into
``child_link_T``. All six joints are revolute.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ikfastpy._kinbody import JointSpec

# (a, alpha, d) per joint, theta=0. Units: meters, radians.
_UR5_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, math.pi / 2, 0.089159),
    (-0.425, 0.0, 0.0),
    (-0.39225, 0.0, 0.0),
    (0.0, math.pi / 2, 0.10915),
    (0.0, -math.pi / 2, 0.09465),
    (0.0, 0.0, 0.0823),
)


def _rot_x(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def _trans(x: float, y: float, z: float) -> NDArray[np.float64]:
    T = np.eye(4, dtype=np.float64)
    T[0, 3], T[1, 3], T[2, 3] = x, y, z
    return T


def _post_rotation_dh(a: float, alpha: float, d: float) -> NDArray[np.float64]:
    """``Trans_z(d) * Trans_x(a) * Rot_x(alpha)``."""
    return _trans(0, 0, d) @ _trans(a, 0, 0) @ _rot_x(alpha)


def ur5_specs() -> list[JointSpec]:
    """Return a list of six revolute ``JointSpec``s for the UR5."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return [
        JointSpec(
            parent_link_T=np.eye(4, dtype=np.float64),
            axis=z_axis,
            joint_type="revolute",
            child_link_T=_post_rotation_dh(a, alpha, d),
            name=f"ur5_joint_{i + 1}",
        )
        for i, (a, alpha, d) in enumerate(_UR5_DH)
    ]


def _rot_z(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def ur5_fk(q: Sequence[float]) -> NDArray[np.float64]:
    """Ground-truth forward kinematics for the UR5 fixture.

    Applies the per-joint transform ``parent_link_T @ R(q_i) @ child_link_T``
    for each of the six revolute joints, in order — matching exactly what
    ``build_kinbody(ur5_specs())`` hands to IKFastSolver. Returns the 4x4
    base-to-EE transform.
    """
    if len(q) != 6:
        raise ValueError(f"UR5 has 6 joints; got q of length {len(q)}")
    T = np.eye(4, dtype=np.float64)
    for spec, qi in zip(ur5_specs(), q, strict=True):
        child = spec.child_link_T if spec.child_link_T is not None else np.eye(4)
        T = T @ spec.parent_link_T @ _rot_z(qi) @ child
    return T
