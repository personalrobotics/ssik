"""Kinova JACO 2 (j2n6s200) kinematics fixture.

6-DOF non-Pieper arm with 60-degree non-orthogonal twists at joints 4-5;
the canonical regression case for the general-6R Raghavan-Roth solver.

Source of truth: ``robot-code/ada_assets/src/ada_assets/models/jaco2.xml``
(MuJoCo MJCF, converted from the upstream Kinova URDF). We transcribe the
six arm-joint frames here so the fixture is self-contained (no MuJoCo
import in the test path).

Chain (parent_pos, parent_quat (w, x, y, z), joint axis in local frame):

  base   -> link_1 : pos=(0, 0, 0.15675),    quat=(0, 0, 1, 0)
  link_1 -> link_2 : pos=(0, 0.0016, -0.11875), quat=(0, 0, -.707107, .707107)
  link_2 -> link_3 : pos=(0, -0.41, 0),      quat=(0, 0, 1, 0)
  link_3 -> link_4 : pos=(0, 0.2073, -0.0114), quat=(0, 0, -.707107, .707107)
  link_4 -> link_5 : pos=(0, -.03703, -.06414), quat=(0, 0, 0.5, 0.866025)
  link_5 -> link_6 : pos=(0, -.03703, -.06414), quat=(0, 0, 0.5, 0.866025)

End-effector ``ee_site`` (inside link_6):
  pos=(0, 0, -0.16), quat=(0, .707107, .707107, 0)

All joints are revolute about local +Z. The 60-degree twist between
joints 4-5 and 5-6 (via the ``quat=(0, 0, 0.5, 0.866025) =
sin(30deg) Z + cos(30deg) W`` rotation) is what makes JACO 2 a non-Pieper
arm: no three consecutive intersecting axes, no parallel pair.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import JointSpec

__all__ = ["JACO2_KEYFRAMES", "jaco2_specs"]


def _quat_wxyz_to_rot(q: tuple[float, float, float, float]) -> NDArray[np.float64]:
    """Convert MuJoCo (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    n = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array(
        [
            [1 - 2*(y*y + z*z),  2*(x*y - z*w),    2*(x*z + y*w)],
            [2*(x*y + z*w),      1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),      2*(y*z + x*w),    1 - 2*(x*x + y*y)],
        ],
        dtype=np.float64,
    )


def _xform(pos: tuple[float, float, float], quat: tuple[float, float, float, float]) -> NDArray[np.float64]:  # noqa: E501
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rot(quat)
    T[:3, 3] = pos
    return T


# Per-joint (pos, quat_wxyz) from the MJCF, in chain order joint_1..joint_6.
_JACO2_JOINT_FRAMES: tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...] = (  # noqa: E501
    ((0.0, 0.0, 0.15675),         (0.0, 0.0, 1.0, 0.0)),
    ((0.0, 0.0016, -0.11875),     (0.0, 0.0, -0.707107, 0.707107)),
    ((0.0, -0.41, 0.0),           (0.0, 0.0, 1.0, 0.0)),
    ((0.0, 0.2073, -0.0114),      (0.0, 0.0, -0.707107, 0.707107)),
    ((0.0, -0.03703, -0.06414),   (0.0, 0.0, 0.5, 0.866025)),
    ((0.0, -0.03703, -0.06414),   (0.0, 0.0, 0.5, 0.866025)),
)

# End-effector site offset inside link_6 (post-joint-6).
_JACO2_EE_FRAME: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
    (0.0, 0.0, -0.16), (0.0, 0.707107, 0.707107, 0.0),
)


def jaco2_specs() -> list[JointSpec]:
    """6 revolute :class:`JointSpec`s for the JACO 2 arm chain.

    Joint 6's ``child_link_T`` carries the EE-site offset so the chain's
    forward kinematics produces the EE pose used by IK targets.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    specs: list[JointSpec] = []
    for i, (pos, quat) in enumerate(_JACO2_JOINT_FRAMES):
        is_last = (i == len(_JACO2_JOINT_FRAMES) - 1)
        ee_pos, ee_quat = _JACO2_EE_FRAME
        child = _xform(ee_pos, ee_quat) if is_last else np.eye(4, dtype=np.float64)
        specs.append(
            JointSpec(
                parent_link_T=_xform(pos, quat),
                axis=z_axis,
                joint_type="revolute",
                child_link_T=child,
                name=f"j2n6s200_joint_{i + 1}",
            )
        )
    return specs


# MJCF keyframes (first 6 entries are arm joints; fingers omitted).
JACO2_KEYFRAMES: dict[str, NDArray[np.float64]] = {
    "above_plate": np.array([-2.579, 3.010, 1.770, -2.076, -1.791, 2.858], dtype=np.float64),
    "resting":     np.array([-1.860, 2.181, 0.364, -5.187, -0.470, -0.814], dtype=np.float64),
    "staging":     np.array([-2.122, 4.496, 4.022, -4.710, -2.493, -1.926], dtype=np.float64),
    "stow":        np.array([-1.521, 2.601, 0.348, -4.000, 0.228, 3.879], dtype=np.float64),
}
