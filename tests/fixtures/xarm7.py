"""UFactory xArm 7 kinematics fixture.

7-DOF arm. Per the dispatch profile in ``scripts/profile_7r_arms.py``,
locking the auto-selected joint and reversing the resulting 6R sub-chain
yields 15/16 samples landing on
:mod:`ssik.solvers.ikgeo.spherical` (``reversed:spherical``) and 1/16 on
the closely-related ``reversed:spherical_two_parallel``. xArm7 is a
**verified Pieper-class wedge**: 7R IK runs at ~3 ms with
``max_solutions=1`` and ~40 ms exhaustive (the lock sweep dispatches to
fast tier-0 inner solvers, no gen_six_dof fallthrough).

Source of truth: ``mujoco_menagerie/ufactory_xarm7/xarm7.xml`` (MuJoCo
MJCF, official upstream from UFactory). We transcribe the seven
arm-joint frames here so the fixture is self-contained (no MuJoCo
import on the test path).

Chain (parent_pos, parent_quat (w, x, y, z), joint axis = local +Z):

  base   -> link_1 : pos=(0, 0, 0.267),         quat=(1, 0, 0, 0)
  link_1 -> link_2 : pos=(0, 0, 0),             quat=(1, -1, 0, 0)
  link_2 -> link_3 : pos=(0, -0.293, 0),        quat=(1, 1, 0, 0)
  link_3 -> link_4 : pos=(0.0525, 0, 0),        quat=(1, 1, 0, 0)
  link_4 -> link_5 : pos=(0.0775, -0.3425, 0),  quat=(1, 1, 0, 0)
  link_5 -> link_6 : pos=(0, 0, 0),             quat=(1, 1, 0, 0)
  link_6 -> link_7 : pos=(0.076, 0.097, 0),     quat=(1, -1, 0, 0)

The xArm7 MJCF defines no ``attachment_site`` inside link_7 (the tool
flange coincides with the post-joint-7 frame), so joint 7's
``child_link_T`` is the identity. All joints revolute about local +Z
(MJCF default ``axis="0 0 1"``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import JointSpec

__all__ = ["XARM7_KEYFRAMES", "xarm7_specs"]


def _quat_wxyz_to_rot(q: tuple[float, float, float, float]) -> NDArray[np.float64]:
    """Convert MuJoCo (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _xform(
    pos: tuple[float, float, float], quat: tuple[float, float, float, float]
) -> NDArray[np.float64]:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rot(quat)
    T[:3, 3] = pos
    return T


# Per-joint (pos, quat_wxyz) from the MJCF, in chain order joint_1..joint_7.
_XARM7_JOINT_FRAMES: tuple[
    tuple[tuple[float, float, float], tuple[float, float, float, float]], ...
] = (
    ((0.0, 0.0, 0.267), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.0, -1.0, 0.0, 0.0)),
    ((0.0, -0.293, 0.0), (1.0, 1.0, 0.0, 0.0)),
    ((0.0525, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)),
    ((0.0775, -0.3425, 0.0), (1.0, 1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)),
    ((0.076, 0.097, 0.0), (1.0, -1.0, 0.0, 0.0)),
)

# Per-joint reachable range from the MJCF ``range="lo hi"`` attributes.
_XARM7_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-6.2832, 6.2832),  # joint1
    (-2.059, 2.0944),  # joint2
    (-6.2832, 6.2832),  # joint3
    (-0.19198, 3.927),  # joint4
    (-6.2832, 6.2832),  # joint5
    (-1.69297, 3.14159),  # joint6
    (-6.2832, 6.2832),  # joint7
)


def xarm7_specs() -> list[JointSpec]:
    """7 revolute :class:`JointSpec`s for the xArm7 arm chain.

    The MJCF defines no end-effector attachment site inside link_7
    (the tool flange is the post-joint-7 frame), so joint 7's
    ``child_link_T`` is the identity. Each joint carries its
    MJCF-supplied reachable range in ``limits``;
    :func:`ssik.solvers.jointlock.seven_r.solve` clamps the default
    ``lock_samples`` sweep to the locked joint's range.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    specs: list[JointSpec] = []
    for i, (pos, quat) in enumerate(_XARM7_JOINT_FRAMES):
        specs.append(
            JointSpec(
                parent_link_T=_xform(pos, quat),
                axis=z_axis,
                joint_type="revolute",
                child_link_T=np.eye(4, dtype=np.float64),
                name=f"xarm7_joint{i + 1}",
                limits=_XARM7_JOINT_LIMITS[i],
            )
        )
    return specs


# MJCF "home" keyframe (the one shipped in xarm7.xml: all joints at zero).
XARM7_KEYFRAMES: dict[str, NDArray[np.float64]] = {
    "home": np.zeros(7, dtype=np.float64),
}
