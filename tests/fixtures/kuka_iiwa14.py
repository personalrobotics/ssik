"""KUKA iiwa LBR (R820 / 14) kinematics fixture.

7-DOF arm with **SRS** (Spherical-Revolute-Spherical) topology:
joints 1, 2, 3 form a spherical shoulder; joint 4 is the elbow;
joints 5, 6, 7 form a spherical wrist. Today this fixture exercises
``ssik.solvers.jointlock.seven_r`` (the universal lock-sweep
fallback) -- with the ``max_solutions=1`` short-circuit (#145) it
matches Franka 7R speed within a small constant.

A future SRS-specific analytical solver (#143, Singh-Kreutz 1989) will
bypass the lock-sweep entirely and produce iiwa IK in ~100 us. This
fixture is the validation target for that solver.

Source of truth: ``mujoco_menagerie/kuka_iiwa_14/iiwa14.xml``
(MuJoCo MJCF, official upstream from KUKA).

Chain (parent_pos, parent_quat (w, x, y, z), joint axis = local +Z):

  base   -> link_1 : pos=(0, 0, 0.1575),  quat=(1, 0, 0, 0)
  link_1 -> link_2 : pos=(0, 0, 0.2025),  quat=(0, 0, 1, 1)
  link_2 -> link_3 : pos=(0, 0.2045, 0),  quat=(0, 0, 1, 1)
  link_3 -> link_4 : pos=(0, 0, 0.2155),  quat=(1, 1, 0, 0)
  link_4 -> link_5 : pos=(0, 0.1845, 0),  quat=(0, 0, 1, 1)
  link_5 -> link_6 : pos=(0, 0, 0.2155),  quat=(1, 1, 0, 0)
  link_6 -> link_7 : pos=(0, 0.081, 0),   quat=(0, 0, 1, 1)

End-effector ``attachment_site`` (inside link_7):
  pos=(0, 0, 0.045), quat=(1, 0, 0, 0)

All joints revolute about local +Z (MJCF default ``axis="0 0 1"``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import JointSpec

__all__ = ["KUKA_IIWA14_KEYFRAMES", "kuka_iiwa14_specs"]


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
_IIWA14_JOINT_FRAMES: tuple[
    tuple[tuple[float, float, float], tuple[float, float, float, float]], ...
] = (
    ((0.0, 0.0, 0.1575), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.2025), (0.0, 0.0, 1.0, 1.0)),
    ((0.0, 0.2045, 0.0), (0.0, 0.0, 1.0, 1.0)),
    ((0.0, 0.0, 0.2155), (1.0, 1.0, 0.0, 0.0)),
    ((0.0, 0.1845, 0.0), (0.0, 0.0, 1.0, 1.0)),
    ((0.0, 0.0, 0.2155), (1.0, 1.0, 0.0, 0.0)),
    ((0.0, 0.081, 0.0), (0.0, 0.0, 1.0, 1.0)),
)

# Per-joint reachable range from the MJCF ``range="lo hi"`` attributes.
# joints 1, 3, 5 use class="joint1" (range -2.96706 .. 2.96706);
# joints 2, 4, 6 use class="joint2" (range -2.0944 .. 2.0944);
# joint 7 uses class="joint3" (range -3.05433 .. 3.05433).
_IIWA14_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.96706, 2.96706),  # joint1
    (-2.0944, 2.0944),  # joint2
    (-2.96706, 2.96706),  # joint3
    (-2.0944, 2.0944),  # joint4
    (-2.96706, 2.96706),  # joint5
    (-2.0944, 2.0944),  # joint6
    (-3.05433, 3.05433),  # joint7
)

# End-effector attachment-site offset inside link_7 (post-joint-7).
_IIWA14_EE_FRAME: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
    (0.0, 0.0, 0.045),
    (1.0, 0.0, 0.0, 0.0),
)


def kuka_iiwa14_specs() -> list[JointSpec]:
    """7 revolute :class:`JointSpec`s for the KUKA iiwa LBR 14 arm chain.

    Joint 7's ``child_link_T`` carries the EE attachment-site offset so
    forward kinematics produces the EE pose used by IK targets. Each
    joint carries its MJCF-supplied reachable range in ``limits``;
    ``ssik.solvers.jointlock.seven_r.solve`` clamps the default
    ``lock_samples`` sweep to the locked joint's range.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    specs: list[JointSpec] = []
    for i, (pos, quat) in enumerate(_IIWA14_JOINT_FRAMES):
        is_last = i == len(_IIWA14_JOINT_FRAMES) - 1
        ee_pos, ee_quat = _IIWA14_EE_FRAME
        child = _xform(ee_pos, ee_quat) if is_last else np.eye(4, dtype=np.float64)
        specs.append(
            JointSpec(
                parent_link_T=_xform(pos, quat),
                axis=z_axis,
                joint_type="revolute",
                child_link_T=child,
                name=f"iiwa_joint{i + 1}",
                limits=_IIWA14_JOINT_LIMITS[i],
            )
        )
    return specs


# MJCF "home" keyframe for the iiwa14 (zero configuration -- no keyframe is
# defined in iiwa14.xml, but ``q = 0`` is the canonical home pose for SRS
# arms with all joints aligned to the world +Z column).
KUKA_IIWA14_KEYFRAMES: dict[str, NDArray[np.float64]] = {
    "home": np.zeros(7, dtype=np.float64),
}
