"""Franka Emika Panda kinematics fixture (no hand).

7-DOF arm. Per EAIK (``find_locked_joint_index`` in
``mj_manipulator/src/mj_manipulator/arms/franka.py:60-64``), locking joint
index 4 (the forearm-adjacent joint) yields a ``SPHERICAL_SECOND_TWO_PARALLEL``
6R sub-chain in EAIK's taxonomy.

ssik dispatches the same structure via chain reversal (#121 Level 1):
the post-lock-4 sub-chain has a spherical wrist at the BASE; reversing
the chain lands the wrist at sub-chain ``(3, 4, 5)`` where the standard
:mod:`ssik.solvers.ikgeo.spherical_two_parallel` solver matches it
directly. Solver name reported by ``_topology_rank`` is
``"reversed:spherical_two_parallel"``.

Source of truth: ``mujoco_menagerie/franka_emika_panda/panda_nohand.xml``
(MuJoCo MJCF, official upstream from Franka Emika). We transcribe the
seven arm-joint frames here so the fixture is self-contained (no MuJoCo
import on the test path).

Chain (parent_pos, parent_quat (w, x, y, z), joint axis = local +Z):

  base   -> link_1 : pos=(0, 0, 0.333),    quat=(1, 0, 0, 0)
  link_1 -> link_2 : pos=(0, 0, 0),        quat=(1, -1, 0, 0)
  link_2 -> link_3 : pos=(0, -0.316, 0),   quat=(1, 1, 0, 0)
  link_3 -> link_4 : pos=(0.0825, 0, 0),   quat=(1, 1, 0, 0)
  link_4 -> link_5 : pos=(-0.0825, 0.384, 0), quat=(1, -1, 0, 0)
  link_5 -> link_6 : pos=(0, 0, 0),        quat=(1, 1, 0, 0)
  link_6 -> link_7 : pos=(0.088, 0, 0),    quat=(1, 1, 0, 0)

End-effector ``attachment_site`` (inside link_7):
  pos=(0, 0, 0.107), quat=(0.3826834, 0, 0, 0.9238795)

All joints revolute about local +Z (MJCF default axis).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import JointSpec

__all__ = ["FRANKA_PANDA_KEYFRAMES", "franka_panda_specs"]


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
_FRANKA_JOINT_FRAMES: tuple[
    tuple[tuple[float, float, float], tuple[float, float, float, float]], ...
] = (
    ((0.0, 0.0, 0.333), (1.0, 0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.0, -1.0, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (1.0, 1.0, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (1.0, -1.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)),
)

# Per-joint reachable range from the MJCF ``range="lo hi"`` attributes
# (joints with no explicit range inherit the panda-class default of
# ``-2.8973 .. 2.8973``).
_FRANKA_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.8973, 2.8973),  # joint1 (default class)
    (-1.7628, 1.7628),  # joint2 (override)
    (-2.8973, 2.8973),  # joint3 (default class)
    (-3.0718, -0.0698),  # joint4 (override)
    (-2.8973, 2.8973),  # joint5 (default class)
    (-0.0175, 3.7525),  # joint6 (override)
    (-2.8973, 2.8973),  # joint7 (default class)
)

# End-effector attachment-site offset inside link_7 (post-joint-7).
_FRANKA_EE_FRAME: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
    (0.0, 0.0, 0.107),
    (0.3826834, 0.0, 0.0, 0.9238795),
)


def franka_panda_specs() -> list[JointSpec]:
    """7 revolute :class:`JointSpec`s for the Franka Panda arm chain.

    Joint 7's ``child_link_T`` carries the EE attachment-site offset so
    forward kinematics produces the EE pose used by IK targets. Each
    joint carries its MJCF-supplied reachable range in ``limits``;
    ``ssik.solvers.jointlock.seven_r.solve`` clamps the default
    ``lock_samples`` sweep to the locked joint's range so we don't waste
    samples outside reachable territory.
    """
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    specs: list[JointSpec] = []
    for i, (pos, quat) in enumerate(_FRANKA_JOINT_FRAMES):
        is_last = i == len(_FRANKA_JOINT_FRAMES) - 1
        ee_pos, ee_quat = _FRANKA_EE_FRAME
        child = _xform(ee_pos, ee_quat) if is_last else np.eye(4, dtype=np.float64)
        specs.append(
            JointSpec(
                parent_link_T=_xform(pos, quat),
                axis=z_axis,
                joint_type="revolute",
                child_link_T=child,
                name=f"panda_joint{i + 1}",
                limits=_FRANKA_JOINT_LIMITS[i],
            )
        )
    return specs


# MJCF "home" keyframe (the one shipped in panda_nohand.xml).
FRANKA_PANDA_KEYFRAMES: dict[str, NDArray[np.float64]] = {
    "home": np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64),
}
