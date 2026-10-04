"""Teleoperation in a few lines: plug your device in here.

Any device that yields 4x4 poses can drive an arm: a VR controller, a
SpaceMouse, a mocap marker. ssik supplies the parts between the device and the
joints (frame helpers, a clutch, and a streaming IK tracker that never switches
branch silently) and no device code. Write a class with a ``poses()`` method
yielding ``(T, t)`` and pass it in.

The arm is mounted in a room the way a torso-mounted arm is: the room has a
world frame, the arm's base sits at a tilted shoulder, and the tracking system
has an origin of its own. The calibration from the tracking frame to the base
is measured with ``calibration_from``, and it equals the composition of the two
known frames. ``docs/teleop.md`` walks through each step.

This runs headless with a scripted source: a hand that draws a circle, lets go
of the grip to reposition, reaches far out of the workspace and comes back.

Run::

    python examples/06_teleop.py
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

import ssik
from ssik.teleop import (
    Clutch,
    PoseSource,
    apply_calibration,
    calibration_from,
    flange_to_tcp,
    invert,
    tcp_to_flange,
)


def pose(axis: int, angle: float, xyz: list[float]) -> np.ndarray:
    """A rigid transform: a rotation by ``angle`` about axis 0, 1 or 2, then ``xyz``."""
    c, s = np.cos(angle), np.sin(angle)
    i, j = [k for k in range(3) if k != axis]
    T = np.eye(4)
    T[i, i], T[i, j], T[j, i], T[j, j] = c, -s, s, c
    T[:3, 3] = xyz
    return T


class ScriptedHand:
    """A deterministic stand-in for a device: 4 s at 50 Hz, in the tracking frame."""

    def poses(self) -> Iterator[tuple[np.ndarray, float]]:
        for k in range(200):
            t = k / 50.0
            T = np.eye(4)
            T[:3, 3] = [0.2 * np.cos(np.pi * t), 0.2 * np.sin(np.pi * t), 0.0]
            if 2.5 <= t < 3.5:  # reach out 3 m (1.5 m for the arm) and back
                T[0, 3] += 3.0 * np.sin(np.pi * (t - 2.5))
            yield T, t

    @staticmethod
    def grip(t: float) -> bool:
        return not 1.0 <= t < 1.4  # released for 0.4 s to reposition


class MyDevice:
    """Where a real device plugs in: an OpenXR controller, a SpaceMouse, ..."""

    def read(self) -> tuple[np.ndarray, float]:
        """Block for the next sample: the 4x4 pose in the tracking frame, seconds."""
        raise NotImplementedError("read your device here")

    def poses(self) -> Iterator[tuple[np.ndarray, float]]:
        while True:
            yield self.read()


arm = ssik.Manipulator.from_prebuilt("ur5e")
tracker = arm.tracker([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], max_joint_speed=2.0)
tool = np.eye(4)
tool[2, 3] = 0.10  # TCP 10 cm beyond the flange: flange_T_tcp

# The arm's mounting (world_T_base): a shoulder 1.1 m up, tilted 30 degrees.
world_T_base = pose(0, np.pi / 6, [0.0, 0.25, 1.1])
# The device setup (world_T_tracking): the tracker's origin on the floor 2 m
# away, turned to face the arm.
world_T_tracking = pose(2, np.pi / 2, [2.0, 0.0, 0.0])

# Calibrate: with the arm at its start pose, hold the device at the TCP with
# its axes on the tool's axes, read it, and pair the reading with the TCP pose.
# Here the reading is simulated from the frames above.
T_tcp = flange_to_tcp(arm.fk(tracker.q), tool)  # base_T_tcp
reading = invert(world_T_tracking) @ world_T_base @ T_tcp  # tracking_T_device
calibration = calibration_from(reading, T_tcp)  # base_T_tracking
assert np.allclose(calibration, invert(world_T_base) @ world_T_tracking, atol=1e-12)
print("calibration: base_T_tracking = invert(world_T_base) @ world_T_tracking")

# The hand's motion in the room is the tool's motion in the room, halved: 1 m of
# hand is 0.5 m of tool, whatever the mount's tilt.
# Clutch(frame="tool") instead moves the tool along its own axes, for jogging.
clutch = Clutch(scale=0.5)

source: PoseSource = ScriptedHand()
last = None
for T_device, t in source.poses():
    D = apply_calibration(calibration, T_device)
    if ScriptedHand.grip(t) and not clutch.engaged:
        clutch.engage(D, flange_to_tcp(arm.fk(tracker.q), tool))  # anchor here
    elif not ScriptedHand.grip(t):
        clutch.release()  # the arm holds while the hand repositions
    T_tcp = clutch.target(D)
    if T_tcp is None:
        continue
    step = tracker.update(tcp_to_flange(T_tcp, tool), t)
    # robot.command(step.q) would go here.
    if (step.status, step.reason) != last:
        last = (step.status, step.reason)
        reason = f" ({step.reason})" if step.reason else ""
        print(f"t={t:4.2f}s  {step.status.name}{reason}  fk_residual={step.fk_residual:.1e}")
print("final q:", np.round(tracker.q, 3))
