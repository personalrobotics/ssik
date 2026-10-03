"""Teleoperation in a few lines: plug your device in here.

Any device that yields 4x4 poses can drive an arm: a VR controller, a
SpaceMouse, a mocap marker. ssik supplies the parts between the device and the
joints (frame helpers, a clutch, and a streaming IK tracker that never switches
branch silently) and no device code. Write a class with a ``poses()`` method
yielding ``(T, t)`` and pass it in.

This runs headless with a scripted source: a hand that draws a circle, lets go
of the grip to reposition, reaches far out of the workspace and comes back.

Run::

    python examples/07_teleop.py
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

import ssik
from ssik.teleop import Clutch, PoseSource, apply_calibration, flange_to_tcp, tcp_to_flange


class ScriptedHand:
    """A deterministic stand-in for a device: 4 s at 50 Hz, in the device's frame."""

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
calibration = np.eye(4)  # base_T_world: the tracking frame in the arm's base frame
clutch = Clutch(scale=0.5)  # 1 m of hand motion is 0.5 m of arm motion

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
