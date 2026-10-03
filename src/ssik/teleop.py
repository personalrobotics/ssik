"""Frame helpers for teleoperation: calibration, tool offset, scaling, clutch.

A teleoperation loop turns device poses into arm targets and hands them to a
:class:`~ssik.Tracker`. The device (a VR controller, a SpaceMouse, a mocap
marker, a transform gizmo) reports poses in its own frame; the arm's IK wants
the flange in the arm's base frame. Everything between the two is a handful of
rigid-transform compositions, collected here so they are written once and
tested for the invariants that matter. ssik ships no device integration: a
device is anything that yields poses, the :class:`PoseSource` protocol.

Conventions. Every pose is a 4x4 homogeneous rigid transform ``a_T_b``: the
pose of frame ``b`` expressed in frame ``a``, so it maps ``b``-coordinates to
``a``-coordinates and composes as ``a_T_c = a_T_b @ b_T_c``. Every function
checks each pose against the rigid-transform rule of ``solve()`` (default
tolerance; ``docs/api.md``, "Input validation") and raises ``TypeError`` or
``ValueError`` as it does.

A typical chain, device reading ``D`` (``world_T_device``) to IK target::

    D_base = apply_calibration(base_T_world, D)       # into the arm's base frame
    T_tcp  = clutch.target(D_base)                    # relative motion, while held
    T      = tcp_to_flange(T_tcp, flange_T_tcp)       # the pose ssik solves for
    step   = tracker.update(T, t)

``docs/api.md`` ("Teleoperation frames") is the normative statement of these
conventions.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY

__all__ = [
    "Clutch",
    "PoseSource",
    "apply_calibration",
    "calibration_from",
    "flange_to_tcp",
    "invert",
    "scale_about",
    "tcp_to_flange",
]


@runtime_checkable
class PoseSource(Protocol):
    """Anything that streams poses: the one interface a device adapter needs.

    ``poses()`` yields ``(T, t)`` pairs: ``T`` a 4x4 rigid transform (the
    device's pose in its own fixed frame) and ``t`` a timestamp in seconds that
    never decreases. The iterator ends when the stream does. ssik ships no
    adapters; ``examples/06_teleop.py`` has a scripted source and a stub showing
    where a VR or SpaceMouse reader plugs in.
    """

    def poses(self) -> Iterator[tuple[NDArray[np.float64], float]]: ...


def _pose(T: Any, name: str) -> NDArray[np.float64]:
    from ssik._solve_inputs import check_pose

    return check_pose(T, policy=DEFAULT_TOLERANCE_POLICY, name=name)


def _scale(s: Any) -> float:
    if isinstance(s, bool) or not isinstance(s, numbers.Real):
        raise TypeError(f"scale must be a real number, got {type(s).__name__}")
    v = float(s)
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"scale must be finite and > 0, got {v}")
    return v


def _inv(T: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.eye(4)
    R = T[:3, :3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def invert(T: ArrayLike) -> NDArray[np.float64]:
    """The inverse of a rigid transform: ``b_T_a`` from ``a_T_b``, exactly
    ``[R^T, -R^T p]`` (no general matrix inverse)."""
    return _inv(_pose(T, "T"))


def calibration_from(T_device: ArrayLike, T_robot: ArrayLike) -> NDArray[np.float64]:
    """The calibration that maps a device reading onto a known robot pose.

    Hold the device at a pose the arm also holds (or define the correspondence
    you want), read ``T_device`` (``world_T_device``) and ``T_robot``
    (``base_T_tcp``), and this returns ``base_T_world = T_robot @ T_device^-1``,
    so that ``apply_calibration(calibration, T_device) == T_robot``.
    """
    return _pose(T_robot, "T_robot") @ _inv(_pose(T_device, "T_device"))


def apply_calibration(calibration: ArrayLike, T_device: ArrayLike) -> NDArray[np.float64]:
    """A device reading ``world_T_device`` in the arm's base frame:
    ``calibration @ T_device`` with ``calibration = base_T_world``."""
    return _pose(calibration, "calibration") @ _pose(T_device, "T_device")


def tcp_to_flange(T_tcp: ArrayLike, flange_T_tcp: ArrayLike) -> NDArray[np.float64]:
    """The flange pose that puts the tool centre point at ``T_tcp``.

    ``T_tcp`` is ``base_T_tcp``; ``flange_T_tcp`` is the tool offset (the TCP
    in the flange frame). Returns ``base_T_flange = T_tcp @ flange_T_tcp^-1``,
    the pose to pass to ``solve()`` or ``Tracker.update()``.
    """
    return _pose(T_tcp, "T_tcp") @ _inv(_pose(flange_T_tcp, "flange_T_tcp"))


def flange_to_tcp(T_flange: ArrayLike, flange_T_tcp: ArrayLike) -> NDArray[np.float64]:
    """Where the tool centre point is when the flange is at ``T_flange``:
    ``T_flange @ flange_T_tcp`` (``FK(q)`` in, TCP pose out). The inverse of
    :func:`tcp_to_flange`."""
    return _pose(T_flange, "T_flange") @ _pose(flange_T_tcp, "flange_T_tcp")


def scale_about(T: ArrayLike, scale: float, anchor: ArrayLike) -> NDArray[np.float64]:
    """Workspace scaling: ``T`` with its position scaled by ``scale`` about
    ``anchor``, its rotation unchanged.

    The position becomes ``a + scale * (p - a)``, with ``a`` the anchor point:
    a ``(3,)`` point, or a 4x4 pose whose position is used. Both are in the
    frame ``T`` is expressed in. ``scale < 1`` makes a large hand motion a small
    arm motion. Rotation is never scaled: a scaled rotation is not a rotation.
    """
    P = _pose(T, "T")
    s = _scale(scale)
    from ssik._solve_inputs import as_real_array

    a = as_real_array(anchor, "anchor")
    if a.shape == (4, 4):
        a = _pose(a, "anchor")[:3, 3]
    elif a.shape != (3,):
        raise ValueError(f"anchor must be a (3,) point or a (4, 4) pose, got {a.shape}")
    elif not all(map(math.isfinite, a.tolist())):
        raise ValueError("anchor must be finite (no NaN or inf)")
    out = P.copy()
    out[:3, 3] = a + s * (P[:3, 3] - a)
    return out


class Clutch:
    """Relative ("clutched") teleoperation: the arm follows the device's motion
    since the clutch engaged, not the device's absolute pose.

    ::

        clutch = Clutch(scale=0.5)
        clutch.engage(T_device, T_robot)            # grip pressed
        T = clutch.target(T_device)                 # each frame while held
        clutch.release()                            # grip released

    On :meth:`engage` the clutch stores the device pose ``A_d = [R_ad, p_ad]``
    and the arm pose ``A_r = [R_ar, p_ar]``, both in one frame: the arm's base
    frame, after :func:`apply_calibration`. While engaged, for a device pose
    ``D = [R_d, p_d]``, ``frame`` picks how the device's motion since engaging
    is applied to the arm:

    ``frame="world"`` (default)::

        p = p_ar + scale * (p_d - p_ad)
        R = (R_d @ R_ad^T) @ R_ar

    The device's translation and rotation since engaging, both expressed in
    the base frame, are applied to the arm in the base frame, the rotation
    about the arm's tool point. Moving the hand 10 cm along the base x axis
    moves the tool ``scale * 10`` cm along the base x axis, whatever either is
    pointing at. This is what an operator looking at the arm expects.

    ``frame="tool"``::

        target(D) = A_r @ S(A_d^-1 @ D)

    ``A_d^-1 @ D`` is the device's displacement in the device's own frame at
    engagement; ``S`` multiplies its translation by ``scale``; the arm makes
    that displacement in its tool frame at the anchor. With ``scale = 1`` this
    is the fixed rigid transform ``(A_r @ A_d^-1) @ D`` of the device's pose:
    the device's axes act as the tool's axes, which suits jogging along the
    tool's own axes (a SpaceMouse held like the tool).

    In both modes the target does not jump at engagement
    (``target(A_d) == A_r``), rotation is never scaled, and releasing and
    engaging again re-anchors both frames, so the operator can reposition the
    device without moving the arm (indexing). The two modes agree when the
    device and arm anchors have the same orientation.
    """

    __slots__ = ("_anchor_device", "_anchor_robot", "_frame", "_scale")

    def __init__(self, *, scale: float = 1.0, frame: Literal["world", "tool"] = "world") -> None:
        if frame not in ("world", "tool"):
            raise ValueError(f"frame must be 'world' or 'tool', got {frame!r}")
        self._scale = _scale(scale)
        self._frame = frame
        self._anchor_device: NDArray[np.float64] | None = None
        self._anchor_robot: NDArray[np.float64] | None = None

    @property
    def scale(self) -> float:
        """The translation scale applied to the device's displacement."""
        return self._scale

    @property
    def frame(self) -> Literal["world", "tool"]:
        """``"world"`` or ``"tool"``: where the device's motion is applied."""
        return self._frame

    @property
    def engaged(self) -> bool:
        """Whether the clutch is engaged."""
        return self._anchor_robot is not None

    def engage(self, T_device: ArrayLike, T_robot: ArrayLike) -> None:
        """Anchor: the device is at ``T_device`` and the arm at ``T_robot``
        now. Engaging while engaged re-anchors."""
        A_d = _pose(T_device, "T_device")
        A_r = _pose(T_robot, "T_robot")
        self._anchor_device = A_d.copy()
        self._anchor_robot = A_r.copy()

    def release(self) -> None:
        """Disengage. :meth:`target` returns ``None`` until the next engage."""
        self._anchor_device = None
        self._anchor_robot = None

    def target(self, T_device: ArrayLike) -> NDArray[np.float64] | None:
        """The arm target for device pose ``T_device``, or ``None`` while
        released (the arm should hold)."""
        D = _pose(T_device, "T_device")
        A_d, A_r = self._anchor_device, self._anchor_robot
        if A_r is None or A_d is None:
            return None
        if self._frame == "tool":
            rel = _inv(A_d) @ D
            rel[:3, 3] *= self._scale
            out: NDArray[np.float64] = A_r @ rel
            return out
        out = np.eye(4)
        out[:3, :3] = (D[:3, :3] @ A_d[:3, :3].T) @ A_r[:3, :3]
        out[:3, 3] = A_r[:3, 3] + self._scale * (D[:3, 3] - A_d[:3, 3])
        return out
