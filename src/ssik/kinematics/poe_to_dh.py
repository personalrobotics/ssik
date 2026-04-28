"""POE -> DH conversion (Spong distal-DH form). Closes #79.

Convert a POE-normalized ``KinBody`` chain to standard DH parameters such that

    FK_DH(theta) = T_pre @ A_1(theta_1) ... A_n(theta_n) @ T_post

matches POE FK at theta = q + theta_offset, where

    A_i(theta_i) = R_z(theta_i) T_z(d_i) T_x(a_i) R_x(alpha_i)    (Spong distal)

Frame placement (Spong "Robot Modeling and Control" \u00a73.2):

    z_i = axis of joint (i+1) in the world frame at q=0      (i = 0..n-1)
    z_n = z-axis of the tool flange                          (free choice)

Frame i origin is at the foot of the common perpendicular between z_{i-1}
and z_i, on the z_i line. Frame i x-axis points along the common
perpendicular from z_{i-1} toward z_i. For i=0 (base frame) we pick world
x; for i=n (tool flange) we project the previous x onto the plane
perpendicular to z_n.

For a joint chain whose first joint axis does NOT align with world z, the
``T_pre`` factor absorbs the discrepancy. Likewise ``T_post`` absorbs the
frame-n to actual-tool-flange discrepancy.

For arms where joint 0's axis IS world z and the tool flange aligns
(commercial arms with conventional URDF base orientation), T_pre and T_post
collapse to identity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody

__all__ = ["DhWithOffset", "poe_to_dh"]


class DhWithOffset:
    """Standard distal-DH parameters plus theta-offset and pre/post transforms.

    Convention:

        FK_DH(q + theta_offset) = T_pre @ A_1 @ ... @ A_n @ T_post

    matches FK_POE(q). Solver wrappers handle the pre/post transforms by
    applying their inverses to T_target before feeding into the DH-internal
    solver, and reversing the theta_offset on returned q values.
    """

    __slots__ = ("a", "alpha", "d", "t_post", "t_pre", "theta_offset")

    def __init__(
        self,
        alpha: NDArray[np.float64],
        a: NDArray[np.float64],
        d: NDArray[np.float64],
        theta_offset: NDArray[np.float64],
        t_pre: NDArray[np.float64],
        t_post: NDArray[np.float64],
    ) -> None:
        self.alpha = np.asarray(alpha, dtype=np.float64)
        self.a = np.asarray(a, dtype=np.float64)
        self.d = np.asarray(d, dtype=np.float64)
        self.theta_offset = np.asarray(theta_offset, dtype=np.float64)
        self.t_pre = np.asarray(t_pre, dtype=np.float64)
        self.t_post = np.asarray(t_post, dtype=np.float64)
        if not (self.alpha.shape == self.a.shape == self.d.shape == self.theta_offset.shape):
            raise ValueError(
                f"DhWithOffset: shape mismatch alpha={self.alpha.shape}, "
                f"a={self.a.shape}, d={self.d.shape}, theta_offset={self.theta_offset.shape}"
            )
        if self.t_pre.shape != (4, 4) or self.t_post.shape != (4, 4):
            raise ValueError("t_pre, t_post must be 4x4")

    def to_dh_tuple(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        return (self.alpha.copy(), self.a.copy(), self.d.copy())


def _rot_axis(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    axis = axis / np.linalg.norm(axis)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x, y, z = axis
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )


def _kinbody_world_axes_origins(
    kb: KinBody,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Joint axes (unit) and origins in world frame at q=0, plus T_home (= POE FK at q=0)."""
    joints = kb.joints
    n = len(joints)
    axes_world = np.zeros((n, 3), dtype=np.float64)
    origins_world = np.zeros((n, 3), dtype=np.float64)

    t_acc = np.eye(4, dtype=np.float64)
    for i, joint in enumerate(joints):
        t_pre = t_acc @ joint.T_left
        axes_world[i] = t_pre[:3, :3] @ joint.axis
        origins_world[i] = t_pre[:3, 3]
        t_acc = t_pre @ joint.T_right

    return axes_world, origins_world, t_acc


def _line_line_perpendicular(
    p1: NDArray[np.float64],
    d1: NDArray[np.float64],
    p2: NDArray[np.float64],
    d2: NDArray[np.float64],
    *,
    parallel_tol: float = 1e-9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Closest-point pair on two infinite lines.

    :returns: ``(foot1, foot2, is_parallel)``.
    """
    cross = np.cross(d1, d2)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < parallel_tol:
        # Parallel. foot1 = p1; foot2 is closest point on L_2 to p1.
        delta = p1 - p2
        t2 = -float(np.dot(delta, d2))
        foot2 = p2 + t2 * d2
        return p1.copy(), foot2, True
    n2 = float(np.dot(cross, cross))
    delta = p2 - p1
    t1 = float(np.dot(np.cross(delta, d2), cross)) / n2
    t2 = float(np.dot(np.cross(delta, d1), cross)) / n2
    return p1 + t1 * d1, p2 + t2 * d2, False


def _signed_angle(
    v1: NDArray[np.float64], v2: NDArray[np.float64], normal: NDArray[np.float64]
) -> float:
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_a = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    sin_a = float(np.dot(np.cross(v1, v2), normal))
    return float(np.arctan2(sin_a, cos_a))


def poe_to_dh(kb: KinBody) -> DhWithOffset:
    """Convert a 6R POE KinBody to standard distal DH parameters.

    Returns a :class:`DhWithOffset` carrying ``(alpha, a, d, theta_offset)``
    of length 6 plus the boundary ``T_pre, T_post`` transforms needed to
    bridge POE-world to DH-frame-0 and DH-frame-n to POE-end.
    """
    joints = kb.joints
    n = len(joints)
    if n != 6:
        raise ValueError(f"poe_to_dh: expected 6 joints, got {n}")
    for joint in joints:
        if joint.joint_type != "revolute":
            raise ValueError(f"poe_to_dh: only revolute supported; got {joint.joint_type}")

    axes_world, origins_world, t_home = _kinbody_world_axes_origins(kb)

    # Build n+1 frames (z, x, origin) in world coords:
    # - z_i for i=0..n-1: joint (i+1)'s axis in world frame = axes_world[i]
    # - z_n: tool flange z-axis (from T_home rotation block)
    # - origin_i for i=1..n-1: foot of common perpendicular between z_{i-1} and
    #   z_i, on z_i's line.
    # - origin_n: tool flange origin = T_home[:3, 3]
    # - origin_0: free choice on z_0's line; we pick origins_world[0] (the
    #   actual joint-1 origin in world coords) so that for arms with joint 1
    #   at world origin (UR5-style) we recover the conventional placement.
    z_axes: list[NDArray[np.float64]] = []
    x_axes: list[NDArray[np.float64]] = []
    origins: list[NDArray[np.float64]] = []

    # Frame 0: z_0 = joint-1's axis (in world frame). x_0 free perpendicular to
    # z_0; align with world x as much as possible (project + renormalize). T_pre
    # then bridges the user's world frame to this DH frame 0.
    z_0 = axes_world[0] / np.linalg.norm(axes_world[0])
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, z_0)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    x_0 = ref - float(np.dot(ref, z_0)) * z_0
    x_0 /= np.linalg.norm(x_0)
    z_axes.append(z_0)
    x_axes.append(x_0)
    origins.append(origins_world[0].copy())

    # Frames 1..n-1: positioned at common perpendicular between z_{i-1} = joints[i-1].axis
    # and z_i = joints[i].axis. Origin at foot on z_i line.
    for i in range(1, n):
        z_prev = axes_world[i - 1]
        p_prev = origins_world[i - 1]
        z_curr = axes_world[i]
        p_curr = origins_world[i]
        foot_prev, foot_curr, is_parallel = _line_line_perpendicular(p_prev, z_prev, p_curr, z_curr)
        z_axes.append(z_curr.copy())
        if is_parallel:
            # x_i is the perpendicular direction from foot_prev to foot_curr
            # (on the parallel-line plane), normalized.
            diff = foot_curr - foot_prev
            diff_norm = float(np.linalg.norm(diff))
            if diff_norm < 1e-9:
                # Coincident axes -- degenerate. Pick any perpendicular to z_curr.
                ref = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(ref, z_curr)) > 0.99:
                    ref = np.array([0.0, 1.0, 0.0])
                x_perp = ref - float(np.dot(ref, z_curr)) * z_curr
                x_perp /= np.linalg.norm(x_perp)
                x_axes.append(x_perp)
            else:
                x_axes.append(diff / diff_norm)
        else:
            # Skew/intersecting: x_i = (z_{i-1} x z_i) normalized, oriented from foot_prev to foot_curr.  # noqa: E501
            cross = np.cross(z_prev, z_curr)
            cross_norm = float(np.linalg.norm(cross))
            x_dir = cross / cross_norm
            # If foot_curr is "behind" foot_prev in the x_dir sense, flip.
            diff = foot_curr - foot_prev
            if np.dot(diff, x_dir) < 0:
                x_dir = -x_dir
            x_axes.append(x_dir)
        origins.append(foot_curr)

    # Frame n = tool flange. z_n from T_home[:3, 2], origin from T_home[:3, 3].
    # x_n: project x_{n-1} onto plane perpendicular to z_n.
    z_n = t_home[:3, 2]
    o_n = t_home[:3, 3]
    z_axes.append(z_n)
    x_prev = x_axes[-1]
    x_proj = x_prev - float(np.dot(x_prev, z_n)) * z_n
    x_proj_norm = float(np.linalg.norm(x_proj))
    if x_proj_norm < 1e-9:
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, z_n)) > 0.99:
            ref = np.array([0.0, 1.0, 0.0])
        x_proj = ref - float(np.dot(ref, z_n)) * z_n
        x_proj_norm = float(np.linalg.norm(x_proj))
    x_axes.append(x_proj / x_proj_norm)
    origins.append(o_n)

    # Compute (alpha_i, a_i, d_i, theta_offset_i) for i=1..n, transitioning
    # from frame i-1 to frame i.
    alpha = np.zeros(n, dtype=np.float64)
    a_arr = np.zeros(n, dtype=np.float64)
    d_arr = np.zeros(n, dtype=np.float64)
    theta_offset = np.zeros(n, dtype=np.float64)

    for k in range(n):
        # Index k in arrays corresponds to Spong's joint i = k+1, transitioning
        # frame k to frame k+1.
        z_a, x_a, o_a = z_axes[k], x_axes[k], origins[k]
        z_b, x_b, o_b = z_axes[k + 1], x_axes[k + 1], origins[k + 1]

        # alpha_i = signed angle from z_a to z_b around x_b
        alpha[k] = _signed_angle(z_a, z_b, x_b)
        # theta_offset_i = signed angle from x_a to x_b around z_a
        theta_offset[k] = _signed_angle(x_a, x_b, z_a)
        # d_i = (o_b - o_a) projected onto z_a (signed offset along previous joint axis)
        delta = o_b - o_a
        d_arr[k] = float(np.dot(delta, z_a))
        # a_i = (o_b - o_a) projected onto x_b (signed offset along common perp)
        # NB: only valid because o_b is on z_b's line at foot of perp; the residual
        # after subtracting d_i z_a should lie along x_b.
        a_arr[k] = float(np.dot(delta, x_b))

    # T_pre: maps the user's world frame -> DH frame 0.
    # Frame 0 sits at origin_0 = joints[0] origin in world, with axes
    # (x_0, y_0=z_0xx_0, z_0). T_pre's columns are these axes expressed in the
    # user's world frame plus the origin translation. For commercial arms whose
    # joint 1 axis is world +z and whose joint 1 sits at the world origin (UR5,
    # Puma 560), T_pre collapses to identity.
    y_0 = np.cross(z_0, x_0)
    t_pre = np.eye(4, dtype=np.float64)
    t_pre[:3, 0] = x_0
    t_pre[:3, 1] = y_0
    t_pre[:3, 2] = z_0
    t_pre[:3, 3] = origins[0]

    # T_post: from "where DH FK ends at theta=offset" to "where POE FK ends
    # at q=0". DH ends at frame n (z_axes[n], x_axes[n], origins[n]). The
    # remaining orientation DOF (y_n = z_n x x_n) is determined; check
    # discrepancy vs t_home.
    z_n = z_axes[n]
    x_n = x_axes[n]
    y_n = np.cross(z_n, x_n)
    t_dh_end = np.eye(4, dtype=np.float64)
    t_dh_end[:3, 0] = x_n
    t_dh_end[:3, 1] = y_n
    t_dh_end[:3, 2] = z_n
    t_dh_end[:3, 3] = origins[n]
    # We need: t_dh_end @ t_post = t_home  ->  t_post = t_dh_end^{-1} @ t_home
    t_post = np.linalg.solve(t_dh_end, t_home)

    return DhWithOffset(
        alpha=alpha,
        a=a_arr,
        d=d_arr,
        theta_offset=theta_offset,
        t_pre=t_pre,
        t_post=t_post,
    )
