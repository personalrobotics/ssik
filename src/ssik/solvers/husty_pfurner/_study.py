"""Study quaternion / dual-quaternion machinery for Husty-Pfurner.

The HP algorithm represents elements of SE(3) as **Study coordinates** -- an
8-vector ``(x_0, x_1, x_2, x_3, y_0, y_1, y_2, y_3)`` corresponding to a
dual quaternion ``sigma = p + epsilon * q`` where:

* ``p = (x_0, x_1, x_2, x_3)`` is the rotation quaternion (``epsilon`` term zero).
* ``q = (y_0, y_1, y_2, y_3)`` is the translation quaternion such that
  ``q = (1/2) * t_quat * p``, where ``t_quat = (0, t_x, t_y, t_z)``.

Valid SE(3) elements lie on the **Study quadric** ``x_0 y_0 + x_1 y_1 +
x_2 y_2 + x_3 y_3 = 0`` with ``(x_0, x_1, x_2, x_3) != 0``.

Dual quaternion arithmetic over the 8-vec form:

* Sum: ``(p_a + eps q_a) + (p_b + eps q_b) = (p_a + p_b) + eps (q_a + q_b)``
* Product: ``(p_a + eps q_a)(p_b + eps q_b) = p_a p_b + eps (p_a q_b + q_a p_b)``
* Conjugate: ``sigma^* = p^* + eps q^*``  (quaternion conjugate of each part).

Phase 5b of GitHub #158 / #162. Algorithmic reference: Capco, Loquias,
Manongsong, Nemenzo (2019), 'Inverse Kinematics of Some General 6R/P
Manipulators', arXiv 1906.07813, Section 2.

This module is pure numpy (no Cython yet). Cython enters only as a
profile-guided escalation in Phase 5h if the perf gate fires.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody

__all__ = [
    "DQ_IDENTITY",
    "dq_chain",
    "dq_conj",
    "dq_from_se3",
    "dq_joint",
    "dq_mul",
    "se3_from_dq",
    "study_quadric_residual",
]


# Identity dual quaternion: identity rotation, zero translation.
DQ_IDENTITY: NDArray[np.float64] = np.array(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64
)


# ---------------------------------------------------------------------------
# Quaternion primitives. Quaternions are stored as 4-vec ``(p_0, p_1, p_2,
# p_3)`` with the scalar in slot 0. All operations are pure numpy / scalar
# math so we can lift them into Cython later without rework.
# ---------------------------------------------------------------------------


def _quat_mul(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product ``p * q`` of two unit quaternions.

    ``(a_0 + a_1 i + a_2 j + a_3 k)(b_0 + b_1 i + b_2 j + b_3 k)`` expands
    to four bilinear scalar combinations; we write them inline so the
    function is allocation-free apart from the result array.
    """
    p0, p1, p2, p3 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
            p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
            p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
            p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
        ],
        dtype=np.float64,
    )


def _quat_conj(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quaternion conjugate ``(p_0, -p_1, -p_2, -p_3)``."""
    return np.array([p[0], -p[1], -p[2], -p[3]], dtype=np.float64)


def _quat_from_rot(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract a unit quaternion from a 3x3 rotation matrix (Shepperd 1978).

    Picks the diagonal element with the largest magnitude as the
    'numerically dominant' branch; this avoids the ill-conditioned
    division near 180-degree rotations that the textbook trace-only
    method suffers from.
    """
    m00, m01, m02 = float(R[0, 0]), float(R[0, 1]), float(R[0, 2])
    m10, m11, m12 = float(R[1, 0]), float(R[1, 1]), float(R[1, 2])
    m20, m21, m22 = float(R[2, 0]), float(R[2, 1]), float(R[2, 2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0  # s = 4 * q0
        q0 = 0.25 * s
        q1 = (m21 - m12) / s
        q2 = (m02 - m20) / s
        q3 = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0  # s = 4 * q1
        q0 = (m21 - m12) / s
        q1 = 0.25 * s
        q2 = (m01 + m10) / s
        q3 = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0  # s = 4 * q2
        q0 = (m02 - m20) / s
        q1 = (m01 + m10) / s
        q2 = 0.25 * s
        q3 = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0  # s = 4 * q3
        q0 = (m10 - m01) / s
        q1 = (m02 + m20) / s
        q2 = (m12 + m21) / s
        q3 = 0.25 * s
    return np.array([q0, q1, q2, q3], dtype=np.float64)


def _rot_from_quat(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """3x3 rotation matrix from a quaternion ``(p_0, p_1, p_2, p_3)``.

    Uses the standard formula. The quaternion does not need to be unit
    length: we normalise as part of the formula to keep the result on
    SO(3) even if upstream arithmetic drifted off the unit sphere.
    """
    p0, p1, p2, p3 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    n = p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (p2 * p2 + p3 * p3), s * (p1 * p2 - p3 * p0), s * (p1 * p3 + p2 * p0)],
            [s * (p1 * p2 + p3 * p0), 1.0 - s * (p1 * p1 + p3 * p3), s * (p2 * p3 - p1 * p0)],
            [s * (p1 * p3 - p2 * p0), s * (p2 * p3 + p1 * p0), 1.0 - s * (p1 * p1 + p2 * p2)],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Dual quaternion primitives. The 8-vec carries ``(p, q)`` packed as
# ``[p_0, p_1, p_2, p_3, q_0, q_1, q_2, q_3]``.
# ---------------------------------------------------------------------------


def dq_mul(sigma_a: NDArray[np.float64], sigma_b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dual-quaternion product ``sigma_a * sigma_b``.

    ``(p_a + eps q_a)(p_b + eps q_b) = (p_a p_b) + eps (p_a q_b + q_a p_b)``.
    """
    p_a = sigma_a[:4]
    q_a = sigma_a[4:]
    p_b = sigma_b[:4]
    q_b = sigma_b[4:]
    p = _quat_mul(p_a, p_b)
    q = _quat_mul(p_a, q_b) + _quat_mul(q_a, p_b)
    return np.concatenate([p, q])


def dq_conj(sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Dual-quaternion conjugate ``(p^*, q^*)``.

    The Study-coordinate inverse on the Study quadric. Useful for
    constructing reverse-chain transforms (``sigma_E sigma_6^{-1}``).
    """
    return np.concatenate([_quat_conj(sigma[:4]), _quat_conj(sigma[4:])])


def dq_from_se3(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a 4x4 SE(3) matrix to its 8-vec dual-quaternion representation.

    ``T = [[R, t], [0, 1]]`` -> ``sigma = (p, (1/2) t_quat p)`` where
    ``p`` is the unit quaternion of ``R`` and ``t_quat = (0, t_x, t_y, t_z)``.
    """
    p = _quat_from_rot(T[:3, :3])
    t = T[:3, 3]
    t_quat = np.array([0.0, float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    q = 0.5 * _quat_mul(t_quat, p)
    return np.concatenate([p, q])


def se3_from_dq(sigma: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert an 8-vec dual quaternion to a 4x4 SE(3) matrix.

    Recovers ``R`` from ``p`` and ``t`` from ``q`` via
    ``t_quat = 2 * q * p^{-1}`` (the inverse of the construction in
    :func:`dq_from_se3`). Also valid for non-unit dual quaternions:
    we divide by ``|p|^2`` so the result is well-defined.
    """
    p = sigma[:4]
    q = sigma[4:]
    R = _rot_from_quat(p)
    p_norm_sq = float(p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3])
    if p_norm_sq == 0.0:
        T = np.eye(4, dtype=np.float64)
        return T
    t_quat = 2.0 * _quat_mul(q, _quat_conj(p)) / p_norm_sq
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t_quat[1:]
    return T


def study_quadric_residual(sigma: NDArray[np.float64]) -> float:
    """Return ``x_0 y_0 + x_1 y_1 + x_2 y_2 + x_3 y_3``.

    Valid SE(3) elements lie on the Study quadric, so this should be
    zero (within floating-point noise) for any pose constructed via
    :func:`dq_from_se3` or :func:`dq_chain`.
    """
    s = sigma
    return float(s[0] * s[4] + s[1] * s[5] + s[2] * s[6] + s[3] * s[7])


# ---------------------------------------------------------------------------
# Joint and chain composition
# ---------------------------------------------------------------------------


def dq_joint(axis: NDArray[np.float64], q: float, joint_type: str) -> NDArray[np.float64]:
    """Dual-quaternion representation of a single joint at parameter ``q``.

    * For ``joint_type == "revolute"``: rotation by ``q`` radians about
      ``axis`` (assumed unit length). Quaternion ``(cos(q/2),
      sin(q/2) * axis)``; translation part zero.
    * For ``joint_type == "prismatic"``: translation by ``q`` units
      along ``axis``. Quaternion ``(1, 0, 0, 0)`` (identity rotation);
      translation part ``(0, q*ax/2, q*ay/2, q*az/2)``.

    Other joint types raise :class:`ValueError`.
    """
    ax = float(axis[0])
    ay = float(axis[1])
    az = float(axis[2])
    if joint_type == "revolute":
        c = math.cos(0.5 * q)
        s = math.sin(0.5 * q)
        return np.array(
            [c, s * ax, s * ay, s * az, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
    if joint_type == "prismatic":
        return np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5 * q * ax, 0.5 * q * ay, 0.5 * q * az],
            dtype=np.float64,
        )
    raise ValueError(f"unsupported joint_type: {joint_type!r}")


def dq_chain(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compose the full kinematic chain into an 8-vec dual quaternion.

    Walks the joints in order, applying ``T_left @ Joint(axis, q) @ T_right``
    as dual-quaternion products. The result represents the same SE(3)
    pose as :func:`ssik.kinematics.poe_fk.poe_forward_kinematics` -- in
    other words, ``se3_from_dq(dq_chain(kb, q)) == poe_forward_kinematics(
    kb, q)`` up to floating-point precision.

    Cross-validates the Study representation: if the chain composition
    here disagrees with the SE(3) FK, the dual-quaternion arithmetic is
    wrong somewhere upstream of HP.
    """
    sigma = DQ_IDENTITY.copy()
    for i, joint in enumerate(kb.joints):
        sigma_l = dq_from_se3(joint.T_left)
        sigma_j = dq_joint(joint.axis, float(q[i]), joint.joint_type)
        sigma_r = dq_from_se3(joint.T_right)
        sigma_step = dq_mul(sigma_l, dq_mul(sigma_j, sigma_r))
        sigma = dq_mul(sigma, sigma_step)
    return sigma
