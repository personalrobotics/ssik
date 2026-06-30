"""Generalized Euler / Davenport decomposition of a rotation into three
rotations about *arbitrary* axes.

Foundation for #354 -- generalizing :mod:`ssik.solvers.seven_r.srs` past its
ZYZ-only gate so it covers any concurrent-axis spherical 7R (e.g. the Galaxea
R1 Pro and OpenArm, whose spherical triples are y-x-z / z-y-x, not z-y-z).

:func:`decompose_3axis` finds angles ``(a, b, c)`` such that

    R == Rot(n1, a) @ Rot(n2, b) @ Rot(n3, c)

for arbitrary unit axes ``n1, n2, n3`` (not necessarily orthogonal or parallel).
Classical ZYZ Euler is the special case ``n1 == n3``. Reference: Shuster &
Markley, "Generalization of the Euler Angles", J. Astronautical Sciences
51(2), 2003.

Derivation sketch. Because ``Rot(n1, a)`` fixes ``n1`` and ``Rot(n3, c)`` fixes
``n3``,

    n1 . R n3 == n1 . Rot(n2, b) n3,

and expanding ``Rot(n2, b) n3`` via Rodrigues gives a single sinusoid in ``b``:

    A cos b + B sin b == C,

with ``A = n1.n3 - (n1.n2)(n2.n3)``, ``B = n1 . (n2 x n3)``,
``C = n1 . R n3 - (n1.n2)(n2.n3)``. That yields up to two ``b`` (the two
elbow-style branches). For each ``b``, ``a`` is the rotation about ``n1``
carrying ``Rot(n2, b) n3`` onto ``R n3``, and ``c`` the rotation about ``n3``
carrying ``R^T n1`` onto ``Rot(n2, b)^T n1``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["decompose_3axis"]

# Below this the b-sinusoid amplitude sqrt(A^2 + B^2) is ~0: the three axes are
# (near-)collinear and no general decomposition exists. SRS shoulder/wrist
# triples are never collinear, so callers there won't hit this.
_DEGENERATE_AMPLITUDE = 1e-12
# Below this a perpendicular component is ~0 (gimbal): the corresponding outer
# angle is indeterminate and we pin it to 0 (a + c is the recoverable quantity).
_GIMBAL_EPS = 1e-9


def _axis_angle_matrix(axis: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    """Rodrigues rotation matrix for ``theta`` about a unit ``axis``."""
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    out: NDArray[np.float64] = np.eye(3, dtype=np.float64) + s * k + (1.0 - c) * (k @ k)
    return out


def _unit_perp(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Some unit vector perpendicular to ``axis``."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = ref - axis * (axis @ ref)
    unit: NDArray[np.float64] = v / np.linalg.norm(v)
    return unit


def _angle_about(
    axis: NDArray[np.float64], v_from: NDArray[np.float64], v_to: NDArray[np.float64]
) -> float:
    """Rotation angle about ``axis`` carrying the component of ``v_from``
    perpendicular to ``axis`` onto that of ``v_to``.

    Returns ``0.0`` when either perpendicular component vanishes (gimbal): the
    angle is then indeterminate and the caller's FK gate sorts out the branch.
    """
    vf = v_from - axis * (axis @ v_from)
    vt = v_to - axis * (axis @ v_to)
    if np.linalg.norm(vf) < _GIMBAL_EPS or np.linalg.norm(vt) < _GIMBAL_EPS:
        return 0.0
    return float(np.arctan2(axis @ np.cross(vf, vt), vf @ vt))


def decompose_3axis(
    R: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    n3: ArrayLike,
) -> list[tuple[float, float, float]]:
    """Decompose ``R`` into ``Rot(n1, a) @ Rot(n2, b) @ Rot(n3, c)``.

    :param R: 3x3 rotation matrix.
    :param n1, n2, n3: rotation axes (need not be unit; normalized internally).
    :returns: up to two ``(a, b, c)`` solution triples (one when the middle
        angle is at a branch boundary). Empty if the axes are (near-)collinear
        so no decomposition exists.
    """
    R = np.asarray(R, dtype=np.float64)
    u1 = np.asarray(n1, dtype=np.float64)
    u2 = np.asarray(n2, dtype=np.float64)
    u3 = np.asarray(n3, dtype=np.float64)
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    u3 = u3 / np.linalg.norm(u3)

    a_coef = u1 @ u3 - (u1 @ u2) * (u2 @ u3)
    b_coef = u1 @ np.cross(u2, u3)
    c_const = u1 @ R @ u3 - (u1 @ u2) * (u2 @ u3)

    amplitude = float(np.hypot(a_coef, b_coef))
    if amplitude < _DEGENERATE_AMPLITUDE:
        return []

    phi = np.arctan2(b_coef, a_coef)
    delta = float(np.arccos(np.clip(c_const / amplitude, -1.0, 1.0)))
    b_branches = (phi + delta,) if delta < 1e-12 else (phi + delta, phi - delta)

    r_n3 = R @ u3
    rt_n1 = R.T @ u1
    out: list[tuple[float, float, float]] = []
    for b_raw in b_branches:
        b = float(b_raw)
        rb = _axis_angle_matrix(u2, b)
        vf = rb @ u3
        # Gimbal: when ``Rot(n2,b) n3`` is parallel to ``n1`` the outer angle
        # ``a`` is indeterminate (only a combination of a, c is fixed). Pin
        # ``a = 0`` and recover ``c`` from the residual ``Rot(n2,-b) R``, which
        # is then a pure ``n3`` rotation. (Hit by symmetric ZYZ triples at
        # b = 0 / pi -- classical Euler gimbal.)
        if np.linalg.norm(vf - u1 * (u1 @ vf)) < _GIMBAL_EPS:
            a = 0.0
            residual = _axis_angle_matrix(u2, -b) @ R
            ref = _unit_perp(u3)
            c = _angle_about(u3, ref, residual @ ref)
        else:
            a = _angle_about(u1, vf, r_n3)
            c = _angle_about(u3, rt_n1, rb.T @ u1)
        out.append((a, b, c))
    return out
