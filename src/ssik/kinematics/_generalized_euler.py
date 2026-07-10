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

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["decompose_3axis"]

# Constant identity -- np.eye(3) allocates + fills on every call (~0.8 us);
# these 3-vector/3x3 hot paths run it hundreds of times per solve.
_EYE3 = np.eye(3, dtype=np.float64)


def _cross3(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scalar 3-vector cross product. ~12x faster than ``np.cross`` on
    length-3 inputs, which pays for axis-normalization + moveaxis machinery
    irrelevant to a single 3-vector."""
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    return np.array(
        [a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], dtype=np.float64
    )


def _norm3(a: NDArray[np.float64]) -> float:
    """Euclidean norm of a 3-vector. ~2x faster than ``np.linalg.norm``,
    which pays for ravel + complex-type checks + dispatcher."""
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

# Below this the b-sinusoid amplitude sqrt(A^2 + B^2) is ~0: the three axes are
# (near-)collinear and no general decomposition exists. SRS shoulder/wrist
# triples are never collinear, so callers there won't hit this.
_DEGENERATE_AMPLITUDE = 1e-12
# Below this a perpendicular component is ~0 (gimbal): the corresponding outer
# angle is indeterminate and we pin it to 0 (a + c is the recoverable quantity).
_GIMBAL_EPS = 1e-9


def _axis_angle_matrix(axis: NDArray[np.float64], theta: float) -> NDArray[np.float64]:
    """Rodrigues rotation matrix for ``theta`` about a unit ``axis``."""
    c, s = math.cos(theta), math.sin(theta)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    out: NDArray[np.float64] = _EYE3 + s * k + (1.0 - c) * (k @ k)
    return out


def _unit_perp(axis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Some unit vector perpendicular to ``axis``."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    v = ref - axis * (axis @ ref)
    unit: NDArray[np.float64] = v / _norm3(v)
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
    if _norm3(vf) < _GIMBAL_EPS or _norm3(vt) < _GIMBAL_EPS:
        return 0.0
    return float(np.arctan2(axis @ _cross3(vf, vt), vf @ vt))


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
    u1 = u1 / _norm3(u1)
    u2 = u2 / _norm3(u2)
    u3 = u3 / _norm3(u3)

    a_coef = u1 @ u3 - (u1 @ u2) * (u2 @ u3)
    b_coef = u1 @ _cross3(u2, u3)
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
        if _norm3(vf - u1 * (u1 @ vf)) < _GIMBAL_EPS:
            a = 0.0
            residual = _axis_angle_matrix(u2, -b) @ R
            ref = _unit_perp(u3)
            c = _angle_about(u3, ref, residual @ ref)
        else:
            a = _angle_about(u1, vf, r_n3)
            c = _angle_about(u3, rt_n1, rb.T @ u1)
        out.append((a, b, c))
    return out


# ---------------------------------------------------------------------------
# Batched variants (vectorise over a leading N axis of rotation matrices).
#
# For a *fixed* triple of axes (n1, n2, n3) -- the SRS-class case, where the
# joint axes are constant across the swivel sweep -- the b-sinusoid amplitude
# and phase are scalars; only the ``u1 . R u3`` term and the per-branch frame
# algebra vary per row. These helpers evaluate the scalar :func:`decompose_3axis`
# math for an ``(N, 3, 3)`` stack of ``R`` in one broadcast, so the general
# SRS sweep runs branch-batched like the canonical ZYZ path.
# ---------------------------------------------------------------------------


def _rodrigues_axis_batch(
    axis: NDArray[np.float64], angles: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Rodrigues matrices for a fixed unit ``axis`` and ``(N,)`` ``angles`` ->
    ``(N, 3, 3)``."""
    c = np.cos(angles)
    s = np.sin(angles)
    omc = 1.0 - c
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    k2 = k @ k
    return (
        _EYE3[None, :, :]
        + s[:, None, None] * k[None, :, :]
        + omc[:, None, None] * k2[None, :, :]
    )


def _angle_about_batch(
    axis: NDArray[np.float64],
    v_from: NDArray[np.float64],
    v_to: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorised :func:`_angle_about` over ``(N, 3)`` vector stacks (``v_from``
    may be ``(3,)`` and broadcasts). Gimbal rows (vanishing perpendicular
    component) return ``0.0``, matching the scalar contract."""
    vf = v_from - axis * (v_from @ axis)[..., None]
    vt = v_to - axis * (v_to @ axis)[..., None]
    vf = np.broadcast_to(vf, vt.shape)
    num = np.cross(vf, vt) @ axis
    den = np.einsum("ni,ni->n", vf, vt)
    out = np.arctan2(num, den)
    nf = np.linalg.norm(vf, axis=1)
    nt = np.linalg.norm(vt, axis=1)
    out[(nf < _GIMBAL_EPS) | (nt < _GIMBAL_EPS)] = 0.0
    return out


def decompose_3axis_batch(
    R: NDArray[np.float64],
    n1: ArrayLike,
    n2: ArrayLike,
    n3: ArrayLike,
) -> tuple[list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]], bool]:
    """Batched :func:`decompose_3axis` for an ``(N, 3, 3)`` stack ``R`` and a
    single fixed axis triple.

    :returns: ``(branches, ok)``. ``ok`` is ``False`` iff the axes are
        (near-)collinear (no decomposition exists for any row) -- then
        ``branches`` is empty. Otherwise ``branches`` holds the two b-solution
        branches, each a ``(a, b, c)`` tuple of ``(N,)`` arrays. Unlike the
        scalar variant this always emits *both* branches (even at the
        ``delta -> 0`` boundary where they coincide); the coincident duplicates
        are collapsed by the caller's dedup pass.
    """
    R = np.asarray(R, dtype=np.float64)
    u1 = np.asarray(n1, dtype=np.float64)
    u2 = np.asarray(n2, dtype=np.float64)
    u3 = np.asarray(n3, dtype=np.float64)
    u1 = u1 / _norm3(u1)
    u2 = u2 / _norm3(u2)
    u3 = u3 / _norm3(u3)

    a_coef = float(u1 @ u3 - (u1 @ u2) * (u2 @ u3))
    b_coef = float(u1 @ _cross3(u2, u3))
    amplitude = float(np.hypot(a_coef, b_coef))
    if amplitude < _DEGENERATE_AMPLITUDE:
        return [], False

    # c_const per row: u1 . R u3.
    c_const = np.einsum("i,nij,j->n", u1, R, u3) - (u1 @ u2) * (u2 @ u3)
    phi = float(np.arctan2(b_coef, a_coef))
    delta = np.arccos(np.clip(c_const / amplitude, -1.0, 1.0))  # (N,)

    r_n3 = R @ u3  # (N, 3)
    rt_n1 = np.einsum("nij,i->nj", R, u1)  # R^T u1, (N, 3)
    ref = _unit_perp(u3)  # (3,), constant

    branches: list[tuple[NDArray, NDArray, NDArray]] = []
    for b in (phi + delta, phi - delta):  # each (N,)
        rb = _rodrigues_axis_batch(u2, b)  # (N, 3, 3)
        vf = np.einsum("nij,j->ni", rb, u3)  # (N, 3)
        # Non-gimbal outer angles.
        a_ng = _angle_about_batch(u1, vf, r_n3)
        rbT_u1 = np.einsum("nij,i->nj", rb, u1)  # rb^T u1, (N, 3)
        c_ng = _angle_about_batch(u3, rt_n1, rbT_u1)
        # Gimbal rows: Rot(n2,b) n3 parallel to n1 -> pin a = 0, recover c from
        # the residual Rot(n2,-b) R (a pure n3 rotation).
        gim = np.linalg.norm(vf - u1 * (vf @ u1)[:, None], axis=1) < _GIMBAL_EPS
        a = np.where(gim, 0.0, a_ng)
        c = c_ng
        if gim.any():
            rb_inv = _rodrigues_axis_batch(u2, -b)
            residual = rb_inv @ R  # (N, 3, 3)
            c_g = _angle_about_batch(u3, ref, residual @ ref)
            c = np.where(gim, c_g, c_ng)
        branches.append((a, b, c))
    return branches, True
