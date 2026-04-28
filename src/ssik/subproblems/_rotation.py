"""Shared Rodrigues rotation + 3-vector primitives for the subproblem solvers.

Kept private (underscore-prefixed) and internal to the subproblems package.

The helpers ``_cross3`` and ``_dot3`` exist because ``np.cross`` /
``np.dot`` carry 2-10 µs of axis-normalisation dispatch overhead per call
that swamps the actual arithmetic on 3-vectors. Hand-rolled versions for
the 3-vector specialisation are 20-50x faster per call. Inside
:func:`rotate` and the SP-N subproblem hot paths these helpers are called
~10k+ times per IK solve; the per-call overhead reduction is the
single largest tier-0 win on UR5 (#93).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["rotate", "rotation_matrix"]


def _cross3(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """3-vector cross product, no dispatch overhead.

    Equivalent to ``np.cross(a, b)`` for 3-element arrays. Several
    subproblem hot paths call this inside Newton / Gauss-Newton loops,
    where the per-call overhead of ``np.cross`` dominates the actual
    arithmetic.
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def _dot3(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """3-vector dot product as ``float``; no dispatch overhead."""
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _norm3(a: NDArray[np.floating]) -> float:
    """3-vector L2 norm, no dispatch overhead. Faster than ``np.linalg.norm``."""
    return float(np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]))


def rotate(k: NDArray[np.float64], theta: float, v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate vector ``v`` by angle ``theta`` about unit axis ``k`` (Rodrigues).

    ``rotate(k, theta, v) = v cos(theta) + (k x v) sin(theta) + k (k . v) (1 - cos(theta))``
    """
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    kv = _dot3(k, v)
    return v * c + _cross3(k, v) * s + k * (kv * (1.0 - c))


def rotation_matrix(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """3x3 rotation matrix around unit ``axis`` by ``angle`` (Rodrigues form).

    Replaces the per-solver ``_rot_mat`` helpers that were duplicated across
    every ikgeo + jointlock solver. Tight closed-form using only float
    multiplies and adds; faster than constructing a skew-symmetric matrix
    and matrix-multiplying.
    """
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )
