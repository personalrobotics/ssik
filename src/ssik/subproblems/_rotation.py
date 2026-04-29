"""Shared Rodrigues rotation primitives for the subproblem solvers.

Kept private (underscore-prefixed) and internal to the subproblems package.
3-vector primitives ``_cross3`` / ``_dot3`` / ``_norm3`` are re-exported
from :mod:`ssik.kinematics._scalar3` so the subproblems hot path uses the
same deterministic, no-BLAS scalar versions as the codegen pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Re-exports (kept underscore-prefixed for backwards-compat with existing
# call sites in the subproblems package).
from ssik.kinematics._scalar3 import _cross3, _dot3, _norm3

__all__ = ["_cross3", "_dot3", "_norm3", "rotate", "rotation_matrix"]


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
