"""Shared Rodrigues rotation primitives for the subproblem solvers.

Kept private (underscore-prefixed) and internal to the subproblems package.
3-vector primitives ``_cross3`` / ``_dot3`` / ``_norm3`` are re-exported
from :mod:`ssik.kinematics._scalar3` so the subproblems hot path uses the
same deterministic, no-BLAS scalar versions as the codegen pipeline.

This module is Cython-compilable in pure-Python mode (#137 Slice 1):
the ``import cython`` block + ``@cython.locals`` / ``@cython.ccall``
decorators are no-ops when interpreted by CPython, but generate
typed C code when compiled by Cython. ``rotation_matrix`` and ``rotate``
are the two hottest functions in any IK solve (per profile of Franka
7R: ``rotate`` accounts for ~21% of total IK time); typing them lets
Cython inline the scalar trig + arithmetic without Python-object
boxing.
"""

from __future__ import annotations

import math

import cython
import numpy as np
from numpy.typing import NDArray

# Re-exports (kept underscore-prefixed for backwards-compat with existing
# call sites in the subproblems package).
from ssik.kinematics._scalar3 import _cross3, _dot3, _norm3

__all__ = ["_cross3", "_dot3", "_norm3", "rotate", "rotation_matrix"]


@cython.ccall
@cython.locals(
    c=cython.double,
    s=cython.double,
    kv=cython.double,
    one_minus_c=cython.double,
)
def rotate(k: NDArray[np.float64], theta: float, v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rotate vector ``v`` by angle ``theta`` about unit axis ``k`` (Rodrigues).

    ``rotate(k, theta, v) = v cos(theta) + (k x v) sin(theta) + k (k . v) (1 - cos(theta))``
    """
    # Scalarized Rodrigues (like ``rotation_matrix``): builds the 3-vector from
    # float ops in a single allocation instead of ~5 numpy temporaries
    # (``_cross3`` array + four element-wise products/sums). ``rotate`` is one of
    # the hottest calls in the SP5/SP6 refine loops, so the per-call heap churn
    # matters. Element order matches the vectorized form exactly (bit-identical).
    c = math.cos(theta)
    s = math.sin(theta)
    kx, ky, kz = float(k[0]), float(k[1]), float(k[2])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    m = (1.0 - c) * (kx * vx + ky * vy + kz * vz)
    return np.array(
        [
            vx * c + (ky * vz - kz * vy) * s + kx * m,
            vy * c + (kz * vx - kx * vz) * s + ky * m,
            vz * c + (kx * vy - ky * vx) * s + kz * m,
        ],
        dtype=np.float64,
    )


@cython.ccall
@cython.locals(
    c=cython.double,
    s=cython.double,
    x=cython.double,
    y=cython.double,
    z=cython.double,
    oc=cython.double,
)
def rotation_matrix(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """3x3 rotation matrix around unit ``axis`` by ``angle`` (Rodrigues form).

    Replaces the per-solver ``_rot_mat`` helpers that were duplicated across
    every ikgeo + jointlock solver. Tight closed-form using only float
    multiplies and adds; faster than constructing a skew-symmetric matrix
    and matrix-multiplying.
    """
    c = math.cos(angle)
    s = math.sin(angle)
    x = float(axis[0])
    y = float(axis[1])
    z = float(axis[2])
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )
