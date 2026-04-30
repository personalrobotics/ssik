"""Deterministic small-vector / small-matrix primitives.

These hand-rolled scalar versions of common 3-vector / 3x3 / 4x4 operations
exist for two reasons:

1. **Speed.** ``np.cross`` / ``np.dot`` / ``np.linalg.norm`` carry 2-10 us of
   axis-normalisation dispatch overhead per call that swamps the actual
   arithmetic on 3-vectors. Hand-rolled versions are 20-50x faster per
   call. Inside hot subproblem paths (#93 tier-1) these are called 10k+
   times per IK solve.

2. **Determinism.** numpy primitives route through BLAS (Accelerate on macOS,
   OpenBLAS on Linux). The two backends produce results that differ by 1 ulp
   due to different summation order / FMA usage / vectorisation. For a 3-element
   dot product this divergence is tiny (~1e-16), but accumulated through a
   chain of 4x4 matrix multiplications inside :func:`ssik.kinematics.poe_to_dh`
   it produces last-bit-different DH parameters across platforms. That
   propagates through ``sympy.cse`` (which is value-sensitive) and produces
   byte-different rendered codegen artifacts on macOS vs Linux.

   The hand-rolled versions express the math as Python-level scalar arithmetic
   with explicit operand ordering. Python ``float`` operations are IEEE 754
   deterministic across every conforming platform, with no FMA reordering and
   no SIMD batching. The codegen pipeline becomes bit-exact across platforms,
   and snapshot tests can enforce byte equality in CI.

The functions are private (underscore-prefixed). Callers inside ``ssik.*``
import them where they need explicit determinism guarantees.
"""

from __future__ import annotations

import math

import cython
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "_cross3",
    "_dot3",
    "_mat3_vec3",
    "_mat4_mat4",
    "_norm3",
    "_se3_inv",
]


@cython.ccall
@cython.locals(
    a0=cython.double,
    a1=cython.double,
    a2=cython.double,
    b0=cython.double,
    b1=cython.double,
    b2=cython.double,
)
def _cross3(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """3-vector cross product. Deterministic, no BLAS dispatch."""
    a0 = float(a[0])
    a1 = float(a[1])
    a2 = float(a[2])
    b0 = float(b[0])
    b1 = float(b[1])
    b2 = float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


@cython.ccall
@cython.locals(
    a0=cython.double,
    a1=cython.double,
    a2=cython.double,
    b0=cython.double,
    b1=cython.double,
    b2=cython.double,
)
def _dot3(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """3-vector dot product as ``float``. Deterministic, no BLAS dispatch."""
    a0 = float(a[0])
    a1 = float(a[1])
    a2 = float(a[2])
    b0 = float(b[0])
    b1 = float(b[1])
    b2 = float(b[2])
    return a0 * b0 + a1 * b1 + a2 * b2


@cython.ccall
@cython.locals(
    a0=cython.double,
    a1=cython.double,
    a2=cython.double,
)
def _norm3(a: NDArray[np.floating]) -> float:
    """3-vector L2 norm. Deterministic, no BLAS dispatch.

    Uses :func:`math.sqrt` which is IEEE 754 correctly-rounded on every
    conforming libm (glibc, Apple, Microsoft) -- bit-identical across
    platforms, unlike vectorised ``np.sqrt`` which can dispatch to
    platform-specific SIMD implementations with different rounding.
    """
    a0 = float(a[0])
    a1 = float(a[1])
    a2 = float(a[2])
    return math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)


def _mat3_vec3(M: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """3x3 matrix times 3-vector. Deterministic, no BLAS dispatch."""
    return np.array(
        [
            M[0, 0] * v[0] + M[0, 1] * v[1] + M[0, 2] * v[2],
            M[1, 0] * v[0] + M[1, 1] * v[1] + M[1, 2] * v[2],
            M[2, 0] * v[0] + M[2, 1] * v[1] + M[2, 2] * v[2],
        ]
    )


def _mat4_mat4(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    """4x4 matrix product. Deterministic, no BLAS dispatch.

    Used in :func:`ssik.kinematics.poe_to_dh` and at every place codegen
    output must be bit-exact across platforms. Per-element scalar form
    expresses the ``A @ B`` math as 16 explicit 4-term sums.
    """
    out = np.empty((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            out[i, j] = (
                A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + A[i, 2] * B[2, j] + A[i, 3] * B[3, j]
            )
    return out


def _se3_inv(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse of an SE(3) matrix as ``[R^T | -R^T t; 0 | 1]``.

    Closed-form, no linear solve, no BLAS dispatch. Caller must ensure
    ``T[:3, :3]`` is orthogonal (otherwise this is *not* an inverse, just
    a transpose-shuffle). Used in :func:`ssik.kinematics.poe_to_dh` where
    we know ``t_dh_end`` is SE(3) by construction.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[0, 0] = R[0, 0]
    out[0, 1] = R[1, 0]
    out[0, 2] = R[2, 0]
    out[1, 0] = R[0, 1]
    out[1, 1] = R[1, 1]
    out[1, 2] = R[2, 1]
    out[2, 0] = R[0, 2]
    out[2, 1] = R[1, 2]
    out[2, 2] = R[2, 2]
    out[0, 3] = -(R[0, 0] * t[0] + R[1, 0] * t[1] + R[2, 0] * t[2])
    out[1, 3] = -(R[0, 1] * t[0] + R[1, 1] * t[1] + R[2, 1] * t[2])
    out[2, 3] = -(R[0, 2] * t[0] + R[1, 2] * t[1] + R[2, 2] * t[2])
    return out
