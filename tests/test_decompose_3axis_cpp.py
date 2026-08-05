"""C++ generalized-Euler decomposition vs the Python oracle (#354).

``decompose_3axis`` (R -> up to two (a,b,c) with R = Rot(n1,a)Rot(n2,b)Rot(n3,c)
for arbitrary axes) is the shoulder/wrist extraction the general SRS solve uses
to cover non-ZYZ concurrent-axis 7R arms. This fuzzes random rotations against
arbitrary axis triples (generic, ZYZ-symmetric, and gimbal-boundary cases) and
checks the C++ port matches Python to machine precision, per branch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ssik.kinematics._generalized_euler import decompose_3axis
from tests._cpp_backend import _load_ext, cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")

_N = 4000


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return np.asarray(q, dtype=np.float64)


def _rand_axis(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    return np.asarray(v / np.linalg.norm(v), dtype=np.float64)


def _match(py: list[Any], cpp: list[Any], tol: float = 1e-11) -> bool:
    if len(py) != len(cpp):
        return False
    # Branch order is identical (both iterate phi+delta then phi-delta).
    return all(
        np.allclose(np.asarray(p), np.asarray(c), atol=tol) for p, c in zip(py, cpp, strict=True)
    )


def test_decompose_3axis_matches_oracle() -> None:
    ext = _load_ext()
    rng = np.random.default_rng(0)
    ez, ey = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    two = single = 0
    mismatch = 0
    for k in range(_N):
        R = _rand_rot(rng)
        # Mix axis triples: generic, classical ZYZ (n1==n3, gimbal-prone), and a
        # near-collinear-middle case to exercise the branch-boundary + degenerate
        # guards.
        r = k % 4
        if r == 0:
            n1, n2, n3 = _rand_axis(rng), _rand_axis(rng), _rand_axis(rng)
        elif r == 1:
            n1, n2, n3 = ez, ey, ez  # ZYZ
        elif r == 2:
            n1, n2, n3 = ez, _rand_axis(rng), ez
        else:
            a = _rand_axis(rng)
            n1, n2, n3 = a, _rand_axis(rng), a  # symmetric outer axes

        py = [tuple(t) for t in decompose_3axis(R, n1, n2, n3)]
        cpp = ext.decompose_3axis_test(R, n1, n2, n3)
        if not _match(py, cpp):
            mismatch += 1
        two += len(py) == 2
        single += len(py) == 1

    assert mismatch == 0, f"{mismatch}/{_N} decompose_3axis mismatches vs Python"
    assert two > 500, f"only {two} two-branch cases -- fuzz not exercising the generic path"
    assert single > 50, f"only {single} single/boundary cases -- gimbal path under-exercised"
