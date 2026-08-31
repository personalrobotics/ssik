"""Parity gate for the native HP f/g kernel (#537).

The C++ ``hp_compute_fg`` (Study-DQ sigma_E injection + Cramer 5x4
evaluation-interpolation + 2-D convolution) must reproduce the Python
``_eliminate.compute_fg_numeric`` bit-for-bit on the shared numeric tier. The
per-arm inputs are the baked ``T_u``/``T_w_pre`` tensors from
``precompute_rrr_chain``; only ``sigma_E`` (the target's Study dual quaternion)
enters at call time, so we sweep random reachable-pose sigma_E and every
``drop_idx``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik.solvers.husty_pfurner._eliminate import compute_fg_numeric, precompute_rrr_chain
from ssik.solvers.husty_pfurner._study import dq_from_se3

from ._cpp_backend import _load_ext, cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="native extension not built")

# Non-degenerate 6R DH sets (from the eliminate test suite), each giving a valid
# precompute the Cramer 7x7 is well-conditioned on.
_DH_SETS = {
    "baseline": dict(
        a_1=0.30,
        l_1=0.40,
        d_2=0.20,
        a_2=0.50,
        l_2=-0.30,
        d_3=0.10,
        a_3=0.40,
        l_3=0.20,
        d_4=0.15,
        a_4=0.25,
        l_4=-0.40,
        d_5=0.10,
        a_5=0.20,
        l_5=0.30,
    ),
    "large_alpha": dict(
        a_1=0.20,
        l_1=1.50,
        d_2=0.30,
        a_2=0.40,
        l_2=-1.20,
        d_3=0.20,
        a_3=0.30,
        l_3=0.80,
        d_4=0.10,
        a_4=0.20,
        l_4=-0.90,
        d_5=0.20,
        a_5=0.30,
        l_5=1.10,
    ),
}


def _random_se3(rng: np.random.Generator) -> np.ndarray:
    """A random SE(3) pose (unit-quaternion rotation + bounded translation)."""
    q = rng.standard_normal(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.uniform(-0.8, 0.8, 3)
    return T


def _assert_coef_close(cpp: np.ndarray, py: np.ndarray, name: str) -> None:
    """Scale-relative coefficient parity. f/g are bivariate-polynomial
    coefficient blocks whose entries span many orders of magnitude (large-alpha
    arms reach O(1e7-1e9)), so per-element rtol is meaningless on the near-zero
    coefficients. The meaningful metric is the diff relative to the block's own
    scale; the C++ Cramer/interp/convolve matches Python to ~1e-12 there (the
    residual is LAPACK-vs-Eigen LU ordering, harmless downstream: the pencil is
    equilibrated and joints are LM-polished to machine precision)."""
    scale = max(float(np.max(np.abs(py))), 1.0)
    worst = float(np.max(np.abs(cpp - py)))
    assert worst <= 1e-9 * scale, f"{name}: worst |Δ|={worst:.2e}, scale={scale:.2e}"


@pytest.mark.parametrize("dh_name", list(_DH_SETS))
@pytest.mark.parametrize("drop_idx", range(8))
def test_hp_fg_matches_python(dh_name: str, drop_idx: int) -> None:
    ext = _load_ext()
    pre = precompute_rrr_chain(**_DH_SETS[dh_name])
    rng = np.random.default_rng(20260831 + drop_idx)
    for _ in range(12):
        sigma_E = dq_from_se3(_random_se3(rng))
        f_py, g_py = compute_fg_numeric(pre, sigma_E, drop_idx=drop_idx)
        f_cpp, g_cpp = ext.hp_compute_fg_test(pre.T_u, pre.T_w_pre, sigma_E, drop_idx)
        _assert_coef_close(f_cpp, f_py, "f")
        _assert_coef_close(g_cpp, g_py, "g")
