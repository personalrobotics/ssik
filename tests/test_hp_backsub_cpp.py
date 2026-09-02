"""Parity gate for the native HP back-substitution (#539, Slice 3b).

`back_substitute_one` recovers the joint tan-half-angles (v_1..v_6) from one
refined (u,w): Cramer cofactor P at (u,w), then two closed-form 2R sub-chain
ZXZ-atan2 decompositions (left = Tv1, right = Tv6/Tv4). This is the subtle
Study-quaternion core, so we hold it to tight parity with Python plus a
round-trip check (feed the true (u,w) from a known joint vector, recover it).
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik.solvers.husty_pfurner._back_substitute import back_substitute_one
from ssik.solvers.husty_pfurner._eliminate import precompute_rrr_chain

from ._cpp_backend import _load_ext, cpp_available
from .test_husty_pfurner_eliminate import _full_6r_chain

pytestmark = pytest.mark.skipif(not cpp_available(), reason="native extension not built")

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


def _ang(v: np.ndarray) -> np.ndarray:
    return np.asarray(2.0 * np.arctan(v), dtype=np.float64)


@pytest.mark.parametrize("dh_name", list(_DH_SETS))
def test_hp_back_substitute(dh_name: str) -> None:
    ext = _load_ext()
    dh = _DH_SETS[dh_name]
    pre = precompute_rrr_chain(**dh)
    rpv = 1 if pre.right_parametric_var == "v_4" else 0
    a = tuple(dh[f"a_{i}"] for i in range(1, 6))
    ls = tuple(dh[f"l_{i}"] for i in range(1, 6))
    d = tuple(dh[f"d_{i}"] for i in range(2, 6))
    dh_a, dh_l, dh_d = np.array(a), np.array(ls), np.array(d)
    rng = np.random.default_rng(3901)
    for _ in range(400):
        v = tuple(float(np.tan(x)) for x in rng.uniform(-1.2, 1.2, 6))
        sigma_E = _full_6r_chain(v=v, a=a, ls=ls, d=d)
        u_true = v[0]
        w_true = v[3] if rpv == 1 else v[5]  # Tv4: w=v_4, Tv6: w=v_6
        v_cpp = np.array(
            ext.hp_back_substitute_test(
                pre.T_u, pre.T_w_pre, sigma_E, u_true, w_true, dh_a, dh_l, dh_d, rpv, 7
            )
        )
        v_py = np.array(back_substitute_one(pre, sigma_E, u_true, w_true, **dh)[0])
        # C++ must match Python's back-substitution at the same (u,w).
        assert np.max(np.abs(v_cpp - v_py)) <= 1e-9, f"{dh_name}: back_substitute parity"
        # And it must round-trip to the generating joint angles (mod 2pi).
        d_ang = (_ang(v_cpp) - _ang(np.array(v)) + np.pi) % (2 * np.pi) - np.pi
        assert np.max(np.abs(d_ang)) <= 1e-8, f"{dh_name}: round-trip to input joints"
