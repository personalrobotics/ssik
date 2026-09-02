"""Completeness gate for the native HP solve_ik kernel (#539, Slice 3c).

`solve_ik` ties the pipeline together: refined (u,w) pairs -> back-substitution
-> the (v_1..v_6) tan-half-angle candidates. This is the full HP numeric kernel
from sigma_E (before the target->sigma_E bridge and POE-FK verify, which are the
artifact-solve wrapper). Gate: for a reachable pose built from a known joint
vector v, some candidate must recover v (angle-wise, mod 2pi). Exact set-parity
isn't the gate -- spurious (u,w) yield spurious candidates that FK-filter out
downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize("dh_name", list(_DH_SETS))
def test_hp_solve_ik_recovers_input(dh_name: str) -> None:
    ext = _load_ext()
    dh = _DH_SETS[dh_name]
    pre = precompute_rrr_chain(**dh)
    rpv = 1 if pre.right_parametric_var == "v_4" else 0
    a = tuple(dh[f"a_{i}"] for i in range(1, 6))
    ls = tuple(dh[f"l_{i}"] for i in range(1, 6))
    d = tuple(dh[f"d_{i}"] for i in range(2, 6))
    dh_a, dh_l, dh_d = np.array(a), np.array(ls), np.array(d)
    rng = np.random.default_rng(3902)
    for _ in range(400):
        v = tuple(float(np.tan(x)) for x in rng.uniform(-1.2, 1.2, 6))
        sigma_E = _full_6r_chain(v=v, a=a, ls=ls, d=d)
        cands = ext.hp_solve_ik_test(pre.T_u, pre.T_w_pre, sigma_E, dh_a, dh_l, dh_d, rpv)
        v_in = 2.0 * np.arctan(np.asarray(v))
        found = any(
            np.max(np.abs((2.0 * np.arctan(np.asarray(c)) - v_in + np.pi) % (2 * np.pi) - np.pi))
            < 1e-6
            for c in cands
        )
        assert found, f"{dh_name}: solve_ik did not recover the generating joint vector"
