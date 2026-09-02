"""Completeness gate for the native HP (u,w) refinement (#539, Slice 3a).

`eliminate_uw_pairs` runs, per Cramer drop index, the f/g stage -> Sylvester
pencil -> a per-root w seed (`_initial_w_for`) -> a 2x2 Newton on [f;g]=0
(`_refine_uw_inline`) -> a 2-D cluster-merge, producing the refined (u,w) pairs
that back-substitution consumes. As with the pencil (test_hp_pencil_cpp), exact
set-parity with Python isn't the gate -- LAPACK vs Eigen produce different
*spurious* (u,w) that never survive back-substitution + FK. What must hold is that
every true root is recovered: for a reachable pose built from joint tan-half
angles v, u=v_1 is a genuine root, so some refined pair must have u ~ v_1.
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
def test_hp_uw_recovers_true_root(dh_name: str) -> None:
    ext = _load_ext()
    dh = _DH_SETS[dh_name]
    pre = precompute_rrr_chain(**dh)
    a = tuple(dh[f"a_{i}"] for i in range(1, 6))
    ls = tuple(dh[f"l_{i}"] for i in range(1, 6))
    d = tuple(dh[f"d_{i}"] for i in range(2, 6))
    rng = np.random.default_rng(539)
    for _ in range(400):
        v = tuple(float(np.tan(x)) for x in rng.uniform(-1.4, 1.4, 6))
        sigma_E = _full_6r_chain(v=v, a=a, ls=ls, d=d)
        pairs = list(ext.hp_eliminate_uw_pairs_test(pre.T_u, pre.T_w_pre, sigma_E, 1e-3))
        assert any(abs(uw[0] - v[0]) <= 1e-4 * (1.0 + abs(v[0])) for uw in pairs), (
            f"{dh_name}: no refined (u,w) pair recovered the true root v1={v[0]:.4f}"
        )
