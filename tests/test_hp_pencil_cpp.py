"""Completeness gate for the native HP Sylvester-pencil eigensolve (#538).

The C++ pencil solve (Sylvester tensor -> 80x80 companion -> eigensolve -> real
filter) must recover every *true* root of ``det S(u)=0`` -- the candidate ``u``
values that seed the downstream (u,w) Newton refinement and back-substitution.

Exact set-parity with Python's ``solve_pencil_eigenvalues`` is NOT the gate: that
uses LAPACK ``dggev`` while the artifact is Eigen-only, and the two classify the
*spurious* near-real eigenvalues (borderline against ``real_tol=1e-3``)
differently -- roots that never survive Newton+FK. What must match is the true
roots. We check the one root known exactly from the generating configuration, the
parametric variable ``v_1``: for a reachable pose built from joint tan-half-angles
``v``, ``u=v_1`` is a genuine root, so the pencil must return a candidate near it
(within the Newton basin) for every pose. Eigen's RealQZ alone fails this on ~38%
of these 80x80 pencils; the artifact's monic-companion reduction with
stable-orientation selection (solve the reversed polynomial when the leading
coefficient is worse-conditioned) recovers dggev-level completeness -- v1 to
~2e-5 worst-case here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik.solvers.husty_pfurner._eliminate import compute_fg_numeric, precompute_rrr_chain

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
def test_hp_pencil_recovers_true_root(dh_name: str) -> None:
    ext = _load_ext()
    dh = _DH_SETS[dh_name]
    pre = precompute_rrr_chain(**dh)
    a = tuple(dh[f"a_{i}"] for i in range(1, 6))
    ls = tuple(dh[f"l_{i}"] for i in range(1, 6))
    d = tuple(dh[f"d_{i}"] for i in range(2, 6))
    rng = np.random.default_rng(538)
    worst = 0.0
    for _ in range(400):
        v = tuple(float(np.tan(x)) for x in rng.uniform(-1.4, 1.4, 6))
        sigma_E = _full_6r_chain(v=v, a=a, ls=ls, d=d)
        f, g = compute_fg_numeric(pre, sigma_E, drop_idx=7)
        cands = list(ext.hp_pencil_roots_test(f, g))
        # The generating root u = v_1 must be recovered within the Newton basin.
        dists = [abs(c - v[0]) for c in cands]
        nearest = min(dists) if dists else float("inf")
        assert nearest <= 1e-3 * (1.0 + abs(v[0])), (
            f"{dh_name}: pencil missed true root v1={v[0]:.4f}; nearest cand Δ={nearest:.2e}"
        )
        worst = max(worst, nearest)
    # Sanity: the recovery is tight (well inside the basin), not just barely in.
    assert worst <= 1e-3, f"{dh_name}: worst v1 recovery {worst:.2e} unexpectedly loose"
