"""Parity gate for the native RR numeric-tensor evaluator (#555).

The RR elimination coefficients (P_sin/P_cos constant, P_one/Q degree-3 in the 12
target entries) are baked as a sparse numeric tensor by
:func:`ssik._native.rr_native_geometry` and evaluated by the generic C++
``rr_eval_coeffs`` -- the HP-style replacement for the per-arm emitted rr_coeffs()
fn, so a single shipped ext covers every RR arm. This asserts the C++ tensor eval
matches the Python lambdified RR reference to machine precision, on real arms.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik._native import rr_native_geometry
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import _cached_best_leftvar, _derive_pq_for_arm

from ._cpp_backend import _load_ext, cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="native extension not built")


@pytest.mark.parametrize("arm", ["jaco2_ik", "piper_ik", "ur5_ik", "puma560_ik"])
def test_rr_tensor_matches_lambdified(arm: str) -> None:
    ext = _load_ext()
    kb = importlib.import_module(f"ssik.prebuilt.{arm}")._KB
    if len(kb.joints) != 6:
        pytest.skip("6R only")
    g = rr_native_geometry(kb)

    # Python lambdified reference (the current emitted-fn source of truth).
    dh = poe_to_dh(kb)
    alpha, a, d = dh.to_dh_tuple()
    at, bt, dt = tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist())
    lin = int(_cached_best_leftvar(at, bt, dt))
    *fns, _meta = _derive_pq_for_arm(at, bt, dt, linearity_joint=lin)

    po_r, po_c, po_m, po_co = g["po_coo"]
    q_r, q_c, q_m, q_co = g["q_coo"]
    po_rc = np.stack([po_r, po_c], axis=1).astype(np.int32)
    q_rc = np.stack([q_r, q_c], axis=1).astype(np.int32)

    def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
        # Uniform-ish random rotation via QR of a Gaussian matrix (sign-fixed).
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return np.asarray(q, dtype=np.float64)

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        T = np.eye(4)
        T[:3, :3] = _rand_rotation(rng)
        T[:3, 3] = rng.normal(size=3)
        t12 = np.ascontiguousarray(T[:3, :].reshape(12), dtype=np.float64)
        got = ext.rr_eval_coeffs_test(
            g["p_sin"], g["p_cos"], g["mono_factors"], po_rc, po_m, po_co, q_rc, q_m, q_co, t12
        )
        ref = [np.asarray(f(*t12)) for f in fns]  # p_sin, p_cos, p_one, q
        for c, r in zip(got, ref, strict=True):
            worst = max(worst, float(np.max(np.abs(np.asarray(c) - r))))
    assert worst < 1e-9, f"{arm}: C++ rr_eval_coeffs vs lambdified worst diff {worst:.2e}"
