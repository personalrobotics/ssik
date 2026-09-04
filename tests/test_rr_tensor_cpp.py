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


@pytest.mark.parametrize("arm", ["jaco2_ik", "piper_ik"])
def test_rr_tensor_full_solve_matches_python(arm: str) -> None:
    """The whole native general_6r solve via the baked tensor recovers every
    Python general_6r solution (relative-completeness + soundness)."""
    from ssik.kinematics.poe_fk import poe_forward_kinematics

    ext = _load_ext()
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    if len(kb.joints) != 6:
        pytest.skip("6R only")
    g = rr_native_geometry(kb)
    axes = np.array([j.axis for j in kb.joints], dtype=np.float64)
    tl = np.array([j.T_left for j in kb.joints], dtype=np.float64)
    tr = np.array([j.T_right for j in kb.joints], dtype=np.float64)
    ty = np.array([0 if j.joint_type == "revolute" else 1 for j in kb.joints], dtype=np.int32)
    po_r, po_c, po_m, po_co = g["po_coo"]
    q_r, q_c, q_m, q_co = g["q_coo"]
    po_rc = np.stack([po_r, po_c], axis=1).astype(np.int32)
    q_rc = np.stack([q_r, q_c], axis=1).astype(np.int32)

    def native(t: np.ndarray) -> list[np.ndarray]:
        (qs,) = ext.general_6r_tensor_solve(
            axes,
            tl,
            tr,
            ty,
            g["alpha"],
            g["a"],
            g["d"],
            g["theta_offset"],
            g["t_pre_inv"],
            g["t_post_inv"],
            g["linearity_joint"],
            g["left_bilinear"],
            g["right_bilinear"],
            g["drop_joint"],
            g["p_sin"],
            g["p_cos"],
            g["mono_factors"],
            po_rc,
            po_m,
            po_co,
            q_rc,
            q_m,
            q_co,
            t,
        )
        return [np.asarray(r) for r in np.asarray(qs).reshape(-1, 6)]

    def wrap_close(x: np.ndarray, y: np.ndarray) -> bool:
        return bool(np.max(np.abs((x - y + np.pi) % (2 * np.pi) - np.pi)) < 1e-4)

    rng = np.random.default_rng(7)
    ranges = [j.limits if j.limits else (-np.pi, np.pi) for j in kb.joints]
    worst_fk = 0.0
    for _ in range(60):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        T = np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)
        nsols = native(T)
        for s in nsols:
            worst_fk = max(worst_fk, float(np.linalg.norm(poe_forward_kinematics(kb, s) - T)))
        oracle = mod.solve(T)  # Python general_6r (this family isn't native-wired yet)
        for o in oracle:
            assert any(wrap_close(o.q, s) for s in nsols), f"{arm}: native RR-tensor missed a sol"
    assert worst_fk <= 1e-6, f"{arm}: native RR-tensor worst FK {worst_fk:.2e}"


@pytest.mark.parametrize("arm", ["jaco2_ik", "piper_ik"])
def test_rr_tensor_production_path_matches_python(arm: str, tmp_path) -> None:
    """The full production runtime path -- bake to a sidecar .npz, load it, then
    ``try_native_solve`` with limits -- recovers every in-limits Python solution
    (relative-completeness + soundness). This exercises the exact code the
    regenerated general_6r artifacts will run under ``native=True``."""
    from ssik._native import bake_rr_tensor_npz, load_rr_native_geometry, try_native_solve
    from ssik.kinematics.poe_fk import poe_forward_kinematics

    if not cpp_available():
        pytest.skip("native extension not built")
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    if len(kb.joints) != 6:
        pytest.skip("6R only")

    npz = str(tmp_path / f"{arm}_rr.npz")
    bake_rr_tensor_npz(kb, npz)
    g = load_rr_native_geometry(npz)

    def wrap_close(x: np.ndarray, y: np.ndarray) -> bool:
        return bool(np.max(np.abs((x - y + np.pi) % (2 * np.pi) - np.pi)) < 1e-4)

    rng = np.random.default_rng(11)
    ranges = [j.limits if j.limits else (-np.pi, np.pi) for j in kb.joints]
    worst_fk = 0.0
    for _ in range(60):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        T = np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)
        nsols = try_native_solve("ikgeo.general_6r", kb, T, respect_limits=True, rr_geometry=g)
        assert nsols is not None, f"{arm}: native general_6r path returned None"
        for s in nsols:
            worst_fk = max(worst_fk, float(np.linalg.norm(poe_forward_kinematics(kb, s.q) - T)))
        oracle = mod.solve(T, respect_limits=True)
        for o in oracle:
            assert any(wrap_close(o.q, s.q) for s in nsols), (
                f"{arm}: production native path missed an in-limits sol"
            )
    assert worst_fk <= 1e-6, f"{arm}: production native path worst FK {worst_fk:.2e}"
