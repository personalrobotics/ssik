"""Gate for the native HP universal-6R artifact solve (#539, Slice 3d).

`hp_artifact_solve` is the full self-contained HP solve: bridge the POE target
into the HP frame (baked transforms), run the numeric kernel (f/g -> pencil ->
refine -> back-sub), convert tan-half-angles to POE q, POE-FK verify + lm_refine,
finalize.

Two properties are gated, matching HP's real dispatch contract:

1. **Soundness on real 6R arms** (jaco2, piper): every C++ solution FK-closes.
   This holds at every pose, including the numerically pathological general-6R
   DH those arms present (jaco2 has alpha_2 = pi; both are highly ill-conditioned
   for HP's tan-half-angle elimination). HP is *never dispatched for general 6R*
   in production -- those route to RR / ikgeo tier-2 -- so completeness there is
   not the contract, but soundness (no wrong answers) must always hold.

2. **Completeness on well-conditioned HP input** (clean synthetic Tv6 and Tv4
   DH): every Python-oracle candidate is recovered by the native kernel. This is
   HP's actual shipping path (symmetric-DH locked-7R sub-chains are
   well-conditioned). The remaining general-6R degenerate-completeness gap is the
   monic-vs-dggev eigensolve limitation tracked in #544.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik._kinbody import KinBody
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.husty_pfurner._back_substitute import solve_ik as py_solve_ik
from ssik.solvers.husty_pfurner._eliminate import precompute_rrr_chain

from ._cpp_backend import _load_ext, cpp_available
from .test_husty_pfurner_eliminate import _full_6r_chain

pytestmark = pytest.mark.skipif(not cpp_available(), reason="native extension not built")

# Two clean, well-conditioned HP DH sets that exercise both right-chain dispatch
# paths: the default Tv6 (w = v_6) and Tv4 (w = v_4, forced by a_4 = 0).
_CLEAN_DH = {
    "tv6": dict(
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
    "tv4": dict(
        a_1=0.30,
        l_1=0.40,
        d_2=0.20,
        a_2=0.50,
        l_2=-0.30,
        d_3=0.10,
        a_3=0.40,
        l_3=0.20,
        d_4=0.15,
        a_4=0.0,
        l_4=0.35,
        d_5=0.10,
        a_5=0.20,
        l_5=0.30,
    ),
}


def _hp_bake(kb: KinBody) -> dict[str, object]:
    """Bake every HpConsts field from a KinBody, mirroring the setup in
    husty_pfurner.general_6r.solve (poe_to_dh bridge + alpha~pi twist cap +
    singular-DH perturbation + precompute_rrr_chain). This is the emit-side
    geometry (Slice 3e will move it into ssik._native as shared emit/runtime
    source)."""
    from ssik.kinematics.poe_to_dh import poe_to_dh
    from ssik.solvers.husty_pfurner.general_6r import _se3_from_dh_offset

    dh = poe_to_dh(kb)
    ls = np.tan(0.5 * dh.alpha)
    ls[:5] = np.clip(ls[:5], -1.0e3, 1.0e3)  # alpha~pi twist cap, mirrors general_6r.solve
    t_z = np.eye(4)
    t_z[2, 3] = -float(dh.d[0])
    t_j6 = _se3_from_dh_offset(a=float(dh.a[5]), alpha=float(dh.alpha[5]), d=float(dh.d[5]))

    st, ep = 1e-9, 1e-3
    a1, l1, a2, l2 = float(dh.a[0]), float(ls[0]), float(dh.a[1]), float(ls[1])
    a4, l4, a5, l5 = float(dh.a[3]), float(ls[3]), float(dh.a[4]), float(ls[4])
    if not (abs(a2) > st and abs(l2) > st) and not (abs(a1) > st and abs(l1) > st):
        if abs(a2) < st:
            a2 = ep
        elif abs(l2) < st:
            l2 = ep
    right_tv6 = abs(a4) > st and abs(l4) > st
    right_tv4 = (abs(a4) < st or abs(l4) < st) and abs(a5) > st and abs(l5) > st
    if not right_tv6 and not right_tv4:
        if abs(a5) < st:
            a5 = ep
        elif abs(l5) < st:
            l5 = ep

    pre = precompute_rrr_chain(
        a_1=a1,
        l_1=l1,
        d_2=float(dh.d[1]),
        a_2=a2,
        l_2=l2,
        d_3=float(dh.d[2]),
        a_3=float(dh.a[2]),
        l_3=float(ls[2]),
        d_4=float(dh.d[3]),
        a_4=a4,
        l_4=l4,
        d_5=float(dh.d[4]),
        a_5=a5,
        l_5=l5,
    )
    return dict(
        t_u=pre.T_u,
        t_w_pre=pre.T_w_pre,
        dh_a=np.array([a1, a2, float(dh.a[2]), a4, a5]),
        dh_l=np.array([l1, l2, float(ls[2]), l4, l5]),
        dh_d=np.array([float(dh.d[1]), float(dh.d[2]), float(dh.d[3]), float(dh.d[4])]),
        theta_offset=np.asarray(dh.theta_offset, dtype=np.float64),
        t_pre_inv=np.linalg.inv(dh.t_pre),
        t_post_inv=np.linalg.inv(dh.t_post),
        t_z_neg_d1=t_z,
        t_joint6_offset_inv=np.linalg.inv(t_j6),
        right_parametric_var=1 if pre.right_parametric_var == "v_4" else 0,
        drop_idx=7,
    )


def _consts(kb: KinBody) -> dict[str, object]:
    js = kb.joints  # type: ignore[attr-defined]
    return dict(
        axes=np.array([j.axis for j in js], dtype=np.float64),
        t_left=np.array([j.T_left for j in js], dtype=np.float64),
        t_right=np.array([j.T_right for j in js], dtype=np.float64),
        types=np.array([0 if j.joint_type == "revolute" else 1 for j in js], dtype=np.int32),
    )


@pytest.mark.parametrize("arm", ["jaco2_ik", "piper_ik"])
def test_hp_artifact_solve_is_sound(arm: str) -> None:
    """Every native HP artifact-solve solution FK-closes -- no wrong answers,
    even on the numerically pathological general-6R DH these arms present."""
    import importlib

    ext = _load_ext()
    kb = importlib.import_module(f"ssik.prebuilt.{arm}")._KB
    if len(kb.joints) != 6:
        pytest.skip("6R only")
    bake = _hp_bake(kb)
    con = _consts(kb)
    rng = np.random.default_rng(3903)
    ranges = [j.limits if j.limits else (-np.pi, np.pi) for j in kb.joints]
    worst_fk = 0.0
    n_sol = 0
    for _ in range(150):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        T = np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)
        qs, _resids = ext.hp_artifact_solve_test(
            con["axes"],
            con["t_left"],
            con["t_right"],
            con["types"],
            bake["t_u"],
            bake["t_w_pre"],
            bake["dh_a"],
            bake["dh_l"],
            bake["dh_d"],
            bake["theta_offset"],
            bake["t_pre_inv"],
            bake["t_post_inv"],
            bake["t_z_neg_d1"],
            bake["t_joint6_offset_inv"],
            bake["right_parametric_var"],
            bake["drop_idx"],
            T,
            True,
        )
        for row in qs:
            c = np.asarray(row)
            fk = float(np.linalg.norm(poe_forward_kinematics(kb, c) - T))
            worst_fk = max(worst_fk, fk)
            n_sol += 1
            assert fk <= 1e-6, f"{arm}: C++ solution FK residual {fk:.2e}"
    assert n_sol > 0, f"{arm}: native HP produced no solutions"
    assert worst_fk <= 1e-6


@pytest.mark.parametrize("dh_name", list(_CLEAN_DH))
def test_hp_kernel_complete_on_wellconditioned_dh(dh_name: str) -> None:
    """On well-conditioned HP DH (the shipping locked-7R contract), the native
    kernel recovers every Python-oracle (v_1..v_6) candidate."""
    ext = _load_ext()
    dh: dict[str, float] = _CLEAN_DH[dh_name]
    pre = precompute_rrr_chain(**dh)
    a = tuple(dh[f"a_{i}"] for i in range(1, 6))
    ls = tuple(dh[f"l_{i}"] for i in range(1, 6))
    d = tuple(dh[f"d_{i}"] for i in range(2, 6))
    rpv = 1 if pre.right_parametric_var == "v_4" else 0
    dh_a, dh_l, dh_d = np.array(a), np.array(ls), np.array(d)
    rng = np.random.default_rng(42)
    for _ in range(80):
        v = tuple(float(np.tan(x)) for x in rng.uniform(-1.4, 1.4, 6))
        sigma_E = np.asarray(_full_6r_chain(v=v, a=a, ls=ls, d=d), dtype=np.float64)
        cpp = np.asarray(
            ext.hp_solve_ik_test(pre.T_u, pre.T_w_pre, sigma_E, dh_a, dh_l, dh_d, rpv)
        ).reshape(-1, 6)
        assert len(cpp), f"{dh_name}: native returned no candidates"
        # True root recovered.
        tv = np.array(v)
        assert np.min(np.max(np.abs(cpp - tv), axis=1)) < 1e-4, (
            f"{dh_name}: native missed the true root"
        )
        # Every Python-oracle candidate recovered (oracle subset of C++).
        oracle = py_solve_ik(
            pre,
            sigma_E,
            a_1=a[0],
            l_1=ls[0],
            d_2=d[0],
            a_2=a[1],
            l_2=ls[1],
            d_3=d[1],
            a_3=a[2],
            l_3=ls[2],
            d_4=d[2],
            a_4=a[3],
            l_4=ls[3],
            d_5=d[3],
            a_5=a[4],
            l_5=ls[4],
        )
        for pt in oracle:
            pt = np.asarray(pt)
            assert np.min(np.max(np.abs(cpp - pt), axis=1)) < 1e-4, (
                f"{dh_name}: native missed a Python-oracle candidate {np.round(pt, 4)}"
            )
