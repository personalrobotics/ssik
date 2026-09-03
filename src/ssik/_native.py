"""Opt-in native (C++) solver dispatch for the artifact ``solve()`` (#507).

Best-effort: :func:`try_native_solve` returns a ``list[Solution]`` when the
shipped native extension (``ssik._ssik_native``, #506) can handle the arm's
solver family, else ``None`` so the caller silently falls back to the Python
path. Native is available only where the wheel bundled the extension (Linux +
macOS) and only for solver families with a native implementation
(``ikgeo.three_parallel`` today).

The generated artifacts call this from ``solve(..., native=True)``; passing
``native=True`` never fails for unavailability -- it is a performance hint.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ssik.core.solution import Solution

# Solver families with a native implementation in _ssik_native. The 6R geometric
# families run the full artifact via try_native_solve; seven_r.srs runs the full
# artifact via try_native_srs_solve (seeded-track + canonical/general core +
# finalize + in-limits fallback, all native).
_NATIVE_SOLVERS = frozenset({"ikgeo.three_parallel", "ikgeo.spherical_two_parallel", "seven_r.srs"})

_ext: Any = None
_ext_tried = False


def _load_ext() -> Any:
    global _ext, _ext_tried
    if _ext_tried:
        return _ext
    _ext_tried = True
    try:
        from ssik import _ssik_native  # type: ignore[attr-defined]

        _ext = _ssik_native
    except ImportError:
        _ext = None
    return _ext


def native_available() -> bool:
    """True when the native extension is importable (shipped for this platform)."""
    return _load_ext() is not None


# Marshalled per-KinBody constants, cached: each artifact has one long-lived _KB,
# so keying on id(kb) is stable and avoids re-marshalling on every solve (which
# would erode the native speedup).
_consts_cache: dict[int, tuple[Any, ...]] = {}


def _consts(solver_name: str, kb: Any) -> tuple[Any, ...]:
    cached = _consts_cache.get(id(kb))
    if cached is not None:
        return cached
    # Per-family geometry preprocessing: spherical_two_parallel needs the wrist
    # gauge canonicalized (the artifact bakes it at build time; the raw _KB lacks
    # it -- 15/18 arms). Canonicalization is FK-identical, so the returned q are
    # physical joint values and the joint limits are unchanged. Done here in
    # Python (cached per-arm), so no preprocessing is ported to C++.
    geom = kb
    if solver_name == "ikgeo.spherical_two_parallel":
        from ssik._kinbody import canonicalize_spherical_wrist
        from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY

        geom = canonicalize_spherical_wrist(kb, DEFAULT_TOLERANCE_POLICY)
    gj = geom.joints
    lj = kb.joints  # limits from the original (physical) joints
    marshalled = (
        np.array([j.axis for j in gj], dtype=np.float64),
        np.array([j.T_left for j in gj], dtype=np.float64),
        np.array([j.T_right for j in gj], dtype=np.float64),
        np.array([0 if j.joint_type == "revolute" else 1 for j in gj], dtype=np.int32),
        np.array([j.limits[0] if j.limits else 0.0 for j in lj], dtype=np.float64),
        np.array([j.limits[1] if j.limits else 0.0 for j in lj], dtype=np.float64),
        np.array([1 if j.limits else 0 for j in lj], dtype=np.int32),
    )
    _consts_cache[id(kb)] = marshalled
    return marshalled


def try_native_solve(
    solver_name: str,
    kb: Any,
    t_target: NDArray[np.float64],
    *,
    respect_limits: bool = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    allow_rescue: bool = True,
    refinement_max_iters: int = 15,
) -> list[Solution] | None:
    """Native artifact solve for a supported family, or ``None`` to fall back.

    Mirrors the ``<arm>_ik.solve()`` contract (limits -> seed -> truncate, force
    refinement); parity with the Python artifact is validated in
    ``tests/test_native_dispatch.py`` + ``tests/test_three_parallel_artifact.py``.
    """
    if solver_name not in _NATIVE_SOLVERS:
        return None
    ext = _load_ext()
    if ext is None:
        return None
    if len(kb.joints) != 6:
        return None

    axes, t_left, t_right, types, lo, hi, has_limits = _consts(solver_name, kb)
    has_seed = q_seed is not None
    seed_arr = (
        np.asarray(q_seed, dtype=np.float64) if has_seed else np.zeros(len(kb.joints), np.float64)
    )
    qs, resids, refine = ext.native_artifact_solve(
        solver_name,
        axes,
        t_left,
        t_right,
        types,
        lo,
        hi,
        has_limits,
        np.asarray(t_target, dtype=np.float64),
        respect_limits,
        has_seed,
        seed_arr,
        seed_metric,
        seed_tolerance is not None,
        seed_tolerance if seed_tolerance is not None else 0.0,
        max_solutions if max_solutions is not None else -1,
        allow_rescue,
        refinement_max_iters,
    )
    return [
        Solution(
            q=np.asarray(qs[i], dtype=np.float64),
            fk_residual=float(resids[i]),
            refinement_used="lm" if int(refine[i]) == 1 else "none",
        )
        for i in range(len(qs))
    ]


# Per-KinBody native-SRS args (baked geometry + marshalled JointConsts) or None.
# Geometry-only, so cache per-arm (keyed on the long-lived artifact _KB's id).
_srs_cache: dict[int, dict[str, Any] | None] = {}


def srs_native_geometry(kb: Any, policy: Any = None) -> dict[str, Any] | None:
    """Baked SRS geometry (the SrsConsts fields + the canonical/general dispatch
    flag) for ANY concurrent-axis SRS 7R arm, or None when the arm is not
    SRS-class. Single source of truth shared by the runtime native path
    (:func:`_srs_native_args` / :func:`try_native_srs_solve`) AND the C++
    emit (``scripts/cpp_emit._srs_bake``), so the pip ``native=True`` backend and
    the self-contained artifact always cover exactly the same arms -- adding an
    SRS arm enrols it in both automatically.

    ``general_path`` mirrors the Python ``use_canonical`` dispatch (reach_slack
    == 0): canonical-ZYZ + offset-free wrist -> the canonical fast-path core,
    everything else (non-ZYZ shoulder/wrist, laterally-offset wrist) -> the
    general Davenport core (#354). Both are native.

    ``policy`` defaults to the strict tolerance policy; :func:`srs_polished_native_
    geometry` passes a relaxed one (axis_intersect = max_drift) so the approximate-
    SRS arms classify.
    """
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.solvers.seven_r.srs import (  # type: ignore[attr-defined]
        _arm_constants,
        _classify_srs_7r_geometric,
    )

    _POL = policy if policy is not None else DEFAULT_TOLERANCE_POLICY

    cls = _classify_srs_7r_geometric(kb, _POL)
    if cls is None or len(kb.joints) != 7:
        return None
    l_se, l_ew, ee_offset, origins = _arm_constants(kb, cls)
    j = kb.joints
    upper = origins[cls.elbow_index] - cls.shoulder_pivot
    u_home = upper / np.linalg.norm(upper)
    ez, ey = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    canonical = (
        np.allclose(j[0].axis, ez)
        and np.allclose(j[1].axis, ey)
        and np.allclose(u_home, ez)
        and np.allclose(j[4].axis, ez)
        and np.allclose(j[5].axis, ey)
        and np.allclose(j[6].axis, ez)
    )
    offset_free = np.allclose(origins[5], cls.wrist_pivot, atol=_POL.axis_intersect)
    return {
        "l_se": float(l_se),
        "l_ew": float(l_ew),
        "ee_offset_local": np.asarray(ee_offset, dtype=np.float64),
        "shoulder_pivot": np.asarray(cls.shoulder_pivot, dtype=np.float64),
        "r_post_wrist": np.asarray(j[6].T_right[:3, :3], dtype=np.float64),
        "elbow_index": int(cls.elbow_index),
        "upper_home": np.asarray(upper, dtype=np.float64),
        "forearm_home": np.asarray(cls.wrist_pivot - origins[cls.elbow_index], dtype=np.float64),
        "general_path": not (canonical and offset_free),
    }


# srs_polished refusal gate (ssik.solvers.seven_r.srs_polished._DEFAULT_MAX_DRIFT_M).
_SRS_POLISHED_MAX_DRIFT_M = 0.04


def srs_polished_native_geometry(kb: Any) -> dict[str, Any] | None:
    """Baked SrsConsts for an approximate-SRS arm (seven_r.srs_polished: gen3,
    j2s7s300, rm75, yumi L/R), or None when not approximately-SRS. Same geometry
    as :func:`srs_native_geometry` but classified under a RELAXED policy
    (axis_intersect = max_drift), mirroring srs_polished.solve's relaxed_policy so
    the small-drift pivots pass the concurrency gate. The C++ srs_polished artifact
    then LM-polishes the resulting cm-off algebraic candidates against the true FK."""
    from dataclasses import replace

    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY

    relaxed = replace(
        DEFAULT_TOLERANCE_POLICY,
        axis_intersect=max(_SRS_POLISHED_MAX_DRIFT_M, DEFAULT_TOLERANCE_POLICY.axis_intersect),
    )
    return srs_native_geometry(kb, policy=relaxed)


def spherical_shoulder_native_geometry(kb: Any, *, polished: bool) -> dict[str, Any] | None:
    """Baked (3,48) affine coefficients for a spherical-shoulder + offset-wrist 7R
    arm (franka/fr3 exact; xarm7/gen72 approximate), or None when the arm isn't in
    the class. Delegates the gate + the reversed-lock-6 coefficient bake to
    ssik.solvers.seven_r.spherical_shoulder (the emit-time compiler step; the
    reverse-chain machinery stays Python-only). ``polished`` selects the drift
    gate (approximate arms) vs the exact gate."""
    from ssik.solvers.seven_r import spherical_shoulder as sh

    if len(kb.joints) != 7:
        return None
    if polished:
        from ssik.solvers.seven_r.spherical_shoulder_polished import (
            is_approximately_spherical_shoulder_7r,
        )

        ok = is_approximately_spherical_shoulder_7r(kb)
    else:
        ok = sh.is_spherical_shoulder_7r(kb)
    if not ok:
        return None
    return {"coef": np.asarray(sh._bake(kb), dtype=np.float64)}  # (3, 48)


def hp_native_geometry(kb: Any) -> dict[str, Any]:
    """Baked Husty-Pfurner ``HpConsts`` fields for a 6R KinBody (any 6R works; HP
    is universal). Single source of truth shared by the native HP artifact solve
    and the C++ emit (``scripts/cpp_emit._hp_bake``). Mirrors the setup in
    :func:`ssik.solvers.husty_pfurner.general_6r.solve`: the ``poe_to_dh`` bridge,
    the ``alpha ~ pi`` twist cap, the singular-DH perturbation + Tv6/Tv4 dispatch,
    and ``precompute_rrr_chain``.

    HP is dispatched for the (well-conditioned, symmetric-DH) locked-7R sub-chains,
    so this is called per locked 6R sub-chain when emitting a jointlock arm whose
    inner solver is HP (kassow, #491).
    """
    from ssik.kinematics.poe_to_dh import poe_to_dh
    from ssik.solvers.husty_pfurner._eliminate import precompute_rrr_chain
    from ssik.solvers.husty_pfurner.general_6r import _se3_from_dh_offset

    if len(kb.joints) != 6:
        raise ValueError(f"hp_native_geometry requires a 6R chain, got {len(kb.joints)}")

    dh = poe_to_dh(kb)
    ls = np.tan(0.5 * dh.alpha)
    ls[:5] = np.clip(ls[:5], -1.0e3, 1.0e3)  # alpha~pi twist cap (mirrors general_6r.solve)
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
    return {
        "t_u": np.asarray(pre.T_u, dtype=np.float64),
        "t_w_pre": np.asarray(pre.T_w_pre, dtype=np.float64),
        "dh_a": np.array([a1, a2, float(dh.a[2]), a4, a5], dtype=np.float64),
        "dh_l": np.array([l1, l2, float(ls[2]), l4, l5], dtype=np.float64),
        "dh_d": np.array(
            [float(dh.d[1]), float(dh.d[2]), float(dh.d[3]), float(dh.d[4])], dtype=np.float64
        ),
        "theta_offset": np.asarray(dh.theta_offset, dtype=np.float64),
        "t_pre_inv": np.linalg.inv(dh.t_pre),
        "t_post_inv": np.linalg.inv(dh.t_post),
        "t_z_neg_d1": t_z,
        "t_joint6_offset_inv": np.linalg.inv(t_j6),
        "right_parametric_var": 1 if pre.right_parametric_var == "v_4" else 0,
        "drop_idx": 7,
    }


def _srs_native_args(kb: Any) -> dict[str, Any] | None:
    """Cached :func:`srs_native_geometry` + the marshalled JointConsts arrays for
    the binding, or None when the arm isn't SRS-class."""
    if id(kb) in _srs_cache:
        return _srs_cache[id(kb)]
    geom = srs_native_geometry(kb)
    result: dict[str, Any] | None = None
    if geom is not None:
        j = kb.joints
        result = {
            "axes": np.array([jt.axis for jt in j], dtype=np.float64),
            "t_left": np.array([jt.T_left for jt in j], dtype=np.float64),
            "t_right": np.array([jt.T_right for jt in j], dtype=np.float64),
            "types": np.array(
                [0 if jt.joint_type == "revolute" else 1 for jt in j], dtype=np.int32
            ),
            # Joint limits for the full-artifact native path (finalize's box).
            "lo": np.array([jt.limits[0] if jt.limits else 0.0 for jt in j], dtype=np.float64),
            "hi": np.array([jt.limits[1] if jt.limits else 0.0 for jt in j], dtype=np.float64),
            "has_limits": np.array([1 if jt.limits else 0 for jt in j], dtype=np.int32),
            **geom,
        }
    _srs_cache[id(kb)] = result
    return result


def try_native_srs_solve(
    solver_name: str,
    kb: Any,
    t_target: NDArray[np.float64],
    *,
    respect_limits: bool = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    allow_rescue: bool = True,
    refinement_max_iters: int = 15,
) -> list[Solution] | None:
    """FULL native SRS artifact solve (the whole ``<arm>.solve()`` contract in
    C++), or ``None`` to fall back. Runs the entire pipeline natively via
    ``srs_artifact_solve`` -- seeded-track fast path, canonical/general core,
    finalize (limits -> seed -> truncate), and the #359 in-limits fallback -- so
    the Python postprocess (which dominated the per-call time) is skipped.

    The T-perturbation rescue is omitted (proven dormant for SRS + guarded), same
    as the 6R native path. Parity with the Python solve is validated across the
    full contract in tests/test_srs_artifact_cpp.py + test_srs_general_cpp.py.
    """
    if solver_name != "seven_r.srs":
        return None
    ext = _load_ext()
    if ext is None:
        return None
    a = _srs_native_args(kb)
    if a is None:
        return None
    has_seed = q_seed is not None
    seed_arr = np.asarray(q_seed, dtype=np.float64) if has_seed else np.zeros(7, np.float64)
    qs, resids, refine = ext.srs_artifact_solve(
        a["axes"],
        a["t_left"],
        a["t_right"],
        a["types"],
        a["l_se"],
        a["l_ew"],
        a["ee_offset_local"],
        a["shoulder_pivot"],
        a["r_post_wrist"],
        a["elbow_index"],
        a["upper_home"],
        a["forearm_home"],
        a["lo"],
        a["hi"],
        a["has_limits"],
        np.asarray(t_target, dtype=np.float64),
        a["general_path"],
        respect_limits,
        has_seed,
        seed_arr,
        seed_metric,
        seed_tolerance is not None,
        seed_tolerance if seed_tolerance is not None else 0.0,
        max_solutions if max_solutions is not None else -1,
        allow_rescue,
        refinement_max_iters,
    )
    return [
        Solution(
            q=np.asarray(qs[i], dtype=np.float64),
            fk_residual=float(resids[i]),
            refinement_used="lm" if int(refine[i]) == 1 else "none",
        )
        for i in range(len(qs))
    ]
