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

import contextlib
import functools
import hashlib
import io
import json
import weakref
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ssik._prep_cache import cache_dir, source_digest, write_atomically
from ssik._solve_inputs import INT32_MAX
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy

# Solver families with a native implementation in _ssik_native. The 6R geometric
# families run the full artifact via try_native_solve; seven_r.srs runs the full
# artifact via try_native_srs_solve (seeded-track + canonical/general core +
# finalize + in-limits fallback, all native).
_NATIVE_SOLVERS = frozenset(
    {
        "ikgeo.three_parallel",
        "ikgeo.spherical_two_parallel",
        "seven_r.srs",
        "seven_r.srs_polished",
        "seven_r.spherical_shoulder",
        "seven_r.spherical_shoulder_polished",
        "ikgeo.general_6r",
        "jointlock.seven_r",
    }
)

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


def honours_options(policy: TolerancePolicy, allow_refinement: bool) -> bool:
    """Whether the native backend implements a solve with these options.

    Native runs fixed internal tolerance gates and never applies the optional
    Newton polish, so it implements only the defaults: ``policy`` equal *by
    value* to :data:`~ssik.core.tolerances.DEFAULT_TOLERANCE_POLICY` (a fresh
    ``TolerancePolicy()`` qualifies) and ``allow_refinement=False``, the
    default of every generated ``solve()``. Any other value makes the
    ``try_native_*`` entry points return ``None``, so the artifact runs the
    Python path, which honours it (#625). Threading the options through the
    native solvers is #575.
    """
    return not allow_refinement and (
        policy is DEFAULT_TOLERANCE_POLICY or policy == DEFAULT_TOLERANCE_POLICY
    )


def _validate_seed_metric(seed_metric: str, q_seed: NDArray[np.float64] | None) -> None:
    """Reject an unknown seed_metric the same way the Python path does (via
    ``ssik.postprocess.nearest_to_seed``), so the contract is identical whether the
    native backend runs or falls back. Only meaningful with a seed (matches Python,
    which reaches the metric only when ranking against ``q_seed``)."""
    if q_seed is not None and seed_metric not in ("wrap_l2", "wrap_linf"):
        raise ValueError(f"unknown metric {seed_metric!r}; expected 'wrap_l2' or 'wrap_linf'")


# The adapters below take the target and seed as ssik._solve_inputs checked them
# (float64 arrays); the binding converts anything else and rejects a malformed
# array itself. The extension's ``int`` options are 32-bit: a cap no result can
# reach is no cap (-1), and an iteration count past 2**31 is unbounded either way,
# so both are clamped inline (#574).


class _KinBodyCache:
    """Per-KinBody cache keyed on object *identity*, not merely ``id()``.

    An artifact has one long-lived ``_KB``, so caching its marshalled geometry
    is what keeps the native path from re-marshalling on every solve. But a bare
    ``id()`` key is unsound: CPython reuses an address once the object is freed.
    Measured, 300 sequentially created-and-freed KinBody objects occupied 16
    distinct ids, one of which served 276 different chains. A caller that builds
    arms dynamically -- ``from_urdf`` in a loop, a test sweeping geometries --
    therefore got **another chain's** baked constants back, and the native
    solver answered for the wrong robot: joint values that are self-consistent,
    pass the solver's own FK filter, and miss the requested pose entirely
    (#570).

    Holding a weakref alongside the entry makes the lookup verify the entry
    still belongs to this object, and the finalizer drops it the moment the
    object dies, so an address can never be inherited with stale data.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[weakref.ReferenceType[Any], Any]] = {}

    def get(self, kb: Any, default: Any = None) -> Any:
        entry = self._entries.get(id(kb))
        if entry is not None and entry[0]() is kb:
            return entry[1]
        return default

    def put(self, kb: Any, value: Any) -> Any:
        key = id(kb)

        def _drop(_ref: Any, key: int = key) -> None:
            self._entries.pop(key, None)

        # Not weakref-able: skip caching rather than key on a reusable address.
        with contextlib.suppress(TypeError):
            self._entries[key] = (weakref.ref(kb, _drop), value)
        return value

    def clear(self) -> None:
        self._entries.clear()


# Distinguishes "absent" from a cached None (these caches store None to mean
# "this arm is not of that solver class").
_MISS: Any = object()

# The native backend runs every family through one class, ``_ssik_native.Solver``
# (``ssik::Solver`` in cpp/include/ssik_cpp/solve.hpp, #707). It is built once
# per arm and family from a solver-data record: the family name, the schema
# number and a flat map of float64 / int32 arrays whose names, shapes and dtypes
# the C++ constructor checks. It prepares the family's constants once (the RR
# coefficient tensors included), so a solve only converts the target and options.
_SOLVER_DATA_SCHEMA = 1


@dataclass(frozen=True)
class _SolverRecord:
    """What ``_ssik_native.Solver`` reads (``ssik_cpp/pybind11/solver_data.hpp``)."""

    family: str
    schema: int
    arrays: dict[str, NDArray[Any]]


def _f64(x: Any) -> NDArray[np.float64]:
    return np.array(x, dtype=np.float64, order="C")


def _i32(x: Any) -> NDArray[np.int32]:
    return np.array(x, dtype=np.int32, order="C")


def _joint_arrays(kb: Any, solver_name: str) -> dict[str, NDArray[Any]]:
    """The joint arrays of every family: the public ``ssik.cpp.joint_data``,
    which also feeds cpp_emit's headers. spherical_two_parallel gets the wrist
    gauge canonicalized there (the raw _KB lacks it -- 15/18 arms);
    canonicalization is FK-identical, so the returned q are physical joint
    values and the joint limits are unchanged."""
    from ssik.cpp import _joint_data

    d = _joint_data(kb, solver_name)
    return {
        "axis": _f64(d.axis),
        "t_left": _f64(d.t_left),
        "t_right": _f64(d.t_right),
        "joint_type": _i32(d.joint_type),
        "lo": _f64(d.lo),
        "hi": _f64(d.hi),
        "present": _i32(d.present),
    }


def _rr_arrays(g: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """:func:`rr_native_geometry`'s constants and tensor as solver-data arrays."""
    po_r, po_c, po_m, po_co = g["po_coo"]
    q_r, q_c, q_m, q_co = g["q_coo"]
    return {
        "alpha": _f64(g["alpha"]),
        "a": _f64(g["a"]),
        "d": _f64(g["d"]),
        "theta_offset": _f64(g["theta_offset"]),
        "t_pre_inv": _f64(g["t_pre_inv"]),
        "t_post_inv": _f64(g["t_post_inv"]),
        "linearity_joint": _i32(int(g["linearity_joint"])),
        "left_bilinear": _i32(g["left_bilinear"]),
        "right_bilinear": _i32(g["right_bilinear"]),
        "drop_joint": _i32(int(g["drop_joint"])),
        "p_sin": _f64(g["p_sin"]),
        "p_cos": _f64(g["p_cos"]),
        "mono_factors": _i32(g["mono_factors"]),
        "po_rc": _i32(np.stack([po_r, po_c], axis=1)),
        "po_mono": _i32(po_m),
        "po_coeff": _f64(po_co),
        "q_rc": _i32(np.stack([q_r, q_c], axis=1)),
        "q_mono": _i32(q_m),
        "q_coeff": _f64(q_co),
    }


def _srs_arrays(geom: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """:func:`srs_native_geometry`'s constants as solver-data arrays."""
    return {
        "l_se": _f64(geom["l_se"]),
        "l_ew": _f64(geom["l_ew"]),
        "ee_offset_local": _f64(geom["ee_offset_local"]),
        "shoulder_pivot": _f64(geom["shoulder_pivot"]),
        "r_post_wrist": _f64(geom["r_post_wrist"]),
        "elbow_index": _i32(int(geom["elbow_index"])),
        "upper_home": _f64(geom["upper_home"]),
        "forearm_home": _f64(geom["forearm_home"]),
        "general_path": _i32(int(bool(geom["general_path"]))),
    }


def _ragged(parts: list[Any], dtype: Any) -> tuple[NDArray[Any], NDArray[np.int32]]:
    """Per-sample arrays concatenated along the first axis, plus the (S + 1,)
    row offsets of each sample."""
    offsets = np.zeros(len(parts) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(p) for p in parts])
    return np.array(np.concatenate(parts), dtype=dtype, order="C"), offsets


def _jointlock_arrays(g: dict[str, Any]) -> dict[str, NDArray[Any]]:
    """A baked jointlock geometry (:func:`load_jointlock_native_geometry`) as
    solver-data arrays: the RR kernel's ragged per-sample tensors become
    concatenated rows plus ``*_offsets``; the HP kernel's sub-chain joints take
    the common joint names."""
    out: dict[str, NDArray[Any]] = {
        "axis": _f64(g["axes"]),
        "t_left": _f64(g["t_left"]),
        "t_right": _f64(g["t_right"]),
        "joint_type": _i32(g["types"]),
        "lo": _f64(g["lo"]),
        "hi": _f64(g["hi"]),
        "present": _i32(g["has_limits"]),
        "lock_idx": _i32(int(g["lock_idx"])),
        "q_lock": _f64(g["q_lock"]),
    }
    if g["kind"] == "hp":
        out["inner_kernel"] = _i32(1)
        renamed = {"sub_axes": "sub_axis", "sub_types": "sub_joint_type"}
        ints = ("right_pv", "drop_idx", "sub_types")
        for k in _JL_HP_KEYS:
            out[renamed.get(k, k)] = _i32(g[k]) if k in ints else _f64(g[k])
        return out
    out["inner_kernel"] = _i32(0)
    for k in ("alpha", "a", "d", "theta_offset", "t_pre_inv", "t_post_inv"):
        out[k] = _f64(g[k])
    for k in ("linearity_joint", "left_bilinear", "right_bilinear", "drop_joint"):
        out[k] = _i32(g[k])
    out["p_sin"] = _f64(np.stack(g["p_sin"]))
    out["p_cos"] = _f64(np.stack(g["p_cos"]))
    out["mono_factors"], out["mono_offsets"] = _ragged(g["mono_factors"], np.int32)
    for prefix in ("po", "q"):
        out[f"{prefix}_rc"], out[f"{prefix}_offsets"] = _ragged(g[f"{prefix}_rc"], np.int32)
        out[f"{prefix}_mono"], _ = _ragged(g[f"{prefix}_mono"], np.int32)
        out[f"{prefix}_coeff"], _ = _ragged(g[f"{prefix}_coeff"], np.float64)
    return out


def _solver_arrays(kb: Any, family: str, geometry: Any = None) -> dict[str, NDArray[Any]] | None:
    """The solver-data arrays of ``kb`` for ``family``, or None when the arm is
    not of that family's class (the SRS and spherical-shoulder classifiers).
    ``geometry`` is the baked data a family cannot derive here: the RR tensor of
    ``ikgeo.general_6r`` (:func:`load_rr_native_geometry`) and the jointlock
    sweep of ``jointlock.seven_r`` (:func:`load_jointlock_native_geometry`)."""
    if family == "jointlock.seven_r":
        return _jointlock_arrays(geometry)
    if family == "ikgeo.general_6r":
        return {**_joint_arrays(kb, family), **_rr_arrays(geometry)}
    if family in ("seven_r.srs", "seven_r.srs_polished"):
        geom = (
            srs_polished_native_geometry(kb)
            if family == "seven_r.srs_polished"
            else srs_native_geometry(kb)
        )
        return None if geom is None else {**_joint_arrays(kb, family), **_srs_arrays(geom)}
    if family in ("seven_r.spherical_shoulder", "seven_r.spherical_shoulder_polished"):
        sh = spherical_shoulder_native_geometry(
            kb, polished=family == "seven_r.spherical_shoulder_polished"
        )
        return None if sh is None else {**_joint_arrays(kb, family), "coef": _f64(sh["coef"])}
    return _joint_arrays(kb, family)


# Per-KinBody prepared solvers: {family: (geometry, Solver or None)}. An entry
# is reused only for the same ``geometry`` object (an artifact passes its one
# lazily loaded sidecar on every call), so other baked data rebuilds it.
_solver_cache = _KinBodyCache()


def _solver(ext: Any, kb: Any, family: str, geometry: Any = None) -> Any:
    """The prepared ``_ssik_native.Solver`` for ``kb`` and ``family``, or None
    when the arm is not of the family's class."""
    per_kb = _solver_cache.get(kb)
    if per_kb is None:
        per_kb = _solver_cache.put(kb, {})
    entry = per_kb.get(family)
    if entry is not None and entry[0] is geometry:
        return entry[1]
    arrays = _solver_arrays(kb, family, geometry)
    solver = (
        None
        if arrays is None
        else ext.Solver(_SolverRecord(family=family, schema=_SOLVER_DATA_SCHEMA, arrays=arrays))
    )
    per_kb[family] = (geometry, solver)
    return solver


def _limit_mode(respect_limits: bool | Literal["wrap"]) -> int:
    """The extension's limit mode: 0 the raw set (``False``), 1 wrap and drop
    (``True``), 2 wrap only (``"wrap"``) -- ``finalize_solutions``' three cases."""
    if respect_limits == "wrap":
        return 2
    return 1 if respect_limits else 0


# The seed argument when there is none (the extension reads it only with a seed).
_NO_SEED = np.zeros(0, dtype=np.float64)


def _solve(
    solver: Any,
    t_target: NDArray[np.float64],
    *,
    respect_limits: bool | Literal["wrap"] = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    allow_rescue: bool = True,
    refinement_max_iters: int = 15,
    enumerate_windings: bool = True,
) -> list[Solution]:
    """One native artifact solve on a prepared Solver."""
    has_seed = q_seed is not None
    qs, resids, refine = solver.solve(
        t_target,
        _limit_mode(respect_limits),
        has_seed,
        q_seed if has_seed else _NO_SEED,
        seed_metric,
        seed_tolerance is not None,
        seed_tolerance if seed_tolerance is not None else 0.0,
        -1 if max_solutions is None or max_solutions >= INT32_MAX else max_solutions,
        allow_rescue,
        min(refinement_max_iters, INT32_MAX),
        enumerate_windings,
    )
    return [
        Solution(
            q=np.asarray(qs[i], dtype=np.float64),
            fk_residual=float(resids[i]),
            refinement_used="lm" if int(refine[i]) == 1 else "none",
        )
        for i in range(len(qs))
    ]


def _try_native(
    family: str, kb: Any, t_target: NDArray[np.float64], geometry: Any = None, **options: Any
) -> list[Solution] | None:
    """The native solve, or None when the extension is absent or the arm is not
    of the family's class."""
    ext = _load_ext()
    if ext is None:
        return None
    solver = _solver(ext, kb, family, geometry)
    if solver is None:
        return None
    return _solve(solver, t_target, **options)


_NATIVE_6R = frozenset({"ikgeo.three_parallel", "ikgeo.spherical_two_parallel", "ikgeo.general_6r"})


def try_native_solve(
    solver_name: str,
    kb: Any,
    t_target: NDArray[np.float64],
    *,
    respect_limits: bool | Literal["wrap"] = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    allow_rescue: bool = True,
    refinement_max_iters: int = 15,
    enumerate_windings: bool = True,
    rr_geometry: dict[str, Any] | None = None,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    allow_refinement: bool = False,
) -> list[Solution] | None:
    """Native artifact solve for a supported 6R family, or ``None`` to fall back.

    Mirrors the ``<arm>_ik.solve()`` contract (limits -> seed -> truncate, force
    refinement); parity with the Python artifact is validated in
    ``tests/test_native_dispatch.py`` + ``tests/test_three_parallel_artifact.py``.

    ``rr_geometry`` is the baked RR tensor (:func:`load_rr_native_geometry`) that
    the ``ikgeo.general_6r`` runtime path needs; the artifact loads it once from
    its sidecar ``.npz`` and passes it here (the ~30s derivation is build-time).

    A ``policy`` or ``allow_refinement`` other than the default returns ``None``
    (:func:`honours_options`).
    """
    if solver_name not in _NATIVE_6R or not honours_options(policy, allow_refinement):
        return None
    _validate_seed_metric(seed_metric, q_seed)
    if len(kb.joints) != 6:
        return None
    if solver_name == "ikgeo.general_6r" and rr_geometry is None:
        return None
    return _try_native(
        solver_name,
        kb,
        t_target,
        rr_geometry if solver_name == "ikgeo.general_6r" else None,
        respect_limits=respect_limits,
        q_seed=q_seed,
        seed_metric=seed_metric,
        seed_tolerance=seed_tolerance,
        max_solutions=max_solutions,
        allow_rescue=allow_rescue,
        refinement_max_iters=refinement_max_iters,
        enumerate_windings=enumerate_windings,
    )


# Per-KinBody native-SRS args (baked geometry + marshalled JointConsts) or None.
# Geometry-only, so cache per-arm (keyed on the long-lived artifact _KB's id).
_srs_cache = _KinBodyCache()


def srs_native_geometry(kb: Any, policy: Any = None) -> dict[str, Any] | None:
    """Baked SRS geometry (the SrsConsts fields + the canonical/general dispatch
    flag) for ANY concurrent-axis SRS 7R arm, or None when the arm is not
    SRS-class. Single source of truth shared by the runtime native path
    (:func:`_srs_native_args` / :func:`try_native_srs_solve`) AND the C++
    emit (``scripts/cpp_emit._srs_bake``), so the pip ``native=True`` backend and
    the self-contained artifact always cover exactly the same arms -- adding an
    SRS arm enrols it in both automatically.

    ``general_path`` mirrors the Python ``use_canonical`` dispatch:
    canonical-ZYZ + offset-free wrist -> the canonical fast-path core,
    everything else (non-ZYZ shoulder/wrist, laterally-offset wrist) -> the
    general Davenport core (#354). Both are native, and the approximate-SRS
    caller follows the same flag (#598).

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


def rr_native_geometry(kb: Any, *, linearity_joint: int | None = None) -> dict[str, Any]:
    """Baked Raghavan-Roth constants + the elimination-coefficient NUMERIC TENSOR
    for a 6R KinBody (#555). The per-arm sympy->C emitted rr_coeffs() is replaced
    by a sparse tensor: p_sin/p_cos are constant in the target; p_one (14x9) and q
    (14x8) are degree-<=3 polynomials in the 12 target entries, baked as COO over a
    shared monomial basis (each monomial = up to 3 target-entry factor indices).
    A generic C++ rr_eval_coeffs evaluates them, so the single shipped ext covers
    every RR arm with no per-arm code. Single source shared by the runtime native
    path and (eventually) the emit; mathematically identical to the lambdified RR
    (validated ~1e-14).

    ``linearity_joint`` overrides the AE-3 choice (default: the cached best, as
    production bakes it), so the native path can be checked under every choice.

    The tensor is a function of the DH ``(alpha, a, d)`` and the linearity request
    alone, and costs seconds to a minute of sympy, so it is served from the
    per-arm preparation cache when present (docs/api.md, "Per-arm preparation
    cache"). The frame fields (``theta_offset``, ``t_pre_inv``, ``t_post_inv``)
    are always computed from ``kb``, so two chains with one DH share an entry.
    """
    from ssik.kinematics.poe_to_dh import poe_to_dh

    if len(kb.joints) != 6:
        raise ValueError(f"rr_native_geometry requires a 6R chain, got {len(kb.joints)}")
    if linearity_joint is not None and linearity_joint not in (0, 1, 2):
        raise ValueError(f"linearity_joint must be 0, 1 or 2; got {linearity_joint!r}")
    dh = poe_to_dh(kb)
    alpha, a, d = dh.to_dh_tuple()
    at, bt, dt = tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist())
    geometry: dict[str, Any] = {
        "alpha": np.asarray(alpha, dtype=np.float64),
        "a": np.asarray(a, dtype=np.float64),
        "d": np.asarray(d, dtype=np.float64),
        "theta_offset": np.asarray(dh.theta_offset, dtype=np.float64),
        "t_pre_inv": np.linalg.inv(dh.t_pre),
        "t_post_inv": np.linalg.inv(dh.t_post),
    }
    lin = None if linearity_joint is None else int(linearity_joint)
    path, key = _rr_tensor_cache_entry(at, bt, dt, lin)
    tensor = None if path is None else _load_cached_rr_tensor(path, key)
    if tensor is None:
        tensor = _derive_rr_tensor(at, bt, dt, lin)
        if path is not None:
            _store_cached_rr_tensor(path, key, {**geometry, **tensor})
    return {**geometry, **tensor}


def _derive_rr_tensor(
    at: tuple[float, ...], bt: tuple[float, ...], dt: tuple[float, ...], lin: int | None
) -> dict[str, Any]:
    """The DH-only part of :func:`rr_native_geometry`, from the symbolic derivation."""
    import sympy as sp

    from ssik.solvers.ikgeo._raghavan_roth import (
        _cached_best_leftvar,
        _cached_derivation,
        _derive_pq_for_arm,
    )

    if lin is None:
        lin = int(_cached_best_leftvar(at, bt, dt))
    # Reuse the process-wide derivation when it carries the symbolic matrices
    # (the AE-3 pick has usually just derived every choice); an entry primed
    # from a shipped artifact carries only callables, so derive afresh then.
    *_fns, meta = _cached_derivation(at, bt, dt, lin)
    if "_sym_q" not in meta:
        *_fns, meta = _derive_pq_for_arm(at, bt, dt, linearity_joint=lin)
    tsyms = list(cast("list[Any]", meta["_sym_t_target"]))

    # Shared monomial basis: exponent tuple -> index; stored as up to-3 factor
    # variable indices (a degree-k monomial repeats a var index k times), -1 pad.
    basis: dict[tuple[int, ...], int] = {}

    def _idx(mono: tuple[int, ...]) -> int:
        if mono not in basis:
            basis[mono] = len(basis)
        return basis[mono]

    def _coo(matrix: Any) -> tuple[list[int], list[int], list[int], list[float]]:
        rows, cols, monos, coeffs = [], [], [], []
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                e = matrix[r, c]
                if e == 0:
                    continue
                poly = sp.Poly(e, *tsyms)
                for mono, co in zip(poly.monoms(), poly.coeffs(), strict=True):
                    rows.append(r)
                    cols.append(c)
                    monos.append(_idx(mono))
                    coeffs.append(float(co))
        return rows, cols, monos, coeffs

    po_row, po_col, po_mono, po_coeff = _coo(meta["_sym_p_one"])
    q_row, q_col, q_mono, q_coeff = _coo(meta["_sym_q"])
    # Monomial factor table (n_mono, 3): the (up to 3) target-var indices whose
    # product is the monomial, -1 padded. basis order == index order.
    factors = np.full((len(basis), 3), -1, dtype=np.int32)
    for mono, i in basis.items():
        f = [v for v, p in enumerate(mono) for _ in range(p)]  # repeat var per power
        factors[i, : len(f)] = f
    lb = cast("tuple[int, int]", meta["left_bilinear"])
    rb = cast("tuple[int, int]", meta["right_bilinear"])
    return {
        "linearity_joint": lin,
        "left_bilinear": np.array(lb, dtype=np.int32),
        "right_bilinear": np.array(rb, dtype=np.int32),
        "drop_joint": int(cast(int, meta["drop_joint"])),
        "p_sin": np.array(meta["_sym_p_sin"], dtype=np.float64),
        "p_cos": np.array(meta["_sym_p_cos"], dtype=np.float64),
        "mono_factors": factors,
        "po_coo": (
            np.array(po_row, np.int32),
            np.array(po_col, np.int32),
            np.array(po_mono, np.int32),
            np.array(po_coeff, np.float64),
        ),
        "q_coo": (
            np.array(q_row, np.int32),
            np.array(q_col, np.int32),
            np.array(q_mono, np.int32),
            np.array(q_coeff, np.float64),
        ),
    }


def _rr_npz_arrays(g: dict[str, Any]) -> dict[str, Any]:
    """:func:`rr_native_geometry`'s result as the flat arrays of the sidecar
    ``.npz`` format. Nested COO tuples are flattened into individual keys
    (np.savez pickles tuples otherwise)."""
    po_r, po_c, po_m, po_co = g["po_coo"]
    q_r, q_c, q_m, q_co = g["q_coo"]
    return {
        "alpha": g["alpha"],
        "a": g["a"],
        "d": g["d"],
        "theta_offset": g["theta_offset"],
        "t_pre_inv": g["t_pre_inv"],
        "t_post_inv": g["t_post_inv"],
        "linearity_joint": np.int32(g["linearity_joint"]),
        "left_bilinear": g["left_bilinear"],
        "right_bilinear": g["right_bilinear"],
        "drop_joint": np.int32(g["drop_joint"]),
        "p_sin": g["p_sin"],
        "p_cos": g["p_cos"],
        "mono_factors": g["mono_factors"],
        "po_row": po_r,
        "po_col": po_c,
        "po_mono": po_m,
        "po_coeff": po_co,
        "q_row": q_r,
        "q_col": q_c,
        "q_mono": q_m,
        "q_coeff": q_co,
    }


def _rr_geometry_from_npz(z: Any) -> dict[str, Any]:
    """Inverse of :func:`_rr_npz_arrays`, on an open ``np.load`` archive."""
    return {
        "alpha": z["alpha"],
        "a": z["a"],
        "d": z["d"],
        "theta_offset": z["theta_offset"],
        "t_pre_inv": z["t_pre_inv"],
        "t_post_inv": z["t_post_inv"],
        "linearity_joint": int(z["linearity_joint"]),
        "left_bilinear": z["left_bilinear"],
        "right_bilinear": z["right_bilinear"],
        "drop_joint": int(z["drop_joint"]),
        "p_sin": z["p_sin"],
        "p_cos": z["p_cos"],
        "mono_factors": z["mono_factors"],
        "po_coo": (z["po_row"], z["po_col"], z["po_mono"], z["po_coeff"]),
        "q_coo": (z["q_row"], z["q_col"], z["q_mono"], z["q_coeff"]),
    }


def bake_rr_tensor_npz(kb: Any, path: str) -> None:
    """Serialize :func:`rr_native_geometry` to a sidecar ``.npz`` (#555).

    The tensor derivation is ~30s of sympy, so it runs ONCE at build time
    (``ssik build`` / codegen), never per-solve. The runtime loads the flat arrays
    via :func:`load_rr_native_geometry`."""
    np.savez_compressed(path, **_rr_npz_arrays(rr_native_geometry(kb)))


def load_rr_native_geometry(path: str) -> dict[str, Any]:
    """Load a baked RR tensor sidecar (:func:`bake_rr_tensor_npz`), reassembling
    the COO tuples that the runtime native path consumes."""
    with np.load(path) as z:
        return _rr_geometry_from_npz(z)


# The native RR tensor in the per-arm preparation cache (ssik._prep_cache): an
# entry is the bake_rr_tensor_npz format plus its key. The key covers the exact
# DH floats, the linearity request (``auto`` or the joint), the source of the two
# modules that shape the tensor (this one, and the derivation, which also makes
# the AE-3 pick) and the sympy/mpmath/numpy versions, so a changed input or
# changed code can only miss. An entry that does not load, does not carry this
# key or fails the range checks in _rr_tensor_well_formed is a miss and is
# overwritten: the indices reach C++ unchecked, so a damaged entry must not.
_RR_TENSOR_CACHE_VERSION = 1
_RR_TENSOR_FIELDS = (
    "linearity_joint",
    "left_bilinear",
    "right_bilinear",
    "drop_joint",
    "p_sin",
    "p_cos",
    "mono_factors",
    "po_coo",
    "q_coo",
)


@functools.cache
def _rr_tensor_cache_stamp() -> dict[str, str]:
    from importlib.metadata import version

    here = Path(__file__).resolve().parent
    return {
        "source": source_digest(
            [here / "_native.py", here / "solvers" / "ikgeo" / "_raghavan_roth.py"]
        ),
        "sympy": version("sympy"),
        "mpmath": version("mpmath"),
        "numpy": np.__version__,
    }


def _rr_tensor_cache_entry(
    at: tuple[float, ...], bt: tuple[float, ...], dt: tuple[float, ...], lin: int | None
) -> tuple[Path | None, bytes]:
    root = cache_dir()
    if root is None:
        return None, b""
    stamp = _rr_tensor_cache_stamp()
    if not stamp["source"]:  # no readable source (e.g. a zipapp): run without the cache
        return None, b""
    key = {
        "kind": "rr-native-tensor",
        "version": _RR_TENSOR_CACHE_VERSION,
        **stamp,
        # repr round-trips a float exactly.
        "dh": [[repr(float(x)) for x in v] for v in (at, bt, dt)],
        "linearity_joint": "auto" if lin is None else lin,
    }
    blob = json.dumps(key, sort_keys=True).encode()
    return root / f"{hashlib.sha256(blob).hexdigest()}.npz", blob


def _load_cached_rr_tensor(path: Path, key: bytes) -> dict[str, Any] | None:
    try:
        with np.load(path) as z:
            if bytes(z["cache_key"]) != key:
                return None
            g = _rr_geometry_from_npz(z)
    except FileNotFoundError:
        return None
    # Truncated, foreign or damaged (zip CRC) file: a miss, re-derived and replaced.
    except (OSError, EOFError, KeyError, ValueError, TypeError, zipfile.BadZipFile):
        return None
    if not _rr_tensor_well_formed(g):
        return None
    return {k: g[k] for k in _RR_TENSOR_FIELDS}


def _rr_tensor_well_formed(g: dict[str, Any]) -> bool:
    """Shapes, dtypes and index ranges the C++ evaluator relies on."""

    def _index(x: Any, hi: int, lo: int = 0) -> bool:
        return bool(x.dtype == np.int32 and (x.size == 0 or lo <= x.min() <= x.max() < hi))

    mono = g["mono_factors"]
    if not (mono.ndim == 2 and mono.shape[1] == 3 and _index(mono, 12, lo=-1)):
        return False
    if not all(g[k].dtype == np.float64 and g[k].shape == (14, 9) for k in ("p_sin", "p_cos")):
        return False
    for name, n_cols in (("po_coo", 9), ("q_coo", 8)):
        rows, cols, monos, coeffs = g[name]
        if not (rows.ndim == 1 and rows.shape == cols.shape == monos.shape == coeffs.shape):
            return False
        if not (_index(rows, 14) and _index(cols, n_cols) and _index(monos, len(mono))):
            return False
        if coeffs.dtype != np.float64:
            return False
    pairs = (g["left_bilinear"], g["right_bilinear"])
    if not all(p.shape == (2,) and _index(p, 6) for p in pairs):
        return False
    return g["linearity_joint"] in (0, 1, 2) and 0 <= g["drop_joint"] < 6


def _store_cached_rr_tensor(path: Path, key: bytes, g: dict[str, Any]) -> None:
    buf = io.BytesIO()
    np.savez_compressed(buf, cache_key=np.bytes_(key), **_rr_npz_arrays(g))
    write_atomically(path, buf.getvalue())


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


def jointlock_hp_native_geometry(kb: Any) -> dict[str, Any]:
    """Baked geometry for an HP-inner jointlock 7R arm (kassow, #491/#554): the
    16-sample lock schedule + per-sample HpConsts (Study-quaternion kernel) + the
    locked 6R sub-chain's JointConsts<6>. All numeric (no per-arm code), stacked
    for the ``jointlock_hp_artifact_solve`` binding. Single source shared by the
    runtime native path and (via _hp_bake) the standalone emit."""
    from ssik.solvers.jointlock.seven_r import _lock_joint, choose_lock_joint

    if len(kb.joints) != 7:
        raise ValueError(f"jointlock_hp_native_geometry requires a 7R chain, got {len(kb.joints)}")
    lock = int(choose_lock_joint(kb))
    lo0, hi0 = kb.joints[lock].limits if kb.joints[lock].limits else (-np.pi, np.pi)
    samples = np.linspace(lo0, hi0, 16, endpoint=False)
    hps, subs = [], []
    for q_lock in samples:
        sub = _lock_joint(kb, lock, float(q_lock))
        hps.append(hp_native_geometry(sub))
        subs.append(sub)

    def _stack(key: str) -> NDArray[np.float64]:
        return np.stack([np.asarray(h[key], dtype=np.float64) for h in hps])

    j7 = kb.joints
    return {
        "lock_idx": lock,
        "q_lock": np.asarray(samples, dtype=np.float64),
        "t_u": _stack("t_u"),
        "t_w_pre": _stack("t_w_pre"),
        "dh_a": _stack("dh_a"),
        "dh_l": _stack("dh_l"),
        "dh_d": _stack("dh_d"),
        "theta_offset": _stack("theta_offset"),
        "t_pre_inv": _stack("t_pre_inv"),
        "t_post_inv": _stack("t_post_inv"),
        "t_z_neg_d1": _stack("t_z_neg_d1"),
        "t_joint6_offset_inv": _stack("t_joint6_offset_inv"),
        "right_pv": np.array([int(h["right_parametric_var"]) for h in hps], dtype=np.int32),
        "drop_idx": np.array([int(h["drop_idx"]) for h in hps], dtype=np.int32),
        "sub_axes": np.stack([np.array([jt.axis for jt in s.joints], np.float64) for s in subs]),
        "sub_t_left": np.stack(
            [np.array([jt.T_left for jt in s.joints], np.float64) for s in subs]
        ),
        "sub_t_right": np.stack(
            [np.array([jt.T_right for jt in s.joints], np.float64) for s in subs]
        ),
        "sub_types": np.stack(
            [
                np.array([0 if jt.joint_type == "revolute" else 1 for jt in s.joints], np.int32)
                for s in subs
            ]
        ),
        "lo": np.array([jt.limits[0] if jt.limits else 0.0 for jt in j7], dtype=np.float64),
        "hi": np.array([jt.limits[1] if jt.limits else 0.0 for jt in j7], dtype=np.float64),
        "has_limits": np.array([1 if jt.limits else 0 for jt in j7], dtype=np.int32),
        "axes": np.array([jt.axis for jt in j7], dtype=np.float64),
        "t_left": np.array([jt.T_left for jt in j7], dtype=np.float64),
        "t_right": np.array([jt.T_right for jt in j7], dtype=np.float64),
        "types": np.array([0 if jt.joint_type == "revolute" else 1 for jt in j7], dtype=np.int32),
    }


def jointlock_rr_native_geometry(kb: Any) -> dict[str, Any]:
    """Baked geometry for an RR-inner jointlock 7R arm (rizon4/rizon10, #554): the
    16-sample lock schedule + per-sample RR numeric tensor (:func:`rr_native_
    geometry` on each locked 6R sub-chain). Fixed-size RrConsts are stacked;
    the variable-length COO tensors are returned as length-16 lists for the
    ``jointlock_rr_artifact_solve`` binding. The ~30s/sub-chain sympy derivation
    (x16) is BUILD-time only. No zero_threshold here: numerical-noise coeffs are
    harmless at runtime (evaluated, not byte-compared like the emit)."""
    from ssik.solvers.jointlock.seven_r import _lock_joint, choose_lock_joint

    if len(kb.joints) != 7:
        raise ValueError(f"jointlock_rr_native_geometry requires a 7R chain, got {len(kb.joints)}")
    lock = int(choose_lock_joint(kb))
    lo0, hi0 = kb.joints[lock].limits if kb.joints[lock].limits else (-np.pi, np.pi)
    samples = np.linspace(lo0, hi0, 16, endpoint=False)
    subs = [rr_native_geometry(_lock_joint(kb, lock, float(q))) for q in samples]

    def _stack(key: str) -> NDArray[np.float64]:
        return np.stack([np.asarray(s[key], dtype=np.float64) for s in subs])

    j7 = kb.joints
    return {
        "lock_idx": lock,
        "q_lock": np.asarray(samples, dtype=np.float64),
        "alpha": _stack("alpha"),
        "a": _stack("a"),
        "d": _stack("d"),
        "theta_offset": _stack("theta_offset"),
        "t_pre_inv": _stack("t_pre_inv"),
        "t_post_inv": _stack("t_post_inv"),
        "linearity_joint": np.array([int(s["linearity_joint"]) for s in subs], dtype=np.int32),
        "left_bilinear": np.stack([np.asarray(s["left_bilinear"], np.int32) for s in subs]),
        "right_bilinear": np.stack([np.asarray(s["right_bilinear"], np.int32) for s in subs]),
        "drop_joint": np.array([int(s["drop_joint"]) for s in subs], dtype=np.int32),
        # Variable-length per sample -> lists (the binding takes py::list).
        "p_sin": [np.asarray(s["p_sin"], np.float64) for s in subs],
        "p_cos": [np.asarray(s["p_cos"], np.float64) for s in subs],
        "mono_factors": [np.asarray(s["mono_factors"], np.int32) for s in subs],
        "po_rc": [
            np.stack([s["po_coo"][0], s["po_coo"][1]], axis=1).astype(np.int32) for s in subs
        ],
        "po_mono": [np.asarray(s["po_coo"][2], np.int32) for s in subs],
        "po_coeff": [np.asarray(s["po_coo"][3], np.float64) for s in subs],
        "q_rc": [np.stack([s["q_coo"][0], s["q_coo"][1]], axis=1).astype(np.int32) for s in subs],
        "q_mono": [np.asarray(s["q_coo"][2], np.int32) for s in subs],
        "q_coeff": [np.asarray(s["q_coo"][3], np.float64) for s in subs],
        "axes": np.array([jt.axis for jt in j7], dtype=np.float64),
        "t_left": np.array([jt.T_left for jt in j7], dtype=np.float64),
        "t_right": np.array([jt.T_right for jt in j7], dtype=np.float64),
        "types": np.array([0 if jt.joint_type == "revolute" else 1 for jt in j7], dtype=np.int32),
        "lo": np.array([jt.limits[0] if jt.limits else 0.0 for jt in j7], dtype=np.float64),
        "hi": np.array([jt.limits[1] if jt.limits else 0.0 for jt in j7], dtype=np.float64),
        "has_limits": np.array([1 if jt.limits else 0 for jt in j7], dtype=np.int32),
    }


# HP-inner jointlock arms (RR-incomplete symmetric-DH sub-chains -> Study-quaternion
# kernel); everything else in the family is RR-inner. Mirrors scripts.cpp_emit.
_HP_JOINTLOCK_ARMS = frozenset({"kassow_kr810_ik"})

# Variable-length (per lock sample) RR tensor COO components, flattened to
# per-sample keys in the sidecar .npz (npz has no ragged arrays).
_JL_RR_RAGGED = ("mono_factors", "po_rc", "po_mono", "po_coeff", "q_rc", "q_mono", "q_coeff")
_JL_HP_KEYS = (
    "t_u",
    "t_w_pre",
    "dh_a",
    "dh_l",
    "dh_d",
    "theta_offset",
    "t_pre_inv",
    "t_post_inv",
    "t_z_neg_d1",
    "t_joint6_offset_inv",
    "right_pv",
    "drop_idx",
    "sub_axes",
    "sub_t_left",
    "sub_t_right",
    "sub_types",
)
_JL_RR_STACKED = (
    "alpha",
    "a",
    "d",
    "theta_offset",
    "t_pre_inv",
    "t_post_inv",
    "linearity_joint",
    "left_bilinear",
    "right_bilinear",
    "drop_joint",
)
_JL_COMMON = ("axes", "t_left", "t_right", "types", "lo", "hi", "has_limits")


def bake_jointlock_npz(kb: Any, path: str, *, use_hp: bool) -> None:
    """Serialize the jointlock native geometry to a sidecar ``.npz`` (#554). RR
    arms (rizon4/rizon10) run 16 ~30s sympy derivations here at BUILD time; kassow
    (HP) is fast. The emitted ``_jointlock_native_geometry()`` loads it lazily."""
    if use_hp:
        g = jointlock_hp_native_geometry(kb)
        np.savez_compressed(
            path,
            kind=np.bytes_(b"hp"),
            lock_idx=np.int32(g["lock_idx"]),
            q_lock=g["q_lock"],
            **{k: g[k] for k in _JL_HP_KEYS + _JL_COMMON},
        )
    else:
        g = jointlock_rr_native_geometry(kb)
        flat = {f"{k}_{i}": arr for k in _JL_RR_RAGGED for i, arr in enumerate(g[k])}
        np.savez_compressed(
            path,
            kind=np.bytes_(b"rr"),
            lock_idx=np.int32(g["lock_idx"]),
            q_lock=g["q_lock"],
            p_sin=np.stack(g["p_sin"]),
            p_cos=np.stack(g["p_cos"]),
            **{k: g[k] for k in _JL_RR_STACKED + _JL_COMMON},
            **flat,
        )


def load_jointlock_native_geometry(path: str) -> dict[str, Any]:
    """Load a baked jointlock sidecar (:func:`bake_jointlock_npz`), reassembling
    the per-sample lists the RR binding needs."""
    with np.load(path) as z:
        kind = bytes(z["kind"]).decode()
        d: dict[str, Any] = {
            "kind": kind,
            "lock_idx": int(z["lock_idx"]),
            "q_lock": z["q_lock"],
            **{k: z[k] for k in _JL_COMMON},
        }
        if kind == "hp":
            d.update({k: z[k] for k in _JL_HP_KEYS})
        else:
            d.update({k: z[k] for k in _JL_RR_STACKED})
            d["p_sin"] = list(z["p_sin"])
            d["p_cos"] = list(z["p_cos"])
            for k in _JL_RR_RAGGED:
                d[k] = [z[f"{k}_{i}"] for i in range(16)]
    return d


def try_native_jointlock_solve(
    solver_name: str,
    kb: Any,
    t_target: NDArray[np.float64],
    *,
    respect_limits: bool | Literal["wrap"] = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    allow_rescue: bool = True,
    refinement_max_iters: int = 15,
    enumerate_windings: bool = True,
    jointlock_geometry: dict[str, Any] | None = None,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    allow_refinement: bool = False,
) -> list[Solution] | None:
    """FULL native jointlock.seven_r artifact solve (#554), or ``None`` to fall
    back. Dispatches by the baked ``kind``: RR-inner (rizon4/rizon10) via the
    16 numeric RR tensors; HP-inner (kassow) via the Study-quaternion kernel.
    Redundant 7R sampling solver -> relative-completeness contract. A
    non-default ``policy`` or ``allow_refinement`` returns ``None``
    (:func:`honours_options`)."""
    if solver_name != "jointlock.seven_r" or jointlock_geometry is None:
        return None
    if not honours_options(policy, allow_refinement):
        return None
    _validate_seed_metric(seed_metric, q_seed)
    return _try_native(
        solver_name,
        kb,
        t_target,
        jointlock_geometry,
        respect_limits=respect_limits,
        q_seed=q_seed,
        seed_metric=seed_metric,
        seed_tolerance=seed_tolerance,
        max_solutions=max_solutions,
        allow_rescue=allow_rescue,
        refinement_max_iters=refinement_max_iters,
        enumerate_windings=enumerate_windings,
    )


def _srs_native_args(kb: Any, solver_name: str = "seven_r.srs") -> dict[str, Any] | None:
    """Cached SRS geometry (strict for ``seven_r.srs``, relaxed/approximate for
    ``seven_r.srs_polished``) + the marshalled JointConsts arrays for the binding,
    or None when the arm isn't SRS-class under that classifier."""
    cached = _srs_cache.get(kb, _MISS)
    if cached is not _MISS:
        return cached  # type: ignore[no-any-return]
    geom = (
        srs_polished_native_geometry(kb)
        if solver_name == "seven_r.srs_polished"
        else srs_native_geometry(kb)
    )
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
    _srs_cache.put(kb, result)
    return result


def try_native_srs_solve(
    solver_name: str,
    kb: Any,
    t_target: NDArray[np.float64],
    *,
    respect_limits: bool | Literal["wrap"] = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    allow_rescue: bool = True,
    refinement_max_iters: int = 15,
    enumerate_windings: bool = True,
) -> list[Solution] | None:
    """FULL native SRS artifact solve (the whole ``<arm>.solve()`` contract in
    C++), or ``None`` to fall back: seeded-track fast path, canonical/general
    core, finalize (limits -> seed -> truncate), and the #359 in-limits
    fallback, so the Python postprocess (which dominated the per-call time) is
    skipped.

    The empty-gated T-perturbation rescue runs in C++, the same definition as
    Python's (#622). Parity with the Python solve is validated across the
    full contract in tests/test_srs_artifact_cpp.py + test_srs_general_cpp.py.
    """
    if solver_name not in ("seven_r.srs", "seven_r.srs_polished"):
        return None
    return _try_native(
        solver_name,
        kb,
        t_target,
        respect_limits=respect_limits,
        q_seed=q_seed,
        seed_metric=seed_metric,
        seed_tolerance=seed_tolerance,
        max_solutions=max_solutions,
        allow_rescue=allow_rescue,
        refinement_max_iters=refinement_max_iters,
        enumerate_windings=enumerate_windings,
    )


def try_native_spherical_shoulder_solve(
    solver_name: str,
    kb: Any,
    t_target: NDArray[np.float64],
    *,
    respect_limits: bool | Literal["wrap"] = True,
    q_seed: NDArray[np.float64] | None = None,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
    max_solutions: int | None = None,
    allow_rescue: bool = True,
    refinement_max_iters: int = 15,
    enumerate_windings: bool = True,
) -> list[Solution] | None:
    """FULL native spherical_shoulder{,_polished} artifact solve (#553), or ``None``
    to fall back. ``_polished`` (xarm7/gen72) LM-refines the approximate candidates;
    the exact arms (franka/fr3) FK-gate. Redundant 7R, so the runtime contract is
    relative-completeness (soundness + coverage), same as srs_polished."""
    if solver_name not in ("seven_r.spherical_shoulder", "seven_r.spherical_shoulder_polished"):
        return None
    return _try_native(
        solver_name,
        kb,
        t_target,
        respect_limits=respect_limits,
        q_seed=q_seed,
        seed_metric=seed_metric,
        seed_tolerance=seed_tolerance,
        max_solutions=max_solutions,
        allow_rescue=allow_rescue,
        refinement_max_iters=refinement_max_iters,
        enumerate_windings=enumerate_windings,
    )


def try_native_solve_7r(
    solver_name: str,
    kb: Any,
    t_target: NDArray[np.float64],
    *,
    policy: TolerancePolicy = DEFAULT_TOLERANCE_POLICY,
    allow_refinement: bool = False,
    **kwargs: Any,
) -> list[Solution] | None:
    """Unified native entry for the thin-wrapper 7R families (the artifact hook
    calls this). Routes by ``solver_name`` to the focused per-family solve, or
    returns ``None`` to fall back to Python: SRS + srs_polished ->
    :func:`try_native_srs_solve`; spherical_shoulder{,_polished} ->
    :func:`try_native_spherical_shoulder_solve`. A non-default ``policy`` or
    ``allow_refinement`` returns ``None`` (:func:`honours_options`)."""
    _validate_seed_metric(kwargs.get("seed_metric", "wrap_linf"), kwargs.get("q_seed"))
    if not honours_options(policy, allow_refinement):
        return None
    if solver_name in ("seven_r.srs", "seven_r.srs_polished"):
        return try_native_srs_solve(solver_name, kb, t_target, **kwargs)
    if solver_name in ("seven_r.spherical_shoulder", "seven_r.spherical_shoulder_polished"):
        return try_native_spherical_shoulder_solve(solver_name, kb, t_target, **kwargs)
    return None
