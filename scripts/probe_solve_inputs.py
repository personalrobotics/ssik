"""Fuzz the public solve boundary and the native bindings with malformed input (#574).

Two probes, each run case by case in a child process so a native crash is
recorded as such instead of ending the run:

- ``public``: every shipped native solver family (one arm each, both jointlock
  kernels) through ``<arm>.solve`` on ``native=True`` and ``native=False``, the
  live ``Manipulator.solve`` (Python only), the artifacts' ``fk``, and the chart
  entry points ``Manipulator.self_motion`` / ``solve_path``. Each case is one
  malformed (or valid-but-unusual) input with everything else valid.
- ``bindings``: every function and class of ``ssik._ssik_native``, called
  directly with shape-valid placeholder arguments of which one at a time is
  malformed (one rank too many, one entry short, NaN/inf in the target or
  seed, an index out of range, an unknown enum value); the chart objects'
  methods, on charts built for a real arm, with an out-of-range chart index.

The cases are fixed (no randomness), so two runs on the same build print the
same table. Usage::

    uv run python scripts/probe_solve_inputs.py            # both probes
    uv run python scripts/probe_solve_inputs.py public --json out.json

Outcomes: ``ok:N`` (returned N solutions or a value), ``<Exception>: message``,
``CRASH:<signal>``, or ``TIMEOUT``. ``docs/api.md`` ("Input validation") is the
contract each outcome is read against; ``tests/test_solve_inputs.py`` asserts it.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

# One arm per native solver family; rizon4 is the jointlock RR kernel, kr810 HP.
ARMS = {
    "ikgeo.three_parallel": "ur5_ik",
    "ikgeo.spherical_two_parallel": "irb6700_ik",
    "ikgeo.general_6r": "hc10_ik",
    "seven_r.srs": "iiwa14_ik",
    "seven_r.srs_polished": "openarm_left_ik",
    "seven_r.spherical_shoulder": "franka_panda_ik",
    "seven_r.spherical_shoulder_polished": "gen72_ik",
    "jointlock.seven_r (RR)": "rizon4_ik",
    "jointlock.seven_r (HP)": "kassow_kr810_ik",
}
CHART_ARMS = ("franka_panda_ik", "iiwa14_ik")
_TIMEOUT_S = 60.0  # per case


def _module(name: str) -> Any:
    from ssik.prebuilt._manifest import load_manifest

    arm = load_manifest()[name]
    return importlib.import_module(arm.hier_module or f"ssik.prebuilt.{name}")


def _valid(mod: Any) -> tuple[np.ndarray, np.ndarray]:
    from ssik.prebuilt._manifest import load_manifest

    name = mod.__name__.rpartition(".")[2]
    arm = load_manifest().get(name)
    q = np.array(arm.sample_q if arm else np.full(mod.DOF, 0.3), dtype=np.float64)
    return mod.fk(q), q


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


# ---------------------------------------------------------------------------
# Public-API cases: name -> f(T, q) -> (T_target, kwargs). Fixed, no RNG.
# ---------------------------------------------------------------------------


def _with(T: Any = None, **kw: Any) -> Callable[[np.ndarray, np.ndarray], tuple[Any, dict]]:
    def make(T0: np.ndarray, q0: np.ndarray) -> tuple[Any, dict]:
        t = T0 if T is None else T(T0, q0)
        out = {k: (v(T0, q0) if callable(v) else v) for k, v in kw.items()}
        return t, out

    return make


def _set(T0: np.ndarray, idx: tuple[int, int], v: float) -> np.ndarray:
    T = T0.copy()
    T[idx] = v
    return T


def _perturb_rot(T0: np.ndarray, eps: float) -> np.ndarray:
    """Rotation off SO(3) by a symmetric stretch: ||R^T R - I||_F ~= 2*sqrt(3)*eps/sqrt(3)."""
    T = T0.copy()
    T[:3, :3] = T0[:3, :3] @ (np.eye(3) + eps * np.diag([1.0, -0.5, 0.25]))
    return T


def _fortran(T0: np.ndarray) -> np.ndarray:
    return np.asfortranarray(T0)


def _strided(T0: np.ndarray) -> np.ndarray:
    big = np.zeros((8, 8))
    big[::2, ::2] = T0
    return big[::2, ::2]


PUBLIC_CASES: dict[str, Callable[[np.ndarray, np.ndarray], tuple[Any, dict]]] = {
    # baseline + valid-but-unusual targets (must solve, same set)
    "T valid": _with(),
    "T list-of-lists": _with(lambda T, q: T.tolist()),
    "T float32": _with(lambda T, q: T.astype(np.float32)),
    "T Fortran-ordered": _with(lambda T, q: _fortran(T)),
    "T non-contiguous view": _with(lambda T, q: _strided(T)),
    "T int identity": _with(lambda T, q: np.eye(4, dtype=np.int64)),
    "T rotation off SO(3) by 1e-9": _with(lambda T, q: _perturb_rot(T, 1e-9)),
    "T rotation off SO(3) by 1e-6": _with(lambda T, q: _perturb_rot(T, 1e-6)),
    # malformed targets
    "T shape (3, 3)": _with(lambda T, q: np.eye(3)),
    "T shape (3, 4)": _with(lambda T, q: T[:3]),
    "T shape (16,)": _with(lambda T, q: T.ravel()),
    "T shape (4,)": _with(lambda T, q: np.ones(4)),
    "T shape (1, 4, 4)": _with(lambda T, q: T[None]),
    "T shape (5, 5)": _with(lambda T, q: np.eye(5)),
    "T empty": _with(lambda T, q: np.zeros((0, 4))),
    "T scalar": _with(lambda T, q: 1.0),
    "T None": _with(lambda T, q: None),
    "T ragged list": _with(lambda T, q: [[1, 0, 0, 0], [0, 1, 0]]),
    "T strings": _with(lambda T, q: np.full((4, 4), "1.0")),
    "T complex": _with(lambda T, q: T.astype(np.complex128)),
    "T bool": _with(lambda T, q: np.eye(4, dtype=bool)),
    "T object of floats": _with(lambda T, q: T.astype(object)),
    "T NaN translation": _with(lambda T, q: _set(T, (0, 3), np.nan)),
    "T NaN rotation": _with(lambda T, q: _set(T, (1, 1), np.nan)),
    "T +inf translation": _with(lambda T, q: _set(T, (2, 3), np.inf)),
    "T -inf rotation": _with(lambda T, q: _set(T, (0, 0), -np.inf)),
    "T rotation scaled 1.01": _with(lambda T, q: _perturb_rot(T, 1e-2)),
    "T rotation off SO(3) by 1e-3": _with(lambda T, q: _perturb_rot(T, 1e-3)),
    "T reflection (det -1)": _with(lambda T, q: _set(T, (slice(0, 3), 0), -T[:3, 0])),
    "T bottom row [0,0,0,2]": _with(lambda T, q: _set(T, (3, 3), 2.0)),
    "T bottom row [1,0,0,1]": _with(lambda T, q: _set(T, (3, 0), 1.0)),
    "T zero matrix": _with(lambda T, q: np.zeros((4, 4))),
    "T huge translation 1e300": _with(lambda T, q: _set(T, (0, 3), 1e300)),
    # seeds
    "q_seed valid": _with(q_seed=lambda T, q: q),
    "q_seed list": _with(q_seed=lambda T, q: q.tolist()),
    "q_seed float32": _with(q_seed=lambda T, q: q.astype(np.float32)),
    "q_seed non-contiguous": _with(q_seed=lambda T, q: np.repeat(q, 2)[::2]),
    "q_seed length 1": _with(q_seed=lambda T, q: q[:1]),
    "q_seed length dof-1": _with(q_seed=lambda T, q: q[:-1]),
    "q_seed length dof+1": _with(q_seed=lambda T, q: np.append(q, 0.0)),
    "q_seed shape (1, dof)": _with(q_seed=lambda T, q: q[None]),
    "q_seed scalar": _with(q_seed=lambda T, q: 0.5),
    "q_seed 0-d array": _with(q_seed=lambda T, q: np.array(0.5)),
    "q_seed NaN": _with(q_seed=lambda T, q: np.where(np.arange(q.size) == 1, np.nan, q)),
    "q_seed inf": _with(q_seed=lambda T, q: np.where(np.arange(q.size) == 0, np.inf, q)),
    "q_seed strings": _with(q_seed=lambda T, q: [str(v) for v in q]),
    "q_seed complex": _with(q_seed=lambda T, q: q.astype(np.complex128)),
    "q_seed + max_solutions=1": _with(q_seed=lambda T, q: q, max_solutions=1),
    "q_seed length 1 + max_solutions=1": _with(q_seed=lambda T, q: q[:1], max_solutions=1),
    # scalar options
    "max_solutions=0": _with(max_solutions=0),
    "max_solutions=0 + T shape (3, 3)": _with(lambda T, q: np.eye(3), max_solutions=0),
    "max_solutions=0 + T NaN": _with(lambda T, q: _set(T, (0, 3), np.nan), max_solutions=0),
    "max_solutions=0 + q_seed length 1": _with(q_seed=lambda T, q: q[:1], max_solutions=0),
    "max_solutions=0 + seed_tolerance without seed": _with(max_solutions=0, seed_tolerance=0.1),
    "max_solutions=-1": _with(max_solutions=-1),
    "max_solutions=-5": _with(max_solutions=-5),
    "max_solutions=2.5": _with(max_solutions=2.5),
    "max_solutions='3'": _with(max_solutions="3"),
    "max_solutions=np.int64(2)": _with(max_solutions=np.int64(2)),
    "max_solutions=2**40": _with(max_solutions=2**40),
    "max_solutions=True": _with(max_solutions=True),
    "respect_limits='wrap'": _with(respect_limits="wrap"),
    "respect_limits='bogus'": _with(respect_limits="bogus"),
    "respect_limits=None": _with(respect_limits=None),
    "seed_metric='bogus' (seeded)": _with(q_seed=lambda T, q: q, seed_metric="bogus"),
    "seed_metric='bogus' (unseeded)": _with(seed_metric="bogus"),
    "seed_tolerance without seed": _with(seed_tolerance=0.1),
    "seed_tolerance=NaN": _with(q_seed=lambda T, q: q, seed_tolerance=float("nan")),
    "seed_tolerance=-1": _with(q_seed=lambda T, q: q, seed_tolerance=-1.0),
    "seed_tolerance=inf": _with(q_seed=lambda T, q: q, seed_tolerance=float("inf")),
    "refinement_max_iters=-1": _with(refinement_max_iters=-1),
    "refinement_max_iters=2**40": _with(refinement_max_iters=2**40),
    "refinement_max_iters=2.5": _with(refinement_max_iters=2.5),
}

# Inputs to the artifact's public ``fk``.
FK_CASES: dict[str, Callable[[np.ndarray], Any]] = {
    "fk q valid": lambda q: q,
    "fk q list": lambda q: q.tolist(),
    "fk q length dof-1": lambda q: q[:-1],
    "fk q length dof+1": lambda q: np.append(q, 0.0),
    "fk q shape (1, dof)": lambda q: q[None],
    "fk q NaN": lambda q: np.full(q.size, np.nan),
    "fk q strings": lambda q: [str(v) for v in q],
}

# Chart entry points (Manipulator.self_motion / solve_path).
CHART_CASES: dict[str, Callable[[np.ndarray, np.ndarray], tuple[str, Any]]] = {
    "self_motion T valid": lambda T, q: ("self_motion", T),
    "self_motion T (3, 3)": lambda T, q: ("self_motion", np.eye(3)),
    "self_motion T NaN": lambda T, q: ("self_motion", _set(T, (0, 3), np.nan)),
    "self_motion T inf": lambda T, q: ("self_motion", _set(T, (1, 1), np.inf)),
    "self_motion T not rigid": lambda T, q: ("self_motion", _perturb_rot(T, 1e-2)),
    "self_motion T strings": lambda T, q: ("self_motion", np.full((4, 4), "1.0")),
    "solve_path valid": lambda T, q: ("solve_path", np.stack([T, T])),
    "solve_path NaN pose": lambda T, q: ("solve_path", np.stack([T, _set(T, (0, 3), np.nan)])),
    "solve_path (2, 3, 3)": lambda T, q: ("solve_path", np.stack([np.eye(3), np.eye(3)])),
    "solve_path q0 length 1": lambda T, q: ("solve_path_q0", (np.stack([T, T]), q[:1])),
    "solve_path q0 NaN": lambda T, q: ("solve_path_q0", (np.stack([T, T]), np.full(7, np.nan))),
}


def _outcome(fn: Callable[[], Any]) -> str:
    try:
        r = fn()
    except Exception as e:  # the probe records whatever the boundary does
        msg = str(e).splitlines()[0] if str(e) else ""
        return f"{type(e).__name__}: {msg[:90]}"
    if isinstance(r, list):
        return f"ok:{len(r)}"
    if isinstance(r, np.ndarray):
        return f"ok:array{r.shape}"
    return f"ok:{type(r).__name__}"


Job = tuple[str, str, str]  # (target, backend, case)
PUBLIC_BACKENDS = ("native", "python", "python-no-ext")


def public_jobs(cases: Iterable[str] | None = None) -> list[Job]:
    """The public-API jobs, in a fixed order. ``python-no-ext`` is the Python
    backend with the input check's numeric kernel forced to its numpy reference,
    as on a platform without the native extension."""
    chosen = [c for c in PUBLIC_CASES if cases is None or c in cases]
    jobs: list[Job] = []
    for arm in ARMS.values():
        jobs += [(arm, b, c) for b in PUBLIC_BACKENDS for c in chosen]
        jobs += [(arm, "fk", c) for c in FK_CASES if cases is None or c in cases]
    jobs += [("ur5_ik", "live", c) for c in chosen]
    for arm in CHART_ARMS:
        for backend in ("chart-native", "chart-python"):
            jobs += [(arm, backend, c) for c in CHART_CASES if cases is None or c in cases]
    return jobs


def _force_numpy_kernel() -> None:
    import ssik._solve_inputs as si

    si._kernel = None


def _run_public(arm: str, backend: str, case: str) -> str:
    mod = _module(arm)
    T0, q0 = _valid(mod)
    if backend == "fk":
        return _outcome(lambda: mod.fk(FK_CASES[case](q0)))
    if backend.startswith("chart-"):
        from ssik import Manipulator

        m = Manipulator.from_prebuilt(arm)
        native = backend == "chart-native"
        kind, arg = CHART_CASES[case](T0, q0)
        if kind == "self_motion":
            return _outcome(lambda: m.self_motion(arg, native=native).charts)
        if kind == "solve_path":
            return _outcome(lambda: m.solve_path(arg, native=native).tracks)
        poses, q_start = arg
        return _outcome(lambda: m.solve_path(poses, q0=q_start, native=native).tracks)
    T, kw = PUBLIC_CASES[case](T0, q0)
    if backend == "live":
        from ssik import Manipulator

        live = Manipulator(mod._KB)
        return _outcome(lambda: live.solve(T, **kw))
    if backend == "python-no-ext":
        with contextlib.suppress(ImportError):  # absent before #574
            _force_numpy_kernel()
    return _outcome(lambda: mod.solve(T, native=backend == "native", **kw))


# ---------------------------------------------------------------------------
# Binding probe: every function and class of ssik._ssik_native, called with a
# shape-valid baseline in which one argument at a time is malformed.
# ---------------------------------------------------------------------------

# Index-like integers (a joint, a lock sample, a Study row).
_INDEX_INTS = ("elbow_index", "lock_idx", "drop_idx", "linearity_joint", "drop_joint")
_FINITE_ARGS = ("target", "q_seed")
# Arrays whose leading length is data, not part of the contract.
_FREE_LENGTH = (
    "mono_factors",
    "po_rc",
    "po_mono",
    "po_coeff",
    "q_rc",
    "q_mono",
    "q_coeff",
    "coeffs",
    "swept",
    "grid",
    "tangent",
)
# No array or index argument a caller can get wrong.
_SKIP = ("get_max_threads", "set_max_threads")
_SEVEN = ("srs", "spherical_shoulder", "jointlock", "SrsCharts", "SphericalShoulderCharts")
_RR_LISTS = (
    "p_sin",
    "p_cos",
    "mono_factors",
    "po_rc",
    "po_mono",
    "po_coeff",
    "q_rc",
    "q_mono",
    "q_coeff",
)


def _f(*shape: int) -> np.ndarray:
    return np.zeros(shape)


def _i(*shape: int, fill: int = 0) -> np.ndarray:
    return np.full(shape, fill, dtype=np.int32)


def _tile(a: Any, n: int) -> np.ndarray:
    a = np.asarray(a)
    return np.tile(a, (n,) + (1,) * a.ndim)


def _baseline_values(fn_name: str) -> dict[str, Any]:
    """Shape-valid arguments by parameter name for one binding. The values are
    placeholders: each probe call malforms one argument, which the binding must
    reject before it computes anything."""
    n = 7 if any(k in fn_name for k in _SEVEN) else 6
    v: dict[str, Any] = {
        # options
        "family": "ikgeo.three_parallel",
        "respect_limits": 1,
        "has_seed": True,
        "seed_metric": "wrap_linf",
        "has_seed_tolerance": False,
        "seed_tolerance": 0.0,
        "max_solutions": -1,
        "allow_rescue": False,
        "refinement_max_iters": 0,
        "enumerate_windings": False,
        "polished": False,
        "general_path": False,
        "allow_refinement": False,
        "bounded": False,
        "fk_atol": 1e-9,
        "feasibility": 1e-9,
        "degeneracy": 1e-12,
        "real_tol": 1e-3,
        "max_magnitude": 1e10,
        "accept_residue_tol": 1e-3,
        "ortho_tol": 3e-5,
        "row_tol": 1e-5,
        "u": 0.0,
        "w": 0.0,
        "l_se": 0.3,
        "l_ew": 0.3,
        "elbow_index": 3,
        "right_parametric_var": 0,
        "lock_idx": 0,
        "metric": None,
        # the chain and the pose
        "axes": _tile([0.0, 0.0, 1.0], n),
        "t_left": _tile(np.eye(4), n),
        "t_right": _tile(np.eye(4), n),
        "types": _i(n),
        "lo": -np.ones(n),
        "hi": np.ones(n),
        "has_limits": _i(n, fill=1),
        "q_seed": _f(n),
        "target": np.eye(4),
        # SRS
        "ee_offset": _f(3),
        "ee_offset_local": _f(3),
        "shoulder_pivot": _f(3),
        "r_post_wrist": np.eye(3),
        "upper_home": _f(3),
        "forearm_home": _f(3),
        # spherical shoulder
        "coef": _f(3, 48),
        # RR tensor of one sub-chain
        "p_sin": _f(14, 9),
        "p_cos": _f(14, 9),
        "mono_factors": _i(0, 3),
        "po_rc": _i(0, 2),
        "po_mono": _i(0),
        "po_coeff": _f(0),
        "q_rc": _i(0, 2),
        "q_mono": _i(0),
        "q_coeff": _f(0),
        "t12": _f(12),
        "alpha": _f(6),
        "a": _f(6),
        "d": _f(6),
        "theta_offset": _f(6),
        "t_pre_inv": np.eye(4),
        "t_post_inv": np.eye(4),
        "linearity_joint": 0,
        "left_bilinear": np.array([1, 2], np.int32),
        "right_bilinear": np.array([3, 4], np.int32),
        "drop_joint": 5,
        # HP
        "t_u": _f(4, 8, 2),
        "t_w_pre": _f(4, 8, 2),
        "sigma_e": _f(8),
        "dh_a": _f(5),
        "dh_l": _f(5),
        "dh_d": _f(4),
        "t_z_neg_d1": np.eye(4),
        "t_joint6_offset_inv": np.eye(4),
        "drop_idx": 7,
        "f": _f(9, 7),
        "g": _f(6, 5),
        # decomposition, feasible arcs, chart tails
        "R": np.eye(3),
        "n1": np.array([0.0, 0.0, 1.0]),
        "n2": np.array([0.0, 1.0, 0.0]),
        "n3": np.array([0.0, 0.0, 1.0]),
        "coeffs": _f(2, 5),
        "swept": _i(1),
        "grid": np.linspace(0.0, 1.0, 4),
        "tangent": _f(2, 7),
    }
    if fn_name == "feasible_arcs_test":
        v["lo"], v["hi"] = -np.ones(2), np.ones(2)
    if "jointlock" in fn_name:  # 16 locked samples, each a 6R sub-chain
        k = 16
        v.update(
            q_lock=_f(k),
            alpha=_f(k, 6),
            a=_f(k, 6),
            d=_f(k, 6),
            theta_offset=_f(k, 6),
            t_pre_inv=_tile(np.eye(4), k),
            t_post_inv=_tile(np.eye(4), k),
            linearity_joint=_i(k),
            left_bilinear=_tile(np.array([1, 2], np.int32), k),
            right_bilinear=_tile(np.array([3, 4], np.int32), k),
            drop_joint=_i(k, fill=5),
            t_u=_f(k, 4, 8, 2),
            t_w_pre=_f(k, 4, 8, 2),
            dh_a=_f(k, 5),
            dh_l=_f(k, 5),
            dh_d=_f(k, 4),
            t_z_neg_d1=_tile(np.eye(4), k),
            t_joint6_offset_inv=_tile(np.eye(4), k),
            right_pv=_i(k),
            drop_idx=_i(k, fill=7),
            sub_axes=_tile(_tile([0.0, 0.0, 1.0], 6), k),
            sub_t_left=_tile(_tile(np.eye(4), 6), k),
            sub_t_right=_tile(_tile(np.eye(4), 6), k),
            sub_types=_i(k, 6),
            **{key: [v[key]] * k for key in _RR_LISTS},
        )
    return v


def _arg_names(fn: Any) -> list[str]:
    """Parameter names, in order, from a pybind signature in ``__doc__``."""
    doc = fn.__init__.__doc__ if isinstance(fn, type) else fn.__doc__
    line = (doc or "").strip().splitlines()[0] if doc else ""
    inside = line[line.find("(") + 1 : line.rfind(")")]
    names = []
    depth = 0
    cur = ""
    for ch in inside:
        depth += ch in "[("
        depth -= ch in "])"
        if ch == "," and depth == 0:
            names.append(cur.split(":")[0].split("=")[0].strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        names.append(cur.split(":")[0].split("=")[0].strip())
    return [n for n in names if n != "self"]


def binding_names(ext: Any) -> list[str]:
    """Every public function and class of the extension the probe covers."""
    return sorted(
        n
        for n in dir(ext)
        if not n.startswith("_") and callable(getattr(ext, n)) and n not in _SKIP
    )


def binding_baseline(ext: Any, name: str) -> tuple[list[str], tuple]:
    names = _arg_names(getattr(ext, name))
    values = _baseline_values(name)
    missing = [n for n in names if n not in values]
    if missing:
        raise KeyError(f"{name}: no baseline for {missing}")
    return names, tuple(values[n] for n in names)


def binding_mutations(names: list[str], args: tuple) -> Iterator[tuple[str, str, tuple]]:
    """``(label, argument, args)``: one malformed variant per argument -- one
    rank too many, one too short, NaN/inf in the target or seed, an index out of
    range, an unknown enum value, a cap below -1."""
    for i, (pname, a) in enumerate(zip(names, args, strict=True)):

        def put(v: Any, i: int = i) -> tuple:
            return (*args[:i], v, *args[i + 1 :])

        if isinstance(a, np.ndarray):
            yield f"{pname} rank+1", pname, put(a[None])
            if a.ndim >= 2:
                yield f"{pname} short inner", pname, put(a[..., :-1])
            if a.size and pname not in _FREE_LENGTH:
                yield f"{pname} short", pname, put(a[:-1])
            if pname in _FINITE_ARGS:
                for label, val in (("NaN", np.nan), ("inf", np.inf)):
                    bad = a.copy()
                    bad.flat[-1] = val
                    yield f"{pname} {label}", pname, put(bad)
            if pname in _INDEX_INTS and a.size:
                bad = a.copy()
                bad.flat[-1] = 99
                yield f"{pname}=99", pname, put(bad)
        elif isinstance(a, list):
            yield f"{pname} list short", pname, put(a[:-1])
        elif pname in _INDEX_INTS:
            yield f"{pname}=99", pname, put(99)
            yield f"{pname}=-1", pname, put(-1)
        elif pname == "seed_metric":
            yield "seed_metric='bogus'", pname, put("bogus")
        elif pname == "respect_limits":
            yield "respect_limits=7", pname, put(7)
        elif pname == "max_solutions":
            yield "max_solutions=-7", pname, put(-7)


_SH_ONLY_CASES: tuple[tuple[str, str, str, str, tuple], ...] = (
    ("SphericalShoulderCharts", "domain", "chart=999", "chart", (999,)),
    ("SphericalShoulderCharts", "nonempty", "chart=999", "chart", (999,)),
)
# Chart-object methods, on an object built for a real arm:
# (class, method, label, argument, args).
_METHOD_CASES: tuple[tuple[str, str, str, str, tuple], ...] = (
    tuple(
        case
        for cls in ("SphericalShoulderCharts", "SrsCharts")
        for case in (
            (cls, "q", "chart=999", "chart", (999, np.zeros(3))),
            (cls, "q", "chart=-1", "chart", (-1, np.zeros(3))),
            (cls, "tangent", "chart=999", "chart", (999, np.zeros(3))),
            (cls, "in_limits", "chart=999", "chart", (999, np.zeros(7), np.ones(7))),
            (cls, "in_limits", "lo short", "lo", (0, np.zeros(3), np.ones(7))),
            (cls, "locate", "q short", "q", (np.zeros(3), 1e-6)),
            (cls, "param_of", "q short", "q", (np.zeros(3),)),
        )
    )
    + _SH_ONLY_CASES
)


def _chart_objects() -> dict[str, Any]:
    """One native chart object per class, built by the chart API for a real arm."""
    import ssik.chart as chart
    from ssik import Manipulator

    built: dict[str, Any] = {}
    real = chart._native_ext

    class Proxy:
        def __init__(self, ext: Any) -> None:
            self._ext = ext

        def __getattr__(self, name: str) -> Any:
            attr = getattr(self._ext, name)
            if name not in ("SphericalShoulderCharts", "SrsCharts"):
                return attr

            def make(*a: Any, **k: Any) -> Any:
                built[name] = attr(*a, **k)
                return built[name]

            return make

    chart._native_ext = lambda: Proxy(real())
    try:
        for arm in CHART_ARMS:
            T, _ = _valid(_module(arm))
            Manipulator.from_prebuilt(arm).self_motion(T, native=True)
    finally:
        chart._native_ext = real
    return built


def binding_jobs() -> list[Job]:
    from ssik._native import _load_ext

    ext = _load_ext()
    jobs: list[Job] = []
    for name in binding_names(ext):
        names, args = binding_baseline(ext, name)
        jobs += [(name, "binding", label) for label, _, _ in binding_mutations(names, args)]
    jobs += [(f"{c}.{m}", "binding", label) for c, m, label, _, _ in _METHOD_CASES]
    return jobs


def binding_argument(name: str, label: str) -> str:
    """The argument a binding job malforms; its error message must name it."""
    if "." in name:
        cls, method = name.split(".")
        return next(a for c, m, lab, a, _ in _METHOD_CASES if (c, m, lab) == (cls, method, label))
    from ssik._native import _load_ext

    names, args = binding_baseline(_load_ext(), name)
    return next(arg for lab, arg, _ in binding_mutations(names, args) if lab == label)


def _run_binding(name: str, label: str) -> str:
    from ssik._native import _load_ext

    ext = _load_ext()
    if "." in name:
        cls, method = name.split(".")
        obj = _chart_objects()[cls]
        margs = next(c[4] for c in _METHOD_CASES if (c[0], c[1], c[2]) == (cls, method, label))
        return _outcome(lambda: getattr(obj, method)(*margs))
    names, args = binding_baseline(ext, name)
    fn = getattr(ext, name)
    mutated = next(a for lab, _, a in binding_mutations(names, args) if lab == label)
    return _outcome(lambda: fn(*mutated))


# ---------------------------------------------------------------------------
# Child protocol and driver
# ---------------------------------------------------------------------------


def _child(jobs_path: str, start: int) -> None:
    jobs = [tuple(j) for j in json.loads(Path(jobs_path).read_text(encoding="utf-8"))]
    for i in range(start, len(jobs)):
        tgt, backend, case = jobs[i]
        print(json.dumps({"i": i, "start": True}), flush=True)
        out = _run_binding(tgt, case) if backend == "binding" else _run_public(tgt, backend, case)
        print(json.dumps({"i": i, "out": out}), flush=True)


def run_jobs(
    jobs: list[Job], *, timeout_s: float = _TIMEOUT_S, verbose: bool = False
) -> list[dict]:
    """Run the jobs in a child process, restarted after a crash or a timeout,
    and return one row per job: target, backend, case and outcome."""
    results: dict[int, str] = {}

    def log(i: int) -> None:
        if verbose:
            print(f"[{i + 1}/{len(jobs)}] {' '.join(jobs[i])}: {results[i]}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as d:
        jobs_path = Path(d) / "jobs.json"
        jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
        start = 0
        while start < len(jobs):
            cmd = [sys.executable, __file__, "--child", str(jobs_path), str(start)]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
            )
            assert proc.stdout is not None
            current = start
            timed_out = False
            deadline = [time.monotonic() + timeout_s]

            def watchdog(p: subprocess.Popen[str] = proc, deadline: list[float] = deadline) -> None:
                nonlocal timed_out
                while p.poll() is None:
                    if time.monotonic() > deadline[0]:
                        timed_out = True
                        p.kill()
                        return
                    time.sleep(0.2)

            threading.Thread(target=watchdog, daemon=True).start()
            for line in proc.stdout:
                rec = json.loads(line)
                current = rec["i"]
                deadline[0] = time.monotonic() + timeout_s
                if "out" in rec:
                    results[current] = rec["out"]
                    log(current)
            rc = proc.wait()
            if current not in results:
                if timed_out:
                    results[current] = "TIMEOUT"
                elif rc < 0:
                    results[current] = f"CRASH:{signal.Signals(-rc).name}"
                else:
                    results[current] = f"CHILD-EXIT:{rc}"
                log(current)
            start = current + 1
    return [
        {"target": t, "backend": b, "case": c, "outcome": results.get(i, "missing")}
        for i, (t, b, c) in enumerate(jobs)
    ]


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2], int(sys.argv[3]))
        return
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("probe", nargs="?", choices=("public", "bindings", "all"), default="all")
    ap.add_argument("--json", type=Path, help="also write the rows here, as JSON")
    a = ap.parse_args()
    jobs: list[Job] = []
    if a.probe in ("public", "all"):
        jobs += public_jobs()
    if a.probe in ("bindings", "all"):
        jobs += binding_jobs()
    rows = run_jobs(jobs, verbose=True)
    for r in rows:
        print(f"{r['target']:34s} {r['backend']:13s} {r['case']:36s} {r['outcome']}")
    bad = [r for r in rows if r["outcome"].startswith(("CRASH", "TIMEOUT", "CHILD"))]
    print(f"\n{len(rows)} cases, {len(bad)} crash/timeout")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
