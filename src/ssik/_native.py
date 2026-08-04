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

# Solver families with a native implementation in _ssik_native.
_NATIVE_SOLVERS = frozenset({"ikgeo.three_parallel"})

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


def _consts(kb: Any) -> tuple[Any, ...]:
    cached = _consts_cache.get(id(kb))
    if cached is not None:
        return cached
    joints = kb.joints
    marshalled = (
        np.array([j.axis for j in joints], dtype=np.float64),
        np.array([j.T_left for j in joints], dtype=np.float64),
        np.array([j.T_right for j in joints], dtype=np.float64),
        np.array([0 if j.joint_type == "revolute" else 1 for j in joints], dtype=np.int32),
        np.array([j.limits[0] if j.limits else 0.0 for j in joints], dtype=np.float64),
        np.array([j.limits[1] if j.limits else 0.0 for j in joints], dtype=np.float64),
        np.array([1 if j.limits else 0 for j in joints], dtype=np.int32),
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

    axes, t_left, t_right, types, lo, hi, has_limits = _consts(kb)
    has_seed = q_seed is not None
    seed_arr = (
        np.asarray(q_seed, dtype=np.float64) if has_seed else np.zeros(len(kb.joints), np.float64)
    )
    qs, resids, refine = ext.three_parallel_artifact_solve(
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
