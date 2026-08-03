"""Adapter exposing the native C++ solver as a ``three_parallel.solve`` drop-in.

Lets the existing Python test suite validate the C++ backend without re-writing
any assertions (#499): the same tests run against ``{python, cpp}``. The C++
extension (``ssik_cpp_ext``, built by ``scripts/build_cpp_ext.py`` into
``cpp/build``) is test-only -- never shipped, never a runtime dependency. Tests
skip the C++ backend when it isn't built.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_REPO = Path(__file__).resolve().parent.parent
_BUILD = _REPO / "cpp" / "build"

_ext: Any = None


def _load_ext() -> Any:
    global _ext
    if _ext is not None:
        return _ext
    if str(_BUILD) not in sys.path:
        sys.path.insert(0, str(_BUILD))
    try:
        import ssik_cpp_ext  # type: ignore[import-not-found]

        _ext = ssik_cpp_ext
    except ImportError:
        _ext = False
    return _ext


def cpp_available() -> bool:
    """True when the native extension is importable (built)."""
    return bool(_load_ext())


@dataclass
class _CppSolution:
    """Minimal Solution stand-in carrying what the tests read (``.q`` etc.)."""

    q: NDArray[np.float64]
    fk_residual: float
    refinement_used: str


def cpp_three_parallel_solve(
    kb: Any,
    t_target: NDArray[np.float64],
    policy: Any = None,
    *,
    allow_refinement: bool = False,
    refinement_max_iters: int = 15,
    max_solutions: int | None = None,
) -> tuple[list[_CppSolution], bool]:
    """Call the native three_parallel solver with the same contract as
    :func:`ssik.solvers.ikgeo.three_parallel.solve`.

    ``policy`` / ``max_solutions`` are accepted for signature compatibility; the
    native path uses its baked tolerances and returns the full set.
    """
    ext = _load_ext()
    if not ext:
        raise RuntimeError("ssik_cpp_ext not built; run scripts/build_cpp_ext.py")
    if len(kb.joints) != 6:
        raise ValueError(f"three_parallel requires a 6-DOF chain; got {len(kb.joints)} joints")

    axes = np.array([j.axis for j in kb.joints], dtype=np.float64)
    t_left = np.array([j.T_left for j in kb.joints], dtype=np.float64)
    t_right = np.array([j.T_right for j in kb.joints], dtype=np.float64)
    types = np.array([0 if j.joint_type == "revolute" else 1 for j in kb.joints], dtype=np.int32)

    qs, resids, is_ls = ext.three_parallel_solve(
        axes,
        t_left,
        t_right,
        types,
        np.asarray(t_target, dtype=np.float64),
        allow_refinement,
        refinement_max_iters,
    )
    sols = [
        _CppSolution(
            q=np.asarray(qs[i], dtype=np.float64),
            fk_residual=float(resids[i]),
            refinement_used="lm" if allow_refinement else "none",
        )
        for i in range(len(qs))
    ]
    if max_solutions is not None:
        sols = sols[:max_solutions]
    return sols, bool(is_ls)


def cpp_artifact_solve(
    kb: Any,
    t_target: NDArray[np.float64],
    *,
    max_solutions: int | None = None,
    q_seed: NDArray[np.float64] | None = None,
    respect_limits: bool = True,
    allow_refinement: bool = False,
    allow_rescue: bool = True,
    policy: Any = None,
    refinement_max_iters: int = 15,
    seed_metric: str = "wrap_linf",
    seed_tolerance: float | None = None,
) -> list[_CppSolution]:
    """Native replica of the ``<arm>_ik.solve()`` artifact contract (#503).

    Same signature + defaults as the generated artifact ``solve()``: force-refined
    core solve, then finalize (limits -> seed -> truncate). Returns
    ``list[Solution]`` (the artifact returns a bare list, not a ``(sols, is_ls)``
    tuple).
    """
    ext = _load_ext()
    if not ext:
        raise RuntimeError("ssik_cpp_ext not built; run scripts/build_cpp_ext.py")
    if seed_tolerance is not None and q_seed is None:
        raise ValueError("seed_tolerance requires q_seed")
    if len(kb.joints) != 6:
        raise ValueError(f"three_parallel requires a 6-DOF chain; got {len(kb.joints)} joints")

    axes = np.array([j.axis for j in kb.joints], dtype=np.float64)
    t_left = np.array([j.T_left for j in kb.joints], dtype=np.float64)
    t_right = np.array([j.T_right for j in kb.joints], dtype=np.float64)
    types = np.array([0 if j.joint_type == "revolute" else 1 for j in kb.joints], dtype=np.int32)
    lo = np.array([j.limits[0] if j.limits else 0.0 for j in kb.joints], dtype=np.float64)
    hi = np.array([j.limits[1] if j.limits else 0.0 for j in kb.joints], dtype=np.float64)
    has_limits = np.array([1 if j.limits else 0 for j in kb.joints], dtype=np.int32)

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
        _CppSolution(
            q=np.asarray(qs[i], dtype=np.float64),
            fk_residual=float(resids[i]),
            refinement_used="lm" if int(refine[i]) == 1 else "none",
        )
        for i in range(len(qs))
    ]
