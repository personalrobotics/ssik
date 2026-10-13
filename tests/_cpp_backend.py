"""Adapter exposing the native C++ solver as a ``three_parallel.solve`` drop-in.

Lets the existing Python test suite validate the C++ backend without re-writing
any assertions (#499): the same tests run against ``{python, cpp}``. Prefers the
shipped ``ssik._ssik_native`` extension (#506) when installed, else falls back to
the test-only build (``scripts/build_cpp_ext.py`` into ``cpp/build``). Tests skip
the C++ backend when neither is available.
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
    # Prefer the shipped extension (ssik._ssik_native, #506) so tests exercise
    # the wheel's native path when installed; fall back to the test-only build
    # under cpp/build produced by scripts/build_cpp_ext.py.
    try:
        from ssik import _ssik_native  # type: ignore[attr-defined]

        _ext = _ssik_native
        return _ext
    except ImportError:
        pass
    if str(_BUILD) not in sys.path:
        sys.path.insert(0, str(_BUILD))
    try:
        import _ssik_native  # type: ignore[import-not-found]

        _ext = _ssik_native
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


def srs_solver(a: dict[str, Any], *, general_path: bool = False) -> Any:
    """``_ssik_native.Solver`` for ``seven_r.srs`` from a test's baked SRS
    arrays (keys as ``srs_resolve_in_limits`` names its arguments)."""
    from ssik._native import _SOLVER_DATA_SCHEMA, _SolverRecord

    arrays: dict[str, Any] = {
        "axis": a["axes"],
        "t_left": a["t_left"],
        "t_right": a["t_right"],
        "joint_type": a["types"],
        "lo": a["lo"],
        "hi": a["hi"],
        "present": a["has_limits"],
        "l_se": np.array(a["l_se"], dtype=np.float64),
        "l_ew": np.array(a["l_ew"], dtype=np.float64),
        "ee_offset_local": a["ee_offset"],
        "shoulder_pivot": a["shoulder_pivot"],
        "r_post_wrist": np.ascontiguousarray(a["r_post_wrist"]),
        "elbow_index": np.array(a["elbow_index"], dtype=np.int32),
        "upper_home": a["upper_home"],
        "forearm_home": a["forearm_home"],
        "general_path": np.array(int(general_path), dtype=np.int32),
    }
    return _load_ext().Solver(_SolverRecord("seven_r.srs", _SOLVER_DATA_SCHEMA, arrays))


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
        raise RuntimeError("_ssik_native not built; run scripts/build_cpp_ext.py")
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
        raise RuntimeError("_ssik_native not built; run scripts/build_cpp_ext.py")
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

    from ssik._native import _SOLVER_DATA_SCHEMA, _SolverRecord

    arrays: dict[str, Any] = {
        "axis": axes,
        "t_left": t_left,
        "t_right": t_right,
        "joint_type": types,
        "lo": lo,
        "hi": hi,
        "present": has_limits,
    }
    solver = ext.Solver(_SolverRecord("ikgeo.three_parallel", _SOLVER_DATA_SCHEMA, arrays))
    qs, resids, refine = solver.solve(
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
