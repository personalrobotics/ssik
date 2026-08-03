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
