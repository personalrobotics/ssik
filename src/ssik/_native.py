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
# families expose the full artifact contract via try_native_solve; seven_r.srs
# exposes only its analytical sweep via try_native_srs_algebraic (the Python
# wrapper keeps seeded-track / resolve_in_limits / finalize).
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


# Per-KinBody native-SRS metadata: (is_canonical, marshalled args) or None. The
# canonical-ZYZ + offset-free check + the SRS geometric constants are geometry-
# only, so cache per-arm (keyed on id(kb)).
_srs_cache: dict[int, Any] = {}


def _srs_native_args(kb: Any) -> Any:
    """Marshalled srs_canonical_solve args for a canonical SRS arm, or None when
    the arm isn't the canonical-ZYZ offset-free path the native core supports."""
    if id(kb) in _srs_cache:
        return _srs_cache[id(kb)]
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY as _POL
    from ssik.solvers.seven_r.srs import (  # type: ignore[attr-defined]
        _arm_constants,
        _classify_srs_7r_geometric,
    )

    result = None
    cls = _classify_srs_7r_geometric(kb, _POL)
    if cls is not None and len(kb.joints) == 7:
        l_se, l_ew, ee_offset, origins = _arm_constants(kb, cls)
        upper = origins[cls.elbow_index] - cls.shoulder_pivot
        u_home = upper / np.linalg.norm(upper)
        j = kb.joints
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
        if canonical and offset_free:
            result = (
                np.array([jt.axis for jt in j], dtype=np.float64),
                np.array([jt.T_left for jt in j], dtype=np.float64),
                np.array([jt.T_right for jt in j], dtype=np.float64),
                np.array([0 if jt.joint_type == "revolute" else 1 for jt in j], dtype=np.int32),
                float(l_se),
                float(l_ew),
                np.asarray(ee_offset, dtype=np.float64),
                np.asarray(cls.shoulder_pivot, dtype=np.float64),
                np.asarray(j[6].T_right[:3, :3], dtype=np.float64),
            )
    _srs_cache[id(kb)] = result
    return result


def try_native_srs_algebraic(
    solver_name: str, kb: Any, t_target: NDArray[np.float64]
) -> list[Solution] | None:
    """Native SRS analytical sweep (the expensive part), or None to fall back.

    Returns the deduped, FK-verified analytical candidate set (pre-finalize) --
    the Python artifact keeps the seeded-track / limits / resolve_in_limits /
    finalize postprocess around it. Supports the canonical-ZYZ offset-free path
    (iiwa14 / xmatepro7); other SRS arms (offset/tilted wrist) return None.
    """
    if solver_name != "seven_r.srs":
        return None
    ext = _load_ext()
    if ext is None:
        return None
    args = _srs_native_args(kb)
    if args is None:
        return None
    axes, t_left, t_right, types, l_se, l_ew, ee_offset, shoulder_pivot, r_post = args
    qs, resids = ext.srs_canonical_solve(
        axes,
        t_left,
        t_right,
        types,
        l_se,
        l_ew,
        ee_offset,
        shoulder_pivot,
        r_post,
        np.asarray(t_target, dtype=np.float64),
    )
    return [
        Solution(q=np.asarray(qs[i], dtype=np.float64), fk_residual=float(resids[i]))
        for i in range(len(qs))
    ]
