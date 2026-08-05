"""Self-contained C++ SRS artifact solve vs the shipped Python prebuilt (#515).

``ssik::srs_artifact_solve`` is the pure-C++ replica of the generated
``<arm>_ik.solve()`` for the seven_r.srs family: seeded_track fast path ->
srs_canonical_solve -> finalize(limits -> seed -> truncate) with the #359
in-limits fallback wired to resolve_in_limits. This is THE gate for a
deployable artifact -- a C++ consumer #includes ``<arm>.hpp`` and calls
``solve(T)`` with zero runtime Python.

The oracle is the shipped ``ssik.prebuilt.kuka.iiwa14_ik`` module (itself the
codegen output). Every artifact-parameter path (respect_limits, seed ranking,
seed tolerance, max_solutions, and the max_solutions==1 + seed seeded_track
fast path) must agree with it.

Skips when the test-only extension isn't built (the cpp CI job builds it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY as _POL
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.prebuilt.kuka import iiwa14_ik
from ssik.solvers.seven_r.srs import (  # type: ignore[attr-defined]
    _arm_constants,
    _classify_srs_7r_geometric,
)
from tests._cpp_backend import _load_ext, cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")

_KB = iiwa14_ik._KB


def _baked() -> dict[str, Any]:
    """SrsConsts (base + branch extras) + JointLimits baked from the prebuilt KB."""
    cls = _classify_srs_7r_geometric(_KB, _POL)
    assert cls is not None
    l_se, l_ew, ee_offset, origins = _arm_constants(_KB, cls)
    j = _KB.joints
    lo, hi, has = [], [], []
    for jt in j:
        lh = jt.limits
        if lh is None or lh[0] is None or lh[1] is None:
            lo.append(-np.pi)
            hi.append(np.pi)
            has.append(0)
        else:
            lo.append(float(lh[0]))
            hi.append(float(lh[1]))
            has.append(1)
    return {
        "axes": np.array([jt.axis for jt in j], dtype=np.float64),
        "t_left": np.array([jt.T_left for jt in j], dtype=np.float64),
        "t_right": np.array([jt.T_right for jt in j], dtype=np.float64),
        "types": np.array([0 if jt.joint_type == "revolute" else 1 for jt in j], dtype=np.int32),
        "l_se": float(l_se),
        "l_ew": float(l_ew),
        "ee_offset": np.asarray(ee_offset, dtype=np.float64),
        "shoulder_pivot": np.asarray(cls.shoulder_pivot, dtype=np.float64),
        "r_post_wrist": np.asarray(j[6].T_right[:3, :3], dtype=np.float64),
        "elbow_index": int(cls.elbow_index),
        "upper_home": np.asarray(origins[cls.elbow_index] - cls.shoulder_pivot, dtype=np.float64),
        "forearm_home": np.asarray(cls.wrist_pivot - origins[cls.elbow_index], dtype=np.float64),
        "lo": np.array(lo, dtype=np.float64),
        "hi": np.array(hi, dtype=np.float64),
        "has_limits": np.array(has, dtype=np.int32),
    }


_ARGS = _baked()
_LIMS = [(float(a), float(b)) for a, b in zip(_ARGS["lo"], _ARGS["hi"], strict=True)]


def _cpp_solve(T: np.ndarray, **kw: Any) -> tuple[list[np.ndarray], list[int]]:
    q_seed = kw.get("q_seed")
    qs, _resids, refine = _load_ext().srs_artifact_solve(
        _ARGS["axes"],
        _ARGS["t_left"],
        _ARGS["t_right"],
        _ARGS["types"],
        _ARGS["l_se"],
        _ARGS["l_ew"],
        _ARGS["ee_offset"],
        _ARGS["shoulder_pivot"],
        _ARGS["r_post_wrist"],
        _ARGS["elbow_index"],
        _ARGS["upper_home"],
        _ARGS["forearm_home"],
        _ARGS["lo"],
        _ARGS["hi"],
        _ARGS["has_limits"],
        np.asarray(T, dtype=np.float64),
        respect_limits=kw.get("respect_limits", True),
        has_seed=q_seed is not None,
        q_seed=(np.asarray(q_seed, dtype=np.float64) if q_seed is not None else np.zeros(7)),
        seed_metric=kw.get("seed_metric", "wrap_linf"),
        has_seed_tolerance=kw.get("seed_tolerance") is not None,
        seed_tolerance=float(kw.get("seed_tolerance") or 0.0),
        max_solutions=(-1 if kw.get("max_solutions") is None else int(kw["max_solutions"])),
        allow_rescue=kw.get("allow_rescue", True),
    )
    return [np.asarray(q) for q in qs], [int(r) for r in refine]


def _py_solve(T: np.ndarray, **kw: Any) -> list[np.ndarray]:
    return [np.asarray(s.q) for s in iiwa14_ik.solve(T, **kw)]


def _wrap_linf(a: np.ndarray, b: np.ndarray) -> float:
    """L-infinity of the wrap-to-pi joint difference: a solution at q=+pi and its
    2*pi twin q=-pi are the same physical config (they appear interchangeably
    when respect_limits=False leaves branches unwrapped)."""
    return float(np.max(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)))


def _agree(py: list[np.ndarray], cpp: list[np.ndarray], tol: float = 1e-7) -> bool:
    """Bijective set agreement: each py solution has a distinct wrap-close cpp
    match. Robust to ordering (a sorted zip misaligns among near-tied branches)."""
    if len(py) != len(cpp):
        return False
    used = [False] * len(cpp)
    for p in py:
        for k, c in enumerate(cpp):
            if not used[k] and _wrap_linf(p, c) <= tol:
                used[k] = True
                break
        else:
            return False
    return True


def test_artifact_matches_prebuilt() -> None:
    rng = np.random.default_rng(0)
    nonempty = seeded = 0
    for _ in range(150):
        q = np.array([rng.uniform(a, b) for a, b in _LIMS])
        T = poe_forward_kinematics(_KB, q)

        # Full solve (all branches), limits on and off.
        for rl in (True, False):
            py = _py_solve(T, respect_limits=rl)
            cpp, _ = _cpp_solve(T, respect_limits=rl)
            assert _agree(py, cpp), f"respect_limits={rl}: {len(py)} vs {len(cpp)}"
            nonempty += len(py) > 0

        # Seed ranking + max_solutions truncation.
        seed = q + rng.uniform(-0.05, 0.05, 7)
        for ms in (None, 1, 4):
            py = _py_solve(T, q_seed=seed, max_solutions=ms)
            cpp, _ = _cpp_solve(T, q_seed=seed, max_solutions=ms)
            assert _agree(py, cpp), f"seed+max_solutions={ms}: {len(py)} vs {len(cpp)}"
            if ms == 1:
                seeded += 1

        # Seed tolerance (drops solutions outside the band).
        py = _py_solve(T, q_seed=seed, seed_tolerance=0.3)
        cpp, _ = _cpp_solve(T, q_seed=seed, seed_tolerance=0.3)
        assert _agree(py, cpp), f"seed_tolerance: {len(py)} vs {len(cpp)}"

    assert nonempty > 200, f"only {nonempty} nonempty solves -- fuzz not exercising the arm"
    assert seeded > 100, "seeded_track fast path not exercised"
