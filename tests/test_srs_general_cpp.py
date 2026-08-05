"""C++ general-path SRS artifact solve vs the shipped Python prebuilts (#354).

``srs_general_solve`` (the Davenport shoulder+wrist swivel sweep) covers the
concurrent-axis SRS arms the canonical ZYZ fast-path cannot: iiwa7 (offset
wrist), r1pro and openarm (non-ZYZ shoulder/wrist). The self-contained C++
artifact must match each arm's shipped Python prebuilt across the full solve()
contract, so these arms become deployable zero-Python artifacts too.

Skips when the test-only extension isn't built (the cpp CI job builds it).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY as _POL
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.seven_r.srs import (  # type: ignore[attr-defined]
    _arm_constants,
    _classify_srs_7r_geometric,
)
from tests._cpp_backend import _load_ext, cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")

# General-path SRS prebuilts: iiwa7 = offset wrist; r1pro/openarm = non-ZYZ.
_ARMS = ["iiwa7_ik", "r1pro_left_ik", "openarm_left_ik"]


def _bake(kb: Any) -> dict[str, Any]:
    cls = _classify_srs_7r_geometric(kb, _POL)
    assert cls is not None
    l_se, l_ew, ee_offset, origins = _arm_constants(kb, cls)
    j = kb.joints
    # use_canonical mirror: canonical-ZYZ + offset-free routes to the canonical
    # path; everything else (these arms) to the general Davenport path.
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
        "upper_home": np.asarray(upper, dtype=np.float64),
        "forearm_home": np.asarray(cls.wrist_pivot - origins[cls.elbow_index], dtype=np.float64),
        "general_path": not (canonical and offset_free),
        "lo": np.array(lo, dtype=np.float64),
        "hi": np.array(hi, dtype=np.float64),
        "has_limits": np.array(has, dtype=np.int32),
    }


def _cpp_solve(a: dict[str, Any], T: np.ndarray, **kw: Any) -> list[np.ndarray]:
    q_seed = kw.get("q_seed")
    qs, _resids, _refine = _load_ext().srs_artifact_solve(
        a["axes"],
        a["t_left"],
        a["t_right"],
        a["types"],
        a["l_se"],
        a["l_ew"],
        a["ee_offset"],
        a["shoulder_pivot"],
        a["r_post_wrist"],
        a["elbow_index"],
        a["upper_home"],
        a["forearm_home"],
        a["lo"],
        a["hi"],
        a["has_limits"],
        np.asarray(T, dtype=np.float64),
        general_path=a["general_path"],
        respect_limits=kw.get("respect_limits", True),
        has_seed=q_seed is not None,
        q_seed=(np.asarray(q_seed, dtype=np.float64) if q_seed is not None else np.zeros(7)),
        max_solutions=(-1 if kw.get("max_solutions") is None else int(kw["max_solutions"])),
    )
    return [np.asarray(q) for q in qs]


def _wrap_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)))


def _agree(py: list[np.ndarray], cpp: list[np.ndarray], tol: float = 1e-6) -> bool:
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


@pytest.mark.parametrize("arm", _ARMS)
def test_general_artifact_matches_prebuilt(arm: str) -> None:
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    a = _bake(kb)
    assert a["general_path"], f"{arm} should route to the general path"
    lims = [(float(lo), float(hi)) for lo, hi in zip(a["lo"], a["hi"], strict=True)]

    rng = np.random.default_rng(0)
    worst = 0.0
    n_nonempty = 0
    for _ in range(120):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        T = poe_forward_kinematics(kb, q)
        for rl in (True, False):
            py = [np.asarray(s.q) for s in mod.solve(T, respect_limits=rl)]
            cpp = _cpp_solve(a, T, respect_limits=rl)
            assert _agree(py, cpp), f"{arm} respect_limits={rl}: {len(py)} vs {len(cpp)}"
            if py:
                n_nonempty += 1
                worst = max(
                    worst,
                    float(max(np.linalg.norm(poe_forward_kinematics(kb, c) - T) for c in cpp)),
                )
        # seed ranking + truncation
        seed = q + rng.uniform(-0.05, 0.05, 7)
        for ms in (None, 1):
            py = [np.asarray(s.q) for s in mod.solve(T, q_seed=seed, max_solutions=ms)]
            cpp = _cpp_solve(a, T, q_seed=seed, max_solutions=ms)
            assert _agree(py, cpp), f"{arm} seed+max={ms}: {len(py)} vs {len(cpp)}"

    assert n_nonempty > 150, f"{arm}: only {n_nonempty} nonempty solves"
    assert worst < 1e-9, f"{arm}: worst C++ FK closure {worst:.2e}"
