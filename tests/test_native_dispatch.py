"""Opt-in native dispatch on the artifact solve() (#507).

Two things:

1. **Fallback** (runs everywhere): with the native extension unavailable,
   ``solve(native=True)`` returns exactly the Python result. Forced by
   monkeypatching the loader, so it holds even where the ext *is* built.
2. **Native-active parity** (runs where ``ssik._ssik_native`` is importable --
   the cpp CI job builds it into ``src/ssik``): ``solve(native=True)`` reproduces
   ``solve()`` across the three_parallel family and its feature matrix. The
   extension's own correctness is covered by ``test_three_parallel_artifact``;
   this validates the wiring (dispatch + constant marshalling in ssik._native).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from ssik import _native
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.prebuilt._manifest import load_manifest

# Redundant / approximate 7R families: native samples the self-motion manifold via
# a different (sound) enumeration than Python, so exact set-match is not the runtime
# contract -- oracle-coverage completeness is validated by the standalone artifact
# gate (#550/#487), and the runtime path is the SAME C++ function. The runtime gate
# here is soundness + non-emptiness (test_native_relative_completeness).
_RELATIVE_NATIVE_SOLVERS = frozenset(
    {
        "seven_r.srs_polished",
        "seven_r.spherical_shoulder",
        "seven_r.spherical_shoulder_polished",
        "jointlock.seven_r",
    }
)

_ARMS = [
    arm.name
    for arm in load_manifest().values()
    if arm.solver in _native._NATIVE_SOLVERS and arm.solver not in _RELATIVE_NATIVE_SOLVERS
]
_RELATIVE_ARMS = [
    arm.name for arm in load_manifest().values() if arm.solver in _RELATIVE_NATIVE_SOLVERS
]


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _close(a: Any, b: Any, tol: float = 1e-3) -> bool:
    return all(abs(_wrap(float(x - y))) < tol for x, y in zip(a, b, strict=True))


def _set_match(A: list[Any], B: list[Any], tol: float = 1e-3) -> bool:
    return len(A) == len(B) and all(any(_close(a.q, b.q, tol) for b in B) for a in A)


def _subset(A: list[Any], B: list[Any], tol: float = 1e-3) -> bool:
    return all(any(_close(a.q, b.q, tol) for b in B) for a in A)


def test_native_true_falls_back_when_unavailable(monkeypatch: Any) -> None:
    """native=True silently returns the Python result when the ext is absent."""
    monkeypatch.setattr(_native, "_ext", None)
    monkeypatch.setattr(_native, "_ext_tried", True)
    assert not _native.native_available()

    mod = importlib.import_module("ssik.prebuilt.ur5_ik")
    t = mod.fk(np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2]))
    py = mod.solve(t)
    nat = mod.solve(t, native=True)
    assert len(py) == len(nat)
    assert all(_close(a.q, b.q) for a, b in zip(py, nat, strict=True)), "fallback must be identical"


def test_native_kwarg_accepted() -> None:
    """The native kwarg exists on three_parallel artifacts (accepts True/False)."""
    mod = importlib.import_module("ssik.prebuilt.ur5_ik")
    t = mod.fk(np.zeros(6))
    assert mod.solve(t, native=False) is not None
    assert mod.solve(t, native=True) is not None  # True is safe even without the ext


@pytest.mark.skipif(
    not _native.native_available(),
    reason="ssik._ssik_native not built (see scripts/build_cpp_ext.py --out-dir src/ssik)",
)
@pytest.mark.parametrize("arm_name", _ARMS)
def test_native_matches_python(arm_name: str) -> None:
    """solve(native=True) reproduces solve() across the feature matrix."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    kb = mod._KB
    ranges = [
        (float(j.limits[0]), float(j.limits[1])) if j.limits else (-np.pi, np.pi) for j in kb.joints
    ]
    rng = np.random.default_rng(2)
    for _ in range(25):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t = np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)
        seed = q + rng.uniform(-0.2, 0.2, len(q))

        assert _set_match(mod.solve(t), mod.solve(t, native=True)), f"{arm_name}: full-set"
        assert _set_match(
            mod.solve(t, respect_limits=False), mod.solve(t, native=True, respect_limits=False)
        ), f"{arm_name}: respect_limits=False"

        py_full = mod.solve(t)
        cpp_m = mod.solve(t, native=True, max_solutions=3)
        assert len(cpp_m) == min(3, len(py_full)), f"{arm_name}: max count"
        assert _subset(cpp_m, py_full), f"{arm_name}: max subset"

        py_s = mod.solve(t, q_seed=seed)
        nat_s = mod.solve(t, q_seed=seed, native=True)
        assert len(py_s) == len(nat_s), f"{arm_name}: seed count"
        if nat_s:
            assert _close(py_s[0].q, nat_s[0].q), f"{arm_name}: nearest-to-seed"

        # seed + max: same count and the nearest (top-1, the tracking guarantee)
        # matches. Deeper ordering is not asserted -- at a near-singular pose two
        # solutions can be near-equidistant to the seed, so which lands k-th flips
        # by BLAS backend (the #56 representative ambiguity).
        py_sm = mod.solve(t, q_seed=seed, max_solutions=2)
        nat_sm = mod.solve(t, q_seed=seed, max_solutions=2, native=True)
        assert len(py_sm) == len(nat_sm), f"{arm_name}: seed+max count"
        if nat_sm:
            assert _close(py_sm[0].q, nat_sm[0].q), f"{arm_name}: seed+max nearest"


@pytest.mark.skipif(
    not _native.native_available(),
    reason="ssik._ssik_native not built (see scripts/build_cpp_ext.py --out-dir src/ssik)",
)
@pytest.mark.parametrize("arm_name", _RELATIVE_ARMS)
def test_native_relative_completeness(arm_name: str) -> None:
    """Redundant/approximate 7R runtime contract (#554): native=True returns SOUND
    solutions (every returned q is a real IK) and is non-empty exactly when the
    Python path is, on stable (non-continuum) poses. It may sample the self-motion
    manifold differently from native=False -- oracle-coverage is the standalone
    artifact gate's job (#550), and the runtime calls the same C++ solve."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    kb = mod._KB
    ranges = [
        (float(j.limits[0]), float(j.limits[1])) if j.limits else (-np.pi, np.pi) for j in kb.joints
    ]
    rng = np.random.default_rng(2)
    kept = 0
    attempts = 0
    while kept < 15 and attempts < 800:
        attempts += 1
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t = np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)
        py = mod.solve(t)
        # Drop near-continuum / near-singular poses (mirror the standalone golden:
        # count <= 130 and stable under a 1e-6 perturbation), where native and
        # Python legitimately return different sound subsets of the manifold.
        if len(py) > 130:
            continue
        stable = all(
            len(mod.solve(np.asarray(poe_forward_kinematics(kb, q + dq), dtype=np.float64)))
            == len(py)
            for dq in (rng.uniform(-1e-6, 1e-6, len(q)), rng.uniform(-1e-6, 1e-6, len(q)))
        )
        if not stable:
            continue
        kept += 1
        nat = mod.solve(t, native=True)
        for s in nat:  # soundness: every native solution is a real IK
            assert np.linalg.norm(poe_forward_kinematics(kb, s.q) - t) <= 1e-6, (
                f"{arm_name}: native returned an unsound solution"
            )
        # non-emptiness parity: native finds solutions exactly when Python does.
        assert bool(nat) == bool(py), f"{arm_name}: native/python emptiness disagree"
    assert kept >= 10, f"{arm_name}: too few stable poses sampled ({kept})"
