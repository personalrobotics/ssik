"""Artifact-layer parity: the native C++ replica of ``<arm>_ik.solve()`` (#503).

The core solver has parity (#500/#502); this validates the full *artifact
contract* the shipped ``<arm>.solve()`` adds on top -- force-refine, limits
(wrap-into-range + drop), seed ranking (wrap_linf / wrap_l2), seed_tolerance,
max_solutions -- against the Python artifact as the oracle. No assertion logic
is re-written: each check is "the native artifact agrees with the Python
artifact" for a given feature configuration.

Assertions are contract-faithful, not naive equality:

- **Full set**: unordered wrap-close agreement (solution *order* without a seed
  is an unspecified implementation detail -- SP6 root order differs between
  numpy and Eigen).
- **max_solutions without a seed**: a correctly-sized *subset* of the full set
  (the docs define this cap as arbitrary without ``q_seed``).
- **q_seed ranking**: the nearest solution (top-1, the tracking-critical one)
  matches and the ordering is monotonic in seed distance; **q_seed +
  max_solutions**: the nearest-k match in order.
- **seed_tolerance**: unordered wrap-close agreement of the kept set.

Skips when the test-only extension isn't built.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.prebuilt._manifest import load_manifest
from tests._cpp_backend import cpp_artifact_solve, cpp_available

pytestmark = pytest.mark.skipif(
    not cpp_available(), reason="ssik_cpp_ext not built (run scripts/build_cpp_ext.py)"
)

_ARMS = [arm.name for arm in load_manifest().values() if arm.solver == "ikgeo.three_parallel"]
_N_POSES = 40


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _close(a: Any, b: Any, tol: float = 1e-3) -> bool:
    return all(abs(_wrap(float(x - y))) < tol for x, y in zip(a, b, strict=True))


def _set_match(A: list[Any], B: list[Any], tol: float = 1e-3) -> bool:
    return len(A) == len(B) and all(any(_close(a.q, b.q, tol) for b in B) for a in A)


def _subset(A: list[Any], B: list[Any], tol: float = 1e-3) -> bool:
    return all(any(_close(a.q, b.q, tol) for b in B) for a in A)


def _seed_linf(q: Any, seed: Any) -> float:
    return max(abs(_wrap(float(x - s))) for x, s in zip(q, seed, strict=True))


@pytest.mark.parametrize("arm_name", _ARMS)
def test_artifact_contract_parity(arm_name: str) -> None:
    """The native artifact-solve reproduces the Python artifact across every
    solve() feature, on reachable poses for one family arm."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    kb = mod._KB
    ranges = [
        (float(j.limits[0]), float(j.limits[1])) if j.limits else (-np.pi, np.pi) for j in kb.joints
    ]
    rng = np.random.default_rng(1)

    for _ in range(_N_POSES):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        t = np.asarray(poe_forward_kinematics(kb, q), dtype=np.float64)
        seed = q + rng.uniform(-0.2, 0.2, len(q))

        # Full set: unordered agreement + native solutions respect limits.
        py_full = mod.solve(t)
        cpp_full = cpp_artifact_solve(kb, t)
        assert _set_match(py_full, cpp_full), f"{arm_name}: full-set mismatch"
        for s in cpp_full:
            for i, (lo, hi) in enumerate(ranges):
                if kb.joints[i].limits is not None:
                    assert lo <= s.q[i] <= hi, f"{arm_name}: native sol out of limits at joint {i}"

        # respect_limits=False: raw set agreement.
        assert _set_match(
            mod.solve(t, respect_limits=False), cpp_artifact_solve(kb, t, respect_limits=False)
        ), f"{arm_name}: respect_limits=False set mismatch"

        # max_solutions without a seed: correctly-sized subset of the full set.
        cpp_m = cpp_artifact_solve(kb, t, max_solutions=3)
        assert len(cpp_m) == min(3, len(py_full)), f"{arm_name}: max_solutions count mismatch"
        assert _subset(cpp_m, py_full), f"{arm_name}: max_solutions not a subset of the full set"

        # q_seed ranking: same count, nearest matches, monotonic order.
        py_s = mod.solve(t, q_seed=seed)
        cpp_s = cpp_artifact_solve(kb, t, q_seed=seed)
        assert len(py_s) == len(cpp_s), f"{arm_name}: seed-ranked count mismatch"
        if cpp_s:
            assert _close(py_s[0].q, cpp_s[0].q), f"{arm_name}: nearest-to-seed mismatch"
            dists = [_seed_linf(s.q, seed) for s in cpp_s]
            assert all(dists[i] <= dists[i + 1] + 1e-9 for i in range(len(dists) - 1)), (
                f"{arm_name}: seed ranking not monotonic"
            )

        # wrap_l2 metric: same count + nearest matches.
        cpp_l2 = cpp_artifact_solve(kb, t, q_seed=seed, seed_metric="wrap_l2")
        py_l2 = mod.solve(t, q_seed=seed, seed_metric="wrap_l2")
        assert len(py_l2) == len(cpp_l2), f"{arm_name}: wrap_l2 count mismatch"
        if cpp_l2:
            assert _close(py_l2[0].q, cpp_l2[0].q), f"{arm_name}: wrap_l2 nearest mismatch"

        # q_seed + max_solutions: nearest-k in order.
        py_sm = mod.solve(t, q_seed=seed, max_solutions=2)
        cpp_sm = cpp_artifact_solve(kb, t, q_seed=seed, max_solutions=2)
        assert len(py_sm) == len(cpp_sm), f"{arm_name}: seed+max_solutions count mismatch"
        assert all(_close(a.q, b.q) for a, b in zip(py_sm, cpp_sm, strict=True)), (
            f"{arm_name}: seed+max_solutions ordered mismatch"
        )

        # seed_tolerance: unordered agreement of the kept set.
        assert _set_match(
            mod.solve(t, q_seed=seed, seed_tolerance=0.5),
            cpp_artifact_solve(kb, t, q_seed=seed, seed_tolerance=0.5),
        ), f"{arm_name}: seed_tolerance set mismatch"


def test_seed_tolerance_requires_q_seed() -> None:
    """The wrapper validation matches the artifact: seed_tolerance needs q_seed."""
    kb = importlib.import_module("ssik.prebuilt.ur5_ik")._KB
    with pytest.raises(ValueError, match="seed_tolerance requires q_seed"):
        cpp_artifact_solve(kb, np.eye(4), seed_tolerance=0.1)
