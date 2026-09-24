"""Seeded artifact solves stay inside the joint limits.

The artifacts filter limits first, then rank by the seed, calling the seed
stage with ``respect_limits=False`` to mean "the limit pass is done". Before
#567 that also made the seed stage limit-blind: it picked each joint's winding
nearest the seed, and on the Panda 60% of seeded solutions came back outside
the joint box (median 1.36 rad past a stop). #567 made ``rewrap_to_seed``
limit-aware; these tests pin that, on both backends.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik import _native
from ssik.kinematics.poe_fk import poe_forward_kinematics

_ARMS = ["franka_panda_ik", "ur5_ik"]


def _outside(q: np.ndarray, lim: np.ndarray) -> float:
    return float(np.max(np.maximum(lim[:, 0] - q, q - lim[:, 1])))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("arm", _ARMS)
def test_seeded_solutions_stay_in_limits(arm: str, native: bool) -> None:
    if native and not _native.native_available():
        pytest.skip("native extension not built")
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    lim = np.array([j.limits for j in kb.joints])
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lim])
        T = poe_forward_kinematics(kb, q)
        seed = np.array([rng.uniform(lo, hi) for lo, hi in lim])
        sols = mod.solve(T, q_seed=seed, native=native)
        for s in sols:
            assert _outside(s.q, lim) <= 1e-9, (arm, native, s.q)
            checked += 1
        # The seeded set is the unseeded one, reordered: nothing lost to the seed.
        assert len(sols) == len(mod.solve(T, native=native))
    assert checked > 100


def test_seeded_nearest_is_the_nearest_holdable_posture() -> None:
    """The top seeded solution is the in-limits representative nearest the seed,
    not a winding outside the box that happens to be nearer."""
    mod = importlib.import_module("ssik.prebuilt.franka_panda_ik")
    kb = mod._KB
    lim = np.array([j.limits for j in kb.joints])
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lim])
        T = poe_forward_kinematics(kb, q)
        seed = np.array([rng.uniform(lo, hi) for lo, hi in lim])
        full = mod.solve(T)
        if not full:
            continue
        best = min(float(np.max(np.abs(s.q - seed))) for s in full)
        (top,) = mod.solve(T, q_seed=seed, max_solutions=1)
        assert _outside(top.q, lim) <= 1e-9
        assert float(np.max(np.abs(top.q - seed))) == pytest.approx(best, abs=1e-9)
