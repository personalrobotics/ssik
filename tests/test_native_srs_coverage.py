"""Every SRS-class arm is native=True accelerated, and native matches non-native.

Guards the invariant that the pip ``native=True`` backend covers EVERY concurrent-
axis SRS arm -- the canonical-ZYZ core (iiwa14/xmatepro7) AND the general
Davenport core (#354; iiwa7/r1pro/openarm) -- with no silent Python fallback, and
that the native path is bit-for-bit set-equivalent to the pure-Python solve. If a
future change re-narrows the native gate (as it once did, only covering canonical
arms), this turns red.

Skips when the test-only extension isn't built (the cpp CI job builds it).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik._native import srs_native_geometry, try_native_srs_algebraic
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.prebuilt._manifest import load_manifest
from tests._cpp_backend import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")

# Every shipped seven_r.srs prebuilt (the family whose native core this covers).
_SRS_ARMS = sorted(a.name for a in load_manifest().values() if a.solver == "seven_r.srs")


def _wrap_linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)))


def _agree(x: list[np.ndarray], y: list[np.ndarray], tol: float = 1e-6) -> bool:
    if len(x) != len(y):
        return False
    used = [False] * len(y)
    for a in x:
        for k, b in enumerate(y):
            if not used[k] and _wrap_linf(a, b) <= tol:
                used[k] = True
                break
        else:
            return False
    return True


@pytest.mark.parametrize("arm", _SRS_ARMS)
def test_native_covers_and_matches(arm: str) -> None:
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    assert srs_native_geometry(kb) is not None, f"{arm}: SRS arm not enrolled in the native gate"

    lims = [
        (float(j.limits[0]), float(j.limits[1])) if j.limits else (-np.pi, np.pi) for j in kb.joints
    ]
    rng = np.random.default_rng(0)
    routed_native = mismatch = nonempty = 0
    for _ in range(60):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        T = poe_forward_kinematics(kb, q)
        if try_native_srs_algebraic("seven_r.srs", kb, np.asarray(T)) is not None:
            routed_native += 1
        na = [np.asarray(s.q) for s in mod.solve(T, native=True)]
        py = [np.asarray(s.q) for s in mod.solve(T, native=False)]
        if not _agree(na, py):
            mismatch += 1
        nonempty += len(py) > 0

    assert routed_native == 60, f"{arm}: native core did not fire on {60 - routed_native}/60 poses"
    assert mismatch == 0, f"{arm}: native != non-native on {mismatch}/60 poses"
    assert nonempty > 45, f"{arm}: only {nonempty}/60 nonempty -- fuzz not exercising the arm"
