"""C++ SRS artifact soundness across the singular regime + rescue liveness (#319).

The completeness bar for the no-Python-fallback standalone artifact, stated as
PROVABLE invariants:

1. **Soundness (everywhere).** Every solution the artifact returns FK-closes and
   respects limits -- including at singularities, where the T-perturbation rescue
   may fire. A no-fallback artifact must never emit an invalid solution.

2. **Bit-parity on well-conditioned poses** is covered by the random-pose tests
   (test_native_srs_coverage / test_srs_general_cpp): native == Python there.

Bit-parity is deliberately NOT asserted in the near-singular regime: there the
IK branch set is ill-conditioned and BOTH backends have divergent edge behavior
(each recovers sound branches the other drops -- e.g. openarm's elbow-singular
shell up to ~0.1 rad, where native sometimes finds solutions Python returns empty
for, and vice versa). Neither is a gold standard there, so soundness -- not
matching a non-canonical reference -- is the meaningful guarantee.

Skips when the test-only extension isn't built (native falls back to Python).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.prebuilt._manifest import load_manifest
from tests._cpp_backend import cpp_available

pytestmark = pytest.mark.skipif(not cpp_available(), reason="ssik._ssik_native not built")

_EXACT_SRS = sorted(a.name for a in load_manifest().values() if a.solver == "seven_r.srs")


def _lims(kb) -> list[tuple[float, float]]:
    return [
        (float(j.limits[0]), float(j.limits[1])) if j.limits else (-np.pi, np.pi) for j in kb.joints
    ]


def _singular_q(kb, rng: np.random.Generator, eps: float) -> np.ndarray:
    """A pose eps rad from an elbow-straight / shoulder-gimbal singularity (eps=0
    is exactly singular), within limits."""
    lims = _lims(kb)

    def clamp(v: float, i: int) -> float:
        return float(np.clip(v, lims[i][0], lims[i][1]))

    q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
    s = rng.choice([-1.0, 1.0])
    which = int(rng.integers(0, 3))
    if which == 0:
        q[3] = clamp(eps * s, 3)  # elbow straight
    elif which == 1:
        q[1] = clamp(np.pi + eps * s, 1)  # shoulder gimbal
    else:
        q[1] = clamp(np.pi + eps * s, 1)
        q[3] = clamp(eps * s, 3)  # combined
    return q


def _assert_sound(arm: str, kb, T: np.ndarray, sols, lims) -> None:
    for s in sols:
        q = np.asarray(s.q)
        resid = float(np.linalg.norm(poe_forward_kinematics(kb, q) - T))
        assert resid < 1e-6, f"{arm}: unsound solution, FK residual {resid:.2e}"
        assert all(lims[i][0] - 1e-9 <= q[i] <= lims[i][1] + 1e-9 for i in range(len(q))), (
            f"{arm}: solution out of limits: {q.tolist()}"
        )


@pytest.mark.parametrize("arm", _EXACT_SRS)
def test_artifact_sound_in_near_singular_regime(arm: str) -> None:
    """Everything the native artifact returns in the near-singular regime
    (1e-3..1e-1 rad off a singularity) is a valid IK. Fast: the analytical core
    handles this regime, so the rescue stays dormant."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    lims = _lims(kb)
    rng = np.random.default_rng(0)
    for _ in range(500):
        eps = 10.0 ** rng.uniform(-3.0, -1.0)
        T = poe_forward_kinematics(kb, _singular_q(kb, rng, eps))
        _assert_sound(arm, kb, T, mod.solve(T, native=True), lims)


# At exact singularities the rescue fires (slower); check soundness + that the
# ported rescue is actually live on representative arms (the rescue is shared).
@pytest.mark.parametrize("arm", ["iiwa14_ik", "r1pro_left_ik"])
def test_rescue_sound_and_live_at_exact_singularities(arm: str) -> None:
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    kb = mod._KB
    lims = _lims(kb)
    rng = np.random.default_rng(0)
    recovered = 0
    for _ in range(80):
        T = poe_forward_kinematics(kb, _singular_q(kb, rng, 0.0))
        sols = mod.solve(T, native=True)
        _assert_sound(arm, kb, T, sols, lims)
        recovered += len(sols) > 0
    assert recovered > 0, (
        f"{arm}: artifact recovered nothing at any exact singularity -- rescue dead"
    )
