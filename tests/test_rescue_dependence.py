"""Rescue-dependence gate: no shipped arm may lean on the numerical rescue
for coverage (#441).

Every prebuilt ``solve()`` has a T-perturbation rescue (numerical LM polish
from a perturbed re-solve) that backfills when the analytical solver returns
nothing. That safety net is legitimate for genuinely-singular or borderline
poses, but it can also *mask a broken analytical solver*: an arm can report
100% coverage on the default path while its native closed-form solver returns
~0 solutions. That is exactly how iiwa7 (#439) looked pre-fix -- native SRS
returned 0 on every pose (wrong wrist reference), the rescue produced 30
numerically-polished sols, and the arm read as "100% covered".

This gate measures coverage twice per arm -- native (``allow_rescue=False``)
vs rescue-augmented (``allow_rescue=True``) -- and fails if the native solver
supplies less than half of what the rescue-augmented path covers. Calibrated
against a full-roster audit: every shipped arm has native == rescue except the
two YuMi arms (native ~0.93-0.97 of rescued, mild near-singular contribution),
which pass the 0.5 band comfortably; an iiwa7-class regression (native 0.0)
fails it.

See #441 for the audit and threshold rationale.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ssik.prebuilt._manifest import load_manifest

if TYPE_CHECKING:
    from types import ModuleType

_MANIFEST = load_manifest()
_ARMS = list(_MANIFEST.values())

# Coverage is sampled over this many in-limits poses per arm. 50 is enough to
# resolve the 0.5 native/rescued band without O(minutes) runtime on the slower
# (HP / srs_polished) solvers.
_N_POSES = 50

# The native analytical solver must supply at least this fraction of the
# rescue-augmented coverage. 0.5 cleanly separates the mild legitimate rescue
# (YuMi ~0.93) from a systemically-broken native solver (iiwa7-class 0.0),
# with wide margin for pose-sampling noise.
_NATIVE_MIN_RATIO = 0.5


def _rescue_dependent(native_cov: float, rescued_cov: float) -> bool:
    """True if the arm leans on the rescue: the native solver covers less than
    ``_NATIVE_MIN_RATIO`` of what the rescue-augmented path covers. Pure so the
    threshold logic is unit-testable independent of any solver (see the teeth
    test below)."""
    return native_cov < _NATIVE_MIN_RATIO * rescued_cov


def _coverage(mod: ModuleType, *, allow_rescue: bool) -> float:
    """Fraction of ``_N_POSES`` in-limits poses for which ``mod.solve`` returns
    at least one solution, with the rescue on or off. ``respect_limits=False``
    so we measure the analytical solver's reach, not URDF limit filtering; the
    sampled q is in-limits so every target is genuinely reachable."""
    kb = mod._KB  # type: ignore[attr-defined]
    rng = np.random.default_rng(20260728)
    limits = [(j.limits if j.limits is not None else (-np.pi, np.pi)) for j in kb.joints]
    covered = 0
    for _ in range(_N_POSES):
        q = np.array([rng.uniform(lo, hi) for lo, hi in limits])
        t_target = mod.fk(q)  # type: ignore[attr-defined]
        sols = mod.solve(t_target, respect_limits=False, allow_rescue=allow_rescue)  # type: ignore[attr-defined]
        covered += bool(sols)
    return covered / _N_POSES


@pytest.mark.parametrize("arm", _ARMS, ids=[a.name for a in _ARMS])
def test_arm_not_rescue_dependent(arm: object) -> None:
    """No shipped arm may depend on the T-perturbation rescue for coverage:
    the native analytical solver must cover >= 50% of what the rescue-augmented
    path covers. A failure means the native solver likely has a bug or
    unhandled geometry that the rescue is silently papering over (cf. iiwa7,
    #439) -- investigate the solver, don't widen this band."""
    name = arm.name  # type: ignore[attr-defined]
    if arm.known_gaps is not None:  # type: ignore[attr-defined]
        pytest.xfail(f"{name}: {arm.known_gaps.xfail_reason}")  # type: ignore[attr-defined]

    mod = importlib.import_module(arm.hier_module or f"ssik.prebuilt.{name}")  # type: ignore[attr-defined]
    if "allow_rescue" not in mod.solve.__code__.co_varnames:  # pragma: no cover
        pytest.skip(f"{name}: solve() has no allow_rescue param")

    rescued = _coverage(mod, allow_rescue=True)
    native = _coverage(mod, allow_rescue=False)
    assert not _rescue_dependent(native, rescued), (
        f"{name}: native analytical solver covers {native:.0%} of poses but the "
        f"rescue-augmented path covers {rescued:.0%} -- the arm depends on the "
        f"numerical T-perturbation rescue for coverage. The native solver likely "
        f"has a bug or unhandled geometry the rescue is masking (cf. iiwa7 #439). "
        f"Investigate the solver; do not widen this gate."
    )


def test_gate_has_teeth() -> None:
    """The threshold flags an iiwa7-class arm (native 0, rescue full) and
    passes a mildly-rescued arm (YuMi-class native ~0.93)."""
    assert _rescue_dependent(0.0, 1.0)  # iiwa7 pre-#439
    assert _rescue_dependent(0.2, 1.0)  # still systemically rescue-dependent
    assert not _rescue_dependent(0.93, 1.0)  # YuMi: mild legitimate rescue
    assert not _rescue_dependent(1.0, 1.0)  # fully native
    assert not _rescue_dependent(0.4, 0.4)  # low-but-fully-native (hard workspace)
