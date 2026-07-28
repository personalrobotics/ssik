"""Perf-regression gate: no shipped arm's solve() may slow down grossly (#436).

A solver refactor that regresses ``solve()`` speed several-fold on some arm class
would ship silently today -- the bulletproof suite checks coverage + FK closure,
not time, and the manifest ``[bench]`` numbers are informational (never asserted).

This gate measures each arm's solve time **relative to a reference arm** (ur5) in
the same run and compares that ratio to a committed baseline
(``tests/_perf_baseline.json``, from ``scripts/regen_perf_baseline.py``). The
ratio is machine-independent -- both arm and reference scale with CPU speed -- so
it holds on any CI runner, unlike absolute ms (the manifest ``ms_mean`` was
measured on a dev machine and would false-fail on a Linux runner).

Only *slowdowns* fail: ``measured_ratio > baseline_ratio * _TOL``. Speedups never
fail (they just leave the baseline stale-high until the next regen). ``_TOL`` = 3
absorbs the ~1.4x run-to-run noise floor of the relative measurement while still
catching a genuine >=3x algorithmic regression.

Runs under ``@pytest.mark.perf`` (CI's serial, non-parallel step -- worker
contention under ``-n auto`` inflates timing, #348).
"""

from __future__ import annotations

import functools
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from _perf import best_call_ms

from ssik.prebuilt._manifest import load_manifest

_BASELINE = json.loads((Path(__file__).parent / "_perf_baseline.json").read_text(encoding="utf-8"))
_REFERENCE: str = _BASELINE["reference"]
_RATIOS: dict[str, float] = _BASELINE["ratios"]
_MANIFEST = load_manifest()

# Slowdown factor over baseline that trips the gate. 3x clears the ~1.4x
# relative-measurement noise floor with margin; a real algorithmic regression
# (an O(n) that became O(n^2), a dropped fast-path) is many-fold, not 3x noise.
_TOL = 3.0
_RUNS = 15


def _regressed(measured_ratio: float, baseline_ratio: float, tol: float = _TOL) -> bool:
    """True iff ``measured_ratio`` exceeds ``baseline_ratio`` by more than
    ``tol``. Pure so the threshold is unit-testable without timing (teeth test)."""
    return measured_ratio > baseline_ratio * tol


@functools.cache
def _solve_ms(name: str) -> float:
    """Best-of-N solve time for one arm at its manifest sample_q. Cached so the
    reference is measured once and each arm once per session."""
    arm = _MANIFEST[name]
    mod = importlib.import_module(arm.hier_module or f"ssik.prebuilt.{name}")
    t_target = mod.fk(np.array(arm.sample_q))
    return float(best_call_ms(lambda: mod.solve(t_target), runs=_RUNS))


_ARMS = sorted(_RATIOS)


@pytest.mark.perf
@pytest.mark.parametrize("arm_name", _ARMS)
def test_solve_time_not_regressed(arm_name: str) -> None:
    """Each arm's solve time relative to ur5 must not exceed its committed
    baseline ratio by more than ``_TOL``. A failure means a solver change made
    this arm several-fold slower -- profile it; do not regen the baseline to
    silence it unless the slowdown is understood and intended."""
    baseline = _RATIOS[arm_name]
    measured = _solve_ms(arm_name) / _solve_ms(_REFERENCE)
    assert not _regressed(measured, baseline), (
        f"{arm_name}: solve time is {measured:.2f}x ur5 vs baseline {baseline:.2f}x "
        f"(> {_TOL}x tolerance) -- a ~{measured / baseline:.1f}x relative slowdown. "
        f"Profile the solver; regenerate tests/_perf_baseline.json only if this "
        f"slowdown is understood and intended."
    )


def test_reference_arm_present() -> None:
    """The reference arm must be a shipped prebuilt (the ratios are meaningless
    otherwise)."""
    assert _REFERENCE in _MANIFEST, f"perf-baseline reference {_REFERENCE!r} not in manifest"


def test_baseline_covers_benched_arms() -> None:
    """Every benched arm must have a baseline ratio, so a newly-onboarded arm
    can't silently escape the perf gate. Regenerate the baseline when adding an
    arm (scripts/regen_perf_baseline.py)."""
    benched = {n for n, a in _MANIFEST.items() if a.bench is not None}
    missing = sorted(benched - set(_RATIOS))
    assert not missing, (
        f"benched arms with no perf baseline: {missing}. "
        f"Run scripts/regen_perf_baseline.py and commit tests/_perf_baseline.json."
    )


def test_gate_has_teeth() -> None:
    """The threshold flags a gross slowdown and ignores noise / speedups."""
    assert _regressed(10.0, 3.0)  # 10x vs 3x baseline (>9x) -> real regression
    assert _regressed(1.0, 0.3)  # 1.0x vs 0.3x baseline (>0.9x) -> 3.3x slower
    assert not _regressed(3.5, 3.0)  # within 3x noise band
    assert not _regressed(0.5, 5.0)  # speedup never fails
