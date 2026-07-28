"""Regenerate ``tests/_perf_baseline.json`` -- the relative solve-time baseline
for the perf-regression gate (#436).

Each arm's baseline is its ``solve()`` wall-clock time **relative to a reference
arm** (ur5), measured with the same best-of-N noise-floor estimator the gate
uses. Ratios are machine-independent (both arm and reference scale with CPU
speed), so the committed baseline is comparable against a measurement on any CI
runner -- unlike absolute ms, which the manifest ``[bench]`` blocks hold and
which drift across machines.

Run this on the reference machine after a deliberate, understood solver
speed change (a new feature that legitimately costs time), then commit the
updated JSON alongside. Do NOT run it to silence a gate failure you don't
understand -- that is the papering-over the gate exists to prevent.

    uv run python scripts/regen_perf_baseline.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tests" / "fixtures"))

from _perf import best_call_ms  # noqa: E402

from ssik.prebuilt._manifest import load_manifest  # noqa: E402

_REFERENCE = "ur5_ik"
_RUNS = 15
_BASELINE_PATH = _ROOT / "tests" / "_perf_baseline.json"


def _solve_ms(name: str) -> float:
    arm = load_manifest()[name]
    mod = importlib.import_module(arm.hier_module or f"ssik.prebuilt.{name}")
    t_target = mod.fk(np.array(arm.sample_q))
    return float(best_call_ms(lambda: mod.solve(t_target), runs=_RUNS))


def main() -> int:
    manifest = load_manifest()
    ref_ms = _solve_ms(_REFERENCE)
    ratios: dict[str, float] = {}
    for name, arm in manifest.items():
        if arm.bench is None:
            continue  # not yet benched; skip
        ratios[name] = round(_solve_ms(name) / ref_ms, 3)
    ratios = dict(sorted(ratios.items()))
    payload = {
        "_note": (
            "Relative solve-time baseline for the perf-regression gate "
            "(tests/test_perf_regression.py, #436). ratios[arm] = "
            "solve_ms(arm) / solve_ms(reference). Regenerate with "
            "scripts/regen_perf_baseline.py on the reference machine."
        ),
        "reference": _REFERENCE,
        "ratios": ratios,
    }
    _BASELINE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(ratios)} ratios to {_BASELINE_PATH.relative_to(_ROOT)} (ref={_REFERENCE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
