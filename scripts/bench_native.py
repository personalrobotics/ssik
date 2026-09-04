"""Native (C++) vs Python benchmark across the prebuilt roster (#554).

Times ``solve(native=False)`` (the annotated-Cython Python path) against
``solve(native=True)`` (the shipped ``_ssik_native`` C++ backend) over the **same**
reachable poses per arm, and writes a committed Markdown table to
``docs/native_benchmark.md``. This is the "blazing fast" evidence for making
``native=True`` the default (#554 step C): every arm must be native-or-faster,
never slower.

    uv run python scripts/bench_native.py                 # all native arms
    uv run python scripts/bench_native.py --arm jaco2_ik  # just one
    uv run python scripts/bench_native.py --out /tmp/x.md # alternate output

Methodology mirrors ``scripts/regen_bench.py``: interior-sampled reachable poses,
a warm-up pass (both paths, so the native sidecar .npz / marshalling caches are
primed), then bootstrap mean +/- 95% CI on per-solve time, over the same poses
for both paths. ``respect_limits=False`` (the raw analytical set; set parity is
gated separately by ``tests/test_native_dispatch.py``, not here).

Timing is **machine-dependent** -- the committed table records the reference
machine; re-run on yours for local numbers. FK residuals are machine-independent.
Requires the compiled perf extensions + the native extension (the guard refuses
to publish numbers measured against an uncompiled dev checkout, #... regen_bench
lesson).
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Reuse the exact sampling + statistics + compiled-extension guard the EAIK
# benchmark uses, so native/python/EAIK numbers are all directly comparable.
from regen_bench import (  # type: ignore[import-not-found]
    _N_TIMED,
    _N_WARMUP,
    _gen_poses,
    _mean_ci95,
    _require_compiled_perf_extensions,
)

from ssik._native import native_available
from ssik.prebuilt._manifest import load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = REPO_ROOT / "docs" / "native_benchmark.md"


def _time_path(mod: object, poses: list[NDArray[np.float64]], *, native: bool) -> dict[str, float]:
    """Warm + time one solve path over the shared pose set."""
    for T in poses[:_N_WARMUP]:
        mod.solve(T, respect_limits=False, native=native)  # type: ignore[attr-defined]
    times: list[float] = []
    worst_fk = 0.0
    sol_counts: list[int] = []
    for T in poses[_N_WARMUP:]:
        t0 = time.perf_counter()
        sols = mod.solve(T, respect_limits=False, native=native)  # type: ignore[attr-defined]
        times.append((time.perf_counter() - t0) * 1e3)
        if sols:
            worst_fk = max(worst_fk, max(s.fk_residual for s in sols))
            sol_counts.append(len(sols))
    mu, half = _mean_ci95(np.array(times))
    return {
        "ms_mean": mu,
        "ms_ci95": half,
        "max_fk": worst_fk,
        "sols_med": float(np.median(sol_counts)) if sol_counts else 0.0,
    }


def bench_arm(arm_name: str) -> dict[str, Any] | None:
    """Native-vs-Python timing for one arm, or None when native is unavailable
    for it (falls back to Python, so no meaningful comparison)."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    poses = _gen_poses(mod, _N_TIMED + _N_WARMUP, np.random.default_rng(0))
    py = _time_path(mod, poses, native=False)
    nat = _time_path(mod, poses, native=True)
    speedup = py["ms_mean"] / nat["ms_mean"] if nat["ms_mean"] > 0 else float("nan")
    return {
        "arm": arm_name,
        "py_ms": py["ms_mean"],
        "py_ci": py["ms_ci95"],
        "nat_ms": nat["ms_mean"],
        "nat_ci": nat["ms_ci95"],
        "speedup": speedup,
        "nat_fk": nat["max_fk"],
        "sols_med": int(nat["sols_med"]),
    }


def _render_md(rows: list[dict[str, Any]]) -> str:
    speeds = np.array([float(r["speedup"]) for r in rows])
    hdr = [
        "# Native (C++) vs Python benchmark",
        "",
        "Per-solve time of `solve(native=True)` (shipped `_ssik_native` C++ backend)",
        "vs `solve(native=False)` (annotated-Cython Python path), same reachable poses",
        f"per arm ({_N_TIMED} timed + {_N_WARMUP} warm-up), bootstrap mean +/- 95% CI,",
        "`respect_limits=False`. Regenerate with `uv run python scripts/bench_native.py`.",
        "",
        f"- **Machine:** {platform.platform()} / Python {platform.python_version()}",
        f"- **Arms:** {len(rows)} native | **median speedup:** {np.median(speeds):.1f}x"
        f" | **range:** {speeds.min():.1f}x - {speeds.max():.1f}x",
        "- Timing is machine-dependent; FK residuals are not. Solution *sets* are",
        "  gated by `tests/test_native_dispatch.py`, not here.",
        "",
        "| Arm | Python (ms) | Native (ms) | Speedup | Native FK | Sols |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda x: -float(x["speedup"])):
        hdr.append(
            f"| {r['arm']} | {r['py_ms']:.2f} ± {r['py_ci']:.2f} "
            f"| {r['nat_ms']:.3f} ± {r['nat_ci']:.3f} "
            f"| {float(r['speedup']):.1f}x | {float(r['nat_fk']):.1e} | {r['sols_med']} |"
        )
    hdr.append("")
    return "\n".join(hdr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", metavar="NAME", help="only these arms")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="output Markdown path")
    parser.add_argument(
        "--allow-uncompiled",
        action="store_true",
        help="measure even against uncompiled perf extensions (numbers not publishable)",
    )
    args = parser.parse_args()

    if not native_available():
        print("error: native extension not built (scripts/build_cpp_ext.py --out-dir src/ssik)")
        return 1
    _require_compiled_perf_extensions(allow_uncompiled=args.allow_uncompiled)

    manifest = load_manifest()
    names = args.arm if args.arm else [a.name for a in manifest.values()]
    rows: list[dict[str, Any]] = []
    for name in names:
        row = bench_arm(name)
        if row is None:
            print(f"  {name}: native unavailable, skipped")
            continue
        rows.append(row)
        print(
            f"  {name}: py {row['py_ms']:.2f} -> native {row['nat_ms']:.3f} ms "
            f"({float(row['speedup']):.1f}x)"
        )

    if not rows:
        print("no arms measured")
        return 1
    args.out.write_text(_render_md(rows))
    speeds = np.array([float(r["speedup"]) for r in rows])
    shown = args.out.relative_to(REPO_ROOT) if args.out.is_relative_to(REPO_ROOT) else args.out
    print(f"\nwrote {shown} ({len(rows)} arms, median {np.median(speeds):.1f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
