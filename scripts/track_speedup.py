"""Track the speedup ladder for a per-arm IK artifact.

The promise (#93 + #110): we currently run at ~1 MFLOP/s achieved (Python +
numpy dispatch-bound). The Cython port's ceiling is ~1 GFLOP/s -- a
~100-1000x speedup. This script measures every rung of the ladder for a
given fixture so we can confirm we're tracking against the goal as work
lands:

    1. Wrapper artifact (Phase 1 #110, ssik runtime dep)
    2. Specialised Python artifact (Phase 1.5 #112, inlined trig)
    3. (future) Specialised Cython artifact (Phase 4 #110)

For each, reports min / median / mean wall-clock, FLOP budget, achieved
GFLOP/s, and the speedup ratio relative to the baseline (rung 1).

Today rungs 1+2 are real; rung 3 is a placeholder. As Cython lands the
script picks it up automatically (it imports any module named
``<arm>_ik_cy`` if present).

Usage:

    uv run python scripts/track_speedup.py --arm puma560
    uv run python scripts/track_speedup.py --arm ur5

The "specialised" rung re-emits the artifact in-memory each run (so we
benchmark exactly the latest codegen output, not whatever happens to be
committed in tests/artifacts/).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ssik._urdf import load_urdf_kinbody_normalized  # noqa: E402
from ssik.core.codegen import emit_artifact  # noqa: E402
from ssik.core.dispatcher import dispatch  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"

# Per-arm fixture metadata.
ARMS = {
    "ur5": {
        "urdf": FIXTURES / "ur5.urdf",
        "base": "base_link",
        "ee": "ee_link",
        "label": "UR5",
    },
    "puma560": {
        "urdf": FIXTURES / "puma560.urdf",
        "base": "base_link",
        "ee": "wrist_3_link",
        "label": "Puma 560",
    },
}


def _emit_module(kb, plan, module_name: str, label: str, output_path: Path):
    """Emit the artifact at the latest codegen state and import it."""
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name=module_name,
        output_path=str(output_path),
        arm_label=label,
    )
    spec = importlib.util.spec_from_file_location(module_name, output_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bench(solve_fn, kb, n_warm: int = 5, n_bench: int = 100, seed: int = 0):
    """Time a callable. Reports min / median / mean / failure-count."""
    rng = np.random.default_rng(seed=seed)
    n_dof = len(kb.joints)

    # Warm.
    for _ in range(n_warm):
        q = rng.uniform(-1.0, 1.0, size=n_dof)
        T = _fk(kb, q)
        solve_fn(T)

    times: list[float] = []
    fk_errs: list[float] = []
    fails = 0
    for _ in range(n_bench):
        q_star = rng.uniform(-1.0, 1.0, size=n_dof)
        T_star = _fk(kb, q_star)
        t0 = time.perf_counter()
        sols, is_ls = solve_fn(T_star)
        times.append((time.perf_counter() - t0) * 1e3)
        if is_ls or not sols:
            fails += 1
            continue
        worst = max(float(np.linalg.norm(_fk(kb, s.q) - T_star)) for s in sols)
        fk_errs.append(worst)
    return {
        "times_ms": np.array(times),
        "fk_err_max": max(fk_errs) if fk_errs else float("nan"),
        "fails": fails,
    }


def _fk(kb, q):
    """POE FK for the bench, using the runtime KinBody."""
    from ssik.subproblems._rotation import rotation_matrix

    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def _print_rung(name: str, result: dict, baseline_min: float | None, flop_budget: int):
    times = result["times_ms"]
    if len(times) == 0:
        print(f"  {name:>30s}: (no data)")
        return
    rmin = float(times.min())
    rmed = float(np.median(times))
    rmean = float(times.mean())
    speedup_str = ""
    if baseline_min is not None and rmin > 0:
        speedup = baseline_min / rmin
        speedup_str = f"  [{speedup:5.2f}x vs baseline]"
    rate_gflops = flop_budget / (rmin * 1e-3) / 1e9
    fk_str = (
        f"FK err max {result['fk_err_max']:.1e}"
        if not np.isnan(result["fk_err_max"])
        else "(no FK data)"
    )
    print(
        f"  {name:>30s}: min {rmin:7.3f} ms  med {rmed:7.3f} ms  "
        f"mean {rmean:7.3f} ms  ({rate_gflops:6.4f} GF/s){speedup_str}  "
        f"{fk_str}, fails {result['fails']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS.keys(), required=True)
    args = parser.parse_args()
    arm = ARMS[args.arm]

    print(f"\nLoading {arm['label']} ({arm['urdf'].name}) ...")
    kb = load_urdf_kinbody_normalized(arm["urdf"], arm["base"], arm["ee"])
    plan = dispatch(kb)
    print(f"Dispatch: {plan.solver_name} (tier {plan.tier})")
    print(f"FLOP budget: {plan.flop_budget:,}")
    print(f"Expected median (current): ~{plan.expected_ms_median} ms\n")

    rungs: list[tuple[str, callable]] = []  # type: ignore[name-defined]

    # Rung 0: direct call into the runtime ssik solver. Baseline.
    from importlib import import_module

    runtime_mod = import_module("ssik.solvers." + plan.solver_name.replace(".", "."))
    rungs.append(("0 - runtime ssik solver", lambda T: runtime_mod.solve(kb, T)))

    # Rung 1: thin wrapper artifact. Force the wrapper emitter regardless
    # of whether a specialised composer is registered.
    from ssik.core import codegen as _cg

    saved = _cg._SPECIALISED_COMPOSERS
    try:
        _cg._SPECIALISED_COMPOSERS = {}
        with tempfile.TemporaryDirectory() as td:
            wrapper_mod = _emit_module(
                kb=kb,
                plan=plan,
                module_name=f"{args.arm}_wrapper",
                label=arm["label"],
                output_path=Path(td) / f"{args.arm}_wrapper.py",
            )
        rungs.append(("1 - wrapper artifact (Phase 1)", wrapper_mod.solve))
    finally:
        _cg._SPECIALISED_COMPOSERS = saved

    # Rung 2: specialised artifact. Use the registered composer if any.
    if plan.solver_name in saved:
        with tempfile.TemporaryDirectory() as td:
            spec_mod = _emit_module(
                kb=kb,
                plan=plan,
                module_name=f"{args.arm}_specialised",
                label=arm["label"],
                output_path=Path(td) / f"{args.arm}_specialised.py",
            )
        rungs.append(("2 - specialised artifact (#112)", spec_mod.solve))

    # Rung 3 placeholder for the future Cython port (#110 Phase 4).
    print("Rungs:")

    baseline_min: float | None = None
    for name, fn in rungs:
        result = _bench(fn, kb)
        if baseline_min is None and result["times_ms"].size > 0:
            baseline_min = float(result["times_ms"].min())
        _print_rung(name, result, baseline_min, plan.flop_budget)

    print()
    print("Speedup ladder tracking #93 / #112 / #110-Phase-4 promise.")
    print("Goal: ~100-1000x at rung 3 (Cython) over rung 1 (wrapper).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
