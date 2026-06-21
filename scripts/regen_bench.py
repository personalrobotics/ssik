"""Regenerate per-arm benchmark numbers in ``MANIFEST.toml``.

Benchmarking a prebuilt arm used to be a hand step: measure ``solve()`` timing /
FK closure / branch count, then hand-edit the ``[arms.<name>.bench]`` block and
re-run ``regen_docs``. This automates the measurement and the manifest write, so
adding an arm (or refreshing the whole set) is one command::

    uv run python scripts/regen_bench.py                 # all arms
    uv run python scripts/regen_bench.py --arm xarm7_ik  # just one
    uv run python scripts/regen_bench.py --docs          # + regenerate docs

The methodology matches ``examples/04_compare_vs_eaik.py`` (the source of the
committed numbers): reachable poses sampled in the joint interior, a warm-up
pass, then bootstrap mean ± 95% CI on solve time, plus worst FK residual and
the branch-count range. ``respect_limits=False`` so the branch count reflects
the analytical solver, not the limit postprocess.

Timing is **machine-dependent** -- run this on the reference machine that
produced the committed numbers (sanity-check: a known arm like ``franka_panda``
should land within CI of its current value). FK and branch counts are
machine-independent.

The manifest is updated in place by surgical line replacement so comments and
hand-curated fields are preserved; only the five ``[bench]`` values change.
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "src" / "ssik" / "prebuilt" / "MANIFEST.toml"

sys.path.insert(0, str(REPO_ROOT))
from ssik.prebuilt._manifest import load_manifest  # noqa: E402

_N_TIMED = 200
_N_WARMUP = 10


def _gen_poses(mod: object, n: int, rng: np.random.Generator) -> list[NDArray[np.float64]]:
    """Reachable poses sampled in the safe joint interior (matches examples/04)."""
    dof = mod.DOF  # type: ignore[attr-defined]
    poses: list[NDArray[np.float64]] = []
    while len(poses) < n:
        q = rng.uniform(-0.5, 0.5, size=dof)
        if dof == 7:
            q[3] = float(rng.uniform(0.3, 0.7))
        try:
            poses.append(mod.fk(q))  # type: ignore[attr-defined]
        except Exception:
            continue
    return poses


def _mean_ci95(samples: NDArray[np.float64], n_boot: int = 1000) -> tuple[float, float]:
    """Bootstrap mean + 95% CI half-width (matches examples/04)."""
    rng = np.random.default_rng(0)
    n = len(samples)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = samples[rng.integers(0, n, size=n)].mean()
    mu = float(samples.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mu, float(max(mu - lo, hi - mu))


def bench_arm(mod: object) -> dict[str, float | int]:
    """Measure one prebuilt module's ``solve()``; return the manifest bench dict."""
    rng = np.random.default_rng(0)
    poses = _gen_poses(mod, _N_TIMED + _N_WARMUP, rng)
    for T in poses[:_N_WARMUP]:
        mod.solve(T, respect_limits=False)  # type: ignore[attr-defined]

    times: list[float] = []
    fk_residuals: list[float] = []
    sol_counts: list[int] = []
    for T in poses[_N_WARMUP:]:
        t0 = time.perf_counter()
        sols = mod.solve(T, respect_limits=False)  # type: ignore[attr-defined]
        times.append((time.perf_counter() - t0) * 1e3)
        if sols:
            fk_residuals.append(max(s.fk_residual for s in sols))
            sol_counts.append(len(sols))

    mu, half = _mean_ci95(np.array(times))
    return {
        "ms_mean": round(mu, 2),
        "ms_ci95": round(half, 2),
        "max_fk": max(fk_residuals) if fk_residuals else float("nan"),
        "sols_min": min(sol_counts) if sol_counts else 0,
        "sols_max": max(sol_counts) if sol_counts else 0,
    }


def _fmt(key: str, value: float | int) -> str:
    if key in ("sols_min", "sols_max"):
        return str(int(value))
    if key == "max_fk":
        return f"{value:.1e}"
    return f"{value:.2f}"


def update_manifest_bench(name: str, bench: dict[str, float | int]) -> None:
    """Surgically rewrite the five values under ``[arms.<name>.bench]`` in place,
    leaving every comment and hand-curated field untouched."""
    lines = MANIFEST.read_text().splitlines(keepends=True)
    header = f"[arms.{name}.bench]"
    start = next((i for i, ln in enumerate(lines) if ln.strip() == header), None)
    if start is None:
        raise KeyError(f"{header} not found in {MANIFEST}")
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("["):  # next section
            break
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in bench:
                lines[i] = f"{key} = {_fmt(key, bench[key])}\n"
    MANIFEST.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", help="benchmark only this arm (e.g. xarm7_ik); default all")
    parser.add_argument("--docs", action="store_true", help="run regen_docs.py afterward")
    args = parser.parse_args()

    manifest = load_manifest()
    names = [args.arm] if args.arm else list(manifest.keys())
    if args.arm and args.arm not in manifest:
        parser.error(f"unknown arm {args.arm!r}; choices: {', '.join(manifest)}")

    print(f"benchmarking {len(names)} arm(s) -> {MANIFEST.relative_to(REPO_ROOT)}")
    for name in names:
        mod = importlib.import_module(f"ssik.prebuilt.{name}")
        bench = bench_arm(mod)
        update_manifest_bench(name, bench)
        print(
            f"  {name}: {bench['ms_mean']} ± {bench['ms_ci95']} ms / "
            f"FK {_fmt('max_fk', bench['max_fk'])} / {bench['sols_min']}-{bench['sols_max']} sols"
        )

    if args.docs:
        print("running regen_docs.py")
        subprocess.run([sys.executable, str(Path(__file__).parent / "regen_docs.py")], check=True)
    else:
        print("note: run scripts/regen_docs.py to refresh README/quickstart, or pass --docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
