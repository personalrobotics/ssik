"""Regenerate per-arm benchmark numbers in ``MANIFEST.toml`` -- both ssik's own
``solve()`` (the ``[bench]`` block) and the EAIK comparison (the ``[eaik]``
block), which together drive the README tables.

Benchmarking a prebuilt arm used to be hand work: measure ssik ``solve()``,
separately run EAIK, then hand-edit the manifest / a lookup table in
``regen_docs``. This automates both measurements and the manifest write, so
adding an arm (or refreshing the whole set) is one command::

    uv run python scripts/regen_bench.py                 # all arms (ssik + EAIK)
    uv run python scripts/regen_bench.py --arm xarm7_ik  # just one
    uv run python scripts/regen_bench.py --eaik-only     # leave ssik numbers alone
    uv run python scripts/regen_bench.py --docs          # + regenerate docs

Methodology (matches ``examples/04_compare_vs_eaik.py``): reachable poses
sampled in the joint interior, a warm-up pass, then bootstrap mean ± 95% CI on
solve time, plus worst FK residual and the branch-count range, over the **same
poses** for ssik and EAIK. ``respect_limits=False`` so the branch count reflects
the analytical solver, not the limit postprocess. The EAIK side records its
support verdict + numbers, or its verbatim refusal (its kinematic-family string
or the first sentence of its error) -- EAIK is fed each manufacturer fixture
as-is (no manual joint-locking), the same chain ssik is given.

EAIK needs the ``[bench]`` extra (``pip install ssik[bench]``); without it the
``[eaik]`` blocks are left untouched.

Timing is **machine-dependent** -- run on the reference machine that produced
the committed numbers (sanity-check: ``franka_panda`` should land within CI of
its current value). FK and branch counts are machine-independent. The manifest
is updated in place, preserving comments and hand-curated fields.
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


def bench_arm(mod: object, poses: list[NDArray[np.float64]]) -> dict[str, float | int]:
    """Measure one prebuilt module's ``solve()``; return the manifest bench dict."""
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


FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _load_eaik(arm: object) -> object:
    """Load an EAIK robot for ``arm`` from its manifest fixture (URDF) or DH
    (specs). Raises on failure (captured as a refusal by the caller)."""
    import eaik.IK_DH  # type: ignore[import-untyped]
    import eaik.IK_URDF  # type: ignore[import-untyped]

    if arm.fixture_kind == "urdf":  # type: ignore[attr-defined]
        return eaik.IK_URDF.UrdfRobot(str(FIXTURES_DIR / arm.fixture))  # type: ignore[attr-defined]
    # specs arm: rebuild the KinBody to extract DH for EAIK's DH adapter.
    sys.path.insert(0, str(FIXTURES_DIR))
    from ssik._kinbody import build_kinbody
    from ssik.kinematics.poe_to_dh import poe_to_dh

    mod = importlib.import_module(arm.fixture)  # type: ignore[attr-defined]
    kb = build_kinbody(getattr(mod, arm.specs_fn)())  # type: ignore[attr-defined]
    dh = poe_to_dh(kb)
    return eaik.IK_DH.DhRobot(dh.alpha, dh.a, dh.d)


def _clean_refusal(msg: str) -> str:
    """Condense an EAIK error into one concise table-friendly sentence (first
    sentence, no trailing punctuation)."""
    first = msg.strip().split(". ")[0].strip().rstrip(".!")
    return first[:90]


def bench_eaik(arm: object, poses: list[NDArray[np.float64]]) -> dict[str, object] | None:
    """Measure EAIK on ``arm`` over the same ``poses`` as the ssik bench.

    Returns the manifest ``[eaik]`` dict: ``{supported: True, ...}`` with
    timing/FK/branch numbers, or ``{supported: False, refusal: <verbatim>}``.
    Returns ``None`` if EAIK isn't installed (caller leaves the block untouched).
    """
    try:
        import eaik  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        robot = _load_eaik(arm)
    except Exception as exc:
        return {"supported": False, "refusal": _clean_refusal(str(exc))}

    if not robot.hasKnownDecomposition():  # type: ignore[attr-defined]
        return {"supported": False, "refusal": str(robot.getKinematicFamily())}  # type: ignore[attr-defined]

    try:
        robot.IK(poses[0])  # type: ignore[attr-defined]
    except Exception as exc:
        return {"supported": False, "refusal": _clean_refusal(str(exc))}

    for T in poses[:_N_WARMUP]:
        robot.IK(T)  # type: ignore[attr-defined]

    times: list[float] = []
    fk_residuals: list[float] = []
    sol_counts: list[int] = []
    for T in poses[_N_WARMUP:]:
        t0 = time.perf_counter()
        sol = robot.IK(T)  # type: ignore[attr-defined]
        times.append((time.perf_counter() - t0) * 1e3)
        is_ls = np.asarray(sol.is_LS, dtype=bool)
        Q = np.asarray(sol.Q)
        exact = Q[~is_ls] if Q.size else Q
        if len(exact):
            sol_counts.append(len(exact))
            fk_residuals.append(
                max(float(np.linalg.norm(robot.fwdKin(np.asarray(q)) - T)) for q in exact)  # type: ignore[attr-defined]
            )

    mu, half = _mean_ci95(np.array(times))
    return {
        "supported": True,
        "family": str(robot.getKinematicFamily()),  # type: ignore[attr-defined]
        "ms_mean": mu,
        "ms_ci95": half,
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


def _toml_str(s: str) -> str:
    """Quote a string as a TOML basic string (escape backslash + double-quote)."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _eaik_block_lines(name: str, eaik: dict[str, object]) -> list[str]:
    """Render the ``[arms.<name>.eaik]`` block (variable schema: supported vs
    refused)."""
    out = [f"[arms.{name}.eaik]\n"]
    if not eaik["supported"]:
        out.append("supported = false\n")
        out.append(f"refusal = {_toml_str(str(eaik['refusal']))}\n")
        return out
    out.append("supported = true\n")
    out.append(f"family = {_toml_str(str(eaik['family']))}\n")
    out.append(f"ms_mean = {float(eaik['ms_mean']):.6g}\n")  # type: ignore[arg-type]
    out.append(f"ms_ci95 = {float(eaik['ms_ci95']):.6g}\n")  # type: ignore[arg-type]
    out.append(f"max_fk = {float(eaik['max_fk']):.1e}\n")  # type: ignore[arg-type]
    out.append(f"sols_min = {int(eaik['sols_min'])}\n")  # type: ignore[call-overload]
    out.append(f"sols_max = {int(eaik['sols_max'])}\n")  # type: ignore[call-overload]
    return out


def update_manifest_eaik(name: str, eaik: dict[str, object]) -> None:
    """Insert or replace the whole ``[arms.<name>.eaik]`` block in place. The
    block's keys depend on whether EAIK supports the arm, so (unlike the bench
    block) the entire block is rewritten, and inserted after ``[<name>.bench]``
    if absent."""
    lines = MANIFEST.read_text().splitlines(keepends=True)
    header = f"[arms.{name}.eaik]"
    block = _eaik_block_lines(name, eaik)
    start = next((i for i, ln in enumerate(lines) if ln.strip() == header), None)
    if start is not None:
        end = start + 1
        while end < len(lines) and not lines[end].lstrip().startswith("["):
            end += 1
        lines[start:end] = block
    else:
        # Insert after the arm's bench block (which ends at the next section).
        bench_hdr = f"[arms.{name}.bench]"
        b = next((i for i, ln in enumerate(lines) if ln.strip() == bench_hdr), None)
        if b is None:
            raise KeyError(f"neither {header} nor {bench_hdr} found in {MANIFEST}")
        end = b + 1
        while end < len(lines) and not lines[end].lstrip().startswith("["):
            end += 1
        # Ensure a single blank line before the inserted block.
        insert = [*([] if end > 0 and lines[end - 1].strip() == "" else ["\n"]), *block, "\n"]
        lines[end:end] = insert
    MANIFEST.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", help="benchmark only this arm (e.g. xarm7_ik); default all")
    parser.add_argument("--docs", action="store_true", help="run regen_docs.py afterward")
    parser.add_argument(
        "--no-eaik", action="store_true", help="skip the EAIK comparison measurement"
    )
    parser.add_argument(
        "--eaik-only",
        action="store_true",
        help="measure only EAIK (leave the ssik [bench] numbers untouched)",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    names = [args.arm] if args.arm else list(manifest.keys())
    if args.arm and args.arm not in manifest:
        parser.error(f"unknown arm {args.arm!r}; choices: {', '.join(manifest)}")

    print(f"benchmarking {len(names)} arm(s) -> {MANIFEST.relative_to(REPO_ROOT)}")
    eaik_skipped = False
    for name in names:
        mod = importlib.import_module(f"ssik.prebuilt.{name}")
        poses = _gen_poses(mod, _N_TIMED + _N_WARMUP, np.random.default_rng(0))
        line = f"  {name}:"
        if not args.eaik_only:
            bench = bench_arm(mod, poses)
            update_manifest_bench(name, bench)
            fk = _fmt("max_fk", bench["max_fk"])
            line += (
                f" ssik {bench['ms_mean']} ± {bench['ms_ci95']} ms / "
                f"FK {fk} / {bench['sols_min']}-{bench['sols_max']} sols"
            )
        if not args.no_eaik:
            eaik = bench_eaik(manifest[name], poses)
            if eaik is None:
                eaik_skipped = True
            else:
                update_manifest_eaik(name, eaik)
                line += (
                    f" | EAIK {round(float(eaik['ms_mean']) * 1000)}µs"  # type: ignore[arg-type]
                    if eaik["supported"]
                    else f" | EAIK refuses ({eaik['refusal']})"
                )
        print(line)

    if eaik_skipped:
        print("note: EAIK not installed (pip install ssik[bench]); [eaik] blocks left untouched")

    if args.docs:
        print("running regen_docs.py")
        subprocess.run([sys.executable, str(Path(__file__).parent / "regen_docs.py")], check=True)
    else:
        print("note: run scripts/regen_docs.py to refresh README/quickstart, or pass --docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
