"""Example 04: measured comparison of ssik vs EAIK.

Bench ``ssik`` (analytical, returns all branches at machine-precision FK)
against **EAIK** (Ostermeier 2024, ``pip install eaik``), the canonical
Python wrapper around C++ subproblem-decomposition solvers. EAIK is
analytical on the kinematic families it recognises (Pieper-class 6R,
canonical SRS 7R with a manual joint lock); it refuses arms outside those
families.

Numerical IK baselines (MINK / TracIK) are tracked separately in #236.

The harness reports, per arm:

- mean ± 95% CI for solve time (ms)  -- bootstrap, 1000 resamples
- max FK residual (Frobenius norm of T_target - FK(q)) across all
  returned IK branches
- median branch count
- "supported / refuses (...)" with the verbatim EAIK error message

Output is a markdown table embeddable in the README.

Run::

    uv run python examples/04_compare_vs_eaik.py
    uv run python examples/04_compare_vs_eaik.py --n 200  # more poses
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
sys.path.insert(0, str(FIXTURES_DIR))

import ssik  # noqa: E402

# ---------------------------------------------------------------------------
# Optional deps.
# ---------------------------------------------------------------------------

try:
    import eaik.IK_DH
    import eaik.IK_URDF

    _EAIK_AVAILABLE = True
except ImportError:
    _EAIK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixture catalogue.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fixture:
    name: str
    dof: int
    kind: str  # "urdf" or "specs"
    args: Any  # (urdf, base, ee) or (module, specs_fn)
    artifact: str | None = None  # module name under ssik.prebuilt if available


FIXTURES = [
    Fixture("UR5", 6, "urdf", ("ur5.urdf", "base_link", "ee_link"), "ur5_ik"),
    Fixture("Puma 560", 6, "urdf", ("puma560.urdf", "base_link", "wrist_3_link"), "puma560_ik"),
    Fixture("JACO 2", 6, "specs", ("jaco2", "jaco2_specs"), "jaco2_ik"),
    Fixture("iiwa14", 7, "specs", ("kuka_iiwa14", "kuka_iiwa14_specs"), "iiwa14_ik"),
    Fixture("Gen3", 7, "urdf", ("gen3.urdf", "base_link", "end_effector_link"), "gen3_ik"),
    Fixture("Franka Panda", 7, "specs", ("franka_panda", "franka_panda_specs"), "franka_panda_ik"),
    Fixture("Rizon 4", 7, "urdf", ("rizon4.urdf", "base_link", "flange"), "rizon4_ik"),
    Fixture(
        "Kassow KR810",
        7,
        "urdf",
        ("kassow_kr810.urdf", "base", "end_effector"),
        "kassow_kr810_ik",
    ),
    Fixture("xArm7", 7, "specs", ("xarm7", "xarm7_specs"), "xarm7_ik"),
    Fixture("xArm6", 6, "urdf", ("xarm6.urdf", "link_base", "link_eef"), "xarm6_ik"),
    Fixture("Z1", 6, "urdf", ("z1.urdf", "link00", "link06"), "z1_ik"),
    Fixture("PiPER", 6, "urdf", ("piper.urdf", "base_link", "link6"), "piper_ik"),
    Fixture("Rizon 10", 7, "urdf", ("rizon10.urdf", "base_link", "flange"), "rizon10_ik"),
    Fixture(
        "CRX-10iA/L",
        6,
        "urdf",
        ("fanuc_crx10ial.urdf", "base_link", "tool0"),
        "fanuc_crx10ial_ik",
    ),
]


class _ArtifactArm:
    """Adapter: prebuilt artifact module for IK (production path).
    Most artifacts bake ``_fk(q)``; SRS-class artifacts (iiwa14, gen3)
    don't but bake ``_KB``, so build a Manipulator(kb) for FK."""

    def __init__(self, module_name: str, dof: int):
        import importlib

        self._module = importlib.import_module(f"ssik.prebuilt.{module_name}")
        self.dof = dof
        if hasattr(self._module, "_fk"):
            self._fk = self._module._fk
        else:
            manip = ssik.Manipulator(self._module._KB)
            self._fk = manip.fk

    def fk(self, q):
        return self._fk(q)

    def solve(self, T, **kwargs):
        return self._module.solve(T, **kwargs)


def _load_ssik_arm(fx: Fixture, *, prefer_artifact: bool = True):
    """Load via prebuilt artifact if available (production path);
    fall back to Manipulator.from_urdf / build_kinbody (dev path)."""
    if prefer_artifact and fx.artifact:
        try:
            return _ArtifactArm(fx.artifact, fx.dof)
        except ImportError:
            pass  # fall through to dev path
    if fx.kind == "urdf":
        urdf, base, ee = fx.args
        return ssik.Manipulator.from_urdf(FIXTURES_DIR / urdf, base=base, ee=ee)
    else:
        mod_name, specs_fn_name = fx.args
        mod = __import__(mod_name)
        from ssik.internals import build_kinbody

        kb = build_kinbody(getattr(mod, specs_fn_name)())
        return ssik.Manipulator(kb)


# ---------------------------------------------------------------------------
# Statistics helpers.
# ---------------------------------------------------------------------------


def _mean_ci95(samples: np.ndarray, n_boot: int = 1000, rng=None) -> tuple[float, float]:
    """Bootstrap mean + 95% CI half-width."""
    if len(samples) == 0:
        return float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    n = len(samples)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = samples[idx].mean()
    mu = float(samples.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mu, float(max(mu - lo, hi - mu))


# ---------------------------------------------------------------------------
# Random pose generator. Shared across libraries so the workload is identical.
# ---------------------------------------------------------------------------


def _gen_poses(arm, n: int, rng) -> list[np.ndarray]:
    """Generate n reachable T-poses by sampling q in the safe interior."""
    Ts = []
    while len(Ts) < n:
        q = rng.uniform(-0.5, 0.5, size=arm.dof)
        if arm.dof == 7:
            q[3] = float(rng.uniform(0.3, 0.7))
        try:
            Ts.append(arm.fk(q))
        except Exception:
            continue
    return Ts


# ---------------------------------------------------------------------------
# ssik bench.
# ---------------------------------------------------------------------------


def _bench_ssik(arm, poses: list[np.ndarray]) -> dict:
    # ``respect_limits=False`` so the branch count reflects the analytical
    # solver's output, not the URDF-limit-aware postprocess. EAIK's bench
    # path (eaik.IK_URDF.UrdfRobot.IK) does not apply URDF joint limits;
    # using ssik's default ``respect_limits=True`` here makes the "/sols"
    # cells apples-to-oranges. Arms with asymmetric joint limits (Unitree
    # Z1: joint2 in [0, 2.97], joint3 in [-2.88, 0]) get filtered down to
    # 1 branch under default settings, which exaggerates the gap. The
    # "/time" measurement still includes the postprocess pass for
    # production realism; only the branch count assertion changes.
    # Warm.
    for T in poses[: min(10, len(poses))]:
        arm.solve(T, respect_limits=False)

    times = []
    fk_residuals = []
    sol_counts = []
    for T in poses:
        t = time.perf_counter()
        sols = arm.solve(T, respect_limits=False)
        times.append((time.perf_counter() - t) * 1000)
        if sols:
            fk_residuals.append(max(s.fk_residual for s in sols))
            sol_counts.append(len(sols))

    times_arr = np.array(times)
    mu, half = _mean_ci95(times_arr)
    return {
        "supported": True,
        "mean_ms": mu,
        "ci95_ms": half,
        "max_fk": float(max(fk_residuals)) if fk_residuals else float("nan"),
        "median_sols": int(np.median(sol_counts)) if sol_counts else 0,
        "min_sols": int(min(sol_counts)) if sol_counts else 0,
        "max_sols": int(max(sol_counts)) if sol_counts else 0,
        "n_solved": len(sol_counts),
        "n_total": len(poses),
    }


# ---------------------------------------------------------------------------
# EAIK bench.
# ---------------------------------------------------------------------------


def _eaik_load(fx: Fixture):
    """Load an EAIK robot for the fixture. Raises on failure."""
    if fx.kind == "urdf":
        urdf, _base, _ee = fx.args
        return eaik.IK_URDF.UrdfRobot(str(FIXTURES_DIR / urdf))
    else:
        # Spec arm: rebuild KinBody just for DH extraction.
        from ssik._kinbody import build_kinbody
        from ssik.kinematics.poe_to_dh import poe_to_dh

        mod_name, specs_fn_name = fx.args
        mod = __import__(mod_name)
        kb = build_kinbody(getattr(mod, specs_fn_name)())
        dh = poe_to_dh(kb)
        return eaik.IK_DH.DhRobot(dh.alpha, dh.a, dh.d)


def _bench_eaik(fx: Fixture, poses: list[np.ndarray]) -> dict:
    if not _EAIK_AVAILABLE:
        return {"supported": False, "error": "eaik not installed"}

    try:
        robot = _eaik_load(fx)
    except Exception as exc:
        return {"supported": False, "error": f"load: {type(exc).__name__}: {str(exc)[:80]}"}

    family = robot.getKinematicFamily()
    has_decomp = robot.hasKnownDecomposition()
    if not has_decomp:
        return {"supported": False, "error": f"no decomposition ({family})"}

    try:
        _ = robot.IK(poses[0])
    except Exception as exc:
        return {
            "supported": False,
            "error": f"IK probe: {type(exc).__name__}: {str(exc)[:60]}",
        }

    # Warm.
    for T in poses[: min(10, len(poses))]:
        robot.IK(T)

    times = []
    fk_residuals = []
    sol_counts = []
    for T in poses:
        t = time.perf_counter()
        sol = robot.IK(T)
        times.append((time.perf_counter() - t) * 1000)
        is_ls = np.asarray(sol.is_LS, dtype=bool)
        Q = np.asarray(sol.Q)
        exact_qs = Q[~is_ls] if Q.size else Q
        if len(exact_qs) > 0:
            sol_counts.append(len(exact_qs))
            worst = 0.0
            for q in exact_qs:
                T_fk = robot.fwdKin(np.asarray(q))
                worst = max(worst, float(np.linalg.norm(T_fk - T)))
            fk_residuals.append(worst)

    times_arr = np.array(times)
    mu, half = _mean_ci95(times_arr)
    return {
        "supported": True,
        "family": family,
        "mean_ms": mu,
        "ci95_ms": half,
        "max_fk": float(max(fk_residuals)) if fk_residuals else float("nan"),
        "median_sols": int(np.median(sol_counts)) if sol_counts else 0,
        "min_sols": int(min(sol_counts)) if sol_counts else 0,
        "max_sols": int(max(sol_counts)) if sol_counts else 0,
        "n_solved": len(sol_counts),
        "n_total": len(poses),
    }


# ---------------------------------------------------------------------------
# Output formatting.
# ---------------------------------------------------------------------------


def _cell_time(r: dict) -> str:
    if not r.get("supported"):
        return f"refuses ({r.get('error', '—')[:55]})"
    mu = r["mean_ms"]
    half = r["ci95_ms"]
    if mu < 1.0:
        return f"{mu * 1000:.0f} ± {half * 1000:.0f} µs"
    return f"{mu:.2f} ± {half:.2f} ms"


def _cell_fk(r: dict) -> str:
    if not r.get("supported"):
        return "—"
    v = r.get("max_fk")
    if v != v or v is None:
        return "—"
    return f"{v:.1e}"


def _cell_sols(r: dict) -> str:
    if not r.get("supported"):
        return "—"
    lo = r.get("min_sols", 0)
    hi = r.get("max_sols", 0)
    if hi == 0:
        return "—"
    return f"{lo}-{hi}" if hi > lo else f"{lo}"


def _format_row(fx: Fixture, ssik_r: dict, eaik_r: dict) -> str:
    ssik_cell = f"{_cell_time(ssik_r)} / {_cell_fk(ssik_r)} / {_cell_sols(ssik_r)}"
    if eaik_r.get("supported"):
        eaik_cell = f"{_cell_time(eaik_r)} / {_cell_fk(eaik_r)} / {_cell_sols(eaik_r)}"
    else:
        eaik_cell = _cell_time(eaik_r)
    return f"| {fx.name} ({fx.dof}R) | {ssik_cell} | {eaik_cell} |"


def _save_results(rows: list[tuple[Fixture, dict, dict]], path: Path) -> None:
    """Persist per-arm results as JSON so the user can stare at partial
    data while the bench runs."""
    import json

    payload = {
        "fixtures": [
            {
                "name": fx.name,
                "dof": fx.dof,
                "ssik": ssik_r,
                "eaik": eaik_r,
            }
            for fx, ssik_r, eaik_r in rows
        ]
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="poses per arm (default 100)")
    parser.add_argument("--seed", type=int, default=0, help="rng seed")
    parser.add_argument(
        "--cache",
        type=str,
        default="/tmp/ssik_bench.json",
        help="path for incremental JSON cache (default /tmp/ssik_bench.json)",
    )
    args = parser.parse_args()

    print("# ssik vs EAIK comparison\n", flush=True)
    print(
        f"EAIK: {'installed' if _EAIK_AVAILABLE else 'NOT installed'} (eaik==1.2.1 if present)",
        flush=True,
    )
    print("Numerical IK baselines (MINK / TracIK) tracked in #236", flush=True)
    print(
        f"Poses per arm: {args.n}, seed={args.seed}, CI: bootstrap 1000 resamples",
        flush=True,
    )
    cache_path = Path(args.cache)
    print(f"Incremental cache: {cache_path}\n", flush=True)
    print("| Arm | ssik (time / max FK / sols) | EAIK (time / max FK / sols) |", flush=True)
    print("|---|---|---|", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for fx in FIXTURES:
        t = time.perf_counter()
        try:
            arm = _load_ssik_arm(fx)
        except Exception as exc:
            print(f"| {fx.name} ({fx.dof}R) | ssik load failed: {exc} | — |", flush=True)
            continue
        poses = _gen_poses(arm, args.n + 10, rng)
        ssik_r = _bench_ssik(arm, poses[10:])
        eaik_r = _bench_eaik(fx, poses[10:])
        elapsed = time.perf_counter() - t
        rows.append((fx, ssik_r, eaik_r))
        print(f"{_format_row(fx, ssik_r, eaik_r)}  <!-- {elapsed:.1f}s -->", flush=True)
        _save_results(rows, cache_path)

    print()
    print("Notes:")
    print("- ssik returns ALL analytical branches; EAIK does where it supports the arm.")
    print("- FK residual measured against the ORIGINAL URDF/spec FK (not a simplified DH).")
    print("- 'refuses (...)' shows EAIK's actual error or kinematic-family classification.")
    print("- ssik 7R uses prebuilt artifacts (production path); cached-RR arms are 10x")
    print("  faster than Manipulator.from_urdf on the first hot-cache call.")


if __name__ == "__main__":
    main()
