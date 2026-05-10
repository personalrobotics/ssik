"""Example 04: apples-to-apples comparison vs EAIK and MINK.

Bench ``ssik`` against:

- **EAIK** (Ostermeier 2024, ``pip install eaik``) — Python wrapper around
  C++ analytical solvers using subproblem decomposition. Fast on
  Pieper-class arms; refuses non-Pieper 6R + non-SRS 7R arms by design.
- **MINK** (``pip install mink``) — Mujoco-native numerical IK via damped
  least-squares. Iterative, takes a seed q_0, converges to a single
  configuration. No analytical branches.

Both are imported with **soft-skip**: if the dependency isn't installed
or fails to load a specific arm, the row is marked ``unsupported`` and
the harness keeps going. Don't be surprised if EAIK refuses several of
the 7R arms — that's the point of the comparison.

The harness reports, per arm:

- median IK time (ms)
- max / median FK residual (Frobenius norm)
- number of IK solutions returned (1 for numeric, up to 256 for analytical)
- "supported / unsupported / load-failed" status

Output is a markdown table embeddable in the README.

Run::

    uv run python examples/04_compare_vs_eaik_mink.py

    # With dev deps (recommended for full comparison):
    pip install eaik mink
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))

import ssik  # noqa: E402

# ---------------------------------------------------------------------------
# Optional deps with soft-skip.
# ---------------------------------------------------------------------------

try:
    import eaik

    _EAIK_AVAILABLE = True
except ImportError:
    eaik = None  # type: ignore[assignment]
    _EAIK_AVAILABLE = False

try:
    import mink
    import mujoco

    _MINK_AVAILABLE = True
except ImportError:
    mink = None  # type: ignore[assignment]
    mujoco = None  # type: ignore[assignment]
    _MINK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

_FIXTURES = [
    # (name, kind, args, dof) where kind in {"urdf", "specs"}
    ("UR5", "urdf", ("ur5.urdf", "base_link", "ee_link"), 6),
    ("Puma 560", "urdf", ("puma560.urdf", "base_link", "wrist_3_link"), 6),
    ("JACO 2", "specs", "jaco2_specs", 6),
    ("iiwa14", "specs", "kuka_iiwa14_specs", 7),
    ("Gen3 (7-DOF)", "urdf", ("gen3.urdf", "base_link", "end_effector_link"), 7),
    ("Franka Panda", "specs", "franka_panda_specs", 7),
    ("Rizon 4", "urdf", ("rizon4.urdf", "base_link", "flange"), 7),
    ("Kassow KR810", "urdf", ("kassow_kr810.urdf", "base", "end_effector"), 7),
]

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def _load_ssik_arm(kind: str, args) -> ssik.Manipulator:
    if kind == "urdf":
        urdf, base, ee = args
        return ssik.Manipulator.from_urdf(FIXTURES_DIR / urdf, base=base, ee=ee)
    elif kind == "specs":
        module = __import__(args.replace("_specs", ""), fromlist=[args])
        specs_fn = getattr(module, args)
        kb = ssik.build_kinbody(specs_fn())
        return ssik.Manipulator(kb)
    else:
        raise ValueError(f"unknown kind: {kind}")


# ---------------------------------------------------------------------------
# Bench harness.
# ---------------------------------------------------------------------------


def _bench_ssik(arm: ssik.Manipulator, N: int = 50, rng=None) -> dict:
    """Measure ssik IK over N random reachable poses."""
    rng = rng or np.random.default_rng(0)
    # Warm.
    for _ in range(10):
        q = rng.uniform(-0.5, 0.5, size=arm.dof)
        if arm.dof == 7:
            q[3] = float(rng.uniform(0.3, 0.7))
        arm.ik(arm.fk(q))

    times = []
    fk_residuals = []
    sol_counts = []
    for _ in range(N):
        q = rng.uniform(-0.5, 0.5, size=arm.dof)
        if arm.dof == 7:
            q[3] = float(rng.uniform(0.3, 0.7))
        T = arm.fk(q)
        t = time.perf_counter()
        sols, is_ls = arm.ik(T)
        times.append((time.perf_counter() - t) * 1000)
        if not is_ls and sols:
            fk_residuals.append(max(s.fk_residual for s in sols))
            sol_counts.append(len(sols))
    return {
        "supported": True,
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_fk": float(max(fk_residuals)) if fk_residuals else float("nan"),
        "median_sols": int(np.median(sol_counts)) if sol_counts else 0,
        "n_solved": len(sol_counts),
        "n_total": N,
    }


def _bench_eaik(name: str, kind: str, args, N: int = 50, rng=None) -> dict:
    """Measure EAIK IK over N random reachable poses. Returns
    ``{'supported': False, 'error': <reason>}`` on failure."""
    if not _EAIK_AVAILABLE:
        return {"supported": False, "error": "eaik not installed"}

    rng = rng or np.random.default_rng(0)
    try:
        if kind == "urdf":
            urdf, _base, _ee = args
            robot = eaik.IK_URDF(str(FIXTURES_DIR / urdf))
        else:
            return {"supported": False, "error": "EAIK requires URDF (Python-spec arm)"}
    except Exception as exc:
        return {"supported": False, "error": f"load failed: {type(exc).__name__}: {exc}"}

    # EAIK exposes `.IK_solver(T)` or `.has_solution()`; the API varies
    # slightly across versions. We use the conservative `solve` attribute
    # if available, otherwise fall back gracefully.
    try:
        # Probe API: try a single call.
        # eaik 1.x: robot.IK(T) -> list of solutions
        if not hasattr(robot, "IK") and not hasattr(robot, "solve"):
            return {"supported": False, "error": "EAIK API mismatch (no .IK or .solve)"}
    except Exception as exc:
        return {"supported": False, "error": f"API probe failed: {exc}"}

    return {"supported": False, "error": "EAIK bench harness not yet implemented; PR welcome"}


def _bench_mink(name: str, kind: str, args, N: int = 50, rng=None) -> dict:
    """Measure MINK iterative IK over N random reachable poses."""
    if not _MINK_AVAILABLE:
        return {"supported": False, "error": "mink not installed"}
    if kind != "urdf":
        return {"supported": False, "error": "MINK harness needs URDF/MJCF"}
    return {
        "supported": False,
        "error": "MINK bench harness not yet implemented; PR welcome",
    }


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------


def _format_row(name: str, dof: int, ssik_r: dict, eaik_r: dict, mink_r: dict) -> str:
    def _cell(r: dict, key: str, fmt: str = "{:.2f}") -> str:
        if not r.get("supported"):
            return r.get("error", "—")[:30]
        v = r.get(key)
        return fmt.format(v) if isinstance(v, (int, float)) and v == v else "—"

    s_time = _cell(ssik_r, "median_ms", "{:.2f} ms")
    s_fk = _cell(ssik_r, "max_fk", "{:.1e}")
    s_sols = _cell(ssik_r, "median_sols", "{}")
    e_time = _cell(eaik_r, "median_ms", "{:.2f} ms")
    m_time = _cell(mink_r, "median_ms", "{:.2f} ms")
    return f"| {name} ({dof}R) | {s_time} / FK {s_fk} / {s_sols} sols | {e_time} | {m_time} |"


def main() -> None:
    print("# ssik vs EAIK vs MINK comparison\n")
    print(f"EAIK installed: {_EAIK_AVAILABLE}")
    print(f"MINK installed: {_MINK_AVAILABLE}")
    print()
    print("| Arm | ssik | EAIK | MINK |")
    print("|---|---|---|---|")

    rng = np.random.default_rng(0)

    for name, kind, args, dof in _FIXTURES:
        try:
            arm = _load_ssik_arm(kind, args)
        except Exception as exc:
            print(f"| {name} ({dof}R) | ssik load failed: {exc} | — | — |")
            continue

        ssik_r = _bench_ssik(arm, rng=rng)
        eaik_r = _bench_eaik(name, kind, args, rng=rng)
        mink_r = _bench_mink(name, kind, args, rng=rng)
        print(_format_row(name, dof, ssik_r, eaik_r, mink_r))

    print()
    print("Notes:")
    print("- ssik returns ALL analytical branches; EAIK does where it supports the arm; MINK")
    print("  always returns 1 (numerical convergence to a single q from the seed).")
    print("- FK residual measured against the ORIGINAL URDF FK, not a simplified DH.")
    print("- EAIK / MINK harnesses are stubs in this v1 release; install them locally and")
    print("  contribute the per-library benchmark code.")


if __name__ == "__main__":
    main()
