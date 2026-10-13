"""End-to-end check of the C++ headers shipped in the ssik wheel (#641, #707).

Run it with the Python of an environment where the ssik WHEEL is installed (not
an editable checkout); it refuses to run against a source tree. It configures
and builds this directory as a standalone CMake project, with
``find_package(ssik_cpp <installed major> CONFIG REQUIRED)`` resolved through
``ssik.get_cmake_dir()`` and Eigen from ``--eigen-prefix``, then checks two
consumers of the installed headers.

``three_parallel_solve`` (#641): UR5's joint data from ``ssik.cpp.joint_data``
and 12 poses go in ``<out>/ur5_three_parallel.input.txt``; the program calls
``three_parallel_artifact_solve`` (``<out>/ur5_three_parallel.solutions.txt``).
Every solution must close FK within 1e-9, and per pose the C++ and Python
(``solve(T, native=False)``) solution sets must be equal modulo 2*pi (same
count, one-to-one within 1e-6 rad).

``solve_any`` (#707): one program, ``ssik::Solver``, for every native family.
Per arm, ``ssik.cpp.solver_data`` goes in ``<out>/<arm>.data.txt`` (family,
schema, then per array its name, dtype, rank, shape and ``repr(float)`` values,
which round-trip exactly) and 8 poses, each solved unseeded with the defaults
and seeded with ``max_solutions=1``, in ``<out>/<arm>.cases.txt``; the program
writes ``<out>/<arm>.solutions.txt``. The arms are one prebuilt arm per native
family (both jointlock kernels) plus runtime-built copies of the Panda and the
JACO 2. Every case must be non-empty; every solution must close FK within 1e-5
both by the arm's own ``fk`` and by ``Solver::fk``; and the C++ set must equal
the in-process native set (same count, one-to-one within 1e-6 rad), where the
reference is ``arm.solve(T, native=True)`` for a prebuilt arm and the native
extension's ``Solver`` on the same ``solver_data`` for a runtime arm. Every
family in ``ssik.cpp.NATIVE_FAMILIES`` must be exercised, and ``solver_data``
must refuse a runtime jointlock arm and a ``spherical_two_intersecting`` arm
with ``UnsupportedFamilyError`` and a non-arm with ``TypeError``.

Everything is summarised in ``<out>/report.json``. The time ``solver_data``
took per arm is printed, not reported, so the report stays deterministic.

Usage, from a clean venv::

    uv build --wheel
    uv venv /tmp/whl && uv pip install --python /tmp/whl/bin/python dist/*.whl
    /tmp/whl/bin/python cpp/examples/wheel_consumer/check.py --out /tmp/wc \\
        --eigen-prefix "$(python scripts/fetch_eigen.py --cmake-prefix)" \\
        --eigen-version "$(python scripts/fetch_eigen.py --version)"

The runtime JACO 2 needs its Raghavan-Roth tensor: about 28 s the first time on
a machine, then milliseconds from the per-arm preparation cache
(``SSIK_DERIVATION_CACHE``). The inputs, the C++ solutions and the report are
deterministic for a given wheel and platform, so ``<out>`` is the inspectable
artifact of the run.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import ssik
from ssik.cpp import NATIVE_FAMILIES, UnsupportedFamilyError, joint_data, solver_data

ARM = "ur5"
N_POSES = 12
SEED = 641
FK_TOL = 1e-9
MATCH_TOL = 1e-6

# solve_any: one prebuilt arm per native family, then runtime-built copies.
SOLVER_ARMS = [
    "ur5_ik",  # ikgeo.three_parallel
    "irb120_ik",  # ikgeo.spherical_two_parallel
    "jaco2_ik",  # ikgeo.general_6r
    "iiwa14_ik",  # seven_r.srs
    "gen3_ik",  # seven_r.srs_polished
    "franka_panda_ik",  # seven_r.spherical_shoulder
    "xarm7_ik",  # seven_r.spherical_shoulder_polished
    "rizon4_ik",  # jointlock.seven_r, Raghavan-Roth inner kernel
    "kassow_kr810_ik",  # jointlock.seven_r, Husty-Pfurner inner kernel
]
RUNTIME_ARMS = ["franka_panda_ik", "jaco2_ik"]
SOLVER_POSES = 8
SOLVER_SEED = 707
SEED_NOISE = 0.05  # rad, the seeded case's offset from the configuration that made the pose
SOLVER_FK_TOL = 1e-5  # subproblem_numerical, the default policy's FK band


def _wrapped(a: np.ndarray, b: np.ndarray) -> float:
    d = np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi
    return float(np.max(np.abs(d)))


def _write_input(path: Path, arm: ssik.Manipulator, poses: list[np.ndarray]) -> None:
    d = joint_data(arm)
    assert d.solver == "ikgeo.three_parallel", d.solver
    tok: list[str] = [str(d.dof)]
    for i in range(d.dof):
        tok += [repr(float(v)) for v in d.axis[i]]
        tok += [repr(float(v)) for v in d.t_left[i].ravel()]
        tok += [repr(float(v)) for v in d.t_right[i].ravel()]
        tok.append(str(int(d.joint_type[i])))
    tok += [repr(float(v)) for v in d.lo]
    tok += [repr(float(v)) for v in d.hi]
    tok += [str(int(v)) for v in d.present]
    tok.append(str(len(poses)))
    for T in poses:
        tok += [repr(float(v)) for v in T.ravel()]
    path.write_text("\n".join(tok) + "\n")


def _read_solutions(path: Path, n_cases: int, n_cols: int) -> list[np.ndarray]:
    lines = path.read_text().splitlines()
    out, k = [], 0
    for _ in range(n_cases):
        n = int(lines[k])
        rows = [np.array(lines[k + 1 + j].split(), dtype=np.float64) for j in range(n)]
        out.append(np.array(rows).reshape(n, n_cols))
        k += 1 + n
    if k != len(lines):
        raise ValueError(f"{path}: {len(lines) - k} trailing lines")
    return out


def _match(cpp: np.ndarray, ref: np.ndarray) -> float | None:
    """One-to-one matching mod 2*pi; the worst matched distance, or None."""
    if len(cpp) != len(ref):
        return None
    free = list(range(len(ref)))
    worst = 0.0
    for q in cpp:
        dists = [_wrapped(q, ref[j]) for j in free]
        if not dists:
            return None
        best = int(np.argmin(dists))
        if dists[best] > MATCH_TOL:
            return None
        worst = max(worst, dists[best])
        free.pop(best)
    return worst


def _sample_q(arm: ssik.Manipulator, rng: np.random.Generator) -> np.ndarray:
    lim = [(-np.pi, np.pi) if b is None else b for b in arm.joint_limits]
    return rng.uniform([b[0] for b in lim], [b[1] for b in lim])


def _check_three_parallel(out: Path, exe: Path) -> tuple[bool, dict[str, Any]]:
    arm = ssik.Manipulator.from_prebuilt(ARM)
    rng = np.random.default_rng(SEED)
    poses = [arm.fk(_sample_q(arm, rng)) for _ in range(N_POSES)]
    inp, sol = out / "ur5_three_parallel.input.txt", out / "ur5_three_parallel.solutions.txt"
    _write_input(inp, arm, poses)
    subprocess.run([str(exe), str(inp), str(sol)], check=True)

    cpp_sets = _read_solutions(sol, N_POSES, 6)
    cases, ok = [], True
    for i, (T, cpp) in enumerate(zip(poses, cpp_sets, strict=True)):
        py = np.array([s.q for s in arm.solve(T, native=False)]).reshape(-1, 6)
        fk = max((float(np.linalg.norm(arm.fk(q) - T)) for q in cpp), default=0.0)
        worst = _match(cpp, py)
        passed = len(cpp) > 0 and fk <= FK_TOL and worst is not None
        ok &= passed
        cases.append(
            {"pose": i, "n_cpp": len(cpp), "n_python": len(py), "max_fk_residual": fk,
             "max_match_rad": worst, "pass": passed}
        )  # fmt: skip
        match = "none" if worst is None else f"{worst:.1e}"
        print(
            f"three_parallel pose {i:2d}: C++ {len(cpp):3d}  Python {len(py):3d}  fk {fk:.1e}  "
            f"match {match}  {'ok' if passed else 'FAIL'}"
        )
    return ok, {"arm": ARM, "fk_tol": FK_TOL, "pass": ok, "cases": cases}


def _write_solver_data(path: Path, d: ssik.cpp.SolverData) -> None:
    lines = [d.family, str(d.schema), str(len(d.arrays))]
    for name in sorted(d.arrays):
        a = d.arrays[name]
        if a.dtype == np.float64:
            dtype, vals = "f64", [repr(float(v)) for v in a.ravel()]
        elif a.dtype == np.int32:
            dtype, vals = "i32", [str(int(v)) for v in a.ravel()]
        else:
            raise TypeError(f"solver data array {name!r} is {a.dtype}, not float64 or int32")
        lines.append(" ".join([name, dtype, str(a.ndim), *map(str, a.shape)]))
        lines.append(" ".join(vals))
    path.write_text("\n".join(lines) + "\n")


def _write_cases(path: Path, cases: list[tuple[np.ndarray, np.ndarray | None, int]]) -> None:
    lines = [str(len(cases))]
    for T, seed, max_solutions in cases:
        tok = [repr(float(v)) for v in T.ravel()]
        tok.append("0" if seed is None else "1")
        if seed is not None:
            tok += [repr(float(v)) for v in seed]
        tok.append(str(max_solutions))
        lines.append(" ".join(tok))
    path.write_text("\n".join(lines) + "\n")


def _solver_arms() -> list[tuple[str, ssik.Manipulator, bool]]:
    arms = [(name, ssik.Manipulator.from_prebuilt(name), False) for name in SOLVER_ARMS]
    for name in RUNTIME_ARMS:
        kb = copy.deepcopy(ssik.Manipulator.from_prebuilt(name).kinbody)
        arms.append((f"runtime_{name}", ssik.Manipulator(kb), True))
    return arms


def _check_solver(out: Path, exe: Path) -> tuple[bool, dict[str, Any]]:
    from ssik import _native

    ext = _native._load_ext()
    if ext is None:
        sys.exit("the installed wheel has no native extension (ssik._ssik_native)")
    rng = np.random.default_rng(SOLVER_SEED)
    rows, families, ok = [], set(), True
    for name, arm, runtime in _solver_arms():
        t0 = time.perf_counter()
        d = solver_data(arm)
        data_s = time.perf_counter() - t0
        families.add(d.family)
        cases: list[tuple[np.ndarray, np.ndarray | None, int]] = []
        for _ in range(SOLVER_POSES):
            q = _sample_q(arm, rng)
            T = arm.fk(q)
            cases.append((T, None, -1))
            cases.append((T, q + rng.normal(scale=SEED_NOISE, size=arm.dof), 1))
        data_p, cases_p, sol_p = (out / f"{name}.{k}.txt" for k in ("data", "cases", "solutions"))
        _write_solver_data(data_p, d)
        _write_cases(cases_p, cases)
        subprocess.run([str(exe), str(data_p), str(cases_p), str(sol_p)], check=True)
        cpp_sets = _read_solutions(sol_p, len(cases), arm.dof + 1)

        ref_solver = ext.Solver(d) if runtime else None
        arm_ok, max_fk, max_cpp_fk, max_match, n_sols, case_rows = True, 0.0, 0.0, 0.0, 0, []
        for i, ((T, seed, cap), rows_cpp) in enumerate(zip(cases, cpp_sets, strict=True)):
            cpp, cpp_fk = rows_cpp[:, : arm.dof], rows_cpp[:, arm.dof]
            kw: dict[str, Any] = {} if seed is None else {"q_seed": seed, "max_solutions": cap}
            if runtime:
                ref_sols = _native._solve(ref_solver, T, **kw)
            else:
                ref_sols = arm.solve(T, native=True, **kw)
            ref = np.array([s.q for s in ref_sols]).reshape(-1, arm.dof)
            fk = max((float(np.max(np.abs(arm.fk(q) - T))) for q in cpp), default=0.0)
            cfk = float(np.max(cpp_fk, initial=0.0))
            worst = _match(cpp, ref)
            passed = (
                len(cpp) > 0 and fk <= SOLVER_FK_TOL and cfk <= SOLVER_FK_TOL and worst is not None
            )
            arm_ok &= passed
            n_sols += len(cpp)
            max_fk, max_cpp_fk = max(max_fk, fk), max(max_cpp_fk, cfk)
            max_match = max(max_match, np.inf if worst is None else worst)
            case_rows.append(
                {"case": i, "pose": i // 2, "seeded": seed is not None, "n_cpp": len(cpp),
                 "n_reference": len(ref), "max_fk_residual": fk, "max_cpp_fk_residual": cfk,
                 "max_match_rad": worst, "pass": passed}
            )  # fmt: skip
        ok &= arm_ok
        rows.append(
            {"arm": name, "runtime": runtime, "family": d.family, "schema": d.schema,
             "dof": d.dof, "n_arrays": len(d.arrays),
             "n_cases": len(cases), "n_solutions": n_sols, "max_fk_residual": max_fk,
             "max_cpp_fk_residual": max_cpp_fk,
             "max_match_rad": None if np.isinf(max_match) else max_match,
             "pass": arm_ok, "cases": case_rows}
        )  # fmt: skip
        print(
            f"Solver {name:26s} {d.family:36s} {n_sols:4d} solutions  fk {max_fk:.1e}  "
            f"C++ fk {max_cpp_fk:.1e}  match {max_match:.1e}  data {data_s:.2f}s  "
            f"{'ok' if arm_ok else 'FAIL'}"
        )

    missing = sorted(NATIVE_FAMILIES - families)
    if missing:
        print(f"Solver: families not exercised: {missing}  FAIL")
        ok = False
    refusals = _check_refusals()
    ok &= all(r["pass"] for r in refusals)
    return ok, {
        "seed": SOLVER_SEED, "poses_per_arm": SOLVER_POSES, "seed_noise_rad": SEED_NOISE,
        "fk_tol": SOLVER_FK_TOL, "families_exercised": sorted(families),
        "families_missing": missing, "refusals": refusals, "arms": rows,
    }  # fmt: skip


def _check_refusals() -> list[dict[str, Any]]:
    """solver_data refuses what it cannot describe, with the documented error."""
    z, y, x = [0, 0, 1], [0, 1, 0], [1, 0, 0]
    spherical_two_intersecting = ssik.Manipulator.from_axes(
        [z, y, x, z, y, z],
        [[0, 0, 0.3], [0, 0, 0], [0, 0, 0.4], [0.4, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0.1]],
    )
    jointlock = ssik.Manipulator(copy.deepcopy(ssik.Manipulator.from_prebuilt("rizon4").kinbody))
    probes: list[tuple[str, Any, type[Exception]]] = [
        ("runtime spherical_two_intersecting", spherical_two_intersecting, UnsupportedFamilyError),
        ("runtime jointlock.seven_r", jointlock, UnsupportedFamilyError),
        ("not a Manipulator", "ur5", TypeError),
    ]
    out = []
    for what, arg, expected in probes:
        try:
            solver_data(arg)
            raised = "nothing"
        except Exception as exc:  # the probe reports whatever was raised
            raised = type(exc).__name__
        passed = raised == expected.__name__
        out.append({"probe": what, "expected": expected.__name__, "raised": raised, "pass": passed})
        print(f"solver_data({what}): {raised}  {'ok' if passed else 'FAIL'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="artifact + build directory")
    ap.add_argument("--eigen-prefix", required=True, help="CMake prefix of Eigen")
    ap.add_argument("--eigen-version", default="", help="require exactly this Eigen")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    src = here.parents[2] / "src"
    pkg = Path(ssik.__file__).resolve().parent
    if pkg.is_relative_to(src):
        sys.exit(f"ssik is imported from the source tree ({pkg}); install the wheel")
    include, cmake_dir = ssik.get_include(), ssik.get_cmake_dir()
    for p in (include, cmake_dir):
        if not Path(p).is_relative_to(pkg):
            sys.exit(f"{p} is not inside the installed ssik package {pkg}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    major = ssik.__version__.split(".")[0]
    build = out / "build"
    subprocess.run(
        [
            "cmake", "-S", str(here), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={cmake_dir};{args.eigen_prefix}",
            f"-DSSIK_CPP_VERSION={major}", f"-DSSIK_EIGEN_VERSION={args.eigen_version}",
        ],
        check=True,
    )  # fmt: skip
    subprocess.run(["cmake", "--build", str(build)], check=True)

    tp_ok, tp_report = _check_three_parallel(out, build / "three_parallel_solve")
    solver_ok, solver_report = _check_solver(out, build / "solve_any")
    ok = tp_ok and solver_ok
    report = {
        "ssik_version": ssik.__version__, "include": include, "cmake_dir": cmake_dir,
        "eigen_version": args.eigen_version or "any", "match_tol": MATCH_TOL, "pass": ok,
        "three_parallel_artifact_solve": tp_report, "solver": solver_report,
    }  # fmt: skip
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wheel consumer: {'PASS' if ok else 'FAIL'} ({out / 'report.json'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
