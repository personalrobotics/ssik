"""End-to-end check of the C++ headers shipped in the ssik wheel (#641).

Run it with the Python of an environment where the ssik WHEEL is installed (not
an editable checkout); it refuses to run against a source tree. It:

1. writes UR5's joint data from ``ssik.cpp.joint_data`` and a fixed set of
   target poses to ``<out>/input.txt``;
2. configures and builds this directory as a standalone CMake project, with
   ``find_package(ssik_cpp <installed major> CONFIG REQUIRED)`` resolved through
   ``ssik.get_cmake_dir()`` and Eigen from ``--eigen-prefix``;
3. runs ``three_parallel_artifact_solve`` on every pose (``<out>/solutions.txt``);
4. checks every C++ solution closes FK within 1e-9 and that, per pose, the C++
   and Python (``solve(T, native=False)``) solution sets are equal modulo 2*pi
   (same count, one-to-one within 1e-6 rad), and writes ``<out>/report.json``.

Usage, from a clean venv::

    uv build --wheel
    uv venv /tmp/whl && uv pip install --python /tmp/whl/bin/python dist/*.whl
    /tmp/whl/bin/python cpp/examples/wheel_consumer/check.py --out /tmp/wc \\
        --eigen-prefix "$(python scripts/fetch_eigen.py --cmake-prefix)" \\
        --eigen-version "$(python scripts/fetch_eigen.py --version)"

The input, the C++ solutions and the report are deterministic for a given
wheel and platform, so ``<out>`` is the inspectable artifact of the run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import ssik
from ssik.cpp import joint_data

ARM = "ur5"
N_POSES = 12
SEED = 641
FK_TOL = 1e-9
MATCH_TOL = 1e-6


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


def _read_solutions(path: Path, n_poses: int) -> list[np.ndarray]:
    lines = path.read_text().splitlines()
    out, k = [], 0
    for _ in range(n_poses):
        n = int(lines[k])
        rows = [np.array(lines[k + 1 + j].split(), dtype=np.float64) for j in range(n)]
        out.append(np.array(rows).reshape(n, 6))
        k += 1 + n
    if k != len(lines):
        raise ValueError(f"{path}: {len(lines) - k} trailing lines")
    return out


def _match(cpp: np.ndarray, py: np.ndarray) -> float | None:
    """One-to-one matching mod 2*pi; the worst matched distance, or None."""
    if len(cpp) != len(py):
        return None
    free = list(range(len(py)))
    worst = 0.0
    for q in cpp:
        dists = [_wrapped(q, py[j]) for j in free]
        if not dists:
            return None
        best = int(np.argmin(dists))
        if dists[best] > MATCH_TOL:
            return None
        worst = max(worst, dists[best])
        free.pop(best)
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="artifact + build directory")
    ap.add_argument("--eigen-prefix", required=True, help="CMake prefix of Eigen")
    ap.add_argument("--eigen-version", default="", help="require exactly this Eigen")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    pkg = Path(ssik.__file__).resolve().parent
    if pkg.is_relative_to(repo):
        sys.exit(f"ssik is imported from the source tree ({pkg}); install the wheel")
    include, cmake_dir = ssik.get_include(), ssik.get_cmake_dir()
    for p in (include, cmake_dir):
        if not Path(p).is_relative_to(pkg):
            sys.exit(f"{p} is not inside the installed ssik package {pkg}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    arm = ssik.Manipulator.from_prebuilt(ARM)
    rng = np.random.default_rng(SEED)
    lo = np.array([lim[0] for lim in arm.joint_limits])
    hi = np.array([lim[1] for lim in arm.joint_limits])
    poses = [arm.fk(rng.uniform(lo, hi)) for _ in range(N_POSES)]
    _write_input(out / "input.txt", arm, poses)

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
    exe = build / "three_parallel_solve"
    subprocess.run([str(exe), str(out / "input.txt"), str(out / "solutions.txt")], check=True)

    cpp_sets = _read_solutions(out / "solutions.txt", N_POSES)
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
            f"pose {i:2d}: C++ {len(cpp):3d}  Python {len(py):3d}  fk {fk:.1e}  "
            f"match {match}  {'ok' if passed else 'FAIL'}"
        )
    report = {
        "arm": ARM, "ssik_version": ssik.__version__, "include": include,
        "cmake_dir": cmake_dir, "eigen_version": args.eigen_version or "any",
        "fk_tol": FK_TOL, "match_tol": MATCH_TOL, "pass": ok, "cases": cases,
    }  # fmt: skip
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wheel consumer: {'PASS' if ok else 'FAIL'} ({out / 'report.json'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
