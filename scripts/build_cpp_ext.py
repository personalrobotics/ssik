#!/usr/bin/env python
"""Build the test-only native-solver Python extension (#499).

Compiles ``cpp/bindings/three_parallel_py.cpp`` (which includes the header-only
native solver) into ``ssik_cpp_ext`` so the existing Python test suite can drive
the C++ backend. This is NOT part of the shipped wheel -- it is a dev/CI test
tool. The Python test suite skips the C++ backend when the extension is absent.

Usage: ``python scripts/build_cpp_ext.py [--out-dir <dir>]`` (default: cpp/build).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11

_REPO = Path(__file__).resolve().parent.parent


def _eigen_include() -> str:
    """Locate the Eigen headers (Homebrew / apt / find_package layout)."""
    candidates = [
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]
    for c in candidates:
        if (Path(c) / "Eigen" / "Dense").exists():
            return c
    raise SystemExit(
        "Eigen not found; install eigen (brew install eigen / apt install libeigen3-dev)"
    )


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = out_dir / f"ssik_cpp_ext{ext_suffix}"
    src = _REPO / "cpp" / "bindings" / "three_parallel_py.cpp"

    cmd = [
        (sys.platform == "darwin" and "clang++") or "c++",
        "-O2",
        "-std=c++20",
        "-shared",
        "-fPIC",
        f"-I{_REPO / 'cpp' / 'include'}",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_path('include')}",
        f"-I{_eigen_include()}",
        str(src),
        "-o",
        str(out),
    ]
    if sys.platform == "darwin":
        cmd += ["-undefined", "dynamic_lookup"]

    print("[build_cpp_ext]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[build_cpp_ext] built {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=_REPO / "cpp" / "build")
    args = ap.parse_args()
    build(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
