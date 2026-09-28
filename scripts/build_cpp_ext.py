#!/usr/bin/env python
"""Build the test-only native-solver Python extension (#499).

Compiles ``cpp/bindings/three_parallel_py.cpp`` (which includes the header-only
native solver) into ``_ssik_native`` so the existing Python test suite can drive
the C++ backend. This is NOT part of the shipped wheel -- it is a dev/CI test
tool. The Python test suite skips the C++ backend when the extension is absent.

Usage: ``python scripts/build_cpp_ext.py [--out-dir <dir>]`` (default: cpp/build).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import fetch_eigen
import pybind11

_REPO = Path(__file__).resolve().parent.parent


def _eigen_include() -> str:
    """The pinned Eigen (scripts/fetch_eigen.py, #606), or SSIK_EIGEN_INCLUDE_DIR
    when a developer deliberately overrides it. Fails rather than falling back
    to whatever Eigen the system has."""
    override = os.environ.get(fetch_eigen.OVERRIDE_ENV)
    if override:
        if not (Path(override) / "Eigen" / "Dense").is_file():
            raise SystemExit(f"{fetch_eigen.OVERRIDE_ENV}={override} has no Eigen/Dense")
        print(
            f"[build_cpp_ext] Eigen {fetch_eigen.header_version(override)} from "
            f"{fetch_eigen.OVERRIDE_ENV}={override} (override; the pin is "
            f"{fetch_eigen.EIGEN_VERSION})"
        )
        return override
    try:
        inc = fetch_eigen.ensure_eigen()
    except fetch_eigen.EigenUnavailable as exc:
        raise SystemExit(
            f"Pinned Eigen {fetch_eigen.EIGEN_VERSION} unavailable: {exc}\n"
            f"Set {fetch_eigen.OVERRIDE_ENV} to build against a local Eigen instead."
        ) from exc
    print(f"[build_cpp_ext] Eigen {fetch_eigen.EIGEN_VERSION} (pinned) from {inc}")
    return str(inc)


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = out_dir / f"_ssik_native{ext_suffix}"
    src = _REPO / "cpp" / "bindings" / "three_parallel_py.cpp"

    cmd = [
        (sys.platform == "darwin" and "clang++") or "c++",
        "-O2",
        # Release build: disable assert()/eigen_assert. The solvers deliberately
        # feed Eigen degenerate matrices (rescue jitters, near-singular poses) and
        # rely on FK-certification to reject the resulting garbage, so an active
        # eigen_assert would abort() the process on exactly the poses the algorithm
        # is designed to survive. Standard for a shipped extension.
        "-DNDEBUG",
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
    else:
        # parallel.hpp (rescue / jointlock sweep, #546) uses std::thread, which
        # needs the pthread library linked on Linux (macOS has it in libSystem).
        cmd += ["-pthread"]

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
