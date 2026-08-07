#!/usr/bin/env bash
# External-consumer smoke: install ssik_cpp to a temp prefix, then configure +
# build + run a STANDALONE downstream project against it via find_package. Proves
# the packaged self-contained artifacts are usable by a real C++ consumer with no
# ssik source tree and no Python. Run from anywhere; paths are resolved here.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cpp_root="$(cd "$here/../.." && pwd)"     # .../cpp
prefix="$(mktemp -d)"
build="$(mktemp -d)"
cons="$(mktemp -d)"
trap 'rm -rf "$prefix" "$build" "$cons"' EXIT

# The per-arm artifacts (cpp/gen/*.hpp) are generated; regenerate if missing.
if [ ! -f "$cpp_root/gen/iiwa14_ik.hpp" ]; then
  ( cd "$cpp_root/.." && python scripts/cpp_emit.py --all )
fi

# The export requires the Eigen3 CMake package (Eigen3::Eigen). On Linux
# (libeigen3-dev) CMake finds it under /usr by default; Homebrew installs it off
# CMake's default search path, so hint it. EIGEN_PREFIX overrides.
eigen_prefix="${EIGEN_PREFIX:-}"
if [ -z "$eigen_prefix" ] && command -v brew >/dev/null 2>&1; then
  eigen_prefix="$(brew --prefix eigen 2>/dev/null || true)"
fi

echo "== configure + install ssik_cpp -> $prefix =="
cmake -S "$cpp_root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DSSIK_CPP_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$prefix" \
  ${eigen_prefix:+-DCMAKE_PREFIX_PATH="$eigen_prefix"} >/dev/null
cmake --install "$build" >/dev/null

echo "== configure + build the standalone consumer against the installed package =="
cmake -S "$cpp_root/examples/consumer" -B "$cons" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$prefix${eigen_prefix:+;$eigen_prefix}" >/dev/null
cmake --build "$cons" >/dev/null

echo "== run =="
"$cons/solve_arm"
echo "C++ consumer smoke: PASS"
