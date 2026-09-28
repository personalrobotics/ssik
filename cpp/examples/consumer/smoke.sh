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

# The export requires the Eigen3 CMake package (Eigen3::Eigen). By default use
# the pinned Eigen every ssik build compiles against (#606), and require exactly
# that version in both the install and the consumer. EIGEN_PREFIX overrides it
# with a CMake prefix of your own Eigen (any version).
eigen_prefix="${EIGEN_PREFIX:-}"
eigen_version=""
if [ -z "$eigen_prefix" ]; then
  eigen_prefix="$(python3 "$cpp_root/../scripts/fetch_eigen.py" --cmake-prefix)"
  eigen_version="$(python3 "$cpp_root/../scripts/fetch_eigen.py" --version)"
fi
echo "== Eigen: ${eigen_version:-any version} from $eigen_prefix =="

echo "== configure + install ssik_cpp -> $prefix =="
cmake -S "$cpp_root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
  -DSSIK_CPP_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DCMAKE_PREFIX_PATH="$eigen_prefix" -DSSIK_EIGEN_VERSION="$eigen_version" >/dev/null
cmake --install "$build" >/dev/null

echo "== configure + build the standalone consumer against the installed package =="
cmake -S "$cpp_root/examples/consumer" -B "$cons" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$prefix;$eigen_prefix" -DSSIK_EIGEN_VERSION="$eigen_version" >/dev/null
cmake --build "$cons" >/dev/null

echo "== run =="
"$cons/solve_arm"
echo "C++ consumer smoke: PASS"
