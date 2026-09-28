#!/usr/bin/env python
"""Provide the one pinned Eigen that every native ssik build compiles against (#606).

The native extension and the C++ conformance build must use the same Eigen on
every platform: QZ convergence differs between Eigen releases (#599), so a
wheel built against a different Eigen is a different numerical artifact from
the one CI verified. This module downloads the pinned upstream release once,
checks its sha256, and unpacks it into a cache directory. It is stdlib-only
because ``hatch_build.py`` loads it inside an isolated build environment.

Usage:
    python scripts/fetch_eigen.py                  # print the include dir
    python scripts/fetch_eigen.py --cmake-prefix   # print a CMake install prefix
    python scripts/fetch_eigen.py --version        # print the pinned version

The cache lives in ``$SSIK_EIGEN_CACHE``, else ``$XDG_CACHE_HOME/ssik`` or
``~/.cache/ssik``. To build against a different Eigen on purpose, set
``SSIK_EIGEN_INCLUDE_DIR`` (Python builds) or pass your own CMake prefix; see
CONTRIBUTING.md.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

EIGEN_VERSION = "3.4.0"
EIGEN_URL = (
    f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_VERSION}/eigen-{EIGEN_VERSION}.tar.gz"
)
EIGEN_SHA256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"

# The explicit escape hatch: build against this Eigen include dir instead of
# the pin. Deliberately ssik-prefixed so a generic EIGEN_INCLUDE_DIR left in a
# developer's environment cannot silently replace the pin.
OVERRIDE_ENV = "SSIK_EIGEN_INCLUDE_DIR"


class EigenUnavailable(RuntimeError):
    """The pinned Eigen could not be downloaded (network, proxy, no curl)."""


def cache_root() -> Path:
    env = os.environ.get("SSIK_EIGEN_CACHE")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "ssik"


def header_version(include_dir: str | Path) -> str:
    """The Eigen version the headers under ``include_dir`` declare.

    Eigen >= 5 writes ``EIGEN_VERSION_STRING`` in ``Eigen/Version``; 3.x only
    has the WORLD.MAJOR.MINOR macros in ``Core/util/Macros.h``.
    """
    inc = Path(include_dir)
    version_h = inc / "Eigen" / "Version"
    if version_h.is_file():
        m = re.search(r'#define\s+EIGEN_VERSION_STRING\s+"([^"]+)"', version_h.read_text())
        if m:
            return m.group(1)
    macros = (inc / "Eigen" / "src" / "Core" / "util" / "Macros.h").read_text()
    parts = []
    for name in ("WORLD", "MAJOR", "MINOR"):
        m = re.search(rf"#define\s+EIGEN_{name}_VERSION\s+(\d+)", macros)
        if m is None:
            raise ValueError(f"no EIGEN_{name}_VERSION in {inc}")
        parts.append(m.group(1))
    return ".".join(parts)


def _download(dest: Path) -> None:
    # curl first: it is on every platform that builds native (Linux, macOS) and
    # uses the system trust store, which python.org macOS interpreters (the
    # ones cibuildwheel uses) lack until "Install Certificates" is run.
    curl = shutil.which("curl")
    if curl:
        r = subprocess.run(
            [curl, "-fsSL", "--retry", "3", "-o", str(dest), EIGEN_URL],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return
        raise EigenUnavailable(f"curl could not download {EIGEN_URL}: {r.stderr.strip()}")
    try:
        with urllib.request.urlopen(EIGEN_URL, timeout=60) as resp, open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    except OSError as exc:
        raise EigenUnavailable(f"could not download {EIGEN_URL}: {exc}") from exc


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract(tarball: Path, into: Path) -> None:
    with tarfile.open(tarball) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(into, filter="data")
        else:  # Python < 3.10.12 / 3.11.4: no extraction filters; check by hand.
            root = into.resolve()
            for member in tf.getmembers():
                target = (into / member.name).resolve()
                if not (member.isfile() or member.isdir()) or not target.is_relative_to(root):
                    raise RuntimeError(f"unexpected entry in Eigen tarball: {member.name}")
            tf.extractall(into)


def ensure_eigen() -> Path:
    """Return the include dir of the pinned Eigen, downloading it if needed.

    Raises EigenUnavailable if it cannot be downloaded, and RuntimeError if the
    download does not match the pinned sha256 or version.
    """
    root = cache_root()
    final = root / f"eigen-{EIGEN_VERSION}-{EIGEN_SHA256[:12]}"
    if (final / "Eigen" / "Dense").is_file():
        return final
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".eigen-") as tmp:
        tmp_dir = Path(tmp)
        tarball = tmp_dir / "eigen.tar.gz"
        _download(tarball)
        got = _sha256(tarball)
        if got != EIGEN_SHA256:
            raise RuntimeError(
                f"Eigen {EIGEN_VERSION} checksum mismatch: {EIGEN_URL} has sha256 {got}, "
                f"expected {EIGEN_SHA256}. Refusing to build against it."
            )
        _extract(tarball, tmp_dir)
        src = tmp_dir / f"eigen-{EIGEN_VERSION}"
        declared = header_version(src)
        if declared != EIGEN_VERSION:
            raise RuntimeError(f"Eigen tarball declares {declared}, expected {EIGEN_VERSION}")
        try:
            src.rename(final)
        except OSError:
            # A concurrent build finished first; its tree is the same bytes.
            if not (final / "Eigen" / "Dense").is_file():
                raise
    return final


def ensure_cmake_prefix() -> Path:
    """Return a CMake install prefix of the pinned Eigen (Eigen3Config.cmake).

    The C++ build and the consumer smoke find Eigen as the ``Eigen3`` CMake
    package, which the raw tarball does not carry; Eigen's own install step
    generates it. Needs ``cmake`` on PATH. Built once per cache.
    """
    src = ensure_eigen()
    prefix = src.with_name(src.name + "-prefix")
    if (prefix / "share" / "eigen3" / "cmake" / "Eigen3Config.cmake").is_file():
        return prefix
    with tempfile.TemporaryDirectory(dir=src.parent, prefix=".eigen-cmake-") as tmp:
        tmp_prefix = Path(tmp) / "prefix"
        build = Path(tmp) / "build"
        # NO_PACKAGE_REGISTRY: Eigen's configure would otherwise record this
        # throwaway build tree in ~/.cmake/packages, where later find_package
        # calls can pick it up.
        for cmd in (
            ["cmake", "-S", str(src), "-B", str(build), f"-DCMAKE_INSTALL_PREFIX={tmp_prefix}",
             "-DBUILD_TESTING=OFF", "-DEIGEN_BUILD_DOC=OFF",
             "-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON"],
            ["cmake", "--install", str(build)],
        ):  # fmt: skip
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout}\n{r.stderr}")
        try:
            tmp_prefix.rename(prefix)
        except OSError:
            if not (prefix / "share" / "eigen3" / "cmake" / "Eigen3Config.cmake").is_file():
                raise
    return prefix


def main() -> int:
    ap = argparse.ArgumentParser(description="Provide the pinned Eigen for native ssik builds.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--cmake-prefix", action="store_true", help="print a CMake install prefix")
    group.add_argument("--version", action="store_true", help="print the pinned version")
    args = ap.parse_args()
    if args.version:
        print(EIGEN_VERSION)
        return 0
    try:
        path = ensure_cmake_prefix() if args.cmake_prefix else ensure_eigen()
    except EigenUnavailable as exc:
        print(f"[fetch_eigen] {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
