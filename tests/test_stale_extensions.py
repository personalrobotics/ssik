"""A stale in-place compiled module never runs silently in a checkout (#614).

hatch_build.py compiles its CYTHON_TARGETS beside their sources, and Python
imports the extension in place of the ``.py``. After a pull changes the source,
the old extension would keep running with no error. Each test copies this
checkout's package (with whatever the build hook left in it) to a temporary
tree, laid out as a checkout (``hatch_build.py`` beside ``src/``) or as an
install, and imports it in a fresh interpreter. The artifact is that
interpreter's exit status and stderr.

Reproduce: ``uv run pytest tests/test_stale_extensions.py``. By hand: append a
line to ``src/ssik/refinement/__init__.py`` after ``uv sync`` and run
``uv run python -c "import ssik"``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest

from ssik._stale_extensions import REBUILD, STAMP_SUFFIX, source_digest

_REPO = Path(__file__).resolve().parent.parent
_SUFFIX = EXTENSION_SUFFIXES[0]
_TARGET = Path("refinement") / "__init__.py"  # one of hatch_build.CYTHON_TARGETS


def _copy(tmp_path: Path, *, checkout: bool) -> Path:
    pkg = tmp_path / "src" / "ssik"
    shutil.copytree(_REPO / "src" / "ssik", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    if checkout:
        shutil.copy(_REPO / "hatch_build.py", tmp_path)
    return pkg


def _extension(pkg: Path) -> Path:
    """The compiled target in ``pkg``. Where this checkout runs pure Python, a
    placeholder recorded against the current source stands in for the build."""
    source = pkg / _TARGET
    ext = source.with_name(source.stem + _SUFFIX)
    if not ext.exists():
        ext.write_bytes(b"placeholder")
        ext.with_name(ext.name + STAMP_SUFFIX).write_text(source_digest(source))
    return ext


def _import(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "src")}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True
    )


def test_fresh_build_imports_compiled_and_ignores_other_interpreters(tmp_path: Path) -> None:
    pkg = _copy(tmp_path, checkout=True)
    source = pkg / _TARGET
    ext = source.with_name(source.stem + _SUFFIX)
    # Another interpreter's extension, never recorded: this one cannot import it.
    version = f"{sys.version_info.major}{sys.version_info.minor}"
    other = source.with_name(source.stem + _SUFFIX.replace(version, "39", 1))
    assert other.name != ext.name
    other.write_bytes(b"placeholder")

    result = _import(tmp_path, "import ssik.refinement as r; print(r.__file__)")
    assert result.returncode == 0, result.stderr
    loaded = Path(result.stdout.strip())
    assert loaded.parent == source.parent
    # Compiled where this checkout was built, the source where it was not.
    assert loaded.name == (ext.name if ext.exists() else source.name)


@pytest.mark.parametrize("staleness", ["source changed", "no record"])
def test_stale_extension_refuses_import_in_checkout(tmp_path: Path, staleness: str) -> None:
    pkg = _copy(tmp_path, checkout=True)
    ext = _extension(pkg)
    if staleness == "source changed":
        with (pkg / _TARGET).open("a") as f:
            f.write("\n# a pulled change\n")
        reason = f"{ext} was built from an older __init__.py"
    else:
        ext.with_name(ext.name + STAMP_SUFFIX).unlink()
        reason = f"{ext} has no record of the source it was built from"

    result = _import(tmp_path, "import ssik")
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert reason in result.stderr
    assert REBUILD in result.stderr


def test_installed_package_is_not_checked(tmp_path: Path) -> None:
    """An install has no hatch_build.py beside it and was built in one piece."""
    pkg = _copy(tmp_path, checkout=False)
    _extension(pkg)
    with (pkg / _TARGET).open("a") as f:
        f.write("\n# differs from the record\n")

    result = _import(tmp_path, "import ssik")
    assert result.returncode == 0, result.stderr
