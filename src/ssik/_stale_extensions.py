"""Refuse stale compiled modules in a source checkout (#614).

``hatch_build.py`` compiles its ``CYTHON_TARGETS`` in place, so ``foo.py`` gains a
``foo.<EXT_SUFFIX>`` beside it and Python imports the extension instead of the
source. Nothing rebuilds an editable install when a pull changes ``foo.py``, so
the old compiled code would keep running with no error. The build hook therefore
records the sha256 of each source it compiles next to the extension
(``<extension>.source-sha256``). In a checkout, ``import ssik`` checks every
extension the running interpreter would load in place of a ``.py`` and refuses
one whose record is missing or names other source. Extensions for other
interpreters (``foo.cpython-310-*.so`` under 3.13) are never imported, so they
are not checked here; their own interpreter checks them.

An installed wheel is not a checkout (there is no ``hatch_build.py`` beside its
``src/``), and its sources and extensions were built together, so the check
costs it one ``stat``.
"""

from __future__ import annotations

import hashlib
import os
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

STAMP_SUFFIX = ".source-sha256"
REBUILD = "uv sync --reinstall-package ssik"


def source_digest(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


def stale_extensions(package_dir: Path) -> list[str]:
    """One line per extension under ``package_dir`` that shadows a ``.py`` and
    was not built from that file's current content."""
    problems = []
    for dirpath, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        names = set(filenames)
        for name in sorted(filenames):
            for suffix in EXTENSION_SUFFIXES:
                source = name.removesuffix(suffix) + ".py"
                if not name.endswith(suffix) or source not in names:
                    continue
                ext = Path(dirpath, name)
                stamp = Path(dirpath, name + STAMP_SUFFIX)
                if not stamp.is_file():
                    problems.append(f"{ext} has no record of the source it was built from")
                elif stamp.read_text().strip() != source_digest(Path(dirpath, source)):
                    problems.append(f"{ext} was built from an older {source}")
    return problems


def check_checkout() -> None:
    package_dir = Path(__file__).resolve().parent
    if not (package_dir.parent.parent / "hatch_build.py").is_file():
        return  # installed, not a source checkout
    problems = stale_extensions(package_dir)
    if problems:
        listed = "\n".join(f"  {p}" for p in problems)
        raise ImportError(
            f"ssik: compiled modules in this checkout do not match their source:\n{listed}\n"
            f"Python would run the old compiled code, not the source. Rebuild them with\n"
            f"  {REBUILD}\n"
            f"or delete the listed files to run the pure-Python source."
        )
