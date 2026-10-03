#!/usr/bin/env python
"""Pick a pull request's CI lanes and which sweeps may run their PR sample (stdlib only).

usage: classify_changes.py < CHANGED_PATHS

Reads the changed paths, one per line, and prints ``full=true|false``,
``cpp=true|false`` and ``pr_sweeps=<comma-separated names>`` for
``$GITHUB_OUTPUT``. CI's ``Classify changed files`` job feeds it a pull
request's diff, and nothing for any other event, so:

- ``full=false`` (the examples lane) only when every path is under
  ``examples/`` or ``docs/``, or is a ``*.md``, ``LICENSE`` or ``.gitignore``,
  and none is under a code directory. No paths means ``full=true``.
- ``cpp=false`` (skip the C++ emit, conformance and reused-suite jobs) only
  when every path is examples or docs, under ``tests/``, or one of the Python
  modules no C++ job imports (:data:`_NOT_CPP_INPUTS`). No paths means
  ``cpp=true``, and ``cpp=true`` always comes with ``full=true``.
- ``pr_sweeps`` names each sweep in tests/_sweeps.py that no path is covered
  by. No paths means none, so every sweep runs in full.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import _sweeps

_CODE_DIRS = ("src/", "tests/", "scripts/", "cpp/", "tools/", ".github/")

# Modules that neither the C++ emit (scripts/cpp_emit.py and the prebuilt arms it
# loads) nor anything they import reaches; tests/test_classify_changes.py checks
# that against the import graph. Everything else under src/ feeds the Python
# oracle the emit and the drift guard compare against, so it runs the C++ jobs.
# A trailing "/" is a directory prefix; anything else is an exact path.
_NOT_CPP_INPUTS = (
    "src/ssik/cli.py",
    "src/ssik/internals/",
    "src/ssik/teleop.py",
    "src/ssik/_validation.py",
)


def _examples_or_docs(path: str) -> bool:
    if path.startswith(_CODE_DIRS):
        return False
    return (
        path.startswith(("examples/", "docs/"))
        or path.endswith(".md")
        or path in ("LICENSE", ".gitignore")
    )


def _outside_cpp(path: str) -> bool:
    """True when ``path`` cannot change what the C++ jobs build, emit or check.

    The C++ jobs read no file under tests/ (the reused suite's tests also run,
    with the native extension built, in every Linux pytest job). scripts/,
    cpp/, the build files and .github/ always count as C++ inputs.
    """
    if _examples_or_docs(path) or path.startswith("tests/"):
        return True
    return any(path.startswith(p) if p.endswith("/") else path == p for p in _NOT_CPP_INPUTS)


def classify(paths: list[str]) -> tuple[bool, bool, list[str]]:
    """``(full, cpp, sampled sweeps)`` for a change touching ``paths``."""
    if not paths:
        return True, True, []
    full = not all(_examples_or_docs(p) for p in paths)
    cpp = not all(_outside_cpp(p) for p in paths)
    sampled = [s for s in _sweeps.SWEEPS if not any(_sweeps.covers(s, p) for p in paths)]
    return full, cpp, sampled


def main() -> int:
    paths = [line.strip() for line in sys.stdin if line.strip()]
    full, cpp, sampled = classify(paths)
    print(f"full={'true' if full else 'false'}")
    print(f"cpp={'true' if cpp else 'false'}")
    print(f"pr_sweeps={','.join(sampled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
