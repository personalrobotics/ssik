#!/usr/bin/env python
"""Pick a pull request's CI lane and which sweeps may run their PR sample (stdlib only).

usage: classify_changes.py < CHANGED_PATHS

Reads the changed paths, one per line, and prints ``full=true|false`` and
``pr_sweeps=<comma-separated names>`` for ``$GITHUB_OUTPUT``. CI's
``Classify changed files`` job feeds it a pull request's diff, and nothing for
any other event, so:

- ``full=false`` (the examples lane) only when every path is under
  ``examples/`` or ``docs/``, or is a ``*.md``, ``LICENSE`` or ``.gitignore``,
  and none is under a code directory. No paths means ``full=true``.
- ``pr_sweeps`` names each sweep in tests/_sweeps.py that no path is covered
  by. No paths means none, so every sweep runs in full.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import _sweeps

_CODE_DIRS = ("src/", "tests/", "scripts/", "cpp/", "tools/", ".github/")


def _examples_or_docs(path: str) -> bool:
    if path.startswith(_CODE_DIRS):
        return False
    return (
        path.startswith(("examples/", "docs/"))
        or path.endswith(".md")
        or path in ("LICENSE", ".gitignore")
    )


def classify(paths: list[str]) -> tuple[bool, list[str]]:
    """``(full, sampled sweeps)`` for a change touching ``paths``."""
    if not paths:
        return True, []
    full = not all(_examples_or_docs(p) for p in paths)
    sampled = [s for s in _sweeps.SWEEPS if not any(_sweeps.covers(s, p) for p in paths)]
    return full, sampled


def main() -> int:
    paths = [line.strip() for line in sys.stdin if line.strip()]
    full, sampled = classify(paths)
    print(f"full={'true' if full else 'false'}")
    print(f"pr_sweeps={','.join(sampled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
