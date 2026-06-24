#!/usr/bin/env python
"""Pre-clone every ``robot_descriptions`` entry the test suite needs (#348).

CI caches ``~/.cache/robot_descriptions``, but on a cache miss the description
repos (some with mesh files: example-robot-data, drake/iiwa, UR) are cloned at
test time -- ~30-55 min and the dominant source of CI wall-clock + run-to-run
variance. Running this once, *serially, before pytest*, (a) populates the cache
so it can be saved for next time and (b) means the parallel (``-n auto``) test
workers only ever read the cache -- no concurrent clones racing on the same
directory.

The list is derived from ``MANIFEST.toml`` provenance lines
(``fixture_source = "robot_descriptions / <desc> ..."``) -- the same source the
parity test reads -- so adding an arm needs no edit here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ssik.prebuilt._manifest import load_manifest

_MARKER = "robot_descriptions / "


def _descriptions() -> list[str]:
    """Distinct ``robot_descriptions`` module names cited in the manifest."""
    out: set[str] = set()
    for arm in load_manifest().values():
        src = arm.fixture_source
        if _MARKER not in src:
            continue
        tail = src.split(_MARKER, 1)[1]
        desc = ""
        for ch in tail:
            if ch.isspace() or ch == "(":
                break
            desc += ch
        if desc:
            out.add(desc)
    return sorted(out)


def main() -> int:
    try:
        from robot_descriptions.loaders.yourdfpy import load_robot_description
    except ImportError:
        print("[prewarm] robot_descriptions not installed; nothing to do")
        return 0

    descs = _descriptions()
    print(f"[prewarm] warming {len(descs)} description(s)")
    failed: list[str] = []
    for desc in descs:
        t0 = time.perf_counter()
        try:
            load_robot_description(desc)
            print(f"[prewarm]   {desc}: ok ({time.perf_counter() - t0:.1f}s)")
        except Exception as exc:
            failed.append(desc)
            print(f"[prewarm]   {desc}: FAILED {type(exc).__name__}: {exc}")
    # Non-fatal: a description that fails to clone here will be retried (and
    # surface its real error) in the test that needs it.
    if failed:
        print(f"[prewarm] {len(failed)} description(s) could not be warmed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
