#!/usr/bin/env python
"""Measure the PR suite's per-test durations for ``pytest --shard`` balancing.

Runs ``pytest -m "not slow and not perf" --durations=0`` (extra arguments are
passed through, e.g. ``-n 4``) and writes ``tests/_shard_durations.json``:
``{nodeid: seconds}`` summed over setup/call/teardown, for tests taking at
least ``_MIN_S``. ``--shard K/N`` (tests/conftest.py) balances the shards by
these numbers; a test missing from the file counts as ``_MIN_S``.

The file only balances CI shards: a stale one makes them uneven, never
incomplete. Regenerate it when the suite's heavy tests change a lot:

  uv run python scripts/regen_test_durations.py -n 4
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_OUT = _REPO / "tests" / "_shard_durations.json"
_MIN_S = 0.1
_LINE = re.compile(r"^([\d.]+)s (setup|call|teardown)\s+(\S+)", re.M)


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not slow and not perf",
        "--durations=0",
        "-q",
        *sys.argv[1:],
    ]
    out = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True).stdout
    total: dict[str, float] = defaultdict(float)
    for m in _LINE.finditer(out):
        total[m.group(3)] += float(m.group(1))
    if not total:
        print(out[-2000:])
        print("[regen_test_durations] no --durations output parsed; not writing")
        return 1
    kept = {k: round(v, 2) for k, v in sorted(total.items()) if v >= _MIN_S}
    _OUT.write_text(json.dumps(kept, indent=0, sort_keys=True) + "\n")
    print(f"[regen_test_durations] {len(kept)} tests >= {_MIN_S}s -> {_OUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
