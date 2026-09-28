#!/usr/bin/env python
"""Prove the CI pytest shards ran the whole suite, once (stdlib only).

usage: check_pytest_shards.py RECORD.json [RECORD.json ...]

Each record is what ``pytest --shard K/N --shard-record PATH`` wrote in one CI
job (tests/conftest.py). Records are grouped by their parent directory's name
(one group per Python version). Per group, fails unless every shard 1..N is
present exactly once with the same N, all shards collected the same tests,
the shards' selections are pairwise disjoint, and their union is the whole
collection. So a shard dropped from the matrix, a mismatched N, or a
partition bug turns CI red instead of silently skipping tests.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def check(group: str, records: list[dict]) -> list[str]:
    errs: list[str] = []
    ns = {r["n"] for r in records}
    if len(ns) != 1:
        return [f"{group}: shards disagree on N: {sorted(ns)}"]
    n = ns.pop()
    ks = sorted(r["k"] for r in records)
    if ks != list(range(1, n + 1)):
        errs.append(f"{group}: expected shards 1..{n}, got {ks}")
    full = set(records[0]["collected"])
    if any(set(r["collected"]) != full for r in records):
        errs.append(f"{group}: shards collected different tests")
    seen: dict[str, int] = {}
    for r in records:
        for nid in r["selected"]:
            if nid in seen:
                errs.append(f"{group}: {nid} ran in shards {seen[nid]} and {r['k']}")
            seen[nid] = r["k"]
    missing = full - seen.keys()
    extra = seen.keys() - full
    if missing:
        errs.append(
            f"{group}: {len(missing)} collected tests ran in no shard, e.g. {sorted(missing)[:3]}"
        )
    if extra:
        errs.append(
            f"{group}: {len(extra)} tests ran but were not collected, e.g. {sorted(extra)[:3]}"
        )
    if not errs:
        sizes = ", ".join(
            f"{r['k']}/{n}: {len(r['selected'])}" for r in sorted(records, key=lambda r: r["k"])
        )
        print(f"{group}: {len(full)} tests, shards {sizes} -- disjoint and complete")
    return errs


def main(paths: list[str]) -> int:
    if not paths:
        print("no shard records given")
        return 1
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        groups[Path(p).parent.name].append(json.loads(Path(p).read_text()))
    errs = [e for g in sorted(groups) for e in check(g, groups[g])]
    for e in errs:
        print("FAIL:", e)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
