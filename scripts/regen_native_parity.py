"""Regenerate the native-parity gate's committed data (#626).

Writes two files next to the gate (tests/test_native_parity.py):

- ``tests/data/native_parity_poses.json``, per arm:
  - ``near_singular``: the sigma_min-minimised poses. The minimiser is an
    optimisation whose argmin can move by an ulp between platforms, so it is
    committed rather than recomputed, and only ``--new-poses`` rewrites it.
  - ``cells`` and ``fast_pins``, per platform (``sys.platform``): the
    (direction:class) cells the full tier fails there, and up to three
    full-tier witness poses per cell. The fast (PR) tier runs the union of
    every platform's pins, so on each platform it fails the same cells as
    the full tier.
- ``tests/data/native_parity_oracle.json``: the chart-free branch oracle's
  verdict at every 6R pose where it was consulted. The gate looks a pose up
  here and runs the oracle live only for a pose this file lacks, so a missing
  entry costs time, never a wrong verdict. Entries are merged across runs and
  platforms; ``--fresh-oracle`` rebuilds them.

Boundary cases (a joint exactly at a limit, a rescue at a singular pose)
resolve differently on Linux and macOS, so each platform's cells come from a
run on that platform: locally for macOS, and for Linux from the nightly
workflow dispatched with ``regen=true``
(``gh workflow run slow.yml --ref <branch> -f regen=true``), whose
``native-parity-data`` artifact holds the regenerated files. The printed
KNOWN_* tables merge every platform recorded so far; paste them into the
gate. A rerun on the same platform reproduces both files byte for byte.

The round-off class (``ROUNDOFF_CLASSES``: exact-limit ties, #624) also
flips between CI runners of one platform, so a cell of it
that a CI run shows but the regeneration run did not is recorded by hand
under the ``ci`` key of that arm's ``cells``, which no rerun overwrites.

    uv run python scripts/regen_native_parity.py            # this platform's cells
    uv run python scripts/regen_native_parity.py --arm ur5_ik --arm fr3_ik
    uv run python scripts/regen_native_parity.py --new-poses  # also re-minimise
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tests import _native_parity as npar  # noqa: E402

# Witness poses kept per (arm, class) cell, first in full-tier order. More
# than one, so a cell still fails on a platform where one witness's numerics
# happen to agree.
_PINS_PER_CELL = 3


def _near_singular(arm: str) -> tuple[str, list[list[float]]]:
    kb = npar.module(arm)._KB
    return arm, [q.tolist() for q in npar.near_singular(arm, kb, npar.FULL["near_singular"])]


def _evaluate(
    arm: str, fresh_oracle: bool
) -> tuple[str, dict[str, Any], dict[str, list[str]], float]:
    # The oracle's verdict depends only on the pose, so committed entries are
    # reused unless asked otherwise; poses no longer consulted are dropped.
    npar.USE_COMMITTED_ORACLE = not fresh_oracle
    t0 = time.perf_counter()
    rep = npar.evaluate(arm, "full")
    oracle = {}
    for pid in sorted(rep.oracle_poses, key=_pose_order):
        adj = npar.adjudicate(arm, pid)
        # An unstabilised set is a lower bound the verdict never reads.
        branches = adj.branches.tolist() if adj.stabilized else []
        oracle[pid] = {"stabilized": adj.stabilized, "branches": branches}
    cells: dict[str, list[str]] = defaultdict(list)
    for direction, gaps in (("forward", rep.forward), ("reverse", rep.reverse)):
        for g in gaps:
            key = f"{direction}:{g.cls}"
            if g.pose not in cells[key]:
                cells[key].append(g.pose)
    return arm, oracle, dict(cells), time.perf_counter() - t0


def _evaluate_job(job: tuple[str, bool]) -> tuple[str, dict[str, Any], dict[str, list[str]], float]:
    return _evaluate(*job)


def _pose_order(pid: str) -> tuple[int, int]:
    s, k = pid.split("/")
    return list(npar.FULL).index(s), int(k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--arm", action="append", help="only these arms (repeatable)")
    ap.add_argument(
        "--new-poses",
        action="store_true",
        help="re-minimise the sigma_min poses (drops every platform's cells and pins)",
    )
    ap.add_argument("--fresh-oracle", action="store_true", help="rerun every oracle adjudication")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()
    arms = args.arm or npar.arm_names()
    here = sys.platform

    pose_data: dict[str, Any] = (
        json.loads(npar.POSES_FILE.read_text())
        if npar.POSES_FILE.exists()
        else {"seed": npar.SEED, "arms": {}}
    )
    oracle_data: dict[str, Any] = (
        json.loads(npar.ORACLE_FILE.read_text())
        if npar.ORACLE_FILE.exists() and not args.fresh_oracle
        else {"arms": {}}
    )
    pose_data["seed"] = npar.SEED
    # Slowest first: the 7R arms dominate the wall-clock.
    arms.sort(key=lambda a: (-npar.module(a).DOF, a))

    with Pool(args.jobs) as pool:
        if args.new_poses:
            for arm, qs in pool.imap_unordered(_near_singular, arms):
                pose_data["arms"][arm] = {"near_singular": qs, "cells": {}, "fast_pins": {}}
                print(f"[poses] {arm}: {len(qs)} near-singular", flush=True)
            _write(npar.POSES_FILE, pose_data)
            npar._pose_data.cache_clear()

        for arm, oracle, cells, dt in pool.imap_unordered(
            _evaluate_job, [(a, args.fresh_oracle) for a in arms]
        ):
            oracle_data["arms"].setdefault(arm, {}).update(oracle)
            oracle_data["arms"][arm] = dict(
                sorted(oracle_data["arms"][arm].items(), key=lambda kv: _pose_order(kv[0]))
            )
            pins: list[str] = []
            for key in sorted(cells):
                pins += [p for p in cells[key][:_PINS_PER_CELL] if p not in pins]
            rec = pose_data["arms"][arm]
            rec.setdefault("cells", {})[here] = sorted(cells)
            rec.setdefault("fast_pins", {})[here] = sorted(pins, key=_pose_order)
            rec["cells"] = dict(sorted(rec["cells"].items()))
            rec["fast_pins"] = dict(sorted(rec["fast_pins"].items()))
            counts = {k: len(v) for k, v in sorted(cells.items())}
            print(f"[gate] {arm} ({dt:.0f} s): {counts or 'clean'}", flush=True)

    oracle_data["arms"] = dict(sorted(oracle_data["arms"].items()))
    pose_data["arms"] = dict(sorted(pose_data["arms"].items()))
    _write(npar.ORACLE_FILE, oracle_data)
    _write(npar.POSES_FILE, pose_data)
    _print_tables(pose_data)


def _print_tables(pose_data: dict[str, Any]) -> None:
    """The KNOWN_* tables of tests/test_native_parity.py, merged over every
    platform recorded in the poses file. A round-off class's cell is known
    everywhere once any platform (or ``ci``) records it."""
    platforms = ["darwin", "linux"]
    print("\nKNOWN_* tables:")
    for direction in ("forward", "reverse"):
        table: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for arm, rec in pose_data["arms"].items():
            for platform, keys in rec.get("cells", {}).items():
                for key in keys:
                    d, cls = key.split(":")
                    if d == direction:
                        table[cls][arm].append(platform)
        print(f"KNOWN_{direction.upper()}: dict[str, dict[str, tuple[str, ...]]] = {{")
        for cls in sorted(table):
            print(f"    {cls!r}: {{")
            for arm, where in sorted(table[cls].items()):
                on = sorted(set(where) & set(platforms))
                spec = "ALL" if cls in npar.ROUNDOFF_CLASSES or on == platforms else repr(tuple(on))
                print(f"        {arm!r}: {spec},")
            print("    },")
        print("}")


def _dumps(x: Any, indent: str = "") -> str:
    """JSON with one configuration (a list of numbers) per line, so a diff
    shows which poses moved."""
    inner = indent + " "
    if isinstance(x, dict) and x:
        items = [f"{inner}{json.dumps(k)}: {_dumps(v, inner)}" for k, v in x.items()]
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"
    if isinstance(x, list) and x and not all(isinstance(v, (int, float)) for v in x):
        return "[\n" + ",\n".join(inner + _dumps(v, inner) for v in x) + "\n" + indent + "]"
    return json.dumps(x)


def _write(path: Path, data: dict[str, Any]) -> None:
    """Atomic: a failed run leaves the last good file in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(_dumps(data) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
