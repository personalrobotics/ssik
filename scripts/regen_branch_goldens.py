"""Regenerate ``tests/data/branch_goldens.json``: the complete branch set of each
curated fixture, found by the chart-free oracle (#573).

Run after adding or editing a fixture in ``tests/_branch_fixtures.py``. The
contract tests read the committed result instead of recomputing it, because a
stabilized oracle run is thousands of LM solves: seconds to minutes per fixture,
which is far too slow for every pull request.

    uv run python scripts/regen_branch_goldens.py

Why a golden rather than a live computation. The oracle is deliberately
independent of the solvers, which means it is a search rather than a
construction, and searches cost time and are only statistically complete. Fixing
the answer in a reviewed file turns that into an asset: the expected branch
structure of each fixture becomes something a reader can see, and a diff to it
in a pull request is a visible claim that the structure changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

from tests._branch_fixtures import FIXTURES, GOLDEN_PATH  # noqa: E402
from tests._branch_oracle import enumerate_branches  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", action="append", help="regenerate only these (repeatable)")
    args = ap.parse_args()

    wanted = set(args.fixture) if args.fixture else None
    existing = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {"fixtures": {}}
    out: dict[str, object] = {"fixtures": dict(existing.get("fixtures", {}))}

    for fx in FIXTURES:
        if wanted is not None and fx.name not in wanted:
            continue
        arm = fx.build()
        t_target = arm.fk(fx.q_star_array())
        t0 = time.perf_counter()
        res = enumerate_branches(arm.kinbody, t_target)
        elapsed = time.perf_counter() - t0

        if not res.stabilized:
            print(
                f"  {fx.name}: NOT STABILIZED at budget {res.budget_used}; the branch "
                f"count was still growing, so this is a lower bound. Raise the budgets "
                f"in tests/_branch_oracle.py before committing.",
                file=sys.stderr,
            )
        out["fixtures"][fx.name] = {  # type: ignore[index]
            "fingerprint": fx.fingerprint(),
            "issue": fx.issue,
            "description": fx.description,
            "n_branches": len(res),
            "branches": [[float(v) for v in q] for q in res.branches],
            "worst_fk": res.worst_fk,
            "budget_used": res.budget_used,
            "stabilized": res.stabilized,
        }
        print(
            f"  {fx.name}: {len(res)} branches, worst FK {res.worst_fk:.1e}, "
            f"budget {res.budget_used}, {elapsed:.1f}s"
        )

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
