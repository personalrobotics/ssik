"""Re-derive the committed branch goldens from scratch (#573).

Marked slow: this is the only place the chart-free oracle actually runs, and a
stabilized run is thousands of LM solves per fixture. The fast contract tests
read ``tests/data/branch_goldens.json`` instead, so a committed golden that
drifted from what the oracle now produces would otherwise go unnoticed. This is
what keeps the reference honest, on the nightly/slow schedule rather than per
pull request.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._branch_fixtures import FIXTURES, load_goldens
from tests._branch_oracle import enumerate_branches, wrapped_linf

_EQUIV_TOL = 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("fixture", FIXTURES, ids=[f.name for f in FIXTURES])
def test_oracle_reproduces_the_golden(fixture) -> None:
    golden = load_goldens()["fixtures"][fixture.name]
    assert golden["fingerprint"] == fixture.fingerprint(), (
        "fixture changed since generation; re-run scripts/regen_branch_goldens.py"
    )

    arm = fixture.build()
    res = enumerate_branches(arm.kinbody, arm.fk(fixture.q_star_array()))
    assert res.stabilized, f"oracle did not stabilize at budget {res.budget_used}"

    expected = [np.asarray(b, dtype=np.float64) for b in golden["branches"]]
    assert len(res) == len(expected), (
        f"{fixture.name}: oracle now finds {len(res)} branches, golden has "
        f"{len(expected)}. If the solver or the fixture changed intentionally, "
        f"regenerate; otherwise this is a real change in branch structure."
    )
    for b in expected:
        assert res.contains(b, _EQUIV_TOL), (
            f"{fixture.name}: golden branch {np.round(b, 4).tolist()} no longer found"
        )
    for got in res.branches:
        assert any(wrapped_linf(got, b) <= _EQUIV_TOL for b in expected), (
            f"{fixture.name}: oracle found a branch absent from the golden: "
            f"{np.round(got, 4).tolist()}"
        )
