"""Soundness, recovery and completeness as three separate contracts (#573).

The acceptance framework for M7. Today's gates establish soundness well and
completeness barely: ``check_solve_coverage`` counts a pose as covered when at
least one IK comes back, and native conformance mostly uses Python as the
completeness oracle, so an omission shared by both passes. These three contracts
are deliberately distinct, because an arm can satisfy any one while failing
another:

**Soundness** every returned ``q`` closes under independent FK. A solver that
returns nothing is vacuously sound.

**Recovery** for a sampled full-rank ``q*``, ``solve(fk(q*))`` returns a
configuration equivalent to ``q*``. This is what a user means by "it found my
pose". A solver returning seven of eight branches usually still passes, because
the sampled ``q*`` is usually one of the seven.

**Completeness** the returned set matches the full branch set. This is the one
#571 fails, and the only one that can fail while the other two pass.

The expected branch sets come from ``tests/data/branch_goldens.json``, computed
offline by a chart-free oracle (``scripts/regen_branch_goldens.py``). Nothing
here runs the oracle: a stabilized run is thousands of LM solves, and the whole
point of committing the result is that per-PR cost stays near zero. The
oracle-versus-golden check lives in ``test_branch_oracle_slow.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ssik.kinematics.poe_fk import poe_forward_kinematics
from tests._branch_fixtures import BY_NAME, FIXTURES, BranchFixture, load_goldens
from tests._branch_oracle import wrapped_linf

_EQUIV_TOL = 1e-6  # two configurations are the same branch below this
_FK_TOL = 1e-9

_GOLDENS = load_goldens()["fixtures"]
_NAMES = [f.name for f in FIXTURES]


def _golden(name: str) -> dict[str, Any]:
    assert name in _GOLDENS, (
        f"no golden for fixture {name!r}; run python scripts/regen_branch_goldens.py"
    )
    golden: dict[str, Any] = _GOLDENS[name]
    return golden


def _branches(name: str) -> list[np.ndarray]:
    return [np.asarray(b, dtype=np.float64) for b in _golden(name)["branches"]]


@pytest.fixture(scope="module")
def solved() -> dict[str, Any]:
    """One solve per fixture, shared by the contracts below."""
    out = {}
    for fx in FIXTURES:
        arm = fx.build()
        t = arm.fk(fx.q_star_array())
        out[fx.name] = (arm, t, arm.solve(t, respect_limits=False))
    return out


# ---------------------------------------------------------------------------
# The goldens have to be trustworthy before anything is measured against them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_golden_matches_its_fixture(name: str) -> None:
    """A golden records the fingerprint of the numbers it was generated from, so
    editing a fixture without regenerating fails here instead of silently
    checking against a stale expectation."""
    fx: BranchFixture = BY_NAME[name]
    g = _golden(name)
    assert g["fingerprint"] == fx.fingerprint(), (
        f"{name}: the fixture changed since its branch set was generated. "
        f"Re-run: python scripts/regen_branch_goldens.py --fixture {name}"
    )
    assert g["stabilized"], (
        f"{name}: golden was recorded from an oracle run that had not stabilized, "
        f"so its branch count is a lower bound and cannot gate completeness"
    )


@pytest.mark.parametrize("name", _NAMES)
def test_golden_branches_are_distinct_and_close_fk(name: str, solved: dict[str, Any]) -> None:
    arm, t, _ = solved[name]
    branches = _branches(name)
    assert branches, f"{name}: golden has no branches"
    for i, a in enumerate(branches):
        resid = float(np.linalg.norm(poe_forward_kinematics(arm.kinbody, a) - t))
        assert resid <= 1e-11, f"{name}: golden branch fails FK at {resid:.2e}"
        for b in branches[i + 1 :]:
            assert wrapped_linf(a, b) > 1e-4, f"{name}: golden holds a duplicate branch"


@pytest.mark.parametrize("name", _NAMES)
def test_golden_covers_everything_the_solver_finds(name: str, solved: dict[str, Any]) -> None:
    """The golden must be a superset of the solver's output.

    A solver branch the golden lacks does not mean the solver is wrong, it means
    the oracle under-searched and the golden cannot be used to judge
    completeness. Checking the direction that invalidates the reference is the
    point: a statistical oracle can stabilize on an incomplete set.
    """
    _, _, sols = solved[name]
    branches = _branches(name)
    missing = [
        s.q
        for s in sols
        if not any(wrapped_linf(np.asarray(s.q), b) <= _EQUIV_TOL for b in branches)
    ]
    assert not missing, (
        f"{name}: solver returned {len(missing)} branch(es) absent from the golden, so the "
        f"golden is an incomplete reference. Raise the oracle budget and regenerate."
    )


# ---------------------------------------------------------------------------
# Contract 1: soundness.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
def test_soundness_every_returned_branch_closes_fk(name: str, solved: dict[str, Any]) -> None:
    arm, t, sols = solved[name]
    assert sols, f"{name}: no solutions at a rank-6 pose"
    for s in sols:
        resid = float(np.linalg.norm(poe_forward_kinematics(arm.kinbody, s.q) - t))
        assert resid <= _FK_TOL, f"{name}: unsound branch, FK residual {resid:.2e}"


# ---------------------------------------------------------------------------
# Contracts 2 and 3: recovery and completeness.
#
# Expectation comes from the fixture's own `issue` field rather than a
# hand-written marker per case: a fixture tagged with an open defect is expected
# to fail, everything else must pass. strict=True, so fixing the defect turns
# these red and forces the tag to be cleared, which is the milestone exit
# condition #573 asks for.
# ---------------------------------------------------------------------------


def _expect(fx: BranchFixture) -> Any:
    if fx.issue:
        return pytest.param(
            fx.name,
            marks=pytest.mark.xfail(
                reason=f"{fx.issue}: a configuration at pi is unrepresentable in the "
                f"affine tan-half coordinate, so this branch is never reconstructed",
                strict=True,
            ),
        )
    return pytest.param(fx.name)


_CASES = [_expect(f) for f in FIXTURES]


@pytest.mark.parametrize("name", _CASES)
def test_recovery_returns_the_seeded_configuration(name: str, solved: dict[str, Any]) -> None:
    fx = BY_NAME[name]
    _, _, sols = solved[name]
    nearest = min(wrapped_linf(np.asarray(s.q), fx.q_star_array()) for s in sols)
    assert nearest <= _EQUIV_TOL, (
        f"{name}: q* not recovered; nearest returned branch is {nearest:.3f} rad away"
    )


@pytest.mark.parametrize("name", _CASES)
def test_completeness_matches_the_golden(name: str, solved: dict[str, Any]) -> None:
    _, _, sols = solved[name]
    branches = _branches(name)
    missing = [
        b for b in branches if not any(wrapped_linf(np.asarray(s.q), b) <= _EQUIV_TOL for s in sols)
    ]
    assert not missing, (
        f"{name}: solver returned {len(sols)} of {len(branches)} branches; "
        f"missing {[np.round(b, 4).tolist() for b in missing]}"
    )


def test_the_three_contracts_are_not_redundant(solved: dict[str, Any]) -> None:
    """Pins the distinction this module exists to make: on #571's fixture the
    solver is sound and incomplete at once, so a gate that only checks FK
    closure reports success while a branch is missing."""
    name = "tan_half_infinity"
    arm, t, sols = solved[name]
    assert all(
        float(np.linalg.norm(poe_forward_kinematics(arm.kinbody, s.q) - t)) <= _FK_TOL for s in sols
    ), "expected soundness to hold here"
    assert len(sols) < len(_branches(name)), (
        "expected this fixture to be incomplete; if the solver now returns every "
        "branch, #571 is fixed and the xfails above should be removed"
    )
