"""Soundness, recovery and completeness as three separate contracts (#573).

The acceptance framework for M7. Today's gates establish soundness well and
completeness barely: ``check_solve_coverage`` counts a pose as covered when at
least one IK comes back, and native conformance mostly uses Python as the
completeness oracle, so an omission shared by both passes. These three
contracts are deliberately distinct, because an arm can satisfy any one of them
while failing another:

**Soundness** every returned ``q`` closes under independent FK. A solver that
returns nothing is vacuously sound.

**Recovery** for a sampled full-rank ``q*``, ``solve(fk(q*))`` returns a
configuration equivalent to ``q*``. This is what a user means by "it found my
pose". A solver returning seven of eight branches usually still passes, because
the sampled ``q*`` is usually one of the seven.

**Completeness** the returned set matches an independent branch oracle. This is
the one #571 fails, and the only one that can fail while the other two pass.

The oracle (``tests._branch_oracle``) shares no algebra with the solvers, which
is the point: it works in joint space, so a configuration at tan-half-angle
infinity is unremarkable to it.
"""

from __future__ import annotations

import numpy as np
import pytest

import ssik
from ssik.kinematics.poe_fk import poe_forward_kinematics
from tests._branch_oracle import OracleResult, enumerate_branches, wrapped_linf

PI = np.pi

# #571's exact reproducer: a regular pose (rank 6, cond ~123) whose linearity
# joint sits at pi, i.e. at tan-half-angle infinity.
_571_ALPHA = [PI / 2, PI / 2, PI / 2, PI / 2, PI / 2, 0.0]
_571_A = [1 / 5, 1 / 4, 1 / 3, 1 / 6, 1 / 7, 1 / 8]
_571_D = [1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5]
_571_QSTAR = np.array([0.0, PI / 2, PI, PI / 2, PI / 2, 0.0])

_EQUIV_TOL = 1e-6  # two configurations are the same branch below this


@pytest.fixture(scope="module")
def arm_571() -> ssik.Manipulator:
    return ssik.Manipulator.from_dh(dh_alpha=_571_ALPHA, dh_a=_571_A, dh_d=_571_D)


@pytest.fixture(scope="module")
def oracle_571(arm_571: ssik.Manipulator) -> OracleResult:
    """Stabilized oracle at #571's pose. Module-scoped: this is thousands of LM
    solves and the result is reused by every contract below."""
    return enumerate_branches(arm_571.kinbody, arm_571.fk(_571_QSTAR))


# ---------------------------------------------------------------------------
# The oracle itself has to be trustworthy before anything is measured with it.
# ---------------------------------------------------------------------------


def test_oracle_stabilizes_and_is_sound(
    arm_571: ssik.Manipulator, oracle_571: OracleResult
) -> None:
    """A growing branch count means the oracle has not finished looking, so an
    unstabilized result is a lower bound and must never be read as completeness."""
    assert oracle_571.stabilized, (
        f"oracle still finding new branches at budget {oracle_571.budget_used}; "
        f"raise the budget before trusting the count"
    )
    assert oracle_571.worst_fk <= 1e-12, f"oracle branch fails FK: {oracle_571.worst_fk:.2e}"
    for i, a in enumerate(oracle_571.branches):
        for b in oracle_571.branches[i + 1 :]:
            assert wrapped_linf(a, b) > 1e-4, "oracle returned a duplicate branch"


def test_oracle_finds_the_configuration_the_chart_cannot_represent(
    oracle_571: OracleResult,
) -> None:
    """The reason the oracle may not share the solver's coordinates (#571).

    ``q*`` has its linearity joint at exactly pi, which the tan-half-angle
    coordinate sends to infinity. A higher-precision run of the same algebra
    would drop it for the same reason the solver does; a joint-space search
    does not care.
    """
    d = oracle_571.nearest(_571_QSTAR)
    assert d <= _EQUIV_TOL, f"oracle missed q* itself (nearest {d:.2e}); it cannot be the reference"


# ---------------------------------------------------------------------------
# Contract 1: soundness. Passes today.
# ---------------------------------------------------------------------------


def test_soundness_every_returned_branch_closes_fk(arm_571: ssik.Manipulator) -> None:
    T = arm_571.fk(_571_QSTAR)
    sols = arm_571.solve(T, respect_limits=False)
    assert sols, "no solutions at a rank-6 pose"
    for s in sols:
        resid = float(np.linalg.norm(poe_forward_kinematics(arm_571.kinbody, s.q) - T))
        assert resid <= 1e-9, f"unsound branch, FK residual {resid:.2e}"


# ---------------------------------------------------------------------------
# Contract 2: recovery. Fails on this fixture, which is the bug.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="#571: the linearity joint at pi is dropped as a nonfinite generalized "
    "eigenvalue, so q* is never reconstructed",
    strict=True,
)
def test_recovery_solve_returns_the_seeded_configuration(arm_571: ssik.Manipulator) -> None:
    T = arm_571.fk(_571_QSTAR)
    sols = arm_571.solve(T, respect_limits=False)
    nearest = min(wrapped_linf(np.asarray(s.q), _571_QSTAR) for s in sols)
    assert nearest <= _EQUIV_TOL, (
        f"q* not recovered; nearest returned branch is {nearest:.3f} rad away"
    )


# ---------------------------------------------------------------------------
# Contract 3: completeness. Fails on this fixture, and is the only contract
# that can fail while soundness and recovery both pass.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="#571: seven of the eight branches are returned; the eighth needs the "
    "projective root at tan-half-angle infinity",
    strict=True,
)
def test_completeness_matches_the_independent_oracle(
    arm_571: ssik.Manipulator, oracle_571: OracleResult
) -> None:
    T = arm_571.fk(_571_QSTAR)
    sols = arm_571.solve(T, respect_limits=False)
    missing = [
        b
        for b in oracle_571.branches
        if not any(wrapped_linf(np.asarray(s.q), b) <= _EQUIV_TOL for s in sols)
    ]
    assert not missing, (
        f"solver returned {len(sols)} of the oracle's {len(oracle_571)} branches; "
        f"missing {[np.round(b, 4).tolist() for b in missing]}"
    )


def test_the_three_contracts_are_not_redundant(
    arm_571: ssik.Manipulator, oracle_571: OracleResult
) -> None:
    """Pins the distinction this module exists to make.

    On this fixture the solver is sound and incomplete at the same time, so a
    gate that only checks FK closure reports success while a branch is missing.
    """
    T = arm_571.fk(_571_QSTAR)
    sols = arm_571.solve(T, respect_limits=False)
    assert all(
        float(np.linalg.norm(poe_forward_kinematics(arm_571.kinbody, s.q) - T)) <= 1e-9
        for s in sols
    ), "expected soundness to hold here"
    assert len(sols) < len(oracle_571), (
        "expected this fixture to be incomplete; if the solver now returns every "
        "branch, #571 is fixed and the xfails above should be removed"
    )
