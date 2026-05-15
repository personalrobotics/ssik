"""Diagnostic / explain mode on Manipulator.solve (#265).

Smoke-tests for ``Manipulator.solve(T, explain=True) -> (sols, Diagnostic)``.
The diagnostic record attributes empty-list failures to a concrete cause
(unreachable / out-of-limits / capped) so callers don't have to guess.

Per-prebuilt explain mode (codegen-template work, three templates total)
is tracked separately as a follow-up; only the Manipulator path is
covered here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik import Diagnostic, Manipulator

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ur5() -> Manipulator:
    return ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")


# ---------------------------------------------------------------------------
# Default path: explain=False preserves the list-only return signature.
# ---------------------------------------------------------------------------


def test_default_returns_list_not_tuple(ur5: Manipulator) -> None:
    q = np.array([0.3, -0.5, 0.7, 0.2, 0.4, -0.1])
    T = ur5.fk(q)
    result = ur5.solve(T)
    assert isinstance(result, list)
    assert all(hasattr(s, "q") for s in result)


def test_explain_false_explicitly_returns_list(ur5: Manipulator) -> None:
    q = np.array([0.3, -0.5, 0.7, 0.2, 0.4, -0.1])
    T = ur5.fk(q)
    result = ur5.solve(T, explain=False)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Explain=True: returns a (sols, Diagnostic) tuple.
# ---------------------------------------------------------------------------


def test_explain_true_returns_tuple_with_diagnostic(ur5: Manipulator) -> None:
    q = np.array([0.3, -0.5, 0.7, 0.2, 0.4, -0.1])
    T = ur5.fk(q)
    result = ur5.solve(T, explain=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    sols, diag = result
    assert isinstance(sols, list)
    assert isinstance(diag, Diagnostic)


def test_explain_diagnostic_fields_on_reachable_pose(ur5: Manipulator) -> None:
    """A reachable pose: raw_candidates > 0, final_count > 0, FK is finite."""
    q = np.array([0.3, -0.5, 0.7, 0.2, 0.4, -0.1])
    T = ur5.fk(q)
    sols, diag = ur5.solve(T, explain=True)
    assert sols
    assert diag.solver_name == "ikgeo.three_parallel"
    assert diag.solver_tier == 0
    assert diag.raw_candidates >= len(sols)
    assert diag.final_count == len(sols)
    assert diag.max_fk_residual < 1e-6
    assert math.isfinite(diag.max_fk_residual)
    assert diag.dispatch_reason  # non-empty
    assert diag.fk_atol > 0
    assert diag.refinement_engaged == 0  # no LM by default on tier-0


def test_explain_diagnostic_summary_is_renderable(ur5: Manipulator) -> None:
    q = np.array([0.3, -0.5, 0.7, 0.2, 0.4, -0.1])
    T = ur5.fk(q)
    _, diag = ur5.solve(T, explain=True)
    summary = diag.summary()
    assert isinstance(summary, str)
    assert "ikgeo.three_parallel" in summary
    assert "tier 0" in summary


# ---------------------------------------------------------------------------
# Empty-list attribution: explain mode distinguishes unreachable from filtered.
# ---------------------------------------------------------------------------


def test_unreachable_pose_diagnostic_reports_zero_raw_candidates(ur5: Manipulator) -> None:
    """Far-away pose: solver returns 0 raw candidates -> diagnostic
    attributes the empty list to 'unreachable', not to filtering."""
    T_unreachable = np.eye(4)
    T_unreachable[:3, 3] = [10.0, 10.0, 10.0]  # outside UR5's ~1m reach
    sols, diag = ur5.solve(T_unreachable, explain=True)
    assert not sols
    assert diag.raw_candidates == 0
    assert diag.final_count == 0
    assert "unreachable" in diag.summary().lower()


def test_diagnostic_reports_max_solutions_truncation(ur5: Manipulator) -> None:
    """``max_solutions=N`` truncation is reflected in dropped_by_max_solutions."""
    q = np.array([0.3, -0.5, 0.7, 0.2, 0.4, -0.1])
    T = ur5.fk(q)
    sols_full, diag_full = ur5.solve(T, explain=True)
    sols_capped, diag_capped = ur5.solve(T, max_solutions=1, explain=True)
    assert len(sols_full) > 1, "test setup requires a multi-branch pose"
    assert len(sols_capped) == 1
    assert diag_capped.dropped_by_max_solutions == len(sols_full) - 1
    assert diag_full.dropped_by_max_solutions == 0


# ---------------------------------------------------------------------------
# Top-level export.
# ---------------------------------------------------------------------------


def test_diagnostic_exposed_at_top_level() -> None:
    """``ssik.Diagnostic`` is the canonical import path for type checks."""
    assert ssik.Diagnostic is Diagnostic
