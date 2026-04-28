"""Smoke tests for :class:`ssik.core.solution.Solution`.

The dataclass itself is trivial; these tests pin the contract that solver
authors and downstream consumers rely on:

- frozen / hashable assumptions match callers' expectations
- defaults match the spec in GitHub #75 (``refinement_used="none"``,
  ``refinement_iters=0``, ``branch_id=None``, ``solver_name=""``)
- field order is what serializers/printing rely on
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from ssik.core.solution import Solution


def test_defaults_match_spec() -> None:
    sol = Solution(q=np.zeros(6), fk_residual=1e-12)
    assert sol.refinement_used == "none"
    assert sol.refinement_iters == 0
    assert sol.branch_id is None
    assert sol.solver_name == ""


def test_frozen_rejects_mutation() -> None:
    sol = Solution(q=np.zeros(6), fk_residual=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sol.fk_residual = 1.0  # type: ignore[misc]


def test_q_is_array_not_copied_on_construction() -> None:
    """Solution stores the ``q`` reference directly. Callers needing
    isolation should ``.copy()`` before constructing."""
    q = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    sol = Solution(q=q, fk_residual=0.0)
    assert sol.q is q


def test_field_order() -> None:
    fields = [f.name for f in dataclasses.fields(Solution)]
    assert fields == [
        "q",
        "fk_residual",
        "refinement_used",
        "refinement_iters",
        "branch_id",
        "solver_name",
    ]
