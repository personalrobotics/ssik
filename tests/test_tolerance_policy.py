"""Tests for the :class:`TolerancePolicy` refactor.

Verifies that subproblem solvers actually consume the policy argument
(rather than the old hardcoded tolerances) by constructing inputs near
each relevant threshold and asserting the same call flips behaviour when
the policy is loosened.
"""

from __future__ import annotations

import numpy as np

from ssik import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.subproblems import sp1, sp4


def test_sp1_feasibility_tolerance_respected() -> None:
    """SP1's is_ls flag flips when the policy loosens the feasibility
    threshold past the actual mismatch."""
    k = np.array([0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    # q has |q| = 1 but k.q differs from k.p by 1e-5 (above default 1e-9).
    q = np.array([1.0, 0.0, 1e-5])

    _, is_ls_default = sp1.solve(k, p, q)
    assert is_ls_default, "default policy should flag mismatch as LS"

    loose = TolerancePolicy(subproblem_feasibility=1e-3)
    _, is_ls_loose = sp1.solve(k, p, q, loose)
    assert not is_ls_loose, "loose policy should absorb the 1e-5 mismatch"


def test_sp4_feasibility_tolerance_respected() -> None:
    """SP4's LS branch fires when the target exceeds reach by more than
    subproblem_feasibility, and falls back to exact with a loose policy."""
    h = np.array([1.0, 0.0, 0.0])
    k = np.array([0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    # Max h . Rot(k, t) p is 1.0; ask for 1.0 + 1e-4 (above default 1e-9).
    d = 1.0 + 1e-4

    _, is_ls_default = sp4.solve(h, k, p, d)
    assert is_ls_default

    loose = TolerancePolicy(subproblem_feasibility=1e-2)
    _, is_ls_loose = sp4.solve(h, k, p, d, loose)
    assert not is_ls_loose


def test_default_policy_is_frozen() -> None:
    """The default policy is a concrete singleton; callers who pass None
    implicitly consume it."""
    assert isinstance(DEFAULT_TOLERANCE_POLICY, TolerancePolicy)
    assert DEFAULT_TOLERANCE_POLICY.subproblem_feasibility == 1e-9
    assert DEFAULT_TOLERANCE_POLICY.subproblem_numerical == 1e-5
    assert DEFAULT_TOLERANCE_POLICY.subproblem_degeneracy == 1e-12
