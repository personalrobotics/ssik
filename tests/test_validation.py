"""The bulletproof gate must actually reject a broken arm.

`ssik._validation.assert_solve_coverage` is what the `ssik add-arm` scaffold
calls. These tests prove it does its job: a good arm passes; an arm that returns
no solutions (the FANUC-M710 / Kinova-[0,0] failure mode from #424) trips the
coverage assertion rather than passing vacuously.

Both directions use a *real* shipped arm's geometry (ur5) so the test exercises
real joint-limit sampling, not a hand-rigged mock.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik._validation import assert_solve_coverage, check_solve_coverage
from ssik.prebuilt import ur5_ik


class _ZeroSolve:
    """Wraps a real arm but returns no solutions for every pose -- the shape of
    a broken solver / a locked-``[0, 0]``-limits fixture."""

    __name__ = "broken_arm"

    def __init__(self, base: object) -> None:
        self._base = base
        self._KB = base._KB  # type: ignore[attr-defined]
        self.DOF = base.DOF  # type: ignore[attr-defined]

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self._base.fk(q)  # type: ignore[attr-defined,no-any-return]

    def solve(self, t_target: np.ndarray, **_kw: object) -> list[object]:
        return []


def test_gate_passes_a_good_arm() -> None:
    coverage, worst_fk = assert_solve_coverage(
        ur5_ik, min_coverage=0.95, fk_ceiling=1e-6, n_poses=32
    )
    assert coverage == 1.0
    assert worst_fk < 1e-6


def test_gate_rejects_a_zero_solution_arm() -> None:
    """The core bulletproof claim: 0 coverage is a loud failure, not a pass."""
    broken = _ZeroSolve(ur5_ik)
    cov, _ = check_solve_coverage(broken, n_poses=32)  # type: ignore[arg-type]
    assert cov == 0.0
    with pytest.raises(AssertionError, match="coverage"):
        assert_solve_coverage(broken, min_coverage=0.95, n_poses=32)  # type: ignore[arg-type]


def test_gate_rejects_bad_fk_closure() -> None:
    """An arm that "solves" but with poor FK closure trips the FK ceiling."""

    class _BadFk(_ZeroSolve):
        __name__ = "bad_fk_arm"

        def solve(self, t_target: np.ndarray, **_kw: object) -> list[object]:
            # One solution, but with a 1 mm FK residual -- above a tight ceiling.
            sol = type("S", (), {"fk_residual": 1e-3})()
            return [sol]

    with pytest.raises(AssertionError, match="FK"):
        assert_solve_coverage(
            _BadFk(ur5_ik),  # type: ignore[arg-type]
            min_coverage=0.5,
            fk_ceiling=1e-6,
            n_poses=16,
        )
