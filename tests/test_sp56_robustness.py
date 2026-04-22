"""Extra robustness tests for SP5 and SP6: scale invariance, input
validation, dedup, count guards, best-LS fallback.

These are the "super robust" checks from the #48 follow-up: ensuring the
hardened subproblems behave consistently across the edge cases a real
pipeline can throw at them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ssik.subproblems import sp5, sp6
from ssik.subproblems._rotation import rotate


def _unit(v: np.ndarray) -> np.ndarray:
    return v / float(np.linalg.norm(v))


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _sp5_consistent_inputs(
    scale: float, rng: np.random.Generator
) -> tuple[Any, Any, Any, Any, Any, Any, Any, float, float, float]:
    """Build an SP5 case at the given length scale with a known seeded triple."""
    k1 = _unit(rng.standard_normal(3))
    k2 = _unit(rng.standard_normal(3))
    k3 = _unit(rng.standard_normal(3))
    p1 = rng.standard_normal(3) * scale
    p2 = rng.standard_normal(3) * scale
    p3 = rng.standard_normal(3) * scale
    t1, t2, t3 = 0.5, -0.7, 1.2
    rhs = rotate(k2, t2, p2 + rotate(k3, t3, p3))
    p0 = rhs - rotate(k1, t1, p1)
    return p0, p1, p2, p3, k1, k2, k3, t1, t2, t3


def _sp6_consistent_inputs(
    scale: float, rng: np.random.Generator
) -> tuple[list[Any], list[Any], list[Any], float, float, float, float]:
    k1 = _unit(rng.standard_normal(3))
    k2 = _unit(rng.standard_normal(3))
    k = [k1, k2, k1, k2]
    p = [rng.standard_normal(3) * scale for _ in range(4)]
    h = [rng.standard_normal(3) for _ in range(4)]
    t1, t2 = 0.6, -1.1
    d1 = float(h[0] @ rotate(k[0], t1, p[0])) + float(h[1] @ rotate(k[1], t2, p[1]))
    d2 = float(h[2] @ rotate(k[2], t1, p[2])) + float(h[3] @ rotate(k[3], t2, p[3]))
    return h, k, p, d1, d2, t1, t2


# ---------------------------------------------------------------------------
# (4) Scale invariance: SP5 / SP6 work at mm and km scales as well as unit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_sp5_scale_invariance(scale: float) -> None:
    """Correctness of SP5 is invariant under uniform scaling of the p
    vectors. Robot kinematics range from millimetre to kilometre scales;
    the solver should not lose precision or return different counts."""
    rng = np.random.default_rng(1234)
    p0, p1, p2, p3, k1, k2, k3, t1, t2, t3 = _sp5_consistent_inputs(scale, rng)
    solutions, is_ls = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    assert not is_ls, f"unexpected LS at scale {scale}"
    assert 1 <= len(solutions) <= 4
    # Every returned solution still satisfies the equation (residual scales
    # linearly with input scale, so widen the check proportionally).
    for s1, s2, s3 in solutions:
        lhs = p0 + rotate(k1, s1, p1)
        rhs = rotate(k2, s2, p2 + rotate(k3, s3, p3))
        assert np.allclose(lhs, rhs, atol=scale * 1e-4)
    # The seeded triple is in the returned set (mod dedup tolerance).
    assert any(
        abs(_wrap(s1 - t1)) < 1e-3 and abs(_wrap(s2 - t2)) < 1e-3 and abs(_wrap(s3 - t3)) < 1e-3
        for s1, s2, s3 in solutions
    )


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_sp6_scale_invariance(scale: float) -> None:
    rng = np.random.default_rng(4567)
    h, k, p, d1, d2, _t1, _t2 = _sp6_consistent_inputs(scale, rng)
    solutions, is_ls = sp6.solve(h, k, p, d1, d2)
    assert not is_ls, f"unexpected LS at scale {scale}"
    assert 1 <= len(solutions) <= 4
    # Residual for SP6 scales with scale (h is O(1), p is O(scale) -> d is O(scale)).
    for s1, s2 in solutions:
        lhs1 = float(h[0] @ rotate(k[0], s1, p[0])) + float(h[1] @ rotate(k[1], s2, p[1]))
        lhs2 = float(h[2] @ rotate(k[2], s1, p[2])) + float(h[3] @ rotate(k[3], s2, p[3]))
        assert abs(lhs1 - d1) < scale * 1e-4
        assert abs(lhs2 - d2) < scale * 1e-4


# ---------------------------------------------------------------------------
# (3) Input validation.
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_sp5_wrong_shape_raises(self) -> None:
        good = np.array([1.0, 0.0, 0.0])
        bad = np.array([1.0, 2.0])  # 2D instead of 3
        with pytest.raises(ValueError, match=r"shape"):
            sp5.solve(bad, good, good, good, good, good, good)

    def test_sp5_nan_input_raises(self) -> None:
        good = np.array([1.0, 0.0, 0.0])
        bad = np.array([np.nan, 0.0, 0.0])
        with pytest.raises(ValueError, match=r"non-finite"):
            sp5.solve(good, good, good, good, bad, good, good)

    def test_sp6_wrong_shape_raises(self) -> None:
        good = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match=r"shape"):
            sp6.solve(
                [np.array([1.0, 2.0]), good, good, good],
                [good, good, good, good],
                [good, good, good, good],
                0.0,
                0.0,
            )

    def test_sp6_nan_d_raises(self) -> None:
        good = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match=r"d1.*d2"):
            sp6.solve(
                [good, good, good, good],
                [good, good, good, good],
                [good, good, good, good],
                float("nan"),
                0.0,
            )

    def test_sp6_wrong_length_raises(self) -> None:
        good = np.array([1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match=r"length-4"):
            sp6.solve([good] * 3, [good] * 4, [good] * 4, 0.0, 0.0)


# ---------------------------------------------------------------------------
# (1) Dedup: SP5 never returns two solutions within subproblem_dedup.
# ---------------------------------------------------------------------------


def test_sp5_no_near_duplicate_solutions_in_output() -> None:
    """Random well-posed SP5 input: each pair of returned solutions is
    separated by at least ``subproblem_dedup`` on at least one joint."""
    rng = np.random.default_rng(9876)
    p0, p1, p2, p3, k1, k2, k3, *_ = _sp5_consistent_inputs(1.0, rng)
    solutions, _ = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    for i, a in enumerate(solutions):
        for b in solutions[i + 1 :]:
            distances = [abs(_wrap(a[j] - b[j])) for j in range(3)]
            assert max(distances) >= 1e-3 - 1e-9, (
                f"near-duplicate solutions survived dedup: {a} vs {b}"
            )


def test_sp6_no_near_duplicate_solutions_in_output() -> None:
    rng = np.random.default_rng(5432)
    h, k, p, d1, d2, *_ = _sp6_consistent_inputs(1.0, rng)
    solutions, _ = sp6.solve(h, k, p, d1, d2)
    for i, a in enumerate(solutions):
        for b in solutions[i + 1 :]:
            distances = [abs(_wrap(a[j] - b[j])) for j in range(2)]
            assert max(distances) >= 1e-3 - 1e-9


# ---------------------------------------------------------------------------
# (5) Best-LS return on infeasibility.
# ---------------------------------------------------------------------------


def test_sp5_infeasible_returns_best_ls() -> None:
    """Input where the LHS magnitude range is disjoint from the RHS
    magnitude range: |p0 + Rot(k1, t1) p1| in [|p0| - |p1|, |p0| + |p1|]
    can never equal |Rot(k2, t2) (p2 + Rot(k3, t3) p3)| = |p2 + Rot(k3, t3) p3|
    in [||p2| - |p3||, |p2| + |p3|]. With p0 large and p1 / p2 / p3 small,
    these ranges don't overlap and SP5 is provably infeasible. Expect
    is_ls=True with at most 1 best-LS triple."""
    p0 = np.array([100.0, 0.0, 0.0])  # |LHS| ~ 100
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.5, 0.0, 0.0])
    p3 = np.array([0.5, 0.0, 0.5])  # |RHS| < 2
    k1 = np.array([0.0, 0.0, 1.0])
    k2 = np.array([0.0, 1.0, 0.0])
    k3 = np.array([1.0, 0.0, 0.0])
    solutions, is_ls = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    # Either a best-LS single solution is returned, or the quartic had no
    # real roots (empty). Both are valid infeasibility signals.
    assert is_ls
    assert len(solutions) <= 1


def test_sp6_infeasible_returns_best_ls() -> None:
    rng = np.random.default_rng(1357)
    h, k, p, d1, d2, *_ = _sp6_consistent_inputs(1.0, rng)
    # Make d1 unreachable (far outside the SP4-like boundary for this p, k, h).
    solutions, is_ls = sp6.solve(h, k, p, d1 + 100.0, d2)
    assert is_ls
    assert len(solutions) <= 1
