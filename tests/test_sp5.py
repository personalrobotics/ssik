"""Tests for :mod:`ssik.subproblems.sp5` (three-rotation composition).

Roundtrip strategy: pick random inputs + seeded angles, evaluate LHS and RHS
to set up a consistent system, run SP5, verify the seeded triple appears in
the returned set (mod 2pi) and that every returned triple satisfies the
defining equation.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from ssik.subproblems import sp5
from ssik.subproblems._rotation import rotate


def _unit(v: np.ndarray) -> np.ndarray:
    return v / float(np.linalg.norm(v))


@st.composite
def _sp5_case(
    draw: st.DrawFn,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    """Generate generic-position SP5 inputs via a seeded numpy RNG.

    Drawing components directly via hypothesis frequently finds measure-zero
    degenerate alignments (p0 perpendicular to ``k_i x k_2``, all vectors
    equal to a common axis, etc.) that don't reflect real-robot kinematics.
    Sampling Gaussian instead keeps the distribution generic.
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    k1 = _unit(rng.standard_normal(3))
    k2 = _unit(rng.standard_normal(3))
    k3 = _unit(rng.standard_normal(3))
    assume(float(np.linalg.norm(np.cross(k1, k2))) > 0.3)
    assume(float(np.linalg.norm(np.cross(k2, k3))) > 0.3)

    p1 = rng.standard_normal(3)
    p2 = rng.standard_normal(3)
    p3 = rng.standard_normal(3)
    for p, k in ((p1, k1), (p3, k3)):
        p_perp_sq = float(np.dot(p, p)) - float(np.dot(k, p)) ** 2
        assume(p_perp_sq > 0.1)

    t1 = float(rng.uniform(-np.pi + 0.1, np.pi - 0.1))
    t2 = float(rng.uniform(-np.pi + 0.1, np.pi - 0.1))
    t3 = float(rng.uniform(-np.pi + 0.1, np.pi - 0.1))

    # Construct a consistent system: solve for p0 so the equation holds at (t1, t2, t3).
    rhs = rotate(k2, t2, p2 + rotate(k3, t3, p3))
    p0 = rhs - rotate(k1, t1, p1)

    return p0, p1, p2, p3, k1, k2, k3, t1, t2, t3


_Sp5Case = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]


@given(_sp5_case())
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
)
def test_roundtrip_seeded_triple_in_solution_set(case: _Sp5Case) -> None:
    p0, p1, p2, p3, k1, k2, k3, t1, t2, t3 = case
    solutions, is_ls = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    assert not is_ls
    assert 1 <= len(solutions) <= 4

    def wrap(a: float) -> float:
        return float(((a + np.pi) % (2 * np.pi)) - np.pi)

    # Quartic-root + sign-branch recovery loses precision on random-position
    # inputs; IK-Geo's reference tests use the same class of tolerances.
    found = any(
        abs(wrap(s1 - t1)) < 1e-3 and abs(wrap(s2 - t2)) < 1e-3 and abs(wrap(s3 - t3)) < 1e-3
        for s1, s2, s3 in solutions
    )
    assert found, f"seeded (t1={t1}, t2={t2}, t3={t3}) not found in {solutions}"


@given(_sp5_case())
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
)
def test_every_solution_satisfies_equation(case: _Sp5Case) -> None:
    p0, p1, p2, p3, k1, k2, k3, _t1, _t2, _t3 = case
    solutions, _ = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    for s1, s2, s3 in solutions:
        lhs = p0 + rotate(k1, s1, p1)
        rhs = rotate(k2, s2, p2 + rotate(k3, s3, p3))
        assert np.allclose(lhs, rhs, atol=1e-3), (
            f"SP5 equation fails at (t1={s1}, t2={s2}, t3={s3}): lhs={lhs}, rhs={rhs}"
        )
