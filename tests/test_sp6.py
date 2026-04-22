"""Tests for :mod:`ssik.subproblems.sp6` (two coupled SP4-like equations).

Roundtrip strategy: pick random inputs and a known ``(theta1, theta2)``,
evaluate the two equations to get ``(d1, d2)``, and verify that SP6 returns
the seeded pair within its solution set.

Inputs are drawn from a seeded Gaussian RNG (same approach as
:file:`test_sp5.py`); this avoids the measure-zero degenerate configurations
IK-Geo's subproblems do not handle (tracked in #48).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from ssik.subproblems import sp6
from ssik.subproblems._rotation import rotate


def _unit(v: np.ndarray) -> np.ndarray:
    return v / float(np.linalg.norm(v))


_Sp6Case = tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    float,
    float,
    float,
    float,
]


@st.composite
def _sp6_case(draw: st.DrawFn) -> _Sp6Case:
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    k1 = _unit(rng.standard_normal(3))
    k2 = _unit(rng.standard_normal(3))
    assume(float(np.linalg.norm(np.cross(k1, k2))) > 0.3)

    p_list = [rng.standard_normal(3) for _ in range(4)]
    h_list = [rng.standard_normal(3) for _ in range(4)]
    for idx, p in enumerate(p_list):
        k_for_this = k1 if idx in (0, 2) else k2
        p_perp_sq = float(np.dot(p, p)) - float(np.dot(k_for_this, p)) ** 2
        assume(p_perp_sq > 0.1)

    t1 = float(rng.uniform(-np.pi + 0.1, np.pi - 0.1))
    t2 = float(rng.uniform(-np.pi + 0.1, np.pi - 0.1))

    k_list = [k1, k2, k1, k2]

    d1 = float(h_list[0] @ rotate(k_list[0], t1, p_list[0])) + float(
        h_list[1] @ rotate(k_list[1], t2, p_list[1])
    )
    d2 = float(h_list[2] @ rotate(k_list[2], t1, p_list[2])) + float(
        h_list[3] @ rotate(k_list[3], t2, p_list[3])
    )

    return h_list, k_list, p_list, d1, d2, t1, t2


@given(_sp6_case())
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
)
def test_roundtrip_seeded_pair_in_solution_set(case: _Sp6Case) -> None:
    h, k, p, d1, d2, t1, t2 = case
    solutions, is_ls = sp6.solve(h, k, p, d1, d2)
    assert not is_ls
    assert 1 <= len(solutions) <= 4

    def wrap(a: float) -> float:
        return float(((a + np.pi) % (2 * np.pi)) - np.pi)

    found = any(abs(wrap(s1 - t1)) < 1e-5 and abs(wrap(s2 - t2)) < 1e-5 for s1, s2 in solutions)
    assert found, f"seeded (t1={t1}, t2={t2}) not in recovered solutions {solutions}"


@given(_sp6_case())
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
)
def test_every_solution_satisfies_equations(case: _Sp6Case) -> None:
    h, k, p, d1, d2, _t1, _t2 = case
    solutions, _ = sp6.solve(h, k, p, d1, d2)
    for s1, s2 in solutions:
        lhs1 = float(h[0] @ rotate(k[0], s1, p[0])) + float(h[1] @ rotate(k[1], s2, p[1]))
        lhs2 = float(h[2] @ rotate(k[2], s1, p[2])) + float(h[3] @ rotate(k[3], s2, p[3]))
        assert abs(lhs1 - d1) < 1e-6, f"eq 1 mismatch at ({s1}, {s2}): {lhs1} vs {d1}"
        assert abs(lhs2 - d2) < 1e-6, f"eq 2 mismatch at ({s1}, {s2}): {lhs2} vs {d2}"


def test_input_shape_validation() -> None:
    h = [np.array([1.0, 0.0, 0.0])] * 3  # only 3, not 4
    k = [np.array([0.0, 0.0, 1.0])] * 4
    p = [np.array([0.0, 1.0, 0.0])] * 4
    with pytest.raises(ValueError, match="length-4"):
        sp6.solve(h, k, p, 0.0, 0.0)
