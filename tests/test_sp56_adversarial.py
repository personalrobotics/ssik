"""Adversarial property tests that allow hypothesis to find pathological
SP5/SP6 inputs (as opposed to the Gaussian-sampled generic-position tests
in :file:`test_sp5.py` / :file:`test_sp6.py`).

The claim being tested: with the beyond-IK-Geo hardening (upfront
degeneracy detection + post-verification against the original equation),
every solution SP5/SP6 returns satisfies the defining equation within
``subproblem_numerical`` -- even when hypothesis finds measure-zero
degeneracies that would silently produce wrong results on stock IK-Geo.

We do **not** require the solver to return the seeded solution on every
input: numerical precision loss in the quartic reduction can drop a
legitimate real root. That's a separate concern (tracked alongside #48
for future work). The property here is narrower: *no returned solution
is wrong*.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from ssik.subproblems import sp5, sp6
from ssik.subproblems._rotation import rotate


def _unit(v: np.ndarray) -> np.ndarray:
    return v / float(np.linalg.norm(v))


_FINITE = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False, width=64)
_ANGLE = st.floats(min_value=-np.pi + 1e-2, max_value=np.pi - 1e-2, allow_nan=False, width=64)


def _vec3(draw: st.DrawFn) -> np.ndarray:
    return np.array([draw(_FINITE), draw(_FINITE), draw(_FINITE)])


# ---------------------------------------------------------------------------
# SP5 adversarial
# ---------------------------------------------------------------------------


@st.composite
def _sp5_adversarial(draw: st.DrawFn) -> tuple[np.ndarray, ...]:
    k1 = _vec3(draw)
    k2 = _vec3(draw)
    k3 = _vec3(draw)
    assume(float(np.linalg.norm(k1)) > 0.1)
    assume(float(np.linalg.norm(k2)) > 0.1)
    assume(float(np.linalg.norm(k3)) > 0.1)
    k1 = _unit(k1)
    k2 = _unit(k2)
    k3 = _unit(k3)

    p0 = _vec3(draw)
    p1 = _vec3(draw)
    p2 = _vec3(draw)
    p3 = _vec3(draw)
    return p0, p1, p2, p3, k1, k2, k3


@given(_sp5_adversarial())
@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
)
def test_sp5_returned_solutions_always_satisfy_equation(case: tuple[np.ndarray, ...]) -> None:
    """Every solution SP5 returns satisfies the defining equation.

    Even when hypothesis finds pathological geometries (k-parallel,
    p-collinear), SP5 either detects the degeneracy upfront (returns
    ``([], True)``) or post-verifies each candidate before returning.
    No silent-nonsense allowed.
    """
    p0, p1, p2, p3, k1, k2, k3 = case
    solutions, _ = sp5.solve(p0, p1, p2, p3, k1, k2, k3)
    for s1, s2, s3 in solutions:
        lhs = p0 + rotate(k1, s1, p1)
        rhs = rotate(k2, s2, p2 + rotate(k3, s3, p3))
        assert np.allclose(lhs, rhs, atol=1e-4), (
            f"SP5 returned invalid solution (t1={s1}, t2={s2}, t3={s3}): "
            f"|lhs - rhs| = {float(np.linalg.norm(lhs - rhs))}"
        )


# ---------------------------------------------------------------------------
# SP6 adversarial
# ---------------------------------------------------------------------------


@st.composite
def _sp6_adversarial(
    draw: st.DrawFn,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float, float]:
    k1 = _vec3(draw)
    k2 = _vec3(draw)
    assume(float(np.linalg.norm(k1)) > 0.1)
    assume(float(np.linalg.norm(k2)) > 0.1)
    k1 = _unit(k1)
    k2 = _unit(k2)
    k = [k1, k2, k1, k2]

    p = [_vec3(draw) for _ in range(4)]
    h = [_vec3(draw) for _ in range(4)]
    d1 = draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    d2 = draw(st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False))
    return h, k, p, d1, d2


@given(_sp6_adversarial())
@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.data_too_large],
)
def test_sp6_returned_solutions_always_satisfy_equations(
    case: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], float, float],
) -> None:
    """Every solution SP6 returns satisfies both defining equations."""
    h, k, p, d1, d2 = case
    solutions, _ = sp6.solve(h, k, p, d1, d2)
    for s1, s2 in solutions:
        lhs1 = float(h[0] @ rotate(k[0], s1, p[0])) + float(h[1] @ rotate(k[1], s2, p[1]))
        lhs2 = float(h[2] @ rotate(k[2], s1, p[2])) + float(h[3] @ rotate(k[3], s2, p[3]))
        assert abs(lhs1 - d1) < 1e-4, (
            f"SP6 eq 1 residual {abs(lhs1 - d1)} at (t1={s1}, t2={s2}) exceeds 1e-4"
        )
        assert abs(lhs2 - d2) < 1e-4, (
            f"SP6 eq 2 residual {abs(lhs2 - d2)} at (t1={s1}, t2={s2}) exceeds 1e-4"
        )
