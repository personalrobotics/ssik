"""Round-trip + edge-case tests for the generalized 3-axis (Davenport)
rotation decomposition (#354 foundation)."""

from __future__ import annotations

import numpy as np
import pytest

from ssik.kinematics._generalized_euler import (
    _axis_angle_matrix,
    decompose_3axis,
)


def _compose(n1, n2, n3, a, b, c):
    return (
        _axis_angle_matrix(np.asarray(n1, float) / np.linalg.norm(n1), a)
        @ _axis_angle_matrix(np.asarray(n2, float) / np.linalg.norm(n2), b)
        @ _axis_angle_matrix(np.asarray(n3, float) / np.linalg.norm(n3), c)
    )


_AXIS_SETS = {
    "ZYZ_symmetric": ([0, 0, 1.0], [0, 1.0, 0], [0, 0, 1.0]),
    "ZYX_asymmetric": ([0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0]),  # R1 Pro wrist class
    "YXZ": ([0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]),  # R1 Pro shoulder class
}


@pytest.mark.parametrize("name", list(_AXIS_SETS))
def test_round_trip_recovers_rotation(name: str) -> None:
    """Every returned (a,b,c) reconstructs R; at least one solution always exists."""
    n1, n2, n3 = _AXIS_SETS[name]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        a, b, c = rng.uniform(-3.1, 3.1, size=3)
        R = _compose(n1, n2, n3, a, b, c)
        sols = decompose_3axis(R, n1, n2, n3)
        assert sols, f"{name}: no decomposition returned"
        best = min(np.abs(_compose(n1, n2, n3, s[0], s[1], s[2]) - R).max() for s in sols)
        worst = max(worst, best)
    assert worst < 1e-9, f"{name}: worst reconstruction error {worst:.2e}"


def test_round_trip_random_axes() -> None:
    """Arbitrary (random, non-orthogonal) axis triples round-trip too."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        m1, m2, m3 = (rng.normal(size=3) for _ in range(3))
        a, b, c = rng.uniform(-3.1, 3.1, size=3)
        R = _compose(m1, m2, m3, a, b, c)
        sols = decompose_3axis(R, m1, m2, m3)
        assert sols
        worst = max(
            worst, min(np.abs(_compose(m1, m2, m3, sa, sb, sc) - R).max() for sa, sb, sc in sols)
        )
    assert worst < 1e-9, f"random axes: worst {worst:.2e}"


def test_two_branches_when_off_boundary() -> None:
    """A generic target yields two distinct middle-angle branches."""
    n1, n2, n3 = _AXIS_SETS["ZYX_asymmetric"]
    R = _compose(n1, n2, n3, 0.4, 0.9, -0.3)
    sols = decompose_3axis(R, n1, n2, n3)
    assert len(sols) == 2
    assert abs(sols[0][1] - sols[1][1]) > 1e-6  # distinct b


def test_gimbal_still_reconstructs() -> None:
    """At a gimbal (middle angle ~0), the decomposition still reconstructs R
    even though the individual outer angles are not unique."""
    n1, n2, n3 = _AXIS_SETS["ZYZ_symmetric"]
    R = _compose(n1, n2, n3, 0.7, 0.0, -0.2)  # b == 0 -> gimbal (a + c determined)
    sols = decompose_3axis(R, n1, n2, n3)
    assert sols
    assert min(np.abs(_compose(n1, n2, n3, s[0], s[1], s[2]) - R).max() for s in sols) < 1e-9


def test_collinear_axes_returns_empty() -> None:
    """Collinear axes admit no general decomposition -> empty, not a crash."""
    assert decompose_3axis(np.eye(3), [0, 0, 1.0], [0, 0, 1.0], [0, 0, 1.0]) == []
