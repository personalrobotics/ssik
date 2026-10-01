"""Tests for :mod:`ssik.subproblems.sp1`.

Coverage:

- **Roundtrip**: pick random ``k, theta, p``; compute ``q = Rot(k, theta) p``;
  assert ``sp1.solve(k, p, q)`` returns ``theta`` mod ``2pi`` with
  ``is_ls=False``.
- **Exact-feasible boundary**: inputs that satisfy the feasibility conditions
  should return ``is_ls=False``; inputs that violate them should return
  ``is_ls=True``.
- **Special cases**: ``p`` along ``k`` (rotation is identity), ``p = q``,
  ``q = -p``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ssik.subproblems import sp1
from ssik.subproblems._rotation import rotate


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _near_equal_mod_2pi(a: float, b: float, tol: float = 1e-7) -> bool:
    return abs(((a - b + np.pi) % (2 * np.pi)) - np.pi) < tol


_FINITE = st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False, width=64)
_ANGLE = st.floats(
    min_value=-np.pi + 1e-3,
    max_value=np.pi - 1e-3,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def _axis_theta_p(draw: st.DrawFn) -> tuple[np.ndarray, float, np.ndarray]:
    from hypothesis import assume

    k_raw = np.array([draw(_FINITE), draw(_FINITE), draw(_FINITE)])
    p = np.array([draw(_FINITE), draw(_FINITE), draw(_FINITE)])
    theta = draw(_ANGLE)
    # Both axis and source vector must have meaningful magnitude.
    assume(float(np.linalg.norm(k_raw)) >= 1e-2)
    assume(float(np.linalg.norm(p)) >= 1e-2)
    k = _unit(k_raw)
    # Reject p collinear with k: rotation is identity, theta is ill-defined.
    p_perp_sq = float(np.dot(p, p)) - float(np.dot(k, p)) ** 2
    assume(p_perp_sq >= 1e-4)
    return k, theta, p


@given(_axis_theta_p())
@settings(max_examples=200, deadline=None)
def test_roundtrip_exact(case: tuple[np.ndarray, float, np.ndarray]) -> None:
    k, theta, p = case
    q = rotate(k, theta, p)
    theta_rec, is_ls = sp1.solve(k, p, q)
    assert not is_ls
    assert _near_equal_mod_2pi(theta_rec, theta), f"expected theta={theta}, got {theta_rec}"


def test_roundtrip_known_values() -> None:
    """Hand-built: rotate the x-axis by pi/4 around z."""
    k = np.array([0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    theta = np.pi / 4
    q = rotate(k, theta, p)
    theta_rec, is_ls = sp1.solve(k, p, q)
    assert not is_ls
    assert abs(theta_rec - theta) < 1e-12


def test_ls_magnitude_mismatch() -> None:
    """|p| != |q| -- no exact solution, should return LS flag."""
    k = np.array([0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([2.0, 0.0, 0.0])  # correct direction, wrong magnitude
    theta, is_ls = sp1.solve(k, p, q)
    assert is_ls
    assert abs(theta) < 1e-9


def test_ls_axial_mismatch() -> None:
    """k.p != k.q -- no exact solution (rotation preserves axial component)."""
    k = np.array([0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 1.0, 1.0])
    _, is_ls = sp1.solve(k, p, q)
    assert is_ls


def test_p_along_k_any_theta_works_ls_chooses_zero() -> None:
    """If p is parallel to k, any theta rotates p to itself. In the exact
    case (q = p), sp1 may return any theta and is_ls=False; in the infeasible
    case (q != p), returns LS."""
    k = np.array([0.0, 0.0, 1.0])
    p = np.array([0.0, 0.0, 3.0])
    # Exact case: q = p. The atan2 will take atan2(0, 0) which is 0.
    theta, is_ls = sp1.solve(k, p, p)
    assert not is_ls
    assert abs(theta) < 1e-12


def test_symmetric_anti_parallel() -> None:
    """q = -p with p perpendicular to k should give theta = pi."""
    k = np.array([0.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([-1.0, 0.0, 0.0])
    theta, is_ls = sp1.solve(k, p, q)
    assert not is_ls
    assert abs(abs(theta) - np.pi) < 1e-12


@given(_axis_theta_p())
@settings(max_examples=100, deadline=None)
def test_roundtrip_ls_when_noise_added(case: tuple[np.ndarray, float, np.ndarray]) -> None:
    """Adding a small perpendicular-to-rotation-plane perturbation should
    flip is_ls to True while keeping theta close to the exact answer."""
    k, theta, p = case
    q = rotate(k, theta, p)
    q_noisy = q + 1e-4 * k  # small axial noise so k.q shifts but |q| shifts too
    # The feasibility tolerance in sp1 is 1e-9, so 1e-4 noise makes is_ls true.
    theta_rec, is_ls = sp1.solve(k, p, q_noisy)
    assert is_ls
    assert _near_equal_mod_2pi(theta_rec, theta, tol=1e-2)


@pytest.mark.parametrize(
    ("p", "q"),
    [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.5, 0.3, 0.2]), np.array([-0.1, 0.5, 0.2])),
    ],
)
def test_parametric_roundtrip_z_axis(p: np.ndarray, q: np.ndarray) -> None:
    """When both p and q are at the same radius and axial position, there's
    an exact theta; sp1 should find it."""
    k = np.array([0.0, 0.0, 1.0])
    # normalize to same axial component and same perpendicular magnitude
    q_fixed = q.copy()
    q_fixed[2] = p[2]
    p_perp = float(np.linalg.norm(p[:2]))
    q_perp = float(np.linalg.norm(q_fixed[:2]))
    q_fixed[:2] = q_fixed[:2] * (p_perp / q_perp if q_perp > 0 else 1)
    theta, is_ls = sp1.solve(k, p, q_fixed)
    assert not is_ls
    # Sanity: rotating p by the recovered theta should reproduce q_fixed.
    q_check = rotate(k, theta, p)
    assert np.allclose(q_check, q_fixed, atol=1e-10)


@pytest.mark.parametrize("delta", [1e-3, 1e-6, 1e-8])
def test_angle_error_scales_as_eps_over_perp_near_the_axis(delta: float) -> None:
    """Near a singularity p and q lie within ``delta`` of a general (not
    axis-aligned) axis. The SP1 angle then carries an error of order eps / delta,
    set by the rounding of q itself. The textbook terms p.q - (k.p)(k.q) and
    (k x p).q cancel O(1) axial parts and cost eps / delta^2 instead, which lost
    wrist branches (#661). Checked through the scalar rule and the batched one."""
    rng = np.random.default_rng(661)
    eps = np.finfo(np.float64).eps
    ks, ps, qs, thetas = [], [], [], []
    for _ in range(200):
        k = _unit(rng.normal(size=3))
        u = _unit(np.cross(k, rng.normal(size=3)))
        w = np.cross(k, u)
        axial = rng.uniform(0.5, 2.0)
        theta = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)
        ks.append(k)
        ps.append(axial * k + delta * u)
        qs.append(axial * k + delta * (np.cos(theta) * u + np.sin(theta) * w))
        thetas.append(theta)
    k_, p_, q_ = np.array(ks), np.array(ps), np.array(qs)
    bound = 64 * eps / delta
    rows = sp1.angle_rows(k_, p_, q_)
    for i, theta in enumerate(thetas):
        scalar, _ = sp1.solve(k_[i], p_[i], q_[i])
        for got in (scalar, float(rows[i])):
            err = abs(((got - theta + np.pi) % (2 * np.pi)) - np.pi)
            assert err <= bound, f"delta={delta}: error {err:.1e} > {bound:.1e}"
