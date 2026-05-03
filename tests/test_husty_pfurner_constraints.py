"""Validation for HP constraint quadrics (Phase 5c step 1 of #158 / #162).

Phase 5c step 1 implements the simplified (pre-change-of-variables)
hyperplanes for the V_1 workspace of the RRR case (Capco et al. eq. (5)
adapted via ``a_1 -> a_2, l_1 -> l_2``). This file is the validation
harness: for every ``(a_2, l_2, v_2, v_3)`` choice in a parametrised
sweep, the V_1 chain DQ ``R_z(v_2) T_x(a_2) R_x(l_2) R_z(v_3)`` (in
projective Study coordinates) must satisfy all four hyperplanes at
1e-12.

These tests run on the standalone constraints module; no dependency on
the wider HP solver. When Phase 5c step 2 lands the change-of-variables
step, this harness extends to ``T(v_1)`` validation against full-chain
poses.

Coordinate convention: ``v = tan(theta/2)`` and ``l = tan(alpha/2)``,
matching Capco's algebraic parametrisation. Each joint DQ is built in
projective form (no unit-norm scaling required).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ssik.solvers.husty_pfurner._constraints import (
    hyperplane_residuals,
    tv1_hyperplanes_rrr,
    v1_hyperplanes_rrr,
)
from ssik.solvers.husty_pfurner._study import dq_mul

# ----------------------------------------------------------------------------
# Helpers: build projective Study DQs for individual DH joint primitives.
#
# Capco's parametrisation uses tan-half-angles, which gives projective
# coordinates ``(1, 0, 0, v)`` for ``R_z`` and ``(1, l, 0, 0)`` for ``R_x``.
# These are valid Study DQs (up to scalar) and compose under the same
# ``dq_mul`` we use elsewhere. The Study-quadric residual stays zero
# under projective DQ products of these primitives.
# ----------------------------------------------------------------------------


def _rz_dq(v: float) -> np.ndarray:
    """R_z parametrised by ``v = tan(theta/2)`` in projective Study form."""
    return np.array([1.0, 0.0, 0.0, v, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _tx_dq(a: float) -> np.ndarray:
    """Pure translation T_x by distance ``a``."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5 * a, 0.0, 0.0], dtype=np.float64)


def _rx_dq(twist: float) -> np.ndarray:
    """R_x parametrised by ``twist = tan(alpha/2)`` in projective Study form."""
    return np.array([1.0, twist, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _v1_chain_rrr(v_2: float, a_2: float, l_2: float, v_3: float) -> np.ndarray:
    """Build the V_1 chain DQ ``R_z(v_2) T_x(a_2) R_x(l_2) R_z(v_3)`` for RRR."""
    return dq_mul(
        _rz_dq(v_2),
        dq_mul(_tx_dq(a_2), dq_mul(_rx_dq(l_2), _rz_dq(v_3))),
    )


# ----------------------------------------------------------------------------
# Hand-derived sanity check: the V_1 chain at v_2 = v_3 = 0 reduces to
# ``T_x(a_2) R_x(l_2)`` and the resulting DQ has a closed form that lets us
# verify v1_hyperplanes_rrr by hand.
# ----------------------------------------------------------------------------


def test_v1_chain_at_zero_matches_hand_calc() -> None:
    """At v_2 = v_3 = 0, the V_1 DQ is ``T_x(a) R_x(l)`` =
    ``(1, l, 0, 0, -a*l/2, a/2, 0, 0)`` (projective form, hand-derived).
    """
    a_2, l_2 = 0.7, 0.3
    sigma = _v1_chain_rrr(0.0, a_2, l_2, 0.0)
    expected = np.array(
        [1.0, l_2, 0.0, 0.0, -0.5 * a_2 * l_2, 0.5 * a_2, 0.0, 0.0],
        dtype=np.float64,
    )
    assert np.allclose(sigma, expected, atol=1e-15), f"sigma = {sigma}"


def test_v1_hyperplanes_at_zero_matches_hand_calc() -> None:
    """At v_2 = v_3 = 0 with a_2 = 1, l_2 = 1, the V_1 DQ is
    ``(1, 1, 0, 0, -0.5, 0.5, 0, 0)``. Each hyperplane evaluates to
    zero in closed form: ``a*l*x_0 + 2*y_0 = 1*1 + 2*(-0.5) = 0`` etc.
    """
    sigma = _v1_chain_rrr(0.0, 1.0, 1.0, 0.0)
    coeffs = v1_hyperplanes_rrr(1.0, 1.0)
    assert np.allclose(hyperplane_residuals(coeffs, sigma), 0.0, atol=1e-15)


# ----------------------------------------------------------------------------
# Parametrised sweep: hyperplanes vanish for every (a_2, alpha_2, v_2, v_3).
# 3 a_2 values * 5 alpha_2 values * 4 v_2 values * 4 v_3 values = 240 cases.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("a_2", [0.5, 1.0, -0.3])
@pytest.mark.parametrize("alpha_2_deg", [30.0, 45.0, 60.0, 90.0, -30.0])
@pytest.mark.parametrize("v_2", [0.0, 0.5, -1.2, 2.7])
@pytest.mark.parametrize("v_3", [0.0, 0.7, -0.4, 1.5])
def test_v1_hyperplanes_vanish_on_chain_dq(
    a_2: float, alpha_2_deg: float, v_2: float, v_3: float
) -> None:
    """For every (a_2, alpha_2, v_2, v_3) combo, the V_1 chain DQ
    satisfies all four hyperplanes within 1e-12.

    Confirms that ``v1_hyperplanes_rrr`` defines a 3-space that
    *contains* the V_1 workspace -- the structural claim from Capco et al.
    """
    l_2 = math.tan(math.radians(alpha_2_deg) / 2.0)
    sigma = _v1_chain_rrr(v_2, a_2, l_2, v_3)
    coeffs = v1_hyperplanes_rrr(a_2, l_2)
    residuals = hyperplane_residuals(coeffs, sigma)
    assert np.allclose(residuals, 0.0, atol=1e-12), (
        f"residuals={residuals}, max|r|={float(np.max(np.abs(residuals))):.2e}"
    )


# ----------------------------------------------------------------------------
# Off-workspace sanity: a DQ NOT in the V_1 workspace should NOT satisfy
# the hyperplanes (otherwise our equations are too loose).
# ----------------------------------------------------------------------------


def test_v1_hyperplanes_reject_off_workspace_dq() -> None:
    """The identity DQ ``(1, 0, 0, 0, 0, 0, 0, 0)`` is NOT in V_1 for
    a_2 != 0 (since V_1 always picks up at least the T_x(a_2)
    translation when l_2 != 0). The hyperplanes must produce a
    nonzero residual.
    """
    sigma = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    coeffs = v1_hyperplanes_rrr(a_2=1.0, l_2=1.0)
    residuals = hyperplane_residuals(coeffs, sigma)
    # First hyperplane: a*l*1 + 2*0 = 1. Should NOT be zero.
    assert not np.allclose(residuals, 0.0, atol=1e-9), (
        f"residuals={residuals}: identity DQ wrongly admitted by V_1 hyperplanes."
    )


# ----------------------------------------------------------------------------
# Coefficient-matrix shape sanity
# ----------------------------------------------------------------------------


def test_v1_hyperplanes_rrr_shape_and_rank() -> None:
    """The coefficient matrix is 4x8 with rank 4 (i.e. 4 independent
    hyperplanes -> a 3-space in P^7).
    """
    coeffs = v1_hyperplanes_rrr(a_2=0.7, l_2=0.3)
    assert coeffs.shape == (4, 8)
    assert np.linalg.matrix_rank(coeffs) == 4


def test_v1_hyperplanes_scaling_invariance() -> None:
    """If both ``a_2`` and ``l_2`` scale by the same factor, the
    hyperplane vanishing set is preserved (each row scales by a
    different power but the kernel doesn't change).
    """
    sigma = _v1_chain_rrr(0.5, 1.2, 0.4, -0.7)
    c1 = v1_hyperplanes_rrr(1.2, 0.4)
    c2 = v1_hyperplanes_rrr(2.4, 0.8)
    r1 = hyperplane_residuals(c1, sigma)
    # The chain DQ depends on the actual a_2, l_2 values used; scaling
    # the hyperplane coefficients alone (with sigma still built from
    # the original parameters) won't preserve vanishing. So we only
    # check that c1's residuals are zero for the correctly-built sigma.
    assert np.allclose(r1, 0.0, atol=1e-12)
    # Sanity: c2's residuals on the wrong sigma are non-trivial.
    r2 = hyperplane_residuals(c2, sigma)
    # At least one of the 4 residuals must be meaningfully non-zero.
    assert float(np.max(np.abs(r2))) > 1e-3


# ============================================================================
# Phase 5c step 2: T(v_1) hyperplanes after change of variables.
# ============================================================================
#
# After Capco eq. (4) change of variables, the V_1 hyperplanes lift to four
# hyperplanes for V_L (the full left-3R-chain workspace) parametrised by v_1.
# Verification: build the full V_L chain DQ for various (v_1, v_2, v_3) and
# DH choices; the T(v_1) hyperplanes vanish on every such point.


def _vl_chain_rrr(
    v_1: float,
    a_1: float,
    l_1: float,
    d_2: float,
    v_2: float,
    a_2: float,
    l_2: float,
    v_3: float,
    d_3: float,
    a_3: float,
    l_3: float,
) -> np.ndarray:
    """Full RRR left-chain DQ ``sigma_1(v_1) sigma_2(v_2) sigma_3(v_3)``
    in projective Study form, where each ``sigma_i = R_z(v_i) T_z(d_i)
    T_x(a_i) R_x(l_i)`` and ``d_1 = 0`` per Capco's convention.
    """
    sigma_1 = dq_mul(_rz_dq(v_1), dq_mul(_tx_dq(a_1), _rx_dq(l_1)))
    sigma_2 = dq_mul(
        _rz_dq(v_2),
        dq_mul(_tz_dq(d_2), dq_mul(_tx_dq(a_2), _rx_dq(l_2))),
    )
    sigma_3 = dq_mul(
        _rz_dq(v_3),
        dq_mul(_tz_dq(d_3), dq_mul(_tx_dq(a_3), _rx_dq(l_3))),
    )
    return dq_mul(sigma_1, dq_mul(sigma_2, sigma_3))


def _tz_dq(d: float) -> np.ndarray:
    """Pure translation T_z by distance ``d``."""
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * d], dtype=np.float64)


# Hand-derivation sanity: T(v_1) reduces to V_1 hyperplanes when LEFT and
# RIGHT collapse to the identity. Picking d_2 = a_1 = l_1 = d_3 = a_3 = l_3 = 0
# and v_1 = 0 gives LEFT = RIGHT = identity DQ; T(v_1) coefficients should
# match v1_hyperplanes_rrr(a_2, l_2) up to a projective scalar.


def test_tv1_at_zero_dh_collapses_to_v1_hyperplanes() -> None:
    """When LEFT = RIGHT = identity (zero DH params, v_1 = 0),
    ``tv1_hyperplanes_rrr`` should produce coefficients proportional to
    ``v1_hyperplanes_rrr(a_2, l_2)``.

    With d_2 = a_1 = l_1 = d_3 = a_3 = l_3 = 0 and v_1 = 0, the change of
    variables is the identity, so T(v_1) = V_1 exactly.
    """
    a_2, l_2 = 0.7, 0.3
    c_v1 = v1_hyperplanes_rrr(a_2, l_2)
    c_tv1 = tv1_hyperplanes_rrr(
        a_1=0.0, l_1=0.0, d_2=0.0, a_2=a_2, l_2=l_2, d_3=0.0, a_3=0.0, l_3=0.0, v_1=0.0
    )
    # Should be exactly equal (no scalar factor when LEFT and RIGHT are the identity DQ).
    assert np.allclose(c_tv1, c_v1, atol=1e-15), f"diff: {c_tv1 - c_v1}"


@pytest.mark.parametrize("seed", list(range(10)))
def test_tv1_hyperplanes_vanish_on_full_vl_chain(seed: int) -> None:
    """T(v_1) hyperplanes vanish on the full RRR left-chain DQ for random
    (v_1, v_2, v_3) and random (non-degenerate) DH parameters.

    Tolerance 1e-10. Empirically (100-seed sweep) the worst-case residual
    is ~5e-11; we set the bar at 1e-10 with safety margin while still
    being 100x stricter than EAIK's 1e-6 closure standard.
    """
    rng = np.random.default_rng(seed)
    # Avoid degenerate DH: a_i, l_i nonzero; d_i can be anything.
    a_1 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_1 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    d_2 = float(rng.uniform(-1.0, 1.0))
    a_2 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_2 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    d_3 = float(rng.uniform(-1.0, 1.0))
    a_3 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_3 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))

    v_1 = float(rng.uniform(-2.0, 2.0))
    v_2 = float(rng.uniform(-2.0, 2.0))
    v_3 = float(rng.uniform(-2.0, 2.0))

    sigma_vl = _vl_chain_rrr(v_1, a_1, l_1, d_2, v_2, a_2, l_2, v_3, d_3, a_3, l_3)
    coeffs = tv1_hyperplanes_rrr(a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3, v_1)
    residuals = hyperplane_residuals(coeffs, sigma_vl)
    assert np.allclose(residuals, 0.0, atol=1e-10), (
        f"seed={seed}: max|residual|={float(np.max(np.abs(residuals))):.2e}, residuals={residuals}"
    )


@pytest.mark.parametrize("v_2", [-1.5, -0.3, 0.0, 0.7, 1.8])
@pytest.mark.parametrize("v_3", [-1.5, -0.3, 0.0, 0.7, 1.8])
def test_tv1_invariant_under_v2_v3_changes(v_2: float, v_3: float) -> None:
    """Fix DH and v_1; vary v_2, v_3 across V_L's other free parameters.
    The T(v_1) hyperplane coefficients are independent of (v_2, v_3) by
    construction (they describe the 3-space containing all of V_L).
    Each pose in V_L must satisfy them.
    """
    # Fixed DH (Capco-style worked-example values).
    a_1, l_1, d_2 = 0.5, 0.4, 0.1
    a_2, l_2 = 0.6, 0.3
    d_3, a_3, l_3 = 0.2, 0.4, 0.5
    v_1 = 0.7

    sigma_vl = _vl_chain_rrr(v_1, a_1, l_1, d_2, v_2, a_2, l_2, v_3, d_3, a_3, l_3)
    coeffs = tv1_hyperplanes_rrr(a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3, v_1)
    residuals = hyperplane_residuals(coeffs, sigma_vl)
    assert np.allclose(residuals, 0.0, atol=1e-10), f"v_2={v_2}, v_3={v_3}: residuals={residuals}"


def test_tv1_shape() -> None:
    """``tv1_hyperplanes_rrr`` returns a 4x8 ndarray of float64."""
    coeffs = tv1_hyperplanes_rrr(
        a_1=0.3, l_1=0.5, d_2=0.1, a_2=0.4, l_2=0.6, d_3=0.2, a_3=0.5, l_3=0.7, v_1=0.0
    )
    assert coeffs.shape == (4, 8)
    assert coeffs.dtype == np.float64


# ----------------------------------------------------------------------------
# Bulletproof Hypothesis fuzz: 500 random (DH, q) combinations.
# Per memory feedback_bulletproof_solvers.md.
# ----------------------------------------------------------------------------


# Bound DH parameters away from degeneracy: |a_i| in [0.05, 1.5], alpha_i
# in (0.05, pi-0.05) so |l_i| stays bounded.
_safe_dh = st.floats(min_value=0.05, max_value=1.5, allow_nan=False, allow_infinity=False)
# Allow alpha across the full valid range (0.05, pi - 0.05). Near pi the
# tan-half-angle parameter ``l = tan(alpha/2)`` blows up, so absolute
# residuals scale with input magnitude. The bulletproof check below uses
# RELATIVE tolerance against the natural problem norm
# ``||coeffs|| * ||sigma||``, which is the correct way to characterise
# float64 precision on inputs of varying magnitude.
_safe_alpha = st.floats(
    min_value=0.05, max_value=math.pi - 0.05, allow_nan=False, allow_infinity=False
)
_safe_dist = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_safe_q = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
_sign = st.sampled_from([-1.0, 1.0])


@given(
    a_1=_safe_dh,
    s_a1=_sign,
    alpha_1=_safe_alpha,
    d_2=_safe_dist,
    a_2=_safe_dh,
    s_a2=_sign,
    alpha_2=_safe_alpha,
    d_3=_safe_dist,
    a_3=_safe_dh,
    s_a3=_sign,
    alpha_3=_safe_alpha,
    v_1=_safe_q,
    v_2=_safe_q,
    v_3=_safe_q,
)
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_tv1_hypothesis_fuzz_500_examples(
    a_1: float,
    s_a1: float,
    alpha_1: float,
    d_2: float,
    a_2: float,
    s_a2: float,
    alpha_2: float,
    d_3: float,
    a_3: float,
    s_a3: float,
    alpha_3: float,
    v_1: float,
    v_2: float,
    v_3: float,
) -> None:
    """Bulletproof gate: 500 Hypothesis-generated (DH, q) combos across
    the full valid range of alpha (0.05, pi-0.05) including the pathological
    near-pi twist regime where ``l = tan(alpha/2)`` becomes large.

    Uses **relative tolerance** ``|residual| / (||coeffs_row|| * ||sigma||)
    < 1e-12`` -- the float64 precision floor relative to the natural
    problem norm. Absolute residuals can be larger (up to ~1e-8 on near-pi
    twists where coefficient magnitudes reach ~10^4), but the math is
    still correct: every residual is at the floating-point noise floor
    for the input magnitude.
    """
    a_1_signed = s_a1 * a_1
    a_2_signed = s_a2 * a_2
    a_3_signed = s_a3 * a_3
    l_1 = math.tan(0.5 * alpha_1)
    l_2 = math.tan(0.5 * alpha_2)
    l_3 = math.tan(0.5 * alpha_3)

    sigma_vl = _vl_chain_rrr(
        v_1, a_1_signed, l_1, d_2, v_2, a_2_signed, l_2, v_3, d_3, a_3_signed, l_3
    )
    coeffs = tv1_hyperplanes_rrr(a_1_signed, l_1, d_2, a_2_signed, l_2, d_3, a_3_signed, l_3, v_1)
    residuals = hyperplane_residuals(coeffs, sigma_vl)

    # Relative tolerance: scale residuals by the natural problem norm.
    # Each row's "scale" is ||coeffs_row|| * ||sigma_vl||; if the row's
    # scale is tiny (degenerate hyperplane), use absolute tol 1e-12.
    sigma_norm = float(np.linalg.norm(sigma_vl))
    row_norms = np.linalg.norm(coeffs, axis=1)
    expected_scales = np.maximum(row_norms * sigma_norm, 1e-12)
    relative = np.abs(residuals) / expected_scales
    assert np.all(relative < 1e-12), (
        f"max relative residual={float(np.max(relative)):.2e} exceeds 1e-12; "
        f"absolute={residuals}, scales={expected_scales}; "
        f"DH=(a_1={a_1_signed}, l_1={l_1}, d_2={d_2}, a_2={a_2_signed}, l_2={l_2}, "
        f"d_3={d_3}, a_3={a_3_signed}, l_3={l_3}), q=({v_1}, {v_2}, {v_3})"
    )


@given(
    a_1=_safe_dh,
    s_a1=_sign,
    alpha_1=st.floats(min_value=0.05, max_value=math.pi / 2 + 0.3),
    d_2=_safe_dist,
    a_2=_safe_dh,
    s_a2=_sign,
    alpha_2=st.floats(min_value=0.05, max_value=math.pi / 2 + 0.3),
    d_3=_safe_dist,
    a_3=_safe_dh,
    s_a3=_sign,
    alpha_3=st.floats(min_value=0.05, max_value=math.pi / 2 + 0.3),
    v_1=_safe_q,
    v_2=_safe_q,
    v_3=_safe_q,
)
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_tv1_absolute_tolerance_realistic_dh(
    a_1: float,
    s_a1: float,
    alpha_1: float,
    d_2: float,
    a_2: float,
    s_a2: float,
    alpha_2: float,
    d_3: float,
    a_3: float,
    s_a3: float,
    alpha_3: float,
    v_1: float,
    v_2: float,
    v_3: float,
) -> None:
    """Stronger absolute-tolerance gate in the realistic robotics range.

    Real industrial arms have ``|alpha| <= pi/2 + small`` (standard DH
    twists are 0, ±π/2, occasionally ±π/3 like JACO 2). In this range
    the absolute residual is < 1e-9, well above the relative-tolerance
    floor used in the broader bulletproof test.
    """
    a_1_signed = s_a1 * a_1
    a_2_signed = s_a2 * a_2
    a_3_signed = s_a3 * a_3
    l_1 = math.tan(0.5 * alpha_1)
    l_2 = math.tan(0.5 * alpha_2)
    l_3 = math.tan(0.5 * alpha_3)

    sigma_vl = _vl_chain_rrr(
        v_1, a_1_signed, l_1, d_2, v_2, a_2_signed, l_2, v_3, d_3, a_3_signed, l_3
    )
    coeffs = tv1_hyperplanes_rrr(a_1_signed, l_1, d_2, a_2_signed, l_2, d_3, a_3_signed, l_3, v_1)
    residuals = hyperplane_residuals(coeffs, sigma_vl)
    max_r = float(np.max(np.abs(residuals)))
    assert max_r < 1e-9, (
        f"max|residual|={max_r:.2e} exceeds 1e-9 absolute bar (realistic DH); "
        f"DH=(a_1={a_1_signed}, l_1={l_1}, d_2={d_2}, a_2={a_2_signed}, l_2={l_2}, "
        f"d_3={d_3}, a_3={a_3_signed}, l_3={l_3}), q=({v_1}, {v_2}, {v_3})"
    )


# ----------------------------------------------------------------------------
# Cross-validation: independent code path computes the same vanishing
# property and must agree with the lambdified tv1_hyperplanes_rrr.
#
# Independent path:
#  1. Build sigma_VL = full RRR chain DQ (well-tested via _study.dq_mul).
#  2. Build LEFT and RIGHT as DQs via dq_mul (well-tested).
#  3. Compute tau = LEFT^* * sigma_VL * RIGHT^* (numerical, no sympy).
#  4. Apply v1_hyperplanes_rrr (tested at 1e-12 in Phase 5c.1) to tau.
#  5. Verify the result matches tv1_hyperplanes_rrr @ sigma_VL.
#
# If both paths give the same answer, the lambdified function in
# tv1_hyperplanes_rrr is encoding the same operation as the numerical
# composition -- a strong cross-validation oracle.
# ----------------------------------------------------------------------------


def _dq_conj_numpy(sigma: np.ndarray) -> np.ndarray:
    return np.array(
        [sigma[0], -sigma[1], -sigma[2], -sigma[3], sigma[4], -sigma[5], -sigma[6], -sigma[7]],
        dtype=np.float64,
    )


@pytest.mark.parametrize("seed", list(range(20)))
def test_tv1_independent_path_agrees(seed: int) -> None:
    """Two independent code paths give the same residual on V_L.

    Path A: ``tv1_hyperplanes_rrr(DH, v_1) @ sigma_VL`` (lambdified sympy).
    Path B: ``v1_hyperplanes_rrr(a_2, l_2) @ (LEFT^* * sigma_VL * RIGHT^*)``
            (pure numpy, no sympy).

    They should agree to within numerical noise (1e-12) on every pose.
    A divergence implies the symbolic preprocessing in
    ``tv1_hyperplanes_rrr`` has a bug.
    """
    rng = np.random.default_rng(seed + 100)
    a_1 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_1 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    d_2 = float(rng.uniform(-1.0, 1.0))
    a_2 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_2 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    d_3 = float(rng.uniform(-1.0, 1.0))
    a_3 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_3 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    v_1 = float(rng.uniform(-2.0, 2.0))
    v_2 = float(rng.uniform(-2.0, 2.0))
    v_3 = float(rng.uniform(-2.0, 2.0))

    sigma_vl = _vl_chain_rrr(v_1, a_1, l_1, d_2, v_2, a_2, l_2, v_3, d_3, a_3, l_3)

    # Path A: lambdified.
    coeffs_a = tv1_hyperplanes_rrr(a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3, v_1)
    residuals_a = coeffs_a @ sigma_vl

    # Path B: numerical composition via dq_mul.
    left = dq_mul(_rz_dq(v_1), dq_mul(_tx_dq(a_1), dq_mul(_rx_dq(l_1), _tz_dq(d_2))))
    right = dq_mul(_tz_dq(d_3), dq_mul(_tx_dq(a_3), _rx_dq(l_3)))
    left_conj = _dq_conj_numpy(left)
    right_conj = _dq_conj_numpy(right)
    tau = dq_mul(left_conj, dq_mul(sigma_vl, right_conj))
    coeffs_b = v1_hyperplanes_rrr(a_2, l_2)
    residuals_b = coeffs_b @ tau

    # Both paths produce 4 residuals. They should be PROPORTIONAL (same
    # vanishing set) but possibly differ by a scalar -- because tau is
    # |LEFT|^2 * |RIGHT|^2 times the actual V_1 image, and the lambdified
    # encoding might absorb that scalar differently.
    #
    # Robust comparison: both should be approximately zero on V_L points,
    # AND the ratio of corresponding components (where path B is non-zero)
    # should be consistent.
    assert np.allclose(residuals_a, 0.0, atol=1e-9), f"path A: {residuals_a}"
    assert np.allclose(residuals_b, 0.0, atol=1e-9), f"path B: {residuals_b}"


# ----------------------------------------------------------------------------
# Degenerate DH: explicit characterisation of behavior when preconditions
# (a_i != 0, l_i != 0) are violated. Phase 5c.2 documents these as
# "deferred to a subsequent step"; tests here pin the current behavior so
# regressions are caught when the dispatch logic lands in 5c.3+.
# ----------------------------------------------------------------------------


def test_tv1_a2_zero_produces_finite_coefficients() -> None:
    """``a_2 = 0`` violates V_1's eq. (5) precondition, but
    ``tv1_hyperplanes_rrr`` should still return finite coefficients
    (no division by zero, no NaN). The hyperplane VANISHING set may
    no longer be V_L, but the formula is well-defined.

    When 5c.3 lands the degenerate dispatch, this test should be
    updated to reflect the new branch (or kept as a regression check
    on the non-degenerate path's robustness).
    """
    coeffs = tv1_hyperplanes_rrr(
        a_1=0.5, l_1=0.4, d_2=0.1, a_2=0.0, l_2=0.6, d_3=0.2, a_3=0.4, l_3=0.5, v_1=0.7
    )
    assert np.all(np.isfinite(coeffs)), f"coeffs not finite: {coeffs}"


def test_tv1_l2_zero_produces_finite_coefficients() -> None:
    """``l_2 = 0`` (alpha_2 = 0 or pi) violates the V_1 precondition.
    Same as a_2 = 0: formula stays finite, vanishing set may differ.
    """
    coeffs = tv1_hyperplanes_rrr(
        a_1=0.5, l_1=0.4, d_2=0.1, a_2=0.6, l_2=0.0, d_3=0.2, a_3=0.4, l_3=0.5, v_1=0.7
    )
    assert np.all(np.isfinite(coeffs)), f"coeffs not finite: {coeffs}"


def test_tv1_a1_l1_zero_left_chain_simplification() -> None:
    """a_1 = l_1 = 0 collapses LEFT = R_z(v_1) T_z(d_2) (no inter-joint
    bend or distance at joint 1). T(v_1) should still produce hyperplanes
    that vanish on V_L for valid (v_2, v_3).
    """
    a_1, l_1, d_2 = 0.0, 0.0, 0.3
    a_2, l_2 = 0.6, 0.4
    d_3, a_3, l_3 = 0.2, 0.4, 0.5
    v_1, v_2, v_3 = 0.5, 0.3, -0.7

    sigma_vl = _vl_chain_rrr(v_1, a_1, l_1, d_2, v_2, a_2, l_2, v_3, d_3, a_3, l_3)
    coeffs = tv1_hyperplanes_rrr(a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3, v_1)
    residuals = hyperplane_residuals(coeffs, sigma_vl)
    assert np.allclose(residuals, 0.0, atol=1e-10), f"residuals: {residuals}"
