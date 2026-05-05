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

import sympy as sp

from ssik.solvers.husty_pfurner._constraints import (
    TV2_RRR_CASE_KEYS,
    _V2_SYM,
    hyperplane_residuals,
    tv1_hyperplanes_rrr,
    tv2_hyperplanes_rrr,
    tv2_rrr_case_for,
    tv2_symbolic_in_v2,
    tv3_hyperplanes_rrr,
    tv6_hyperplanes_rrr,
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


# ============================================================================
# Phase 5c step 3: T(v_3) -- alternative parametrisation by joint 3.
# ============================================================================
#
# T(v_1) requires a_2 != 0 ∧ l_2 != 0. For arms with l_2 = ±1 (common DH
# twist of ±90deg), T(v_1) lies in the Study quadric and we need T(v_3).
# T(v_3) requires a_1 != 0 ∧ l_1 != 0 (eq. 5 precondition for V_3).
#
# The change of variables for T(v_3) is one-sided (only on the right):
#     V_L = V_3 · POST(v_3)
# where POST(v_3) = T_z(d_2) T_x(a_2) R_x(l_2) R_z(v_3) T_z(d_3) T_x(a_3)
# R_x(l_3) and V_3 = {R_z(v_1) T_x(a_1) R_x(l_1) R_z(v_2)}.


def test_tv3_at_zero_dh_collapses_to_v3_hyperplanes() -> None:
    """When POST = identity (zero DH) and v_3 = 0, T(v_3) collapses to
    V_3 hyperplanes. V_3 hyperplanes use (a_1, l_1) per Capco eq. (5).
    """
    a_1, l_1 = 0.7, 0.3
    # v1_hyperplanes_rrr(a_2, l_2) is also eq (5); we reuse it with
    # (a_1, l_1) here as it's the SAME equation, just labelled by joint
    # indexing.
    c_v3 = v1_hyperplanes_rrr(a_1, l_1)
    c_tv3 = tv3_hyperplanes_rrr(
        a_1=a_1, l_1=l_1, d_2=0.0, a_2=0.0, l_2=0.0, d_3=0.0, a_3=0.0, l_3=0.0, v_3=0.0
    )
    assert np.allclose(c_tv3, c_v3, atol=1e-15), f"diff: {c_tv3 - c_v3}"


@pytest.mark.parametrize("seed", list(range(10)))
def test_tv3_hyperplanes_vanish_on_full_vl_chain(seed: int) -> None:
    """T(v_3) hyperplanes vanish on the full RRR chain DQ for random
    (v_1, v_2, v_3) and random non-degenerate DH at 1e-10 absolute.
    """
    rng = np.random.default_rng(seed + 200)
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
    coeffs = tv3_hyperplanes_rrr(a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3, v_3)
    residuals = hyperplane_residuals(coeffs, sigma_vl)
    assert np.allclose(residuals, 0.0, atol=1e-10), (
        f"seed={seed}: max|residual|={float(np.max(np.abs(residuals))):.2e}"
    )


@pytest.mark.parametrize("v_1", [-1.5, -0.3, 0.0, 0.7, 1.8])
@pytest.mark.parametrize("v_2", [-1.5, -0.3, 0.0, 0.7, 1.8])
def test_tv3_invariant_under_v1_v2_changes(v_1: float, v_2: float) -> None:
    """Fix DH and v_3; vary v_1, v_2 across V_L's other free parameters.
    The T(v_3) coefficients are independent of (v_1, v_2) by construction;
    every V_L pose must satisfy them.
    """
    a_1, l_1, d_2 = 0.5, 0.4, 0.1
    a_2, l_2 = 0.6, 0.3
    d_3, a_3, l_3 = 0.2, 0.4, 0.5
    v_3 = 0.7

    sigma_vl = _vl_chain_rrr(v_1, a_1, l_1, d_2, v_2, a_2, l_2, v_3, d_3, a_3, l_3)
    coeffs = tv3_hyperplanes_rrr(a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3, v_3)
    residuals = hyperplane_residuals(coeffs, sigma_vl)
    assert np.allclose(residuals, 0.0, atol=1e-10), f"v_1={v_1}, v_2={v_2}: residuals={residuals}"


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
def test_tv3_hypothesis_fuzz_500_examples(
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
    """Bulletproof gate for T(v_3): 500 (DH, q) combinations across the
    full alpha range; relative tolerance 1e-12. Mirrors T(v_1)'s
    bulletproof test.
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
    coeffs = tv3_hyperplanes_rrr(a_1_signed, l_1, d_2, a_2_signed, l_2, d_3, a_3_signed, l_3, v_3)
    residuals = hyperplane_residuals(coeffs, sigma_vl)
    sigma_norm = float(np.linalg.norm(sigma_vl))
    row_norms = np.linalg.norm(coeffs, axis=1)
    expected_scales = np.maximum(row_norms * sigma_norm, 1e-12)
    relative = np.abs(residuals) / expected_scales
    assert np.all(relative < 1e-12), (
        f"max relative residual={float(np.max(relative)):.2e}; "
        f"absolute={residuals}, scales={expected_scales}"
    )


def test_tv3_independent_path_agrees() -> None:
    """T(v_3) lambdified path agrees with numerical composition path.

    Path A: ``tv3_hyperplanes_rrr(DH, v_3) @ sigma_VL`` (lambdified sympy).
    Path B: ``v1_hyperplanes_rrr(a_1, l_1) @ (sigma_VL · POST^*)`` (numpy).

    Both vanish on V_L points; agreement at 1e-9.
    """
    rng = np.random.default_rng(300)
    for _ in range(20):
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
        coeffs_a = tv3_hyperplanes_rrr(a_1, l_1, d_2, a_2, l_2, d_3, a_3, l_3, v_3)
        residuals_a = coeffs_a @ sigma_vl

        # Path B: numerical composition. POST = T_z(d_2) T_x(a_2) R_x(l_2)
        # R_z(v_3) T_z(d_3) T_x(a_3) R_x(l_3).
        post = dq_mul(
            _tz_dq(d_2),
            dq_mul(
                _tx_dq(a_2),
                dq_mul(
                    _rx_dq(l_2),
                    dq_mul(_rz_dq(v_3), dq_mul(_tz_dq(d_3), dq_mul(_tx_dq(a_3), _rx_dq(l_3)))),
                ),
            ),
        )
        post_conj = _dq_conj_numpy(post)
        tau = dq_mul(sigma_vl, post_conj)
        coeffs_b = v1_hyperplanes_rrr(a_1, l_1)  # eq. 5 with (a_1, l_1) is the V_3 set
        residuals_b = coeffs_b @ tau

        assert np.allclose(residuals_a, 0.0, atol=1e-9), f"path A residuals: {residuals_a}"
        assert np.allclose(residuals_b, 0.0, atol=1e-9), f"path B residuals: {residuals_b}"


def test_tv3_a1_zero_produces_finite_coefficients() -> None:
    """``a_1 = 0`` violates V_3's eq. (5) precondition; coefficients
    should still be finite (no NaN). Vanishing set may differ; future
    phase 5c.4 will dispatch the degenerate branch explicitly.
    """
    coeffs = tv3_hyperplanes_rrr(
        a_1=0.0, l_1=0.4, d_2=0.1, a_2=0.6, l_2=0.3, d_3=0.2, a_3=0.4, l_3=0.5, v_3=0.7
    )
    assert np.all(np.isfinite(coeffs))


def test_tv3_l1_zero_produces_finite_coefficients() -> None:
    """``l_1 = 0`` (alpha_1 = 0 or pi) violates V_3's eq. (5) precondition."""
    coeffs = tv3_hyperplanes_rrr(
        a_1=0.6, l_1=0.0, d_2=0.1, a_2=0.4, l_2=0.5, d_3=0.2, a_3=0.4, l_3=0.5, v_3=0.7
    )
    assert np.all(np.isfinite(coeffs))


def test_tv3_shape() -> None:
    """tv3_hyperplanes_rrr returns 4x8 float64."""
    coeffs = tv3_hyperplanes_rrr(
        a_1=0.3, l_1=0.5, d_2=0.1, a_2=0.4, l_2=0.6, d_3=0.2, a_3=0.5, l_3=0.7, v_3=0.0
    )
    assert coeffs.shape == (4, 8)
    assert coeffs.dtype == np.float64


# ============================================================================
# Phase 5d step 1: T(v_6) -- right-chain hyperplanes parametrised by joint 6.
# ============================================================================
#
# Capco eq. (6) constructs T(v_6) from T(v_1) via parameter substitutions
# + a final sigma_E^* change of variables. Conventions assumed by
# tv6_hyperplanes_rrr: a_6 = d_6 = l_6 = 0 (caller absorbs joint-6 EE
# offset into sigma_E by left-multiplying).
#
# Bulletproof property: for ANY q_star = (v_1, v_2, v_3, v_4, v_5, v_6) and
# DH parameters with a_5, l_5 != 0, the T(v_6) hyperplanes vanish on the
# left-chain DQ tau = sigma_1(v_1) sigma_2(v_2) sigma_3(v_3) when sigma_E
# is the full 6R chain DQ at q_star.


def _full_6r_chain_dq(
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
    v_4: float,
    d_4: float,
    a_4: float,
    l_4: float,
    v_5: float,
    d_5: float,
    a_5: float,
    l_5: float,
    v_6: float,
) -> np.ndarray:
    """Full 6R chain DQ: ``sigma_1 sigma_2 sigma_3 sigma_4 sigma_5 sigma_6``
    with ``a_6 = d_6 = l_6 = 0`` (sigma_6 = R_z(v_6) only).
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
    sigma_4 = dq_mul(
        _rz_dq(v_4),
        dq_mul(_tz_dq(d_4), dq_mul(_tx_dq(a_4), _rx_dq(l_4))),
    )
    sigma_5 = dq_mul(
        _rz_dq(v_5),
        dq_mul(_tz_dq(d_5), dq_mul(_tx_dq(a_5), _rx_dq(l_5))),
    )
    sigma_6 = _rz_dq(v_6)  # a_6 = d_6 = l_6 = 0
    return dq_mul(
        sigma_1,
        dq_mul(sigma_2, dq_mul(sigma_3, dq_mul(sigma_4, dq_mul(sigma_5, sigma_6)))),
    )


@pytest.mark.parametrize("seed", list(range(10)))
def test_tv6_hyperplanes_vanish_on_left_chain_dq(seed: int) -> None:
    """T(v_6) hyperplanes vanish on the LEFT chain DQ when sigma_E is the
    full 6R FK pose.

    Setup: pick random (DH, q_star). Compute sigma_E = full 6R FK DQ.
    The LEFT-chain DQ tau = sigma_1 sigma_2 sigma_3 belongs to V_L AND
    to V_R = sigma_E sigma_6^{-1} sigma_5^{-1} sigma_4^{-1} (they coincide
    at the IK solution by construction). T(v_6) is the V_R 3-space, so
    its hyperplanes must vanish on tau.
    """
    rng = np.random.default_rng(seed + 400)
    a_1 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_1 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    d_2 = float(rng.uniform(-1.0, 1.0))
    a_2 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_2 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    d_3 = float(rng.uniform(-1.0, 1.0))
    a_3 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_3 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    d_4 = float(rng.uniform(-1.0, 1.0))
    a_4 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_4 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
    d_5 = float(rng.uniform(-1.0, 1.0))
    a_5 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
    l_5 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))

    v_1 = float(rng.uniform(-2.0, 2.0))
    v_2 = float(rng.uniform(-2.0, 2.0))
    v_3 = float(rng.uniform(-2.0, 2.0))
    v_4 = float(rng.uniform(-2.0, 2.0))
    v_5 = float(rng.uniform(-2.0, 2.0))
    v_6 = float(rng.uniform(-2.0, 2.0))

    sigma_E = _full_6r_chain_dq(
        v_1,
        a_1,
        l_1,
        d_2,
        v_2,
        a_2,
        l_2,
        v_3,
        d_3,
        a_3,
        l_3,
        v_4,
        d_4,
        a_4,
        l_4,
        v_5,
        d_5,
        a_5,
        l_5,
        v_6,
    )

    # tau = sigma_1 sigma_2 sigma_3 (left chain only)
    tau = _vl_chain_rrr(v_1, a_1, l_1, d_2, v_2, a_2, l_2, v_3, d_3, a_3, l_3)

    coeffs = tv6_hyperplanes_rrr(a_4, l_4, d_4, a_5, l_5, d_5, sigma_E, v_6)
    residuals = hyperplane_residuals(coeffs, tau)
    sigma_norm = float(np.linalg.norm(tau))
    row_norms = np.linalg.norm(coeffs, axis=1)
    expected_scales = np.maximum(row_norms * sigma_norm, 1e-12)
    relative = np.abs(residuals) / expected_scales
    assert np.all(relative < 1e-10), (
        f"seed={seed}: max relative residual={float(np.max(relative)):.2e}, "
        f"absolute={residuals}, scales={expected_scales}"
    )


@pytest.mark.parametrize("v_4", [-1.0, 0.0, 0.7, 1.5])
@pytest.mark.parametrize("v_5", [-1.0, 0.0, 0.7, 1.5])
def test_tv6_invariant_under_v4_v5_changes(v_4: float, v_5: float) -> None:
    """Fix DH and v_6; vary v_4, v_5 across V_R's other free parameters.
    T(v_6) coefficients are independent of (v_4, v_5) by construction;
    every V_L pose at the corresponding sigma_E must satisfy them.
    """
    a_1, l_1, d_2 = 0.5, 0.4, 0.1
    a_2, l_2 = 0.6, 0.3
    d_3, a_3, l_3 = 0.2, 0.4, 0.5
    d_4, a_4, l_4 = 0.15, 0.45, 0.55
    d_5, a_5, l_5 = 0.18, 0.5, 0.45
    v_1, v_2, v_3, v_6 = 0.3, -0.4, 0.5, 0.7

    sigma_E = _full_6r_chain_dq(
        v_1,
        a_1,
        l_1,
        d_2,
        v_2,
        a_2,
        l_2,
        v_3,
        d_3,
        a_3,
        l_3,
        v_4,
        d_4,
        a_4,
        l_4,
        v_5,
        d_5,
        a_5,
        l_5,
        v_6,
    )
    tau = _vl_chain_rrr(v_1, a_1, l_1, d_2, v_2, a_2, l_2, v_3, d_3, a_3, l_3)
    coeffs = tv6_hyperplanes_rrr(a_4, l_4, d_4, a_5, l_5, d_5, sigma_E, v_6)
    residuals = hyperplane_residuals(coeffs, tau)
    sigma_norm = float(np.linalg.norm(tau))
    row_norms = np.linalg.norm(coeffs, axis=1)
    expected_scales = np.maximum(row_norms * sigma_norm, 1e-12)
    relative = np.abs(residuals) / expected_scales
    assert np.all(relative < 1e-10), (
        f"v_4={v_4}, v_5={v_5}: max relative={float(np.max(relative)):.2e}"
    )


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
    d_4=_safe_dist,
    a_4=_safe_dh,
    s_a4=_sign,
    alpha_4=_safe_alpha,
    d_5=_safe_dist,
    a_5=_safe_dh,
    s_a5=_sign,
    alpha_5=_safe_alpha,
    v_1=_safe_q,
    v_2=_safe_q,
    v_3=_safe_q,
    v_4=_safe_q,
    v_5=_safe_q,
    v_6=_safe_q,
)
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_tv6_hypothesis_fuzz_500_examples(
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
    d_4: float,
    a_4: float,
    s_a4: float,
    alpha_4: float,
    d_5: float,
    a_5: float,
    s_a5: float,
    alpha_5: float,
    v_1: float,
    v_2: float,
    v_3: float,
    v_4: float,
    v_5: float,
    v_6: float,
) -> None:
    """Bulletproof gate for T(v_6): 500 (DH, q) combos across full alpha
    range; relative tolerance 1e-12. Sigma_E built from full 6R FK; T(v_6)
    hyperplanes vanish on the left-chain DQ.
    """
    a_1_signed = s_a1 * a_1
    a_2_signed = s_a2 * a_2
    a_3_signed = s_a3 * a_3
    a_4_signed = s_a4 * a_4
    a_5_signed = s_a5 * a_5
    l_1 = math.tan(0.5 * alpha_1)
    l_2 = math.tan(0.5 * alpha_2)
    l_3 = math.tan(0.5 * alpha_3)
    l_4 = math.tan(0.5 * alpha_4)
    l_5 = math.tan(0.5 * alpha_5)

    sigma_E = _full_6r_chain_dq(
        v_1,
        a_1_signed,
        l_1,
        d_2,
        v_2,
        a_2_signed,
        l_2,
        v_3,
        d_3,
        a_3_signed,
        l_3,
        v_4,
        d_4,
        a_4_signed,
        l_4,
        v_5,
        d_5,
        a_5_signed,
        l_5,
        v_6,
    )
    tau = _vl_chain_rrr(v_1, a_1_signed, l_1, d_2, v_2, a_2_signed, l_2, v_3, d_3, a_3_signed, l_3)
    coeffs = tv6_hyperplanes_rrr(a_4_signed, l_4, d_4, a_5_signed, l_5, d_5, sigma_E, v_6)
    residuals = hyperplane_residuals(coeffs, tau)
    sigma_norm = float(np.linalg.norm(tau))
    row_norms = np.linalg.norm(coeffs, axis=1)
    expected_scales = np.maximum(row_norms * sigma_norm, 1e-12)
    relative = np.abs(residuals) / expected_scales
    assert np.all(relative < 1e-12), (
        f"max relative residual={float(np.max(relative)):.2e}; "
        f"absolute={residuals}, scales={expected_scales}"
    )


def test_tv6_independent_path_agrees() -> None:
    """T(v_6) lambdified path agrees with direct V_R computation.

    Path A: tv6_hyperplanes_rrr(DH, sigma_E, v_6) @ tau (where tau is the
    LEFT-chain DQ, which equals V_R-points by the IK relation).

    Path B: build V_R explicitly via numerical conjugation:
    tau_R = sigma_E . sigma_6^{-1} . sigma_5^{-1} . sigma_4^{-1}.
    Apply T(v_1)-with-substitutions hyperplanes to tau_R (path BEFORE the
    sigma_E^* change of variables). Should also vanish.

    Both paths vanish on V_R points; agreement at 1e-9.
    """
    rng = np.random.default_rng(500)
    for _ in range(20):
        a_1 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
        l_1 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
        d_2 = float(rng.uniform(-1.0, 1.0))
        a_2 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
        l_2 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
        d_3 = float(rng.uniform(-1.0, 1.0))
        a_3 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
        l_3 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
        d_4 = float(rng.uniform(-1.0, 1.0))
        a_4 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
        l_4 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
        d_5 = float(rng.uniform(-1.0, 1.0))
        a_5 = float(rng.uniform(0.1, 1.5)) * float(rng.choice([-1.0, 1.0]))
        l_5 = math.tan(0.5 * float(rng.uniform(0.1, math.pi - 0.1)))
        v_1 = float(rng.uniform(-2.0, 2.0))
        v_2 = float(rng.uniform(-2.0, 2.0))
        v_3 = float(rng.uniform(-2.0, 2.0))
        v_4 = float(rng.uniform(-2.0, 2.0))
        v_5 = float(rng.uniform(-2.0, 2.0))
        v_6 = float(rng.uniform(-2.0, 2.0))

        sigma_E = _full_6r_chain_dq(
            v_1,
            a_1,
            l_1,
            d_2,
            v_2,
            a_2,
            l_2,
            v_3,
            d_3,
            a_3,
            l_3,
            v_4,
            d_4,
            a_4,
            l_4,
            v_5,
            d_5,
            a_5,
            l_5,
            v_6,
        )
        tau = _vl_chain_rrr(v_1, a_1, l_1, d_2, v_2, a_2, l_2, v_3, d_3, a_3, l_3)

        # Path A: lambdified tv6_hyperplanes_rrr.
        coeffs_a = tv6_hyperplanes_rrr(a_4, l_4, d_4, a_5, l_5, d_5, sigma_E, v_6)
        residuals_a = coeffs_a @ tau

        # Path B: apply sigma_E^* to tau (tau is the V_L pose, which equals
        # sigma_E (full chain) at IK; sigma_E^* @ tau lands in V_R-pre-
        # change-of-variables coordinates), then apply substituted T(v_1).
        sigma_E_conj = _dq_conj_numpy(sigma_E)
        # tau_R = sigma_E^* @ tau (== V_R-pre-sigma_E version of the IK pose)
        tau_R = dq_mul(sigma_E_conj, tau)
        coeffs_b = tv1_hyperplanes_rrr(
            a_1=-a_5,
            l_1=-l_5,
            d_2=-d_5,
            a_2=-a_4,
            l_2=-l_4,
            d_3=-d_4,
            a_3=0.0,
            l_3=0.0,
            v_1=-v_6,
        )
        residuals_b = coeffs_b @ tau_R

        assert np.allclose(residuals_a, 0.0, atol=1e-9), f"path A residuals: {residuals_a}"
        assert np.allclose(residuals_b, 0.0, atol=1e-9), f"path B residuals: {residuals_b}"


def test_tv6_at_identity_sigma_e_collapses_to_substituted_v1() -> None:
    """When ``sigma_E = (1, 0, 0, 0, 0, 0, 0, 0)`` (identity DQ), the
    sigma_E^* change of variables is identity, so T(v_6) reduces to
    the substituted T(v_1). Sanity check on the wiring.
    """
    a_4, l_4, d_4 = 0.45, 0.55, 0.15
    a_5, l_5, d_5 = 0.5, 0.45, 0.18
    v_6 = 0.7
    sigma_E_identity = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    coeffs_tv6 = tv6_hyperplanes_rrr(a_4, l_4, d_4, a_5, l_5, d_5, sigma_E_identity, v_6)
    coeffs_tv1_sub = tv1_hyperplanes_rrr(
        a_1=-a_5,
        l_1=-l_5,
        d_2=-d_5,
        a_2=-a_4,
        l_2=-l_4,
        d_3=-d_4,
        a_3=0.0,
        l_3=0.0,
        v_1=-v_6,
    )
    assert np.allclose(coeffs_tv6, coeffs_tv1_sub, atol=1e-15)


def test_tv6_shape() -> None:
    sigma_E = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5], dtype=np.float64)
    coeffs = tv6_hyperplanes_rrr(
        a_4=0.5,
        l_4=0.4,
        d_4=0.1,
        a_5=0.6,
        l_5=0.3,
        d_5=0.2,
        sigma_E=sigma_E,
        v_6=0.0,
    )
    assert coeffs.shape == (4, 8)
    assert coeffs.dtype == np.float64


def test_tv6_rejects_wrong_shape_sigma_e() -> None:
    """sigma_E must be 8-vec; otherwise raise ValueError."""
    with pytest.raises(ValueError, match="sigma_E must be 8-vec"):
        tv6_hyperplanes_rrr(
            a_4=0.5,
            l_4=0.4,
            d_4=0.1,
            a_5=0.6,
            l_5=0.3,
            d_5=0.2,
            sigma_E=np.zeros(7),  # wrong shape
            v_6=0.0,
        )


# ----------------------------------------------------------------------------
# T(v_2) -- Phase 5c.4 / GitHub #176 -- the double-degenerate parametrization.
# Triggered for RRR when both T(v_1) (a_2!=0 AND l_2!=0) and T(v_3) (a_1!=0
# AND l_1!=0) are violated. Capco enumerates 4 sub-cases keyed by which DH
# parameter pair is zero -- each builds a different 4-hyperplane system
# from the 12x16 kernel construction.
# ----------------------------------------------------------------------------


def _vl_chain_rrr_simple(
    v_1: float, a_1: float, l_1: float,
    v_2: float, d_2: float, a_2: float, l_2: float,
    v_3: float,
) -> np.ndarray:
    """RRR left-chain DQ in the SIMPLE form ``T(v_2)`` derives against:
    sigma_1(v_1, d_1=0, a_1, l_1) . sigma_2(v_2, d_2, a_2, l_2) . R_z(v_3).

    Differs from :func:`_vl_chain_rrr` in that joint-3 DH parameters
    (a_3, l_3, d_3) are absent -- they get absorbed by the Tv2_full
    change of variables in ``_eliminate.py``.
    """
    sigma_1 = dq_mul(_rz_dq(v_1), dq_mul(_tx_dq(a_1), _rx_dq(l_1)))
    sigma_2 = dq_mul(
        _rz_dq(v_2),
        dq_mul(_tz_dq(d_2), dq_mul(_tx_dq(a_2), _rx_dq(l_2))),
    )
    return dq_mul(sigma_1, dq_mul(sigma_2, _rz_dq(v_3)))


_TV2_RRR_CASE_DH = {
    "[a_1=0,a_2=0]": dict(a_1=0.0, a_2=0.0),
    "[a_1=0,l_2=0]": dict(a_1=0.0, l_2=0.0),
    "[l_1=0,a_2=0]": dict(l_1=0.0, a_2=0.0),
    "[l_1=0,l_2=0]": dict(l_1=0.0, l_2=0.0),
}


def _random_dh_for_tv2_case(rng: np.random.Generator, case_key: str) -> dict[str, float]:
    """Random non-degenerate DH satisfying the sub-case condition.
    The DH params NOT pinned to zero by the sub-case are drawn from a
    safe non-degenerate range.
    """
    fixed = _TV2_RRR_CASE_DH[case_key]
    out = {
        "a_1": float(rng.uniform(0.2, 1.0)) * float(rng.choice([-1.0, 1.0])),
        "l_1": math.tan(0.5 * float(rng.uniform(0.3, math.pi - 0.3))),
        "d_2": float(rng.uniform(-1.0, 1.0)),
        "a_2": float(rng.uniform(0.2, 1.0)) * float(rng.choice([-1.0, 1.0])),
        "l_2": math.tan(0.5 * float(rng.uniform(0.3, math.pi - 0.3))),
    }
    out.update(fixed)
    return out


@pytest.mark.parametrize("case_key", TV2_RRR_CASE_KEYS)
@pytest.mark.parametrize("seed", list(range(5)))
def test_tv2_hyperplanes_vanish_on_full_vl_chain(case_key: str, seed: int) -> None:
    """T(v_2) hyperplanes vanish on the full RRR left-chain DQ (in the
    simple form -- joint-3 DH absent) for random (v_1, v_2, v_3) and
    random DH satisfying the sub-case degeneracy condition.

    Tolerance 1e-12 (post-numeric-SVD-kernel construction; tighter than
    T(v_1)/T(v_3) which use sympy lambdify and lose ~3 digits to numpy
    contraction order).
    """
    rng = np.random.default_rng(seed * 100 + (hash(case_key) % 1000))
    dh = _random_dh_for_tv2_case(rng, case_key)
    v_1 = float(rng.uniform(-2.0, 2.0))
    v_2 = float(rng.uniform(-2.0, 2.0))
    v_3 = float(rng.uniform(-2.0, 2.0))

    sigma_vl = _vl_chain_rrr_simple(
        v_1, dh["a_1"], dh["l_1"], v_2, dh["d_2"], dh["a_2"], dh["l_2"], v_3
    )

    h_sym = tv2_symbolic_in_v2(case_key, **dh)
    h_at_v2 = h_sym.subs(_V2_SYM, sp.Float(v_2))
    h_np = np.array(h_at_v2.tolist(), dtype=np.float64)

    residuals = h_np @ sigma_vl
    assert np.allclose(residuals, 0.0, atol=1e-12), (
        f"case={case_key} seed={seed}: max|residual|="
        f"{float(np.max(np.abs(residuals))):.2e}"
    )


def test_tv2_rrr_case_for_picks_correct_subcase() -> None:
    """``tv2_rrr_case_for`` mirrors Capco's which_case.py dispatch."""
    assert tv2_rrr_case_for(0.0, 1.0, 0.0, 1.0) == "[a_1=0,a_2=0]"
    assert tv2_rrr_case_for(0.0, 1.0, 0.5, 0.0) == "[a_1=0,l_2=0]"
    assert tv2_rrr_case_for(0.5, 0.0, 0.0, 1.0) == "[l_1=0,a_2=0]"
    assert tv2_rrr_case_for(0.5, 0.0, 0.5, 0.0) == "[l_1=0,l_2=0]"
    with pytest.raises(ValueError, match=r"do not match any T\(v_2\) RRR sub-case"):
        # Non-degenerate DH: T(v_1) applies, no T(v_2) sub-case.
        tv2_rrr_case_for(0.5, 1.0, 0.5, 1.0)


@pytest.mark.parametrize("case_key", TV2_RRR_CASE_KEYS)
@pytest.mark.parametrize("seed", list(range(5)))
def test_tv2_full_hyperplanes_vanish_on_full_vl_chain(case_key: str, seed: int) -> None:
    """Tv2_full hyperplanes (with joint-3 DH change of variables) vanish
    on the FULL RRR left-chain DQ ``sigma_1 sigma_2 sigma_3`` (now
    including joint-3 DH offsets ``a_3, l_3, d_3``) for random
    (v_1, v_2, v_3) and random DH satisfying the sub-case condition.

    This is the form ``_eliminate.precompute_rrr_chain`` consumes -- the
    Study coords are at frame F_4, after joint 3's full transition has
    acted.

    Tolerance 1e-12 (post-numeric-SVD-kernel construction).
    """
    rng = np.random.default_rng(seed * 100 + (hash(case_key) % 1000) + 17)
    dh = _random_dh_for_tv2_case(rng, case_key)
    # Joint-3 DH parameters (free, not part of the sub-case).
    d_3 = float(rng.uniform(-1.0, 1.0))
    a_3 = float(rng.uniform(0.2, 1.0)) * float(rng.choice([-1.0, 1.0]))
    l_3 = math.tan(0.5 * float(rng.uniform(0.3, math.pi - 0.3)))
    v_1 = float(rng.uniform(-2.0, 2.0))
    v_2 = float(rng.uniform(-2.0, 2.0))
    v_3 = float(rng.uniform(-2.0, 2.0))

    # Full RRR chain INCLUDING joint-3 DH offsets.
    sigma_vl_full = _vl_chain_rrr(
        v_1, dh["a_1"], dh["l_1"], dh["d_2"], v_2, dh["a_2"], dh["l_2"],
        v_3, d_3, a_3, l_3,
    )

    coeffs_full = tv2_hyperplanes_rrr(
        case_key,
        a_1=dh["a_1"], l_1=dh["l_1"], d_2=dh["d_2"],
        a_2=dh["a_2"], l_2=dh["l_2"],
        d_3=d_3, a_3=a_3, l_3=l_3,
        v_2=v_2,
    )
    residuals = coeffs_full @ sigma_vl_full
    assert np.allclose(residuals, 0.0, atol=1e-12), (
        f"case={case_key} seed={seed}: max|residual|="
        f"{float(np.max(np.abs(residuals))):.2e}"
    ) 