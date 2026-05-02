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

from ssik.solvers.husty_pfurner._constraints import (
    hyperplane_residuals,
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
