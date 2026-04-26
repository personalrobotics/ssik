"""Self-consistency tests for the Raghavan-Roth (P, Q) builder.

The (P, Q) matrices encode the loop-closure equation as

    (P_sin[i] s_2 + P_cos[i] c_2 + P_one[i]) . v_left(q_3, q_4)
    = Q[i] . v_right(q_0, q_1)

for each row ``i``. So if we evaluate at a *known* IK solution ``(q_0, ...,
q_4)`` and the corresponding ``T_target = FK(q*)``, every row must vanish.

These tests check that on a standard DH 6R chain (the Manocha-Canny Table I
geometry, which has 16 real IK solutions for a generic pose):

1. Substituting q* yields ``P_sin, P_cos, P_one, Q`` whose rows close to
   machine precision -- proves the symbolic derivation got the monomial
   structure right.
2. The closure holds across multiple random q* seeds -- proves no special
   cancellation (would catch a sign-flip in one row, etc.).
3. The closure holds across multiple random DH parameter sets -- proves the
   derivation is generic, not over-fit to MC Table I.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ssik.solvers.ikgeo._raghavan_roth import build_pq


def _dh_matrix(theta: float, alpha: float, a: float, d: float) -> NDArray[np.float64]:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _fk_dh(q: NDArray[np.float64], alpha: NDArray[np.float64], a: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward kinematics for a standard-DH 6R chain."""
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh_matrix(float(q[i]), float(alpha[i]), float(a[i]), float(d[i]))
    return T


# Manocha-Canny Table I (IEEE T-RA 1994, page 654). The synthetic 6R has
# 16 real IK solutions for the published end-effector pose.
# Twist angles given in the paper as 90.0 / 1.0 -- we read 1.0 as 1 radian
# (the column header reads "Twist angle"; the column has both 90.0 and 1.0,
# clearly mixing degrees and radians). Until verified, treat as a synthetic
# fixture only -- exact match to the paper's matrices is a stretch goal.
_MC_TABLE_I_ALPHA = np.array([np.pi / 2, 1.0, np.pi / 2, 1.0, np.pi / 2, 1.0])
_MC_TABLE_I_A = np.array([0.3, 1.0, 0.0, 1.5, 0.0, 0.0])
_MC_TABLE_I_D = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0])


def _v_left(q3: float, q4: float) -> NDArray[np.float64]:
    s3, c3 = np.sin(q3), np.cos(q3)
    s4, c4 = np.sin(q4), np.cos(q4)
    return np.array([s3 * s4, s3 * c4, c3 * s4, c3 * c4, s3, c3, s4, c4, 1.0])


def _v_right(q0: float, q1: float) -> NDArray[np.float64]:
    s0, c0 = np.sin(q0), np.cos(q0)
    s1, c1 = np.sin(q1), np.cos(q1)
    return np.array([s0 * s1, s0 * c1, c0 * s1, c0 * c1, s0, c0, s1, c1])


# ---------------------------------------------------------------------------
# Self-consistency on MC Table I geometry across multiple seeded q*.
# ---------------------------------------------------------------------------

_SEEDED_Q = [
    np.array([0.3, -0.7, 0.5, 1.1, -0.4, 0.6]),
    np.array([-1.2, 0.4, -0.9, 0.2, 0.8, -1.5]),
    np.array([0.0, 0.1, -0.2, 0.3, -0.4, 0.5]),
    np.array([2.1, -1.3, 0.7, -0.8, 1.4, -2.0]),
]


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_pq_closes_at_seeded_q_mc_table_i(q_star: NDArray[np.float64]) -> None:
    """For the MC Table I 6R, P_sin, P_cos, P_one, Q must close at q*."""
    alpha = _MC_TABLE_I_ALPHA
    a = _MC_TABLE_I_A
    d = _MC_TABLE_I_D
    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)

    s2, c2 = np.sin(q_star[2]), np.cos(q_star[2])
    p_eff = p_sin * s2 + p_cos * c2 + p_one  # 6x9

    v_left = _v_left(q_star[3], q_star[4])
    v_right = _v_right(q_star[0], q_star[1])

    residual = p_eff @ v_left - q_mat @ v_right  # 6-vector
    assert np.allclose(residual, 0.0, atol=1e-10), (
        f"P @ v_left != Q @ v_right at seeded q*; residual={residual}"
    )


# ---------------------------------------------------------------------------
# Random DH parameters: prove the derivation is generic.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_pq_closes_random_dh_random_q(seed: int) -> None:
    """For random-DH 6R chains and random q*, the (P, Q) closure must hold."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-np.pi, np.pi, size=6)
    a = rng.uniform(-1.0, 1.0, size=6)
    d = rng.uniform(-1.0, 1.0, size=6)
    q_star = rng.uniform(-np.pi, np.pi, size=6)

    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)

    s2, c2 = np.sin(q_star[2]), np.cos(q_star[2])
    p_eff = p_sin * s2 + p_cos * c2 + p_one
    residual = p_eff @ _v_left(q_star[3], q_star[4]) - q_mat @ _v_right(q_star[0], q_star[1])

    assert np.allclose(residual, 0.0, atol=1e-10), (
        f"random DH closure failed (seed={seed}); residual norm={np.linalg.norm(residual):.3e}"
    )


# ---------------------------------------------------------------------------
# Output shape and dtype contract.
# ---------------------------------------------------------------------------


def test_pq_output_shapes() -> None:
    alpha = _MC_TABLE_I_ALPHA
    a = _MC_TABLE_I_A
    d = _MC_TABLE_I_D
    q_star = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)

    assert p_sin.shape == (14, 9)
    assert p_cos.shape == (14, 9)
    assert p_one.shape == (14, 9)
    assert q_mat.shape == (14, 8)
    assert p_sin.dtype == np.float64
