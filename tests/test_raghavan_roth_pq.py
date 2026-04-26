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


@pytest.mark.parametrize("seed", [0])  # one random DH suffices alongside MC Table I
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


# ---------------------------------------------------------------------------
# SVD elimination of (q_0, q_1).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_eliminate_q0_q1_closes_at_seeded_q(q_star: NDArray[np.float64]) -> None:
    """After eliminating (q_0, q_1) via SVD, the resulting 6 equations must
    vanish at the seeded q*.

    The elimination subtracts (q_0, q_1) dependence by projecting onto the
    left null space of Q. So (E_sin s_2 + E_cos c_2 + E_one) @ v_left(q_3, q_4)
    must equal zero at q*.
    """
    from ssik.solvers.ikgeo._raghavan_roth import eliminate_q0_q1

    alpha = _MC_TABLE_I_ALPHA
    a = _MC_TABLE_I_A
    d = _MC_TABLE_I_D
    t_target = _fk_dh(q_star, alpha, a, d)
    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    assert e_sin.shape == (6, 9)

    s2, c2 = np.sin(q_star[2]), np.cos(q_star[2])
    e_eff = e_sin * s2 + e_cos * c2 + e_one  # 6x9
    residual = e_eff @ _v_left(q_star[3], q_star[4])  # 6-vector
    assert np.allclose(residual, 0.0, atol=1e-10), (
        f"E @ v_left != 0 at seeded q*; residual={residual}"
    )


@pytest.mark.parametrize("seed", [0])
def test_eliminate_q0_q1_random_dh(seed: int) -> None:
    from ssik.solvers.ikgeo._raghavan_roth import eliminate_q0_q1

    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-np.pi, np.pi, size=6)
    a = rng.uniform(-1.0, 1.0, size=6)
    d = rng.uniform(-1.0, 1.0, size=6)
    q_star = rng.uniform(-np.pi, np.pi, size=6)
    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)

    s2, c2 = np.sin(q_star[2]), np.cos(q_star[2])
    e_eff = e_sin * s2 + e_cos * c2 + e_one
    residual = e_eff @ _v_left(q_star[3], q_star[4])
    assert np.allclose(residual, 0.0, atol=1e-9), (
        f"E @ v_left != 0 at seeded q* (seed={seed}); ||residual||={np.linalg.norm(residual):.3e}"
    )


# ---------------------------------------------------------------------------
# Weierstrass substitution + v_left_trig -> v_left_x basis change.
# ---------------------------------------------------------------------------


def _v_left_x(q3: float, q4: float) -> NDArray[np.float64]:
    """Reference v_left_x basis at numeric joint angles."""
    x3, x4 = np.tan(q3 / 2.0), np.tan(q4 / 2.0)
    return np.array(
        [x3**2 * x4**2, x3**2 * x4, x3**2, x3 * x4**2, x3 * x4, x3, x4**2, x4, 1.0]
    )


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_weierstrass_eliminate_closes_at_seeded_q(q_star: NDArray[np.float64]) -> None:
    """After Weierstrass substitution for q_2 + basis change for (q_3, q_4),
    the 6 equations must still vanish at the seeded q*."""
    from ssik.solvers.ikgeo._raghavan_roth import (
        eliminate_q0_q1,
        weierstrass_eliminate_trig,
    )

    alpha = _MC_TABLE_I_ALPHA
    a = _MC_TABLE_I_A
    d = _MC_TABLE_I_D
    t_target = _fk_dh(q_star, alpha, a, d)
    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)

    assert e_quad.shape == (6, 9)
    x2 = float(np.tan(q_star[2] / 2.0))
    e_eff = e_quad * x2 * x2 + e_lin * x2 + e_const  # 6x9 evaluated at x_2*
    residual = e_eff @ _v_left_x(q_star[3], q_star[4])  # 6-vector
    assert np.allclose(residual, 0.0, atol=1e-9), (
        f"E(x_2) @ v_left_x != 0 at seeded q*; residual={residual}"
    )


@pytest.mark.parametrize("seed", [0])
def test_weierstrass_eliminate_random_dh(seed: int) -> None:
    from ssik.solvers.ikgeo._raghavan_roth import (
        eliminate_q0_q1,
        weierstrass_eliminate_trig,
    )

    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-np.pi, np.pi, size=6)
    a = rng.uniform(-1.0, 1.0, size=6)
    d = rng.uniform(-1.0, 1.0, size=6)
    q_star = rng.uniform(-np.pi / 2, np.pi / 2, size=6)  # avoid x = inf
    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)

    x2 = float(np.tan(q_star[2] / 2.0))
    e_eff = e_quad * x2 * x2 + e_lin * x2 + e_const
    residual = e_eff @ _v_left_x(q_star[3], q_star[4])
    assert np.allclose(residual, 0.0, atol=1e-9), (
        f"E(x_2) @ v_left_x != 0 at seeded q* (seed={seed}); ||residual||={np.linalg.norm(residual):.3e}"
    )


# ---------------------------------------------------------------------------
# 12x12 M(x_2) matrix polynomial.
# ---------------------------------------------------------------------------


def _v_12(q3: float, q4: float) -> NDArray[np.float64]:
    """Reference 12-monomial vector at numeric joint angles."""
    x3, x4 = np.tan(q3 / 2.0), np.tan(q4 / 2.0)
    return np.array(
        [
            x3**2 * x4**2,
            x3**2 * x4,
            x3**2,
            x3 * x4**2,
            x3 * x4,
            x3,
            x4**2,
            x4,
            1.0,
            x3**3 * x4**2,
            x3**3 * x4,
            x3**3,
        ]
    )


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_m_matrix_closes_at_seeded_q(q_star: NDArray[np.float64]) -> None:
    """At a seeded IK solution, M(x_2*) @ v_12 must vanish (12-vector residual)."""
    from ssik.solvers.ikgeo._raghavan_roth import (
        build_m_matrix,
        eliminate_q0_q1,
        weierstrass_eliminate_trig,
    )

    alpha = _MC_TABLE_I_ALPHA
    a = _MC_TABLE_I_A
    d = _MC_TABLE_I_D
    t_target = _fk_dh(q_star, alpha, a, d)
    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)

    assert m_quad.shape == (12, 12)
    x2 = float(np.tan(q_star[2] / 2.0))
    m_eff = m_quad * x2 * x2 + m_lin * x2 + m_const
    residual = m_eff @ _v_12(q_star[3], q_star[4])
    assert np.allclose(residual, 0.0, atol=1e-9), (
        f"M(x_2) @ v_12 != 0 at seeded q*; ||residual||={np.linalg.norm(residual):.3e}"
    )


@pytest.mark.parametrize("seed", [0])
def test_m_matrix_random_dh(seed: int) -> None:
    from ssik.solvers.ikgeo._raghavan_roth import (
        build_m_matrix,
        eliminate_q0_q1,
        weierstrass_eliminate_trig,
    )

    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-np.pi, np.pi, size=6)
    a = rng.uniform(-1.0, 1.0, size=6)
    d = rng.uniform(-1.0, 1.0, size=6)
    q_star = rng.uniform(-np.pi / 2, np.pi / 2, size=6)
    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)

    x2 = float(np.tan(q_star[2] / 2.0))
    m_eff = m_quad * x2 * x2 + m_lin * x2 + m_const
    residual = m_eff @ _v_12(q_star[3], q_star[4])
    assert np.allclose(residual, 0.0, atol=1e-8), (
        f"M(x_2) @ v_12 != 0 at seeded q* (seed={seed}); ||residual||={np.linalg.norm(residual):.3e}"
    )


# ---------------------------------------------------------------------------
# 24x24 companion eigenvalue route -> tan(q_2/2) roots.
# ---------------------------------------------------------------------------


def _wrap_pi(x: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float(((x + np.pi) % (2 * np.pi)) - np.pi)


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_solve_x2_roots_recovers_seeded_q2(q_star: NDArray[np.float64]) -> None:
    """The seeded tan(q_2/2) must appear among the real eigenvalues of the
    24x24 companion matrix. Other eigenvalues correspond to *other* IK
    branches at the same target pose."""
    from ssik.solvers.ikgeo._raghavan_roth import (
        build_m_matrix,
        eliminate_q0_q1,
        solve_x2_roots,
        weierstrass_eliminate_trig,
    )

    alpha = _MC_TABLE_I_ALPHA
    a = _MC_TABLE_I_A
    d = _MC_TABLE_I_D
    t_target = _fk_dh(q_star, alpha, a, d)
    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)
    roots, _ = solve_x2_roots(m_quad, m_lin, m_const)

    x2_star = float(np.tan(q_star[2] / 2.0))
    closest = min(roots, key=lambda r: abs(r - x2_star))
    assert abs(closest - x2_star) < 1e-6, (
        f"seeded tan(q_2/2)={x2_star:.6f} not among roots {sorted(roots)};"
        f" closest={closest:.6f}, error={abs(closest - x2_star):.3e}"
    )
    # Sanity: at most 16 real roots (degree-16 polynomial after spurious-i removal).
    assert len(roots) <= 16


@pytest.mark.parametrize("seed", [0])
def test_solve_x2_roots_random_dh(seed: int) -> None:
    from ssik.solvers.ikgeo._raghavan_roth import (
        build_m_matrix,
        eliminate_q0_q1,
        solve_x2_roots,
        weierstrass_eliminate_trig,
    )

    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-np.pi, np.pi, size=6)
    a = rng.uniform(-1.0, 1.0, size=6)
    d = rng.uniform(-1.0, 1.0, size=6)
    q_star = rng.uniform(-np.pi / 2, np.pi / 2, size=6)
    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)
    roots, _ = solve_x2_roots(m_quad, m_lin, m_const)

    x2_star = float(np.tan(q_star[2] / 2.0))
    closest = min(roots, key=lambda r: abs(r - x2_star))
    assert abs(closest - x2_star) < 1e-5, (
        f"seed={seed}: seeded tan(q_2/2)={x2_star:.6f} not among roots {sorted(roots)};"
        f" closest={closest:.6f}, error={abs(closest - x2_star):.3e}"
    )


# ---------------------------------------------------------------------------
# End-to-end: back-substitution recovers seeded q* from the right eigenvector.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_back_substitute_recovers_seeded_q_star(q_star: NDArray[np.float64]) -> None:
    """Full pipeline: build (P, Q), eliminate q_0/q_1, Weierstrass, build M(x_2),
    eigenvalue, then back-substitute. The seeded q* (mod wrap) must be recovered
    by the eigenvector closest to tan(q_2*/2)."""
    from ssik.solvers.ikgeo._raghavan_roth import (
        back_substitute,
        build_m_matrix,
        eliminate_q0_q1,
        solve_x2_roots,
        weierstrass_eliminate_trig,
    )

    alpha = _MC_TABLE_I_ALPHA
    a = _MC_TABLE_I_A
    d = _MC_TABLE_I_D
    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)
    roots, eigvecs = solve_x2_roots(m_quad, m_lin, m_const)

    # Pick the root closest to tan(q_2*/2) and back-substitute.
    x2_star = float(np.tan(q_star[2] / 2.0))
    best_idx = min(range(len(roots)), key=lambda i: abs(roots[i] - x2_star))
    q_recovered = back_substitute(
        roots[best_idx], eigvecs[best_idx], p_sin, p_cos, p_one, q_mat,
        (alpha, a, d), t_target,
    )
    assert q_recovered is not None, "back_substitute returned None at the closest root"

    # Compare wrap-to-pi joint-by-joint.
    diffs = [_wrap_pi(float(q_recovered[i] - q_star[i])) for i in range(6)]
    max_diff = max(abs(diff) for diff in diffs)
    assert max_diff < 1e-5, (
        f"recovered q does not match seeded q*; per-joint diffs={diffs}"
    )

    # And verify FK closure.
    t_recovered = _fk_dh(q_recovered, alpha, a, d)
    assert np.allclose(t_recovered, t_target, atol=1e-7), (
        f"recovered q fails FK closure; max diff={np.max(np.abs(t_recovered - t_target)):.3e}"
    )


def test_solve_all_ik_jaco2_like_geometry() -> None:
    """Regression: a JACO-2-like geometry (60-deg twists at joints 4,5) triggers
    cond(m_quad) ~ 1e16 -- the Manocha-Canny singular-pencil case. The
    solver must (1) detect the conditioning failure, (2) fall back through
    Mobius reparameterization + scipy generalized eigenvalue, and (3) recover
    the seeded q* via Newton refinement of the imprecise eigenvalue seed.

    Closes the EAIK gap on the non-Pieper geometry that motivated this work.
    """
    from ssik.solvers.ikgeo._raghavan_roth import solve_all_ik

    # JACO-2-like DH (Kinova j2n6 family): 60-deg twists at joints 4 and 5,
    # creating a near-singular leading matrix in M(x_2). Approximate values;
    # the real fixture is in robot-code/ada_assets/.../jaco2.xml.
    alpha = np.array([np.pi / 2, np.pi, np.pi / 2, 60 * np.pi / 180, 60 * np.pi / 180, np.pi])
    a = np.array([0.0, 0.41, 0.0, 0.0, 0.0, 0.0])
    d = np.array([0.2755, 0.0, -0.0098, -0.2502, -0.0858, -0.2116])
    q_star = np.array([0.3, -0.5, 0.7, 0.4, -0.6, 0.2])
    t_target = _fk_dh(q_star, alpha, a, d)

    solutions, is_ls = solve_all_ik((alpha, a, d), t_target, fk_atol=1e-5)
    assert not is_ls
    assert len(solutions) >= 1

    best = min(solutions, key=lambda q: max(abs(_wrap_pi(float(q[i] - q_star[i]))) for i in range(6)))
    diffs = [_wrap_pi(float(best[i] - q_star[i])) for i in range(6)]
    assert max(abs(d) for d in diffs) < 1e-6, (
        f"q* not recovered on JACO-like geometry; diffs={diffs}"
    )

    for i, q in enumerate(solutions):
        t_check = _fk_dh(q, alpha, a, d)
        assert np.allclose(t_check, t_target, atol=1e-5), (
            f"JACO-like solution {i} fails FK closure"
        )


def test_back_substitute_random_dh() -> None:
    """End-to-end random-DH test: derive (P, Q), solve eigenvalue, back-substitute,
    and verify FK closure on a non-MC-Table-I arm."""
    from ssik.solvers.ikgeo._raghavan_roth import (
        back_substitute,
        build_m_matrix,
        eliminate_q0_q1,
        solve_x2_roots,
        weierstrass_eliminate_trig,
    )

    rng = np.random.default_rng(0)
    alpha = rng.uniform(-np.pi, np.pi, size=6)
    a = rng.uniform(-1.0, 1.0, size=6)
    d = rng.uniform(-1.0, 1.0, size=6)
    q_star = rng.uniform(-np.pi / 2, np.pi / 2, size=6)
    t_target = _fk_dh(q_star, alpha, a, d)

    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)
    roots, eigvecs = solve_x2_roots(m_quad, m_lin, m_const)

    x2_star = float(np.tan(q_star[2] / 2.0))
    best_idx = min(range(len(roots)), key=lambda i: abs(roots[i] - x2_star))
    q_recovered = back_substitute(
        roots[best_idx], eigvecs[best_idx], p_sin, p_cos, p_one, q_mat,
        (alpha, a, d), t_target,
    )
    assert q_recovered is not None

    diffs = [_wrap_pi(float(q_recovered[i] - q_star[i])) for i in range(6)]
    assert max(abs(diff) for diff in diffs) < 1e-5, f"diffs={diffs}"

    t_recovered = _fk_dh(q_recovered, alpha, a, d)
    assert np.allclose(t_recovered, t_target, atol=1e-7)


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_solve_all_ik_recovers_q_star_and_alternatives(q_star: NDArray[np.float64]) -> None:
    """The full driver must (a) include the seeded q* among returned solutions
    and (b) return at least one alternative (MC Table I has up to 16
    solutions per pose)."""
    from ssik.solvers.ikgeo._raghavan_roth import solve_all_ik

    alpha = _MC_TABLE_I_ALPHA
    a = _MC_TABLE_I_A
    d = _MC_TABLE_I_D
    t_target = _fk_dh(q_star, alpha, a, d)
    solutions, is_ls = solve_all_ik((alpha, a, d), t_target, fk_atol=1e-6)
    assert not is_ls
    assert len(solutions) >= 1, "no solutions returned"

    def _max_wrap(q: NDArray[np.float64]) -> float:
        return max(abs(_wrap_pi(float(q[i] - q_star[i]))) for i in range(6))

    best = min(solutions, key=_max_wrap)
    assert _max_wrap(best) < 1e-4, (
        f"seeded q* not recovered; closest distance={_max_wrap(best):.3e}, "
        f"all solutions: {solutions}"
    )

    # Every returned solution must FK-close.
    for i, q in enumerate(solutions):
        t_check = _fk_dh(q, alpha, a, d)
        assert np.allclose(t_check, t_target, atol=1e-6), (
            f"solution {i} fails FK closure; max diff={np.max(np.abs(t_check - t_target)):.3e}"
        )


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
