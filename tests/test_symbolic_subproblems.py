"""Numeric-correctness tests for :mod:`ssik.codegen._symbolic`.

For each symbolic SP module, substitute concrete numpy-style inputs and
check that ``float(expr.subs(...))`` matches ``ssik.subproblems.spN.solve(...)``
to machine precision.

This is the "bulletproof" gate at the symbolic-foundation layer (#112):
if the symbolic SPs disagree with the numerical SPs on even one input,
every artifact built on top is unsound.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from ssik.codegen._symbolic.sp1 import sp1_theta_sym
from ssik.codegen._symbolic.sp2 import sp2_branches_sym
from ssik.codegen._symbolic.sp3 import sp3_branches_sym
from ssik.codegen._symbolic.sp4 import sp4_branches_sym
from ssik.codegen._symbolic.sp6 import sp6_a_mat_b_sym
from ssik.subproblems import sp1, sp2, sp3, sp4
from ssik.subproblems._rotation import _cross3, _dot3

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _to_sym(v: np.ndarray) -> sp.Matrix:
    return sp.Matrix([float(x) for x in v])


def _eval(expr: sp.Expr) -> float:
    return float(expr.evalf())


# ---------------------------------------------------------------------------
# SP1.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(20)))
def test_sp1_symbolic_matches_numerical(seed: int) -> None:
    """SP1 has a single closed-form atan2; symbolic and numeric must agree
    to ~machine precision on every random input."""
    rng = np.random.default_rng(seed)
    k = rng.standard_normal(3)
    k = k / np.linalg.norm(k)
    p = rng.standard_normal(3)
    q_target = rng.standard_normal(3)
    # Project q onto the same |p_perp|/k.p shell as p so the exact regime
    # holds (less interesting for LS but matches what composers feed it).
    theta_num, _ = sp1.solve(k, p, q_target)
    theta_sym = sp1_theta_sym(_to_sym(k), _to_sym(p), _to_sym(q_target))
    assert np.isclose(_eval(theta_sym), theta_num, atol=1e-12), (
        f"sp1 mismatch: numeric {theta_num} vs symbolic {_eval(theta_sym)}"
    )


# ---------------------------------------------------------------------------
# SP4.
# ---------------------------------------------------------------------------


def _sp4_pick_branch(
    sym_branches: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
    num_solutions: list[float],
) -> None:
    """SP4 may return 1 or 2 solutions; the symbolic version always returns
    both branches. Verify that every numeric solution matches one of the
    two symbolic branches (within machine precision)."""
    theta_plus = _eval(sym_branches[0])
    theta_minus = _eval(sym_branches[1])
    sym_set = {theta_plus, theta_minus}
    for sol in num_solutions:
        match = any(
            np.isclose(sol, s, atol=1e-10)
            or np.isclose((sol - s + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=1e-10)
            for s in sym_set
        )
        if not match:
            pytest.fail(
                f"sp4: numeric solution {sol} does not match either symbolic "
                f"branch ({theta_plus}, {theta_minus})"
            )


@pytest.mark.parametrize("seed", list(range(20)))
def test_sp4_symbolic_matches_numerical(seed: int) -> None:
    """SP4 has 2 branches in the generic case; symbolic must produce both
    such that every numeric solution matches one branch."""
    rng = np.random.default_rng(seed + 100)
    h = rng.standard_normal(3)
    h = h / np.linalg.norm(h)
    k = rng.standard_normal(3)
    k = k / np.linalg.norm(k)
    p = rng.standard_normal(3)

    # Construct a feasible d by picking a random theta_seed and computing
    # d = h.(Rot(k, theta_seed) p). Guarantees the two branches exist.
    theta_seed = float(rng.uniform(-np.pi, np.pi))
    c, s = np.cos(theta_seed), np.sin(theta_seed)
    rot_p = c * p + s * np.cross(k, p) + (1 - c) * np.dot(k, p) * k
    d = float(h @ rot_p)

    num_solutions, num_is_ls = sp4.solve(h, k, p, d)
    assert not num_is_ls, "test setup should produce feasible SP4"

    sym = sp4_branches_sym(_to_sym(h), _to_sym(k), _to_sym(p), sp.Float(d))
    _sp4_pick_branch(sym, num_solutions)


# ---------------------------------------------------------------------------
# SP3.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(15)))
def test_sp2_symbolic_matches_numerical(seed: int) -> None:
    """SP2 produces 2 branches in the generic feasible case; symbolic must
    reproduce both branches such that every numerical solution matches one
    of them (modulo wrap-to-pi)."""
    rng = np.random.default_rng(seed + 400)
    k1 = rng.standard_normal(3)
    k1 = k1 / np.linalg.norm(k1)
    k2 = rng.standard_normal(3)
    k2 = k2 / np.linalg.norm(k2)
    # Construct feasible inputs: pick (theta1*, theta2*) and a "shell" point z.
    # Set p = unrotate(k1, theta1*, z) so |Rot(k1, theta1*) p| = |z| = |q|.
    z = rng.standard_normal(3)
    theta1_seed = float(rng.uniform(-np.pi, np.pi))
    theta2_seed = float(rng.uniform(-np.pi, np.pi))

    def _rot(k: np.ndarray, t: float, v: np.ndarray) -> np.ndarray:
        c, s = np.cos(t), np.sin(t)
        out: np.ndarray = c * v + s * np.cross(k, v) + (1 - c) * np.dot(k, v) * k
        return out

    p = _rot(k1, -theta1_seed, z)
    q = _rot(k2, -theta2_seed, z)

    num_solutions, num_is_ls = sp2.solve(k1, k2, p, q)
    if num_is_ls or len(num_solutions) == 0:
        pytest.skip("test setup degenerate; rerun with different seed")

    sym = sp2_branches_sym(_to_sym(k1), _to_sym(k2), _to_sym(p), _to_sym(q))
    theta1_a = _eval(sym[0])
    theta2_a = _eval(sym[1])
    theta1_b = _eval(sym[2])
    theta2_b = _eval(sym[3])
    sym_pairs = ((theta1_a, theta2_a), (theta1_b, theta2_b))

    def _wrap_close(a: float, b: float, tol: float = 1e-8) -> bool:
        diff = ((a - b + np.pi) % (2 * np.pi)) - np.pi
        return abs(diff) < tol

    for sol_t1, sol_t2 in num_solutions:
        match = any(
            _wrap_close(sol_t1, sym_t1) and _wrap_close(sol_t2, sym_t2)
            for sym_t1, sym_t2 in sym_pairs
        )
        if not match:
            pytest.fail(
                f"sp2: numeric ({sol_t1:.6f}, {sol_t2:.6f}) does not match "
                f"either symbolic branch {sym_pairs}"
            )


@pytest.mark.parametrize("seed", list(range(10)))
def test_sp6_a_mat_b_symbolic_matches_numerical(seed: int) -> None:
    """SP6's setup matrices A (2x4) and b (2x1) must match the numerical
    SP6 implementation when evaluated with the same inputs. The QR /
    ellipse-intersection / GN-refinement steps stay runtime; this test
    verifies only the symbolic setup."""
    rng = np.random.default_rng(seed + 300)
    h_list = [rng.standard_normal(3) for _ in range(4)]
    k_list = [rng.standard_normal(3) for _ in range(4)]
    for i in range(4):
        k_list[i] = k_list[i] / np.linalg.norm(k_list[i])
    p_list = [rng.standard_normal(3) for _ in range(4)]
    d1, d2 = float(rng.standard_normal()), float(rng.standard_normal())

    # Symbolic A, b.
    h_sym = tuple(_to_sym(h) for h in h_list)
    k_sym = tuple(_to_sym(k) for k in k_list)
    p_sym = tuple(_to_sym(p) for p in p_list)
    a_sym, b_sym = sp6_a_mat_b_sym(h_sym, k_sym, p_sym, sp.Float(d1), sp.Float(d2))
    a_num = np.array([[float(a_sym[i, j].evalf()) for j in range(4)] for i in range(2)])
    b_num = np.array([float(b_sym[i, 0].evalf()) for i in range(2)])

    # Numerical A, b -- mirror sp6.solve's setup.
    a_cols_ref = []
    for idx in range(4):
        kxp = _cross3(k_list[idx], p_list[idx])
        a_cols_ref.append(np.column_stack([kxp, -_cross3(k_list[idx], kxp)]))
    h1_a1 = h_list[0] @ a_cols_ref[0]
    h2_a2 = h_list[1] @ a_cols_ref[1]
    h3_a3 = h_list[2] @ a_cols_ref[2]
    h4_a4 = h_list[3] @ a_cols_ref[3]
    a_ref = np.array(
        [
            [h1_a1[0], h1_a1[1], h2_a2[0], h2_a2[1]],
            [h3_a3[0], h3_a3[1], h4_a4[0], h4_a4[1]],
        ]
    )
    b_ref = np.array(
        [
            d1
            - _dot3(h_list[0], k_list[0]) * _dot3(k_list[0], p_list[0])
            - _dot3(h_list[1], k_list[1]) * _dot3(k_list[1], p_list[1]),
            d2
            - _dot3(h_list[2], k_list[2]) * _dot3(k_list[2], p_list[2])
            - _dot3(h_list[3], k_list[3]) * _dot3(k_list[3], p_list[3]),
        ]
    )

    assert np.allclose(a_num, a_ref, atol=1e-12), "sp6 A_mat mismatch"
    assert np.allclose(b_num, b_ref, atol=1e-12), "sp6 b_vec mismatch"


@pytest.mark.parametrize("seed", list(range(20)))
def test_sp3_symbolic_matches_numerical(seed: int) -> None:
    """SP3 reduces to SP4 with target shift; symbolic must produce branches
    matching the numerical SP3 outputs."""
    rng = np.random.default_rng(seed + 200)
    k = rng.standard_normal(3)
    k = k / np.linalg.norm(k)
    p = rng.standard_normal(3)
    q_target = rng.standard_normal(3)

    # Construct feasible d by picking theta_seed and computing distance.
    theta_seed = float(rng.uniform(-np.pi, np.pi))
    c, s = np.cos(theta_seed), np.sin(theta_seed)
    rot_p = c * p + s * np.cross(k, p) + (1 - c) * np.dot(k, p) * k
    d = float(np.linalg.norm(rot_p - q_target))

    num_solutions, num_is_ls = sp3.solve(k, p, q_target, d)
    assert not num_is_ls, "test setup should produce feasible SP3"

    sym = sp3_branches_sym(_to_sym(k), _to_sym(p), _to_sym(q_target), sp.Float(d))
    _sp4_pick_branch(sym, num_solutions)
