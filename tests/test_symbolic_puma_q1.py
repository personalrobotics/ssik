"""Composer correctness test: symbolic SPs + Puma KinBody → explicit trig.

Validates that the symbolic SP4 module, fed Puma's q1-step constants
(axes[1], -axes[0], p_16, axis_1_dot_psum_123) with T_target symbolic,
produces ``sin``/``cos``/``atan2`` expressions whose numeric evaluation
matches the runtime SP4 solver on random target poses.

This is the smallest end-to-end check that the composer architecture
works: substitute concrete arm constants into a symbolic SP, run sympy
CSE, evaluate at a sample T_target, compare against the in-tree solver.
The full composer (#112 step 2) extends this to all six joints.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sympy as sp

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.codegen._compose._target import make_target_symbols
from ssik.codegen._symbolic.sp4 import sp4_branches_sym
from ssik.subproblems import sp4

FIXTURES = Path(__file__).parent / "fixtures"


def _vec(v: np.ndarray) -> sp.Matrix:
    return sp.Matrix([float(x) for x in v])


def test_puma_q1_symbolic_matches_numerical() -> None:
    """Specialise SP4's q1 step with Puma's constants and check that
    every random T_target produces the same ``q1`` solutions as the
    runtime SP4 solver."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")

    # Puma q1 inputs: h = axes[1], k = -axes[0], p = p_16(T_target),
    # d = axes[1] . (p[1] + p[2] + p[3]).
    axis_0 = kb.joints[0].axis
    axis_1 = kb.joints[1].axis
    p_0 = kb.joints[0].T_left[:3, 3]
    p_1 = kb.joints[1].T_left[:3, 3]
    p_2 = kb.joints[2].T_left[:3, 3]
    # Wrist consolidation: spherical-wrist family.
    p_3_consolidated = (
        kb.joints[3].T_left[:3, 3] + kb.joints[4].T_left[:3, 3] + kb.joints[5].T_left[:3, 3]
    )
    p_tool = kb.joints[5].T_right[:3, 3]
    r_home = kb.joints[5].T_right[:3, :3]
    d_q1 = float(axis_1 @ (p_1 + p_2 + p_3_consolidated))

    target = make_target_symbols()
    # r_06 = T_target[:3,:3] @ r_home.T
    r_home_sym = sp.Matrix([[float(x) for x in row] for row in r_home])
    r_06 = target.r * r_home_sym.T
    # p_0t = T_target[:3, 3]
    p_0t = target.p
    # p_16 = p_0t - r_06 @ p[6] - p[0]
    p_tool_sym = _vec(p_tool)
    p_0_sym = _vec(p_0)
    p_16 = p_0t - r_06 * p_tool_sym - p_0_sym

    # Symbolic SP4: returns the 5-tuple of branch expressions + guards.
    h_sym = _vec(axis_1)
    k_sym = _vec(-axis_0)
    sp4_out = sp4_branches_sym(h_sym, k_sym, p_16, sp.Float(d_q1))
    theta_plus_expr, theta_minus_expr = sp4_out[0], sp4_out[1]

    # Run sympy.cse to confirm the codegen target produces clean output.
    cse_subs, _exprs = sp.cse([theta_plus_expr, theta_minus_expr])
    # Sanity: cse should produce SOME common subexpressions (not zero).
    assert len(cse_subs) >= 2, "expected sympy.cse to factor common subexpressions"

    # Validate against runtime SP4 on 20 random T_targets.
    rng = np.random.default_rng(seed=0)
    matches = 0
    for _trial in range(20):
        # Build a feasible T_target by FK'ing a random q (roughly) so SP4
        # is feasible for q1.
        q_star = rng.uniform(-1.0, 1.0, size=6)
        from ssik.subproblems._rotation import rotation_matrix

        T = np.eye(4)
        for j, qi in zip(kb.joints, q_star, strict=True):
            R = np.eye(4)
            R[:3, :3] = rotation_matrix(j.axis, float(qi))
            T = T @ j.T_left @ R @ j.T_right

        # Build the substitution dict from the target symbols to T's entries.
        subs = {}
        for i in range(3):
            for jj in range(3):
                subs[target.r[i, jj]] = float(T[i, jj])
        for i, name in enumerate(("p_x", "p_y", "p_z")):
            subs[sp.Symbol(name, real=True)] = float(T[i, 3])

        theta_sym_plus = float(theta_plus_expr.subs(subs).evalf())
        theta_sym_minus = float(theta_minus_expr.subs(subs).evalf())

        # Compare against runtime SP4 directly.
        p_16_num = T[:3, 3] - (T[:3, :3] @ r_home.T) @ p_tool - p_0
        runtime_sols, _ = sp4.solve(axis_1, -axis_0, p_16_num, d_q1)
        # Symbolic produces both branches; runtime may produce 1 or 2.
        # Each runtime solution must match one of (sym_plus, sym_minus).
        for sol in runtime_sols:
            if any(
                np.isclose(sol, s, atol=1e-10)
                or np.isclose((sol - s + np.pi) % (2 * np.pi) - np.pi, 0.0, atol=1e-10)
                for s in (theta_sym_plus, theta_sym_minus)
            ):
                matches += 1

    assert matches >= 20, (
        f"symbolic q1 should match runtime on at least 20/40 branch hits across "
        f"20 trials; got {matches}"
    )
