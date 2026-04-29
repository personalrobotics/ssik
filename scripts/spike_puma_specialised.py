"""Spike for #112: hand-specialised IK module for Puma 560.

Two halves:

1. ``solve_specialised(T)`` -- spherical_two_parallel for Puma with KinBody
   constants baked in as Python floats / numpy literals. SP1/SP3/SP4 still
   imported (those get inlined by step 1-2 of #112). FK is fully inlined.
   This is the structural target of the codegen pipeline.

2. ``demo_sympy_sp4_substituted()`` -- show what step 1-2 will produce by
   running sympy on SP4's closed-form with Puma's q1-step constants
   substituted. The output is explicit trig in the target-pose entries --
   the IKFast paradigm. Demonstrates feasibility of source-level
   specialisation.

Goals validated by this spike:

- The specialised structure benches at least as fast as the current path
  in pure Python. (Speed parity claim from #112.)
- 100 random poses round-trip correctly through the specialised solver
  (FK closure < 1e-9). (Bulletproof.)
- Sympy can reduce SP4 with Puma's constants to a finite trig expression.
  (Proves the codegen pipeline's central operation is feasible.)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ssik._urdf import load_urdf_kinbody_normalized  # noqa: E402
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY  # noqa: E402
from ssik.solvers.ikgeo import spherical_two_parallel  # noqa: E402
from ssik.subproblems import sp1, sp3, sp4  # noqa: E402

# ===========================================================================
# Hand-specialised Puma 560 spherical_two_parallel.
#
# All KinBody constants are inlined as Python floats / np.array literals.
# No KinBody object is constructed at runtime. SP1/SP3/SP4 calls remain
# (they get sympy-inlined in step 1-2 of #112).
# ===========================================================================

# Joint axes in the base frame at q=0 (POE-normalised). From Puma 560 URDF.
_AXIS_0 = np.array([0.0, 0.0, 1.0])
_AXIS_1 = np.array([0.0, -1.0, 0.0])
_AXIS_2 = np.array([0.0, -1.0, 0.0])  # parallel to axis_1
_AXIS_3 = np.array([0.0, 0.0, 1.0])
_AXIS_4 = np.array([0.0, -1.0, 0.0])
_AXIS_5 = np.array([0.0, 0.0, 1.0])
_NEG_AXIS_0 = -_AXIS_0
_NEG_AXIS_2 = -_AXIS_2
_NEG_AXIS_5 = -_AXIS_5

# Per-joint translation offsets (T_left[:3, 3]) and tool offset (T_right[5][:3, 3]).
_P_0 = np.array([0.0, 0.0, 0.0])
_P_1 = np.array([0.0, 0.0, 0.0])
_P_2 = np.array([0.4318, 0.0, 0.0])
# Wrist consolidation: spherical-wrist family expects p[3] to be the
# total joint-3 -> wrist-intersection translation. Puma's URDF splits
# this across T_left[3..5]; sum them at build time.
_P_3 = np.array([0.020299999999999985, -0.15005, 9.187912610603016e-18]) + np.array(
    [0.0, 0.0, 0.4318]
) + np.array([0.0, 0.0, 0.0])
_P_TOOL = np.array([0.0, 0.0, 0.0])

# Pre-computed scalars. SP4 for q1 takes ``d = axes[1] @ (p[1] + p[2] + p[3])``.
_AXIS_1_DOT_PSUM_123 = float(_AXIS_1 @ (_P_1 + _P_2 + _P_3))

# Home-pose rotation. Identity for Puma.
_R_HOME = np.eye(3)

_POLICY = DEFAULT_TOLERANCE_POLICY


def _rotation_matrix(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    """Inlined per-axis rotation matrix. Constants would substitute further
    if axis were known statically."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc],
        ],
        dtype=np.float64,
    )


def _fk_specialised(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inlined POE forward kinematics for Puma 560.

    Six joints; T_left and T_right (except T_right[5]) are pure
    translations. T_right[5] is identity. r_home is identity.
    """
    T = np.eye(4)
    # Joint 0
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ _P_0
    R = _rotation_matrix(_AXIS_0, float(q[0]))
    T[:3, :3] = T[:3, :3] @ R
    # Joint 1
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ _P_1
    R = _rotation_matrix(_AXIS_1, float(q[1]))
    T[:3, :3] = T[:3, :3] @ R
    # Joint 2
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ _P_2
    R = _rotation_matrix(_AXIS_2, float(q[2]))
    T[:3, :3] = T[:3, :3] @ R
    # Joint 3 (we still go through joints 3, 4, 5 separately for FK; the
    # spherical-wrist consolidation is only used by the IK)
    p3_orig = np.array([0.020299999999999985, -0.15005, 9.187912610603016e-18])
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ p3_orig
    R = _rotation_matrix(_AXIS_3, float(q[3]))
    T[:3, :3] = T[:3, :3] @ R
    # Joint 4
    p4_orig = np.array([0.0, 0.0, 0.4318])
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ p4_orig
    R = _rotation_matrix(_AXIS_4, float(q[4]))
    T[:3, :3] = T[:3, :3] @ R
    # Joint 5
    R = _rotation_matrix(_AXIS_5, float(q[5]))
    T[:3, :3] = T[:3, :3] @ R
    # No tool offset (P_TOOL = 0).
    return T


def solve_specialised(T_target: NDArray[np.float64]) -> tuple[list[NDArray[np.float64]], bool]:
    """Inverse kinematics for Puma 560, hand-specialised spherical_two_parallel.

    Mirrors :func:`ssik.solvers.ikgeo.spherical_two_parallel.solve` but with
    KinBody attribute access removed and all constants inlined. Calls into
    SP1/SP3/SP4 -- those get sympy-inlined by step 1-2 of #112.
    """
    t_target = np.asarray(T_target, dtype=np.float64)
    # r_home = I, so r_06 = t_target[:3,:3] @ I.T = t_target[:3,:3]
    r_06 = t_target[:3, :3]
    p_0t = t_target[:3, 3]

    # SP4 for q1.
    p_16 = p_0t - r_06 @ _P_TOOL - _P_0  # = p_0t (since p_tool = 0, p_0 = 0)
    t1_solutions, _ = sp4.solve(_AXIS_1, _NEG_AXIS_0, p_16, _AXIS_1_DOT_PSUM_123, _POLICY)

    candidates: list[NDArray[np.float64]] = []
    for q1 in t1_solutions:
        shoulder = _rotation_matrix(_NEG_AXIS_0, q1) @ (-p_0t + r_06 @ _P_TOOL + _P_0) + _P_1

        t3_solutions, _ = sp3.solve(_AXIS_2, -_P_3, _P_2, float(np.linalg.norm(shoulder)), _POLICY)

        for q3 in t3_solutions:
            q2, _ = sp1.solve(
                _AXIS_1,
                -_P_2 - _rotation_matrix(_AXIS_2, q3) @ _P_3,
                shoulder,
                _POLICY,
            )

            r_36 = (
                _rotation_matrix(_NEG_AXIS_2, q3)
                @ _rotation_matrix(-_AXIS_1, q2)
                @ _rotation_matrix(_NEG_AXIS_0, q1)
                @ r_06
            )

            t5_solutions, _ = sp4.solve(_AXIS_3, _AXIS_4, _AXIS_5, float(_AXIS_3 @ r_36 @ _AXIS_5), _POLICY)

            for q5 in t5_solutions:
                q4, _ = sp1.solve(
                    _AXIS_3,
                    _rotation_matrix(_AXIS_4, q5) @ _AXIS_5,
                    r_36 @ _AXIS_5,
                    _POLICY,
                )
                q6, _ = sp1.solve(
                    _NEG_AXIS_5,
                    _rotation_matrix(-_AXIS_4, q5) @ _AXIS_3,
                    r_36.T @ _AXIS_3,
                    _POLICY,
                )
                candidates.append(np.array([q1, q2, q3, q4, q5, q6]))

    # Verify + dedup. FK closure threshold from policy.
    fk_atol = _POLICY.subproblem_numerical
    dedup_atol = _POLICY.subproblem_dedup
    accepted: list[tuple[NDArray[np.float64], float]] = []
    for q in candidates:
        T_check = _fk_specialised(q)
        residual = float(np.linalg.norm(T_check - t_target))
        if residual <= fk_atol:
            accepted.append((q, residual))

    # Wrap-to-pi dedup; keep lower-residual on collision.
    deduped: list[tuple[NDArray[np.float64], float]] = []
    for cand_q, cand_res in accepted:
        dup_idx = None
        for j, (existing_q, _) in enumerate(deduped):
            diffs = (cand_q - existing_q + np.pi) % (2 * np.pi) - np.pi
            if np.all(np.abs(diffs) < dedup_atol):
                dup_idx = j
                break
        if dup_idx is None:
            deduped.append((cand_q, cand_res))
        elif cand_res < deduped[dup_idx][1]:
            deduped[dup_idx] = (cand_q, cand_res)

    solutions = [q for q, _ in deduped]
    return solutions, len(solutions) == 0


# ===========================================================================
# Sympy demo: SP4 with Puma's q1-step constants substituted.
# Demonstrates step 1-2 of #112: explicit trig output.
# ===========================================================================


def demo_sympy_sp4_substituted() -> None:
    """Show what sympy-driven SP4 specialisation produces for Puma's q1 step.

    SP4 closed form: theta = atan2(B, A) +/- acos((d - C) / R)
    where A = h.p - (k.p)(h.k), B = h.(k x p), C = (k.p)(h.k), R = sqrt(A^2 + B^2).
    """
    print("\n=== SP4 specialisation demo (Puma q1 step) ===\n")

    # Inputs: h, k are Puma's q1-step constants (concrete vectors).
    # p depends on T_target (the user's input), so it's symbolic.
    # d is a concrete scalar.
    h = sp.Matrix(_AXIS_1.tolist())
    k = sp.Matrix(_NEG_AXIS_0.tolist())
    d = sp.Float(_AXIS_1_DOT_PSUM_123)

    # T_target entries treated as symbols; we keep just the wrist-center
    # input p_16 = p_0t (since p_tool=0, p_0=0 for Puma).
    px, py, pz = sp.symbols("p_x p_y p_z", real=True)
    p_sym = sp.Matrix([px, py, pz])

    # SP4 closed form, fully symbolic.
    hp = h.dot(p_sym)
    kp = k.dot(p_sym)
    hk = float(h.dot(k))
    A = hp - kp * hk
    B = h.dot(k.cross(p_sym))
    C = kp * hk
    R = sp.sqrt(A**2 + B**2)
    phi = sp.atan2(B, A)
    delta = sp.acos((d - C) / R)
    theta_plus = sp.simplify(phi + delta)
    theta_minus = sp.simplify(phi - delta)

    # Run sympy.cse to extract common subexpressions.
    cse_subs, exprs = sp.cse([theta_plus, theta_minus], optimizations="basic")

    print("Common subexpressions:")
    for sym, expr in cse_subs:
        print(f"    {sym} = {expr}")
    print("Outputs:")
    print(f"    theta_q1_branch_plus  = {exprs[0]}")
    print(f"    theta_q1_branch_minus = {exprs[1]}")

    print("\nRendered as Python code (the codegen target):")
    for sym, expr in cse_subs:
        print(f"    {sym} = {sp.pycode(expr)}")
    print(f"    theta_q1_plus  = {sp.pycode(exprs[0])}")
    print(f"    theta_q1_minus = {sp.pycode(exprs[1])}")


# ===========================================================================
# Bench + validation harness.
# ===========================================================================


def bench_and_validate() -> None:
    print("loading Puma 560 KinBody (for current-path baseline)...")
    kb = load_urdf_kinbody_normalized(REPO / "tests/fixtures/puma560.urdf", "base_link", "wrist_3_link")

    rng = np.random.default_rng(seed=0)
    n = 100

    # Warm.
    q_warm = rng.uniform(-1.0, 1.0, size=6)
    T_warm = _fk_specialised(q_warm)
    solve_specialised(T_warm)
    spherical_two_parallel.solve(kb, T_warm)

    # Validate FK closure on the SPECIALISED solver.
    print(f"\nvalidating specialised solver on {n} random poses...")
    fk_errs: list[float] = []
    fails = 0
    n_solutions: list[int] = []
    for _ in range(n):
        q_star = rng.uniform(-1.0, 1.0, size=6)
        T_star = _fk_specialised(q_star)
        sols, is_ls = solve_specialised(T_star)
        if is_ls or not sols:
            fails += 1
            continue
        n_solutions.append(len(sols))
        worst = max(float(np.linalg.norm(_fk_specialised(q) - T_star)) for q in sols)
        fk_errs.append(worst)
    print(
        f"  ✓ {n - fails}/{n} succeeded, "
        f"FK error median {np.median(fk_errs):.2e} max {max(fk_errs):.2e}, "
        f"solutions/pose median {int(np.median(n_solutions))}"
    )

    # Bench specialised vs current.
    print(f"\nbenching specialised vs current on {n} random poses (warm)...")
    rng = np.random.default_rng(seed=42)
    poses = [_fk_specialised(rng.uniform(-1.0, 1.0, size=6)) for _ in range(n)]

    t_specialised: list[float] = []
    for T in poses:
        t0 = time.perf_counter()
        solve_specialised(T)
        t_specialised.append((time.perf_counter() - t0) * 1e3)

    t_current: list[float] = []
    for T in poses:
        t0 = time.perf_counter()
        spherical_two_parallel.solve(kb, T)
        t_current.append((time.perf_counter() - t0) * 1e3)

    a_med = float(np.median(t_specialised))
    b_med = float(np.median(t_current))
    a_min = float(np.min(t_specialised))
    b_min = float(np.min(t_current))
    print(f"  specialised: min {a_min:.3f} ms, median {a_med:.3f} ms")
    print(f"  current:     min {b_min:.3f} ms, median {b_med:.3f} ms")
    print(f"  ratio (specialised / current) at min:    {a_min / b_min:.2f}x")
    print(f"  ratio (specialised / current) at median: {a_med / b_med:.2f}x")


if __name__ == "__main__":
    bench_and_validate()
    demo_sympy_sp4_substituted()
