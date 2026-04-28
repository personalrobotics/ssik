"""JACO 2 benchmark for analytical exhaustion (#67 / AE-1..AE-6).

Run after each AE change to measure conditioning + closure improvements:

    uv run python scripts/bench_jaco2.py

Reports per stage:
- cond(m_quad)               -- the structural conditioning of the pencil
- num eigenvalue roots        -- how many real x_2 candidates the eigensolver finds
- num back_substitute survivors  -- candidates that pass the cross-checks
- best FK error (no refinement) -- pure-algebraic residual for the closest-to-q* candidate
- best FK error (LM refinement) -- after scipy LM polish
- best wrap-pi distance to seeded q*

JACO 2 (Kinova j2n6s200) standard DH parameters: 60-deg non-orthogonal
twists at joints 4-5 are the defining structural feature of the j2n6s2*
family. Values typed from the standard published Kinova DH table; will be
validated against `robot-code/ada_assets/.../jaco2.xml` when POE\u2192DH (#79)
lands.
"""

from __future__ import annotations

import functools
import time

print = functools.partial(print, flush=True)  # stream output as it happens

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from ssik.solvers.ikgeo._raghavan_roth import (  # noqa: E402
    _derive_pq_for_arm,
    _fk_dh,
    _newton_refine,
    back_substitute,
    build_m_matrix,
    build_pq,
    eliminate_q0_q1,
    solve_x2_roots_mobius,
    weierstrass_eliminate_trig,
)


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


def _fk(q: NDArray[np.float64], alpha, a, d) -> NDArray[np.float64]:
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh_matrix(q[i], alpha[i], a[i], d[i])
    return T


def _wrap_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def _max_diff(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> float:
    return max(abs(_wrap_pi(float(q1[i] - q2[i]))) for i in range(6))


def main() -> None:
    # JACO 2 (Kinova j2n6s200) standard published DH
    alpha = np.array([np.pi / 2, np.pi, np.pi / 2, 60 * np.pi / 180, 60 * np.pi / 180, np.pi])
    a = np.array([0.0, 0.41, 0.0, 0.0, 0.0, 0.0])
    d = np.array([0.2755, 0.0, -0.0098, -0.2502, -0.0858, -0.2116])

    # Seeded q* for repeatable benchmark
    q_star = np.array([0.3, -0.5, 0.7, 0.4, -0.6, 0.2])
    t_target = _fk(q_star, alpha, a, d)

    print("=" * 70)
    print("JACO 2 benchmark")
    print("=" * 70)
    print(f"q* = {q_star}")
    print(f"alpha = {alpha}  (60deg twists at joints 4,5)")
    print()

    # Stage 1: build (P, Q)
    print("building (P, Q) symbolically (cached after first call; expect 30-100s on cold cache)...")
    t0 = time.time()
    p_sin, p_cos, p_one, q_mat = build_pq((alpha, a, d), t_target)
    t_pq = time.time() - t0
    print(f"build_pq:                    {t_pq:6.2f}s  (cached after first call)")

    # Stage 2: eliminate (q_0, q_1)
    t0 = time.time()
    e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
    print(f"eliminate_q0_q1:             {time.time()-t0:6.2f}s  -> 6x9 E system")

    # Stage 3: Weierstrass for q_2 + W transform for (q_3, q_4)
    t0 = time.time()
    e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
    print(f"weierstrass_eliminate_trig:  {time.time()-t0:6.2f}s  -> 6x9 quadratic-in-x_2")

    # Stage 4: build 12x12 M(x_2)
    t0 = time.time()
    m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)
    cond_a = float(np.linalg.cond(m_quad))
    cond_b = float(np.linalg.cond(m_lin))
    cond_c = float(np.linalg.cond(m_const))
    print(f"build_m_matrix:              {time.time()-t0:6.2f}s")
    print(f"  cond(A=m_quad)  = {cond_a:.3e}")
    print(f"  cond(B=m_lin)   = {cond_b:.3e}")
    print(f"  cond(C=m_const) = {cond_c:.3e}")

    # AE-1: log equilibrated cond
    from ssik.solvers.ikgeo._raghavan_roth import _equilibrate_pencil
    a_eq, _b_eq, _c_eq, _, _ = _equilibrate_pencil(m_quad, m_lin, m_const)
    print(f"  cond(A_eq) [AE-1]   = {np.linalg.cond(a_eq):.3e}  "
          f"(\u00d7 {cond_a / np.linalg.cond(a_eq):.2e} reduction)")

    # AE-3 (#70): try alternative leftvar choices (linearity_joint = 0, 1, 2).
    # AE-4 (#71): also test SO(3) reduction on the best leftvar.

    def _build_at_leftvar(linearity_joint: int, apply_so3: bool):
        fns = _derive_pq_for_arm(
            tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist()),
            linearity_joint=linearity_joint, apply_so3=apply_so3,
        )
        p_sin_fn, p_cos_fn, p_one_fn, q_fn, _meta = fns
        args = [*t_target[0, :].tolist(), *t_target[1, :].tolist(), *t_target[2, :].tolist()]
        return (
            np.asarray(p_sin_fn(*args), dtype=np.float64),
            np.asarray(p_cos_fn(*args), dtype=np.float64),
            np.asarray(p_one_fn(*args), dtype=np.float64),
            np.asarray(q_fn(*args), dtype=np.float64),
        )

    print("\n--- AE-3 alternative leftvars (no SO(3)) ---")
    for lj in (0, 1, 2):
        if lj == 2:
            ps, pc, po, qm = p_sin, p_cos, p_one, q_mat  # already computed
        else:
            t0 = time.time()
            ps, pc, po, qm = _build_at_leftvar(lj, apply_so3=False)
            print(f"  build_pq(linearity={lj}): {time.time()-t0:5.1f}s")
        es, ec, eo = eliminate_q0_q1(ps, pc, po, qm)
        eq, el, ec_const = weierstrass_eliminate_trig(es, ec, eo)
        mq, _ml, _mc = build_m_matrix(eq, el, ec_const)
        cond_lj = float(np.linalg.cond(mq))
        print(f"  linearity={lj}: cond(A) = {cond_lj:.3e}  "
              f"(vs baseline {cond_a:.3e}: \u00d7 {cond_a / cond_lj:.2e})")
    print()

    # Full pipeline at linearity=q_1 -- expect pure-algebraic FK closure
    print("--- Full pipeline at linearity=q_1 (pure algebraic) ---")
    from ssik.solvers.ikgeo._raghavan_roth import solve_all_ik
    t0 = time.time()
    sols_q1, is_ls_q1 = solve_all_ik(
        (alpha, a, d), t_target,
        fk_atol=1e-9,  # tight target -- expect pass with no LM polish
        linearity_joint=1,
    )
    elapsed_q1 = time.time() - t0
    print(
        f"solve_all_ik(linearity=1):   {elapsed_q1:6.2f}s  -> "
        f"{len(sols_q1)} solutions, is_ls={is_ls_q1}"
    )
    if sols_q1:
        best_q1 = min(sols_q1, key=lambda s: _max_diff(s.q, q_star))
        print(f"  best |q-q*|: {_max_diff(best_q1.q, q_star):.3e}")
        fk_errs_q1 = [
            float(np.linalg.norm(_fk_dh(s.q, (alpha, a, d)) - t_target))
            for s in sols_q1
        ]
        print(
            f"  FK errors: max={max(fk_errs_q1):.3e}, "
            f"all<1e-9: {all(e<1e-9 for e in fk_errs_q1)}"
        )
    print()

    # Stage 5: eigenvalue route
    t0 = time.time()
    roots, eigvecs = solve_x2_roots_mobius(m_quad, m_lin, m_const)
    t_eig = time.time() - t0
    x2_star = float(np.tan(q_star[2] / 2.0))
    closest_root = min(roots, key=lambda r: abs(r - x2_star)) if roots else None
    print(f"solve_x2_roots_mobius:       {t_eig:6.2f}s  -> {len(roots)} real roots")
    if closest_root is not None:
        print(f"  closest root to tan(q_2*/2)={x2_star:.4f}: {closest_root:.4f}  "
              f"(error {abs(closest_root - x2_star):.3e})")

    # Stage 6: back-substitution + FK validation, no refinement
    t0 = time.time()
    candidates_alg = []
    for r, ev in zip(roots, eigvecs, strict=False):
        q_cand = back_substitute(r, ev, p_sin, p_cos, p_one, q_mat, (alpha, a, d), t_target)
        if q_cand is None:
            continue
        fk_err = float(np.linalg.norm(_fk_dh(q_cand, (alpha, a, d)) - t_target))
        candidates_alg.append((q_cand, fk_err))
    t_alg = time.time() - t0
    print(f"back_substitute (no LS):     {t_alg:6.2f}s  -> {len(candidates_alg)} survivors")
    if candidates_alg:
        best_alg = min(candidates_alg, key=lambda c: _max_diff(c[0], q_star))
        print(f"  best |q-q*| (algebraic only): {_max_diff(best_alg[0], q_star):.3e}")
        print(f"  best FK err  (algebraic only): {min(c[1] for c in candidates_alg):.3e}")

    # Stage 7: back-sub + LM refinement (FK-tolerance-driven termination)
    t0 = time.time()
    candidates_ref = []
    iters_per_cand = []
    for r, ev in zip(roots, eigvecs, strict=False):
        q_cand = back_substitute(r, ev, p_sin, p_cos, p_one, q_mat, (alpha, a, d), t_target)
        if q_cand is None:
            continue
        result = _newton_refine(q_cand, (alpha, a, d), t_target, fk_atol=1e-9, max_iters=30)
        if result is None:
            continue
        q_ref, iters = result
        iters_per_cand.append(iters)
        fk_err = float(np.linalg.norm(_fk_dh(q_ref, (alpha, a, d)) - t_target))
        candidates_ref.append((q_ref, fk_err))
    t_ref = time.time() - t0
    print(f"back_substitute + LM:        {t_ref:6.2f}s  -> {len(candidates_ref)} survivors")
    if iters_per_cand:
        print(f"  LM iterations per candidate: {iters_per_cand}")
    if candidates_ref:
        best_ref = min(candidates_ref, key=lambda c: _max_diff(c[0], q_star))
        print(f"  best |q-q*| (with LM):        {_max_diff(best_ref[0], q_star):.3e}")
        print(f"  best FK err  (with LM):       {min(c[1] for c in candidates_ref):.3e}")

    print()
    print(f"VERDICT (algebraic-only): "
          f"{'PASS' if candidates_alg and min(c[1] for c in candidates_alg) < 1e-9 else 'FAIL'} "
          f"@ fk_atol=1e-9")
    print(f"VERDICT (with LM polish): "
          f"{'PASS' if candidates_ref and min(c[1] for c in candidates_ref) < 1e-9 else 'FAIL'} "
          f"@ fk_atol=1e-9")


if __name__ == "__main__":
    main()
