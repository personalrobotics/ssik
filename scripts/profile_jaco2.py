"""cProfile + per-stage timing for solve_all_ik on JACO 2.

Identifies the hot spots in the per-IK pipeline so we know where to spend
optimization effort. Run after the cache is warm (so per-arm setup doesn't
dominate).

    uv run python -u scripts/profile_jaco2.py

Sections:
  1. Per-stage timing (FK build, eliminate_q0_q1, weierstrass, build_M,
     eigendecomp, back_substitute, FK validation, LM polish)
  2. cProfile top-30 by cumtime, then by tottime
  3. Per-step time over 100 random poses, report distribution per stage
"""

from __future__ import annotations

import cProfile
import functools
import pstats
import time
from io import StringIO

import numpy as np

print = functools.partial(print, flush=True)

from ssik.solvers.ikgeo._raghavan_roth import (
    _cached_best_leftvar,
    _fk_dh,
    _newton_refine,
    back_substitute,
    build_m_matrix,
    build_pq,
    eliminate_q0_q1,
    solve_all_ik,
    solve_x2_roots_mobius,
    weierstrass_eliminate_trig,
)


def _dh_matrix(theta, alpha, a, d):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0., sa, ca, d], [0., 0., 0., 1.]])


def _fk(q, alpha, a, d):
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh_matrix(q[i], alpha[i], a[i], d[i])
    return T


def main() -> None:
    alpha = np.array([np.pi/2, np.pi, np.pi/2, 60*np.pi/180, 60*np.pi/180, np.pi])
    a     = np.array([0.0, 0.41, 0.0, 0.0, 0.0, 0.0])
    d     = np.array([0.2755, 0.0, -0.0098, -0.2502, -0.0858, -0.2116])
    dh = (alpha, a, d)

    # Warm cache
    print("warming cache...")
    q_warm = np.array([0.3, -0.5, 0.7, 0.4, -0.6, 0.2])
    T_warm = _fk(q_warm, alpha, a, d)
    t0 = time.time()
    sols, _ = solve_all_ik(dh, T_warm, fk_atol=1e-9, linearity_joint='auto')
    print(f"cold-cache: {time.time()-t0:.1f}s, {len(sols)} solutions")

    # Per-stage timing
    print("\n=== per-stage timing (warm cache, 100 random poses) ===")
    rng = np.random.default_rng(42)
    stage_times: dict[str, list[float]] = {
        "build_pq": [], "eliminate_q0_q1": [], "weierstrass": [],
        "build_M": [], "eigenvalue": [], "back_substitute_total": [],
        "fk_validate_total": [], "lm_polish_total": [], "total": [],
    }
    leftvar = _cached_best_leftvar(tuple(alpha.tolist()), tuple(a.tolist()), tuple(d.tolist()))
    print(f"cached leftvar: q_{leftvar}")

    for _ in range(100):
        q_random = rng.uniform(-1, 1, size=6)
        T_target = _fk(q_random, alpha, a, d)

        t_total_start = time.perf_counter()

        t = time.perf_counter()
        p_sin, p_cos, p_one, q_mat, meta = build_pq(
            dh, T_target, linearity_joint=leftvar, return_metadata=True
        )
        stage_times["build_pq"].append(time.perf_counter() - t)

        t = time.perf_counter()
        e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
        stage_times["eliminate_q0_q1"].append(time.perf_counter() - t)

        t = time.perf_counter()
        e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
        stage_times["weierstrass"].append(time.perf_counter() - t)

        t = time.perf_counter()
        m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)
        stage_times["build_M"].append(time.perf_counter() - t)

        t = time.perf_counter()
        roots, eigvecs = solve_x2_roots_mobius(m_quad, m_lin, m_const)
        stage_times["eigenvalue"].append(time.perf_counter() - t)

        t_bs = 0.0
        t_fk = 0.0
        t_lm = 0.0
        for r, ev in zip(roots, eigvecs):
            t = time.perf_counter()
            q_cand = back_substitute(r, ev, p_sin, p_cos, p_one, q_mat, dh, T_target, metadata=meta)
            t_bs += time.perf_counter() - t
            if q_cand is None:
                continue
            t = time.perf_counter()
            t_check = _fk_dh(q_cand, dh)
            fk_err = float(np.linalg.norm(t_check - T_target))
            t_fk += time.perf_counter() - t
            if fk_err > 1e-9:
                t = time.perf_counter()
                _newton_refine(q_cand, dh, T_target, fk_atol=1e-9)
                t_lm += time.perf_counter() - t
        stage_times["back_substitute_total"].append(t_bs)
        stage_times["fk_validate_total"].append(t_fk)
        stage_times["lm_polish_total"].append(t_lm)

        stage_times["total"].append(time.perf_counter() - t_total_start)

    # Report
    print(f"\n{'stage':<26} {'min ms':>9} {'median ms':>11} {'mean ms':>11} {'p95 ms':>9} {'max ms':>9}")
    print("-" * 80)
    for name, ts in stage_times.items():
        ts_arr = np.array(ts) * 1000
        print(f"{name:<26} {ts_arr.min():>9.3f} {np.median(ts_arr):>11.3f} "
              f"{ts_arr.mean():>11.3f} {np.percentile(ts_arr, 95):>9.3f} {ts_arr.max():>9.3f}")

    # cProfile run for top-time identification
    print("\n=== cProfile top-30 by cumtime (50 iters) ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        q_random = rng.uniform(-1, 1, size=6)
        T_target = _fk(q_random, alpha, a, d)
        solve_all_ik(dh, T_target, fk_atol=1e-9, linearity_joint='auto')
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(30)
    print(s.getvalue())


if __name__ == "__main__":
    main()
