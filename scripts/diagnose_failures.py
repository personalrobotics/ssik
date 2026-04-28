"""Rigorous failure-mode diagnostic for the Raghavan-Roth pipeline.

For each (arm, random q*) pair, classify every eigenvalue route candidate
into one of five outcomes:

  algebraic_pass    -- back_substitute candidate FK-closes immediately
  newton_pass       -- Newton refinement succeeded from algebraic seed
  newton_diverged   -- Newton diverged (likely spurious eigenvalue root)
  newton_no_converge - Newton hit cap without reaching fk_atol
  back_sub_none     -- back_substitute returned None (cross-check failed)

Per pose: succeeds if >=1 candidate ends in algebraic_pass or newton_pass.
Per arm: aggregate cond, candidate-outcome distribution, and the FK error
distribution at each stage to identify root causes of failure.

    uv run python -u scripts/diagnose_failures.py
"""

from __future__ import annotations

import functools
from collections import Counter

import numpy as np

print = functools.partial(print, flush=True)

from ssik.solvers.ikgeo._raghavan_roth import (  # noqa: E402
    _newton_refine,
    back_substitute,
    build_m_matrix,
    build_pq,
    eliminate_q0_q1,
    pick_best_leftvar,
    solve_x2_roots_mobius,
    weierstrass_eliminate_trig,
)


def _dh_matrix(theta, alpha, a, d):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0., sa, ca, d], [0., 0., 0., 1.]])  # noqa: E501


def _fk(q, alpha, a, d):
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh_matrix(q[i], alpha[i], a[i], d[i])
    return T


def diagnose_arm(name: str, alpha, a, d, *, n_poses: int = 50, fk_atol: float = 1e-6, q_range: float = 1.0) -> None:  # noqa: E501
    """Run diagnostic on `n_poses` random poses for a given arm."""
    print(f"\n{'='*78}")
    print(f"ARM: {name}")
    print(f"{'='*78}")
    print(f"alpha = {alpha}")
    print(f"a     = {a}")
    print(f"d     = {d}")

    dh = (alpha, a, d)
    rng = np.random.default_rng(0)

    # AE-3 leftvar selection
    print("\n--- AE-3 leftvar selection ---")
    best_lj, conds = pick_best_leftvar(dh)
    print(f"  best leftvar: q_{best_lj}")
    for lj in sorted(conds.keys()):
        marker = " <- chosen" if lj == best_lj else ""
        print(f"    cond(linearity={lj}) = {conds[lj]:.3e}{marker}")

    # Per-pose stats
    pose_outcomes: list[bool] = []           # whether pose has >=1 valid candidate
    candidate_class_counts: Counter[str] = Counter()
    fk_err_by_class: dict[str, list[float]] = {
        "algebraic_pass": [], "newton_pass": [], "newton_diverged": [],
        "newton_no_converge": [], "back_sub_none": []
    }
    seed_qdiff_by_class: dict[str, list[float]] = {
        "algebraic_pass": [], "newton_pass": [], "newton_diverged": [],
        "newton_no_converge": [],
    }
    newton_iters_by_class: dict[str, list[int]] = {
        "newton_pass": [], "newton_no_converge": [],
    }
    cond_at_solve: list[float] = []
    cond_b_at_solve: list[float] = []
    cond_c_at_solve: list[float] = []
    cond_sigma_at_solve: list[float] = []
    norm_ainvb: list[float] = []
    norm_ainvc: list[float] = []
    n_roots_per_pose: list[int] = []

    print(f"\n--- {n_poses} random poses (q* in [-{q_range}, {q_range}]) ---")
    for _pose_idx in range(n_poses):
        q_star = rng.uniform(-q_range, q_range, size=6)
        T_target = _fk(q_star, alpha, a, d)

        # Build pipeline
        p_sin, p_cos, p_one, q_mat, meta = build_pq(
            dh, T_target, linearity_joint=best_lj, return_metadata=True
        )
        e_sin, e_cos, e_one = eliminate_q0_q1(p_sin, p_cos, p_one, q_mat)
        e_quad, e_lin, e_const = weierstrass_eliminate_trig(e_sin, e_cos, e_one)
        m_quad, m_lin, m_const = build_m_matrix(e_quad, e_lin, e_const)

        # Pencil conditioning measurements (#71 detail probe).
        cond_at_solve.append(float(np.linalg.cond(m_quad)))
        cond_b_at_solve.append(float(np.linalg.cond(m_lin)))
        cond_c_at_solve.append(float(np.linalg.cond(m_const)))
        a_inv_b = np.linalg.solve(m_quad, m_lin)
        a_inv_c = np.linalg.solve(m_quad, m_const)
        norm_ainvb.append(float(np.linalg.norm(a_inv_b, ord=2)))
        norm_ainvc.append(float(np.linalg.norm(a_inv_c, ord=2)))
        # Companion matrix conditioning (Σ = [[0, I], [-A^-1 C, -A^-1 B]]).
        sigma = np.zeros((24, 24))
        sigma[:12, 12:] = np.eye(12)
        sigma[12:, :12] = -a_inv_c
        sigma[12:, 12:] = -a_inv_b
        cond_sigma_at_solve.append(float(np.linalg.cond(sigma)))

        roots, eigvecs = solve_x2_roots_mobius(m_quad, m_lin, m_const)
        n_roots_per_pose.append(len(roots))

        pose_succeeded = False
        for r, ev in zip(roots, eigvecs, strict=False):
            q_cand = back_substitute(r, ev, p_sin, p_cos, p_one, q_mat, dh, T_target, metadata=meta)
            if q_cand is None:
                candidate_class_counts["back_sub_none"] += 1
                fk_err_by_class["back_sub_none"].append(np.nan)
                continue
            fk_err_alg = float(np.linalg.norm(_fk(q_cand, alpha, a, d) - T_target))
            qdiff_seed = float(max(abs((q_cand[i] - q_star[i] + np.pi) % (2*np.pi) - np.pi) for i in range(6)))  # noqa: E501
            if fk_err_alg <= fk_atol:
                candidate_class_counts["algebraic_pass"] += 1
                fk_err_by_class["algebraic_pass"].append(fk_err_alg)
                seed_qdiff_by_class["algebraic_pass"].append(qdiff_seed)
                pose_succeeded = True
                continue
            # Need Newton -- with trajectory tracking.
            refined = _newton_refine(
                q_cand, dh, T_target, fk_atol=fk_atol, return_trajectory=True,
            )
            if refined is None:
                # Newton hit max_iters without converging. Distinguish:
                # (a) likely-spurious seed (fk_err high, |q-q*| large)
                # (b) close seed that genuinely should have converged
                if fk_err_alg > 1e-1:
                    candidate_class_counts["newton_diverged"] += 1
                    fk_err_by_class["newton_diverged"].append(fk_err_alg)
                    seed_qdiff_by_class["newton_diverged"].append(qdiff_seed)
                else:
                    candidate_class_counts["newton_no_converge"] += 1
                    fk_err_by_class["newton_no_converge"].append(fk_err_alg)
                    seed_qdiff_by_class["newton_no_converge"].append(qdiff_seed)
                continue
            q_ref, iters_used, _trajectory = refined
            fk_err_ref = float(np.linalg.norm(_fk(q_ref, alpha, a, d) - T_target))
            candidate_class_counts["newton_pass"] += 1
            fk_err_by_class["newton_pass"].append(fk_err_ref)
            seed_qdiff_by_class["newton_pass"].append(qdiff_seed)
            newton_iters_by_class["newton_pass"].append(iters_used)
            pose_succeeded = True

        pose_outcomes.append(pose_succeeded)

    # Report
    print(f"\nResults over {n_poses} poses:")
    n_passed = sum(pose_outcomes)
    print(f"  pose-level: {n_passed}/{n_poses} succeeded ({100*n_passed/n_poses:.0f}%)")
    cond_arr = np.array(cond_at_solve)
    print(f"  cond(A=m_quad)         median={np.median(cond_arr):.3e}, max={cond_arr.max():.3e}")
    print(f"  cond(B=m_lin)          median={np.median(cond_b_at_solve):.3e}, max={max(cond_b_at_solve):.3e}")  # noqa: E501
    print(f"  cond(C=m_const)        median={np.median(cond_c_at_solve):.3e}, max={max(cond_c_at_solve):.3e}")  # noqa: E501
    print(f"  ||A^-1 B|| (op-norm)   median={np.median(norm_ainvb):.3e}, max={max(norm_ainvb):.3e}")
    print(f"  ||A^-1 C|| (op-norm)   median={np.median(norm_ainvc):.3e}, max={max(norm_ainvc):.3e}")
    print(f"  cond(\u03a3 companion 24x24) median={np.median(cond_sigma_at_solve):.3e}, "
          f"max={max(cond_sigma_at_solve):.3e}")
    n_roots = np.array(n_roots_per_pose)
    print(f"  num real roots per pose: median={int(np.median(n_roots))}, "
          f"min={n_roots.min()}, max={n_roots.max()}")
    total_cands = sum(candidate_class_counts.values())
    print(f"  total candidates evaluated: {total_cands}")
    for cls in ("algebraic_pass", "newton_pass", "back_sub_none", "newton_diverged", "newton_no_converge"):  # noqa: E501
        n = candidate_class_counts[cls]
        pct = 100 * n / total_cands if total_cands else 0
        line = f"    {cls:<22} {n:>5} ({pct:5.1f}%)"
        if fk_err_by_class.get(cls):
            errs = np.array([e for e in fk_err_by_class[cls] if not np.isnan(e)])
            if len(errs):
                line += f"  fk_err: median={np.median(errs):.3e}, max={errs.max():.3e}"
        if seed_qdiff_by_class.get(cls):
            qd = np.array(seed_qdiff_by_class[cls])
            line += f"  |seed-q*|: median={np.median(qd):.3f}"
        if newton_iters_by_class.get(cls):
            ni = np.array(newton_iters_by_class[cls])
            line += f"  iters: median={int(np.median(ni))}, max={ni.max()}"
        print(line)


def main() -> None:
    # JACO 2 (Kinova j2n6s200) -- known good case after AE-3
    diagnose_arm(
        "JACO 2 (j2n6s200, 60-deg twists at joints 4-5)",
        alpha=np.array([np.pi/2, np.pi, np.pi/2, 60*np.pi/180, 60*np.pi/180, np.pi]),
        a=np.array([0.0, 0.41, 0.0, 0.0, 0.0, 0.0]),
        d=np.array([0.2755, 0.0, -0.0098, -0.2502, -0.0858, -0.2116]),
        n_poses=50,
        fk_atol=1e-6,
        q_range=1.0,
    )

    # MC Table I -- failing case
    diagnose_arm(
        "Manocha-Canny Table I (mixed-alpha synthetic)",
        alpha=np.array([np.pi/2, 1.0, np.pi/2, 1.0, np.pi/2, 1.0]),
        a=np.array([0.3, 1.0, 0.0, 1.5, 0.0, 0.0]),
        d=np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0]),
        n_poses=50,
        fk_atol=1e-6,
        q_range=1.0,
    )


if __name__ == "__main__":
    main()
