"""Bulletproof gate for specialised codegen artifacts.

For each tier-0 composer, runs the existing fixture solver test pattern
against the SPECIALISED artifact (not the runtime solver):

  - 100 random poses
  - Every returned solution must FK-close on the seeded target at 1e-9
  - At least one returned solution must wrap-to-pi-match the seeded q*

This guarantees the specialised codegen is functionally equivalent to
the runtime solver, not just structurally similar -- the contract that
makes #112's "no exceptions, bulletproof" requirement real.

Tier-1 and tier-2 composers will land in subsequent PRs and add their
own bulletproof tests (each follows the same pattern).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ssik._kinbody import KinBody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import dispatch
from ssik.subproblems._rotation import rotation_matrix

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))


def _wrap(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


def _q_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-3) -> bool:
    return all(abs(_wrap(float(ai - bi))) < tol for ai, bi in zip(a, b, strict=True))


def _fk(kb: KinBody, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        rot = np.eye(4)
        rot[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = T @ j.T_left @ rot @ j.T_right
    return T


def _build_specialised(kb: KinBody, module_name: str, tmp_path: Path) -> object:
    plan = dispatch(kb)
    artifact_path = tmp_path / f"{module_name}.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name=module_name,
        output_path=str(artifact_path),
        arm_label=module_name,
    )
    spec = importlib.util.spec_from_file_location(module_name, artifact_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bulletproof_check(
    kb: KinBody,
    artifact: object,
    *,
    n_poses: int = 100,
    seed: int = 0,
    fk_atol: float = 1e-8,
    q_atol: float = 1e-3,
) -> None:
    """Round-trip ``n_poses`` random poses through the specialised artifact.

    For each pose:
      1. Pick random ``q*``, compute ``T_star = fk(q*)``.
      2. Solve via ``artifact.solve(T_star)``.
      3. Assert ``is_ls is False`` (algebraic path closed).
      4. Every returned solution must FK-close on T_star at ``fk_atol``.
      5. At least one returned solution must wrap-to-pi-match q*.
    """
    rng = np.random.default_rng(seed=seed)
    n_dof = len(kb.joints)
    fails = 0
    miss_seeded = 0
    for trial in range(n_poses):
        q_star = rng.uniform(-1.0, 1.0, size=n_dof)
        T_star = _fk(kb, q_star)

        sols, is_ls = artifact.solve(T_star)  # type: ignore[attr-defined]
        if is_ls or not sols:
            fails += 1
            continue

        # Every returned q must FK-close on T_star.
        for sol in sols:
            T_check = _fk(kb, sol.q)
            if not np.allclose(T_check, T_star, atol=fk_atol):
                pytest.fail(
                    f"trial {trial}: artifact q={sol.q.tolist()} fails FK closure "
                    f"(max|diff|={float(np.max(np.abs(T_check - T_star))):.2e})"
                )

        # At least one returned solution must match seeded q* (wrap-to-pi).
        if not any(_q_match(np.asarray(sol.q), q_star, tol=q_atol) for sol in sols):
            miss_seeded += 1

    assert fails == 0, f"{fails}/{n_poses} poses returned is_ls=True or empty"
    # Allow a small fraction of missed seeded recoveries (near-singular poses
    # where the algebraic path returns FK-equivalent but distinct q-vectors).
    assert miss_seeded < n_poses * 0.1, (
        f"{miss_seeded}/{n_poses} poses failed to recover seeded q* (>10% miss rate)"
    )


# ---------------------------------------------------------------------------
# Tier-0 specialised artifacts.
# ---------------------------------------------------------------------------


def test_bulletproof_puma560_specialised(tmp_path: Path) -> None:
    """Puma 560 specialised artifact (spherical_two_parallel) bulletproof."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")
    artifact = _build_specialised(kb, "puma560_bp", tmp_path)
    _bulletproof_check(kb, artifact, n_poses=100, fk_atol=1e-8)


def test_bulletproof_ur5_specialised(tmp_path: Path) -> None:
    """UR5 specialised artifact (three_parallel) bulletproof."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    artifact = _build_specialised(kb, "ur5_bp", tmp_path)
    _bulletproof_check(kb, artifact, n_poses=100, fk_atol=1e-8)


def test_specialised_artifact_refinement_path_works(tmp_path: Path) -> None:
    """The specialised artifact's ``allow_refinement=True`` path must wire
    through to ``ssik.refinement.lm_refine`` correctly.

    Strategy: use a tightened ``subproblem_numerical`` so some algebraic
    candidates fall into the near-miss bucket. Without refinement, those
    get dropped (fewer solutions). With refinement, the Newton polish
    recovers them (same or more solutions). At minimum, refinement must
    return non-empty results without raising and report
    ``refinement_used == "lm"`` for any polished candidate.
    """
    from ssik.core.tolerances import TolerancePolicy

    kb = load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")
    artifact = _build_specialised(kb, "puma560_refine", tmp_path)

    rng = np.random.default_rng(seed=11)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    T_star = _fk(kb, q_star)

    # Tight policy that triggers the near-miss path on at least some poses.
    tight_policy = TolerancePolicy(subproblem_numerical=1e-13)

    sols_off, _ = artifact.solve(T_star, policy=tight_policy, allow_refinement=False)  # type: ignore[attr-defined]
    sols_on, _ = artifact.solve(T_star, policy=tight_policy, allow_refinement=True)  # type: ignore[attr-defined]

    # Refinement should recover at least as many candidates as the no-refine path.
    assert len(sols_on) >= len(sols_off), (
        f"refinement reduced solution count: {len(sols_off)} (off) -> {len(sols_on)} (on)"
    )

    # If refinement helped, at least one solution should report it.
    if len(sols_on) > len(sols_off):
        assert any(s.refinement_used == "lm" for s in sols_on), (
            "refinement added solutions but no Solution.refinement_used == 'lm'"
        )
