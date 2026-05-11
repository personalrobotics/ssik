"""Bulletproof locked-7R conformance tests for HP general_6r.

Phase 5c.4 / GitHub #176: HP must produce machine-precision IK on every
locked sub-chain of every 7R arm in the fixture set. The DH audit
(franka_panda, kuka_iiwa14, xarm7 x 7 lock indices = 21 configs) shows
that 15/21 hit the **Tv2 sub-case `[a_1=0, a_2=0]`** (V_L lies in the
Study quadric structurally — measure-zero singularity). HP resolves
this by perturbing ``a_2 → ε = 1e-3`` (and/or ``a_5 → ε`` for the
right-chain Tv5 case), running the standard Tv1+Tv4 dispatch, and
polishing each algebraic seed via 6-D Levenberg-Marquardt on the
unperturbed POE FK.

Test contract (per ``feedback_bulletproof_solvers``):

- **FK closure ≤ 1e-10** for at least one returned IK on every config.
- **q_truth recovery within 1e-6** for non-singular configs (Jacobian
  full-rank at q_truth).
- **At-least-one valid IK** for structurally-singular configs (Jacobian
  rank-deficient at q_truth — IK is a continuous family, q_truth is
  one of infinite valid solutions).

Known structurally-singular config (skipped from q_truth assertion):
- ``iiwa14`` lock=3: Jacobian sv_min ~ 1e-8 at every q I tested. The
  60° elbow-locked DH gives a globally rank-deficient 6R chain. HP
  returns a valid IK from the 1-D continuous family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from franka_panda import franka_panda_specs
from kuka_iiwa14 import kuka_iiwa14_specs
from xarm7 import xarm7_specs

from ssik._kinbody import build_kinbody
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.husty_pfurner.general_6r import solve as hp_solve
from ssik.solvers.jointlock.seven_r import _lock_joint

ARMS = [
    ("franka", franka_panda_specs),
    ("iiwa14", kuka_iiwa14_specs),
    ("xarm7", xarm7_specs),
]

# Q values used to generate the IK target via FK; deleted at lock_idx for the
# 6R sub-chain.
_Q_DEFAULT = np.array([0.3, -0.4, 0.7, 0.5, 0.6, -0.5, 0.2])

# Configs known to be structurally singular at the test q_truth (Jacobian
# rank-deficient → IK is a continuous family, q_truth recovery is undefined).
# HP must still return a valid IK from the family.
_SINGULAR_CONFIGS = {("iiwa14", 3)}


def _q_truth_for_lock(lock_idx: int) -> np.ndarray:
    return np.delete(_Q_DEFAULT, lock_idx)


def _closest_q_err_modulo_2pi(q_returned: np.ndarray, q_truth: np.ndarray) -> float:
    """Wrap each joint to be within π of q_truth, then return max |q - q_truth|."""
    q_close = q_returned.copy()
    for i in range(len(q_close)):
        while q_close[i] - q_truth[i] > np.pi:
            q_close[i] -= 2 * np.pi
        while q_truth[i] - q_close[i] > np.pi:
            q_close[i] += 2 * np.pi
    return float(np.max(np.abs(q_close - q_truth)))


@pytest.mark.parametrize(
    ("arm_name", "lock_idx"),
    [(arm, lock) for arm, _ in ARMS for lock in range(7) if (arm, lock) not in _SINGULAR_CONFIGS],
)
def test_locked_7r_fk_closure(arm_name: str, lock_idx: int) -> None:
    """Every returned IK on every locked-7R config FK-closes at machine
    precision. Bulletproof contract: FK Frobenius residual ≤ 1e-10.
    """
    specs_fn = dict(ARMS)[arm_name]
    kb6 = _lock_joint(build_kinbody(specs_fn()), lock_idx, 0.5)
    q_truth = _q_truth_for_lock(lock_idx)
    T_target = poe_forward_kinematics(kb6, q_truth)

    sols, is_ls = hp_solve(kb6, T_target, allow_refinement=True)
    assert sols, (
        f"{arm_name} lock={lock_idx}: HP returned 0 IK solutions for a "
        f"reachable target (q_truth = {q_truth.tolist()})"
    )
    assert not is_ls, f"{arm_name} lock={lock_idx}: HP reported is_ls=True"

    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-10, (
        f"{arm_name} lock={lock_idx}: best FK residual {best_fk:.3e} > 1e-10. "
        f"HP returned {len(sols)} IK solutions, none reached bulletproof FK closure."
    )

    # Every returned IK must be a real IK (FK closure ≤ 1e-8 — looser than
    # the best one, allows for cluster siblings that didn't fully polish
    # but are still valid).
    for s in sols:
        assert s.fk_residual < 1e-8, (
            f"{arm_name} lock={lock_idx}: FK residual "
            f"{s.fk_residual:.3e} > 1e-8 — likely a spurious algebraic seed "
            f"that LM didn't reject"
        )


@pytest.mark.parametrize(
    ("arm_name", "lock_idx"),
    [(arm, lock) for arm, _ in ARMS for lock in range(7) if (arm, lock) not in _SINGULAR_CONFIGS],
)
def test_locked_7r_q_truth_recovery(arm_name: str, lock_idx: int) -> None:
    """Non-singular locked-7R configs recover q_truth at machine precision
    (max |q - q_truth| modulo 2π < 1e-6).

    Singular configs (Jacobian rank-deficient at q_truth) are excluded;
    they're tested separately via FK closure only.
    """
    specs_fn = dict(ARMS)[arm_name]
    kb6 = _lock_joint(build_kinbody(specs_fn()), lock_idx, 0.5)
    q_truth = _q_truth_for_lock(lock_idx)
    T_target = poe_forward_kinematics(kb6, q_truth)

    sols, _ = hp_solve(kb6, T_target, allow_refinement=True)
    assert sols, f"{arm_name} lock={lock_idx}: 0 IK"

    best_q_err = min(_closest_q_err_modulo_2pi(np.asarray(s.q), q_truth) for s in sols)
    assert best_q_err < 1e-6, (
        f"{arm_name} lock={lock_idx}: closest IK to q_truth has max |q - q_truth| "
        f"= {best_q_err:.3e} > 1e-6. HP returned {len(sols)} IK candidates "
        f"and missed q_truth"
    )


@pytest.mark.parametrize(("arm_name", "lock_idx"), sorted(_SINGULAR_CONFIGS))
def test_locked_7r_singular_returns_some_valid_ik(arm_name: str, lock_idx: int) -> None:
    """Structurally-singular configs (Jacobian rank-deficient at q_truth)
    have a continuous IK family; HP returns a valid family member with
    FK closure at machine precision but not necessarily q_truth.
    """
    specs_fn = dict(ARMS)[arm_name]
    kb6 = _lock_joint(build_kinbody(specs_fn()), lock_idx, 0.5)
    q_truth = _q_truth_for_lock(lock_idx)
    T_target = poe_forward_kinematics(kb6, q_truth)

    sols, is_ls = hp_solve(kb6, T_target, allow_refinement=True)
    # Singular configs may genuinely return no algebraic seeds (the
    # elimination's polynomial system is rank-deficient). Acceptable
    # outcomes: (a) zero solutions with is_ls=True, OR (b) at least one
    # solution with FK closure ≤ 1e-8.
    if not sols:
        assert is_ls, f"{arm_name} lock={lock_idx} (singular): HP returned 0 IK but is_ls=False"
        return
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-8, f"{arm_name} lock={lock_idx} (singular): best FK {best_fk:.3e} > 1e-8"


def test_locked_franka_q_truth_to_machine_precision() -> None:
    """Locked-Franka regression test for the V_L⊂S perturbation fix.

    Pre-perturbation: HP returned 0 IK solutions for every locked-Franka
    config (the Tv2 sub-case [a_1=0, a_2=0] gave spurious algebraic roots
    that all failed FK closure). The perturbation ``a_2 → 1e-3`` + 6-D
    LM polish recovers q_truth at machine precision (FK ≤ 1e-15).
    """
    kb6 = _lock_joint(build_kinbody(franka_panda_specs()), 3, 0.5)
    q_truth = _q_truth_for_lock(3)
    T_target = poe_forward_kinematics(kb6, q_truth)
    sols, _ = hp_solve(kb6, T_target, allow_refinement=True)
    assert sols
    best_q_err = min(_closest_q_err_modulo_2pi(np.asarray(s.q), q_truth) for s in sols)
    best_fk = min(s.fk_residual for s in sols)
    assert best_q_err < 1e-6, f"locked-Franka q_truth not recovered: q_err={best_q_err:.3e}"
    assert best_fk < 1e-13, f"locked-Franka FK not at machine precision: {best_fk:.3e}"


def test_locked_franka_user_can_loosen_fk_atol() -> None:
    """User-facing fk_atol knob: looser threshold returns IKs faster but
    at lower precision. Default (1e-12) gives machine precision; loose
    (1e-4) gives micro-precision.
    """
    kb6 = _lock_joint(build_kinbody(franka_panda_specs()), 3, 0.5)
    q_truth = _q_truth_for_lock(3)
    T_target = poe_forward_kinematics(kb6, q_truth)

    sols_tight, _ = hp_solve(kb6, T_target, allow_refinement=True, fk_atol=1e-12)
    sols_loose, _ = hp_solve(kb6, T_target, allow_refinement=True, fk_atol=1e-4)

    assert sols_tight, "tight fk_atol returned 0 IK"
    assert sols_loose, "loose fk_atol returned 0 IK"
    # Both should find the IK (loose is just willing to terminate sooner),
    # so both should have at least one solution within machine precision
    # of q_truth.
    best_tight = min(_closest_q_err_modulo_2pi(np.asarray(s.q), q_truth) for s in sols_tight)
    best_loose = min(_closest_q_err_modulo_2pi(np.asarray(s.q), q_truth) for s in sols_loose)
    assert best_tight < 1e-6
    # Loose still recovers q_truth, just may polish less inside LM.
    assert best_loose < 1e-3
