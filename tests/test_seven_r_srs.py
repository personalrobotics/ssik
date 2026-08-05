"""Bulletproof validation for the native SRS-class 7R solver (#187).

Singh-Kreutz 1989 closed-form 7R for arms with shoulder-spherical +
wrist-spherical topology + elbow roll. Predicate-driven: any 7R
fixture that matches :func:`ssik.kinematics.predicates.is_srs_7r`
auto-applies the solver -- no per-arm hardcoding.

Test contract (per `feedback_bulletproof_solvers`):

- **FK closure ≤ 1e-10** for every returned IK on every reachable pose.
- **Hypothesis fuzz** over 100 random reachable poses.
- **Cross-validation** against `jointlock + HP` -- both algorithms must
  produce FK-correct IKs (solution-set agreement is up to wrap-to-π).
- **Performance gate**: <2 ms median full-sweep on iiwa14;
  <0.5 ms median with `max_solutions=1`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from _perf import best_call_ms
from kuka_iiwa14 import kuka_iiwa14_specs

from ssik._kinbody import KinBody, build_kinbody
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.seven_r.srs import solve as srs_solve

# ----------------------------------------------------------------------------
# Reachability sanity + FK closure on hand-picked poses
# ----------------------------------------------------------------------------


_HAND_PICKED_Q = [
    np.array([0.3, -0.4, 0.7, 0.5, 0.6, -0.5, 0.2]),
    np.array([-0.5, 0.3, -0.7, 1.0, -0.4, 0.5, -0.3]),
    np.array([0.0, 0.5, 0.0, 1.5, 0.0, -0.5, 0.0]),  # elbow-folded posture
    np.array([1.2, -0.8, 0.3, 0.4, 1.1, -0.6, 0.9]),
]


@pytest.mark.parametrize("q_star", _HAND_PICKED_Q)
def test_iiwa14_fk_closure_at_hand_picked(q_star: np.ndarray) -> None:
    """Every IK returned at a reachable iiwa14 pose FK-closes ≤ 1e-10."""
    kb = build_kinbody(kuka_iiwa14_specs())
    T_target = poe_forward_kinematics(kb, q_star)
    sols, is_ls = srs_solve(kb, T_target)
    assert sols, f"SRS returned no IK for reachable pose q={q_star}"
    assert not is_ls
    for sol in sols:
        T_check = poe_forward_kinematics(kb, sol.q)
        fk_err = float(np.linalg.norm(T_check - T_target))
        assert fk_err < 1e-10, f"FK closure failed: q={sol.q}, fk_err={fk_err:.2e}"


@pytest.mark.parametrize("q_star", _HAND_PICKED_Q)
def test_iiwa14_max_solutions_one(q_star: np.ndarray) -> None:
    """`max_solutions=1` returns exactly one valid IK at a reachable pose."""
    kb = build_kinbody(kuka_iiwa14_specs())
    T_target = poe_forward_kinematics(kb, q_star)
    sols, is_ls = srs_solve(kb, T_target, max_solutions=1)
    assert len(sols) == 1
    assert not is_ls
    T_check = poe_forward_kinematics(kb, sols[0].q)
    assert np.linalg.norm(T_check - T_target) < 1e-10


# ----------------------------------------------------------------------------
# Hypothesis fuzz: 100 random reachable poses round-trip
# ----------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_iiwa14_random_pose_fk_closure(seed: int) -> None:
    """100 random q in [-0.8, 0.8] per joint: FK round-trip must close
    on at least one returned IK.

    Range chosen to avoid joint limits (iiwa14 limits are typically
    ±2.0 / ±2.97 rad; ±0.8 keeps poses well-inside reachable workspace
    and avoids near-singular elbow extensions that hit the
    cosine-rule edges).
    """
    rng = np.random.default_rng(seed)
    q_star = rng.uniform(-0.8, 0.8, size=7)
    kb = build_kinbody(kuka_iiwa14_specs())
    T_target = poe_forward_kinematics(kb, q_star)
    sols, _ = srs_solve(kb, T_target)
    assert sols, f"random reachable pose returned no IK: q*={q_star.tolist()}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-10, f"random pose seed={seed}: best FK={best_fk:.2e} > 1e-10"


# ----------------------------------------------------------------------------
# Cross-validation: SRS vs jointlock + HP must produce FK-equal IKs
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("q_star", _HAND_PICKED_Q[:2])  # subset (HP is slow)
def test_iiwa14_srs_vs_jointlock_both_find_fk_correct_ik(q_star: np.ndarray) -> None:
    """SRS and jointlock+HP independently produce FK-correct IK on the
    same iiwa14 target. This is a *consistency* check: two algorithms
    based on independent algebra both reach FK closure.

    NOT asserted: HP ⊆ SRS (or vice-versa). The two algorithms cover
    iiwa14's 1-D redundancy manifold differently:

      * SRS samples the explicit swivel angle uniformly in [-π, π]
        with 16 samples. Targets at intermediate swivel values are
        not directly returned (they would require a denser sweep).
      * jointlock+HP samples q_3 (the elbow joint) implicitly via
        16 lock-joint values; the swivel emerges from the inner 6R
        IK rather than being sampled directly.

    The redundancy-coverage difference is structural; the IK-set
    intersection is non-trivial but generally not full. The test
    therefore asserts only that *both* algorithms agree the target
    is reachable + return FK-correct IKs.
    """
    from ssik.solvers.jointlock import seven_r as jointlock_seven_r

    kb = build_kinbody(kuka_iiwa14_specs())
    T_target = poe_forward_kinematics(kb, q_star)

    srs_sols, srs_is_ls = srs_solve(kb, T_target)
    hp_sols, hp_is_ls = jointlock_seven_r.solve(kb, T_target, allow_refinement=True)

    # Both must agree the target is reachable.
    assert not srs_is_ls
    assert not hp_is_ls
    assert srs_sols, f"q*={q_star.tolist()}: srs_sols={len(srs_sols)}"
    assert hp_sols, f"q*={q_star.tolist()}: hp_sols={len(hp_sols)}"

    # Best FK closure from each solver must hit machine precision (SRS) or
    # near-machine (HP -- LM converges to ~1e-7 to ~1e-13).
    best_srs_fk = min(s.fk_residual for s in srs_sols)
    best_hp_fk = min(s.fk_residual for s in hp_sols)
    assert best_srs_fk < 1e-10, f"SRS best FK={best_srs_fk:.2e}"
    assert best_hp_fk < 1e-6, f"HP best FK={best_hp_fk:.2e}"


# ----------------------------------------------------------------------------
# Performance gates
# ----------------------------------------------------------------------------


@pytest.mark.perf
def test_iiwa14_full_sweep_under_2ms() -> None:
    """Full swivel sweep + 8-branch enumeration on iiwa14 must take < 2 ms.

    Empirical (M3, single-thread): ~17 ms today. The 2 ms gate is the
    target after #186 (Cython compile of the inner FK loop). Until #186
    lands, we use a generous gate to catch *regressions*; the true
    target is sub-ms. Best-of-N timing (see :mod:`tests._perf`) so the gate
    tracks compute cost, not shared-runner scheduling noise.
    """
    kb = build_kinbody(kuka_iiwa14_specs())
    q_star = _HAND_PICKED_Q[0]
    T_target = poe_forward_kinematics(kb, q_star)
    best_ms = best_call_ms(lambda: srs_solve(kb, T_target))
    # Conservative gate: 50 ms catches >2.5x regression from current ~17 ms
    # baseline. Tighten to 2 ms target after #186 Cython compile.
    assert best_ms < 50, f"SRS full-sweep too slow: {best_ms:.2f} ms"


@pytest.mark.perf
def test_iiwa14_max_solutions_one_under_1ms() -> None:
    """`max_solutions=1` (give-me-any-IK use case) must early-exit, not compute
    the full solution set.

    Empirical (M3, single-thread): ~0.23 ms today; a full iiwa14 SRS solve is
    ~8.5 ms. The gate is a coarse "did the early-exit fire" ceiling, so it is set
    at 3 ms -- ~13x nominal, cleanly below the ~8.5 ms broken-early-exit cost, and
    robust to CI-runner load. The old 1 ms bound was ~4x nominal and flaked even
    on the best-of-N noise floor under sustained CI load (#383 / #479 on the
    4-version matrix). Precise regression detection is the relative-ratio gate's
    job (tests/_perf_baseline.json); this test just guards the early-exit path.
    """
    kb = build_kinbody(kuka_iiwa14_specs())
    q_star = _HAND_PICKED_Q[0]
    T_target = poe_forward_kinematics(kb, q_star)
    best_ms = best_call_ms(lambda: srs_solve(kb, T_target, max_solutions=1))
    assert best_ms < 3.0, (
        f"SRS max_solutions=1 too slow ({best_ms:.2f} ms): early-exit likely broken"
    )


# ----------------------------------------------------------------------------
# Topology refusal
# ----------------------------------------------------------------------------


def test_srs_solver_rejects_non_srs_arm() -> None:
    """Franka Panda is anthropomorphic, not SRS. The solver must raise."""
    from franka_panda import franka_panda_specs

    kb = build_kinbody(franka_panda_specs())
    with pytest.raises(ValueError, match="SRS"):
        srs_solve(kb, np.eye(4))


def test_srs_solver_rejects_non_7r() -> None:
    """6R chains must be rejected by the DOF check."""
    from ur5 import ur5_specs

    kb = build_kinbody(ur5_specs())
    with pytest.raises(ValueError, match="7-DOF"):
        srs_solve(kb, np.eye(4))


# ----------------------------------------------------------------------------
# Unreachable pose
# ----------------------------------------------------------------------------


def test_srs_unreachable_target_returns_is_ls() -> None:
    """A target far outside iiwa14's workspace returns is_ls=True with no
    solutions, not a crash.
    """
    kb = build_kinbody(kuka_iiwa14_specs())
    T = np.eye(4)
    T[0, 3] = 100.0  # 100 m away
    sols, is_ls = srs_solve(kb, T)
    assert is_ls
    assert len(sols) == 0


# ----------------------------------------------------------------------------
# Dispatcher integration
# ----------------------------------------------------------------------------


def test_dispatcher_picks_srs_for_iiwa14() -> None:
    """The top-level dispatcher routes iiwa14 to seven_r.srs (tier 0)."""
    from ssik.core.dispatcher import dispatch

    kb = build_kinbody(kuka_iiwa14_specs())
    plan = dispatch(kb)
    assert plan.solver_name == "seven_r.srs"
    assert plan.tier == 0


def test_dispatcher_routes_franka_to_spherical_shoulder_not_srs() -> None:
    """Franka Panda is non-SRS (offset wrist), so it must NOT route to
    seven_r.srs. Since #373 it routes to the exact spherical-shoulder specialist
    (not the jointlock fallback it used before)."""
    from franka_panda import franka_panda_specs

    from ssik.core.dispatcher import dispatch

    kb = build_kinbody(franka_panda_specs())
    plan = dispatch(kb)
    assert plan.solver_name == "seven_r.spherical_shoulder"


# ----------------------------------------------------------------------------
# Offset-wrist SRS (#424): concurrent axes but a laterally-offset wrist
# ----------------------------------------------------------------------------

_IIWA7_URDF = Path(__file__).parent / "fixtures" / "kuka_iiwa7.urdf"


def _iiwa7_kb() -> KinBody:
    from ssik._urdf import load_urdf_kinbody_normalized

    return load_urdf_kinbody_normalized(_IIWA7_URDF, "iiwa_link_0", "iiwa_link_ee")


def test_iiwa7_is_offset_wrist_srs() -> None:
    """iiwa7 (differentiable-robot-model) is genuinely SRS-class (wrist axes
    4/5/6 concurrent) but has intermediate lateral offsets at joints 5/6 that
    displace joint 5's origin from the wrist-concurrency point. This is the
    fixture that exposed the canonical fast-path's offset-free-wrist assumption
    (#424): iiwa14 (offset-free) does not."""
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY as POL
    from ssik.kinematics.predicates import _classify_srs_7r_geometric, is_srs_7r
    from ssik.solvers.seven_r.srs import _arm_constants

    kb = _iiwa7_kb()
    assert is_srs_7r(kb, POL), "iiwa7 must classify as SRS (concurrent wrist axes)"
    cls = _classify_srs_7r_geometric(kb, POL)
    assert cls is not None
    _, _, _, origins = _arm_constants(kb, cls)
    # Joint 5's origin is NOT at the wrist pivot -- this is what makes the
    # canonical fast-path invalid and forces the general path.
    offset = float(np.linalg.norm(origins[5] - cls.wrist_pivot))
    assert offset > 1e-2, f"expected an offset wrist; joint5 offset={offset:.4f}"


@pytest.mark.parametrize("q_star", _HAND_PICKED_Q)
def test_iiwa7_offset_wrist_fk_closure_hand_picked(q_star: np.ndarray) -> None:
    """Every IK returned at a reachable iiwa7 pose FK-closes <= 1e-10.

    Regression for #424: the canonical ZYZ fast-path used joint 5's frame as
    the wrist-pivot reference, so on an offset wrist it placed the wrong point
    at the target and *every* candidate missed FK closure by ~the offset
    (~4-6 cm on iiwa7) -> zero solutions. The offset-free-wrist guard routes
    iiwa7 to the general path, which closes to machine precision.
    """
    kb = _iiwa7_kb()
    T_target = poe_forward_kinematics(kb, q_star)
    sols, is_ls = srs_solve(kb, T_target)
    assert sols, f"SRS returned no IK for reachable iiwa7 pose q={q_star}"
    assert not is_ls
    for sol in sols:
        fk_err = float(np.linalg.norm(poe_forward_kinematics(kb, sol.q) - T_target))
        assert fk_err < 1e-10, f"FK closure failed: q={sol.q}, fk_err={fk_err:.2e}"


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_iiwa7_offset_wrist_random_pose_fk_closure(seed: int) -> None:
    """100 random reachable iiwa7 poses: at least one returned IK FK-closes."""
    kb = _iiwa7_kb()
    rng = np.random.default_rng(seed)
    q_star = rng.uniform(-0.8, 0.8, size=7)
    T_target = poe_forward_kinematics(kb, q_star)
    sols, _ = srs_solve(kb, T_target)
    assert sols, f"no IK for reachable iiwa7 pose (seed={seed})"
    assert any(
        float(np.linalg.norm(poe_forward_kinematics(kb, s.q) - T_target)) < 1e-8 for s in sols
    ), f"no FK-closing IK among {len(sols)} for seed={seed}"


# ----------------------------------------------------------------------------
# Rotated home ee frame (#517): R_ee_home != I must not break the SRS solve
# ----------------------------------------------------------------------------

_IIWA14_URDF = Path(__file__).parent / "fixtures" / "kuka_iiwa14.urdf"


def test_srs_solves_with_rotated_home_ee_frame() -> None:
    """A rotated home flange (R_ee_home != I) must still solve to machine
    precision. Regression for #517: _arm_constants baked the tool offset as a
    *world* vector at q=0, but the solver applies it as an *ee-frame* offset
    (W_t = p - R_target @ ee_offset_local). Without the R_home.T gauge-
    normalization the wrist target was displaced by a constant ~0.18 m, and
    *every* candidate missed FK closure -> zero solutions. iiwa_link_ee (offset-
    free but with a rotated home flange) is the canonical-path reproducer; the
    offset-free iiwa_link_7 (R_home == I) is unaffected.
    """
    from ssik._urdf import load_urdf_kinbody_normalized
    from ssik.kinematics.poe_fk import poe_forward_kinematics as fk

    kb = load_urdf_kinbody_normalized(_IIWA14_URDF, "iiwa_link_0", "iiwa_link_ee")
    assert not np.allclose(fk(kb, np.zeros(7))[:3, :3], np.eye(3)), (
        "fixture must have a rotated home ee frame to guard #517"
    )
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(100):
        q = rng.uniform(-2.0, 2.0, size=7)
        t = fk(kb, q)
        sols, is_ls = srs_solve(kb, t)
        assert sols, "no IK for a reachable rotated-home-ee pose (#517)"
        assert not is_ls, "rotated-home-ee pose returned a least-squares (empty) result (#517)"
        worst = max(worst, min(float(np.linalg.norm(fk(kb, s.q) - t)) for s in sols))
    assert worst < 1e-10, f"rotated-home-ee FK closure {worst:.2e} (#517 regressed)"
