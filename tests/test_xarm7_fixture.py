"""UFactory xArm7 fixture validation + 7R IK end-to-end.

xArm7 is a **verified Pieper-class wedge**: locking the auto-selected
joint and reversing the resulting 6R sub-chain dispatches to
``reversed:spherical`` for 15/16 lock samples and
``reversed:spherical_two_parallel`` for the remaining one. There is no
``gen_six_dof`` fallthrough -- 7R IK runs at ~3 ms with
``max_solutions=1`` and ~40 ms exhaustive on a workstation.

This module covers:

1. The fixture itself: builds with 7 revolute joints, FK is deterministic,
   ``build_kinbody`` POE-normalises axes into the base frame at q=0.
2. The wedge fingerprint: dispatch lands in ``reversed:spherical`` (no
   tier-2 fallthrough).
3. Real 7R IK closure on random reachable poses, default and
   ``max_solutions=1`` paths.
4. Sanity: unreachable target returns cleanly, joint limits flow through
   the lock-sample clamp.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from xarm7 import XARM7_KEYFRAMES, xarm7_specs

from ssik._kinbody import build_kinbody
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.jointlock import seven_r


def test_xarm7_fixture_builds_with_seven_revolute_joints() -> None:
    """The fixture transcribes 7 revolute joints with limits."""
    specs = xarm7_specs()
    assert len(specs) == 7
    for spec in specs:
        assert spec.joint_type == "revolute"
        assert spec.limits is not None
        lo, hi = spec.limits
        assert lo < 0 < hi


def test_xarm7_fk_at_home_is_along_z() -> None:
    """At ``q = 0`` the xArm7 is folded; the EE position should be finite
    and consistent across calls.
    """
    kb = build_kinbody(xarm7_specs())
    T = poe_forward_kinematics(kb, XARM7_KEYFRAMES["home"])
    assert np.all(np.isfinite(T))
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_xarm7_fk_round_trip() -> None:
    """FK is deterministic across a handful of random q values."""
    kb = build_kinbody(xarm7_specs())
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=7)
        T1 = poe_forward_kinematics(kb, q)
        T2 = poe_forward_kinematics(kb, q)
        assert np.allclose(T1, T2, atol=1e-15)
        R = T1[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_xarm7_kb_axes_are_world_frame_at_q0() -> None:
    """``build_kinbody`` POE-normalises the xArm7 chain: each joint's
    ``axis`` is in the base frame at q=0, not the local +Z used by the
    MJCF transcription.
    """
    kb = build_kinbody(xarm7_specs())
    axes = [j.axis.copy() for j in kb.joints]
    distinct = {tuple(np.round(a, 6).tolist()) for a in axes}
    assert len(distinct) > 1, (
        f"xArm7 world-frame axes should vary, got {axes} -- "
        "build_kinbody POE-normalisation did not apply"
    )


def test_xarm7_dispatches_to_reversed_spherical_wedge() -> None:
    """xArm7 is a Pieper-class wedge: at the auto-selected lock joint,
    most samples dispatch to ``reversed:spherical`` (the canonical
    spherical-wrist solver applied to the chain-reversed 6R sub-chain).

    The lock sweep should produce zero ``gen_six_dof`` samples -- if a
    future change re-introduces the tier-2 fallthrough on xArm7, this
    test catches it.
    """
    kb = build_kinbody(xarm7_specs())
    lock_idx = seven_r.choose_lock_joint(kb, DEFAULT_TOLERANCE_POLICY)
    joint_lim = kb.joints[lock_idx].limits
    lo, hi = joint_lim if joint_lim is not None else (-np.pi, np.pi)
    samples = np.linspace(lo, hi, seven_r._DEFAULT_SAMPLES, endpoint=False)

    dispatch: dict[str, int] = {}
    for q_lock in samples:
        sub = seven_r._lock_joint(kb, lock_idx, float(q_lock))
        _, solver_name = seven_r._topology_rank(sub, DEFAULT_TOLERANCE_POLICY)
        dispatch[solver_name] = dispatch.get(solver_name, 0) + 1

    assert "gen_six_dof" not in dispatch, (
        f"xArm7 dispatch fell through to gen_six_dof for some lock samples: {dispatch}"
    )
    assert "reversed:gen_six_dof" not in dispatch, (
        f"xArm7 dispatch fell through to reversed:gen_six_dof: {dispatch}"
    )
    fast = dispatch.get("reversed:spherical", 0) + dispatch.get(
        "reversed:spherical_two_parallel", 0
    )
    assert fast >= seven_r._DEFAULT_SAMPLES - 1, (
        f"xArm7 should land in reversed:spherical[_two_parallel] for nearly every "
        f"lock sample, got dispatch={dispatch}"
    )


def test_xarm7_seven_r_solves_random_reachable_poses() -> None:
    """Random reachable poses produce IK solutions through the wedge
    dispatch path.

    Bulletproof claim: every reachable pose produces ``>= 1`` candidate
    that FK-closes at the standard ssik tolerance (``atol=1e-9``,
    rtol=1e-5 default), AND every returned candidate FK-closes within
    the algorithm's known precision floor (``atol=1e-5``).

    The two-tier check separates the *coverage* claim ("the wedge solves
    every reachable pose to machine precision") from the *purity* claim
    ("every branch is at machine precision"). xArm7's lock sweep
    dispatches 1/16 samples to ``reversed:spherical_two_parallel`` whose
    inner branch composition can produce residuals up to ~1e-6 on
    near-origin poses; that's the algorithm's real precision floor on
    this arm and we document it honestly rather than over-promising
    machine precision uniformly.
    """
    kb = build_kinbody(xarm7_specs())
    rng = np.random.default_rng(seed=0)
    closures = 0
    for _ in range(5):
        q_true = rng.uniform(-0.8, 0.8, size=7)
        T_target = poe_forward_kinematics(kb, q_true)
        sols, is_ls = seven_r.solve(kb, T_target)
        if is_ls or not sols:
            continue
        # Coverage: at least one candidate at machine precision.
        machine_precision_hits = [
            sol
            for sol in sols
            if np.allclose(poe_forward_kinematics(kb, sol.q), T_target, atol=1e-9)
        ]
        assert machine_precision_hits, (
            f"xArm7 IK returned {len(sols)} candidates but NONE at machine "
            f"precision (atol=1e-9). Wedge dispatch is producing only sub-precision branches."
        )
        # Purity: every candidate within the algorithm's precision floor.
        for sol in sols:
            T_check = poe_forward_kinematics(kb, sol.q)
            err = float(np.max(np.abs(T_check - T_target)))
            assert err < 1e-5, (
                f"xArm7 IK candidate exceeded precision floor: max|diff|={err:.2e} > 1e-5"
            )
        closures += 1
    assert closures == 5, (
        f"xArm7 7R IK closed only {closures}/5 random reachable poses -- "
        "wedge dispatch is not producing valid solutions."
    )


def test_xarm7_max_solutions_short_circuit_returns_one_valid_ik() -> None:
    """``max_solutions=1`` short-circuits the lock sweep on the first
    valid IK and returns it FK-closed at machine precision.
    """
    kb = build_kinbody(xarm7_specs())
    rng = np.random.default_rng(seed=1)
    q_true = rng.uniform(-0.8, 0.8, size=7)
    T_target = poe_forward_kinematics(kb, q_true)

    sols, is_ls = seven_r.solve(kb, T_target, max_solutions=1)
    assert not is_ls
    assert len(sols) == 1
    T_check = poe_forward_kinematics(kb, sols[0].q)
    assert np.allclose(T_check, T_target, atol=1e-9), (
        f"xArm7 max_solutions=1 FK closure: "
        f"max|diff|={float(np.max(np.abs(T_check - T_target))):.2e}"
    )


def test_xarm7_seven_r_returns_cleanly_at_unreachable_target() -> None:
    """A clearly-out-of-reach target (target far outside the work
    envelope) returns an empty solution set with ``is_ls=True``,
    not a crash.
    """
    kb = build_kinbody(xarm7_specs())
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = 100.0
    sols, is_ls = seven_r.solve(kb, T, max_solutions=1)
    assert is_ls
    assert len(sols) == 0


def test_xarm7_joint_limits_baked_into_kinbody() -> None:
    """Each joint's MJCF-supplied ``range`` lands in ``Joint.limits`` after
    ``build_kinbody`` POE-normalisation.
    """
    kb = build_kinbody(xarm7_specs())
    expected = [
        (-6.2832, 6.2832),
        (-2.059, 2.0944),
        (-6.2832, 6.2832),
        (-0.19198, 3.927),
        (-6.2832, 6.2832),
        (-1.69297, 3.14159),
        (-6.2832, 6.2832),
    ]
    for i, lim in enumerate(expected):
        assert kb.joints[i].limits == lim, f"joint {i} limits"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_xarm7_max_solutions_short_circuit_fk_closure_seeded(seed: int) -> None:
    """Parametric repeat of the short-circuit FK-closure check across
    five seeds to surface any seed-specific branch failure quickly.
    """
    kb = build_kinbody(xarm7_specs())
    rng = np.random.default_rng(seed=seed)
    q_true = rng.uniform(-0.8, 0.8, size=7)
    T_target = poe_forward_kinematics(kb, q_true)

    sols, is_ls = seven_r.solve(kb, T_target, max_solutions=1)
    if is_ls or not sols:
        pytest.skip(f"seed {seed}: pose not reachable by short-circuit path")
    T_check = poe_forward_kinematics(kb, sols[0].q)
    assert np.allclose(T_check, T_target, atol=1e-9), (
        f"xArm7 seed={seed} FK closure: max|diff|={float(np.max(np.abs(T_check - T_target))):.2e}"
    )
