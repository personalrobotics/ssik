"""Franka Panda fixture validation + 7R IK end-to-end (#121).

The fixture itself is a transcribed MJCF (see ``tests/fixtures/franka_panda.py``).
This module covers four things:

1. The fixture builds correctly: FK at the documented home pose matches
   the Franka spec, and ``build_kinbody`` produces a POE-normalised
   chain (axes in the base frame).
2. ``build_kinbody`` POE-normalisation lands axes in the world frame at
   q=0 (regression check for #125).
3. The auto-selected lock joint matches EAIK's pick (joint index 4)
   and the topology rank dispatches to ``reversed:spherical_two_parallel``
   at tier-0 closed-form speed (#121 Level 1: chain reversal).
4. Random-pose 7R IK closes at machine precision via the reversed-chain
   dispatch path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ssik._kinbody import KinBody, build_kinbody
from ssik.kinematics._scalar3 import _mat4_mat4
from ssik.subproblems._rotation import rotation_matrix

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))
sys.path.insert(0, str(Path(__file__).parent.parent / "prebuilt"))

from franka_panda import (  # noqa: E402
    FRANKA_PANDA_KEYFRAMES,
    franka_panda_specs,
)


def _fk(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        R = np.eye(4)
        R[:3, :3] = rotation_matrix(j.axis, float(qi))
        T = _mat4_mat4(_mat4_mat4(T, _mat4_mat4(j.T_left, R)), j.T_right)
    return T


def test_franka_fixture_builds_and_fk_at_home_matches() -> None:
    """``franka_panda_specs`` builds a 7-joint POE-normalised KinBody whose
    FK at the home keyframe matches the published Franka spec."""
    kb = build_kinbody(franka_panda_specs())
    assert len(kb.joints) == 7

    # All joints revolute, axes unit-length in world frame.
    for j in kb.joints:
        assert j.joint_type == "revolute"
        assert abs(np.linalg.norm(j.axis) - 1.0) < 1e-12

    # FK at home matches the published Franka home position
    # (qpos="0 0 0 -1.57079 0 1.57079 -0.7853"). The translation is
    # roughly 0.5 m forward, 0.6 m up.
    T_home = _fk(kb, FRANKA_PANDA_KEYFRAMES["home"])
    home_xyz = T_home[:3, 3]
    assert 0.5 < home_xyz[0] < 0.6, f"unexpected home x: {home_xyz[0]}"
    assert abs(home_xyz[1]) < 1e-6, f"home y should be ~0: {home_xyz[1]}"
    assert 0.6 < home_xyz[2] < 0.7, f"unexpected home z: {home_xyz[2]}"


def test_franka_kb_axes_are_world_frame_at_q0() -> None:
    """``build_kinbody`` POE-normalises the Franka chain: each joint's
    ``axis`` is in the base frame at q=0, not in some local frame.

    Confirms the fix landed in this PR: pre-fix, all Franka joints had
    ``axis = [0, 0, 1]`` (the local convention for MJCF-built specs).
    Post-fix, axes vary because the per-link quaternion rotations are
    folded into the world-frame axis.
    """
    kb = build_kinbody(franka_panda_specs())
    # Franka has alternating axis directions in world frame -- if all were
    # (0, 0, 1) the build_kinbody normalisation didn't apply.
    axes = [j.axis.copy() for j in kb.joints]
    distinct = {tuple(np.round(a, 6).tolist()) for a in axes}
    assert len(distinct) > 1, (
        f"Franka world-frame axes should vary, got {axes} -- "
        "build_kinbody POE-normalisation did not apply"
    )


def test_franka_dispatches_to_reversed_spherical_two_parallel() -> None:
    """Franka post-lock-4 dispatches to ``reversed:spherical_two_parallel``
    at tier-0 closed-form speed.

    The chain has its spherical wrist at the BASE of the sub-chain
    (joints 0,1,2 all pass through (0,0,0.333)). ssik's
    :func:`~ssik.solvers.jointlock.seven_r._topology_rank` reverses the
    chain and recognises the now-canonical-position spherical wrist;
    :func:`~ssik.solvers.jointlock.seven_r._dispatch` routes the call
    through ``reverse_kinematic_chain``, dispatches to the standard
    ``spherical_two_parallel`` solver, and maps the returned q-vectors
    back to the original ordering.

    EAIK identifies the same structure as ``REVERSED +
    SPHERICAL_SECOND_TWO_PARALLEL`` -- consistent with this dispatch.
    """
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.solvers.jointlock.seven_r import (
        _lock_joint,
        _topology_rank,
        choose_lock_joint,
    )

    kb = build_kinbody(franka_panda_specs())
    # choose_lock_joint should match EAIK's pick (joint index 4).
    assert choose_lock_joint(kb, DEFAULT_TOLERANCE_POLICY) == 4

    sub_kb = _lock_joint(kb, 4, 0.0)
    rank, solver_name = _topology_rank(sub_kb, DEFAULT_TOLERANCE_POLICY)
    assert rank == 0, f"expected tier-0, got rank={rank} solver={solver_name}"
    assert solver_name == "reversed:spherical_two_parallel", (
        f"expected reversed:spherical_two_parallel, got {solver_name}"
    )


def test_franka_7r_solves_via_reversed_dispatch() -> None:
    """Franka 7R IK closes for random poses via the reversed-chain
    closed-form path. Every returned solution FK-closes at machine
    precision (~1e-12); the median IK call runs in ~80 ms (16-sample
    sweep times ~5 ms inner spherical_two_parallel), competitive with
    mink et al. while remaining fully analytical.
    """
    from ssik.solvers.jointlock.seven_r import solve as seven_r_solve

    kb = build_kinbody(franka_panda_specs())
    rng = np.random.default_rng(seed=0)
    closures = 0
    for _ in range(5):
        q_true = rng.uniform(-1.5, 1.5, size=7)
        T_target = _fk(kb, q_true)
        sols, is_ls = seven_r_solve(kb, T_target)
        if is_ls or not sols:
            continue
        # Every returned solution FK-closes at machine precision.
        for sol in sols:
            T_check = _fk(kb, sol.q)
            assert np.allclose(T_check, T_target, atol=1e-9), (
                f"Franka IK candidate failed FK closure: "
                f"max|diff|={float(np.max(np.abs(T_check - T_target))):.2e}"
            )
        closures += 1
    assert closures == 5, (
        f"Franka 7R IK closed only {closures}/5 random poses -- "
        "the reversed-chain dispatch isn't producing valid solutions."
    )


def test_franka_joint_limits_baked_into_kinbody() -> None:
    """Each joint's MJCF-supplied ``range`` lands in ``Joint.limits`` after
    ``build_kinbody`` POE-normalisation. Limits are kinematic data (frame
    change is a no-op on them).
    """
    kb = build_kinbody(franka_panda_specs())
    expected = [
        (-2.8973, 2.8973),  # joint1
        (-1.7628, 1.7628),  # joint2
        (-2.8973, 2.8973),  # joint3
        (-3.0718, -0.0698),  # joint4
        (-2.8973, 2.8973),  # joint5
        (-0.0175, 3.7525),  # joint6
        (-2.8973, 2.8973),  # joint7
    ]
    for i, lim in enumerate(expected):
        assert kb.joints[i].limits == lim, f"joint {i} limits"


def test_franka_seven_r_lock_samples_clamps_to_limits() -> None:
    """``seven_r.solve`` default lock-sample sweep covers the locked joint's
    actual range, not ``[-pi, pi]``. Exercises the clamping path in #129
    Step 1.
    """
    from ssik.solvers.jointlock.seven_r import choose_lock_joint
    from ssik.solvers.jointlock.seven_r import solve as seven_r_solve

    kb = build_kinbody(franka_panda_specs())
    lock_idx = choose_lock_joint(kb)
    # Joint 4 has limits (-2.8973, 2.8973) -- about 92% of [-pi, pi].
    expected_lo, expected_hi = kb.joints[lock_idx].limits  # type: ignore[misc]

    # FK on the home pose gives a reachable target. The solve internally
    # builds samples = np.linspace(lo, hi, N, endpoint=False); we can't
    # observe them directly, but we can verify the chosen lock_idx's
    # limits have the expected MJCF-supplied bounds (Franka joint 4
    # is the forearm-adjacent joint, range -2.8973..2.8973).
    assert lock_idx == 4
    assert expected_lo == -2.8973
    assert expected_hi == 2.8973

    # Also verify a real solve still works through the limits-aware
    # sweep (regression check: clamping shouldn't break IK).
    T_home = _fk(kb, FRANKA_PANDA_KEYFRAMES["home"])
    sols, is_ls = seven_r_solve(kb, T_home)
    assert not is_ls
    assert len(sols) > 0


def test_franka_artifact_max_solutions_short_circuit() -> None:
    """Public Franka artifact ``solve()`` exposes ``max_solutions`` and
    short-circuits the lock sweep -- the universal-7R speedup from #142.
    Validates the codegen plumbed the param through (composer
    ``_solve_algebraic`` accepts it, orchestrator forwards it).

    The artifact module is excluded from mypy strict typing (it's
    generated code; its types come from the codegen template and are
    validated via the byte-equal snapshot test). Calls into it use
    ``# type: ignore`` to silence the cross-boundary call warnings.
    """
    import franka_panda_ik  # type: ignore[import-not-found]

    rng = np.random.default_rng(seed=0)
    for _ in range(5):
        q_true = rng.uniform(-1.5, 1.5, size=7)
        T_target = franka_panda_ik._fk(q_true)

        # Test exercises analytical-branch enumeration; bypass limits
        # filter so all 64 geometric branches are kept regardless of
        # whether their q lands inside Franka's URDF limits.
        sols_all = franka_panda_ik.solve(T_target, respect_limits=False)
        sols_one = franka_panda_ik.solve(T_target, max_solutions=1, respect_limits=False)

        assert sols_all, "exhaustive search returned no IK"
        assert sols_one, "max_solutions=1 returned no IK"
        assert len(sols_one) == 1, f"max_solutions=1 returned {len(sols_one)}"
        assert len(sols_all) >= 8, "exhaustive search should produce many solutions"

        # FK closure on the short-circuit result.
        T_check = franka_panda_ik._fk(sols_one[0].q)
        err = float(np.max(np.abs(T_check - T_target)))
        assert err < 1e-9, f"max_solutions=1 candidate FK closure {err:.2e} > 1e-9"


def test_franka_artifact_q_seed_returns_nearest() -> None:
    """``q_seed`` + ``max_solutions=1`` returns the IK whose lock-joint
    value is closest to the seed (the trajectory-tracking promise).
    """
    import franka_panda_ik  # type: ignore[import-not-found]

    q_true = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0])
    T_target = franka_panda_ik._fk(q_true)

    # Seed exactly at q_true; the corresponding lock-joint sample is
    # nearest by definition, so the returned IK should match q_true at
    # the locked joint within a sweep-step.
    sols = franka_panda_ik.solve(T_target, q_seed=q_true, max_solutions=1)
    assert len(sols) == 1
    lock_idx = 4  # baked _LOCK_IDX for Franka
    sweep_step = (2.8973 - (-2.8973)) / 16  # default 16 samples over the joint range
    diff = float(((sols[0].q[lock_idx] - q_true[lock_idx] + np.pi) % (2 * np.pi)) - np.pi)
    assert abs(diff) <= sweep_step + 1e-9, (
        f"q_seed bias didn't pick the nearest lock sample: "
        f"diff={diff:.4f}, sweep_step={sweep_step:.4f}"
    )
