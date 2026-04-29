"""Franka Panda fixture validation + topology baseline (#121).

The fixture itself is a transcribed MJCF (see ``tests/fixtures/franka_panda.py``).
This test validates two things:

1. The fixture builds correctly: FK at the documented home pose matches
   the Franka spec, and ``build_kinbody`` produces a POE-normalised
   chain (axes in the base frame).
2. The current state of the topology-rank dispatch on Franka post-lock-4.
   This is recorded as ``xfail`` so the suite keeps running; when #121
   Step 2 lands and the topology rank can dispatch the spherical-wrist-
   at-base case, this test will start passing and the xfail is removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from numpy.typing import NDArray

from ssik._kinbody import KinBody, build_kinbody
from ssik.kinematics._scalar3 import _mat4_mat4
from ssik.subproblems._rotation import rotation_matrix

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

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


@pytest.mark.xfail(
    reason=(
        "Franka post-lock-4 has a spherical wrist at the chain BASE (joints "
        "0,1,2 all pass through (0,0,0.333)) plus joints (3,4) parallel. "
        "EAIK calls this REVERSED + SPHERICAL_SECOND_TWO_PARALLEL after "
        "chain reversal. ssik's `_topology_rank` only matches spherical "
        "wrists at the END of the sub-chain (positions 3,4,5); generalising "
        "it (chain-reversal pre-pass + position-flexible dispatch) is "
        "tracked as #121 Step 2."
    ),
    strict=True,
)
def test_franka_dispatches_to_spherical_wrist_class() -> None:
    """When #121 Step 2 lands, Franka should dispatch to a tier-0 closed-form
    solver (some spherical-wrist class) for the post-lock-4 sub-chain.

    Today it falls through to ``gen_six_dof`` or ``two_parallel`` with the
    inner solver returning ``is_ls=True`` for every pose -- documented here
    via xfail so the suite keeps running until the fix lands.
    """
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.solvers.jointlock.seven_r import _lock_joint, _topology_rank

    kb = build_kinbody(franka_panda_specs())
    sub_kb = _lock_joint(kb, 4, 0.0)
    rank, solver_name = _topology_rank(sub_kb, DEFAULT_TOLERANCE_POLICY)
    # When fixed, this should land on tier-0 (rank 0, closed-form).
    assert rank == 0, f"expected tier-0, got rank={rank} solver={solver_name}"
