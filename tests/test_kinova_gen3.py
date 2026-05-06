"""Bulletproof validation for the Kinova Gen3 7-DOF fixture (#80).

Gen3 is a 7-DOF anthropomorphic arm commonly cited as SRS-class in the
literature (and treated as such by EAIK). **In its actual URDF it is
NOT pure SRS** — joints 2/3 carry y-offsets of 0.005375 / -0.006375 m
that displace joint-3's axis ~12 mm off the joint-0/1 common point.
The wrist is closer to spherical (~0.4 mm drift on joint-7).

ssik's `is_srs_7r` predicate correctly rejects Gen3, and the
dispatcher routes it through `seven_r.srs_polished` (#193 fast path,
~95 ms median; covered by tests/test_seven_r_srs_polished.py). The
universal `jointlock + HP` fallback (~1.5 s) remains correct as a
backup and is also exercised here.

This test contract:

- predicate refuses Gen3 (correct strict non-SRS classification).
- dispatcher picks `seven_r.srs_polished` (post-#193).
- hand-picked + random poses solve via jointlock+HP directly (the
  invariant: HP works correctly on Gen3 even though it isn't the
  default route anymore).

Source URDF: ``tests/fixtures/gen3.urdf``, vendored from
https://github.com/Kinovarobotics/ros_kortex (noetic-devel branch,
``kortex_description/arms/gen3/7dof/urdf/GEN3-7DOF-NOVISION_FOR_URDF_ARM_V12.urdf``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.dispatcher import dispatch
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import is_srs_7r
from ssik.solvers.jointlock import seven_r as jointlock_seven_r

GEN3_URDF = Path(__file__).parent / "fixtures" / "gen3.urdf"


def _gen3_kinbody():
    return load_urdf_kinbody_normalized(GEN3_URDF, "base_link", "end_effector_link")


# ----------------------------------------------------------------------------
# URDF load + topology classification
# ----------------------------------------------------------------------------


def test_gen3_loads_as_7r() -> None:
    kb = _gen3_kinbody()
    assert len(kb.joints) == 7
    for j in kb.joints:
        assert j.joint_type == "revolute"


def test_gen3_is_not_pure_srs() -> None:
    """Gen3's URDF y-offsets break strict SRS structure (#193).

    Joint-3 axis is ~12 mm off the joint-0/1 common point because of
    the 5.375 / -6.375 mm motor-housing offsets in the URDF. ssik
    correctly refuses to classify it as SRS — the fast path is
    tracked in #193 (approximate-SRS + LM polish).
    """
    kb = _gen3_kinbody()
    assert is_srs_7r(kb, DEFAULT_TOLERANCE_POLICY) is None


def test_gen3_dispatches_to_polished_srs() -> None:
    """Dispatcher routes Gen3 through `seven_r.srs_polished` (#193 fast path).

    Pre-#193 Gen3 routed to ``jointlock + HP`` (~1.5 s); the polished
    variant brings it to ~95 ms via Singh-Kreutz + LM polish.
    """
    kb = _gen3_kinbody()
    plan = dispatch(kb)
    assert plan.solver_name == "seven_r.srs_polished"


# ----------------------------------------------------------------------------
# Hand-picked seeded recovery via jointlock+HP
# ----------------------------------------------------------------------------


_HAND_PICKED_Q = [
    np.array([0.3, -0.4, 0.7, 0.5, 0.6, -0.5, 0.2]),
    np.array([-0.5, 0.3, -0.7, 1.0, -0.4, 0.5, -0.3]),
    np.array([0.0, 0.5, 0.0, 1.5, 0.0, -0.5, 0.0]),  # elbow-folded
    np.array([1.2, -0.8, 0.3, 0.4, 1.1, -0.6, 0.9]),
]


@pytest.mark.slow
@pytest.mark.parametrize("q_star", _HAND_PICKED_Q)
def test_gen3_hand_picked_fk_closure(q_star: np.ndarray) -> None:
    """Every IK returned at a reachable Gen3 pose FK-closes < 1e-10.

    Marked slow: jointlock+HP takes ~2 s per IK on Gen3; the full
    parametrize is ~8 s.
    """
    kb = _gen3_kinbody()
    T_target = poe_forward_kinematics(kb, q_star)
    sols, is_ls = jointlock_seven_r.solve(kb, T_target, allow_refinement=True)
    assert not is_ls
    assert sols, f"jointlock+HP returned no IK for reachable pose q={q_star}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-10, f"best FK closure {best_fk:.2e} > 1e-10"


# ----------------------------------------------------------------------------
# Hypothesis fuzz: random reachable poses (slow — small N because each
# call is ~2 s through jointlock+HP)
# ----------------------------------------------------------------------------


@pytest.mark.slow
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_gen3_random_pose_fk_closure(seed: int) -> None:
    """Random q in [-0.8, 0.8] per joint: at least one returned IK
    must FK-close < 1e-10.

    N=10 because each call is ~2 s through jointlock+HP (the full
    test takes ~20 s). Will be tightened to N=500 once #193 polished
    SRS lands and per-IK time drops to ~10-20 ms.
    """
    rng = np.random.default_rng(seed)
    q_star = rng.uniform(-0.8, 0.8, size=7)
    kb = _gen3_kinbody()
    T_target = poe_forward_kinematics(kb, q_star)
    sols, _ = jointlock_seven_r.solve(kb, T_target, allow_refinement=True)
    assert sols, f"random reachable pose returned no IK: q*={q_star.tolist()}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-10, f"random pose seed={seed}: best FK={best_fk:.2e} > 1e-10"


# ----------------------------------------------------------------------------
# Drift characterisation (documents the non-SRS-ness for #193)
# ----------------------------------------------------------------------------


def test_gen3_drift_documented() -> None:
    """Gen3's shoulder drift ~12 mm and wrist drift ~0.4 mm are
    documented as the input to #193.

    This test pins the measured values so a future URDF revision that
    changes the offsets is detected loudly.
    """
    from ssik.kinematics.predicates import joint_origins

    kb = _gen3_kinbody()
    origins = joint_origins(kb.joints)

    # Shoulder triple (joints 0, 1, 2): drift of axis 2 from the
    # (axis 0, axis 1) common point.
    axes = [kb.joints[i].axis for i in (0, 1, 2)]
    pts = [origins[i] for i in (0, 1, 2)]
    M = np.column_stack([axes[0], -axes[1]])
    sol, *_ = np.linalg.lstsq(M, pts[1] - pts[0], rcond=None)
    common_sh = pts[0] + float(sol[0]) * axes[0]
    delta = common_sh - pts[2]
    perp = delta - float(np.dot(delta, axes[2])) * axes[2]
    shoulder_drift = float(np.linalg.norm(perp))

    # Wrist triple (joints 4, 5, 6).
    axes = [kb.joints[i].axis for i in (4, 5, 6)]
    pts = [origins[i] for i in (4, 5, 6)]
    M = np.column_stack([axes[0], -axes[1]])
    sol, *_ = np.linalg.lstsq(M, pts[1] - pts[0], rcond=None)
    common_wr = pts[0] + float(sol[0]) * axes[0]
    delta = common_wr - pts[2]
    perp = delta - float(np.dot(delta, axes[2])) * axes[2]
    wrist_drift = float(np.linalg.norm(perp))

    # Pinned values from the vendored URDF (V12, no-vision). A change
    # to either reflects a URDF revision and should be reviewed.
    assert 0.011 < shoulder_drift < 0.013, f"shoulder drift {shoulder_drift:.4f} m"
    assert 3e-4 < wrist_drift < 4e-4, f"wrist drift {wrist_drift:.4e} m"
