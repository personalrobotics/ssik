"""Bulletproof validation for the Kassow Robots KR810 7-DOF fixture (#80).

KR810 is a 7-DOF arm in the Kassow KR* series (KR810 / KR1018 /
KR1410 / KR1805). Often grouped with the SRS family in the 7R
literature, but its **URDF carries motor-housing offsets that put
both shoulder and wrist well outside strict SRS structure**:

  * shoulder drift ~86 mm (joint-3 axis displaced from the
    joint-0/1 common point)
  * wrist drift ~111 mm (joint-7 axis displaced from the joint-4/5
    common point)

Both drifts exceed Newton's basin of attraction (~3-5 cm task
space), so the future #193 polished-SRS solver is expected to
**refuse** KR810. The universal `jointlock + HP` fallback
(~1.5 s per IK on M3) is the only correct path unless a dedicated
non-spherical-wrist 7R solver lands later.

ssik's `is_srs_7r` predicate correctly rejects KR810; dispatcher
routes through `jointlock + HP`.

Source URDF: ``tests/fixtures/kassow_kr810.urdf``, hand-rendered
from the xacro at https://github.com/rcruzoliver/kr_ros2 (main
branch), ``kr_robot_description/kr810/urdf/kr810_description.urdf.xacro``.
The xacro represents the chain with phantom fixed links between
active joints (carrying motor-housing offsets); the flat URDF
inlines those into each active joint's origin.
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

KR810_URDF = Path(__file__).parent / "fixtures" / "kassow_kr810.urdf"


def _kr810_kinbody():
    return load_urdf_kinbody_normalized(KR810_URDF, "base", "end_effector")


# ----------------------------------------------------------------------------
# URDF load + topology classification
# ----------------------------------------------------------------------------


def test_kr810_loads_as_7r() -> None:
    kb = _kr810_kinbody()
    assert len(kb.joints) == 7
    for j in kb.joints:
        assert j.joint_type == "revolute"


def test_kr810_is_not_pure_srs() -> None:
    """KR810's shoulder + wrist offsets break strict SRS structure
    by enough that #193's polished-SRS path is also expected to
    refuse it (drift gate ~40 mm).
    """
    kb = _kr810_kinbody()
    assert is_srs_7r(kb, DEFAULT_TOLERANCE_POLICY) is None


def test_kr810_dispatches_to_jointlock_hp() -> None:
    """Dispatcher routes KR810 through jointlock+HP (universal fallback)."""
    kb = _kr810_kinbody()
    plan = dispatch(kb)
    assert plan.solver_name == "jointlock.seven_r"


# ----------------------------------------------------------------------------
# Hand-picked seeded recovery via jointlock+HP
# ----------------------------------------------------------------------------


_HAND_PICKED_Q = [
    np.array([0.3, -0.4, 0.7, 0.5, 0.6, -0.5, 0.2]),
    np.array([-0.5, 0.3, -0.7, 1.0, -0.4, 0.5, -0.3]),
    np.array([0.0, 0.5, 0.0, 1.5, 0.0, -0.5, 0.0]),
    np.array([1.2, -0.8, 0.3, 0.4, 1.1, -0.6, 0.9]),
]


@pytest.mark.slow
@pytest.mark.parametrize("q_star", _HAND_PICKED_Q)
def test_kr810_hand_picked_fk_closure(q_star: np.ndarray) -> None:
    """Every IK returned at a reachable KR810 pose FK-closes < 1e-10.

    Marked slow: jointlock+HP takes ~1.5 s per IK.
    """
    kb = _kr810_kinbody()
    T_target = poe_forward_kinematics(kb, q_star)
    sols, is_ls = jointlock_seven_r.solve(kb, T_target, allow_refinement=True)
    assert not is_ls
    assert sols, f"jointlock+HP returned no IK for reachable pose q={q_star}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-10, f"best FK closure {best_fk:.2e} > 1e-10"


@pytest.mark.slow
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_kr810_random_pose_fk_closure(seed: int) -> None:
    """Random q in [-0.8, 0.8] per joint: at least one returned IK
    must FK-close < 1e-10. N=10 because each call is ~1.5 s.
    """
    rng = np.random.default_rng(seed)
    q_star = rng.uniform(-0.8, 0.8, size=7)
    kb = _kr810_kinbody()
    T_target = poe_forward_kinematics(kb, q_star)
    sols, _ = jointlock_seven_r.solve(kb, T_target, allow_refinement=True)
    assert sols, f"random reachable pose returned no IK: q*={q_star.tolist()}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-10, f"random pose seed={seed}: best FK={best_fk:.2e} > 1e-10"


# ----------------------------------------------------------------------------
# Drift characterisation
# ----------------------------------------------------------------------------


def test_kr810_drift_documented() -> None:
    """Pin shoulder ~86 mm, wrist ~111 mm. Both exceed Newton's
    basin (~3-5 cm), so #193 polished-SRS will refuse KR810. A
    future xacro revision changing these values fails this test
    loudly.
    """
    from ssik.kinematics.predicates import joint_origins

    kb = _kr810_kinbody()
    origins = joint_origins(kb.joints)

    def _drift(idxs: tuple[int, int, int]) -> float:
        axes = [kb.joints[i].axis for i in idxs]
        pts = [origins[i] for i in idxs]
        M = np.column_stack([axes[0], -axes[1]])
        sol, *_ = np.linalg.lstsq(M, pts[1] - pts[0], rcond=None)
        common = pts[0] + float(sol[0]) * axes[0]
        delta = common - pts[2]
        perp = delta - float(np.dot(delta, axes[2])) * axes[2]
        return float(np.linalg.norm(perp))

    shoulder_drift = _drift((0, 1, 2))
    wrist_drift = _drift((4, 5, 6))

    assert 0.080 < shoulder_drift < 0.090, f"shoulder drift {shoulder_drift:.4f} m"
    assert 0.105 < wrist_drift < 0.115, f"wrist drift {wrist_drift:.4f} m"
