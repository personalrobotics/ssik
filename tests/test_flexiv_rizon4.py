"""Bulletproof validation for the Flexiv Rizon 4 7-DOF fixture (#80).

Rizon 4 is a 7-DOF arm cited as SRS-class in the literature (and
treated as such by EAIK). **In its actual URDF the wrist is not even
approximately spherical** — joint-7's axis line is ~151 mm off the
joint-4/5 common point. The shoulder also has a larger offset (~65 mm)
than Gen3's (~12 mm).

The wrist drift in particular is **outside Newton's basin of
attraction** (~3-5 cm in task space empirically). This means Rizon 4
will likely be **refused** by the future #193 polished-SRS solver —
its drift gate caps at ~40 mm. Rizon 4 stays on the universal
jointlock+HP fallback unless a dedicated non-spherical-wrist 7R
solver lands.

ssik's `is_srs_7r` predicate correctly rejects Rizon 4, and the
dispatcher routes it through `jointlock + HP` for slow-but-correct
IK.

Source URDF: ``tests/fixtures/rizon4.urdf``, hand-rendered from the
xacro + YAML configs at
https://github.com/flexivrobotics/flexiv_description (humble branch):

- ``urdf/common/rizon_arm.xacro`` — joint chain + axis layout
- ``config/Rizon4/default_kinematics.yaml`` — per-joint xyz/rpy
- ``config/Rizon4/joint_limits.yaml`` — lower/upper/effort/velocity
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

RIZON4_URDF = Path(__file__).parent / "fixtures" / "rizon4.urdf"


def _rizon4_kinbody():
    return load_urdf_kinbody_normalized(RIZON4_URDF, "base_link", "flange")


# ----------------------------------------------------------------------------
# URDF load + topology classification
# ----------------------------------------------------------------------------


def test_rizon4_loads_as_7r() -> None:
    kb = _rizon4_kinbody()
    assert len(kb.joints) == 7
    for j in kb.joints:
        assert j.joint_type == "revolute"


def test_rizon4_is_not_pure_srs() -> None:
    """Rizon 4's wrist is not even approximately spherical (~151 mm
    drift), and the shoulder has ~65 mm offsets. ssik correctly
    refuses to classify it as SRS — and #193's drift gate at ~40 mm
    is expected to refuse Rizon 4 too. Universal fallback stays the
    only correct path.
    """
    kb = _rizon4_kinbody()
    assert is_srs_7r(kb, DEFAULT_TOLERANCE_POLICY) is None


def test_rizon4_dispatches_to_jointlock_hp() -> None:
    """Dispatcher routes Rizon 4 through jointlock+HP (universal fallback)."""
    kb = _rizon4_kinbody()
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
def test_rizon4_hand_picked_fk_closure(q_star: np.ndarray) -> None:
    """Every IK returned at a reachable Rizon 4 pose FK-closes < 1e-10.

    Marked slow: jointlock+HP takes ~1.5 s per IK on Rizon 4.
    """
    kb = _rizon4_kinbody()
    T_target = poe_forward_kinematics(kb, q_star)
    sols, is_ls = jointlock_seven_r.solve(kb, T_target, allow_refinement=True)
    assert not is_ls
    assert sols, f"jointlock+HP returned no IK for reachable pose q={q_star}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-10, f"best FK closure {best_fk:.2e} > 1e-10"


# ----------------------------------------------------------------------------
# Hypothesis fuzz: small N because each call is ~1.5 s
# ----------------------------------------------------------------------------


@pytest.mark.slow
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_rizon4_random_pose_fk_closure(seed: int) -> None:
    """Random q in [-0.8, 0.8] per joint: at least one returned IK
    must FK-close < 1e-10. N=10 because each call is ~1.5 s.
    """
    rng = np.random.default_rng(seed)
    q_star = rng.uniform(-0.8, 0.8, size=7)
    kb = _rizon4_kinbody()
    T_target = poe_forward_kinematics(kb, q_star)
    sols, _ = jointlock_seven_r.solve(kb, T_target, allow_refinement=True)
    assert sols, f"random reachable pose returned no IK: q*={q_star.tolist()}"
    best_fk = min(s.fk_residual for s in sols)
    assert best_fk < 1e-10, f"random pose seed={seed}: best FK={best_fk:.2e} > 1e-10"


# ----------------------------------------------------------------------------
# Drift characterisation (input to #193 — drift gate decides whether
# Rizon 4 is a viable target for polished-SRS or stays on jointlock+HP)
# ----------------------------------------------------------------------------


def test_rizon4_drift_documented() -> None:
    """Pin the measured drift values: shoulder ~65 mm, wrist ~151 mm.

    The wrist drift is outside Newton's basin of attraction (~3-5 cm
    task space). #193 will refuse Rizon 4 unless a dedicated
    non-spherical-wrist 7R solver is added.

    A future YAML revision that changes these values fails this test
    loudly.
    """
    from ssik.kinematics.predicates import joint_origins

    kb = _rizon4_kinbody()
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

    # Pinned ranges from the rendered URDF (Humble-branch YAML).
    assert 0.060 < shoulder_drift < 0.070, f"shoulder drift {shoulder_drift:.4f} m"
    assert 0.145 < wrist_drift < 0.155, f"wrist drift {wrist_drift:.4f} m"
