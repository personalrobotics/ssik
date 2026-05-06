r"""KUKA iiwa LBR 14 fixture validation.

iiwa is **SRS** (Spherical-Revolute-Spherical) topology with non-zero
inter-wrist-joint translations -- the wrist axes meet at a common point
but the joint origins are spread along that axis. The IK-Geo \`spherical\`
solver family can't handle this consolidation; the strict-coincidence
fix in :func:`ssik.kinematics.predicates.three_consecutive_intersecting`
(#155) keeps the dispatcher from silently mis-classifying iiwa as a
spherical-wrist arm.

iiwa IK currently routes through ``husty_pfurner.general_6r`` (HP's
universal-6R fallback in jointlock) at ~120 ms per inner 6R x 16 lock
samples = ~2 s per 7R IK. #143 (native SRS analytical solver) will
land the right path with sub-millisecond timing.

Pre-#176/#177: iiwa14 routed through the deleted ``gen_six_dof`` grid
search at 18+ seconds with **zero** solutions returned. The HP
perturbation path (#176) handles iiwa14's measure-zero Tv2 singularity
correctly, polishing each algebraic seed via 6-D LM to machine
precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from kuka_iiwa14 import KUKA_IIWA14_KEYFRAMES, kuka_iiwa14_specs

from ssik._kinbody import build_kinbody
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.predicates import (
    three_consecutive_intersecting,
)
from ssik.solvers.jointlock import seven_r


def test_iiwa14_fixture_builds_with_seven_revolute_joints() -> None:
    """The fixture transcribes 7 revolute joints with limits + the EE site."""
    specs = kuka_iiwa14_specs()
    assert len(specs) == 7
    for spec in specs:
        assert spec.joint_type == "revolute"
        assert spec.limits is not None
        lo, hi = spec.limits
        assert lo < 0 < hi  # all iiwa joints are symmetric around 0


def test_iiwa14_fk_at_home_is_along_z() -> None:
    """At ``q = 0`` the iiwa is fully extended along world +Z. The EE
    position should be at (0, 0, total_z_extent).
    """
    kb = build_kinbody(kuka_iiwa14_specs())
    T = poe_forward_kinematics(kb, KUKA_IIWA14_KEYFRAMES["home"])
    pos = T[:3, 3]
    # x and y at home should be exactly zero (or numerical noise).
    assert abs(pos[0]) < 1e-12
    assert abs(pos[1]) < 1e-12
    # z is the cumulative reach. iiwa14 spec: ~1.306 m fully extended.
    assert 1.30 < pos[2] < 1.31


def test_iiwa14_fk_round_trip() -> None:
    """FK is deterministic and stable across a handful of random q values."""
    kb = build_kinbody(kuka_iiwa14_specs())
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, size=7)
        T1 = poe_forward_kinematics(kb, q)
        T2 = poe_forward_kinematics(kb, q)
        assert np.allclose(T1, T2, atol=1e-15)
        # The pose is finite + the rotation block is orthogonal.
        assert np.all(np.isfinite(T1))
        R = T1[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_iiwa14_predicate_rejects_loose_spherical_wrist() -> None:
    """Regression test for #155: iiwa's wrist axes meet at a common point
    but the wrist joint origins are spread along the axis. The IK-Geo
    consolidation requires the *last two* origins to coincide with the
    intersection; iiwa violates this. The strict-coincidence predicate
    must return ``None`` (i.e., refuse to classify iiwa as having a
    spherical wrist).
    """
    kb = build_kinbody(kuka_iiwa14_specs())
    triple = three_consecutive_intersecting(kb.joints)
    assert triple is None, (
        f"three_consecutive_intersecting wrongly admitted iiwa as having a "
        f"spherical wrist at {triple}; this is the #155 regression."
    )


def test_iiwa14_chooses_a_real_lock_joint() -> None:
    """The lock-joint chooser still returns a valid index for iiwa even
    after the predicate tightening. Topology rank may be 2 or 3 (tier-1
    or tier-2 fallback) but should not error.
    """
    kb = build_kinbody(kuka_iiwa14_specs())
    lock_idx = seven_r.choose_lock_joint(kb)
    assert 0 <= lock_idx < 7


def test_iiwa14_seven_r_returns_cleanly_at_unreachable_target() -> None:
    """Sanity: a clearly-out-of-reach target (target far outside the
    work envelope) returns an empty solution set with ``is_ls=True``,
    not a crash. We use ``max_solutions=1`` so the solver short-circuits
    quickly even though iiwa's tier-2 fallback is otherwise slow.
    """
    kb = build_kinbody(kuka_iiwa14_specs())
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = 100.0  # 100 m away -- well outside iiwa's ~1.3 m reach
    sols, is_ls = seven_r.solve(kb, T, max_solutions=1)
    assert is_ls
    assert len(sols) == 0


def test_iiwa14_ik_via_hp_fallback() -> None:
    """With #143 (native SRS analytical solver) still absent, iiwa14 IK
    routes through ``husty_pfurner.general_6r`` for each post-lock 6R
    sub-chain. HP's perturbation path (#176) handles the symmetric-DH
    Tv2 case that arises in iiwa14's locked sub-chains; 6-D LM polish
    then converges each algebraic seed to machine precision.

    Per-IK budget: ~120 ms x 16 lock samples = ~2 s. Acceptable while
    #143 remains open; previously this path went through ``gen_six_dof``
    grid-search at 18+ seconds with 0 solutions returned.
    """
    kb = build_kinbody(kuka_iiwa14_specs())
    rng = np.random.default_rng(7)
    q_star = rng.uniform(-0.5, 0.5, size=7)
    T = poe_forward_kinematics(kb, q_star)
    sols, is_ls = seven_r.solve(kb, T, max_solutions=1, allow_refinement=True)
    assert sols, "iiwa14 reachable pose should produce at least one IK via HP fallback"
    assert not is_ls
    T_check = poe_forward_kinematics(kb, sols[0].q)
    assert np.allclose(T_check, T, atol=1e-6), (
        f"FK closure failed: max|diff|={np.max(np.abs(T_check - T)):.2e}"
    )
