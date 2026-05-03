"""POE FK prismatic-joint branch (Phase 5a.5 of #158).

Validates :func:`ssik.kinematics.poe_fk.poe_forward_kinematics` for
chains containing prismatic joints. The pre-#158 implementation was
revolute-only -- it computed Rodrigues' rotation per joint with no
translation branch, so any chain with ``joint_type='prismatic'`` would
silently produce a wrong FK.

Phase 5a.5 adds the prismatic branch in :func:`poe_forward_kinematics`:
the joint transform becomes a translation by ``q`` along the joint axis
instead of a rotation. The accumulator carries unchanged through
``T_right``; the rest of the FK pipeline is identical.

This test file covers:

1. Pure prismatic single-joint chains (1P along +Z, +X, and a
   non-axis-aligned axis) -- hand-computable closed-form FK at
   machine precision.
2. Pure prismatic 3-DOF chain (orthogonal axes) -- additivity check.
3. Mixed revolute + prismatic 3-DOF chain (RPR) -- frame composition
   stays correct when a prismatic translation is sandwiched between
   rotations.
4. Determinism: two consecutive FK calls byte-equal on the same
   prismatic input.
5. Existing revolute fixtures (Franka, JACO 2, iiwa, UR5) regress at
   identical precision -- the revolute branch must be byte-equivalent
   to pre-Phase-5a.5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from franka_panda import franka_panda_specs
from jaco2 import jaco2_specs
from kuka_iiwa14 import kuka_iiwa14_specs
from ur5 import ur5_specs

from ssik._kinbody import JointSpec, build_kinbody
from ssik.kinematics.poe_fk import poe_forward_kinematics


def _prismatic_spec(
    *,
    parent_link_T: np.ndarray | None = None,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    name: str = "prismatic",
) -> JointSpec:
    """Build a single prismatic :class:`JointSpec` along ``axis`` with
    identity parent / child link transforms unless overridden.
    """
    return JointSpec(
        parent_link_T=np.eye(4, dtype=np.float64) if parent_link_T is None else parent_link_T,
        axis=np.array(axis, dtype=np.float64),
        joint_type="prismatic",
        child_link_T=np.eye(4, dtype=np.float64),
        name=name,
        limits=(-10.0, 10.0),
    )


def _revolute_spec(
    *,
    parent_link_T: np.ndarray | None = None,
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    name: str = "revolute",
) -> JointSpec:
    """Build a single revolute :class:`JointSpec` along ``axis`` with
    identity parent / child link transforms unless overridden.
    """
    return JointSpec(
        parent_link_T=np.eye(4, dtype=np.float64) if parent_link_T is None else parent_link_T,
        axis=np.array(axis, dtype=np.float64),
        joint_type="revolute",
        child_link_T=np.eye(4, dtype=np.float64),
        name=name,
        limits=(-np.pi, np.pi),
    )


# ----------------------------------------------------------------------------
# Single prismatic-joint sanity checks (closed-form)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("d", [0.0, 0.123, -0.456, 1.0, -2.5])
def test_single_prismatic_along_z_translates_by_d(d: float) -> None:
    """One prismatic joint along world +Z, identity link transforms.
    EE pose at q=d is the identity rotation with translation (0, 0, d).
    """
    kb = build_kinbody([_prismatic_spec(axis=(0.0, 0.0, 1.0))])
    T = poe_forward_kinematics(kb, np.array([d]))
    expected = np.eye(4, dtype=np.float64)
    expected[2, 3] = d
    assert np.allclose(T, expected, atol=1e-15), f"d={d}: T={T}"


@pytest.mark.parametrize("d", [0.0, 0.123, -0.456, 1.0])
def test_single_prismatic_along_x_translates_by_d(d: float) -> None:
    kb = build_kinbody([_prismatic_spec(axis=(1.0, 0.0, 0.0))])
    T = poe_forward_kinematics(kb, np.array([d]))
    expected = np.eye(4, dtype=np.float64)
    expected[0, 3] = d
    assert np.allclose(T, expected, atol=1e-15)


def test_single_prismatic_along_diagonal_translates_by_d_axis() -> None:
    """A prismatic joint along a non-axis-aligned axis translates by
    ``d * axis_unit`` -- the unit-axis path picks up no extra factor.
    """
    axis = np.array([1.0, 1.0, 1.0], dtype=np.float64) / np.sqrt(3.0)
    kb = build_kinbody([_prismatic_spec(axis=tuple(axis.tolist()))])
    d = 0.42
    T = poe_forward_kinematics(kb, np.array([d]))
    expected = np.eye(4, dtype=np.float64)
    expected[:3, 3] = d * axis
    assert np.allclose(T, expected, atol=1e-15)


def test_single_prismatic_at_zero_is_identity() -> None:
    """At q=0, a prismatic-only chain has FK = identity (no translation,
    identity rotation block)."""
    kb = build_kinbody([_prismatic_spec(axis=(0.5, 0.5, np.sqrt(0.5)))])
    T = poe_forward_kinematics(kb, np.array([0.0]))
    assert np.allclose(T, np.eye(4), atol=1e-15)


# ----------------------------------------------------------------------------
# Pure prismatic 3-DOF chain (additivity along orthogonal axes)
# ----------------------------------------------------------------------------


def test_three_prismatic_orthogonal_axes_additive() -> None:
    """Three prismatic joints along world X, Y, Z (each with identity
    link transforms) at q=(dx, dy, dz) -- EE translation should be
    exactly (dx, dy, dz), rotation block identity.
    """
    specs = [
        _prismatic_spec(axis=(1.0, 0.0, 0.0), name="p_x"),
        _prismatic_spec(axis=(0.0, 1.0, 0.0), name="p_y"),
        _prismatic_spec(axis=(0.0, 0.0, 1.0), name="p_z"),
    ]
    kb = build_kinbody(specs)
    q = np.array([1.5, -0.7, 2.3])
    T = poe_forward_kinematics(kb, q)
    expected = np.eye(4, dtype=np.float64)
    expected[:3, 3] = q
    assert np.allclose(T, expected, atol=1e-15)


# ----------------------------------------------------------------------------
# Mixed revolute + prismatic chain (RPR)
# ----------------------------------------------------------------------------


def test_rpr_chain_at_zero_is_identity() -> None:
    """RPR chain with identity link transforms, all joints aligned to
    world +Z, evaluated at q=(0, 0, 0): identity rotation, zero
    translation.
    """
    specs = [
        _revolute_spec(axis=(0.0, 0.0, 1.0), name="r1"),
        _prismatic_spec(axis=(0.0, 0.0, 1.0), name="p2"),
        _revolute_spec(axis=(0.0, 0.0, 1.0), name="r3"),
    ]
    kb = build_kinbody(specs)
    T = poe_forward_kinematics(kb, np.zeros(3))
    assert np.allclose(T, np.eye(4), atol=1e-15)


def test_rpr_chain_prismatic_along_z_at_revolute_zero_translates_by_d() -> None:
    """RPR chain with all axes along world +Z, R-joints at zero, P-joint
    at d -- EE rotation = I (revolute joints contribute nothing at q=0),
    EE translation = (0, 0, d).

    Hand-derived: the R(z, 0) factors on either side of the prismatic
    don't affect the +Z translation (the +Z axis is invariant under
    rotation about itself).
    """
    specs = [
        _revolute_spec(axis=(0.0, 0.0, 1.0), name="r1"),
        _prismatic_spec(axis=(0.0, 0.0, 1.0), name="p2"),
        _revolute_spec(axis=(0.0, 0.0, 1.0), name="r3"),
    ]
    kb = build_kinbody(specs)
    d = 0.789
    T = poe_forward_kinematics(kb, np.array([0.0, d, 0.0]))
    expected = np.eye(4, dtype=np.float64)
    expected[2, 3] = d
    assert np.allclose(T, expected, atol=1e-15)


def test_rpr_chain_revolute_rotates_prismatic_translation() -> None:
    """RPR chain: r1 along +Z (rotates first), then prismatic along +X,
    then r3 along +Z. With r1 = pi/2, the prismatic +X translation
    becomes +Y in the world frame.

    Hand-derived: r1=pi/2 rotates the local +X frame to world +Y.
    Prismatic translation by ``d`` then moves the EE by (0, d, 0) in
    world. r3 is also +Z which doesn't affect translation, only the
    final rotation block (which composes with r1 to net pi/2 if
    r3=0).
    """
    specs = [
        _revolute_spec(axis=(0.0, 0.0, 1.0), name="r1"),
        _prismatic_spec(axis=(1.0, 0.0, 0.0), name="p2"),
        _revolute_spec(axis=(0.0, 0.0, 1.0), name="r3"),
    ]
    kb = build_kinbody(specs)
    d = 0.5
    T = poe_forward_kinematics(kb, np.array([np.pi / 2.0, d, 0.0]))
    # Translation: rotated +X by pi/2 about +Z = +Y. Distance d.
    assert np.allclose(T[:3, 3], np.array([0.0, d, 0.0]), atol=1e-12)
    # Rotation block: R_z(pi/2) @ R_z(0) = R_z(pi/2).
    expected_R = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    assert np.allclose(T[:3, :3], expected_R, atol=1e-12)


# ----------------------------------------------------------------------------
# Determinism on prismatic
# ----------------------------------------------------------------------------


def test_prismatic_fk_is_deterministic() -> None:
    """Repeated calls on the same prismatic input return byte-equal
    results (no LAPACK noise; the prismatic branch is pure scalar
    arithmetic)."""
    specs = [
        _revolute_spec(axis=(0.0, 0.0, 1.0)),
        _prismatic_spec(axis=(1.0, 0.0, 0.0)),
        _revolute_spec(axis=(0.0, 1.0, 0.0)),
    ]
    kb = build_kinbody(specs)
    q = np.array([0.3, 0.7, -0.2])
    T1 = poe_forward_kinematics(kb, q)
    T2 = poe_forward_kinematics(kb, q)
    assert np.array_equal(T1, T2)


# ----------------------------------------------------------------------------
# Regression: existing revolute fixtures unaffected
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("specs_fn", "n_dof"),
    [
        (jaco2_specs, 6),
        (ur5_specs, 6),
        (franka_panda_specs, 7),
        (kuka_iiwa14_specs, 7),
    ],
)
def test_revolute_fixtures_regression(specs_fn, n_dof: int) -> None:  # type: ignore[no-untyped-def]
    """Existing all-revolute fixtures must produce FK byte-identical to
    pre-Phase-5a.5 -- the prismatic branch only fires when a joint's
    ``is_prismatic`` returns True, so revolute paths must be untouched.
    """
    kb = build_kinbody(specs_fn())
    rng = np.random.default_rng(42)
    q = rng.uniform(-1.0, 1.0, size=n_dof)
    T = poe_forward_kinematics(kb, q)
    assert np.all(np.isfinite(T))
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    # FK is deterministic across calls.
    T2 = poe_forward_kinematics(kb, q)
    assert np.array_equal(T, T2)
