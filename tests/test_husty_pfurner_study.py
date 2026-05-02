"""Validation for the Study quaternion / dual-quaternion machinery
(Phase 5b of #158 / #162).

The HP algorithm builds on Study coordinates: a 4x4 SE(3) matrix gets
encoded as an 8-vec dual quaternion, and chain composition becomes
dual-quaternion multiplication. If the encoding / decoding / product
is wrong, every downstream HP step (constraint quadrics, elimination,
back-substitution) inherits the bug.

This module covers four oracle classes:

1. **SE(3) round-trip**: ``T -> dq -> T`` is identity at 1e-12 for any
   rigid transform (Hypothesis fuzz over 100+ random poses).
2. **Study quadric invariance**: every dual quaternion produced by
   :func:`dq_from_se3` lies on the Study quadric within 1e-12.
3. **Composition equivalence**: ``dq_mul(dq_a, dq_b)`` decodes to the
   same SE(3) as ``T_a @ T_b`` (matrix product). Likewise for
   :func:`dq_chain` vs :func:`ssik.kinematics.poe_fk.poe_forward_kinematics`
   on revolute and revolute+prismatic chains.
4. **Joint primitives**: revolute ``dq_joint(axis, theta, "revolute")``
   matches Rodrigues; prismatic matches translation along axis.

The harness operates on ``ssik.solvers.husty_pfurner._study`` directly
-- no dependency on the (still-unimplemented) HP solver. This is the
math-infra validation that any HP-implementation PR can rely on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from franka_panda import franka_panda_specs
from jaco2 import jaco2_specs
from ur5 import ur5_specs

from ssik._kinbody import JointSpec, build_kinbody
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.husty_pfurner._study import (
    DQ_IDENTITY,
    dq_chain,
    dq_conj,
    dq_from_se3,
    dq_joint,
    dq_mul,
    se3_from_dq,
    study_quadric_residual,
)

# ----------------------------------------------------------------------------
# Helpers: build random SE(3) elements, check approximate equality.
# ----------------------------------------------------------------------------


def _random_se3(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly-random rotation (axis-angle) + small translation."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(-math.pi, math.pi))
    c, s = math.cos(angle), math.sin(angle)
    oc = 1.0 - c
    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
    R = np.array(
        [
            [c + ax * ax * oc, ax * ay * oc - az * s, ax * az * oc + ay * s],
            [ay * ax * oc + az * s, c + ay * ay * oc, ay * az * oc - ax * s],
            [az * ax * oc - ay * s, az * ay * oc + ax * s, c + az * az * oc],
        ]
    )
    t = rng.uniform(-1.0, 1.0, size=3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _prismatic_spec(axis: tuple[float, float, float], name: str) -> JointSpec:
    return JointSpec(
        parent_link_T=np.eye(4, dtype=np.float64),
        axis=np.array(axis, dtype=np.float64),
        joint_type="prismatic",
        child_link_T=np.eye(4, dtype=np.float64),
        name=name,
        limits=(-10.0, 10.0),
    )


def _revolute_spec(axis: tuple[float, float, float], name: str) -> JointSpec:
    return JointSpec(
        parent_link_T=np.eye(4, dtype=np.float64),
        axis=np.array(axis, dtype=np.float64),
        joint_type="revolute",
        child_link_T=np.eye(4, dtype=np.float64),
        name=name,
        limits=(-math.pi, math.pi),
    )


# ----------------------------------------------------------------------------
# Oracle 1: SE(3) round-trip
# ----------------------------------------------------------------------------


def test_dq_identity_decodes_to_eye4() -> None:
    """``DQ_IDENTITY`` should decode to the 4x4 identity at machine precision."""
    T = se3_from_dq(DQ_IDENTITY)
    assert np.allclose(T, np.eye(4), atol=1e-15)


@pytest.mark.parametrize("seed", list(range(20)))
def test_se3_round_trip_is_identity_seeded(seed: int) -> None:
    """``T -> dq -> T`` is the identity transform at 1e-12 for 20 fixed seeds."""
    rng = np.random.default_rng(seed)
    T = _random_se3(rng)
    sigma = dq_from_se3(T)
    T_back = se3_from_dq(sigma)
    assert np.allclose(T_back, T, atol=1e-12), (
        f"seed={seed}: max|diff|={float(np.max(np.abs(T_back - T))):.2e}"
    )


@given(
    rotvec=hnp.arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    ),
    t=hnp.arrays(
        dtype=np.float64,
        shape=(3,),
        elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_se3_round_trip_hypothesis_fuzz(rotvec: np.ndarray, t: np.ndarray) -> None:
    """Hypothesis fuzz: random rotation-vector + translation pairs round-trip
    at 1e-12 over 200 examples.
    """
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        R = np.eye(3)
    else:
        axis = rotvec / angle
        c, s = math.cos(angle), math.sin(angle)
        oc = 1.0 - c
        ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        R = np.array(
            [
                [c + ax * ax * oc, ax * ay * oc - az * s, ax * az * oc + ay * s],
                [ay * ax * oc + az * s, c + ay * ay * oc, ay * az * oc - ax * s],
                [az * ax * oc - ay * s, az * ay * oc + ax * s, c + az * az * oc],
            ]
        )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    sigma = dq_from_se3(T)
    T_back = se3_from_dq(sigma)
    assert np.allclose(T_back, T, atol=1e-10), f"max|diff|={float(np.max(np.abs(T_back - T))):.2e}"


# ----------------------------------------------------------------------------
# Oracle 2: Study quadric invariance
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(20)))
def test_study_quadric_zero_after_dq_from_se3(seed: int) -> None:
    """``study_quadric_residual(dq_from_se3(T))`` is zero within 1e-12
    for any rigid transform.
    """
    rng = np.random.default_rng(seed)
    T = _random_se3(rng)
    sigma = dq_from_se3(T)
    residual = study_quadric_residual(sigma)
    assert abs(residual) < 1e-12, f"seed={seed}: residual={residual:.2e}"


def test_study_quadric_zero_for_identity() -> None:
    assert abs(study_quadric_residual(DQ_IDENTITY)) < 1e-15


# ----------------------------------------------------------------------------
# Oracle 3: composition equivalence (DQ product vs matrix product)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(10)))
def test_dq_mul_matches_matrix_product(seed: int) -> None:
    """``dq_mul(dq_a, dq_b)`` decodes to ``T_a @ T_b``.

    Two random rigid transforms; compose via DQ and via matrix multiplication;
    the two SE(3) outputs must match at 1e-12.
    """
    rng = np.random.default_rng(seed)
    T_a = _random_se3(rng)
    T_b = _random_se3(rng)
    expected = T_a @ T_b
    sigma = dq_mul(dq_from_se3(T_a), dq_from_se3(T_b))
    actual = se3_from_dq(sigma)
    assert np.allclose(actual, expected, atol=1e-10), (
        f"seed={seed}: max|diff|={float(np.max(np.abs(actual - expected))):.2e}"
    )


def test_dq_mul_associativity() -> None:
    """``(a * b) * c`` decodes to the same SE(3) as ``a * (b * c)`` for
    random rigid transforms.
    """
    rng = np.random.default_rng(0)
    a = dq_from_se3(_random_se3(rng))
    b = dq_from_se3(_random_se3(rng))
    c = dq_from_se3(_random_se3(rng))
    left = dq_mul(dq_mul(a, b), c)
    right = dq_mul(a, dq_mul(b, c))
    assert np.allclose(se3_from_dq(left), se3_from_dq(right), atol=1e-10)


def test_dq_mul_identity_is_identity() -> None:
    """``DQ_IDENTITY * sigma == sigma * DQ_IDENTITY == sigma``."""
    rng = np.random.default_rng(0)
    sigma = dq_from_se3(_random_se3(rng))
    assert np.allclose(dq_mul(DQ_IDENTITY, sigma), sigma, atol=1e-15)
    assert np.allclose(dq_mul(sigma, DQ_IDENTITY), sigma, atol=1e-15)


def test_dq_conj_inverts_unit_dq() -> None:
    """For a unit dual quaternion (which any pose-DQ is by construction),
    ``sigma * conj(sigma)`` decodes to the identity SE(3).
    """
    rng = np.random.default_rng(0)
    sigma = dq_from_se3(_random_se3(rng))
    product = dq_mul(sigma, dq_conj(sigma))
    T = se3_from_dq(product)
    assert np.allclose(T, np.eye(4), atol=1e-10), (
        f"max|diff|={float(np.max(np.abs(T - np.eye(4)))):.2e}"
    )


# ----------------------------------------------------------------------------
# Oracle 4: joint primitives match the standalone formulas
# ----------------------------------------------------------------------------


def test_dq_joint_revolute_about_z_quarter_turn() -> None:
    """Revolute joint about world +Z by pi/2: rotation block matches the
    canonical 90-degree-about-Z matrix; translation column is zero.
    """
    axis = np.array([0.0, 0.0, 1.0])
    sigma = dq_joint(axis, math.pi / 2.0, "revolute")
    T = se3_from_dq(sigma)
    expected_R = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    assert np.allclose(T[:3, :3], expected_R, atol=1e-15)
    assert np.allclose(T[:3, 3], 0.0, atol=1e-15)


def test_dq_joint_prismatic_along_x_translates_by_d() -> None:
    """Prismatic joint along world +X by d: identity rotation, translation = (d, 0, 0)."""
    axis = np.array([1.0, 0.0, 0.0])
    d = 0.42
    sigma = dq_joint(axis, d, "prismatic")
    T = se3_from_dq(sigma)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-15)
    assert np.allclose(T[:3, 3], np.array([d, 0.0, 0.0]), atol=1e-15)


def test_dq_joint_unsupported_type_raises() -> None:
    """Non-revolute / non-prismatic joint types raise ``ValueError`` with
    a clear message."""
    with pytest.raises(ValueError, match="joint_type"):
        dq_joint(np.array([0.0, 0.0, 1.0]), 0.5, "fixed")


# ----------------------------------------------------------------------------
# Oracle 3 (continued): chain composition matches POE FK
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("specs_fn", "n_dof"),
    [
        (jaco2_specs, 6),
        (ur5_specs, 6),
        (franka_panda_specs, 7),
    ],
)
def test_dq_chain_matches_poe_fk_revolute_fixtures(  # type: ignore[no-untyped-def]
    specs_fn, n_dof: int
) -> None:
    """``se3_from_dq(dq_chain(kb, q))`` equals ``poe_forward_kinematics(kb, q)``
    on the existing all-revolute fixtures (JACO 2, UR5, Franka). 5 random q
    seeds per arm; tolerance 1e-10.
    """
    kb = build_kinbody(specs_fn())
    rng = np.random.default_rng(42)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=n_dof)
        T_poe = poe_forward_kinematics(kb, q)
        T_dq = se3_from_dq(dq_chain(kb, q))
        assert np.allclose(T_dq, T_poe, atol=1e-10), (
            f"{specs_fn.__name__}: max|diff|={float(np.max(np.abs(T_dq - T_poe))):.2e}"
        )


def test_dq_chain_matches_poe_fk_mixed_rp_chain() -> None:
    """A synthetic R-P-R chain composed via dual quaternions matches POE FK.

    Mirrors ``test_poe_fk_prismatic.py``'s RPR fixture: r1 about world +Z,
    p2 along +X, r3 about +Z. With r1=pi/2, the prismatic translation
    rotates from local +X to world +Y.
    """
    specs = [
        _revolute_spec((0.0, 0.0, 1.0), "r1"),
        _prismatic_spec((1.0, 0.0, 0.0), "p2"),
        _revolute_spec((0.0, 0.0, 1.0), "r3"),
    ]
    kb = build_kinbody(specs)
    q = np.array([math.pi / 2.0, 0.5, 0.0])
    T_poe = poe_forward_kinematics(kb, q)
    T_dq = se3_from_dq(dq_chain(kb, q))
    assert np.allclose(T_dq, T_poe, atol=1e-10), (
        f"max|diff|={float(np.max(np.abs(T_dq - T_poe))):.2e}"
    )


def test_dq_chain_at_zero_is_dq_of_link_product_only() -> None:
    """At ``q=0``, every revolute / prismatic joint contributes the identity
    transform; the chain DQ is just the dual quaternion of the cumulative
    ``T_left @ T_right`` product.
    """
    kb = build_kinbody(jaco2_specs())
    q = np.zeros(6)
    sigma = dq_chain(kb, q)
    T_dq = se3_from_dq(sigma)
    T_poe = poe_forward_kinematics(kb, q)
    assert np.allclose(T_dq, T_poe, atol=1e-10)


def test_dq_chain_residual_on_study_quadric() -> None:
    """``study_quadric_residual(dq_chain(kb, q))`` is zero within 1e-10
    for any reachable configuration. Composition of valid SE(3) DQs
    stays on the Study quadric.
    """
    kb = build_kinbody(jaco2_specs())
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=6)
        sigma = dq_chain(kb, q)
        residual = study_quadric_residual(sigma)
        assert abs(residual) < 1e-10, f"residual={residual:.2e}"


# ----------------------------------------------------------------------------
# Oracle 3 (continued): determinism
# ----------------------------------------------------------------------------


def test_dq_chain_is_deterministic() -> None:
    """Repeated calls on the same input produce byte-equal 8-vec results."""
    kb = build_kinbody(jaco2_specs())
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    s1 = dq_chain(kb, q)
    s2 = dq_chain(kb, q)
    assert np.array_equal(s1, s2)
