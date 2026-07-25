"""POE -> DH conversion validation.

Round-trip test: for each KinBody fixture, the converted DH parameters must
satisfy

    FK_POE(kb, q*) == FK_DH(alpha, a, d, q* + theta_offset)

at machine precision (1e-12) for 100 random q*.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ssik._kinbody import JointSpec, KinBody, build_kinbody
from ssik.kinematics.poe_to_dh import poe_to_dh
from tests.fixtures.ur5 import ur5_specs


def _rot_axis(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    axis = axis / np.linalg.norm(axis)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x, y, z = axis
    oc = 1.0 - c
    return np.array(
        [
            [c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s, 0],
            [y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s, 0],
            [z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def _fk_poe(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    """POE forward kinematics from a KinBody."""
    T = np.eye(4)
    for joint, qi in zip(kb.joints, q, strict=True):
        T = T @ joint.T_left @ _rot_axis(joint.axis, float(qi)) @ joint.T_right
    return T


def _fk_dh(
    alpha: NDArray[np.float64],
    a: NDArray[np.float64],
    d: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Spong distal-DH FK."""
    T = np.eye(4)
    for i in range(len(alpha)):
        c, s = float(np.cos(theta[i])), float(np.sin(theta[i]))
        ca, sa = float(np.cos(alpha[i])), float(np.sin(alpha[i]))
        A_i = np.array(
            [
                [c, -s * ca, s * sa, a[i] * c],
                [s, c * ca, -c * sa, a[i] * s],
                [0, sa, ca, d[i]],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        T = T @ A_i
    return T


@pytest.fixture(scope="module")
def ur5_kb() -> KinBody:
    return build_kinbody(ur5_specs())


def test_poe_to_dh_ur5_alpha_magnitudes(ur5_kb: KinBody) -> None:
    """The conversion of UR5's KinBody should recover the published UR5 alpha
    magnitudes (sign may differ due to perpendicular-direction convention at
    intersecting axes; what matters is FK round-trip, validated separately)."""
    dh = poe_to_dh(ur5_kb)
    expected_alpha_abs = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.0])
    np.testing.assert_allclose(np.abs(dh.alpha), expected_alpha_abs, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_poe_to_dh_fk_roundtrip_ur5(ur5_kb: KinBody, seed: int) -> None:
    """FK_POE(kb, q*) == T_pre @ FK_DH(dh, q* + theta_offset) @ T_post at machine precision."""
    _assert_invariant(ur5_kb, seed)


def _assert_invariant(kb: KinBody, seed: int, n: int = 100) -> None:
    """The core poe_to_dh contract, over ``n`` random poses at 1e-10:

    FK_POE(kb, q) == T_pre @ FK_DH(alpha, a, d, q + theta_offset) @ T_post
    """
    dh = poe_to_dh(kb)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, size=len(kb.joints))
        T_poe = _fk_poe(kb, q)
        T_dh = dh.t_pre @ _fk_dh(dh.alpha, dh.a, dh.d, q + dh.theta_offset) @ dh.t_post
        assert np.allclose(T_poe, T_dh, atol=1e-10), (
            f"POE/DH invariant broken (seed={seed}); max diff = {np.max(np.abs(T_poe - T_dh)):.3e}"
        )


def _post_rotation_dh(a: float, alpha: float, d: float) -> NDArray[np.float64]:
    """``Trans_z(d) Trans_x(a) Rot_x(alpha)`` -- the fixed part of a DH link."""
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = d
    Tx = np.eye(4, dtype=np.float64)
    Tx[0, 3] = a
    return T @ Tx @ _rot_axis(np.array([1.0, 0.0, 0.0]), alpha)


def _dh_kb(rows: list[tuple[float, float, float]]) -> KinBody:
    """Build a 6R KinBody from ``(a, alpha, d)`` DH rows (theta=0)."""
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    specs = [
        JointSpec(
            parent_link_T=np.eye(4, dtype=np.float64),
            axis=z,
            joint_type="revolute",
            child_link_T=_post_rotation_dh(a, alpha, d),
            name=f"j{i}",
        )
        for i, (a, alpha, d) in enumerate(rows)
    ]
    return build_kinbody(specs)


# The terminal joint's twist ``alpha[5]`` sets the angle between joint-6's axis
# and the flange z-axis: 0/±pi are parallel/anti-parallel wrists (all currently
# shipped RR arms), everything in between is a *skew* wrist. With a non-zero
# terminal ``a[5]`` (flange offset perpendicular to joint 6) the skew case is
# exactly the FANUC CRX non-/L geometry that #418 broke: the old terminal-frame
# construction violated the DH constraint ``x_n ⟂ z_{n-1}`` and the invariant
# failed by O(1). This sweep is the self-contained guard.
@pytest.mark.parametrize(
    "alpha6",
    [
        0.0,
        np.pi / 6,
        np.pi / 4,
        np.pi / 3,
        np.pi / 2,
        2 * np.pi / 3,
        3 * np.pi / 4,
        np.pi,
        -np.pi / 2,
    ],
)
def test_poe_to_dh_invariant_terminal_wrist_sweep(alpha6: float) -> None:
    rows = [
        (0.10, np.pi / 2, 0.30),
        (0.40, 0.0, 0.0),
        (0.35, 0.0, 0.0),
        (0.0, np.pi / 2, 0.15),
        (0.0, -np.pi / 2, 0.10),
        (0.12, alpha6, 0.16),  # offset + arbitrary twist terminal wrist
    ]
    _assert_invariant(_dh_kb(rows), seed=3)


# Real-arm regression: every shipped RR (tier-2) arm must satisfy the invariant,
# including the FANUC CRX-10iA base -- the concrete skew-wrist geometry that
# regressed (#418; joint 6 ⟂ flange with the flange origin on joint 6's axis,
# a projection degeneracy the synthetic sweep above does not reproduce).
_RR_FIXTURES = [
    ("xarm6", "xarm6.urdf", "link_base", "link_eef"),
    ("piper", "piper.urdf", "base_link", "link6"),
    ("yam", "yam.urdf", "base_link", "link_6"),
    ("crx10ial", "fanuc_crx10ial.urdf", "base_link", "tool0"),
    ("fanuc_crx10ia", "fanuc_crx10ia.urdf", "base_link", "tool0"),
]


@pytest.mark.parametrize(("name", "fixture", "base", "ee"), _RR_FIXTURES)
def test_poe_to_dh_invariant_rr_arms(name: str, fixture: str, base: str, ee: str) -> None:
    from pathlib import Path

    from ssik._urdf import load_urdf_kinbody_normalized

    path = Path(__file__).resolve().parent / "fixtures" / fixture
    if not path.is_file():
        pytest.skip(f"fixture {fixture} not present")
    _assert_invariant(load_urdf_kinbody_normalized(path, base, ee), seed=5)
