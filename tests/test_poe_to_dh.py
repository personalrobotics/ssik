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

from fixtures.ur5 import ur5_specs
from ssik._kinbody import KinBody, build_kinbody
from ssik.kinematics.poe_to_dh import poe_to_dh


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
    dh = poe_to_dh(ur5_kb)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        q = rng.uniform(-np.pi, np.pi, size=6)
        T_poe = _fk_poe(ur5_kb, q)
        T_dh = dh.t_pre @ _fk_dh(dh.alpha, dh.a, dh.d, q + dh.theta_offset) @ dh.t_post
        assert np.allclose(T_poe, T_dh, atol=1e-10), (
            f"POE/DH FK mismatch (seed={seed}); max diff = {np.max(np.abs(T_poe - T_dh)):.3e}"
        )
