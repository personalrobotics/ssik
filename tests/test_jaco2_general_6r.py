"""End-to-end JACO 2 (j2n6s200) round-trip via the general-6R solver.

JACO 2 is the canonical EAIK-gap target: 6-DOF non-Pieper, non-orthogonal
60-degree twists at joints 4-5. No three consecutive intersecting axes, no
parallel pair, so subproblem-composition solvers don't apply.

This test wires the real MJCF source-of-truth (transcribed in
:mod:`fixtures.jaco2`) through:

    KinBody -> poe_to_dh -> solve_all_ik -> theta -> q (POE frame)

and checks that ``solve(kb, FK(q*))`` recovers a solution that FK-closes
on the POE chain. AE-3 leftvar selection picks ``q_1`` for JACO 2,
collapsing ``cond(m_quad)`` from ~3.75e16 to ~127 (#70).

Marked ``slow`` because the first call triggers sympy preprocessing for
the leftvar selection (~30-100s cold cache); subsequent calls are
single-digit milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ssik._kinbody import KinBody, build_kinbody
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo import general_6r
from tests.fixtures.jaco2 import JACO2_KEYFRAMES, jaco2_specs


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
    T = np.eye(4)
    for joint, qi in zip(kb.joints, q, strict=True):
        T = T @ joint.T_left @ _rot_axis(joint.axis, float(qi)) @ joint.T_right
    return T


@pytest.fixture(scope="module")
def jaco2_kb() -> KinBody:
    return build_kinbody(jaco2_specs())


def test_jaco2_poe_to_dh_recovers_60deg_twists(jaco2_kb: KinBody) -> None:
    """The MJCF -> POE -> DH conversion should expose the 60-degree
    non-orthogonal twists at joints 4-5 that make JACO 2 non-Pieper.

    The published synthetic DH used in the diagnostic harness has
    ``|alpha| = (90, 180, 90, 60, 60, 180) deg``. Sign convention may
    differ, so we check magnitudes."""
    dh = poe_to_dh(jaco2_kb)
    deg = np.abs(dh.alpha) * 180.0 / np.pi
    # The two 60-degree twists are the JACO 2 fingerprint.
    sixty_deg_count = int(np.sum(np.isclose(deg, 60.0, atol=0.5)))
    assert sixty_deg_count == 2, (
        f"expected 2 alpha entries near 60 deg (joints 4-5); got |alpha|={deg}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0])
def test_jaco2_general_6r_recovers_seed(jaco2_kb: KinBody, seed: int) -> None:
    """``solve(kb, FK(q*))`` recovers a valid solution that FK-closes."""
    rng = np.random.default_rng(seed)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    T_star = _fk_poe(jaco2_kb, q_star)

    solutions, is_ls = general_6r.solve(jaco2_kb, T_star)
    assert not is_ls, f"general_6r returned no solution for JACO 2 seed={seed}, q_star={q_star}"
    assert len(solutions) >= 1

    for sol in solutions:
        T_check = _fk_poe(jaco2_kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-5), (
            f"FK round-trip violated; max|diff|={np.max(np.abs(T_check - T_star)):.3e}"
        )
        # JACO 2's AE-3 leftvar avoids the singular pencil; algebraic FK
        # is exact and refinement should not fire by default.
        assert sol.refinement_used == "none", sol

    def _wrap_max(q: NDArray[np.float64]) -> float:
        return float(
            max(
                abs(((float(qi - qs) + np.pi) % (2 * np.pi)) - np.pi)
                for qi, qs in zip(q, q_star, strict=True)
            )
        )

    best = min(_wrap_max(sol.q) for sol in solutions)
    assert best < 1e-3, f"seeded q* not recovered; best wrap-max diff = {best:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["above_plate", "resting", "staging", "stow"])
def test_jaco2_general_6r_at_keyframes(jaco2_kb: KinBody, name: str) -> None:
    """At each MJCF keyframe pose, FK -> IK round-trips on the POE chain."""
    q_star = JACO2_KEYFRAMES[name]
    T_star = _fk_poe(jaco2_kb, q_star)

    solutions, is_ls = general_6r.solve(jaco2_kb, T_star)
    assert not is_ls, f"general_6r returned no solution at keyframe {name!r}"
    for sol in solutions:
        T_check = _fk_poe(jaco2_kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-5)

    # The seeded keyframe must be among the recovered solutions (mod 2pi).
    def _wrap_max(q: NDArray[np.float64]) -> float:
        return float(
            max(
                abs(((float(qi - qs) + np.pi) % (2 * np.pi)) - np.pi)
                for qi, qs in zip(q, q_star, strict=True)
            )
        )

    best = min(_wrap_max(sol.q) for sol in solutions)
    assert best < 1e-3, f"keyframe {name!r} q* not recovered; best wrap-max diff = {best:.3e}"
