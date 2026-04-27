"""KinBody-input round-trip for :mod:`ssik.solvers.ikgeo.general_6r`.

Validates that the wrapper

  KinBody --(poe_to_dh)--> DhWithOffset --(solve_all_ik)--> theta -> q

closes round-trip at machine precision for a known-good arm (UR5).

UR5 is a Pieper-class arm (spherical wrist + two-parallel shoulder), so this
test is intentionally easy for the general-6R solver. The harder targets
(Kinova JACO 2 with 60-deg twists, MC Table I) live in
:mod:`tests.test_raghavan_roth_pq`. This test exists to catch regressions in
the *KinBody-input bridge* (poe_to_dh + the t_pre/t_post inverse on the
target + theta_offset removal) -- not in the underlying numeric pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ssik._kinbody import KinBody, build_kinbody
from ssik.solvers.ikgeo import general_6r
from fixtures.ur5 import ur5_specs


def _rot_axis(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    axis = axis / np.linalg.norm(axis)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x, y, z = axis
    oc = 1.0 - c
    return np.array(
        [
            [c + x*x*oc, x*y*oc - z*s, x*z*oc + y*s, 0],
            [y*x*oc + z*s, c + y*y*oc, y*z*oc - x*s, 0],
            [z*x*oc - y*s, z*y*oc + x*s, c + z*z*oc, 0],
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
def ur5_kb() -> KinBody:
    return build_kinbody(ur5_specs())


@pytest.mark.slow
def test_general_6r_ur5_recovers_seed(ur5_kb: KinBody) -> None:
    """``solve(kb, FK(q*))`` returns at least one solution and recovers q*."""
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    T_star = _fk_poe(ur5_kb, q_star)

    solutions, is_ls = general_6r.solve(ur5_kb, T_star)
    assert not is_ls, "general_6r should find at least one solution for a feasible UR5 pose"
    assert len(solutions) >= 1

    for sol in solutions:
        T_check = _fk_poe(ur5_kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-5), (
            f"FK round-trip violated; max|diff|={np.max(np.abs(T_check - T_star)):.3e}"
        )
        # Default-off refinement: pure-algebraic path on UR5.
        assert sol.refinement_used == "none", sol
        assert sol.solver_name == "ikgeo.general_6r"

    def _wrap_max(q: NDArray[np.float64]) -> float:
        return float(max(abs(((float(qi - qs) + np.pi) % (2*np.pi)) - np.pi)
                         for qi, qs in zip(q, q_star, strict=True)))
    best = min(_wrap_max(sol.q) for sol in solutions)
    assert best < 1e-3, f"seeded q* not recovered; best wrap-max diff = {best:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42])
def test_general_6r_ur5_fk_roundtrip(ur5_kb: KinBody, seed: int) -> None:
    """For 10 random q*, every returned solution FK-closes on the POE chain."""
    rng = np.random.default_rng(seed)
    for _ in range(10):
        q_star = rng.uniform(-1.0, 1.0, size=6)
        T_star = _fk_poe(ur5_kb, q_star)
        solutions, is_ls = general_6r.solve(ur5_kb, T_star)
        assert not is_ls, f"no solution for seed={seed}, q_star={q_star}"
        for sol in solutions:
            T_check = _fk_poe(ur5_kb, sol.q)
            assert np.allclose(T_check, T_star, atol=1e-5)


def test_general_6r_wrong_dof_raises(ur5_kb: KinBody) -> None:
    short_kb = KinBody(links=ur5_kb.links[:5], joints=ur5_kb.joints[:4])
    with pytest.raises(ValueError, match="6-DOF"):
        general_6r.solve(short_kb, np.eye(4))
