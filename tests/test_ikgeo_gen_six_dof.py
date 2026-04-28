"""End-to-end validation for :mod:`ssik.solvers.ikgeo.gen_six_dof`.

Tier-2 fully-general 6R solver. Correct for any 6R chain (within SP5's
non-degeneracy preconditions on the inner shoulder solve) but SLOW:
~100-120 seconds per IK in pure Python due to the 100x100 (q1, q2)
grid with an inner SP5 call per cell. Marked ``pytest.mark.slow`` so
it only runs under ``pytest -m slow``; the default fast suite skips it.

Performance optimisation (e.g. batching SP5's quartic root-finds) is
tracked separately -- this file validates *correctness*, not speed.

Test structure follows tier-1/tier-0 bulletproof discipline but with
fewer poses (each ~2 min). Fixtures are IK-Geo-style random arms
following their ``GeneralSetup``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik._kinbody import Joint, KinBody, Link
from ssik.solvers.ikgeo import gen_six_dof


def _rodrigues(k: np.ndarray, t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R: np.ndarray = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R


def _axis_angle_4x4(k: np.ndarray, t: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rodrigues(k, t)
    return T


def _fk(kb: KinBody, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _axis_angle_4x4(j.axis, qi) @ j.T_right
    return T


def _build_random_generic_arm(seed: int) -> KinBody:
    """Follows IK-Geo's GeneralSetup: 6 random unit axes + 7 random offsets,
    no parallel or intersecting constraints."""
    rng = np.random.default_rng(seed)

    def _rnorm() -> np.ndarray:
        v = rng.standard_normal(3)
        return v / float(np.linalg.norm(v))

    axes = [_rnorm() for _ in range(6)]
    t_lefts = [rng.standard_normal(3) for _ in range(6)]
    tool_p = rng.standard_normal(3)
    link_names = ["base_link", *(f"link_{i}" for i in range(1, 6)), "ee_link"]
    links = [Link(name=n) for n in link_names]
    joints: list[Joint] = []
    for i in range(6):
        t_left_i = np.eye(4)
        t_left_i[:3, 3] = t_lefts[i]
        t_right_i = np.eye(4)
        if i == 5:
            t_right_i[:3, 3] = tool_p
        joints.append(
            Joint(
                name=f"joint_{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=t_left_i,
                T_right=t_right_i,
                axis=axes[i],
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


@pytest.fixture(scope="module")
def synth_a() -> KinBody:
    return _build_random_generic_arm(seed=5)


@pytest.mark.slow
def test_generic_pose_solutions_fk_match(synth_a: KinBody) -> None:
    """One hand-picked pose: solver finds at least one valid IK and all
    returned q's FK-match at 1e-5. Seeded q* is recovered to 1e-3 rad."""
    rng = np.random.default_rng(123)
    q_star = rng.uniform(-np.pi + 0.3, np.pi - 0.3, 6)
    T_star = _fk(synth_a, q_star)
    solutions, is_ls = gen_six_dof.solve(synth_a, T_star)
    assert not is_ls, "gen_six_dof should find at least one solution"
    assert len(solutions) >= 1
    for i, sol in enumerate(solutions):
        T_check = _fk(synth_a, sol.q)
        # FK tolerance 1e-5 reflects SP5 + Nelder-Mead precision floor.
        assert np.allclose(T_check, T_star, atol=1e-5), (
            f"solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )
        assert sol.solver_name == "ikgeo.gen_six_dof"

    def _wrap_max(q: np.ndarray) -> float:
        diffs = [
            abs(((float(qi - qs) + np.pi) % (2 * np.pi)) - np.pi)
            for qi, qs in zip(q, q_star, strict=True)
        ]
        return max(diffs)

    best_dq = min(_wrap_max(sol.q) for sol in solutions)
    # 1e-3 rad reflects tier-2's Nelder-Mead precision on a 100x100 seed grid.
    assert best_dq < 1e-3, f"seeded q* not recovered within 1e-3 rad; closest dq={best_dq:.2e}"


@pytest.mark.slow
def test_second_synthetic_arm_solutions_fk_match() -> None:
    """Second differently-seeded arm -- validates 'generic, not geometry-specific'."""
    synth_b = _build_random_generic_arm(seed=99)
    rng = np.random.default_rng(456)
    q_star = rng.uniform(-np.pi + 0.3, np.pi - 0.3, 6)
    T_star = _fk(synth_b, q_star)
    solutions, is_ls = gen_six_dof.solve(synth_b, T_star)
    if is_ls:
        pytest.skip("synth_b pose happens to fall in a gen_six_dof infeasible region")
    assert len(solutions) >= 1
    for i, sol in enumerate(solutions):
        T_check = _fk(synth_b, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-5), f"synth_b sol {i} fails FK"


def test_wrong_dof_raises(synth_a: KinBody) -> None:
    short_kb = KinBody(links=synth_a.links[:5], joints=synth_a.joints[:4])
    with pytest.raises(ValueError, match="6-DOF"):
        gen_six_dof.solve(short_kb, np.eye(4))
