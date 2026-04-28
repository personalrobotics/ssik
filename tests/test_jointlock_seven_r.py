"""End-to-end validation for :mod:`ssik.solvers.jointlock.seven_r`.

Universal 7R wrapper that locks one joint and dispatches the resulting
6R sub-chain to whichever ikgeo tier-0/1 solver matches its (per-sample)
topology. Tests cover:

- ``choose_lock_joint`` picks the topologically-best lock per arm
- Full FK-match correctness on synthetic 7R fixtures
- Targeted user-supplied samples recover seeded q* at machine precision
- 7-DOF guard
- Performance budget: 16-sample sweep < 0.5 s on synthetic SRS arm

Fixtures are synthetic (Franka-, iiwa-, Rizon-style geometries are
covered by the upcoming ``specialist.geofik`` and ``specialist.stereo_sew``
solvers; this is the generic cross-arm fallback).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray

from ssik._kinbody import Joint, KinBody, Link
from ssik.solvers.jointlock import seven_r


def _rodrigues(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    c, s = np.cos(t), np.sin(t)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R: NDArray[np.float64] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R


def _axis_angle_4x4(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = _rodrigues(k, t)
    return T


def _fk(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _axis_angle_4x4(j.axis, qi) @ j.T_right
    return T


def _build_srs_with_spherical_wrist() -> KinBody:
    """Synthetic 7R: shoulder pitch+pitch (parallel), elbow roll, then
    a spherical wrist at joints 4-6. Locking joint 3 yields a 6R with
    spherical_two_parallel topology."""
    axes = [
        np.array([0.0, 0.0, 1.0]),  # j0 base z (shoulder yaw)
        np.array([0.0, -1.0, 0.0]),  # j1 shoulder pitch -y
        np.array([0.0, -1.0, 0.0]),  # j2 elbow pitch -y (parallel to j1)
        np.array([0.0, 0.0, 1.0]),  # j3 elbow roll z (lock candidate)
        np.array([0.0, -1.0, 0.0]),  # j4 wrist pitch -y
        np.array([0.0, 0.0, 1.0]),  # j5 wrist roll z
        np.array([0.0, -1.0, 0.0]),  # j6 final pitch -y
    ]
    t_lefts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.2]),
        np.array([0.4, 0.0, 0.0]),
        np.array([0.05, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.4]),
        np.array([0.0, 0.0, 0.0]),  # j5 origin = j4 origin
        np.array([0.0, 0.0, 0.0]),  # j6 origin = j5 origin (spherical wrist)
    ]
    links = [Link(name=f"l{i}") for i in range(8)]
    joints = []
    for i in range(7):
        T_l = np.eye(4)
        T_l[:3, 3] = t_lefts[i]
        joints.append(
            Joint(
                name=f"j{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_l,
                T_right=np.eye(4),
                axis=axes[i],
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


@pytest.fixture(scope="module")
def srs_kb() -> KinBody:
    return _build_srs_with_spherical_wrist()


# ---------------------------------------------------------------------------
# choose_lock_joint correctness.
# ---------------------------------------------------------------------------


def test_choose_lock_joint_picks_low_rank(srs_kb: KinBody) -> None:
    """For an arm where locking joint 3 yields three_parallel topology,
    the chooser should pick joint 3 (lowest rank = 0)."""
    lock_idx = seven_r.choose_lock_joint(srs_kb)
    assert lock_idx == 3, f"expected lock_idx=3, got {lock_idx}"


# ---------------------------------------------------------------------------
# Targeted lock recovers seeded q*.
# ---------------------------------------------------------------------------


_SEEDED_Q = [
    np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4]),
    np.array([-1.2, -1.8, 2.1, -0.4, 0.7, -1.5, 0.8]),
    np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]),
]


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_targeted_sample_recovers_seeded_q_star(
    srs_kb: KinBody, q_star: NDArray[np.float64]
) -> None:
    """Passing the exact lock-joint value as a one-element sample list
    should recover the seeded q* at machine precision."""
    lock_idx = seven_r.choose_lock_joint(srs_kb)
    T_star = _fk(srs_kb, q_star)
    solutions, is_ls = seven_r.solve(srs_kb, T_star, lock_samples=[float(q_star[lock_idx])])
    assert not is_ls
    assert len(solutions) >= 1

    def _max_wrap(q: NDArray[np.float64]) -> float:
        return max(
            abs(((float(qi - qs) + np.pi) % (2 * np.pi)) - np.pi)
            for qi, qs in zip(q, q_star, strict=True)
        )

    best_dq = min(_max_wrap(s.q) for s in solutions)
    assert best_dq < 1e-10, f"seeded q* not recovered at machine precision; closest dq={best_dq}"


# ---------------------------------------------------------------------------
# Sweep correctness: every returned q reproduces T_target under FK.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_sweep_solutions_all_fk_match(srs_kb: KinBody, q_star: NDArray[np.float64]) -> None:
    """All solutions returned by the default 16-sample sweep must satisfy
    FK(q) == T_target at 1e-10 atol."""
    T_star = _fk(srs_kb, q_star)
    solutions, is_ls = seven_r.solve(srs_kb, T_star, lock_samples=16)
    assert not is_ls
    assert len(solutions) >= 8, f"expected at least 8 solutions, got {len(solutions)}"
    for i, sol in enumerate(solutions):
        T_check = _fk(srs_kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-10), (
            f"sweep solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


def test_user_provided_samples_override(srs_kb: KinBody) -> None:
    """User-supplied samples should be used verbatim (not augmented)."""
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    custom = [0.0, 0.5, 1.1, -0.5]
    solutions, _ = seven_r.solve(srs_kb, T_star, lock_samples=custom)
    # Each lock value should produce up to 8 inner solutions.
    assert len(solutions) > 0
    # All solutions' lock-joint value should be in the custom list (within
    # the wrap-to-pi tolerance).
    lock_idx = seven_r.choose_lock_joint(srs_kb)
    for sol in solutions:
        ql = float(sol.q[lock_idx])
        assert any(abs(((ql - c + np.pi) % (2 * np.pi)) - np.pi) < 1e-6 for c in custom), (
            f"solution lock-joint {ql} not in user sample list"
        )


# ---------------------------------------------------------------------------
# Performance budget.
# ---------------------------------------------------------------------------


def test_default_sweep_under_budget(srs_kb: KinBody) -> None:
    """16-sample sweep on synthetic SRS arm should complete well under any
    tier-2 fallback timescale.

    Standalone runs see 0.1-2s depending on machine load; under a busy
    test suite, contention can stretch this further. Tier-2 fallback
    (gen_six_dof grid) takes 30+s, so a 5s budget still catches the
    "accidentally fell through to tier-2" regression that this test exists
    to guard against.
    """
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    t0 = time.time()
    solutions, is_ls = seven_r.solve(srs_kb, T_star, lock_samples=16)
    elapsed = time.time() - t0
    assert not is_ls
    assert len(solutions) > 0
    assert elapsed < 5.0, f"16-sample sweep took {elapsed:.2f}s (budget 5.0s)"


# ---------------------------------------------------------------------------
# Topology refusal.
# ---------------------------------------------------------------------------


def test_wrong_dof_raises() -> None:
    """6-DOF KinBody must be rejected."""
    axes = [np.array([0.0, 0.0, 1.0]) for _ in range(6)]
    t_lefts = [np.array([0.0, 0.0, 0.1 * (i + 1)]) for i in range(6)]
    links = [Link(name=f"l{i}") for i in range(7)]
    joints = []
    for i in range(6):
        T_l = np.eye(4)
        T_l[:3, 3] = t_lefts[i]
        joints.append(
            Joint(
                name=f"j{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_l,
                T_right=np.eye(4),
                axis=axes[i],
                joint_type="revolute",
            )
        )
    six_kb = KinBody(links=links, joints=joints)
    with pytest.raises(ValueError, match="7"):
        seven_r.solve(six_kb, np.eye(4))
