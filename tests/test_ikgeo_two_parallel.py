"""End-to-end validation for :mod:`ssik.solvers.ikgeo.two_parallel`.

Tier-1 univariate-search solver for 6R arms with ``axes[1] || axes[2]``
and no stronger wrist specialization. Same precision / completeness
caveats as ``two_intersecting`` (see ``project_tier1_search_completeness.md``
memory): ``search_1d``'s 200-sample grid can miss zero crossings at
SP6-feasibility boundaries, so the solver may return fewer than the
generic 8 IK solutions for random poses.

What we do test:
- every returned q reproduces T_target under FK at 1e-8,
- topology refusal (wrong DOF, non-parallel joints 1,2).

Fixtures are synthetic, following IK-Geo's own ``TwoParallelSetup``
pattern (random unit axes with ``axes[2] = axes[1]`` + random offsets).
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik._kinbody import Joint, KinBody, Link
from ssik.solvers.ikgeo import two_parallel


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


def _build_random_two_parallel_arm(seed: int) -> KinBody:
    """Follows IK-Geo's own TwoParallelSetup: 6 random unit axes with
    axes[2] = axes[1] (the 'two parallel' constraint) and 7 random offsets."""
    rng = np.random.default_rng(seed)

    def _rnorm() -> np.ndarray:
        v = rng.standard_normal(3)
        return v / float(np.linalg.norm(v))

    axes = [_rnorm() for _ in range(6)]
    axes[2] = axes[1].copy()
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
    return _build_random_two_parallel_arm(seed=7)


@pytest.fixture(scope="module")
def synth_b() -> KinBody:
    return _build_random_two_parallel_arm(seed=42)


# ---------------------------------------------------------------------------
# Returned-solution correctness: every q that the solver emits must FK-close
# on T_target. Does not assert completeness (see module docstring).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", list(range(5)))
def test_random_pose_returned_solutions_fk_match(synth_a: KinBody, seed: int) -> None:
    """For each of 5 random poses on synth_a, all returned q's FK-match."""
    rng = np.random.default_rng(seed + 1000)
    q_star = rng.uniform(-np.pi + 0.3, np.pi - 0.3, 6)
    T_star = _fk(synth_a, q_star)
    solutions, is_ls = two_parallel.solve(synth_a, T_star)
    # Tier-1 sparsity: solver may return 0 solutions on a pose even if a
    # valid IK exists (search_1d misses crossings at SP6-feasibility
    # boundaries). Both outcomes are "acceptable" for our correctness
    # tests; what we verify is that *returned* solutions are correct.
    if is_ls:
        return
    assert len(solutions) >= 1
    for i, sol in enumerate(solutions):
        q = sol.q
        T_check = _fk(synth_a, q)
        assert np.allclose(T_check, T_star, atol=1e-8), (
            f"solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


@pytest.mark.parametrize("seed", list(range(3)))
def test_second_synthetic_arm_returned_solutions_fk_match(synth_b: KinBody, seed: int) -> None:
    """Same correctness check on a differently-seeded synth arm
    (validates "generic, not geometry-specific")."""
    rng = np.random.default_rng(seed + 2000)
    q_star = rng.uniform(-np.pi + 0.3, np.pi - 0.3, 6)
    T_star = _fk(synth_b, q_star)
    solutions, is_ls = two_parallel.solve(synth_b, T_star)
    if is_ls:
        return
    assert len(solutions) >= 1
    for i, sol in enumerate(solutions):
        q = sol.q
        T_check = _fk(synth_b, q)
        assert np.allclose(T_check, T_star, atol=1e-8), f"synth_b sol {i} fails FK"


def test_solver_finds_at_least_one_solution_over_population(synth_a: KinBody) -> None:
    """Over 20 random poses, the solver should find at least one valid IK
    on *some* of them. This is a regression guard -- below the observed
    baseline indicates the solver has become universally non-functional.

    **Tier-1 sparsity baseline**: measured on this synth with 200
    search_1d samples, ~15-30% of random non-singular poses yield at
    least one valid IK. The rest fail because SP6's feasibility window
    over q1 is narrow AND branch-index tracking across search_1d
    samples is unstable -- SP6's (q6, q4) branches reorder between
    adjacent q1 values, so the index-based crossing detection in
    search_1d misses zeros. Proper geometric-continuity branch
    tracking (instead of index-based) is a substantial rework tracked
    separately (post-v0.1). Threshold below is set to catch regressions
    into zero-completeness; see ``project_tier1_search_completeness.md``.
    """
    rng = np.random.default_rng(31415)
    n_found = 0
    n_total = 20
    for _ in range(n_total):
        q_star = rng.uniform(-np.pi + 0.3, np.pi - 0.3, 6)
        T_star = _fk(synth_a, q_star)
        solutions, is_ls = two_parallel.solve(synth_a, T_star)
        if not is_ls and len(solutions) >= 1:
            n_found += 1
    # With geometric branch-matched search_1d (see _univariate.py),
    # observed is 10-12/20 (~55%) at the time of writing. Floor at
    # 7/20 (35%) guards against regressions; the remaining misses
    # are poses where SP6 is infeasible across most of [-pi, pi] and
    # branch-matching has no pairs to track.
    assert n_found >= 7, (
        f"solver found solutions in only {n_found}/{n_total} poses -- "
        f"tier-1 completeness has regressed below the minimum threshold (7/20)"
    )


# ---------------------------------------------------------------------------
# Topology refusal.
# ---------------------------------------------------------------------------


def test_wrong_dof_raises(synth_a: KinBody) -> None:
    short_kb = KinBody(links=synth_a.links[:5], joints=synth_a.joints[:4])
    with pytest.raises(ValueError, match="6-DOF"):
        two_parallel.solve(short_kb, np.eye(4))


def test_wrong_topology_raises_non_parallel_shoulder() -> None:
    """UR5 has three parallel axes (1,2,3), not just two -- should still
    pass the two_parallel check because axes[1]||axes[2] holds. Use an
    arm whose axes[1] is NOT parallel to axes[2]."""
    from dataclasses import replace
    from pathlib import Path

    from ssik._urdf import load_urdf_kinbody_normalized

    kb = load_urdf_kinbody_normalized(
        Path(__file__).parent / "fixtures" / "ur5.urdf", "base_link", "ee_link"
    )
    # Break parallelism by setting axes[2] orthogonal to axes[1].
    new_axis = np.array([1.0, 0.0, 0.0])
    new_joint = replace(kb.joints[2], axis=new_axis)
    broken_kb = KinBody(
        links=kb.links,
        joints=[new_joint if i == 2 else kb.joints[i] for i in range(6)],
    )
    with pytest.raises(ValueError, match="parallel"):
        two_parallel.solve(broken_kb, np.eye(4))
