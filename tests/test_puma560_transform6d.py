"""End-to-end integration: normalized Puma 560 → ikfast Transform6D → walker.

This test is the empirical proof-point for #33 normalization: it shows that
POE-normalizing a real URDF makes ikfast's ``TestIntersectingAxes``
pattern-matcher recognize the spherical-wrist structure and take the fast
Pieper-style decomposition path, producing a Transform6D chaintree in ~30s
on a laptop.

Puma 560 is a **canonical** Pieper-compatible arm (d₅=d₆=0, a₄=a₅=a₆=0);
it's the textbook example in every analytical-IK course since 1982 and
EAIK's own canonical URDF example. The exact DH numbers come from Peter
Corke's Robotics Toolbox.

Marked ``slow`` — generation takes ~30s wall time. Opt in via
``pytest -m slow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).parent / "fixtures"
URDF_PATH = FIXTURES / "puma560.urdf"


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        return np.eye(4)
    k = axis / norm
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return T


def _fk(kinbody: Any, q: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    for j, qi in zip(kinbody.joints, q, strict=True):
        T = T @ j.T_left @ _axis_angle(j.axis, qi) @ j.T_right
    return T


@pytest.fixture(scope="module")
def puma560_kinbody() -> Any:
    from ssik._urdf import load_urdf_kinbody_normalized

    return load_urdf_kinbody_normalized(URDF_PATH, "base_link", "wrist_3_link")


@pytest.fixture(scope="module")
def puma560_chaintree(puma560_kinbody: Any) -> Any:
    """Generate the Transform6D chaintree once (~30s) and share it across
    tests in this module. Caching via pickle would be nice, but 30s is
    tolerable for a slow-marked test."""
    from ssik._vendor.ikfast import IKFastSolver

    solver = IKFastSolver(kinbody=puma560_kinbody)
    # maxcasedepth=1 matches andyzeng/ikfastpy's empirical choice for UR5;
    # for Puma it doesn't matter since TestIntersectingAxes succeeds before
    # the case-depth-bounded paths are reached.
    solver.maxcasedepth = 1
    return solver.generateIkSolver(
        baselink="base_link",
        eelink="wrist_3_link",
        freeindices=[],
        solvefn=IKFastSolver.solveFullIK_6D,
    )


def test_normalized_puma560_parallel_axes(puma560_kinbody: Any) -> None:
    """Sanity-check the normalization before the slow generation kicks in:
    Puma 560's shoulder-and-elbow cluster (joints 1, 2, 4) should all be
    aligned along the base-frame ``-y`` axis."""
    for idx in (1, 2, 4):
        ax = puma560_kinbody.joints[idx].axis
        assert np.allclose(ax, [0.0, -1.0, 0.0], atol=1e-10), (
            f"joint {idx} axis = {ax.tolist()}, expected (0, -1, 0)"
        )
    # Joints 0, 3, 5 should be +z.
    for idx in (0, 3, 5):
        ax = puma560_kinbody.joints[idx].axis
        assert np.allclose(ax, [0.0, 0.0, 1.0], atol=1e-10), (
            f"joint {idx} axis = {ax.tolist()}, expected (0, 0, 1)"
        )


def test_transform6d_generation_succeeds(puma560_chaintree: Any) -> None:
    """The main claim: ikfast Transform6D successfully generates for the
    normalized Puma 560 via the ``TestIntersectingAxes`` + Pieper-style
    decomposition path. Prior to #33's normalization this test would be
    impossible to write — see #33 and #35 for the UR5-on-DH-style analog
    that gets stuck before reaching this point."""
    assert puma560_chaintree is not None
    cls = type(puma560_chaintree).__name__
    assert cls == "SolverIKChainTransform6D", f"unexpected chaintree type {cls}"


def test_chaintree_carries_expected_symbolic_state(puma560_chaintree: Any) -> None:
    """The chaintree's fields we need for evaluation are populated."""
    # solvejointvars lists the 6 joints we're solving for.
    assert len(puma560_chaintree.solvejointvars) == 6
    # Tee is a symbolic 4x4 — the EE-pose remapping function.
    assert puma560_chaintree.Tee is not None
    # jointtree is the sequential list of solver nodes (SolverCheckZeros /
    # SolverSolution / etc.) that the walker will execute.
    assert len(puma560_chaintree.jointtree) >= 1


def test_walker_produces_candidates_at_a_known_pose(
    puma560_kinbody: Any, puma560_chaintree: Any
) -> None:
    """Pick a non-singular ``q*``, compute T* = FK(q*), feed T* through the
    Transform6D walker, and verify at least one candidate solution actually
    satisfies FK(candidate) ≈ T*. This is the full roundtrip — identical in
    spirit to #30/#31 but on Transform6D instead of Translation3D, and on a
    second robot to prove the walker isn't UR5-specific."""
    from fk_ik_eval import eval_chaintree_6d

    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2])
    T_star = _fk(puma560_kinbody, q_star)

    candidates = eval_chaintree_6d(puma560_chaintree, q_free={}, target_pose=T_star)
    assert len(candidates) > 0, "walker produced no candidates at a generic q*"

    matches = []
    for cand in candidates:
        q = np.array([cand[f"j{i}"] for i in range(6)])
        if np.allclose(_fk(puma560_kinbody, q), T_star, atol=1e-6):
            matches.append(q)

    assert len(matches) > 0, (
        f"no walker candidate satisfies T_star within 1e-6. "
        f"raw candidates produced = {len(candidates)}"
    )
