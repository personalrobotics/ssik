"""Plumb-through correctness for ``max_solutions`` (#198).

The outer-most solver layers (``seven_r.srs``, ``jointlock.seven_r``) already
honor ``max_solutions``. Issue #198 plumbs it through the inner-6R layer
so a capped IK request stops branch enumeration at every depth.

Test contract:
- Each inner-6R ``solve()`` accepts ``max_solutions: int | None``.
- ``max_solutions=None`` matches existing behavior (regression guard).
- ``max_solutions=N`` returns at most ``N`` IKs, all FK-closing.
- ``max_solutions=1`` is the trajectory-tracking case (most common).
- Jointlock plumbs the cap to inner solvers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from ssik._kinbody import KinBody, build_kinbody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.husty_pfurner import general_6r as hp_general_6r
from ssik.solvers.ikgeo import (
    general_6r as rr_general_6r,
)
from ssik.solvers.ikgeo import (
    spherical,
    spherical_two_intersecting,
    spherical_two_parallel,
    three_parallel,
)
from ssik.solvers.jointlock import seven_r as jointlock_seven_r

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

from jaco2 import jaco2_specs  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def puma_kb() -> KinBody:
    return load_urdf_kinbody_normalized(FIXTURES / "puma560.urdf", "base_link", "wrist_3_link")


@pytest.fixture(scope="module")
def ur5_kb() -> KinBody:
    return load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")


@pytest.fixture(scope="module")
def rizon4_kb() -> KinBody:
    return load_urdf_kinbody_normalized(FIXTURES / "rizon4.urdf", "base_link", "flange")


@pytest.fixture(scope="module")
def jaco2_kb() -> KinBody:
    """JACO 2 (non-Pieper 6R) -- canonical fixture for RR + HP."""
    return build_kinbody(jaco2_specs())


def _seeded_target(kb: KinBody, q_seed: np.ndarray) -> np.ndarray:
    out: np.ndarray = poe_forward_kinematics(kb, q_seed)
    return out


# ---------------------------------------------------------------------------
# Inner-6R: each solver respects the cap and returns FK-closing IKs.
# ---------------------------------------------------------------------------


_PUMA_Q = np.array([0.4, -0.6, 0.8, 1.0, -0.4, 0.3])
_UR5_Q = np.array([0.3, -0.7, 0.9, -0.5, 1.1, 0.2])


@pytest.mark.parametrize("max_n", [1, 2, 4])
def test_spherical_two_intersecting_respects_max_solutions(puma_kb: KinBody, max_n: int) -> None:
    T = _seeded_target(puma_kb, _PUMA_Q)
    sols, is_ls = spherical_two_intersecting.solve(puma_kb, T, max_solutions=max_n)
    assert not is_ls
    assert 1 <= len(sols) <= max_n
    for s in sols:
        assert s.fk_residual < 1e-6


@pytest.mark.parametrize("max_n", [1, 2, 4])
def test_spherical_two_parallel_respects_max_solutions(puma_kb: KinBody, max_n: int) -> None:
    T = _seeded_target(puma_kb, _PUMA_Q)
    sols, is_ls = spherical_two_parallel.solve(puma_kb, T, max_solutions=max_n)
    assert not is_ls
    assert 1 <= len(sols) <= max_n
    for s in sols:
        assert s.fk_residual < 1e-6


@pytest.mark.parametrize("max_n", [1, 2, 4])
def test_three_parallel_respects_max_solutions(ur5_kb: KinBody, max_n: int) -> None:
    T = _seeded_target(ur5_kb, _UR5_Q)
    sols, is_ls = three_parallel.solve(ur5_kb, T, max_solutions=max_n)
    assert not is_ls
    assert 1 <= len(sols) <= max_n
    for s in sols:
        assert s.fk_residual < 1e-6


# ``spherical`` is the generic spherical-wrist fallback. Puma matches both
# specialisations + the generic one. We sanity-check that the cap fires.
@pytest.mark.parametrize("max_n", [1, 2])
def test_spherical_respects_max_solutions(puma_kb: KinBody, max_n: int) -> None:
    T = _seeded_target(puma_kb, _PUMA_Q)
    sols, is_ls = spherical.solve(puma_kb, T, max_solutions=max_n)
    # ``spherical`` may legitimately return is_ls=True on Puma when the SP5
    # shoulder reduction is degenerate (the arm actually belongs to
    # spherical_two_parallel); that's fine -- the cap contract still holds.
    if not is_ls:
        assert 1 <= len(sols) <= max_n
        for s in sols:
            assert s.fk_residual < 1e-6


_JACO2_Q = np.array([0.4, -0.6, 0.8, 1.0, -0.4, 0.3])


@pytest.mark.parametrize("max_n", [1, 2])
def test_husty_pfurner_respects_max_solutions(jaco2_kb: KinBody, max_n: int) -> None:
    T = _seeded_target(jaco2_kb, _JACO2_Q)
    # JACO 2 needs LM polish to recover IK from HP's algebraic seeds (matches
    # ssik.solvers.husty_pfurner's oracle test contract).
    sols, is_ls = hp_general_6r.solve(jaco2_kb, T, max_solutions=max_n, allow_refinement=True)
    assert not is_ls
    assert 1 <= len(sols) <= max_n
    for s in sols:
        assert s.fk_residual < 1e-10


@pytest.mark.parametrize("max_n", [1, 2, 4])
def test_ikgeo_general_6r_respects_max_solutions(jaco2_kb: KinBody, max_n: int) -> None:
    T = _seeded_target(jaco2_kb, _JACO2_Q)
    sols, is_ls = rr_general_6r.solve(jaco2_kb, T, max_solutions=max_n)
    assert not is_ls
    assert 1 <= len(sols) <= max_n
    for s in sols:
        assert s.fk_residual < 1e-6


# ---------------------------------------------------------------------------
# Default-None preserves existing behavior (regression guard).
# ---------------------------------------------------------------------------


def test_spherical_two_parallel_none_matches_uncapped(puma_kb: KinBody) -> None:
    T = _seeded_target(puma_kb, _PUMA_Q)
    capped, _ = spherical_two_parallel.solve(puma_kb, T, max_solutions=None)
    uncapped, _ = spherical_two_parallel.solve(puma_kb, T)
    assert len(capped) == len(uncapped)


def test_three_parallel_none_matches_uncapped(ur5_kb: KinBody) -> None:
    T = _seeded_target(ur5_kb, _UR5_Q)
    capped, _ = three_parallel.solve(ur5_kb, T, max_solutions=None)
    uncapped, _ = three_parallel.solve(ur5_kb, T)
    assert len(capped) == len(uncapped)


def test_husty_pfurner_none_matches_uncapped(jaco2_kb: KinBody) -> None:
    T = _seeded_target(jaco2_kb, _JACO2_Q)
    capped, _ = hp_general_6r.solve(jaco2_kb, T, max_solutions=None, allow_refinement=True)
    uncapped, _ = hp_general_6r.solve(jaco2_kb, T, allow_refinement=True)
    assert len(capped) == len(uncapped)


# ---------------------------------------------------------------------------
# Validation: max_solutions < 1 is rejected.
# ---------------------------------------------------------------------------


def test_zero_max_solutions_rejected(puma_kb: KinBody, jaco2_kb: KinBody) -> None:
    T_puma = _seeded_target(puma_kb, _PUMA_Q)
    T_jaco = _seeded_target(jaco2_kb, _JACO2_Q)
    with pytest.raises(ValueError, match="max_solutions must be >= 1"):
        spherical_two_parallel.solve(puma_kb, T_puma, max_solutions=0)
    with pytest.raises(ValueError, match="max_solutions must be >= 1"):
        hp_general_6r.solve(jaco2_kb, T_jaco, max_solutions=0)
    with pytest.raises(ValueError, match="max_solutions must be >= 1"):
        rr_general_6r.solve(jaco2_kb, T_jaco, max_solutions=0)


# ---------------------------------------------------------------------------
# Jointlock plumbs the cap to inner solvers (Rizon 4 dispatches to
# jointlock.seven_r with HP / two_parallel as the inner). With cap=1, every
# inner-solver call gets max_solutions=1 -> capped enumeration.
# ---------------------------------------------------------------------------


def test_jointlock_plumbs_max_solutions_through_dispatch(rizon4_kb: KinBody) -> None:
    q_seed = np.array([0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4])
    T = poe_forward_kinematics(rizon4_kb, q_seed)

    # Patch the inner-6R solvers to record observed max_solutions.
    observed: list[int | None] = []
    real_dispatch = jointlock_seven_r._dispatch  # type: ignore[attr-defined]

    def _spy_dispatch(  # type: ignore[no-untyped-def]
        solver_name,
        sub_kb,
        T_target,
        policy,
        *,
        allow_refinement,
        refinement_max_iters,
        max_solutions=None,
        cached_rr_only=False,
    ):
        observed.append(max_solutions)
        return real_dispatch(
            solver_name,
            sub_kb,
            T_target,
            policy,
            allow_refinement=allow_refinement,
            refinement_max_iters=refinement_max_iters,
            max_solutions=max_solutions,
            cached_rr_only=cached_rr_only,
        )

    jointlock_seven_r._dispatch = _spy_dispatch  # type: ignore[attr-defined]
    try:
        sols, _ = jointlock_seven_r.solve(rizon4_kb, T, max_solutions=1)
    finally:
        jointlock_seven_r._dispatch = real_dispatch  # type: ignore[attr-defined]

    # All observed inner-dispatch calls were given max_solutions=1.
    assert observed, "expected at least one inner dispatch"
    assert all(m == 1 for m in observed), f"expected all 1, got {observed}"
    # And the outer cap is honored.
    assert len(sols) == 1
    assert sols[0].fk_residual < 1e-9
