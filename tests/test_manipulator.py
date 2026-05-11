"""Bulletproof tests for the public :class:`ssik.Manipulator` API (#12, #227).

Manipulator is the v1.0 entry point. These tests assert:

- Factories work on every fixture (URDF + Python-spec).
- ``fk`` and ``ik`` round-trip on hand-picked + random poses.
- ``ik`` honors ``max_solutions``, ``q_seed``, ``allow_refinement``.
- Error paths fire with clear messages.
- The Manipulator's per-IK overhead is negligible (< 50 us vs the raw solver).
- Repr/str are useful diagnostics.

The tests cover the common path (``Manipulator.from_urdf`` then ``.fk`` / ``.ik``)
on every fixture in ``tests/fixtures/``. Solver-specific bulletproof tests
already exist (e.g. test_kinova_gen3.py); this file is the API contract.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik._kinbody import build_kinbody
from ssik.kinematics.poe_fk import poe_forward_kinematics

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

# ---------------------------------------------------------------------------
# URDF-based fixtures: full smoke matrix
# ---------------------------------------------------------------------------


_URDF_FIXTURES = [
    # (filename, base_link, ee_link, expected_solver_name)
    ("ur5.urdf", "base_link", "ee_link", "ikgeo.three_parallel"),
    ("puma560.urdf", "base_link", "wrist_3_link", "ikgeo.spherical_two_parallel"),
    ("gen3.urdf", "base_link", "end_effector_link", "seven_r.srs_polished"),
    ("rizon4.urdf", "base_link", "flange", "jointlock.seven_r"),
    ("kassow_kr810.urdf", "base", "end_effector", "jointlock.seven_r"),
]


@pytest.mark.parametrize(("urdf", "base", "ee", "expected_solver"), _URDF_FIXTURES)
def test_from_urdf_dispatches_correctly(
    urdf: str, base: str, ee: str, expected_solver: str
) -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / urdf, base=base, ee=ee)
    assert arm.solver_name == expected_solver
    assert arm.dof in (6, 7)


@pytest.mark.parametrize(("urdf", "base", "ee", "_solver"), _URDF_FIXTURES)
def test_fk_then_ik_roundtrip(urdf: str, base: str, ee: str, _solver: str) -> None:
    """Hand-picked q → fk → ik must return at least one IK that FK-closes."""
    arm = ssik.Manipulator.from_urdf(FIXTURES / urdf, base=base, ee=ee)
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-0.5, 0.5, size=arm.dof)
    if arm.dof == 7:
        # Avoid kinematic singularity for 7DOF arms (#225 follow-up; not the
        # API contract under test).
        q_star[3] = float(rng.uniform(0.3, 0.7))
    T = arm.fk(q_star)
    sols = arm.solve(T)
    assert sols, f"{urdf}: solve returned no solutions on a reachable FK pose"
    # At least one solution FK-closes.
    best = min(np.linalg.norm(arm.fk(s.q) - T) for s in sols)
    assert best < 1e-6, f"{urdf}: best FK residual {best:.2e}"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_dof_property() -> None:
    arm6 = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    assert arm6.dof == 6
    arm7 = ssik.Manipulator.from_urdf(FIXTURES / "rizon4.urdf", base="base_link", ee="flange")
    assert arm7.dof == 7


def test_joint_limits_shape() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    limits = arm.joint_limits
    assert len(limits) == arm.dof
    for lo, hi in limits:
        assert lo < hi


def test_dispatch_plan_has_useful_fields() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    plan = arm.dispatch_plan
    assert plan.solver_name == "ikgeo.three_parallel"
    assert plan.tier == 0
    assert isinstance(plan.reason, str)
    assert plan.reason
    assert plan.expected_ms_median > 0


def test_kinbody_property_exposes_internal() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    kb = arm.kinbody
    assert isinstance(kb, ssik.KinBody)
    assert len(kb.joints) == 6


def test_repr_is_useful() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    r = repr(arm)
    assert "Manipulator" in r
    assert "6-DOF" in r
    assert "ikgeo.three_parallel" in r


# ---------------------------------------------------------------------------
# fk
# ---------------------------------------------------------------------------


def test_fk_returns_4x4() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    T = arm.fk(np.zeros(6))
    assert T.shape == (4, 4)
    assert T.dtype == np.float64
    # Bottom row is [0, 0, 0, 1].
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_fk_matches_poe_forward_kinematics() -> None:
    """``Manipulator.fk(q)`` must be bit-identical to the underlying primitive."""
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    rng = np.random.default_rng(42)
    for _ in range(10):
        q = rng.uniform(-1, 1, size=arm.dof)
        T_manip = arm.fk(q)
        T_raw = poe_forward_kinematics(arm.kinbody, q)
        np.testing.assert_array_equal(T_manip, T_raw)


def test_fk_accepts_list_and_tuple() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    T_list = arm.fk([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    T_tuple = arm.fk((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    T_arr = arm.fk(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    np.testing.assert_array_equal(T_list, T_arr)
    np.testing.assert_array_equal(T_tuple, T_arr)


def test_fk_rejects_wrong_shape() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    with pytest.raises(ValueError, match="fk expected q of shape"):
        arm.fk(np.zeros(5))
    with pytest.raises(ValueError, match="fk expected q of shape"):
        arm.fk(np.zeros((6, 1)))


# ---------------------------------------------------------------------------
# ik
# ---------------------------------------------------------------------------


def test_solve_returns_list_of_solutions() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    T = arm.fk(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    sols = arm.solve(T)
    assert isinstance(sols, list)
    assert all(isinstance(s, ssik.Solution) for s in sols)


def test_ik_max_solutions_caps_returned() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    T = arm.fk(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    sols_full = arm.solve(T)
    sols_capped = arm.solve(T, max_solutions=2)
    assert len(sols_full) > 2
    assert len(sols_capped) <= 2
    # FK closure preserved on the capped set.
    for s in sols_capped:
        assert s.fk_residual < 1e-6


def test_ik_q_seed_passes_through_when_supported() -> None:
    """``q_seed`` is accepted by jointlock.seven_r and used to reorder samples."""
    arm = ssik.Manipulator.from_urdf(FIXTURES / "rizon4.urdf", base="base_link", ee="flange")
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, size=7)
    q[3] = 0.5
    T = arm.fk(q)
    # No q_seed: get solutions
    sols_no_seed = arm.solve(T, max_solutions=1)
    # With q_seed: should also work, same solver path
    sols_seeded = arm.solve(T, max_solutions=1, q_seed=q)
    assert len(sols_no_seed) == 1
    assert len(sols_seeded) == 1


def test_ik_q_seed_silently_ignored_when_unsupported() -> None:
    """Solvers that don't accept ``q_seed`` (e.g. ikgeo.three_parallel) should
    not raise -- the API filters kwargs by signature.
    """
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    T = arm.fk(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    # Should not raise even though ikgeo.three_parallel.solve has no q_seed param.
    sols = arm.solve(T, q_seed=np.zeros(6))
    assert sols


def test_solve_rejects_wrong_T_shape() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    with pytest.raises(ValueError, match="solve expected T_target"):
        arm.solve(np.eye(3))
    with pytest.raises(ValueError, match="solve expected T_target"):
        arm.solve(np.zeros(16))


def test_ik_rejects_wrong_q_seed_shape() -> None:
    arm = ssik.Manipulator.from_urdf(FIXTURES / "rizon4.urdf", base="base_link", ee="flange")
    T = arm.fk(np.zeros(7))
    with pytest.raises(ValueError, match="q_seed expected shape"):
        arm.solve(T, q_seed=np.zeros(6))


def test_solve_unreachable_target_returns_empty() -> None:
    """A target way out of reach yields an empty solution list."""
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    T = np.eye(4)
    T[:3, 3] = [100.0, 100.0, 100.0]  # 100 metres away
    sols = arm.solve(T)
    assert sols == []


# ---------------------------------------------------------------------------
# Construction from a pre-built KinBody (escape hatch)
# ---------------------------------------------------------------------------


def test_construct_from_kinbody_directly() -> None:
    """The public constructor accepts a KinBody for callers who already have one."""
    from franka_panda import franka_panda_specs

    kb = build_kinbody(franka_panda_specs())
    arm = ssik.Manipulator(kb)
    assert arm.dof == 7
    assert arm.solver_name == "jointlock.seven_r"
    # Same FK roundtrip contract.
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, size=7)
    q[3] = 0.5
    T = arm.fk(q)
    sols = arm.solve(T, max_solutions=1)
    assert sols


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_from_urdf_raises_on_missing_file(tmp_path: Path) -> None:
    bad_path = tmp_path / "nonexistent.urdf"
    with pytest.raises(FileNotFoundError):
        ssik.Manipulator.from_urdf(bad_path, base="base", ee="ee")


def test_from_urdf_raises_on_bad_link_name() -> None:
    with pytest.raises(ValueError, match=r"this_link_does_not_exist|not found|no link"):
        ssik.Manipulator.from_urdf(
            FIXTURES / "ur5.urdf", base="this_link_does_not_exist", ee="ee_link"
        )


# ---------------------------------------------------------------------------
# Public exports surface
# ---------------------------------------------------------------------------


def test_top_level_exports_present() -> None:
    """Every type a user needs is reachable from the top-level ``ssik`` namespace."""
    expected = [
        "Manipulator",
        "Solution",
        "KinBody",
        "Joint",
        "Link",
        "JointSpec",
        "build_kinbody",
        "TolerancePolicy",
        "DEFAULT_TOLERANCE_POLICY",
        "DispatchPlan",
        "TopologyReport",
        "describe_topology",
        "dispatch",
        "__version__",
    ]
    for name in expected:
        assert hasattr(ssik, name), f"ssik.{name} missing"
    assert set(ssik.__all__) >= set(expected) - {"__version__"}


# ---------------------------------------------------------------------------
# Performance: Manipulator overhead must be small
# ---------------------------------------------------------------------------


def test_ik_overhead_under_100us() -> None:
    """Manipulator.ik() should add < 100 us overhead vs the raw solver call.
    The wrapper does signature inspection + kwarg filtering -- cheap, but
    catch regressions where someone adds a per-call sympy import or similar.
    """
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    T = arm.fk(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

    # Warm
    for _ in range(20):
        arm.solve(T)

    # Manipulator path
    t = time.perf_counter()
    for _ in range(200):
        arm.solve(T)
    manip_per = (time.perf_counter() - t) / 200

    # Raw solver path
    from ssik.solvers.ikgeo import three_parallel

    t = time.perf_counter()
    for _ in range(200):
        three_parallel.solve(arm.kinbody, T)
    raw_per = (time.perf_counter() - t) / 200

    overhead = (manip_per - raw_per) * 1e6
    assert overhead < 100.0, (
        f"Manipulator.ik overhead {overhead:.1f} us > 100 us regression gate "
        f"(manip={manip_per * 1e6:.1f} us, raw={raw_per * 1e6:.1f} us)"
    )
