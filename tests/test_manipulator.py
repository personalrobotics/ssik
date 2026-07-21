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
from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik._kinbody import build_kinbody
from ssik.kinematics.poe_fk import poe_forward_kinematics

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))

from _perf import best_call_ms  # noqa: E402

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
    for lim in limits:
        if lim is None:  # continuous joint -- unconstrained
            continue
        lo, hi = lim
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
    from ssik.internals import KinBody

    assert isinstance(kb, KinBody)
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
    # Bypass limits filtering: this test exercises q_seed plumbing, not
    # reachability. respect_limits=True can drop branches on Rizon 4
    # which masks the q_seed kwarg behavior under test.
    sols_no_seed = arm.solve(T, max_solutions=1, respect_limits=False)
    sols_seeded = arm.solve(T, max_solutions=1, q_seed=q, respect_limits=False)
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
    # Franka: exact spherical-shoulder specialist (#373).
    assert arm.solver_name == "seven_r.spherical_shoulder"
    # Same FK roundtrip contract.
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, size=7)
    q[3] = 0.5
    T = arm.fk(q)
    # Bypass URDF limit filter: this test exercises construction + dispatch,
    # not reachability under Franka's tight joint limits.
    sols = arm.solve(T, max_solutions=1, respect_limits=False)
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
    """Top-level ``ssik`` namespace is the user-facing v1.0 surface."""
    expected = [
        "Manipulator",
        "Solution",
        "TolerancePolicy",
        "DEFAULT_TOLERANCE_POLICY",
        "__version__",
    ]
    for name in expected:
        assert hasattr(ssik, name), f"ssik.{name} missing"
    assert set(ssik.__all__) >= set(expected) - {"__version__"}


def test_contributor_surface_via_ssik_internals() -> None:
    """Contributor / debugging surface lives under ``ssik.internals``."""
    from ssik import internals

    expected = [
        "KinBody",
        "Joint",
        "Link",
        "JointSpec",
        "build_kinbody",
        "DispatchPlan",
        "TopologyReport",
        "describe_topology",
        "dispatch",
    ]
    for name in expected:
        assert hasattr(internals, name), f"ssik.internals.{name} missing"


# ---------------------------------------------------------------------------
# Performance: Manipulator overhead must be small
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_ik_overhead_under_300us() -> None:
    """Manipulator.solve() should add < 300 us overhead vs the raw solver call.
    Overhead comes from signature inspection + the always-on postprocess
    pass (wrap_to_limits + respect_limits when respect_limits=True; nearest_to_seed
    when q_seed given; truncate when max_solutions given). Catches regressions
    where someone adds a per-call sympy import or similar heavy work.
    """
    arm = ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")
    T = arm.fk(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    from ssik.solvers.ikgeo import three_parallel

    # Overhead = wrapper-path best minus raw-solver best. Best-of-N on each side
    # (see tests._perf) is the noise floor, so their difference isolates the
    # Manipulator wrapper's true per-call cost instead of the differenced
    # scheduler noise of two means (which can even go negative under load).
    manip_ms = best_call_ms(lambda: arm.solve(T), warmup=20, runs=200)
    raw_ms = best_call_ms(lambda: three_parallel.solve(arm.kinbody, T), warmup=20, runs=200)

    overhead = (manip_ms - raw_ms) * 1e3  # ms -> us
    assert overhead < 300.0, (
        f"Manipulator.solve overhead {overhead:.1f} us > 300 us regression gate "
        f"(manip={manip_ms * 1e3:.1f} us, raw={raw_ms * 1e3:.1f} us)"
    )


# ---------------------------------------------------------------------------
# #328: from_urdf cold-path coverage parity (rescue) + discoverability (warn).
# ---------------------------------------------------------------------------


def test_from_urdf_jointlock_solves_ridge_via_rescue() -> None:
    """A rank-deficient ridge pose that the analytical path misses is recovered
    by the T-perturbation rescue, so ``from_urdf`` matches the baked artifact's
    coverage instead of silently returning [] (#328)."""
    arm = ssik.Manipulator.from_urdf(FIXTURES / "rizon4.urdf", base="base_link", ee="flange")
    q = np.array([0.0, 1.0, -2.0, 0.375, -2.0, 1.0, 0.0])  # rizon4 #304 ridge
    T = arm.fk(q)
    rescued = arm.solve(T, respect_limits=False)
    assert rescued, "ridge pose should be recovered via rescue"
    for s in rescued:
        T_fk = poe_forward_kinematics(arm.kinbody, np.asarray(s.q))
        assert np.allclose(T_fk, T, atol=1e-6), (
            f"rescued solution FK off by {np.abs(T_fk - T).max():.1e}"
        )
    # Rescue never reduces the analytical result.
    analytic = arm.solve(T, respect_limits=False, allow_rescue=False)
    assert len(rescued) >= len(analytic)


def test_from_urdf_jointlock_warns_once_about_cold_coverage() -> None:
    """The cold jointlock-7R path warns once that coverage/speed differ from the
    baked artifact, so the difference is discoverable rather than silent (#328)."""
    import warnings

    arm = ssik.Manipulator.from_urdf(FIXTURES / "rizon4.urdf", base="base_link", ee="flange")
    T = arm.fk(np.zeros(7))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        arm.solve(T)
        arm.solve(T)  # second call must NOT repeat the warning
    cold = [
        w for w in caught if issubclass(w.category, UserWarning) and "ssik build" in str(w.message)
    ]
    assert len(cold) == 1, f"expected exactly one cold-coverage warning, got {len(cold)}"
