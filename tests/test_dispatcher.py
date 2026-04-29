"""End-to-end correctness for :mod:`ssik.core.dispatcher`.

Asserts that ``dispatch(kb)`` picks the right solver for every real URDF /
MJCF fixture we ship, plus a synthetic non-Pieper 6R for the EAIK-gap case.
The numeric estimate fields (``expected_ms_median``, ``flop_budget``) are
checked for plausible orders of magnitude, not exact values -- they're
order-of-magnitude estimates, not contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from ssik._kinbody import Joint, KinBody, Link, build_kinbody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.dispatcher import DispatchPlan, dispatch

FIXTURES = Path(__file__).parent / "fixtures"
sys.path.insert(0, str(FIXTURES))
from jaco2 import jaco2_specs  # noqa: E402


def _build_synth_non_pieper() -> KinBody:
    """Random-axis 6R chain with no parallel/intersecting structure -- the
    classic EAIK-gap shape that should land on ``ikgeo.general_6r``.
    """
    rng = np.random.default_rng(seed=1234)

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
        T_l = np.eye(4)
        T_l[:3, 3] = t_lefts[i]
        T_r = np.eye(4)
        if i == 5:
            T_r[:3, 3] = tool_p
        joints.append(
            Joint(
                name=f"joint_{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_l,
                T_right=T_r,
                axis=axes[i],
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


@pytest.mark.parametrize(
    ("urdf", "base", "ee", "expected_solver", "expected_tier"),
    [
        ("ur5.urdf", "base_link", "ee_link", "ikgeo.three_parallel", 0),
        (
            "puma560.urdf",
            "base_link",
            "wrist_3_link",
            "ikgeo.spherical_two_parallel",
            0,
        ),
    ],
)
def test_real_urdf_dispatch(
    urdf: str, base: str, ee: str, expected_solver: str, expected_tier: int
) -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / urdf, base, ee)
    plan = dispatch(kb)
    assert plan.solver_name == expected_solver
    assert plan.tier == expected_tier
    assert plan.expected_ms_median > 0
    assert plan.flop_budget > 0
    # Tier 0/1 should not need symbolic precompute.
    assert plan.needs_symbolic_precompute is False
    assert plan.estimated_precompute_seconds is None
    # Reason is a multi-line user-facing string.
    assert "\n" in plan.reason
    assert len(plan.reason) > 50


def test_jaco2_dispatch_to_general_6r() -> None:
    """JACO 2 (real MJCF, 60-degree non-orthogonal twists) is the canonical
    non-Pieper 6R fixture; it must land on the production tier-2 path."""
    kb = build_kinbody(jaco2_specs())
    plan = dispatch(kb)
    assert plan.solver_name == "ikgeo.general_6r"
    assert plan.tier == 2
    assert plan.needs_symbolic_precompute is True


def test_synth_non_pieper_dispatch_to_general_6r() -> None:
    """The defining EAIK-gap test: a chain with no Pieper specialisation
    must dispatch to the production tier-2 path (``general_6r``), not the
    grid-search oracle (``gen_six_dof``)."""
    kb = _build_synth_non_pieper()
    plan = dispatch(kb)
    assert plan.solver_name == "ikgeo.general_6r"
    assert plan.tier == 2
    assert plan.needs_symbolic_precompute is True
    assert plan.estimated_precompute_seconds is not None
    assert plan.estimated_precompute_seconds > 0
    # Reason mentions the EAIK-gap framing.
    assert "EAIK" in plan.reason or "Raghavan" in plan.reason


def test_dispatch_rejects_non_6dof() -> None:
    """7R chains aren't supported by this dispatcher iteration -- they
    route through ``jointlock.seven_r``. The error message should say so."""
    rng = np.random.default_rng(0)
    links = [Link(name=f"l{i}") for i in range(8)]
    joints = []
    for i in range(7):
        T_l = np.eye(4)
        T_l[:3, 3] = rng.standard_normal(3)
        ax = rng.standard_normal(3)
        ax = ax / float(np.linalg.norm(ax))
        joints.append(
            Joint(
                name=f"j{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_l,
                T_right=np.eye(4),
                axis=ax,
                joint_type="revolute",
            )
        )
    kb = KinBody(links=links, joints=joints)
    with pytest.raises(ValueError, match="7R support"):
        dispatch(kb)


def test_dispatch_plan_is_immutable() -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    plan = dispatch(kb)
    assert isinstance(plan, DispatchPlan)
    with pytest.raises((AttributeError, Exception)):  # frozen dataclass
        plan.solver_name = "ikgeo.spherical"  # type: ignore[misc]
