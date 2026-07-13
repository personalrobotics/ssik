"""Exact feasible-swivel joint-limit resolution for SRS-class 7R (#359).

The blind elbow-swivel sweep in ``seven_r.srs`` can sample no in-limits
candidate for a reachable in-limits pose (the in-limits swivel arc is narrower
than the sampling). :func:`resolve_in_limits` computes the feasible-swivel arcs
in closed form and returns the in-limits solution(s) exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.seven_r._swivel_limits import resolve_in_limits

FIXTURES = Path(__file__).parent / "fixtures"

_SRS_ARMS = [
    ("r1pro_left", "left_arm_base_link", "left_arm_link7"),
    ("r1pro_right", "right_arm_base_link", "right_arm_link7"),
    ("openarm_left", "openarm_left_base_link", "openarm_left_ee_base_link"),
    ("openarm_right", "openarm_right_base_link", "openarm_right_ee_base_link"),
]


def _limits(kb):
    out = []
    for j in kb.joints:
        lo_hi = j.limits
        if lo_hi is None or lo_hi[0] is None or lo_hi[1] is None:
            out.append((-np.pi, np.pi))
        else:
            out.append((float(lo_hi[0]), float(lo_hi[1])))
    return out


@pytest.mark.parametrize(("name", "base", "ee"), _SRS_ARMS, ids=[a[0] for a in _SRS_ARMS])
def test_resolves_every_in_limits_pose(name: str, base: str, ee: str) -> None:
    """Every reachable *in-limits* pose has an exact in-limits IK, and the
    resolver returns one: in-limits + FK-closing to machine precision. This is
    the guarantee the blind sweep cannot make (it samples, the resolver solves).
    """
    kb = load_urdf_kinbody_normalized(FIXTURES / f"{name}.urdf", base, ee)
    lims = _limits(kb)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(60):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        T = poe_forward_kinematics(kb, q)
        sols = resolve_in_limits(kb, T)
        assert sols, f"{name}: no in-limits solution for a reachable in-limits pose"
        in_lim = [
            s
            for s in sols
            if all(lims[i][0] - 1e-9 <= s.q[i] <= lims[i][1] + 1e-9 for i in range(7))
        ]
        assert in_lim, f"{name}: resolver returned only out-of-limits solutions"
        best = min(float(np.linalg.norm(poe_forward_kinematics(kb, s.q) - T)) for s in in_lim)
        worst = max(worst, best)
    assert worst < 1e-9, f"{name}: worst in-limits FK closure {worst:.2e}"


def test_resolves_approximate_srs_gen3() -> None:
    """Kinova Gen3 is only *approximately* SRS (~12 mm axis drift), so the exact
    resolver no-ops; the approximate path (#370) seeds from best-fit pivots and
    LM-polishes to the true FK. At the #299/#359 gen3 gap pose (raw solve returns
    a set none of which is in-limits) it must return in-limits ``"lm"`` solutions.
    """
    kb = load_urdf_kinbody_normalized(FIXTURES / "gen3.urdf", "base_link", "end_effector_link")
    lims = _limits(kb)
    q_gap = np.array(
        [
            1.6474334178,
            0.5114245864,
            -1.1049209077,
            1.1166184280,
            -0.0972974297,
            2.0879156524,
            1.7343580230,
        ]
    )
    T = poe_forward_kinematics(kb, q_gap)
    sols = resolve_in_limits(kb, T)
    assert sols, "gen3: approximate resolver returned no in-limits solution at the gap pose"
    for s in sols:
        assert all(lims[i][0] - 1e-9 <= s.q[i] <= lims[i][1] + 1e-9 for i in range(7))
        assert s.refinement_used == "lm"
        assert float(np.linalg.norm(poe_forward_kinematics(kb, s.q) - T)) < 1e-8


def test_returns_empty_for_non_srs_chain() -> None:
    """Non-SRS chains (no spherical shoulder+wrist) are not resolvable this way;
    the resolver must no-op so it is safe as a universal thin-wrapper fallback."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "franka_panda.urdf", "panda_link0", "panda_link8")
    T = poe_forward_kinematics(kb, np.zeros(len(kb.joints)))
    assert resolve_in_limits(kb, T) == []
