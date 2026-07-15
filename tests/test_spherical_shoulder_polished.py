"""Bulletproof coverage for the approximately-spherical-shoulder 7R class (#373).

uFactory xArm7 is *nearly* the exact spherical-shoulder class -- the closed-form
q_i(q6) recipe gives excellent seeds but a ~1e-8 residual (this was the #159
precision floor). LM polish recovers machine precision, and the resolver keeps
the exact-in-limits guarantee: every reachable in-limits pose returns an
in-limits, FK-closing solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.seven_r.spherical_shoulder_polished import (
    is_approximately_spherical_shoulder_7r,
    resolve_in_limits,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _xarm7():
    return load_urdf_kinbody_normalized(FIXTURES / "xarm7.urdf", "link_base", "link7")


def _limits(kb):
    return [(float(j.limits[0]), float(j.limits[1])) for j in kb.joints]


def test_predicate_accepts_xarm7_only() -> None:
    """The approximate predicate accepts xArm7 but not the exact class (Franka),
    nor arms whose wrist drift is too large (rizon4)."""
    assert is_approximately_spherical_shoulder_7r(_xarm7())
    franka = load_urdf_kinbody_normalized(
        FIXTURES / "franka_panda.urdf", "panda_link0", "panda_link8"
    )
    assert not is_approximately_spherical_shoulder_7r(franka)  # exact class, no polish
    rizon4 = load_urdf_kinbody_normalized(FIXTURES / "rizon4.urdf", "base_link", "flange")
    assert not is_approximately_spherical_shoulder_7r(rizon4)  # 32 mm drift, too large


def test_resolves_every_in_limits_pose_xarm7() -> None:
    """Every reachable in-limits xArm7 pose returns an in-limits, FK-closing
    solution at machine precision -- 0 coverage misses. Closes the #159 precision
    floor (blind sweep gave ~1e-6; the polished closed form gives ~1e-11)."""
    kb = _xarm7()
    lims = _limits(kb)
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(120):
        q = np.array([rng.uniform(lo, hi) for lo, hi in lims])
        T = poe_forward_kinematics(kb, q)
        sols = resolve_in_limits(kb, T)
        assert sols, "xarm7: no in-limits solution for a reachable in-limits pose"
        in_lim = [
            s
            for s in sols
            if all(lims[i][0] - 1e-9 <= s.q[i] <= lims[i][1] + 1e-9 for i in range(7))
        ]
        assert in_lim, "xarm7: resolver returned only out-of-limits solutions"
        best = min(float(np.linalg.norm(poe_forward_kinematics(kb, s.q) - T)) for s in in_lim)
        worst = max(worst, best)
    assert worst < 1e-9, f"xarm7: worst in-limits FK closure {worst:.2e}"


def test_returns_empty_for_non_7r_chain() -> None:
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    T = poe_forward_kinematics(kb, np.zeros(len(kb.joints)))
    assert resolve_in_limits(kb, T) == []
