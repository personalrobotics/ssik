"""Bulletproof coverage for the spherical-shoulder 7R specialist (#373).

The jointlock blind sweep (and any lock-swap shortcut) drops a small fraction of
reachable in-limits poses whose redundancy interval is narrow. This solver
resolves the q6 redundancy exactly (SP3 reachability bracket x in-limits
feasible arcs on the closed-form q_i(q6)), so **every** reachable in-limits pose
returns an in-limits, FK-closing solution -- the guarantee the sweep cannot make.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.seven_r.spherical_shoulder import resolve_in_limits

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"

_ARMS = [
    ("franka_panda", "panda_link0", "panda_link8"),
    ("fr3", "fr3_link0", "fr3_link8"),
]


def _limits(kb):
    return [(float(j.limits[0]), float(j.limits[1])) for j in kb.joints]


@pytest.mark.parametrize(("name", "base", "ee"), _ARMS, ids=[a[0] for a in _ARMS])
def test_resolves_every_in_limits_pose(name: str, base: str, ee: str) -> None:
    """Every reachable in-limits pose gets an in-limits, FK-closing solution --
    0 coverage misses, machine-precision FK. This is the bulletproof guarantee
    the blind lock-sweep drops ~1% of the time."""
    kb = load_urdf_kinbody_normalized(FIXTURES / f"{name}.urdf", base, ee)
    lims = _limits(kb)
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(150):
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


def test_returns_empty_for_non_7r_chain() -> None:
    """A non-7R chain is not this solver's class -- it must no-op (safe as a
    thin-wrapper fallback)."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "ur5.urdf", "base_link", "ee_link")
    T = poe_forward_kinematics(kb, np.zeros(len(kb.joints)))
    assert resolve_in_limits(kb, T) == []
