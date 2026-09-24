"""In-limits arc finding (``ssik.solvers.seven_r._feasible_param``, shared by
the SRS and spherical-shoulder in-limits resolvers) and the seeded
``max_solutions=1`` cap on a solver that takes no seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics

FIXTURES = Path(__file__).parent / "fixtures"


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _random_q(kb, rng):
    return np.array([rng.uniform(lo, hi) for lo, hi in (j.limits for j in kb.joints)])


def test_feasible_arcs_catch_an_excursion_between_grid_points() -> None:
    """A joint that leaves its range and returns between two points of the
    bracketing grid: a bump 0.24 past the stop and 0.004 wide (a Panda branch
    had one 0.243 rad past a stop over 0.01 of its coordinate). The grid is now
    refined wherever a joint steps or bends more than 0.02 rad."""
    from ssik.solvers.seven_r._feasible_param import PARAM_GRID, feasible_arcs

    def q_scalar(t):
        bump = 1.24 * np.exp(-(((t - 0.3) / 0.002) ** 2))
        return np.array([0.5 * np.sin(t) + bump])

    q_grid = np.array([q_scalar(t) for t in PARAM_GRID])
    arcs = feasible_arcs(q_scalar, q_grid, (0,), [(-1.0, 1.0)], PARAM_GRID)
    t = np.linspace(-np.pi, np.pi, 400001, endpoint=False)
    inside = np.array([abs(q_scalar(x)[0]) <= 1.0 for x in t])
    covered = np.zeros_like(inside)
    for lo, hi in arcs:
        covered |= (t >= lo - 1e-9) & (t <= hi + 1e-9)
    assert not np.any(covered & ~inside), arcs  # nothing out of range is kept
    assert not np.any(inside & ~covered), arcs  # nothing in range is lost


def test_feasible_arc_survives_a_flip_at_its_midpoint() -> None:
    """At a wrist gimbal lock a branch's coordinate flips by pi at one point. When
    that point is exactly the midpoint of a feasible arc, a midpoint test threw the
    whole arc away (an iiwa14 joint in range over 92% of the circle came back never
    in range). The grid samples inside the arc decide it instead."""
    from ssik.solvers.seven_r._feasible_param import PARAM_GRID, arcs_for_joint

    lo, hi = -2.9, 2.9
    bad = (-2.6, -2.4)  # a genuine out-of-range stretch: q = 3.0 there

    def smooth(t):
        return 3.0 if bad[0] < t < bad[1] else 0.1

    # The long feasible arc runs from -2.4 round to -2.6 + 2pi; its midpoint:
    mid = float(((bad[1] + bad[0] + 2 * np.pi) / 2 + np.pi) % (2 * np.pi) - np.pi)

    def flipped(t):
        # 1e-9 wide: far below the grid step, wide enough to hold the arc's
        # midpoint, which the bisected boundaries place within ~1e-12 of `mid`.
        return 0.1 + np.pi if abs(t - mid) < 1e-9 else smooth(t)  # -3.04: out of range

    q_col = np.array([smooth(t) for t in PARAM_GRID])
    for q_of in (smooth, flipped):
        arcs = arcs_for_joint(q_of, lo, hi, PARAM_GRID, q_col)
        covered = sum(b - a for a, b in arcs)
        assert covered == pytest.approx(2 * np.pi - 0.2, abs=1e-6), (q_of.__name__, arcs)


def test_seeded_solve_capped_to_one_returns_the_nearest() -> None:
    """The tracking idiom `solve(T, q_seed=q, max_solutions=1, respect_limits=False)`
    on a solver that takes no seed (SRS): the cap used to reach the solver before
    the seed ranking, returning its first branch instead of the nearest."""
    kb = load_urdf_kinbody_normalized(FIXTURES / "kuka_iiwa14.urdf", "base", "iiwa_link_ee_kuka")
    arm = ssik.Manipulator(kb)
    rng = np.random.default_rng(31)
    for _ in range(5):
        q = _random_q(kb, rng)
        T = poe_forward_kinematics(kb, q)
        seed = q + rng.uniform(-0.05, 0.05, 7)
        full = arm.solve(T, respect_limits=False)
        nearest = min(float(np.max(np.abs(_wrap(s.q - seed)))) for s in full)
        (top,) = arm.solve(T, q_seed=seed, max_solutions=1, respect_limits=False)
        assert float(np.max(np.abs(_wrap(top.q - seed)))) == pytest.approx(nearest, abs=1e-9)
