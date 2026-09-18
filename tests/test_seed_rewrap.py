"""Seed-relative rewrapping for wide-limit / continuous joints (#562, step 1).

A seeded solve must return the ``q_i + 2*pi*k`` representative nearest the seed
(within finite limits), so ``solve(T, q_seed=q_current, max_solutions=1)`` never
commands a gratuitous 2*pi turn on a joint whose limits span > 2*pi (UR family,
continuous joints). This is a returned-value fix only: unseeded result counts are
unchanged, and native == Python.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik._native import native_available
from ssik.core.solution import Solution
from ssik.postprocess import rewrap_to_seed


def _kb(arm: str):
    return importlib.import_module(f"ssik.prebuilt.{arm}")._KB


def test_rewrap_helper_finite_wide_limit_picks_seed_nearest_winding() -> None:
    kb = _kb("universal_robots.ur5e_ik")  # j0 has limits [-2pi, 2pi]
    # principal-ish value near -0.28; seed near its +2pi winding (6.0).
    sol = Solution(q=np.array([-0.283, 0.0, 0.0, 0.0, 0.0, 0.0]), fk_residual=0.0)
    seed = np.array([6.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    (out,) = rewrap_to_seed([sol], kb, seed)
    assert out.q[0] == pytest.approx(-0.283 + 2 * np.pi, abs=1e-9)  # 6.0-side winding
    assert -2 * np.pi <= out.q[0] <= 2 * np.pi


def test_rewrap_helper_boundary_zero_principal() -> None:
    # principal 0 with [-2pi, 2pi] admits {-2pi, 0, 2pi}; seed near +2pi -> +2pi.
    kb = _kb("universal_robots.ur5e_ik")
    sol = Solution(q=np.zeros(6), fk_residual=0.0)
    seed = np.array([6.2, 0, 0, 0, 0, 0])
    (out,) = rewrap_to_seed([sol], kb, seed)
    assert out.q[0] == pytest.approx(2 * np.pi, abs=1e-9)


def test_rewrap_helper_continuous_joint_nearest_turn() -> None:
    # jaco2 has continuous (limits=None) joints; nearest turn, no clamp.
    kb = _kb("kinova.jaco2_ik")
    cont = next(i for i, j in enumerate(kb.joints) if j.limits is None)
    q = np.zeros(6)
    q[cont] = 0.1
    seed = np.zeros(6)
    seed[cont] = 0.1 + 4 * np.pi  # two turns away
    (out,) = rewrap_to_seed([Solution(q=q, fk_residual=0.0)], kb, seed)
    assert out.q[cont] == pytest.approx(0.1 + 4 * np.pi, abs=1e-9)


@pytest.mark.skipif(not native_available(), reason="native extension not built")
@pytest.mark.parametrize("native", [True, False])
def test_seeded_solve_returns_seed_winding_ur5e(native: bool) -> None:
    m = importlib.import_module("ssik.prebuilt.universal_robots.ur5e_ik")
    # a real config with joints wound near +2pi (inside [-2pi, 2pi]).
    q = np.array([6.0, -0.6, 0.9, 5.9, -0.5, 0.2])
    T = m.fk(q)
    sols = m.solve(T, q_seed=q, max_solutions=1, native=native)
    assert sols, "seeded solve returned nothing"
    s = sols[0].q
    assert np.max(np.abs(s - q)) < 1e-6, f"seeded solve jumped: {s.tolist()} vs seed {q.tolist()}"


@pytest.mark.skipif(not native_available(), reason="native extension not built")
def test_seeded_native_matches_python_ur5e() -> None:
    m = importlib.import_module("ssik.prebuilt.universal_robots.ur5e_ik")
    rng = np.random.default_rng(0)
    ranges = [j.limits if j.limits else (-np.pi, np.pi) for j in m._KB.joints]
    for _ in range(20):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        T = m.fk(q)
        seed = q + rng.uniform(-0.1, 0.1, 6)
        nat = m.solve(T, q_seed=seed, max_solutions=1, native=True)
        pyth = m.solve(T, q_seed=seed, max_solutions=1, native=False)
        assert len(nat) == len(pyth)
        if nat:
            assert np.allclose(nat[0].q, pyth[0].q, atol=1e-6), "native/python seed rep differ"


def test_unseeded_count_unchanged_ur5e() -> None:
    # Step 1 must not change unseeded counts (MINOR-safe): principal reps only.
    m = importlib.import_module("ssik.prebuilt.universal_robots.ur5e_ik")
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.uniform(-2.0, 2.0, 6)
        assert 1 <= len(m.solve(m.fk(q))) <= 8
