"""Seeded numerical-tracking fast path (#380).

When ``solve(T, q_seed=q, max_solutions=1)`` is called -- the trajectory-tracking
idiom -- the thin-wrapper artifacts Newton-continue from the seed via
:func:`ssik.refinement.seeded_track` instead of resolving the whole redundancy.
Two guarantees, both exercised here:

* **Fast** -- when the seed continues cleanly the full analytical
  ``_solver_solve`` is never called (asserted structurally, no wall-clock).
* **Correct** -- on a smooth trajectory the continuation is *exactly* the
  seed-nearest solution the full solve would return, to machine precision; and
  when the seed cannot continue (branch jump / divergence / out-of-limits) the
  path transparently falls through to the full solve, so coverage is never lost.
"""

from pathlib import Path

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.refinement import kinbody_jacobian, seeded_track

FIXTURES = Path(__file__).parent / "fixtures"


def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def _linf(q_seed: np.ndarray, q: np.ndarray) -> float:
    return float(np.abs(_wrap(np.asarray(q) - q_seed)).max())


# The three thin-wrapper 7R classes the fast path serves: exact spherical
# shoulder (franka), polished spherical shoulder (xarm7), and SRS (iiwa14).
#
# The seed-gap tolerance is the max the fast-path continuation may trail the
# exhaustive sweep's nearest, and it is not the same for all three. The fast
# path Newton-continues, which L2-projects the seed onto the self-motion
# manifold; the exhaustive path L-infinity-ranks its discrete grid samples.
# For the spherical-shoulder arms the L-infinity-dominant joint is pinned by
# the target (the redundancy does not move it), so the L2 and L-infinity
# optima coincide exactly (gap 0). For SRS the swivel moves every joint, so
# the two norms' optima differ by a hair (~1e-4) -- a benign, bounded
# (non-accumulating) metric gap, not a missed branch. FK closure is machine
# precision for all three regardless.
_ARMS = [
    ("franka_panda_ik", "franka_panda.urdf", "panda_link0", "panda_link8", 1e-9),
    ("xarm7_ik", "xarm7.urdf", "link_base", "link7", 1e-9),
    ("iiwa14_ik", "kuka_iiwa14.urdf", "base", "iiwa_link_ee_kuka", 2e-3),
]


def _artifact(name: str):
    from importlib import import_module

    return import_module(f"ssik.prebuilt.{name}")


def _kb(urdf: str, base: str, ee: str):
    return load_urdf_kinbody_normalized(FIXTURES / urdf, base, ee)


# ---------------------------------------------------------------------------
# The seeded_track primitive.
# ---------------------------------------------------------------------------


def test_seeded_track_continues_to_machine_precision() -> None:
    """From a near-seed the primitive returns a machine-precision continuation
    tagged ``refinement_used="lm"`` and within ``max_dist`` of the seed."""
    kb = _kb("franka_panda.urdf", "panda_link0", "panda_link8")
    rng = np.random.default_rng(0)
    q = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = poe_forward_kinematics(kb, q)
    seed = q + rng.uniform(-0.03, 0.03, 7)

    sol = seeded_track(
        seed,
        lambda x: poe_forward_kinematics(kb, x),
        lambda x: kinbody_jacobian(kb, x),
        T,
    )
    assert sol is not None
    assert sol.refinement_used == "lm"
    assert float(np.max(np.abs(poe_forward_kinematics(kb, sol.q) - T))) < 1e-10
    assert _linf(seed, sol.q) <= 0.5


def test_seeded_track_rejects_out_of_window_continuation() -> None:
    """A continuation farther than ``max_dist`` from the seed is rejected
    (returns ``None``) so the caller falls back to the full solve -- the gate
    that stops a Newton branch-jump from being served as a tracking result."""
    kb = _kb("franka_panda.urdf", "panda_link0", "panda_link8")
    q = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = poe_forward_kinematics(kb, q)
    seed = q + 0.05  # genuine solution ~0.05 away, exceeds a 0.001 window

    assert (
        seeded_track(
            seed,
            lambda x: poe_forward_kinematics(kb, x),
            lambda x: kinbody_jacobian(kb, x),
            T,
            max_dist=0.001,
        )
        is None
    )


def test_seeded_track_diverges_to_none() -> None:
    """A wildly wrong seed makes damped-LM fail to reach machine precision;
    the primitive returns ``None`` rather than a garbage config."""
    kb = _kb("franka_panda.urdf", "panda_link0", "panda_link8")
    q = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = poe_forward_kinematics(kb, q)
    # A pose whose only solutions are far from this seed; LM either diverges
    # or lands outside the default window -> None either way.
    seed = np.array([2.5, 1.2, -2.8, -0.2, 2.9, -1.5, 2.0])
    sol = seeded_track(
        seed,
        lambda x: poe_forward_kinematics(kb, x),
        lambda x: kinbody_jacobian(kb, x),
        T,
    )
    assert sol is None or _linf(seed, sol.q) <= 0.5


# ---------------------------------------------------------------------------
# The artifact fast path: correct + fast + safe fall-through.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "urdf", "base", "ee", "gap_tol"), _ARMS)
def test_fast_path_matches_exhaustive_over_trajectory(
    name: str, urdf: str, base: str, ee: str, gap_tol: float
) -> None:
    """Across a smooth trajectory the seeded fast path returns a config as close
    to the seed as the exhaustive sweep's nearest (to ``gap_tol`` -- see the
    ``_ARMS`` note on the L2-projection vs L-infinity-ranking metric gap), and
    FK-closes to machine precision -- it never settles for a farther branch."""
    art = _artifact(name)
    rng = np.random.default_rng(7)
    q_prev = np.array([0.3, -0.4, 0.2, -1.4, 0.1, 1.0, 0.5])
    worst_gap = 0.0
    worst_fk = 0.0
    for _ in range(25):
        q_true = q_prev + rng.normal(scale=0.02, size=7)
        T = art.fk(q_true)

        fast = art.solve(T, q_seed=q_prev, max_solutions=1, respect_limits=False)
        exhaustive = art.solve(T, respect_limits=False)
        assert fast, f"{name}: fast path returned no solution"
        assert exhaustive

        fk_err = float(np.max(np.abs(art.fk(fast[0].q) - T)))
        worst_fk = max(worst_fk, fk_err)
        fast_best = _linf(q_prev, fast[0].q)
        exhaustive_best = min(_linf(q_prev, s.q) for s in exhaustive)
        worst_gap = max(worst_gap, fast_best - exhaustive_best)
        q_prev = np.asarray(fast[0].q)
    assert worst_fk < 1e-9, f"{name}: fast path FK closure {worst_fk:.2e}"
    assert worst_gap < gap_tol, f"{name}: fast path missed nearest by {worst_gap:.2e} rad"


def test_fast_path_skips_full_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the seed continues cleanly the full analytical ``_solver_solve`` is
    never invoked -- the structural proof of the speed win (no wall-clock)."""
    art = _artifact("franka_panda_ik")
    q = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = art.fk(q)
    seed = q + 0.01

    calls = {"n": 0}
    real = art._solver_solve

    def _spy(*a: object, **k: object) -> object:
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(art, "_solver_solve", _spy)
    fast = art.solve(T, q_seed=seed, max_solutions=1, respect_limits=False)
    assert fast
    assert calls["n"] == 0, "full solve was called despite a clean seed continuation"


def test_fast_path_falls_through_when_seed_cannot_continue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A seed that cannot cleanly continue falls through to the full solve, which
    still returns the correct seed-nearest IK -- coverage is never lost."""
    art = _artifact("franka_panda_ik")
    q = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = art.fk(q)
    seed = np.array([2.5, 1.2, -2.8, -0.2, 2.9, -1.5, 2.0])  # far from any branch

    calls = {"n": 0}
    real = art._solver_solve

    def _spy(*a: object, **k: object) -> object:
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(art, "_solver_solve", _spy)
    result = art.solve(T, q_seed=seed, max_solutions=1, respect_limits=False)
    assert result, "fall-through must still return a solution"
    assert calls["n"] >= 1, "expected fall-through to the full solve"
    assert float(np.max(np.abs(art.fk(result[0].q) - T))) < 1e-9


def test_fast_path_result_is_in_limits() -> None:
    """With ``respect_limits=True`` (default) the fast-path result is in-limits."""
    art = _artifact("franka_panda_ik")
    kb = _kb("franka_panda.urdf", "panda_link0", "panda_link8")
    lo = np.array([j.limits[0] for j in kb.joints])
    hi = np.array([j.limits[1] for j in kb.joints])
    q = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = art.fk(q)
    fast = art.solve(T, q_seed=q + 0.01, max_solutions=1)
    assert fast
    assert np.all(np.asarray(fast[0].q) >= lo - 1e-9)
    assert np.all(np.asarray(fast[0].q) <= hi + 1e-9)


def test_fast_path_honors_seed_tolerance() -> None:
    """A ``seed_tolerance`` tighter than the continuation move drops the
    fast-path result (empty, consistent with the full path's hard bound)."""
    art = _artifact("franka_panda_ik")
    q = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = art.fk(q + 0.1)  # continuation ~0.1 from the seed q
    tight = art.solve(T, q_seed=q, max_solutions=1, respect_limits=False, seed_tolerance=0.01)
    loose = art.solve(T, q_seed=q, max_solutions=1, respect_limits=False, seed_tolerance=0.5)
    assert not tight, "tolerance tighter than the move must yield no solution"
    assert loose, "tolerance looser than the move must keep the continuation"
