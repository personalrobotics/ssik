"""Seeded nearest-to-seed tracking: ``seed_metric`` (#330) + the 7R
lock-outward-from-seed fast path (#331).

Two coupled guarantees on the ``q_seed`` contract:

* **#330** -- ``solve(T, q_seed=q, max_solutions=k)`` returns the ``k``
  solutions nearest ``q_seed`` ranked by ``seed_metric``. The default is
  ``"wrap_linf"`` (minimise the *largest* single-joint move), which holds the
  branch during trajectory tracking where the summed-distance ``"wrap_l2"``
  would let a big single-joint flip hide behind smaller moves elsewhere.
* **#331** -- on jointlock-7R arms the seeded path walks lock samples outward
  from ``q_seed[lock_idx]`` and L-infinity-ranks the first yielding slice's
  *full* branch set, so it returns the same nearest config as the exhaustive
  sweep in ~1 sub-solve instead of ~16.
"""

from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik._kinbody import KinBody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.postprocess import nearest_to_seed
from ssik.solvers.jointlock import seven_r as jointlock_seven_r

FIXTURES = Path(__file__).parent / "fixtures"

# A franka pose + seed where the L2-nearest and L-infinity-nearest branches
# differ (found by enumeration; see #330). The L-infinity pick has a strictly
# smaller worst-single-joint move -- the anti-flip property.
_DIVERGENT_QTRUE = np.array([-0.5455, 0.0149, 0.819, -0.2425, -0.2675, -1.247, 0.1981])
_DIVERGENT_QSEED = np.array([-1.1469, 0.4888, 1.9194, -0.2181, -1.391, 1.9702, -0.4396])


def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def _linf(q_seed: np.ndarray, sol: object) -> float:
    return float(np.abs(_wrap(np.asarray(sol.q) - q_seed)).max())  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def franka_kb() -> KinBody:
    return load_urdf_kinbody_normalized(
        FIXTURES / "franka_panda.urdf", "panda_link0", "panda_link8"
    )


# ---------------------------------------------------------------------------
# #330 -- seed_metric plumbs through the baked artifact solve().
# ---------------------------------------------------------------------------


def test_seed_metric_plumbs_through_artifact_solve() -> None:
    """``seed_metric`` is forwarded to the artifact ``solve()`` ranking pass:
    both metrics are accepted and return a real nearest IK, and the default is
    ``wrap_linf``. (The metrics' *divergence* is unit-tested directly on
    ``nearest_to_seed`` in test_postprocess; Franka now uses the spherical-shoulder
    specialist, whose dense redundancy sampling makes the two metrics agree.)"""
    from ssik.prebuilt import franka_panda_ik

    T = franka_panda_ik.fk(_DIVERGENT_QTRUE)

    l2 = franka_panda_ik.solve(
        T, q_seed=_DIVERGENT_QSEED, max_solutions=1, respect_limits=False, seed_metric="wrap_l2"
    )
    linf = franka_panda_ik.solve(
        T, q_seed=_DIVERGENT_QSEED, max_solutions=1, respect_limits=False, seed_metric="wrap_linf"
    )
    default = franka_panda_ik.solve(
        T, q_seed=_DIVERGENT_QSEED, max_solutions=1, respect_limits=False
    )
    assert l2
    assert linf
    assert default
    # Default is wrap_linf.
    assert np.allclose(default[0].q, linf[0].q)
    # Both return a real IK that FK-closes.
    for sols in (l2, linf):
        err = float(np.max(np.abs(franka_panda_ik.fk(sols[0].q) - T)))
        assert err < 1e-9, f"FK closure {err:.2e}"


def test_bad_seed_metric_raises() -> None:
    from ssik.prebuilt import franka_panda_ik

    T = franka_panda_ik.fk(_DIVERGENT_QTRUE)
    with pytest.raises(ValueError, match="unknown metric"):
        franka_panda_ik.solve(T, q_seed=_DIVERGENT_QSEED, seed_metric="bogus")


def test_seed_metric_plumbs_through_manipulator() -> None:
    """``Manipulator.solve`` forwards ``seed_metric`` to the ranking pass: both
    metrics are accepted and return a real nearest IK (divergence itself is
    unit-tested on ``nearest_to_seed`` in test_postprocess)."""
    arm = ssik.Manipulator.from_urdf(
        FIXTURES / "franka_panda.urdf", base="panda_link0", ee="panda_link8"
    )
    T = arm.fk(_DIVERGENT_QTRUE)
    l2 = arm.solve(
        T, q_seed=_DIVERGENT_QSEED, max_solutions=1, respect_limits=False, seed_metric="wrap_l2"
    )
    linf = arm.solve(
        T, q_seed=_DIVERGENT_QSEED, max_solutions=1, respect_limits=False, seed_metric="wrap_linf"
    )
    assert l2
    assert linf
    for sols in (l2, linf):
        assert float(np.max(np.abs(arm.fk(sols[0].q) - T))) < 1e-9


# ---------------------------------------------------------------------------
# #331 -- 7R seeded fast path matches the exhaustive nearest, cheaply.
# ---------------------------------------------------------------------------


def test_jointlock_seeded_matches_exhaustive_nearest(franka_kb: KinBody) -> None:
    """Across a smooth trajectory, the seeded fast path's nearest solution is
    exactly as close to the seed as the exhaustive sweep's nearest -- it never
    settles for a farther branch."""
    rng = np.random.default_rng(2)
    q_prev = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    worst_gap = 0.0
    for _ in range(15):
        q_true = q_prev + rng.normal(scale=0.02, size=7)
        T = poe_forward_kinematics(franka_kb, q_true)

        seeded, _ = jointlock_seven_r.solve(
            franka_kb, T, q_seed=q_prev, max_solutions=1, respect_limits=False
        )
        seeded = nearest_to_seed(seeded, q_prev, metric="wrap_linf")
        exhaustive, _ = jointlock_seven_r.solve(franka_kb, T, respect_limits=False)
        assert seeded
        assert exhaustive

        seeded_best = _linf(q_prev, seeded[0])
        exhaustive_best = min(_linf(q_prev, s) for s in exhaustive)
        worst_gap = max(worst_gap, seeded_best - exhaustive_best)
        q_prev = np.asarray(seeded[0].q)
    assert worst_gap < 1e-9, f"seeded path missed the nearest branch by {worst_gap:.2e} rad"


def test_jointlock_seeded_returns_full_branch_set(franka_kb: KinBody) -> None:
    """When seeded, jointlock hands back the first yielding slice's *full*
    branch set (not a single branch) so the caller can rank by ``seed_metric``.
    This guards the "skip the in-solver trim when seeded" change in #331."""
    q_seed = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = poe_forward_kinematics(franka_kb, q_seed)
    seeded, _ = jointlock_seven_r.solve(
        franka_kb, T, q_seed=q_seed, max_solutions=1, respect_limits=False
    )
    assert len(seeded) > 1, "seeded solve must return the full branch set for ranking, not 1"


def test_jointlock_seeded_dispatches_fewer_than_exhaustive(franka_kb: KinBody) -> None:
    """The lock-outward early-exit (#331) evaluates far fewer inner sub-solves
    than the exhaustive sweep -- the speed win, asserted structurally (no
    wall-clock)."""
    q_seed = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])
    T = poe_forward_kinematics(franka_kb, q_seed)
    real_dispatch = jointlock_seven_r._dispatch  # type: ignore[attr-defined]

    def _count(*args: object, **kwargs: object) -> object:
        _count.n += 1  # type: ignore[attr-defined]
        return real_dispatch(*args, **kwargs)  # type: ignore[arg-type]

    def _run(**solve_kwargs: object) -> int:
        _count.n = 0  # type: ignore[attr-defined]
        jointlock_seven_r._dispatch = _count  # type: ignore[attr-defined,assignment]
        try:
            jointlock_seven_r.solve(franka_kb, T, respect_limits=False, **solve_kwargs)  # type: ignore[arg-type]
        finally:
            jointlock_seven_r._dispatch = real_dispatch  # type: ignore[attr-defined]
        return _count.n  # type: ignore[attr-defined,no-any-return]

    seeded_dispatches = _run(q_seed=q_seed, max_solutions=1)
    exhaustive_dispatches = _run()
    assert seeded_dispatches < exhaustive_dispatches, (
        f"seeded {seeded_dispatches} not < exhaustive {exhaustive_dispatches}"
    )


# ---------------------------------------------------------------------------
# #333 -- seed_tolerance: hard per-joint deviation bound (postprocess filter).
# ---------------------------------------------------------------------------

_TRACK_SEED = np.array([0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5])


def test_seed_tolerance_requires_q_seed() -> None:
    from ssik.prebuilt import franka_panda_ik

    T = franka_panda_ik.fk(np.zeros(7))
    with pytest.raises(ValueError, match="seed_tolerance requires q_seed"):
        franka_panda_ik.solve(T, seed_tolerance=0.1)


def test_seed_tolerance_manipulator_requires_q_seed() -> None:
    arm = ssik.Manipulator.from_urdf(
        FIXTURES / "franka_panda.urdf", base="panda_link0", ee="panda_link8"
    )
    with pytest.raises(ValueError, match="seed_tolerance requires q_seed"):
        arm.solve(arm.fk(np.zeros(7)), seed_tolerance=0.1)


def test_seed_tolerance_returned_solutions_respect_bound() -> None:
    """The guarantee: every returned solution is within the per-joint bound.

    Seeding at an actual solution (deviation 0 from itself) keeps the set
    non-empty, so the bound check is never vacuous.
    """
    from ssik.prebuilt import franka_panda_ik

    rng = np.random.default_rng(0)
    tol = np.deg2rad(6)
    checked = 0
    for _ in range(100):
        q = rng.uniform(-1.5, 1.5, size=7)
        T = franka_panda_ik.fk(q)
        any_sols = franka_panda_ik.solve(T, respect_limits=False)
        if not any_sols:
            continue
        seed = np.asarray(any_sols[0].q)
        sols = franka_panda_ik.solve(T, q_seed=seed, seed_tolerance=tol, respect_limits=False)
        assert sols, "seeding at an actual solution must keep at least itself"
        for s in sols:
            assert np.abs(_wrap(np.asarray(s.q) - seed)).max() <= tol + 1e-9
            checked += 1
    assert checked > 0


def test_seed_tolerance_large_keeps_all_zero_keeps_none() -> None:
    from ssik.prebuilt import franka_panda_ik

    T = franka_panda_ik.fk(_TRACK_SEED)
    base = franka_panda_ik.solve(T, q_seed=_TRACK_SEED, respect_limits=False)
    huge = franka_panda_ik.solve(T, q_seed=_TRACK_SEED, seed_tolerance=10.0, respect_limits=False)
    assert {tuple(np.round(s.q, 9)) for s in huge} == {tuple(np.round(s.q, 9)) for s in base}
    # max wrap-to-pi deviation is always <= pi, so 10 rad keeps everything.
    zero = franka_panda_ik.solve(T, q_seed=_TRACK_SEED, seed_tolerance=0.0, respect_limits=False)
    assert zero == []


def test_seed_tolerance_none_matches_no_tolerance() -> None:
    from ssik.prebuilt import franka_panda_ik

    T = franka_panda_ik.fk(_TRACK_SEED)
    a = franka_panda_ik.solve(T, q_seed=_TRACK_SEED, max_solutions=1, respect_limits=False)
    b = franka_panda_ik.solve(
        T, q_seed=_TRACK_SEED, max_solutions=1, seed_tolerance=None, respect_limits=False
    )
    assert [tuple(np.round(s.q, 12)) for s in a] == [tuple(np.round(s.q, 12)) for s in b]


def test_seed_tolerance_manipulator_honors_bound() -> None:
    arm = ssik.Manipulator.from_urdf(
        FIXTURES / "franka_panda.urdf", base="panda_link0", ee="panda_link8"
    )
    T = arm.fk(_TRACK_SEED)
    any_sols = arm.solve(T, respect_limits=False)
    seed = np.asarray(any_sols[0].q)
    tol = np.deg2rad(6)
    sols = arm.solve(T, q_seed=seed, seed_tolerance=tol, respect_limits=False)
    assert sols
    for s in sols:
        assert np.abs(_wrap(np.asarray(s.q) - seed)).max() <= tol + 1e-9
