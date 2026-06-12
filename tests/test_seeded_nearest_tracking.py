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
    """``seed_metric`` selects the ranking on the public artifact ``solve()``;
    ``wrap_linf`` (the default) yields a strictly smaller worst-joint move than
    ``wrap_l2`` on a divergent pose."""
    from ssik.prebuilt import franka_panda_ik

    T = franka_panda_ik._fk(_DIVERGENT_QTRUE)

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

    # The two metrics genuinely disagree, and wrap_linf wins on worst-joint move.
    assert not np.allclose(l2[0].q, linf[0].q), "metrics should pick different branches here"
    assert _linf(_DIVERGENT_QSEED, linf[0]) < _linf(_DIVERGENT_QSEED, l2[0])

    # Default is wrap_linf.
    assert np.allclose(default[0].q, linf[0].q)

    # Both are real IK solutions.
    for sols in (l2, linf):
        err = float(np.max(np.abs(franka_panda_ik._fk(sols[0].q) - T)))
        assert err < 1e-9, f"FK closure {err:.2e}"


def test_bad_seed_metric_raises() -> None:
    from ssik.prebuilt import franka_panda_ik

    T = franka_panda_ik._fk(_DIVERGENT_QTRUE)
    with pytest.raises(ValueError, match="unknown metric"):
        franka_panda_ik.solve(T, q_seed=_DIVERGENT_QSEED, seed_metric="bogus")


def test_seed_metric_plumbs_through_manipulator() -> None:
    """``Manipulator.solve`` forwards ``seed_metric`` to the ranking pass."""
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
    assert not np.allclose(l2[0].q, linf[0].q)
    assert _linf(_DIVERGENT_QSEED, linf[0]) < _linf(_DIVERGENT_QSEED, l2[0])


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
