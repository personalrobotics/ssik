"""End-to-end validation for :mod:`ssik.solvers.jointlock.seven_r`.

Universal 7R wrapper that locks one joint and dispatches the resulting
6R sub-chain to whichever ikgeo tier-0/1 solver matches its (per-sample)
topology. Tests cover:

- ``choose_lock_joint`` picks the topologically-best lock per arm
- Full FK-match correctness on synthetic 7R fixtures
- Targeted user-supplied samples recover seeded q* at machine precision
- 7-DOF guard
- Performance budget: 16-sample sweep < 0.5 s on synthetic SRS arm

Fixtures are synthetic (Franka-, iiwa-, Rizon-style geometries are
covered by the upcoming ``specialist.geofik`` and ``specialist.stereo_sew``
solvers; this is the generic cross-arm fallback).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from numpy.typing import NDArray

from ssik._kinbody import Joint, KinBody, Link
from ssik.postprocess import nearest_to_seed
from ssik.solvers.jointlock import seven_r


def _rodrigues(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    c, s = np.cos(t), np.sin(t)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R: NDArray[np.float64] = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R


def _axis_angle_4x4(k: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    T = np.eye(4)
    T[:3, :3] = _rodrigues(k, t)
    return T


def _fk(kb: KinBody, q: NDArray[np.float64]) -> NDArray[np.float64]:
    T = np.eye(4)
    for j, qi in zip(kb.joints, q, strict=True):
        T = T @ j.T_left @ _axis_angle_4x4(j.axis, qi) @ j.T_right
    return T


def _build_srs_with_spherical_wrist() -> KinBody:
    """Synthetic 7R: shoulder pitch+pitch (parallel), elbow roll, then
    a spherical wrist at joints 4-6. Locking joint 3 yields a 6R with
    spherical_two_parallel topology."""
    axes = [
        np.array([0.0, 0.0, 1.0]),  # j0 base z (shoulder yaw)
        np.array([0.0, -1.0, 0.0]),  # j1 shoulder pitch -y
        np.array([0.0, -1.0, 0.0]),  # j2 elbow pitch -y (parallel to j1)
        np.array([0.0, 0.0, 1.0]),  # j3 elbow roll z (lock candidate)
        np.array([0.0, -1.0, 0.0]),  # j4 wrist pitch -y
        np.array([0.0, 0.0, 1.0]),  # j5 wrist roll z
        np.array([0.0, -1.0, 0.0]),  # j6 final pitch -y
    ]
    t_lefts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.2]),
        np.array([0.4, 0.0, 0.0]),
        np.array([0.05, -0.1, 0.0]),
        np.array([0.0, 0.0, 0.4]),
        np.array([0.0, 0.0, 0.0]),  # j5 origin = j4 origin
        np.array([0.0, 0.0, 0.0]),  # j6 origin = j5 origin (spherical wrist)
    ]
    links = [Link(name=f"l{i}") for i in range(8)]
    joints = []
    for i in range(7):
        T_l = np.eye(4)
        T_l[:3, 3] = t_lefts[i]
        joints.append(
            Joint(
                name=f"j{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_l,
                T_right=np.eye(4),
                axis=axes[i],
                joint_type="revolute",
            )
        )
    return KinBody(links=links, joints=joints)


@pytest.fixture(scope="module")
def srs_kb() -> KinBody:
    return _build_srs_with_spherical_wrist()


# ---------------------------------------------------------------------------
# choose_lock_joint correctness.
# ---------------------------------------------------------------------------


def test_choose_lock_joint_picks_low_rank(srs_kb: KinBody) -> None:
    """For an arm where locking joint 3 yields three_parallel topology,
    the chooser should pick joint 3 (lowest rank = 0)."""
    lock_idx = seven_r.choose_lock_joint(srs_kb)
    assert lock_idx == 3, f"expected lock_idx=3, got {lock_idx}"


@pytest.mark.parametrize("lock_idx", range(7))
def test_lock_joint_fk_consistent_all_indices(srs_kb: KinBody, lock_idx: int) -> None:
    """The locked 6R sub-chain's FK must match the full 7R FK for *every* lock
    index, including the last joint. Locking the last joint previously dropped
    the locked transform entirely, yielding a ~2 m FK mismatch and silent
    wrong solutions (#374)."""
    rng = np.random.default_rng(lock_idx)
    for _ in range(8):
        q = rng.uniform(-np.pi, np.pi, size=7)
        sub = seven_r._lock_joint(srs_kb, lock_idx, float(q[lock_idx]))
        q_free = np.delete(q, lock_idx)
        assert np.allclose(_fk(sub, q_free), _fk(srs_kb, q), atol=1e-12), (
            f"lock_idx={lock_idx}: 6R sub-chain FK diverges from full 7R FK"
        )


# ---------------------------------------------------------------------------
# Targeted lock recovers seeded q*.
# ---------------------------------------------------------------------------


_SEEDED_Q = [
    np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4]),
    np.array([-1.2, -1.8, 2.1, -0.4, 0.7, -1.5, 0.8]),
    np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]),
]


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_targeted_sample_recovers_seeded_q_star(
    srs_kb: KinBody, q_star: NDArray[np.float64]
) -> None:
    """Passing the exact lock-joint value as a one-element sample list
    should recover the seeded q* at machine precision."""
    lock_idx = seven_r.choose_lock_joint(srs_kb)
    T_star = _fk(srs_kb, q_star)
    solutions, is_ls = seven_r.solve(srs_kb, T_star, lock_samples=[float(q_star[lock_idx])])
    assert not is_ls
    assert len(solutions) >= 1

    def _max_wrap(q: NDArray[np.float64]) -> float:
        return max(
            abs(((float(qi - qs) + np.pi) % (2 * np.pi)) - np.pi)
            for qi, qs in zip(q, q_star, strict=True)
        )

    best_dq = min(_max_wrap(s.q) for s in solutions)
    assert best_dq < 1e-10, f"seeded q* not recovered at machine precision; closest dq={best_dq}"


# ---------------------------------------------------------------------------
# Sweep correctness: every returned q reproduces T_target under FK.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_star", _SEEDED_Q)
def test_sweep_solutions_all_fk_match(srs_kb: KinBody, q_star: NDArray[np.float64]) -> None:
    """All solutions returned by the default 16-sample sweep must satisfy
    FK(q) == T_target at 1e-10 atol."""
    T_star = _fk(srs_kb, q_star)
    solutions, is_ls = seven_r.solve(srs_kb, T_star, lock_samples=16)
    assert not is_ls
    assert len(solutions) >= 8, f"expected at least 8 solutions, got {len(solutions)}"
    for i, sol in enumerate(solutions):
        T_check = _fk(srs_kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-10), (
            f"sweep solution {i} fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


def test_user_provided_samples_override(srs_kb: KinBody) -> None:
    """User-supplied samples should be used verbatim (not augmented)."""
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    custom = [0.0, 0.5, 1.1, -0.5]
    solutions, _ = seven_r.solve(srs_kb, T_star, lock_samples=custom)
    # Each lock value should produce up to 8 inner solutions.
    assert len(solutions) > 0
    # All solutions' lock-joint value should be in the custom list (within
    # the wrap-to-pi tolerance).
    lock_idx = seven_r.choose_lock_joint(srs_kb)
    for sol in solutions:
        ql = float(sol.q[lock_idx])
        assert any(abs(((ql - c + np.pi) % (2 * np.pi)) - np.pi) < 1e-6 for c in custom), (
            f"solution lock-joint {ql} not in user sample list"
        )


# ---------------------------------------------------------------------------
# Performance budget.
# ---------------------------------------------------------------------------


@pytest.mark.perf
def test_default_sweep_under_budget(srs_kb: KinBody) -> None:
    """16-sample sweep on synthetic SRS arm should complete well under any
    tier-2 fallback timescale.

    Standalone runs see 0.1-2s depending on machine load; under a busy
    test suite, contention can stretch this further. Tier-2 fallback
    (HP at ~120 ms x 16 lock samples = ~2 s) is acceptable; the 5s
    budget catches the "no IK at all" regression.
    """
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    t0 = time.time()
    solutions, is_ls = seven_r.solve(srs_kb, T_star, lock_samples=16)
    elapsed = time.time() - t0
    assert not is_ls
    assert len(solutions) > 0
    assert elapsed < 5.0, f"16-sample sweep took {elapsed:.2f}s (budget 5.0s)"


# ---------------------------------------------------------------------------
# Topology refusal.
# ---------------------------------------------------------------------------


def test_wrong_dof_raises() -> None:
    """6-DOF KinBody must be rejected."""
    axes = [np.array([0.0, 0.0, 1.0]) for _ in range(6)]
    t_lefts = [np.array([0.0, 0.0, 0.1 * (i + 1)]) for i in range(6)]
    links = [Link(name=f"l{i}") for i in range(7)]
    joints = []
    for i in range(6):
        T_l = np.eye(4)
        T_l[:3, 3] = t_lefts[i]
        joints.append(
            Joint(
                name=f"j{i}",
                dof_index=i,
                parent_link=links[i],
                T_left=T_l,
                T_right=np.eye(4),
                axis=axes[i],
                joint_type="revolute",
            )
        )
    six_kb = KinBody(links=links, joints=joints)
    with pytest.raises(ValueError, match="7"):
        seven_r.solve(six_kb, np.eye(4))


# ---------------------------------------------------------------------------
# Early-exit + seed-bias (#142 items 1+2).
#
# Universal speedup for non-SRS 7R arms: ``max_solutions`` short-circuits
# the lock-sweep once enough deduplicated solutions are collected;
# ``q_seed`` reorders samples by wrap-to-pi distance to the seed so
# trajectory-tracking callers find their match in the first sample(s).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_max_solutions_returns_at_most_n(srs_kb: KinBody, k: int) -> None:
    """``max_solutions=k`` returns at most ``k`` deduplicated solutions, and
    every returned solution FK-closes against the target."""
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    solutions, is_ls = seven_r.solve(srs_kb, T_star, max_solutions=k)
    assert not is_ls
    assert 1 <= len(solutions) <= k
    for sol in solutions:
        T_check = _fk(srs_kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-10), (
            f"early-exit solution fails FK: max|diff|={np.max(np.abs(T_check - T_star))}"
        )


def test_max_solutions_subset_of_full_sweep(srs_kb: KinBody) -> None:
    """Solutions returned with ``max_solutions=k`` must be a wrap-to-pi
    subset of the full-sweep result -- early exit reduces *count* but
    never returns *different* solutions."""
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    full, _ = seven_r.solve(srs_kb, T_star)
    short, _ = seven_r.solve(srs_kb, T_star, max_solutions=3)
    full_qs = [s.q for s in full]
    for sol in short:
        # Each early-exit solution must match (mod 2pi) some full-sweep
        # solution.
        match = False
        for q_full in full_qs:
            diff = ((sol.q - q_full + np.pi) % (2 * np.pi)) - np.pi
            if np.max(np.abs(diff)) < 1e-6:
                match = True
                break
        assert match, f"early-exit solution {sol.q.tolist()} not in full sweep"


def test_q_seed_returns_nearest_first(srs_kb: KinBody) -> None:
    """With ``q_seed=q*``, the nearest-ranked solution should be the one
    closest to ``q*`` (in wrap-to-pi joint distance) -- the trajectory-tracking
    promise. Since #331, jointlock hands back the first yielding lock slice's
    *full* branch set when seeded (so the caller can rank by ``seed_metric``);
    the caller's ``nearest_to_seed`` -- applied here as the orchestrator and
    Manipulator do -- delivers the nearest config."""
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    sols_seeded, _ = seven_r.solve(srs_kb, T_star, q_seed=q_star, max_solutions=1)
    assert len(sols_seeded) >= 1
    sol_seeded = nearest_to_seed(sols_seeded, q_star, metric="wrap_linf")[0]
    # Compare against the full sweep to confirm we picked the nearest one
    # to the seed (or one of them, modulo 2pi wrap on non-locked joints).
    full, _ = seven_r.solve(srs_kb, T_star)
    lock_idx = seven_r.choose_lock_joint(srs_kb)

    def _lock_dist(q: NDArray[np.float64]) -> float:
        d = float(((q[lock_idx] - q_star[lock_idx] + np.pi) % (2 * np.pi)) - np.pi)
        return abs(d)

    seeded_lock_dist = _lock_dist(sol_seeded.q)
    full_lock_dists = [_lock_dist(s.q) for s in full]
    # Seeded result's lock-joint distance must be the minimum (the seed
    # bias only ranks lock-joint values; tie-breaks on inner-solver
    # outputs are not specified, so we don't assert on the *full* q
    # vector).
    assert seeded_lock_dist <= min(full_lock_dists) + 1e-9


def test_q_seed_speedup_visits_fewer_samples(srs_kb: KinBody) -> None:
    """``q_seed`` + ``max_solutions=1`` should evaluate fewer lock samples
    than the unseeded variant, on average. Empirical bound: with the
    seed at the true q*, we should never need more than 4 samples (the
    nearest-first ordering puts the matching sample at index 0)."""
    import logging

    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    seven_r._LOG.setLevel(logging.INFO)

    # Capture the log message that reports samples-evaluated count.
    handler_records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            handler_records.append(record)

    h = _Capture()
    seven_r._LOG.addHandler(h)
    try:
        seven_r.solve(srs_kb, T_star, q_seed=q_star, max_solutions=1)
        msg = handler_records[-1].getMessage()
        # The log line is "...lock_idx=K, M/N samples -> ..."; pull M.
        # Just sanity-check that M <= 4.
        seg = msg.split(",")[1].strip()  # "M/N samples -> ..."
        m_evaluated = int(seg.split("/")[0])
        assert m_evaluated <= 4, (
            f"q_seed + max_solutions=1 evaluated {m_evaluated} samples; expected <= 4"
        )
    finally:
        seven_r._LOG.removeHandler(h)


def test_default_unchanged(srs_kb: KinBody) -> None:
    """Calling without ``max_solutions`` and ``q_seed`` should produce the
    same solution count as before -- backwards-compat guard."""
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    sols_before, _ = seven_r.solve(srs_kb, T_star)
    sols_after, _ = seven_r.solve(srs_kb, T_star, max_solutions=None, q_seed=None)
    assert len(sols_before) == len(sols_after)


def test_invalid_max_solutions_raises(srs_kb: KinBody) -> None:
    T_star = _fk(srs_kb, np.zeros(7))
    for bad in (0, -1, -10):
        with pytest.raises(ValueError, match="max_solutions"):
            seven_r.solve(srs_kb, T_star, max_solutions=bad)


def test_invalid_q_seed_shape_raises(srs_kb: KinBody) -> None:
    T_star = _fk(srs_kb, np.zeros(7))
    for bad in (np.zeros(6), np.zeros(8), np.zeros((7, 1))):
        with pytest.raises(ValueError, match="q_seed"):
            seven_r.solve(srs_kb, T_star, q_seed=bad)


def test_max_solutions_fk_bulletproof(srs_kb: KinBody) -> None:
    """100 random poses, ``max_solutions=1``: every returned solution
    must FK-close at machine precision. The early-exit path must never
    return a non-converged candidate."""
    rng = np.random.default_rng(20260430)
    failures = 0
    fk_max = 0.0
    for _ in range(100):
        q_star = rng.uniform(-1.5, 1.5, size=7)
        T_star = _fk(srs_kb, q_star)
        sols, is_ls = seven_r.solve(srs_kb, T_star, max_solutions=1)
        if is_ls or not sols:
            failures += 1
            continue
        T_check = _fk(srs_kb, sols[0].q)
        err = float(np.max(np.abs(T_check - T_star)))
        fk_max = max(fk_max, err)
        if err > 1e-9:
            failures += 1
    assert failures == 0, f"{failures}/100 random poses failed FK closure (max err {fk_max:.2e})"
    assert fk_max < 1e-9, f"max FK closure error {fk_max:.2e} above 1e-9"


# ---------------------------------------------------------------------------
# Codegen-time topology cache (#142 item 4).
# ---------------------------------------------------------------------------


def test_dispatch_cache_matches_uncached(srs_kb: KinBody) -> None:
    """When the dispatch cache is correctly pre-computed, the cached
    path must produce the same solution set as the uncached path. This
    is the safety net under the codegen-time topology cache: if the
    cached dispatch names ever drift from what ``_topology_rank`` would
    pick at runtime, this test catches it.
    """
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.solvers.jointlock.seven_r import (
        _DEFAULT_SAMPLES,
        _lock_joint,
        _topology_rank,
    )

    lock_idx = seven_r.choose_lock_joint(srs_kb)
    joint_lim = srs_kb.joints[lock_idx].limits
    if joint_lim is None:
        lo, hi = -np.pi, np.pi
    else:
        lo, hi = joint_lim
    samples = np.linspace(lo, hi, _DEFAULT_SAMPLES, endpoint=False)
    cache = []
    for q_lock in samples:
        sub_kb = _lock_joint(srs_kb, lock_idx, float(q_lock))
        _, name = _topology_rank(sub_kb, DEFAULT_TOLERANCE_POLICY)
        cache.append(name)

    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    sols_uncached, _ = seven_r.solve(srs_kb, T_star, lock_samples=samples)
    sols_cached, _ = seven_r.solve(srs_kb, T_star, lock_samples=samples, dispatch_cache=cache)

    # Same solution count.
    assert len(sols_uncached) == len(sols_cached), (
        f"cache changed solution count: uncached={len(sols_uncached)}, cached={len(sols_cached)}"
    )
    # Each cached solution matches some uncached solution at machine
    # precision (mod 2pi).
    for sc in sols_cached:
        match = False
        for su in sols_uncached:
            diff = ((sc.q - su.q + np.pi) % (2 * np.pi)) - np.pi
            if np.max(np.abs(diff)) < 1e-9:
                match = True
                break
        assert match, f"cached solution {sc.q.tolist()} not in uncached set"


def test_dispatch_cache_length_mismatch_raises(srs_kb: KinBody) -> None:
    """``dispatch_cache`` length must match ``lock_samples`` length."""
    T_star = _fk(srs_kb, np.zeros(7))
    samples = [0.0, 0.5, 1.0]  # 3 samples
    cache = ["spherical_two_parallel", "spherical_two_parallel"]  # 2 entries
    with pytest.raises(ValueError, match="dispatch_cache length"):
        seven_r.solve(srs_kb, T_star, lock_samples=samples, dispatch_cache=cache)


def test_dispatch_cache_with_q_seed_reordering(srs_kb: KinBody) -> None:
    """When ``q_seed`` reorders samples, the cache must be permuted
    alongside so each cached entry still corresponds to its sample."""
    q_star = np.array([0.3, -0.7, 0.9, 1.1, -0.5, 0.2, 0.4])
    T_star = _fk(srs_kb, q_star)
    samples = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    # All-correct cache (matches what _topology_rank would pick).
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.solvers.jointlock.seven_r import _lock_joint, _topology_rank

    lock_idx = seven_r.choose_lock_joint(srs_kb)
    cache = []
    for q_lock in samples:
        sub_kb = _lock_joint(srs_kb, lock_idx, float(q_lock))
        _, name = _topology_rank(sub_kb, DEFAULT_TOLERANCE_POLICY)
        cache.append(name)

    sols_seeded, _ = seven_r.solve(
        srs_kb,
        T_star,
        lock_samples=samples,
        dispatch_cache=cache,
        q_seed=q_star,
        max_solutions=1,
    )
    # Seeded returns the first yielding slice's full branch set (#331); every
    # branch must FK-close at machine precision -- a mis-permuted cache would
    # dispatch the wrong inner solver and yield garbage / non-closing branches.
    assert len(sols_seeded) >= 1
    for sol in sols_seeded:
        T_check = _fk(srs_kb, sol.q)
        assert np.allclose(T_check, T_star, atol=1e-10)
