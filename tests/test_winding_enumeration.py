"""Default-on winding enumeration for finite wide-limit joints (#562, step 2).

A revolute joint whose limits span more than 2*pi admits several distinct
in-limit representatives of the *same* geometric branch: with limits
[-2*pi, 2*pi], -10deg and +350deg have identical FK but are different
admissible configurations, at different distances and with different motions
available from the robot's current state. ssik now returns all of them.

These are finite-limit lifts, not extra geometric branches, and the two counts
are reported separately so they are never conflated.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pytest

import ssik
from ssik._native import native_available
from ssik.core.solution import Solution
from ssik.postprocess import _windings_topk as topk
from ssik.postprocess import (
    count_windings,
    expand_windings,
    finalize_solutions,
    nearest_to_seed,
    winding_joints,
)

FIXTURES = Path(__file__).parent / "fixtures"
TWO_PI = 2.0 * np.pi


@pytest.fixture(scope="module")
def ur5() -> ssik.Manipulator:
    return ssik.Manipulator.from_urdf(FIXTURES / "ur5.urdf", base="base_link", ee="ee_link")


@pytest.fixture(scope="module")
def panda() -> ssik.Manipulator:
    return ssik.Manipulator.from_urdf(
        FIXTURES / "franka_panda.urdf", base="panda_link0", ee="panda_link8"
    )


def _sol(q: list[float]) -> Solution:
    return Solution(q=np.array(q, dtype=np.float64), fk_residual=0.0)


# ---------------------------------------------------------------------------
# Which joints are eligible at all.
# ---------------------------------------------------------------------------


def test_winding_joints_selects_only_wide_finite_revolute(ur5: ssik.Manipulator) -> None:
    """Eligibility needs a tolerance, not a bare ``span > 2*pi``.

    This very fixture proves it: the UR5 URDF writes joint 2's limits as
    +/-3.14159265359, which is 4e-10 *wider* than a half turn. A bare
    comparison would call it a wide joint and lift it into a second
    representative that exists only at the exact boundary and is the same
    physical configuration.
    """
    wide = {i for i, _, _ in winding_joints(ur5.kinbody)}
    assert wide == {0, 1, 3, 4, 5}
    spans = [j.limits[1] - j.limits[0] for j in ur5.kinbody.joints if j.limits]
    assert spans[2] > TWO_PI, "fixture no longer exercises the round-off case"
    assert spans[2] - TWO_PI < 1e-9


def test_narrow_and_continuous_joints_are_never_enumerated(panda: ssik.Manipulator) -> None:
    # Panda's joints are all narrower than a full turn -> nothing to lift.
    assert winding_joints(panda.kinbody) == []
    T = panda.fk(np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7]))
    assert len(panda.solve(T)) == len(panda.solve(T, enumerate_windings=False))


def test_continuous_joints_do_not_change_the_count() -> None:
    """A continuous joint has an infinite lift family and no privileged member,
    so it must not be enumerated (its seed handling is the nearest turn)."""
    import importlib

    kb = importlib.import_module("ssik.prebuilt.kinova.jaco2_ik")._KB
    continuous = [i for i, j in enumerate(kb.joints) if j.limits is None]
    assert continuous, "fixture should have continuous joints"
    assert winding_joints(kb) == []

    q = np.array([1.0, 2.5, 1.0, 1.0, 1.0, 1.0])
    raw = [_sol(list(q)), _sol(list(q + 0.4))]
    assert len(expand_windings(raw, kb)) == len(raw)

    # A seed two turns away still selects the nearest turn, from a family that
    # was never materialised.
    seed = q.copy()
    seed[continuous[0]] += 2 * TWO_PI
    (near, _) = finalize_solutions(raw, kb, q_seed=seed)
    assert float(near.q[continuous[0]]) == pytest.approx(float(seed[continuous[0]]), abs=1e-9)


# ---------------------------------------------------------------------------
# The representatives themselves.
# ---------------------------------------------------------------------------


def test_expansion_derives_representatives_from_the_limits(ur5: ssik.Manipulator) -> None:
    """A principal value of 0 under [-2*pi, 2*pi] admits {-2*pi, 0, +2*pi}: the
    count follows from the limits, not from a fixed multiplier."""
    kb = ur5.kinbody
    j0 = winding_joints(kb)[0][0]
    q = np.zeros(len(kb.joints))
    out = expand_windings([_sol(list(q))], kb)
    vals = sorted({round(float(s.q[j0]), 9) for s in out})
    assert vals == [round(-TWO_PI, 9), 0.0, round(TWO_PI, 9)]


def test_every_representative_is_in_limits_and_fk_identical(ur5: ssik.Manipulator) -> None:
    q = np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2])
    T = ur5.fk(q)
    sols = ur5.solve(T)
    assert len(sols) == 256, "8 geometric branches x 2^5 wide joints"
    for s in sols:
        for i, joint in enumerate(ur5.kinbody.joints):
            assert joint.limits is not None
            assert joint.limits[0] <= s.q[i] <= joint.limits[1]
        assert np.linalg.norm(ur5.fk(s.q) - T) < 1e-9, "a lift must not move the end effector"


def test_enumerate_windings_false_returns_one_per_geometric_branch(ur5: ssik.Manipulator) -> None:
    T = ur5.fk(np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2]))
    plain = ur5.solve(T, enumerate_windings=False)
    assert len(plain) == 8
    # Every plain solution appears in the enumerated set (as some winding).
    full = ur5.solve(T)
    for p in plain:
        assert any(np.allclose(p.q, f.q, atol=1e-12) for f in full)


def test_lifts_are_distinct_configurations_not_duplicates(ur5: ssik.Manipulator) -> None:
    """Turn-aware behaviour: representatives differing by 2*pi on a wide joint
    must both survive, where a modulo-2*pi dedup would collapse them."""
    T = ur5.fk(np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2]))
    sols = ur5.solve(T)
    exact = {tuple(np.round(s.q, 9)) for s in sols}
    assert len(exact) == len(sols), "no exact duplicates"
    collapsed = {tuple(np.round(np.mod(s.q, TWO_PI), 9)) for s in sols}
    assert len(collapsed) < len(sols), "a modulo-2pi view would have merged these"


# ---------------------------------------------------------------------------
# Seed interaction: ordering and truncation.
# ---------------------------------------------------------------------------


def test_seed_prefers_its_own_winding_over_the_equivalent_one(ur5: ssik.Manipulator) -> None:
    """A seed at +350deg must rank the +350deg representative ahead of the
    equivalent -10deg one, and return it under max_solutions=1."""
    kb = ur5.kinbody
    j0 = winding_joints(kb)[0][0]
    q = np.zeros(len(kb.joints))
    q[j0] = np.deg2rad(-10.0)
    T = ur5.fk(q)

    seed = np.zeros(len(kb.joints))
    seed[j0] = np.deg2rad(350.0)
    ranked = ur5.solve(T, q_seed=seed)
    near = np.deg2rad(350.0)
    far = np.deg2rad(-10.0)
    first = float(ranked[0].q[j0])
    assert abs(first - near) < 1e-9, f"seed's own winding should come first, got {first}"

    (only,) = ur5.solve(T, q_seed=seed, max_solutions=1)
    assert abs(float(only.q[j0]) - near) < 1e-9
    assert abs(float(only.q[j0]) - far) > np.pi, "must not command a full turn back"


def test_finite_joint_ranking_uses_ordinary_not_modular_distance(ur5: ssik.Manipulator) -> None:
    """Ranking a finite joint modulo 2*pi makes distinct windings tie and
    understates real motion: a joint that must travel ~2*pi is not 'close'."""
    kb = ur5.kinbody
    j0 = winding_joints(kb)[0][0]
    seed = np.zeros(len(kb.joints))
    a, b = np.zeros(len(kb.joints)), np.zeros(len(kb.joints))
    a[j0] = 0.1
    b[j0] = 0.1 - TWO_PI  # same FK, a full turn away
    ranked = nearest_to_seed([_sol(list(b)), _sol(list(a))], seed, metric="wrap_linf", kb=kb)
    assert ranked[0].q[j0] == pytest.approx(0.1), "the genuinely nearer winding must win"


@pytest.mark.parametrize("metric", ["wrap_linf", "wrap_l2"])
@pytest.mark.parametrize("k", [1, 2, 3, 8, 25, 300])
def test_topk_equals_expand_rank_truncate(ur5: ssik.Manipulator, metric: str, k: int) -> None:
    """The pruned top-k must be *identical* to enumerating the complete set,
    ranking it and truncating -- that equivalence is what lets the capped paths
    skip building what they would discard."""
    kb = ur5.kinbody
    rng = np.random.default_rng(7)
    for _ in range(6):
        q = rng.uniform(-2.0, 2.0, 6)
        raw = ur5.solve(ur5.fk(q), enumerate_windings=False)
        for seed in (rng.uniform(-6.0, 6.0, 6), q + rng.uniform(-0.05, 0.05, 6)):
            full = nearest_to_seed(expand_windings(raw, kb), seed, metric=metric, kb=kb)[:k]
            pruned = topk(raw, kb, seed, metric, k)
            assert len(pruned) == len(full)
            for got, want in zip(pruned, full, strict=True):
                assert np.array_equal(got.q, want.q)


def test_capped_solve_matches_complete_enumeration(ur5: ssik.Manipulator) -> None:
    """End to end: max_solutions returns the same prefix the complete set would."""
    q = np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2])
    T = ur5.fk(q)
    seed = q + 0.02
    full = ur5.solve(T, q_seed=seed)
    for k in (1, 4, 17):
        capped = ur5.solve(T, q_seed=seed, max_solutions=k)
        assert len(capped) == k
        for got, want in zip(capped, full[:k], strict=True):
            assert np.array_equal(got.q, want.q)


def test_unseeded_cap_is_a_prefix_of_the_full_set(ur5: ssik.Manipulator) -> None:
    T = ur5.fk(np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2]))
    full = ur5.solve(T)
    for k in (1, 5, 99):
        capped = ur5.solve(T, max_solutions=k)
        assert [s.q.tolist() for s in capped] == [s.q.tolist() for s in full[:k]]


# ---------------------------------------------------------------------------
# Diagnostics, determinism, and the respect_limits contract.
# ---------------------------------------------------------------------------


def test_diagnostics_separate_branches_from_lifts(ur5: ssik.Manipulator) -> None:
    T = ur5.fk(np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2]))
    _, diag = ur5.solve(T, explain=True)
    assert diag.geometric_branches == 8
    assert diag.winding_representatives == 256
    assert "8 geometric branches -> 256 in-limit configurations" in diag.summary()


def test_diagnostics_stay_truthful_under_a_cap(ur5: ssik.Manipulator) -> None:
    """The complete-set size is reported even though the cap meant it was never
    built, so the cap's own count cannot silently under-report."""
    T = ur5.fk(np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2]))
    sols, diag = ur5.solve(T, max_solutions=3, explain=True)
    assert len(sols) == 3
    assert diag.winding_representatives == 256
    assert diag.dropped_by_max_solutions == 253


def test_count_windings_matches_the_expansion(ur5: ssik.Manipulator) -> None:
    rng = np.random.default_rng(3)
    for _ in range(5):
        raw = ur5.solve(ur5.fk(rng.uniform(-2.0, 2.0, 6)), enumerate_windings=False)
        assert count_windings(raw, ur5.kinbody) == len(expand_windings(raw, ur5.kinbody))


def test_ordering_is_independent_of_candidate_order(ur5: ssik.Manipulator) -> None:
    """Both backends must agree on the order, and they do not generate
    candidates in the same sequence, so the order may not depend on it."""
    kb = ur5.kinbody
    q = np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2])
    raw = ur5.solve(ur5.fk(q), enumerate_windings=False)
    seed = q + 0.02
    forward = finalize_solutions(raw, kb, q_seed=seed, max_solutions=10)
    backward = finalize_solutions(list(reversed(raw)), kb, q_seed=seed, max_solutions=10)
    assert [s.q.tolist() for s in forward] == [s.q.tolist() for s in backward]


def test_respect_limits_false_does_not_clamp_a_seeded_solve(panda: ssik.Manipulator) -> None:
    """Regression (found while building #562 step 2, shipped broken in v5.2.0):
    with respect_limits=False the caller wants the raw geometric set, so limits
    must play no part in postprocessing either. Rewrapping a solution *into*
    limits there returned a representative a full turn from the seed -- exactly
    the gratuitous motion seed-relative rewrapping exists to prevent."""
    q = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.7])
    T = panda.fk(q)
    seed = np.asarray(panda.solve(T, respect_limits=False)[0].q)
    sols = panda.solve(T, q_seed=seed, respect_limits=False)
    assert sols
    assert float(np.max(np.abs(np.asarray(sols[0].q) - seed))) < 1e-9, (
        "the seed's own configuration must come back unchanged, not a turn away"
    )


def test_enumeration_requires_respect_limits(ur5: ssik.Manipulator) -> None:
    """The set being enumerated is defined by the limits, so there is nothing to
    enumerate without them."""
    T = ur5.fk(np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2]))
    assert len(ur5.solve(T, respect_limits=False)) == 8


def test_expansion_order_is_the_ascending_cartesian_product(ur5: ssik.Manipulator) -> None:
    """The documented, stable order the ranking and truncation rules are
    defined against."""
    kb = ur5.kinbody
    wind = winding_joints(kb)
    raw = ur5.solve(ur5.fk(np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2])), enumerate_windings=False)
    got = expand_windings(raw, kb)
    want = []
    for sol in raw:
        opts = [
            sorted(v for k in range(-3, 4) if lo <= (v := float(sol.q[i]) + TWO_PI * k) <= hi)
            for i, lo, hi in wind
        ]
        for combo in itertools.product(*opts):
            q = sol.q.copy()
            for (i, _, _), v in zip(wind, combo, strict=True):
                q[i] = v
            want.append(q)
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        assert np.array_equal(a.q, b)


# ---------------------------------------------------------------------------
# Native parity. The C++ finalize mirrors the Python pipeline, so both backends
# must return the same configurations in the same order (#562's parity clause).
# ---------------------------------------------------------------------------

# One arm per native solver family whose candidate set is exact on both
# backends. Redundant / approximate 7R arms (srs_polished, spherical_shoulder,
# jointlock, and RR general_6r on some poses) sample a continuum and already
# differ between backends before #562 -- that is the relative-completeness
# contract from #487/#554, not a winding question -- so they are covered by
# soundness below instead of set equality.
_EXACT_NATIVE_ARMS = [
    "universal_robots.ur5e_ik",  # three_parallel, 5 wide joints -> x32
    "universal_robots.ur3e_ik",  # joint 2 limits exactly [-pi, pi]: both ends of the cut
    "fanuc.m710ic_ik",  # spherical_two_parallel
    "ufactory.xarm6_ik",  # general_6r (RR)
    "abb.irb1600_ik",  # wide joint with a 3-representative span
    "standard_bots.thor_ik",  # 4 wide joints
    "rokae.xmatepro7_ik",  # srs
    "franka.panda_ik",  # spherical_shoulder, no wide joint
    "kinova.jaco2_ik",  # continuous joints, never enumerated
]

# Poses with a joint exactly on the +-pi cut (#596), where each solver picks a
# side ad hoc and the backends used to return representatives 2*pi apart.
_CUT_POSES = {
    # Home-like pose: theta_offset[0] = pi puts the DH cut at q0 = 0, which came
    # back as q0 = -2*pi from both backends, and other continuous joints flipped.
    "kinova.jaco2_ik": [[0.0, 4.207047497, 3.386762076, 2.733640661, 1.98456641, -3.12438615]],
    # Elbow folded exactly to pi on a [-pi, pi] joint: +pi on one backend and -pi
    # on the other, which also reordered the seeded ranking.
    "universal_robots.ur3e_ik": [[1.2, -2.0, np.pi, -1.3, 1.1, -0.6]],
}
# At the fold the elbow is a double root, so the backends agree on the other
# joints only to ~1e-6 there; a 2*pi flip is still six orders beyond this.
_CUT_TOL = 1e-5


def _module(arm: str):
    import importlib

    return importlib.import_module(f"ssik.prebuilt.{arm}")


def _continuous(kb) -> np.ndarray:
    return np.array([j.joint_type == "revolute" and j.limits is None for j in kb.joints])


def _same_config(kb, a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """The parity contract: a continuous joint is compared on the circle (its
    coordinate is a representative, not a configuration), every other joint by
    its coordinate, since a finite joint's +-pi really are 2*pi apart."""
    d = np.abs(np.asarray(a) - np.asarray(b))
    circ = np.abs((d + np.pi) % TWO_PI - np.pi)
    return float(np.max(np.where(_continuous(kb), circ, d))) <= tol


def _same_set(kb, A: list[Solution], B: list[Solution], tol: float) -> bool:
    if len(A) != len(B):
        return False
    left = list(B)
    for a in A:
        k = next((k for k, b in enumerate(left) if _same_config(kb, a.q, b.q, tol)), None)
        if k is None:
            return False
        left.pop(k)
    return True


@pytest.mark.skipif(not native_available(), reason="native extension not built")
@pytest.mark.parametrize("arm", _EXACT_NATIVE_ARMS)
def test_native_enumeration_matches_python(arm: str) -> None:
    m = _module(arm)
    kb = m._KB
    cont = _continuous(kb)
    rng = np.random.default_rng(0)
    ranges = [j.limits if j.limits else (-np.pi, np.pi) for j in kb.joints]
    poses = [(np.array([rng.uniform(lo, hi) for lo, hi in ranges]), 1e-6, False) for _ in range(6)]
    poses += [(np.array(q), _CUT_TOL, True) for q in _CUT_POSES.get(arm, [])]
    for q, tol, cut in poses:
        T = m.fk(q)
        nat, pyth = m.solve(T, native=True), m.solve(T, native=False)
        assert len(nat) == len(pyth), f"{arm}: {len(nat)} native vs {len(pyth)} python"
        assert _same_set(kb, nat, pyth, tol), f"{arm}: different lifts at q={q.tolist()}"
        # One representative for a continuous joint: (-pi, pi], never -2*pi.
        for s in nat + pyth:
            assert np.all((s.q[cont] > -np.pi) & (s.q[cont] <= np.pi)), f"{arm}: {s.q}"

        # Ranked order must agree too, not just the set -- from the seed itself
        # and, for a finite joint on the cut, from the far end of its range.
        seeds = [q]
        if cut and np.any(np.isclose(q, np.pi)):
            seeds.append(np.where(np.isclose(q, np.pi), -np.pi, q))
        for seed in seeds:
            a = m.solve(T, q_seed=seed, max_solutions=12, native=True)
            b = m.solve(T, q_seed=seed, max_solutions=12, native=False)
            assert len(a) == len(b)
            for i, (u, v) in enumerate(zip(a, b, strict=True)):
                assert _same_config(kb, u.q, v.q, tol), f"{arm}: rank {i} differs, seed {seed}"
            if cut:
                # With a seed the nearest solution is stable: the posed
                # configuration, in the representative at the seed's end of a
                # [-pi, pi] joint (enumeration off: that span is not lifted).
                for native in (True, False):
                    top = m.solve(
                        T, q_seed=seed, max_solutions=1, native=native, enumerate_windings=False
                    )
                    assert _same_config(kb, top[0].q, seed, tol), f"{arm}: nearest {top[0].q}"


# Poses with one joint exactly at a limit (#624): (q, that joint). The backends
# compute the angle an ulp or two apart, and a strict limit test dropped the
# branch on whichever side landed outside: on macOS, native at the IRB 6700's
# joint 2 upper limit, and Python at the GP8's joint 1 lower limit. The IRB
# 6700's joint 5 spans [-2*pi, 2*pi], so its lifts are compared too.
_LIMIT_POSES = {
    "abb.irb6700_ik": ([1.9, -0.1, 1.2217304763960306, -4.9, 1.2, 0.5], 2),
    "yaskawa.gp8_ik": ([-1.8, -1.1344640137963142, 2.2, -1.5, -0.1, 6.0], 1),
}


@pytest.mark.skipif(not native_available(), reason="native extension not built")
@pytest.mark.parametrize("arm", sorted(_LIMIT_POSES))
def test_both_backends_keep_a_configuration_exactly_at_a_limit(arm: str) -> None:
    """A robot parked at a hard stop gets its configuration back on both
    backends, on the limit exactly, with the same lifts."""
    m = _module(arm)
    kb = m._KB
    q, j = np.array(_LIMIT_POSES[arm][0]), _LIMIT_POSES[arm][1]
    assert q[j] in kb.joints[j].limits, "fixture no longer sits exactly on a limit"
    T = m.fk(q)
    nat, pyth = m.solve(T, native=True), m.solve(T, native=False)
    assert _same_set(kb, nat, pyth, 1e-6), f"{arm}: {len(nat)} native vs {len(pyth)} python"
    for backend, sols in (("native", nat), ("python", pyth)):
        for s in sols:
            for i, joint in enumerate(kb.joints):
                assert joint.limits[0] <= s.q[i] <= joint.limits[1], f"{backend}: {s.q}"
        posed = [s for s in sols if _same_config(kb, s.q, q, 1e-6)]
        assert posed, f"{backend} lost the posed configuration"
        assert all(s.q[j] == q[j] for s in posed), f"{backend}: {[s.q[j] for s in posed]}"


# Poses where finalize moves a value after the solver measured fk_residual
# (#645): the +-pi snap onto a limit (#601), the limit clamp (#635), and a
# winding representative clamped onto a limit. Puma 560: joint 0 1e-7 inside
# its limit, a few ulps inside pi, is snapped onto the limit, which moves the
# tool by 1.6e-7 while the solver's residual said 8e-16. UR5e: joints 0 and 5
# 5e-10 off 0, whose -2*pi / +2*pi lifts land within the band of a limit.
_MOVED_POSES = {
    "unimation.puma560_ik": [
        [
            3.14159255358979,
            1.018871849146357,
            -2.375096920883292,
            2.370593504371684,
            -1.7105133287808816,
            3.14159265358879,
        ]
    ],
    "universal_robots.ur5e_ik": [[5e-10, -0.7, 0.9, -0.4, 0.8, -5e-10]],
    "universal_robots.ur3e_ik": _CUT_POSES["universal_robots.ur3e_ik"],
    "kinova.jaco2_ik": _CUT_POSES["kinova.jaco2_ik"],
    **{arm: [q] for arm, (q, _) in _LIMIT_POSES.items()},
}


@pytest.mark.skipif(not native_available(), reason="native extension not built")
@pytest.mark.parametrize("arm", sorted(_MOVED_POSES))
def test_fk_residual_describes_the_returned_configuration(arm: str) -> None:
    """Every returned ``fk_residual`` is ``||FK(q) - T||_F`` at the ``q``
    returned, on both backends and in every mode, including after a snap or a
    clamp moved ``q`` off the value the solver measured (#645). Measured
    agreement is within 5e-15; 1e-12 is round-off, far below the 1e-7 the
    stale residual missed by."""
    m = _module(arm)
    for pose in _MOVED_POSES[arm]:
        q = np.array(pose)
        T = m.fk(q)
        modes: list[dict[str, object]] = [
            {},
            {"respect_limits": "wrap"},
            {"respect_limits": False},
            {"q_seed": q, "max_solutions": 1},
            {"q_seed": q, "max_solutions": 12},
        ]
        for kw in modes:
            for native in (True, False):
                sols = m.solve(T, native=native, **kw)
                assert sols, f"{arm}: no solution, native={native}, {kw}"
                for s in sols:
                    actual = float(np.linalg.norm(m.fk(s.q) - T))
                    assert abs(s.fk_residual - actual) <= 1e-12, (
                        f"{arm} native={native} {kw}: reported {s.fk_residual:.1e}, "
                        f"actual {actual:.1e} at {s.q.tolist()}"
                    )


@pytest.mark.skipif(not native_available(), reason="native extension not built")
@pytest.mark.parametrize("arm", ["universal_robots.ur5e_ik", "standard_bots.thor_ik"])
def test_native_lifts_are_sound(arm: str) -> None:
    """Every configuration the native path returns is in limits and reproduces
    the target pose: a lift may never move the end effector."""
    m = _module(arm)
    kb = m._KB
    rng = np.random.default_rng(5)
    ranges = [j.limits for j in kb.joints]
    for _ in range(4):
        q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
        T = m.fk(q)
        for s in m.solve(T, native=True):
            for i, joint in enumerate(kb.joints):
                assert joint.limits[0] <= s.q[i] <= joint.limits[1]
            assert np.linalg.norm(m.fk(s.q) - T) < 1e-6


@pytest.mark.skipif(not native_available(), reason="native extension not built")
def test_native_enumerate_windings_false_restores_pre_60_counts() -> None:
    m = _module("universal_robots.ur5e_ik")
    q = np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2])
    T = m.fk(q)
    for native in (True, False):
        assert len(m.solve(T, native=native, enumerate_windings=False)) == 8
        assert len(m.solve(T, native=native)) == 256


@pytest.mark.skipif(not native_available(), reason="native extension not built")
def test_native_does_not_double_expand() -> None:
    """A solve runs the finalize pipeline several times (limit pass, rescue
    pass, ranking pass). Lifting in more than one of them silently multiplies
    the result set, which is why only the final pass enumerates."""
    m = _module("universal_robots.ur5e_ik")
    q = np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2])
    T = m.fk(q)
    for native in (True, False):
        sols = m.solve(T, native=native)
        assert len(sols) == 256, f"native={native}: {len(sols)} (256 expected, 8192 = double)"
        assert len({tuple(np.round(s.q, 9)) for s in sols}) == 256, "duplicates present"


@pytest.mark.skipif(not native_available(), reason="native extension not built")
@pytest.mark.parametrize("arm", ["universal_robots.ur5e_ik", "standard_bots.thor_ik"])
def test_flag_matrix_agrees_across_backends(arm: str) -> None:
    """Both flags, both backends, all four combinations.

    The native path reaches finalize two different ways -- some entry points run
    it once themselves, others go through an artifact solver whose final pass
    runs with respect_limits=false by then -- and an early version honoured
    enumerate_windings but not respect_limits on the first of those, so
    `solve(T, respect_limits=False)` lifted when it should not have. The
    benchmark harness calls exactly that, which is how it surfaced.
    """
    m = _module(arm)
    q = np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2])[: len(m._KB.joints)]
    T = m.fk(q)
    for respect_limits in (True, False):
        for enumerate_windings in (True, False):
            counts = {
                native: len(
                    m.solve(
                        T,
                        native=native,
                        respect_limits=respect_limits,
                        enumerate_windings=enumerate_windings,
                    )
                )
                for native in (True, False)
            }
            assert counts[True] == counts[False], (
                f"{arm}: respect_limits={respect_limits} "
                f"enumerate_windings={enumerate_windings} -> {counts}"
            )
            if not (respect_limits and enumerate_windings):
                # Only the both-on corner lifts.
                assert counts[True] == len(
                    m.solve(T, native=True, respect_limits=respect_limits, enumerate_windings=False)
                )


@pytest.mark.perf
@pytest.mark.skipif(not native_available(), reason="native extension not built")
def test_capped_solve_does_not_pay_for_discarded_lifts() -> None:
    """Enumeration costs per configuration *returned* -- that is inherent, and a
    UR returning 256 instead of 8 is the point of #562. What must not happen is
    a capped solve paying for the configurations it throws away: the tracking
    idiom asks for one solution and must not build the other 255.
    """
    m = _module("universal_robots.ur5e_ik")
    q = np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2])
    T = m.fk(q)

    def best(call, warmup: int = 20, runs: int = 120) -> float:
        for _ in range(warmup):
            call()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            call()
            times.append(time.perf_counter() - t0)
        return min(times)

    plain = best(lambda: m.solve(T, enumerate_windings=False))
    full = best(lambda: m.solve(T))
    tracked = best(lambda: m.solve(T, q_seed=q, max_solutions=1))

    assert len(m.solve(T)) == 256
    assert full > 2.0 * plain, "expected lifting 8 branches to 256 to cost something"
    # The cap keeps it at the un-enumerated cost, not the 256-configuration one.
    assert tracked < plain + 0.35 * (full - plain), (
        f"seeded max_solutions=1 pays for discarded lifts: tracked={tracked * 1e6:.0f}us "
        f"plain={plain * 1e6:.0f}us full={full * 1e6:.0f}us"
    )


# ---------------------------------------------------------------------------
# Plumbing guards. The flag has to reach three places in every emitted artifact,
# and a miss is silent: the arm keeps returning *valid* IK, just the wrong set.
# One family's native hook was in fact missed, and only a 15x count difference
# against a clean pre-6.0 build revealed it.
# ---------------------------------------------------------------------------


def _artifact_sources() -> list[tuple[str, str]]:
    root = Path("src/ssik/prebuilt")
    return [
        (str(p.relative_to(root)), p.read_text())
        for p in sorted(root.rglob("*_ik.py"))
        if not p.name.startswith("_")
    ]


def test_every_artifact_plumbs_the_flag() -> None:
    """Signature, native hook, and the single lifting stage -- in every arm."""
    sources = _artifact_sources()
    assert len(sources) >= 70, f"expected the full prebuilt set, found {len(sources)}"
    for name, src in sources:
        assert "enumerate_windings: bool = True," in src, f"{name}: solve() lacks the kwarg"
        assert "enumerate_windings=enumerate_windings and respect_limits," in src, (
            f"{name}: no lifting stage, or it does not honour respect_limits"
        )
        if "_try_native" in src:
            assert "enumerate_windings=enumerate_windings,\n" in src, (
                f"{name}: native hook does not forward the flag -- the native path "
                f"would ignore enumerate_windings and silently return a different set"
            )


@pytest.mark.skipif(not native_available(), reason="native extension not built")
@pytest.mark.parametrize(
    "arm",
    [
        "universal_robots.ur5e_ik",  # three_parallel
        "fanuc.m710ic_ik",  # spherical_two_parallel
        "ufactory.xarm6_ik",  # general_6r (RR)
        "abb.yumi_left_ik",  # srs_polished
        "ufactory.xarm7_ik",  # spherical_shoulder_polished
        "kassow.kr810_ik",  # jointlock
    ],
)
def test_native_actually_responds_to_the_flag(arm: str) -> None:
    """On an arm that has lifts, the native path must return fewer solutions with
    enumeration off. A hook that drops the flag returns the lifted set either
    way, which is easy to miss because every solution is still a valid IK."""
    m = _module(arm)
    kb = m._KB
    assert winding_joints(kb), f"{arm} has no wide-limit joint; pick another arm"
    rng = np.random.default_rng(2)
    ranges = [j.limits if j.limits else (-np.pi, np.pi) for j in kb.joints]
    q = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
    T = m.fk(q)
    lifted = len(m.solve(T, native=True))
    plain = len(m.solve(T, native=True, enumerate_windings=False))
    assert lifted > plain, f"{arm}: native ignored enumerate_windings ({lifted} either way)"
    # The Python path must respond the same way. Only the ratio is compared:
    # redundant 7R arms sample the self-motion manifold differently on the two
    # backends (the #487/#554 relative-completeness contract), so their absolute
    # counts legitimately differ while the lift multiplier must not.
    py_lifted = len(m.solve(T, native=False))
    py_plain = len(m.solve(T, native=False, enumerate_windings=False))
    assert py_lifted > py_plain, f"{arm}: python ignored enumerate_windings"
    assert lifted / plain == pytest.approx(py_lifted / py_plain, rel=0.5), (
        f"{arm}: backends disagree on the lift multiplier "
        f"(native {lifted}/{plain}, python {py_lifted}/{py_plain})"
    )
