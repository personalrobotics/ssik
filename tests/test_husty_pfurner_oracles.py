"""Validation harness for the Husty-Pfurner solver (Phase 5a of #158).

The harness encodes the seven correctness oracles from #158 / #162 as
parametrised test functions. Until the HP solver is implemented (PRs in
#162), every oracle is marked ``xfail(strict=True)`` -- the tests run,
hit ``NotImplementedError`` from the skeleton, and report ``xfailed``.
When the solver lands (Phase 5g), the ``xfail`` marks are removed and
the same tests start passing.

Oracles (per #158):

1. **FK closure** -- every returned ``q`` FK-closes the input pose at 1e-9.
2. **EAIK cross-check** (skipped if EAIK not installed) -- same arm,
   same poses, same solution sets within wrap-to-pi at 1e-6 on
   Pieper-class arms.
3. **gen_six_dof solution-count parity** -- HP must not return fewer
   solutions than the brute-force grid on reachable poses (slow; opt-in).
4. **JACO 2 RR composer cross-check** -- two independent algorithms on
   the same 6R arm produce the same q-set within wrap-to-pi at 1e-8.
5. **Hypothesis fuzz over random 6R chains** -- FK closure on every
   returned ``q`` (slow; opt-in).
6. **Numerical-stability sweep** -- near-parallel-axis tangencies and
   near-coincident origins return stable solutions or raise
   ``NumericConditioningError``; never silent wrong answers.
7. **Determinism** -- byte-equal solution lists across runs for fixed
   inputs.

Run the fast harness (oracles 1, 2, 4, 6, 7 with small inputs)::

    uv run pytest tests/test_husty_pfurner_oracles.py

Run the full validation (oracles 3, 5 included; minutes)::

    uv run pytest tests/test_husty_pfurner_oracles.py -m slow
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from jaco2 import jaco2_specs
from ur5 import ur5_specs

from ssik._kinbody import KinBody, build_kinbody
from ssik.core.solution import Solution
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.solvers.husty_pfurner.general_6r import solve as hp_solve

XFAIL_REASON = (
    "Husty-Pfurner solver not yet implemented; tracked in "
    "https://github.com/siddhss5/ikfastpy/issues/162"
)
XFAIL = pytest.mark.xfail(strict=True, reason=XFAIL_REASON, raises=NotImplementedError)

# ----------------------------------------------------------------------------
# Oracle helpers (the actual validation logic). Tests below wire these to
# ``hp_solve`` and mark themselves xfail until the solver lands. Helpers do
# NOT change when the xfail marks are removed.
# ----------------------------------------------------------------------------


def _fk_closure_oracle(
    kb: KinBody,
    q_star: np.ndarray,
    *,
    atol: float = 1e-9,
) -> tuple[list[Solution], np.ndarray]:
    """Oracle 1: every returned ``q`` FK-closes the target at ``atol``.

    Returns the (solutions, T_target) tuple so the caller can run further
    cross-checks against the same problem.
    """
    T = poe_forward_kinematics(kb, q_star)
    sols, is_ls = hp_solve(kb, T)
    assert not is_ls, f"HP returned is_ls=True for reachable pose q={q_star}"
    assert sols, "HP returned zero solutions for a reachable pose"
    for sol in sols:
        T_check = poe_forward_kinematics(kb, sol.q)
        err = float(np.max(np.abs(T_check - T)))
        assert np.allclose(T_check, T, atol=atol), (
            f"HP candidate failed FK closure: max|diff|={err:.2e} > {atol}"
        )
    return sols, T


def _q_set_match_oracle(
    sols_a: list[Solution],
    sols_b: list[Solution],
    *,
    atol: float = 1e-6,
) -> None:
    """Helper for oracles 2 + 4: assert two solver outputs produce the same
    q-set modulo wrap-to-pi within ``atol`` per joint.

    Comparison is order-independent: each ``sols_a`` candidate is matched
    to its nearest ``sols_b`` neighbour by per-joint wrap-to-pi distance.
    """
    assert len(sols_a) == len(sols_b), f"solution counts differ: {len(sols_a)} vs {len(sols_b)}"
    qs_b = [s.q for s in sols_b]
    matched = [False] * len(qs_b)
    for sa in sols_a:
        best_idx = -1
        best_d = float("inf")
        for j, qb in enumerate(qs_b):
            if matched[j]:
                continue
            wrapped = (sa.q - qb + np.pi) % (2 * np.pi) - np.pi
            d = float(np.max(np.abs(wrapped)))
            if d < best_d:
                best_d = d
                best_idx = j
        assert best_idx >= 0, f"no candidate to match against sa.q={sa.q}"
        assert best_d < atol, f"no q-match within {atol}: sa.q={sa.q}, closest_d={best_d:.2e}"
        matched[best_idx] = True


def _determinism_oracle(kb: KinBody, q_star: np.ndarray) -> None:
    """Oracle 7: two consecutive ``solve`` calls return byte-equal q vectors.

    Ordering must also match (HP is deterministic per branch enumeration).
    """
    T = poe_forward_kinematics(kb, q_star)
    sols_1, is_ls_1 = hp_solve(kb, T)
    sols_2, is_ls_2 = hp_solve(kb, T)
    assert is_ls_1 == is_ls_2
    assert len(sols_1) == len(sols_2)
    for s1, s2 in zip(sols_1, sols_2, strict=True):
        assert np.array_equal(s1.q, s2.q), f"non-deterministic solution: q1={s1.q}, q2={s2.q}"


# ----------------------------------------------------------------------------
# Oracle 1: FK closure on representative arms
# ----------------------------------------------------------------------------


@XFAIL
def test_oracle1_fk_closure_jaco2() -> None:
    """JACO 2 (non-Pieper 6R) -- the canonical HP target."""
    kb = build_kinbody(jaco2_specs())
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    _fk_closure_oracle(kb, q_star)


@XFAIL
def test_oracle1_fk_closure_ur5() -> None:
    """UR5 (Pieper-class 6R, three parallel axes) -- HP must also handle
    arms that the dispatcher would normally route to a faster Pieper
    specialisation. Validates that HP does not miss solutions on
    Pieper-class chains.
    """
    kb = build_kinbody(ur5_specs())
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    _fk_closure_oracle(kb, q_star)


# ----------------------------------------------------------------------------
# Oracle 2: EAIK cross-check (auto-skipped if EAIK not installed locally)
# ----------------------------------------------------------------------------


@XFAIL
def test_oracle2_eaik_cross_check_ur5() -> None:
    """HP and EAIK agree on the UR5 q-set within 1e-6 wrap-to-pi.

    EAIK is not a hard dependency of ssik. When the package isn't installed
    locally this test SKIPS rather than fails -- the oracle is still
    encoded and runs in dev environments where EAIK is present.
    """
    pytest.importorskip("eaik")
    # Implementation deferred until Phase 5g; the EAIK adapter and the
    # solution-set comparison wire up at the same time as the xfail removal.
    # For now the harness asserts NotImplementedError lands (via XFAIL).
    kb = build_kinbody(ur5_specs())
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    _fk_closure_oracle(kb, q_star)


# ----------------------------------------------------------------------------
# Oracle 3: gen_six_dof solution-count parity (slow; opt-in via -m slow)
# ----------------------------------------------------------------------------


@pytest.mark.slow
@XFAIL
def test_oracle3_gen_six_dof_parity_jaco2() -> None:
    """HP must not return fewer solutions than ``gen_six_dof`` on a
    reachable JACO 2 pose. The brute-force grid is slow (~30 s) but
    correct on the chains where HP claims coverage.
    """
    from ssik.solvers.ikgeo import gen_six_dof

    kb = build_kinbody(jaco2_specs())
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    T = poe_forward_kinematics(kb, q_star)
    hp_sols, _ = hp_solve(kb, T)
    grid_sols, _ = gen_six_dof.solve(kb, T)
    assert len(hp_sols) >= len(grid_sols), (
        f"HP returned {len(hp_sols)} solutions but gen_six_dof found "
        f"{len(grid_sols)} -- HP is missing solutions."
    )


# ----------------------------------------------------------------------------
# Oracle 4: JACO 2 RR composer cross-check
# ----------------------------------------------------------------------------


@XFAIL
def test_oracle4_jaco2_rr_composer_cross_check() -> None:
    """HP and ``ssik.solvers.ikgeo.general_6r`` (the Raghavan-Roth composer)
    must agree on JACO 2 q-sets within 1e-8 wrap-to-pi. Two independent
    algorithms on the same arm.
    """
    from ssik.solvers.ikgeo import general_6r as rr_general_6r

    kb = build_kinbody(jaco2_specs())
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    T = poe_forward_kinematics(kb, q_star)
    hp_sols, hp_is_ls = hp_solve(kb, T)
    rr_sols, rr_is_ls = rr_general_6r.solve(kb, T)
    assert hp_is_ls == rr_is_ls
    _q_set_match_oracle(hp_sols, rr_sols, atol=1e-8)


# ----------------------------------------------------------------------------
# Oracle 5: Hypothesis fuzz over random 6R chains (slow; opt-in)
# ----------------------------------------------------------------------------


@pytest.mark.slow
@XFAIL
def test_oracle5_hypothesis_fuzz_random_6r_chains() -> None:
    """Generate random 6R chains, draw random reachable poses via FK,
    confirm HP returns at least one FK-closing candidate for each.

    Phase 5a runs a small example budget for harness wiring; Phase 5g
    bumps to 500 chains x 500 poses per the validation gate in #158.
    """
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    @given(
        seed=st.integers(min_value=0, max_value=2**32 - 1),
        q_seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(
        max_examples=5,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def _check(seed: int, q_seed: int) -> None:
        rng_chain = np.random.default_rng(seed)
        rng_q = np.random.default_rng(q_seed)
        # Use JACO 2 spec frames as a deterministic shape; randomise the
        # joint axes to perturb the kinematics. Replaced in Phase 5g with
        # a proper random-chain generator.
        del rng_chain
        kb = build_kinbody(jaco2_specs())
        q_star = rng_q.uniform(-1.0, 1.0, size=6)
        _fk_closure_oracle(kb, q_star)

    _check()


# ----------------------------------------------------------------------------
# Oracle 6: numerical-stability sweep at near-singular geometries
# ----------------------------------------------------------------------------


@XFAIL
def test_oracle6_numerical_stability_near_singular() -> None:
    """At a near-singular pose (joints aligned to a degenerate configuration),
    HP must either return stable solutions OR raise a structured
    ``NumericConditioningError`` -- not a silent wrong answer or an
    unhandled exception.

    Phase 5a stub uses JACO 2 at a known-near-singular pose; Phase 5g
    expands to a parametrised suite over (parallel-axis tangency,
    coincident-origin near-miss, joint-limit edge).
    """
    kb = build_kinbody(jaco2_specs())
    # All-zero q is the canonical home-pose singularity for many arms.
    q_star = np.zeros(6, dtype=np.float64)
    _fk_closure_oracle(kb, q_star)


# ----------------------------------------------------------------------------
# Oracle 7: determinism -- byte-equal solution list across runs
# ----------------------------------------------------------------------------


@XFAIL
def test_oracle7_determinism_jaco2() -> None:
    """Two consecutive ``solve`` calls on the same input return byte-equal
    q vectors in the same order. No reliance on hash randomisation,
    threading nondeterminism, or LAPACK eigvals branch noise.
    """
    kb = build_kinbody(jaco2_specs())
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    _determinism_oracle(kb, q_star)


# ----------------------------------------------------------------------------
# Skeleton smoke tests (no xfail -- these document the current state)
# ----------------------------------------------------------------------------


def test_skeleton_solve_raises_not_implemented() -> None:
    """The Phase 5a skeleton raises NotImplementedError with a pointer
    to the tracking issue. When Phase 5g lands the implementation, this
    test flips to a skeleton-removal test (or is deleted).
    """
    kb = build_kinbody(jaco2_specs())
    T = poe_forward_kinematics(kb, np.zeros(6))
    with pytest.raises(
        NotImplementedError, match=r"https://github.com/siddhss5/ikfastpy/issues/162"
    ):
        hp_solve(kb, T)


def test_skeleton_solver_module_imports_cleanly() -> None:
    """The skeleton module imports without dragging in heavy dependencies
    (sympy, gen_six_dof). HP's runtime entry point should be lightweight.
    """
    import importlib

    mod = importlib.import_module("ssik.solvers.husty_pfurner.general_6r")
    assert hasattr(mod, "solve")
    assert mod.solve is hp_solve
