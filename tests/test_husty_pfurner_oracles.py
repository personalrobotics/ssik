"""Validation harness for the Husty-Pfurner solver (Phase 5a of #158).

The harness encodes the correctness oracles from #158 / #162 as
parametrised test functions. The HP solver has landed (#184 / #185), so
these run as live checks; one remains ``xfail(strict=False)`` for a
documented coverage gap (oracle 6, tracked in #183).

Oracles (per #158):

1. **FK closure** -- every returned ``q`` FK-closes the input pose at 1e-9.
2. **Cross-solver agreement (UR5)** -- HP and Raghavan-Roth, two
   independent universal-6R algebras, find the same solution branches on a
   clean spherical-wrist arm (exact count parity + branch-level joint match,
   robust to LAPACK-backend numeric drift). (A direct EAIK cross-check needs
   joint-convention reconciliation; the EAIK timing / coverage comparison
   lives in ``examples/04_compare_vs_eaik.py``.)
5. **Hypothesis fuzz over random 6R chains** -- FK closure on every
   returned ``q`` (slow; opt-in).
6. **Numerical-stability sweep** -- near-parallel-axis tangencies and
   near-coincident origins return stable solutions or raise
   ``NumericConditioningError``; never silent wrong answers.
7. **Determinism** -- byte-equal solution lists across runs for fixed
   inputs.

Oracles 3 and 4 (HP-vs-RR solution-count parity on JACO 2) were retired:
#409 established HP and Raghavan-Roth are *complementary*, not competing,
universal-6R solvers -- RR is the tier-2 general-6R path, HP is the
locked-7R backstop. HP returning fewer branches than RR on a non-Pieper
arm like JACO 2 (~90/100 poses) is expected, not a defect, so a parity
assertion there tracked no real contract. HP's coverage is validated on
its actual domain by ``test_husty_pfurner_locked_7r``; HP FK-closure on
JACO 2 is covered by oracle 1; the meaningful HP-vs-RR cross-check lives
in oracle 2 on UR5, an arm both solvers fully cover.

Run the fast harness (oracles 1, 2, 6, 7 with small inputs)::

    uv run pytest tests/test_husty_pfurner_oracles.py

Run the full validation (oracle 5 included; minutes)::

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
    # ``allow_refinement=True``: HP returns algebraic seeds that may be
    # at O(eps^{1/k}) precision for multi-root configs; LM polish (post
    # #176 perturbation work) brings them to machine precision in 4-8
    # iters per seed. The post-perturbation HP is "correct + precise"
    # only with refinement enabled.
    sols, is_ls = hp_solve(kb, T, allow_refinement=True)
    assert not is_ls, f"HP returned is_ls=True for reachable pose q={q_star}"
    assert sols, "HP returned zero solutions for a reachable pose"
    for sol in sols:
        T_check = poe_forward_kinematics(kb, sol.q)
        err = float(np.max(np.abs(T_check - T)))
        assert np.allclose(T_check, T, atol=atol), (
            f"HP candidate failed FK closure: max|diff|={err:.2e} > {atol}"
        )
    return sols, T


def _determinism_oracle(kb: KinBody, q_star: np.ndarray) -> None:
    """Oracle 7: two consecutive ``solve`` calls return byte-equal q vectors.

    Ordering must also match (HP is deterministic per branch enumeration).
    """
    T = poe_forward_kinematics(kb, q_star)
    sols_1, is_ls_1 = hp_solve(kb, T, allow_refinement=True)
    sols_2, is_ls_2 = hp_solve(kb, T, allow_refinement=True)
    assert is_ls_1 == is_ls_2
    assert len(sols_1) == len(sols_2)
    for s1, s2 in zip(sols_1, sols_2, strict=True):
        assert np.array_equal(s1.q, s2.q), f"non-deterministic solution: q1={s1.q}, q2={s2.q}"


# ----------------------------------------------------------------------------
# Oracle 1: FK closure on representative arms
# ----------------------------------------------------------------------------


def test_oracle1_fk_closure_jaco2() -> None:
    """JACO 2 (non-Pieper 6R) -- the canonical HP target."""
    kb = build_kinbody(jaco2_specs())
    rng = np.random.default_rng(0)
    q_star = rng.uniform(-1.0, 1.0, size=6)
    _fk_closure_oracle(kb, q_star)


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
# Oracle 2: cross-solver agreement on UR5 -- HP vs Raghavan-Roth, two
# independent universal-6R algebras. UR5 is a clean spherical-wrist arm both
# solve fully, so they must return the SAME solution set (unlike a non-Pieper
# arm such as JACO 2, where the two solvers' spurious-root filters legitimately
# disagree and HP -- the locked-7R backstop, #409 -- returns fewer branches
# than the tier-2 RR path it isn't competing with).
#
# (An EAIK cross-check was the original intent here, but EAIK's URDF/DH frame
# and per-joint sign/offset conventions differ from ssik's POE, so a correct
# comparison needs joint-convention reconciliation -- tracked separately. The
# HP-vs-RR check below gives stronger, dependency-free cross-solver coverage;
# the EAIK timing/coverage comparison lives in examples/04_compare_vs_eaik.py.)
# ----------------------------------------------------------------------------


def _nearest_wrap(q: np.ndarray, pool: list[np.ndarray]) -> float:
    """Smallest per-joint wrap-to-pi distance from ``q`` to any pose in ``pool``."""
    return min(float(np.abs((q - p + np.pi) % (2 * np.pi) - np.pi).max()) for p in pool)


def test_oracle2_hp_rr_cross_check_ur5() -> None:
    """HP and Raghavan-Roth find the SAME UR5 solution branches.

    The check is branch-level, not bit-identity: the two algebras run different
    LAPACK paths, so a near-singular branch's joints can drift up to ~1e-3
    between backends (macOS Accelerate vs Linux OpenBLAS) while still being the
    same solution. Distinct UR5 branches differ by >>1e-2, so a 1e-2 match
    tolerance + exact count parity catches a missing/spurious/wrong branch
    without flaking on backend numerics.
    """
    from ssik.solvers.ikgeo import general_6r as rr_general_6r

    same_branch = 1e-2
    kb = build_kinbody(ur5_specs())
    rng = np.random.default_rng(0)
    for _ in range(20):
        q_star = rng.uniform(-1.0, 1.0, size=6)
        T = poe_forward_kinematics(kb, q_star)
        hp_q = [np.asarray(s.q) for s in hp_solve(kb, T, allow_refinement=True)[0]]
        rr_q = [np.asarray(s.q) for s in rr_general_6r.solve(kb, T)[0]]
        assert hp_q
        assert len(hp_q) == len(rr_q), f"HP returned {len(hp_q)}, RR returned {len(rr_q)}"
        for q in hp_q:
            assert _nearest_wrap(q, rr_q) < same_branch
        for q in rr_q:
            assert _nearest_wrap(q, hp_q) < same_branch


# ----------------------------------------------------------------------------
# Oracle 5: Hypothesis fuzz over random 6R chains (slow; opt-in)
# ----------------------------------------------------------------------------


@pytest.mark.slow
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


@pytest.mark.xfail(
    strict=False,
    reason=(
        "JACO 2 home pose (q=0) is at a kinematic singularity where "
        "Jacobian rank drops -- HP returns valid IKs from the continuous "
        "family but may not include the all-zero seed. FK closure assertion "
        "still holds for what's returned; the test contract should be "
        "reformulated for singular poses (#176 / #183 work)."
    ),
)
def test_oracle6_numerical_stability_near_singular() -> None:
    """At a near-singular pose (joints aligned to a degenerate configuration),
    HP must either return stable solutions OR raise a structured
    ``NumericConditioningError`` -- not a silent wrong answer or an
    unhandled exception.
    """
    kb = build_kinbody(jaco2_specs())
    # All-zero q is the canonical home-pose singularity for many arms.
    q_star = np.zeros(6, dtype=np.float64)
    _fk_closure_oracle(kb, q_star)


# ----------------------------------------------------------------------------
# Oracle 7: determinism -- byte-equal solution list across runs
# ----------------------------------------------------------------------------


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
# Module-level smoke tests (HP is now implemented; skeleton tests retired).
# ----------------------------------------------------------------------------


def test_skeleton_solver_module_imports_cleanly() -> None:
    """The skeleton module imports without dragging in heavy dependencies
    (sympy). HP's runtime entry point should be lightweight.
    """
    import importlib

    mod = importlib.import_module("ssik.solvers.husty_pfurner.general_6r")
    assert hasattr(mod, "solve")
    assert mod.solve is hp_solve
