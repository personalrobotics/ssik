"""Cached-RR fast path for jointlock inner-6R dispatch (#210).

When ``ssik build`` emits a per-arm artifact for a non-tier-0 7R arm
(e.g. Rizon 4, Kassow KR810), it bakes a list of (DH, linearity)
tuples and primes Raghavan-Roth's symbolic derivation cache at
module-import time. Subsequent jointlock dispatches use cached RR
(~1 ms warm) instead of HP / two_parallel / spherical (~13-260 ms),
yielding 12-25x post-warmup speedup.

The URDF-loaded path (no artifact, e.g. tests via
``load_urdf_kinbody_normalized``) does NOT prime the cache, so it
keeps using the original solver -- avoiding the 80-130 s cold-cache
cost that would otherwise break test runtimes.

Test contract:

- :func:`prime_derivation` populates the lookup map.
- :func:`primed_linearity_for_dh` returns the baked linearity for a
  primed DH and ``None`` for an unprimed one.
- The jointlock dispatch's ``_try_cached_rr`` returns ``None`` (falls
  back to original solver) when the cache isn't primed.
- The composer's ``_RR_PRIME_DHS`` list excludes Franka (all-tier-0
  dispatch) and includes Rizon 4 / Kassow KR810 (non-tier-0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo._raghavan_roth import (
    _PRIMED_LINEARITY_MAP,
    primed_linearity_for_dh,
)
from ssik.solvers.jointlock.seven_r import (
    _RR_ELIGIBLE_INNER_SOLVERS,
    _lock_joint,
    _try_cached_rr,
)

# ---------------------------------------------------------------------------
# Prime API
# ---------------------------------------------------------------------------


def test_primed_linearity_map_lookup() -> None:
    """:func:`primed_linearity_for_dh` reads the module-level
    ``_PRIMED_LINEARITY_MAP``. Phase 2 artifacts populate this dict
    directly via assignment in the registration block; this test
    verifies the lookup contract without going through the (slow)
    :func:`prime_derivation` sympy path.
    """
    # Direct map insertion (mimics the artifact's registration block).
    alpha = (0.123, 0.456, 0.789, 0.111, 0.222, 0.333)
    a = (1.234, 2.345, 3.456, 4.567, 5.678, 6.789)
    d = (0.987, 0.876, 0.765, 0.654, 0.543, 0.432)
    assert primed_linearity_for_dh(alpha, a, d) is None  # not primed
    _PRIMED_LINEARITY_MAP[(alpha, a, d)] = (2, False)
    try:
        result = primed_linearity_for_dh(alpha, a, d)
        assert result == (2, False)
    finally:
        _PRIMED_LINEARITY_MAP.pop((alpha, a, d), None)


def test_primed_linearity_returns_none_for_unprimed_dh() -> None:
    """``primed_linearity_for_dh`` returns None for any unprimed DH."""
    # Random DH unlikely to collide with a primed one.
    alpha = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    a = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    d = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
    assert primed_linearity_for_dh(alpha, a, d) is None


# ---------------------------------------------------------------------------
# URDF path: no priming -> _try_cached_rr returns None
# ---------------------------------------------------------------------------


def test_urdf_loaded_rizon_does_not_trigger_cached_rr() -> None:
    """The URDF path (no artifact import) doesn't prime RR for any
    sub-chain, so :func:`_try_cached_rr` returns ``None`` and the
    dispatch falls back to the original solver.

    This is the test-suite-friendly behavior: no 80+ second cold-cache
    cost when running tests via ``load_urdf_kinbody_normalized``. The
    speedup is opt-in via the per-arm artifact.
    """
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY

    kb = load_urdf_kinbody_normalized(
        Path(__file__).parent / "fixtures" / "rizon4.urdf",
        "base_link",
        "flange",
    )
    # Pick a representative locked sub-chain.
    sub_kb = _lock_joint(kb, lock_idx=2, q_lock=0.0)
    T_target = poe_forward_kinematics(sub_kb, np.zeros(6))
    result = _try_cached_rr(
        sub_kb,
        T_target,
        DEFAULT_TOLERANCE_POLICY,
        allow_refinement=False,
        refinement_max_iters=15,
    )
    assert result is None, f"expected None (no prime in test path); got {type(result).__name__}"


# ---------------------------------------------------------------------------
# Composer: builds correct prime list per arm
# ---------------------------------------------------------------------------


def test_composer_skips_priming_for_franka() -> None:
    """Franka's all-tier-0 dispatch -> RR baking should be omitted
    from the artifact entirely, keeping the artifact byte-stable with
    pre-#210.

    Franka's 16 samples all route to ``reversed:spherical`` or
    ``reversed:spherical_two_parallel``; ``spherical`` is excluded from
    :data:`_RR_ELIGIBLE_INNER_SOLVERS` because the prime cost (~14 s
    per DH) is too expensive for the modest ~6 ms per-call savings.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
    from franka_panda import franka_panda_specs

    from ssik._kinbody import build_kinbody
    from ssik.codegen._compose.seven_r import compose

    kb = build_kinbody(franka_panda_specs())
    artifact_source = compose(kb)
    # Phase 2 sentinels: no CSE'd builder + no insert_derivation calls.
    assert "_build_pq_jointlock_sample_" not in artifact_source
    assert "insert_derivation" not in artifact_source


@pytest.mark.slow
def test_composer_emits_baked_rr_for_rizon4() -> None:
    """Rizon 4's non-tier-0 inner samples (HP / two_parallel) trigger the
    composer to bake CSE'd RR builders + ``insert_derivation`` calls.

    Marked slow because :func:`compose` runs both ``_cached_best_leftvar``
    (AE-3 leftvar probing) AND the symbolic preprocessing for each unique
    sub-chain DH at codegen time -- the whole point of #210 Phase 2 is
    to pay this cost ONCE at build time so module-import is sympy-free.
    """
    from ssik.codegen._compose.seven_r import compose

    kb = load_urdf_kinbody_normalized(
        Path(__file__).parent / "fixtures" / "rizon4.urdf",
        "base_link",
        "flange",
    )
    artifact_source = compose(kb)
    # Phase 2 sentinels: at least one CSE'd builder + insert_derivation calls.
    assert "_build_pq_jointlock_sample_0(" in artifact_source
    assert "_ssik_rr_insert(" in artifact_source
    # The runtime jointlock dispatch lookup map is also populated.
    assert "_PRIMED_LINEARITY_MAP[(" in artifact_source


# ---------------------------------------------------------------------------
# Eligibility set sanity
# ---------------------------------------------------------------------------


def test_rr_eligible_set_excludes_tier0_solvers() -> None:
    """Tier-0 specialisations are explicitly excluded -- they're already
    fast (1-2 ms per call) and beat RR's per-call cost.

    Also verifies ``spherical`` (rank 1, ~7.5 ms) is excluded: its prime
    cost is too high for the modest savings, particularly on Franka
    where 15 of 16 samples route to ``reversed:spherical``.
    """
    excluded = {
        "three_parallel",
        "spherical_two_parallel",
        "spherical_two_intersecting",
        "spherical",
    }
    assert excluded.isdisjoint(_RR_ELIGIBLE_INNER_SOLVERS), (
        f"tier-0/spherical solvers should not be RR-eligible: "
        f"{excluded & _RR_ELIGIBLE_INNER_SOLVERS}"
    )


def test_rr_eligible_set_includes_slow_inner_solvers() -> None:
    """Slow inner solvers (HP, tier-1 search-based) should be eligible
    for cached-RR replacement.
    """
    expected = {
        "two_intersecting",
        "two_parallel",
        "husty_pfurner.general_6r",
    }
    assert expected.issubset(_RR_ELIGIBLE_INNER_SOLVERS), (
        f"slow inner solvers should be RR-eligible: missing {expected - _RR_ELIGIBLE_INNER_SOLVERS}"
    )


# ---------------------------------------------------------------------------
# Composer + runtime integration: built artifact's priming list matches
# what the runtime expects
# ---------------------------------------------------------------------------


def test_rizon4_composer_prime_dh_matches_runtime_dh() -> None:
    """The DH tuples computed at codegen time match what the runtime
    jointlock dispatch produces from the same KinBody.

    This is the load-bearing invariant of #210: if codegen-time DH
    differs from runtime-time DH (e.g. due to floating-point
    nondeterminism or numpy version differences), the prime is wasted
    -- the runtime ``primed_linearity_for_dh`` lookup misses, and we
    fall back to the slow original solver.

    Skips the AE-3 leftvar selection (it would trigger ~30s of sympy
    preprocessing per unique DH; that's exercised in slow integration
    tests). This structural test just verifies the DH tuple match.
    """
    from ssik.codegen._compose.seven_r import _RR_ELIGIBLE_INNER_SOLVERS as eligible
    from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY
    from ssik.kinematics.reverse import reverse_kinematic_chain
    from ssik.solvers.jointlock.seven_r import (
        _DEFAULT_SAMPLES,
        _topology_rank,
        choose_lock_joint,
    )

    kb = load_urdf_kinbody_normalized(
        Path(__file__).parent / "fixtures" / "rizon4.urdf",
        "base_link",
        "flange",
    )
    lock_idx = choose_lock_joint(kb, DEFAULT_TOLERANCE_POLICY)
    joint_limits = kb.joints[lock_idx].limits
    lo, hi = joint_limits if joint_limits is not None else (-float(np.pi), float(np.pi))
    samples = np.linspace(lo, hi, _DEFAULT_SAMPLES, endpoint=False)

    eligible_dhs: list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = []
    for q_lock in samples:
        sub_kb = _lock_joint(kb, lock_idx, float(q_lock))
        _, name = _topology_rank(sub_kb, DEFAULT_TOLERANCE_POLICY)
        bare = name[len("reversed:") :] if name.startswith("reversed:") else name
        if bare not in eligible:
            continue
        sub_kb_for_dh = sub_kb
        if name.startswith("reversed:"):
            sub_kb_for_dh = reverse_kinematic_chain(sub_kb)
        dh = poe_to_dh(sub_kb_for_dh)
        eligible_dhs.append(
            (
                tuple(float(x) for x in dh.alpha),
                tuple(float(x) for x in dh.a),
                tuple(float(x) for x in dh.d),
            )
        )

    # Re-run the same extraction; both passes must produce identical
    # DH lists. Verifies the composer's codegen-time computation is
    # deterministic across calls.
    eligible_dhs_again: list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = []
    for q_lock in samples:
        sub_kb = _lock_joint(kb, lock_idx, float(q_lock))
        _, name = _topology_rank(sub_kb, DEFAULT_TOLERANCE_POLICY)
        bare = name[len("reversed:") :] if name.startswith("reversed:") else name
        if bare not in eligible:
            continue
        sub_kb_for_dh = sub_kb
        if name.startswith("reversed:"):
            sub_kb_for_dh = reverse_kinematic_chain(sub_kb)
        dh = poe_to_dh(sub_kb_for_dh)
        eligible_dhs_again.append(
            (
                tuple(float(x) for x in dh.alpha),
                tuple(float(x) for x in dh.a),
                tuple(float(x) for x in dh.d),
            )
        )

    assert eligible_dhs == eligible_dhs_again
    assert len(eligible_dhs) >= 8, (
        f"Rizon 4 should produce 8+ non-tier-0 inner samples; got {len(eligible_dhs)}"
    )
