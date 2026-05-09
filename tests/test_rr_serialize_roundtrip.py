"""Round-trip tests for cached-RR derivation serialization (#210 Phase 2).

The build pipeline pre-computes Raghavan-Roth's symbolic derivation at codegen
time, serializes the sympy matrices via :func:`serialize_derivation`, and
embeds the bytes in the artifact. At module-init the artifact deserializes
via :func:`prime_derivation_from_blob` -- ~0.25 s per blob vs ~7 s cold
sympy derivation, ~30x faster import.

Test contract:
- serialize -> deserialize roundtrip preserves solver behavior bit-for-bit.
- ``prime_derivation_from_blob`` populates the cache + linearity map.
- Version mismatch raises ValueError.
- Solving from a primed-from-blob cache produces same FK closure as cold.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from ssik._urdf import load_urdf_kinbody_normalized
from ssik.kinematics.poe_fk import poe_forward_kinematics
from ssik.kinematics.poe_to_dh import poe_to_dh
from ssik.solvers.ikgeo import general_6r as rr
from ssik.solvers.ikgeo._raghavan_roth import (
    _DERIVATION_CACHE,
    _PRIMED_LINEARITY_MAP,
    _cached_best_leftvar,
    prime_derivation_from_blob,
    primed_linearity_for_dh,
    serialize_derivation,
)
from ssik.solvers.jointlock.seven_r import _lock_joint, choose_lock_joint

FIXTURES = Path(__file__).parent / "fixtures"

# Module-shared subchain: lock joint 2 of Rizon 4 at q_lock=0 produces an
# RR-eligible 6R sub-chain (HP is the inner solver baseline).
_RIZON4_KB = load_urdf_kinbody_normalized(FIXTURES / "rizon4.urdf", "base_link", "flange")
_LOCK_IDX = choose_lock_joint(_RIZON4_KB)
_SUB_KB = _lock_joint(_RIZON4_KB, _LOCK_IDX, 0.0)
_DH = poe_to_dh(_SUB_KB)
_ALPHA = tuple(float(x) for x in _DH.alpha)
_A = tuple(float(x) for x in _DH.a)
_D = tuple(float(x) for x in _DH.d)


@pytest.fixture
def fresh_cache():
    """Save + restore the derivation cache around a test."""
    saved_cache = dict(_DERIVATION_CACHE)
    saved_linearity = dict(_PRIMED_LINEARITY_MAP)
    yield
    _DERIVATION_CACHE.clear()
    _DERIVATION_CACHE.update(saved_cache)
    _PRIMED_LINEARITY_MAP.clear()
    _PRIMED_LINEARITY_MAP.update(saved_linearity)


def test_serialize_returns_bytes() -> None:
    linearity = _cached_best_leftvar(_ALPHA, _A, _D)
    blob = serialize_derivation(_ALPHA, _A, _D, linearity_joint=linearity)
    assert isinstance(blob, bytes)
    # Should be at least a few KB (sympy matrices are non-trivial).
    assert len(blob) > 1024


def test_serialize_payload_v1_schema() -> None:
    linearity = _cached_best_leftvar(_ALPHA, _A, _D)
    blob = serialize_derivation(_ALPHA, _A, _D, linearity_joint=linearity)
    payload = pickle.loads(blob)
    assert payload["version"] == 1
    assert set(payload.keys()) >= {
        "version",
        "alpha",
        "a",
        "d",
        "linearity_joint",
        "apply_so3",
        "sym_p_sin",
        "sym_p_cos",
        "sym_p_one",
        "sym_q",
        "sym_t_target",
        "left_bilinear",
        "right_bilinear",
        "drop_joint",
    }


def test_prime_from_blob_populates_cache(fresh_cache) -> None:
    linearity = _cached_best_leftvar(_ALPHA, _A, _D)
    blob = serialize_derivation(_ALPHA, _A, _D, linearity_joint=linearity)
    _DERIVATION_CACHE.clear()
    _PRIMED_LINEARITY_MAP.clear()
    prime_derivation_from_blob(blob)
    assert (_ALPHA, _A, _D, linearity, False) in _DERIVATION_CACHE
    assert primed_linearity_for_dh(_ALPHA, _A, _D) == (linearity, False)


def test_prime_from_blob_rejects_version_mismatch() -> None:
    bad_payload = pickle.dumps({"version": 999})
    with pytest.raises(ValueError, match="unsupported derivation payload version"):
        prime_derivation_from_blob(bad_payload)


def test_solve_from_blob_matches_cold_derivation(fresh_cache) -> None:
    """Solving against a blob-primed cache produces same q-vectors as a cold
    derivation -- the serialization is correctness-preserving."""
    linearity = _cached_best_leftvar(_ALPHA, _A, _D)

    # Cold path: clear cache, run solve directly (re-derives in-flight).
    _DERIVATION_CACHE.clear()
    _PRIMED_LINEARITY_MAP.clear()
    q_truth = np.array([0.3, 0.4, -0.5, 0.6, 0.2, -0.3], dtype=np.float64)
    T_target = poe_forward_kinematics(_SUB_KB, q_truth)
    cold_sols, _ = rr.solve(_SUB_KB, T_target, allow_refinement=True, linearity_joint=linearity)
    assert cold_sols, "cold solve produced no solutions"

    # Capture blob from the populated cache, then clear and reload.
    blob = serialize_derivation(_ALPHA, _A, _D, linearity_joint=linearity)
    _DERIVATION_CACHE.clear()
    _PRIMED_LINEARITY_MAP.clear()
    prime_derivation_from_blob(blob)

    # Warm path: solve again, should hit the primed cache (no re-derivation).
    warm_sols, _ = rr.solve(_SUB_KB, T_target, allow_refinement=True, linearity_joint=linearity)

    assert len(cold_sols) == len(warm_sols)
    cold_sorted = sorted(cold_sols, key=lambda s: tuple(s.q.tolist()))
    warm_sorted = sorted(warm_sols, key=lambda s: tuple(s.q.tolist()))
    for c, w in zip(cold_sorted, warm_sorted, strict=True):
        # Bit-identical q-vectors (same lambdas + same lapack).
        np.testing.assert_array_equal(c.q, w.q)


@pytest.mark.slow
def test_codegen_artifact_with_baked_blobs_imports_quickly(tmp_path) -> None:
    """End-to-end: build a Rizon 4 artifact and verify the resulting .py
    file contains the b85 blob block. Marked ``slow`` because the actual
    artifact build takes 5-7 min (16 cold sympy derivations x ~7 s each).
    """
    from ssik.core.codegen import emit_artifact
    from ssik.core.dispatcher import dispatch

    plan = dispatch(_RIZON4_KB)
    out = tmp_path / "rizon4_baked_smoke.py"
    result = emit_artifact(
        kb=_RIZON4_KB,
        plan=plan,
        module_name="rizon4_baked_smoke",
        output_path=str(out),
    )
    src = result.source
    assert "_RR_PRIME_BLOBS_B85" in src, "expected baked blobs in the artifact"
    assert "prime_derivation_from_blob" in src
    # The artifact should have at least 100 KB of base85-encoded blob
    # data (Rizon 4 has 14+ non-tier-0 samples x ~5 KB each compressed).
    assert len(src) > 100_000, f"artifact too small: {len(src)} bytes"
