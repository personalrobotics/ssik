"""The on-disk Raghavan-Roth derivation cache returns the derivation, or nothing.

Every process that derives an arm's (P, Q) shares the result through a cache
directory, so a cache entry stands in for ~10-45 s of sympy work. Failure modes
the solver suites cannot expose, because they only ever see a working cache:

- an entry left truncated or corrupt (a killed writer) is returned, or crashes
  the solve, instead of being re-derived and replaced;
- an entry differs from what a fresh derivation returns;
- an entry outlives a change to the derivation code, so a changed derivation is
  masked by the old cached result.
"""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from ssik.solvers.ikgeo import _raghavan_roth as rr

# PUMA 560 standard DH: the cheapest derivation (~1.5 s).
_DH = (
    (math.pi / 2, 0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, 0.0),
    (0.0, 0.4318, 0.0203, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.15, 0.4318, 0.0, 0.0),
)


class _Derived(Exception):
    """Raised in place of the symbolic derivation: the cache missed."""


def _no_derivation(*_args: object, **_kwargs: object) -> None:
    raise _Derived


def test_disk_cache_returns_the_fresh_derivation_and_misses_on_code_change(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SSIK_DERIVATION_CACHE", str(tmp_path))
    path = rr._derivation_cache_path(*_DH, linearity_joint=2, apply_so3=False)
    assert path is not None
    assert path.parent == tmp_path

    # A truncated entry is re-derived and replaced, not returned.
    path.write_bytes(b"\x80\x05truncated")
    fresh = rr._derive_pq_for_arm(*_DH)
    assert pickle.loads(path.read_bytes())["key"]["source"] == rr._DERIVATION_SOURCE_HASH
    assert [p.name for p in tmp_path.iterdir()] == [path.name]  # no temp files left

    # A hit does no symbolic work and returns bit-identical (P, Q).
    monkeypatch.setattr(rr, "_derive_pq_symbolic", _no_derivation)
    cached = rr._derive_pq_for_arm(*_DH)
    t_args = np.random.default_rng(0).standard_normal(12).tolist()
    for f_fresh, f_cached in zip(fresh[:4], cached[:4], strict=True):
        np.testing.assert_array_equal(f_cached(*t_args), f_fresh(*t_args))
    assert cached[4] == fresh[4]

    # The key covers the derivation's source and the symbolic stack's versions,
    # so a changed derivation can never be answered from an older entry.
    monkeypatch.setattr(rr, "_DERIVATION_SOURCE_HASH", "0" * 64)
    with pytest.raises(_Derived):
        rr._derive_pq_for_arm(*_DH)

    # SSIK_DERIVATION_CACHE=off disables it: no entry is read.
    monkeypatch.undo()
    monkeypatch.setenv("SSIK_DERIVATION_CACHE", "off")
    assert rr._derivation_cache_path(*_DH, linearity_joint=2, apply_so3=False) is None
