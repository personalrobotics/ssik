"""Shared pytest fixtures.

``ssik.solvers.ikgeo._raghavan_roth`` keeps two process-global caches that
are populated at artifact import (the AOT-prime path, #210 / #320):

- ``_DERIVATION_CACHE`` -- per-arm symbolic (P, Q) derivations
- ``_PRIMED_LINEARITY_MAP`` -- per-arm AE-3 leftvar selection

Several tests deliberately clear / pop these to exercise the cold-start and
re-derivation paths (``test_aot_prime``, ``test_cached_rr_jointlock``,
``test_rr_serialize_roundtrip``). Because the caches are module-global, that
mutation leaks across tests: an arm already imported into ``sys.modules``
won't re-run its AOT prime, so after its primed entry is wiped a later
``solve()`` falls back to runtime re-derivation -- which on the cached-RR
jointlock-7R arms (Kassow / Rizon) returns 0 / low-precision candidates.
That surfaced as order-dependent failures once those arms' uniform-fuzz
sweeps were un-xfailed (#319).

This autouse fixture restores any cache entries a test cleared or popped,
*additively*: it re-adds entries that were present before the test and are
now missing, but never removes entries the test legitimately added (e.g. a
freshly imported arm). The entries are deterministic functions of the arm's
DH, so re-adding a wiped entry is exact.
"""

from __future__ import annotations

import pytest

from ssik.solvers.ikgeo import _raghavan_roth as _rr_mod


@pytest.fixture(autouse=True)
def _restore_rr_global_caches():
    deriv_before = dict(_rr_mod._DERIVATION_CACHE)
    lin_before = dict(_rr_mod._PRIMED_LINEARITY_MAP)
    try:
        yield
    finally:
        for dkey, dval in deriv_before.items():
            _rr_mod._DERIVATION_CACHE.setdefault(dkey, dval)
        for lkey, lval in lin_before.items():
            _rr_mod._PRIMED_LINEARITY_MAP.setdefault(lkey, lval)


# xarm7's analytical IK is ~140x slower on Linux/OpenBLAS than macOS/Accelerate
# (#350): its fuzz / aot-prime / tight-policy tests take 30+ min on Linux CI and
# single-handedly set the wall-clock floor, while staying ~13s on macOS (even
# under xdist -- so this is a genuine platform pathology, not test flake or a
# cache-coldness artifact). Defer them to `-m slow` (the macOS pre-push hook +
# nightly, where xarm7 is fast) until the Linux perf bug is root-caused. xarm7's
# other (fast) tests still run per-PR; only these three heavyweights move.
_XARM7_SLOW_ON_CI = (
    "test_prebuilt_7r_random_q_roundtrip",
    "test_prebuilt_7r_tight_policy_machine_precision",
    "test_aot_primed_solve_matches_fixed_pose_fingerprint",
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "xarm7_ik" in item.nodeid and any(p in item.nodeid for p in _XARM7_SLOW_ON_CI):
            item.add_marker(pytest.mark.slow)
