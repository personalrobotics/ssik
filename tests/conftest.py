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

import os

import pytest
from hypothesis import settings

from ssik.solvers.ikgeo import _raghavan_roth as _rr_mod

# CI determinism (#479). Unseeded Hypothesis boundary-hunting occasionally lands
# in a measure-zero near-cancellation shell (#466: openarm exact-SRS at a
# specific near-alignment) that random sampling never hits -- a run-to-run flake
# the 3.10-3.13 matrix (#478) amplifies ~4x. The "ci" profile fixes the example
# set so a CI failure is reproducible (and a real regression fails every run);
# local dev keeps the exploratory (random) default so it still finds new cases.
# GitHub Actions sets CI=true. Per-test ``@settings(...)`` inherit ``derandomize``
# from the active profile unless they set it explicitly.
settings.register_profile("ci", derandomize=True)
settings.register_profile("dev", derandomize=False)
settings.load_profile("ci" if os.environ.get("CI") else "dev")


@pytest.fixture(params=["python", "cpp"])
def three_parallel_backend(request):
    """A ``three_parallel.solve``-compatible callable, once per backend (#499).

    Parametrising a test over this fixture runs the *same* assertions against the
    Python reference and the native C++ solver, so the C++ backend inherits the
    full Python rigor (500-pose fuzz, singular coverage, #56/#362 regressions)
    with zero duplicated assertion logic. The ``cpp`` leg skips when the test-only
    extension isn't built (see tests/_cpp_backend.py + scripts/build_cpp_ext.py).
    """
    if request.param == "python":
        from ssik.solvers.ikgeo import three_parallel

        return three_parallel.solve

    from tests._cpp_backend import cpp_available, cpp_three_parallel_solve

    if not cpp_available():
        pytest.skip("ssik_cpp_ext not built (run scripts/build_cpp_ext.py)")
    return cpp_three_parallel_solve


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
