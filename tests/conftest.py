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

import argparse
import json
import os
import re
from pathlib import Path

import pytest
from hypothesis import settings

from ssik.solvers.ikgeo import _raghavan_roth as _rr_mod
from tests import _sweeps

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


# --shard K/N: CI splits each Python version's suite across N jobs. Every job
# collects the same items and keeps a deterministic slice, so the N slices are
# disjoint and together cover the collection exactly. Slices are balanced by
# the measured durations in _shard_durations.json (regenerate with
# scripts/regen_test_durations.py); a stale file only unbalances them.
# --shard-record writes {k, n, collected, selected} so a CI job can prove the
# union of the N shards is the whole collection (scripts/check_pytest_shards.py).
_SHARD_DURATIONS = Path(__file__).with_name("_shard_durations.json")
_SHARD_DEFAULT_S = 0.1


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard", type=_parse_shard, metavar="K/N", help="run only shard K of N (1-based)"
    )
    parser.addoption("--shard-record", metavar="PATH", help="write the shard's selection here")


def pytest_configure(config: pytest.Config) -> None:
    try:
        _sweeps.sampled()
    except ValueError as e:
        raise pytest.UsageError(str(e)) from None


def pytest_report_header(config: pytest.Config) -> str:
    sampled = _sweeps.sampled()
    full = [s for s in _sweeps.SWEEPS if s not in sampled]
    return f"sweeps: PR sample {sorted(sampled) or 'none'}, full {full or 'none'}"


def _parse_shard(spec: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)/(\d+)", spec)
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise argparse.ArgumentTypeError(f"wants K/N with 1 <= K <= N, got {spec!r}")
    return int(m.group(1)), int(m.group(2))


def shard_partition(nodeids: list[str], n: int, cost: dict[str, float]) -> list[list[str]]:
    """Split ``nodeids`` into ``n`` disjoint lists covering them all: greedy
    longest-first onto the least-loaded shard, ties broken by nodeid."""
    loads = [0.0] * n
    shards: list[list[str]] = [[] for _ in range(n)]
    for nid in sorted(nodeids, key=lambda x: (-cost.get(x, _SHARD_DEFAULT_S), x)):
        i = min(range(n), key=lambda i: (loads[i], i))
        loads[i] += cost.get(nid, _SHARD_DEFAULT_S)
        shards[i].append(nid)
    return shards


@pytest.hookimpl(trylast=True)  # after -m / -k deselection: shard what would run
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    shard = config.getoption("--shard")
    if not shard:
        return
    k, n = shard
    cost = json.loads(_SHARD_DURATIONS.read_text()) if _SHARD_DURATIONS.exists() else {}
    collected = [it.nodeid for it in items]
    mine = set(shard_partition(collected, n, cost)[k - 1])
    keep = [it for it in items if it.nodeid in mine]
    config.hook.pytest_deselected(items=[it for it in items if it.nodeid not in mine])
    items[:] = keep
    record = config.getoption("--shard-record")
    # Under xdist every worker collects the same items; one of them records.
    if record and os.environ.get("PYTEST_XDIST_WORKER", "gw0") == "gw0":
        payload = {"k": k, "n": n, "collected": collected, "selected": [it.nodeid for it in keep]}
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        Path(record).write_text(json.dumps(payload))


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
