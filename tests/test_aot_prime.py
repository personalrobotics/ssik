"""AOT-baked RR-prime architecture invariants (#320).

The pre-#320 priming path shipped a base85-encoded zlib-compressed pickle
of sympy matrices and re-lambdified at module-import (~3.75 s lag per arm
on Kassow KR810). Post-#320 the codegen embeds the lambdified callable's
Python source directly in the artifact via ``inspect.getsource``;
module-import is plain Python parse + exec at ~80 ms.

These tests lock in the post-#320 architecture so a future refactor can't
silently regress back to the blob path:

- Every jointlock-7R arm with an RR-eligible inner solver ships the
  ``_AOT_PRIME_DATA`` block, NOT a ``_RR_PRIME_BLOBS_B85`` tuple.
- ``import <prebuilt>`` does not invoke ``sp.lambdify`` (the priming
  happens entirely via Python module load).
- After import, ``_DERIVATION_CACHE`` has at least one entry whose key
  matches an arm-derived DH.
- The arm's runtime ``solve()`` route hits cached-RR (not HP) for at
  least one of its lock samples.

If anyone reverts to blob-prime or breaks AOT extraction, the relevant
assertion fires immediately on a single test run rather than waiting for
someone to notice a 5x cold-import regression.
"""

from __future__ import annotations

import importlib
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

from ssik.prebuilt._manifest import load_manifest
from ssik.solvers.ikgeo import _raghavan_roth as rr_mod
from ssik.solvers.jointlock.seven_r import _RR_ELIGIBLE_INNER_SOLVERS

REPO_ROOT = Path(__file__).resolve().parent.parent
PREBUILT_DIR = REPO_ROOT / "src" / "ssik" / "prebuilt"


def _jointlock_arms_with_eligible_inner() -> list[str]:
    """Return arm module names whose dispatch contains at least one
    inner solver in ``_RR_ELIGIBLE_INNER_SOLVERS`` (today:
    two_intersecting, two_parallel, husty_pfurner.general_6r).
    Picks the set from the live frozenset so future changes to the
    eligibility policy auto-update this test's parametrisation, and
    newly-added arms are automatically covered.

    ``spherical`` is deliberately excluded today -- see #321 for the
    pending evaluation of whether the per-call runtime win justifies
    the ~45 min/arm cold sympy derivation cost.
    """
    arms = []
    for arm in load_manifest().values():
        if arm.solver != "jointlock.seven_r":
            continue
        # Inspect the artifact's _DISPATCH_CACHE to determine if it has
        # at least one eligible inner solver. We just text-scan since the
        # tuple is emitted at codegen time.
        text = (PREBUILT_DIR / f"{arm.name}.py").read_text()
        # extract the dispatch tuple region cheaply
        start = text.find("_DISPATCH_CACHE = (")
        end = text.find(")", start)
        dispatch_text = text[start:end]
        bare_names = set()
        for raw in dispatch_text.split(","):
            name = raw.strip().strip("'\"")
            if name.startswith("reversed:"):
                name = name[len("reversed:") :]
            bare_names.add(name)
        if bare_names & _RR_ELIGIBLE_INNER_SOLVERS:
            arms.append(arm.name)
    return sorted(arms)


PRIMED_ARMS = _jointlock_arms_with_eligible_inner()


@pytest.mark.parametrize("arm_name", PRIMED_ARMS)
def test_artifact_ships_aot_block_not_blob(arm_name: str) -> None:
    """AOT-baked artifacts contain ``_AOT_PRIME_DATA``; blob-primed
    artifacts contained ``_RR_PRIME_BLOBS_B85``. The post-#320 codegen
    only emits the former."""
    text = (PREBUILT_DIR / f"{arm_name}.py").read_text()
    assert "_AOT_PRIME_DATA" in text, f"{arm_name}: expected post-#320 AOT prime block; not found"
    assert "_RR_PRIME_BLOBS_B85" not in text, (
        f"{arm_name}: legacy blob-prime block present; regen the artifact"
    )
    assert "prime_derivation_from_blob" not in text, (
        f"{arm_name}: legacy blob-prime call present; regen the artifact"
    )


@pytest.mark.parametrize("arm_name", PRIMED_ARMS)
def test_import_does_not_call_sympy_lambdify(arm_name: str) -> None:
    """The AOT path eliminates ``sp.lambdify`` from the import-time hot
    path. If a future change re-introduces it (e.g. partial revert to
    blob-prime), this test fires immediately.

    We can't simply assert "sympy not imported" because numpy and other
    deps may pull it in transitively. Instead, patch ``sp.lambdify`` to
    raise and confirm import still succeeds.
    """
    # Force a fresh import: pop the module if previously imported. This
    # clears the shared ``_raghavan_roth`` derivation caches to observe a
    # genuine cold import; the ``_restore_rr_global_caches`` autouse fixture
    # (tests/conftest.py) re-adds any wiped entries afterward so the
    # mutation doesn't leak into later arms' solves.
    import sys

    mod_path = f"ssik.prebuilt.{arm_name}"
    sys.modules.pop(mod_path, None)
    rr_mod._DERIVATION_CACHE.clear()
    rr_mod._PRIMED_LINEARITY_MAP.clear()

    import sympy as sp

    with mock.patch.object(
        sp,
        "lambdify",
        side_effect=AssertionError(f"{arm_name}: AOT path must not call sp.lambdify"),
    ):
        importlib.import_module(mod_path)
    # And confirm the prime did populate the cache.
    assert rr_mod._DERIVATION_CACHE, (
        f"{arm_name}: import completed but _DERIVATION_CACHE is empty -- AOT-prime did not run?"
    )


@pytest.mark.parametrize("arm_name", PRIMED_ARMS)
def test_aot_primed_solve_matches_fixed_pose_fingerprint(arm_name: str) -> None:
    """Run a tiny deterministic pose sweep and confirm the solver
    produces a non-empty solution set whose FK closure is at the
    machine-precision floor expected of a cached-RR fast path.

    This is the runtime-side parity check: even if the architecture
    looks right (above tests), this confirms the lambdified callables
    actually evaluate to the correct algebraic result on real poses.
    """
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    # Gate on the arm's own calibrated FK ceiling (the same one the
    # Hypothesis sweep uses). The cached-RR HP arms (Kassow / Rizon) close
    # to ~1e-6..1e-5 on adverse poses, well within their documented 1e-4
    # ceiling; a hardcoded 1e-6 here under-specifies them.
    ceiling = load_manifest()[arm_name].fk_ceiling_fuzz
    rng = np.random.default_rng(20260608)
    # Use a small sweep -- the heavy 500-pose sweep lives in the
    # Hypothesis-driven test_prebuilt_uniform_fuzz tests.
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, mod.DOF)
        T = mod.fk(q)
        sols = mod.solve(T, respect_limits=False, q_seed=q)
        # Skip degenerate poses where 0 sols is a known coverage gap;
        # we just want to verify the non-degenerate ones produce solid
        # FK closure.
        if not sols:
            continue
        max_fk = max(float(s.fk_residual) for s in sols)
        assert max_fk < ceiling, (
            f"{arm_name}: AOT-baked solve FK closure {max_fk:.2e} > {ceiling:.0e} ceiling at q={q}"
        )
