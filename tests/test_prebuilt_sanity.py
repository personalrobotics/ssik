"""Sanity check for every shipped prebuilt artifact (#132).

Confirms that each ``ssik.prebuilt.<arm>_ik`` module:

1. Imports cleanly under the namespaced location.
2. Exposes ``solve(T)`` and (where applicable) ``fk(q)``.
3. Returns at least one IK candidate for an FK-seeded target.
4. Every returned candidate FK-closes within ``1e-6`` (looser than the
   per-arm bulletproof contracts; this is a packaging smoke test, not a
   solver correctness test).

Doubles as the wheel-build smoke gate referenced from
``pyproject.toml``'s ``[tool.cibuildwheel]`` ``test-command``: if a wheel
ships missing ``.so`` files or a broken prebuilt import, this test fires.

The per-arm parametrisation is driven by ``MANIFEST.toml`` via the
loader at :mod:`ssik.prebuilt._manifest`. Adding a new arm therefore
needs no edits in this file — populate the manifest and the test picks
it up automatically.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik.prebuilt._manifest import load_manifest

_MANIFEST = load_manifest()


@pytest.mark.parametrize(
    "arm_name",
    list(_MANIFEST),
    ids=list(_MANIFEST),
)
def test_prebuilt_arm_imports_and_solves(arm_name: str) -> None:
    arm = _MANIFEST[arm_name]
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")

    # Public surface check.
    assert hasattr(mod, "solve"), f"{arm_name} missing solve()"
    assert hasattr(mod, "fk"), f"{arm_name} missing fk()"

    q = np.array(arm.sample_q, dtype=np.float64)
    assert len(q) == arm.dof, f"sample q length mismatch for {arm_name}"

    T = mod.fk(q)
    assert T.shape == (4, 4)

    sols = mod.solve(T)
    assert sols, f"{arm_name}: solve() returned empty list for FK-seeded target"

    # FK closure on every returned candidate. 1e-6 is the wheel-build smoke
    # bar; per-arm bulletproof contracts (in test_<arm>.py) gate machine
    # precision separately.
    max_fk = max(s.fk_residual for s in sols)
    assert max_fk < 1e-6, f"{arm_name}: max FK residual {max_fk:.2e} > 1e-6"
