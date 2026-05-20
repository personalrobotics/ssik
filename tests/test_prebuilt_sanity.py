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
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# The prebuilts shipped in the wheel. Each entry is (module_name,
# dof, sample_q) where sample_q is a config chosen to land inside the
# arm's reachable set so the round-trip will return at least one IK.
PREBUILT_ARMS: list[tuple[str, int, list[float]]] = [
    ("ur5_ik", 6, [0.3, -0.5, 0.7, 0.2, 0.4, -0.1]),
    ("puma560_ik", 6, [0.4, -0.6, 0.8, 1.0, -0.4, 0.3]),
    ("jaco2_ik", 6, [0.5, -0.8, 0.9, 1.1, -0.5, 0.4]),
    ("xarm6_ik", 6, [0.3, 1.0, -0.8, 0.2, 0.4, -0.1]),
    ("z1_ik", 6, [0.3, 0.8, -0.5, 0.2, 0.4, -0.1]),
    ("piper_ik", 6, [0.3, 0.7, -0.5, 0.2, 0.4, -0.1]),
    ("iiwa14_ik", 7, [0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4]),
    ("gen3_ik", 7, [0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4]),
    ("franka_panda_ik", 7, [0.2, 0.3, -0.4, -1.2, 0.3, 1.4, 0.5]),
    ("rizon4_ik", 7, [0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4]),
    ("kassow_kr810_ik", 7, [0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4]),
    ("xarm7_ik", 7, [0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4]),
    ("rizon10_ik", 7, [0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4]),
]


@pytest.mark.parametrize(("arm_name", "dof", "q_sample"), PREBUILT_ARMS)
def test_prebuilt_arm_imports_and_solves(arm_name: str, dof: int, q_sample: list[float]) -> None:
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")

    # Public surface check.
    assert hasattr(mod, "solve"), f"{arm_name} missing solve()"
    assert hasattr(mod, "fk"), f"{arm_name} missing fk()"

    q = np.array(q_sample, dtype=np.float64)
    assert len(q) == dof, f"sample q length mismatch for {arm_name}"

    T = mod.fk(q)
    assert T.shape == (4, 4)

    sols = mod.solve(T)
    assert sols, f"{arm_name}: solve() returned empty list for FK-seeded target"

    # FK closure on every returned candidate. 1e-6 is the wheel-build smoke
    # bar; per-arm bulletproof contracts (in test_<arm>.py) gate machine
    # precision separately.
    max_fk = max(s.fk_residual for s in sols)
    assert max_fk < 1e-6, f"{arm_name}: max FK residual {max_fk:.2e} > 1e-6"
