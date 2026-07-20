"""In-limits resolver wiring for thin-wrapper 7R prebuilts (D1, #389).

Each redundant-7R solver family ships its own exact in-limits resolver
(SRS swivel via ``_swivel_limits`` vs spherical-shoulder q6 via
``spherical_shoulder`` / ``spherical_shoulder_polished``). The thin-wrapper
codegen must wire the resolver that *matches the arm's solver* — importing the
SRS resolver into a spherical-shoulder artifact is a silent no-op (it returns
``[]`` for a non-SRS chain), so the arm's exact in-limits recovery is dead code
(the original D1 bug: Franka/FR3 recovered 0 where their own resolver recovers
10/5).

The structural guard below iterates **every** prebuilt and asserts the wired
resolver is exactly the one codegen's has-attr rule selects, so this
mis-wiring class cannot be reintroduced without turning CI red.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from ssik.prebuilt._manifest import load_manifest

# The has-attr derivation codegen uses: a solver family that defines its own
# ``resolve_in_limits`` owns it; everything else falls back to the SRS resolver.
_SRS_RESOLVER = "ssik.solvers.seven_r._swivel_limits"


def _expected_resolver(solver_name: str):
    solver_module = importlib.import_module(f"ssik.solvers.{solver_name}")
    module = solver_module if hasattr(solver_module, "resolve_in_limits") else None
    if module is None:
        module = importlib.import_module(_SRS_RESOLVER)
    return module.resolve_in_limits


def _thin_wrapper_prebuilts() -> list[str]:
    names = []
    for arm in load_manifest():
        try:
            mod = importlib.import_module(f"ssik.prebuilt.{arm}")
        except Exception:
            continue
        # Thin-wrapper artifacts are exactly the ones that emit ``_resolve_in_limits``.
        if hasattr(mod, "_resolve_in_limits"):
            names.append(arm)
    return names


_THIN_WRAPPER = _thin_wrapper_prebuilts()


@pytest.mark.parametrize("arm", _THIN_WRAPPER)
def test_prebuilt_wires_solver_matched_resolver(arm: str) -> None:
    """Structural guard: every thin-wrapper prebuilt imports the in-limits
    resolver from the module that matches its own solver — never a mismatched
    no-op (D1). Fails CI the instant a new/renamed solver drifts."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    expected = _expected_resolver(mod.SOLVER_NAME)
    assert mod._resolve_in_limits is expected, (
        f"{arm} (solver {mod.SOLVER_NAME}) wired {mod._resolve_in_limits.__module__}."
        f"resolve_in_limits but its solver family provides "
        f"{expected.__module__}.resolve_in_limits — the D1 mis-wiring."
    )


@pytest.mark.parametrize(
    ("arm", "q"),
    [
        ("franka_panda_ik", [0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5]),
        ("fr3_ik", [0.3, -0.4, 0.2, -1.6, 0.1, 1.3, 0.5]),
        ("xarm7_ik", [0.2, -0.3, 0.5, 1.0, 0.4, 1.0, 0.3]),
    ],
)
def test_spherical_shoulder_in_limits_recovery_is_live(arm: str, q: list[float]) -> None:
    """Behavioral regression: the spherical-shoulder prebuilts' in-limits
    resolver actually recovers solutions (it was the SRS no-op returning 0)."""
    mod = importlib.import_module(f"ssik.prebuilt.{arm}")
    t = mod.fk(np.asarray(q))
    sols = mod._resolve_in_limits(mod._KB, t)
    assert sols, f"{arm}: in-limits resolver recovered nothing (D1 dead-code regression)"
    worst = max(float(np.max(np.abs(mod.fk(s.q) - t))) for s in sols)
    assert worst < 1e-9, f"{arm}: in-limits solution FK closure {worst:.2e}"
