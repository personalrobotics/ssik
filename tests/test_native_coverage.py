"""Every shipped arm must actually reach the native backend (#568).

``native=True`` has been the default for all 72 prebuilt arms since v5.0, and
the C++ path is what users get unless they ask otherwise. The dispatch is
deliberately forgiving -- ``_try_native_*`` returns ``None`` and the Python
solver takes over -- so an arm that stops resolving natively keeps returning
correct IK and says nothing. The only visible symptom is being slower, which is
exactly the kind of regression that hides for a release or two.

These tests make that failure loud: the extension must be present, and every
arm must take the native path through it.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from ssik._native import native_available
from ssik.prebuilt._manifest import load_manifest

_MANIFEST = load_manifest()
_ARMS = sorted(_MANIFEST)

# The one hook each artifact calls, by family. Every arm has exactly one.
_HOOKS = ("_try_native_solve", "_try_native_solve_7r", "_try_native_jointlock_solve")


def test_native_extension_is_available() -> None:
    """Fail rather than skip.

    Native is the shipped default, so a run without the extension is not
    exercising what users get. Every native test in the suite is guarded by
    ``skipif(not native_available())``, which means an environment missing the
    extension quietly reports success while testing the fallback instead. This
    is the one place that refuses to be quiet about it.

    Build it with ``python scripts/build_cpp_ext.py --out-dir src/ssik``.
    """
    assert native_available(), (
        "ssik._ssik_native is not built, so the native backend -- the default "
        "for all shipped arms -- is not under test here, and every native test "
        "in this suite is silently skipping. Build it with: "
        "python scripts/build_cpp_ext.py --out-dir src/ssik"
    )


def _hook_name(mod: Any) -> str:
    found = [h for h in _HOOKS if hasattr(mod, h)]
    assert len(found) == 1, f"expected exactly one native hook, found {found}"
    return found[0]


@pytest.mark.skipif(not native_available(), reason="covered by the test above")
@pytest.mark.parametrize("arm_name", _ARMS)
def test_arm_resolves_to_the_native_backend(arm_name: str, monkeypatch: Any) -> None:
    """The arm's native hook must return solutions, not ``None``.

    Spies on the hook the emitted artifact actually calls, so this follows the
    real dispatch rather than re-deriving it: a geometry that stops matching its
    native family, a missing baked sidecar, or a binding signature drift all
    surface here as a fallback to Python.
    """
    arm = _MANIFEST[arm_name]
    mod = importlib.import_module(arm.hier_module or f"ssik.prebuilt.{arm_name}")
    hook = _hook_name(mod)
    original = getattr(mod, hook)
    calls: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(mod, hook, spy)
    sols = mod.solve(mod.fk(np.array(arm.sample_q)))

    assert calls, f"{arm_name}: solve() never called {hook} -- native dispatch is not wired"
    assert calls[0] is not None, (
        f"{arm_name}: {hook} returned None, so this arm fell back to the Python "
        f"solver. Correct IK, but not the backend we ship."
    )
    assert sols, f"{arm_name}: native path returned no solutions at its manifest sample_q"
