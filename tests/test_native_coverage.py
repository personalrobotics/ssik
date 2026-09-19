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
from pathlib import Path
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


# ---------------------------------------------------------------------------
# #570: the native geometry caches must be keyed on KinBody *identity*.
# ---------------------------------------------------------------------------


def test_kinbody_cache_survives_id_reuse() -> None:
    """CPython reuses an object's address once it is freed, so a cache keyed on
    a bare ``id(kb)`` hands a later, different chain the previous one's baked
    geometry -- and the native solver answers for the wrong robot.

    Measured while diagnosing #570: 300 sequentially created-and-freed KinBody
    objects occupied 16 distinct ids, one of them serving 276 different chains.
    """
    import gc

    from ssik._native import _KinBodyCache

    cache = _KinBodyCache()
    kb = _MANIFEST[_ARMS[0]]
    first = importlib.import_module(
        _MANIFEST[_ARMS[0]].hier_module or f"ssik.prebuilt.{_ARMS[0]}"
    )._KB
    cache.put(first, "first")
    assert cache.get(first) == "first"

    class _Body:  # stands in for a transient KinBody
        pass

    transient = _Body()
    cache.put(transient, "transient")
    key = id(transient)
    del transient
    gc.collect()
    assert key not in cache._entries, "entry outlived its KinBody; its id can be reused"
    del kb


def test_dynamically_built_arms_do_not_cross_contaminate() -> None:
    """The end-to-end shape of #570: build and discard many differing chains,
    solving each natively. A cache that lets one chain's geometry leak into
    another shows up as a solution that does not reproduce its own target."""
    import gc

    import ssik

    fixtures = Path(__file__).parent / "fixtures"
    specs = [
        (fixtures / "ur5.urdf", "base_link", "ee_link"),
        (fixtures / "franka_panda.urdf", "panda_link0", "panda_link8"),
    ]
    q_by_dof = {6: np.array([0.3, -0.7, 0.9, -0.4, 0.8, 0.2]), 7: np.zeros(7) + 0.3}

    for i in range(24):
        path, base, ee = specs[i % len(specs)]
        arm = ssik.Manipulator.from_urdf(path, base=base, ee=ee)
        q = q_by_dof[arm.dof]
        T = arm.fk(q)
        sols = arm.solve(T, enumerate_windings=False)
        assert sols, f"iteration {i}: no solutions for {path.name}"
        worst = max(float(np.linalg.norm(arm.fk(s.q) - T)) for s in sols)
        assert worst < 1e-6, (
            f"iteration {i} ({path.name}): FK {worst:.2e} -- a solution that does not "
            f"reproduce its own target means another chain's geometry was used"
        )
        del arm
        gc.collect()
