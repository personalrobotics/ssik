"""Guards that the Cython perf extensions are measured + shipped compiled.

A dev checkout that built ``poe_fk.so`` but not ``refinement.so`` ran the
refinement layer in pure-Python mode. ``regen_bench`` measured that and the
inflated numbers were committed as authoritative (gen3 51 vs 37 ms, piper 2.5
vs 1.1 ms). The annotated ``.py`` sources import fine either way, so nothing
functional caught it -- only the timing was wrong.

Three guards now prevent a recurrence, and this module pins their invariants:

1. ``regen_bench`` refuses to measure ssik timing against uncompiled perf
   extensions (unless ``--allow-uncompiled``).
2. The cibuildwheel ``test-command`` asserts each perf-target module loads from
   a compiled ``.so`` in the built wheel.
3. Both of the above are single-sourced from ``hatch_build.CYTHON_TARGETS``;
   the drift test below fails if a new target is added without updating the
   wheel smoke check.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

import regen_bench  # type: ignore[import-not-found]  # noqa: E402


def _cython_target_modules() -> list[str]:
    """Module names of hatch_build.CYTHON_TARGETS (parsed, no hatchling import)."""
    tree = ast.parse((_REPO / "hatch_build.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            named = isinstance(node.target, ast.Name) and node.target.id == "CYTHON_TARGETS"
            value = node.value
        elif isinstance(node, ast.Assign):
            named = any(isinstance(t, ast.Name) and t.id == "CYTHON_TARGETS" for t in node.targets)
            value = node.value
        else:
            continue
        if named and value is not None:
            paths = ast.literal_eval(value)
            return [
                p.removeprefix("src/")
                .removesuffix(".py")
                .removesuffix("/__init__")
                .replace("/", ".")
                for p in paths
            ]
    raise AssertionError("CYTHON_TARGETS not found in hatch_build.py")


def _wheel_smoke_command() -> str:
    """The full cibuildwheel test-command string (all checks concatenated)."""
    cfg = tomllib.loads((_REPO / "pyproject.toml").read_text())
    return "\n".join(cfg["tool"]["cibuildwheel"]["test-command"])


def test_wheel_smoke_covers_every_cython_target() -> None:
    """Every compiled perf target must be asserted-compiled by the wheel smoke
    check. Adding a CYTHON_TARGET without a matching wheel assertion (so a wheel
    could ship it uncompiled and undetected) fails here."""
    smoke = _wheel_smoke_command()
    for mod in _cython_target_modules():
        assert mod in smoke, (
            f"{mod} is a hatch_build.CYTHON_TARGETS compile target but the "
            f"cibuildwheel test-command does not assert it loads compiled. Add a "
            f'\'{mod}.__file__.endswith((".so", ".pyd"))\' check to test-command.'
        )
    # And the check must actually verify a compiled extension, not merely import.
    assert ".so" in smoke
    assert "endswith" in smoke


def test_regen_bench_gate_blocks_uncompiled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bench gate hard-errors on uncompiled perf extensions and passes when
    they are compiled -- independent of this checkout's own build state."""
    monkeypatch.setattr(regen_bench, "_uncompiled_perf_extensions", lambda: ["ssik.refinement"])
    with pytest.raises(SystemExit, match="refusing to measure"):
        regen_bench._require_compiled_perf_extensions(allow_uncompiled=False)
    # Escape hatch warns but proceeds.
    regen_bench._require_compiled_perf_extensions(allow_uncompiled=True)
    # All-compiled passes.
    monkeypatch.setattr(regen_bench, "_uncompiled_perf_extensions", list)
    regen_bench._require_compiled_perf_extensions(allow_uncompiled=False)


def test_uncompiled_detector_parses_targets() -> None:
    """The detector resolves the same modules as CYTHON_TARGETS and returns a
    subset of them (whichever are currently uncompiled in this env)."""
    targets = set(_cython_target_modules())
    assert targets == {"ssik.refinement", "ssik.kinematics.poe_fk"}
    assert set(regen_bench._uncompiled_perf_extensions()) <= targets
