"""Guard the single-source solver registry (U1, #389).

``ssik.core.solver_registry.SOLVERS`` replaced ~six parallel ``name -> X`` tables
that used to be edited in lockstep (dispatcher estimates, codegen import paths,
manipulator module paths, codegen composers, fk-atol / force-refine overrides).
These tests are the structural guard that keeps a new/renamed solver from
shipping half-wired: every row must import, every composer must exist, the
dispatcher may not emit a name the table lacks, and the one value still mirrored
in a live solver (three_parallel's FK gate) must match. Any drift turns CI red
here instead of KeyError-ing in the field.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from ssik.core.solver_registry import SOLVERS, SolverSpec

_SRC = Path(__file__).resolve().parents[1] / "src" / "ssik"


@pytest.mark.parametrize("name", sorted(SOLVERS))
def test_every_solver_module_imports_and_solves(name: str) -> None:
    """Each spec's ``module_path`` imports and exposes a callable ``solve``."""
    spec = SOLVERS[name]
    assert spec.name == name, "SOLVERS key must equal the spec's name"
    module = importlib.import_module(spec.module_path)
    assert callable(getattr(module, "solve", None)), (
        f"{name}: {spec.module_path} has no callable solve()"
    )


@pytest.mark.parametrize("name", sorted(n for n, s in SOLVERS.items() if s.composer))
def test_every_composer_imports_with_entry_points(name: str) -> None:
    """Each specialised solver's ``composer`` module exposes ``compose`` and
    ``render_constants_header`` (the codegen contract)."""
    composer = SOLVERS[name].composer
    assert composer is not None
    mod = importlib.import_module(composer)
    assert callable(getattr(mod, "compose", None)), f"{name}: {composer}.compose missing"
    assert callable(getattr(mod, "render_constants_header", None)), (
        f"{name}: {composer}.render_constants_header missing"
    )


def test_dispatcher_only_emits_registered_solvers() -> None:
    """Every solver name the dispatcher can hand to ``_make_plan`` is in SOLVERS.

    Parses the dispatcher source for the string literals passed to ``_make_plan``
    so a new dispatch branch that references an unregistered solver fails here
    rather than KeyError-ing at ``dispatch()`` / ``ssik build`` time.
    """
    src = (_SRC / "core" / "dispatcher.py").read_text()
    emitted = set(re.findall(r'_make_plan\(\s*"([^"]+)"', src))
    assert emitted, "no _make_plan calls found -- regex or dispatcher structure changed"
    missing = emitted - set(SOLVERS)
    assert not missing, f"dispatcher emits solvers absent from SOLVERS: {sorted(missing)}"


def test_three_parallel_fk_gate_matches_live_solver() -> None:
    """The FK-verify gate baked into three_parallel's artifact must equal the
    live solver's constant -- the value the old codegen mirror could silently
    drift from (#362)."""
    from ssik.solvers.ikgeo import three_parallel

    # fk_atol_expr is a Python expression string emitted into the artifact;
    # compare its numeric value (not text) to the live gate.
    assert float(SOLVERS["ikgeo.three_parallel"].fk_atol_expr) == three_parallel._FK_VERIFY_ATOL


def test_every_prebuilt_solver_name_is_registered() -> None:
    """Every shipped prebuilt artifact's ``SOLVER_NAME`` is a registered solver."""
    from ssik.prebuilt._manifest import load_manifest

    for arm in load_manifest():
        try:
            mod = importlib.import_module(f"ssik.prebuilt.{arm}")
        except Exception:
            continue
        assert mod.SOLVER_NAME in SOLVERS, f"{arm}: SOLVER_NAME {mod.SOLVER_NAME} not in SOLVERS"


def test_module_path_is_derived_not_stored() -> None:
    """``module_path`` is a pure derivation of the name (the property that let us
    delete the two byte-identical import-path/module-path tables)."""
    for name, spec in SOLVERS.items():
        assert spec.module_path == f"ssik.solvers.{name}"


def test_spec_is_frozen() -> None:
    """The registry is immutable -- no runtime mutation of solver metadata."""
    spec = next(iter(SOLVERS.values()))
    with pytest.raises((AttributeError, TypeError)):
        spec.tier = 99  # type: ignore[misc]
    assert isinstance(spec, SolverSpec)
