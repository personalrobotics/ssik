"""CI's lane and sweep-tier classification fails safe (scripts/classify_changes.py).

The classification decides what a pull request's CI runs, and CI cannot check
its own decision afterwards: a code change classified as examples-only skips
the whole suite, a change to the C++ jobs' inputs classified as Python-only
skips the C++ emit, drift guard and conformance, and a change to a sweep's
covered code classified as uncovered runs only the sweep's PR sample. All pass
green while testing less. So every path that is not plainly examples or docs
must choose the full suite, every path that is not plainly outside the C++
jobs' inputs must run them, lookalike paths must not pass as either, and a
sweep may run its sample only when no changed path is covered by it. No paths
(a push, a failed diff) must mean everything in full.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from tests import _sweeps

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "classify_changes.py"


def _lanes(*paths: str) -> tuple[bool, bool, set[str]]:
    out = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        input="".join(f"{p}\n" for p in paths),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fields = dict(line.split("=", 1) for line in out.splitlines())
    assert set(fields) == {"full", "cpp", "pr_sweeps"}
    assert fields["full"] in ("true", "false")
    assert fields["cpp"] in ("true", "false")
    full, cpp = fields["full"] == "true", fields["cpp"] == "true"
    assert full or not cpp, "the C++ jobs run only in the full suite"
    return full, cpp, {s for s in fields["pr_sweeps"].split(",") if s}


def _classify(*paths: str) -> tuple[bool, set[str]]:
    full, _, sampled = _lanes(*paths)
    return full, sampled


def test_no_paths_runs_everything_in_full() -> None:
    assert _lanes() == (True, True, set())


@pytest.mark.parametrize(
    "paths",
    [
        ("examples/05_viser_interactive_ik.py",),
        ("examples/README.md", "docs/api.md", "README.md", "LICENSE", ".gitignore"),
    ],
)
def test_examples_and_docs_take_the_examples_lane(paths: tuple[str, ...]) -> None:
    full, cpp, _ = _lanes(*paths)
    assert not full
    assert not cpp


@pytest.mark.parametrize(
    "path",
    [
        "src/ssik/README.md",  # a .md under a code directory
        "tests/fixtures/notes.md",
        ".github/workflows/ci.yml",
        "scripts/run_examples.py",
        "examples-old/demo.py",  # lookalike prefixes
        "docs.py",
        "src/LICENSE",
        "pyproject.toml",
        "uv.lock",
    ],
)
def test_anything_else_takes_the_full_suite(path: str) -> None:
    full, _ = _classify("examples/05_viser_interactive_ik.py", path)
    assert full


def test_a_sweep_samples_only_when_no_path_is_covered() -> None:
    # Code no sweep covers: every sweep may run its sample.
    assert _classify("src/ssik/cli.py", "tests/test_cli_add_arm.py") == (True, set(_sweeps.SWEEPS))
    # Shared code: no sweep may.
    for path in ("src/ssik/solvers/seven_r/srs.py", "cpp/include/ssik_cpp/chart.hpp"):
        assert _classify("src/ssik/cli.py", path) == (True, set())
    # A sweep's own code runs that sweep in full and leaves the others sampled.
    for sweep in _sweeps.SWEEPS:
        own = [p for p in _sweeps.COVERED[sweep] if p.startswith("tests/test_")]
        assert own, sweep
        _, sampled = _classify(*own)
        assert sampled == set(_sweeps.SWEEPS) - {sweep}
    # Lookalikes of covered paths are not covered.
    assert _classify("src/ssik/chart_extra.py")[1] == set(_sweeps.SWEEPS)
    assert _classify("src/ssik/chart.py")[1] == set(_sweeps.SWEEPS) - {"chart_native"}


@pytest.mark.parametrize(
    "paths",
    [
        ("src/ssik/teleop.py",),
        ("src/ssik/cli.py", "src/ssik/_validation.py", "src/ssik/internals/__init__.py"),
        ("tests/test_teleop.py", "tests/conftest.py", "tests/fixtures/ur5.urdf"),
        ("src/ssik/teleop.py", "tests/test_teleop.py", "examples/06_teleop.py", "README.md"),
    ],
)
def test_python_only_changes_skip_the_cpp_jobs(paths: tuple[str, ...]) -> None:
    full, cpp, _ = _lanes(*paths)
    assert full
    assert not cpp


@pytest.mark.parametrize(
    "path",
    [
        # The C++ sources, the oracle the emit and drift guard compare against,
        # the shipped artifacts and the build.
        "cpp/include/ssik_cpp/fk.hpp",
        "cpp/README.md",
        "src/ssik/solvers/seven_r/srs.py",
        "src/ssik/core/codegen.py",
        "src/ssik/codegen/_compose/seven_r.py",
        "src/ssik/prebuilt/MANIFEST.toml",
        "src/ssik/prebuilt/universal_robots/ur5_ik.py",
        "src/ssik/prebuilt/README.md",
        "src/ssik/subproblems/sp1.py",
        "src/ssik/kinematics/poe_fk.py",
        "src/ssik/refinement/polish.py",
        "src/ssik/_native.py",
        "src/ssik/_kinbody.py",
        "src/ssik/cpp/__init__.py",
        "src/ssik/__init__.py",
        # Imported by the solvers, so part of the oracle.
        "src/ssik/manipulator.py",
        "src/ssik/tracker.py",
        "src/ssik/chart.py",
        # A module nobody has shown to be outside the oracle.
        "src/ssik/new_module.py",
        "scripts/cpp_emit.py",
        "scripts/build_cpp_ext.py",
        "scripts/fetch_eigen.py",
        "scripts/classify_changes.py",
        "scripts/regen_docs.py",
        "tools/encode_demo_assets.py",
        "pyproject.toml",
        "uv.lock",
        "hatch_build.py",
        ".github/workflows/ci.yml",
        # Lookalikes of the Python-only modules and of tests/.
        "src/ssik/teleop_extra.py",
        "src/ssik/teleop.pyx",
        "src/ssik/internals.py",
        "src/ssik/internals_extra/__init__.py",
        "src/ssik/solvers/cli.py",
        "src/tests/test_x.py",
        "tests.py",
    ],
)
def test_anything_else_runs_the_cpp_jobs(path: str) -> None:
    full, cpp, _ = _lanes("src/ssik/teleop.py", "tests/test_teleop.py", path)
    assert full
    assert cpp


def _ssik_imports(source: Path, module: str) -> set[str]:
    """Every ``ssik`` module ``source`` (module ``module``) imports, at any depth
    in the file (function-level imports included), as dotted names."""
    package = module if source.name == "__init__.py" else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*parent, *([node.module] if node.module else [])])
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return {n for n in names if n == "ssik" or n.startswith("ssik.")}


def test_python_only_modules_are_outside_what_the_cpp_jobs_import() -> None:
    # The C++ jobs run the Python oracle through scripts/cpp_emit.py, which also
    # imports every prebuilt arm by name. Walk the import graph from there (each
    # module's parent packages included) and require that none of the modules the
    # classifier lets skip the C++ jobs is reachable.
    repo = _SCRIPT.parent.parent
    src = repo / "src"

    def file_of(module: str) -> Path | None:
        base = src.joinpath(*module.split("."))
        for candidate in (base / "__init__.py", base.with_suffix(".py")):
            if candidate.is_file():
                return candidate
        return None

    roots = [repo / "scripts" / "cpp_emit.py", *sorted((src / "ssik" / "prebuilt").rglob("*.py"))]
    todo = [m for root in roots for m in _ssik_imports(root, "")]
    reached: set[Path] = set()
    seen: set[str] = set()
    while todo:
        parts = todo.pop().split(".")
        for i in range(1, len(parts) + 1):
            module = ".".join(parts[:i])
            if module in seen:
                continue
            seen.add(module)
            path = file_of(module)
            if path is not None:
                reached.add(path)
                todo.extend(_ssik_imports(path, module))
    assert repo / "src" / "ssik" / "solvers" / "seven_r" / "srs.py" in reached  # the walk works

    spec = importlib.util.spec_from_file_location("classify_changes", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    classify_changes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classify_changes)
    for entry in classify_changes._NOT_CPP_INPUTS:
        listed = repo / entry
        assert listed.exists(), f"{entry} is listed but does not exist"
        hits = sorted(p for p in reached if p == listed or listed in p.parents)
        assert not hits, f"{entry} is listed as outside the C++ jobs, but they import {hits}"
