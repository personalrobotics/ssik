"""CI's lane and sweep-tier classification fails safe (scripts/classify_changes.py).

The classification decides what a pull request's CI runs, and CI cannot check
its own decision afterwards: a code change classified as examples-only skips
the whole suite, and a change to a sweep's covered code classified as
uncovered runs only the sweep's PR sample. Both pass green while testing less.
So every path that is not plainly examples or docs must choose the full suite,
lookalike paths must not pass as examples or docs, and a sweep may run its
sample only when no changed path is covered by it. No paths (a push, a failed
diff) must mean everything in full.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests import _sweeps

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "classify_changes.py"


def _classify(*paths: str) -> tuple[bool, set[str]]:
    out = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        input="".join(f"{p}\n" for p in paths),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    fields = dict(line.split("=", 1) for line in out.splitlines())
    assert set(fields) == {"full", "pr_sweeps"}
    assert fields["full"] in ("true", "false")
    return fields["full"] == "true", {s for s in fields["pr_sweeps"].split(",") if s}


def test_no_paths_runs_everything_in_full() -> None:
    assert _classify() == (True, set())


@pytest.mark.parametrize(
    "paths",
    [
        ("examples/05_viser_interactive_ik.py",),
        ("examples/README.md", "docs/api.md", "README.md", "LICENSE", ".gitignore"),
    ],
)
def test_examples_and_docs_take_the_examples_lane(paths: tuple[str, ...]) -> None:
    full, _ = _classify(*paths)
    assert not full


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
