"""End-to-end tests for :mod:`ssik.cli` -- the ``ssik`` console script.

Exercises both ``ssik classify`` and ``ssik build`` via the in-process
``main()`` entry point so we get a fast loop without spawning subprocesses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ssik.cli import main as cli_main

FIXTURES = Path(__file__).parent / "fixtures"


def test_classify_ur5(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(
        [
            "classify",
            str(FIXTURES / "ur5.urdf"),
            "--base",
            "base_link",
            "--ee",
            "ee_link",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "ikgeo.three_parallel" in out
    assert "tier 0" in out
    assert "Three consecutive parallel" in out


def test_classify_puma(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli_main(
        [
            "classify",
            str(FIXTURES / "puma560.urdf"),
            "--base",
            "base_link",
            "--ee",
            "wrist_3_link",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "ikgeo.spherical_two_parallel" in out
    assert "tier 0" in out


def test_build_ur5_emits_and_validates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = tmp_path / "ur5_ik_cli.py"
    rc = cli_main(
        [
            "build",
            str(FIXTURES / "ur5.urdf"),
            "--base",
            "base_link",
            "--ee",
            "ee_link",
            "--out",
            str(artifact),
            "--module-name",
            "ur5_ik_cli",
            "--validate-poses",
            "10",
        ]
    )
    assert rc == 0
    assert artifact.exists()
    out = capsys.readouterr().out
    assert "Wrote" in out
    assert "0 failures" in out
    assert "ikgeo.three_parallel" in out

    # Import + smoke-solve to confirm the artifact is functional from disk.
    spec = importlib.util.spec_from_file_location("ur5_ik_cli", artifact)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ur5_ik_cli"] = mod
    spec.loader.exec_module(mod)
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.1, 0.3]
    sols = mod.solve(T)
    # Random free pose may or may not have a solution; just assert callable.
    assert isinstance(sols, list)


def test_build_no_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = tmp_path / "ur5_skipval.py"
    rc = cli_main(
        [
            "build",
            str(FIXTURES / "ur5.urdf"),
            "--base",
            "base_link",
            "--ee",
            "ee_link",
            "--out",
            str(artifact),
            "--module-name",
            "ur5_skipval",
            "--no-validate",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Validation skipped" in out
    assert artifact.exists()


def test_build_default_module_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``--module-name`` is omitted, the artifact is named after the
    URDF stem (e.g. ``ur5.urdf`` → ``ur5_ik``)."""
    monkeypatch.chdir(tmp_path)
    rc = cli_main(
        [
            "build",
            str(FIXTURES / "ur5.urdf"),
            "--base",
            "base_link",
            "--ee",
            "ee_link",
            "--validate-poses",
            "5",
        ]
    )
    assert rc == 0
    assert (tmp_path / "ur5_ik.py").exists()
