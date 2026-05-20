"""Manifest cross-validation: every shipped prebuilt artifact has a
manifest entry whose metadata matches the artifact's own bake-time
constants.

Adding a new arm: ``ssik add-arm`` (or its manual equivalent) populates
the manifest. Editing an existing arm: bump the manifest, then re-emit
the artifact so the bake-time constants stay in sync. This test fails
loudly if those two ever drift.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ssik.prebuilt._manifest import Arm, load_manifest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def manifest() -> dict[str, Arm]:
    return load_manifest()


def test_manifest_loads_cleanly(manifest: dict[str, Arm]) -> None:
    """``MANIFEST.toml`` parses without errors and is non-empty."""
    assert manifest, "manifest is empty"
    # Stable iteration order matters: doc generators rely on it for
    # table-row ordering. TOML preserves declaration order; this just
    # documents the invariant.
    names = list(manifest.keys())
    assert len(set(names)) == len(names), "manifest has duplicate arm names"


def test_every_shipped_prebuilt_has_a_manifest_entry(manifest: dict[str, Arm]) -> None:
    """Every ``ssik.prebuilt.<arm>_ik`` module on disk must be in the
    manifest, and vice versa.

    The manifest is the single source of truth -- a prebuilt artifact
    that's missing from it would silently drop out of doc tables, test
    parametrisations, and the bench. A manifest entry without a
    corresponding artifact would attempt to load nothing.
    """
    prebuilt_dir = Path(importlib.import_module("ssik.prebuilt").__file__).parent
    on_disk = {
        p.stem
        for p in prebuilt_dir.glob("*_ik.py")
        if not p.stem.startswith("_") and p.stem != "__init__"
    }
    in_manifest = set(manifest.keys())
    missing_from_manifest = on_disk - in_manifest
    missing_from_disk = in_manifest - on_disk
    assert not missing_from_manifest, (
        f"these prebuilt artifacts are on disk but absent from "
        f"MANIFEST.toml: {sorted(missing_from_manifest)}"
    )
    assert not missing_from_disk, (
        f"these manifest entries have no corresponding prebuilt artifact "
        f"on disk: {sorted(missing_from_disk)} (regenerate with "
        f"`uv run python scripts/regen_artifacts.py` and `--include-slow` "
        f"if any are slow_build = true)"
    )


@pytest.mark.parametrize(
    "arm_name",
    [a for a in load_manifest()],
    ids=list(load_manifest()),
)
def test_manifest_matches_artifact_constants(arm_name: str, manifest: dict[str, Arm]) -> None:
    """Each manifest entry's ``solver`` / ``base_link`` / ``ee_link`` /
    ``dof`` must equal the constants the artifact baked at emit time.

    If they diverge, either the manifest is stale (bump it) or the
    artifact was regenerated with different inputs (re-run ``ssik
    build`` against the manifest's fixture).
    """
    arm = manifest[arm_name]
    mod = importlib.import_module(f"ssik.prebuilt.{arm_name}")
    assert arm.solver == mod.SOLVER_NAME, (
        f"{arm_name}: manifest.solver={arm.solver!r}, artifact.SOLVER_NAME={mod.SOLVER_NAME!r}"
    )
    assert arm.base_link == mod.BASE_LINK, (
        f"{arm_name}: manifest.base_link={arm.base_link!r}, artifact.BASE_LINK={mod.BASE_LINK!r}"
    )
    assert arm.ee_link == mod.EE_LINK, (
        f"{arm_name}: manifest.ee_link={arm.ee_link!r}, artifact.EE_LINK={mod.EE_LINK!r}"
    )
    assert arm.dof == mod.DOF, f"{arm_name}: manifest.dof={arm.dof}, artifact.DOF={mod.DOF}"


@pytest.mark.parametrize(
    "arm_name",
    [a for a in load_manifest()],
    ids=list(load_manifest()),
)
def test_manifest_fixture_file_exists(arm_name: str, manifest: dict[str, Arm]) -> None:
    """The fixture pointed at by each manifest entry must exist."""
    arm = manifest[arm_name]
    if arm.fixture_kind == "urdf":
        path = FIXTURES_DIR / arm.fixture
        assert path.exists(), f"{arm_name}: fixture {path} does not exist"
    else:
        # specs: a Python builder module under tests/fixtures
        path = FIXTURES_DIR / f"{arm.fixture}.py"
        assert path.exists(), f"{arm_name}: fixture {path} does not exist"


@pytest.mark.parametrize(
    "arm_name",
    [a for a in load_manifest()],
    ids=list(load_manifest()),
)
def test_manifest_sample_q_length_matches_dof(arm_name: str, manifest: dict[str, Arm]) -> None:
    """``sample_q`` length must equal ``dof``."""
    arm = manifest[arm_name]
    assert len(arm.sample_q) == arm.dof, (
        f"{arm_name}: sample_q has {len(arm.sample_q)} elements but dof = {arm.dof}"
    )


def test_drift_markers_present_when_platform_drift_true(manifest: dict[str, Arm]) -> None:
    """Arms with ``platform_drift = true`` must declare ``drift_markers``
    so the snapshot test can fall back to structural checks on non-macOS.
    """
    for name, arm in manifest.items():
        if arm.platform_drift:
            assert arm.drift_markers, f"{name}: platform_drift = true but drift_markers is empty"
