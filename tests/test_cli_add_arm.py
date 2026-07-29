"""End-to-end tests for the ``ssik add-arm`` subcommand (#196).

The CLI takes a URDF + base/ee link names + an arm name, vendors the
URDF into ``tests/fixtures/`` and emits a per-arm bulletproof test
scaffold. The generated tests assert URDF load + dispatcher routing
+ hand-picked / random FK closure on every retained IK.

Test contract:

- The generated test file is valid Python and pytest-collectable.
- The generated tests pass when run against the real fixture URDF.
- Reusing an existing name without ``--force`` errors out cleanly.
- Solver-specific scaffolds work for every dispatch class we care
  about (jointlock, srs, srs_polished, ikgeo.* — exercised on the
  fixtures we already have).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Mock repository root with a populated ``tests/fixtures`` dir.

    The CLI writes to ``<repo-root>/tests/fixtures/`` and
    ``<repo-root>/tests/``, so we need both directories. We create
    them and return the workspace root.
    """
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    return tmp_path


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``ssik`` as a subprocess. Returns the completed-process result."""
    return subprocess.run(
        [sys.executable, "-m", "ssik.cli", *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_add_arm_builds_and_validates(workspace: Path) -> None:
    """Default (no ``--no-validate``): add-arm builds the emitted artifact and
    runs the coverage gate locally, reporting a verdict, and the printed stanza
    carries derived ``fk_ceiling_fuzz`` / ``platform_drift`` (not placeholders).
    Uses UR5 (fast three-parallel build)."""
    result = _run_cli(
        "add-arm",
        str(REPO_ROOT / "tests" / "fixtures" / "ur5.urdf"),
        "--base",
        "base_link",
        "--ee",
        "ee_link",
        "--name",
        "ur5_validate_test",
        "--vendor",
        "universal_robots",
        "--repo-root",
        str(workspace),
    )
    assert result.returncode == 0, result.stderr
    assert "VALIDATED" in result.stdout, f"no validation verdict:\n{result.stdout}"
    assert "platform_drift = false" in result.stdout
    # fk_ceiling came from the smoke, not the 1e-4 placeholder.
    assert "fk_ceiling_fuzz = 1e-04" not in result.stdout


def test_add_arm_derives_class_metadata(workspace: Path) -> None:
    """The printed stanza derives kinematic_class / short_class / class_tags from
    the dispatched solver (not TODO), a verified-solvable sample_q (not the
    near-home [0.1, ...] placeholder), and a best-effort display_name -- so the
    only hand-fill left is the fixture_source provenance. UR5 -> three_parallel."""
    result = _run_cli(
        "add-arm",
        "--no-validate",
        str(REPO_ROOT / "tests" / "fixtures" / "ur5.urdf"),
        "--base",
        "base_link",
        "--ee",
        "ee_link",
        "--name",
        "ur5_class_test",
        "--vendor",
        "universal_robots",
        "--repo-root",
        str(workspace),
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert 'kinematic_class = "three-parallel 6R (UR-class)"' in out
    assert 'class_tags = ["three-parallel", "6R", "Pieper"]' in out
    assert 'kinematic_class = "TODO"' not in out
    assert 'class_tags = ["TODO"]' not in out
    # sample_q was replaced by a verified pose, not the [0.1, 0.1, ...] default.
    assert "sample_q = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]" not in out
    assert 'display_name = "Universal Robots' in out


def test_add_arm_generates_files(workspace: Path) -> None:
    """``ssik add-arm`` vendors the URDF and emits a test scaffold."""
    result = _run_cli(
        "add-arm",
        "--no-validate",
        str(REPO_ROOT / "tests" / "fixtures" / "rizon4.urdf"),
        "--base",
        "base_link",
        "--ee",
        "flange",
        "--name",
        "rizon4_addarm_test",
        "--vendor",
        "flexiv",
        "--repo-root",
        str(workspace),
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"

    urdf_dest = workspace / "tests" / "fixtures" / "rizon4_addarm_test.urdf"
    test_dest = workspace / "tests" / "test_rizon4_addarm_test.py"
    assert urdf_dest.exists(), "URDF not vendored"
    assert test_dest.exists(), "test scaffold not written"
    # Vendored URDF is stripped to kinematics-only (no mesh/visual/collision)
    # but FK-identical to the source (#341).
    text = urdf_dest.read_text()
    assert "package://" not in text
    assert "<visual" not in text
    assert "<collision" not in text
    import numpy as np

    from ssik._urdf import load_urdf_kinbody_normalized
    from ssik.kinematics.poe_fk import poe_forward_kinematics

    src = REPO_ROOT / "tests" / "fixtures" / "rizon4.urdf"
    kb_vendored = load_urdf_kinbody_normalized(urdf_dest, "base_link", "flange")
    kb_src = load_urdf_kinbody_normalized(src, "base_link", "flange")
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1.0, 1.0, size=len(kb_src.joints))
        drift = np.abs(poe_forward_kinematics(kb_vendored, q) - poe_forward_kinematics(kb_src, q))
        assert drift.max() < 1e-12, f"stripped URDF FK drift {drift.max():.2e}"


def test_add_arm_refuses_overwrite_without_force(workspace: Path) -> None:
    """If the fixture already exists, the CLI refuses unless ``--force``."""
    target = workspace / "tests" / "fixtures" / "preexisting_arm.urdf"
    target.write_text("<robot/>")
    result = _run_cli(
        "add-arm",
        "--no-validate",
        str(REPO_ROOT / "tests" / "fixtures" / "rizon4.urdf"),
        "--base",
        "base_link",
        "--ee",
        "flange",
        "--name",
        "preexisting_arm",
        "--vendor",
        "flexiv",
        "--repo-root",
        str(workspace),
    )
    assert result.returncode != 0
    assert "already exists" in result.stdout
    # File contents preserved.
    assert target.read_text() == "<robot/>"


def test_add_arm_force_overwrites(workspace: Path) -> None:
    """``--force`` overwrites the existing fixture."""
    target_urdf = workspace / "tests" / "fixtures" / "force_test.urdf"
    target_test = workspace / "tests" / "test_force_test.py"
    target_urdf.write_text("<robot/>")
    target_test.write_text("# stale")
    result = _run_cli(
        "add-arm",
        "--no-validate",
        str(REPO_ROOT / "tests" / "fixtures" / "rizon4.urdf"),
        "--base",
        "base_link",
        "--ee",
        "flange",
        "--name",
        "force_test",
        "--vendor",
        "flexiv",
        "--repo-root",
        str(workspace),
        "--force",
    )
    assert result.returncode == 0
    # Fresh contents (URDF byte-equal to Rizon source; test file no longer "# stale").
    assert "<robot/>" not in target_urdf.read_text()
    assert "# stale" not in target_test.read_text()


def test_add_arm_generated_test_is_valid_python(workspace: Path) -> None:
    """The generated test file imports cleanly (compile check)."""
    result = _run_cli(
        "add-arm",
        "--no-validate",
        str(REPO_ROOT / "tests" / "fixtures" / "rizon4.urdf"),
        "--base",
        "base_link",
        "--ee",
        "flange",
        "--name",
        "compile_test",
        "--vendor",
        "flexiv",
        "--repo-root",
        str(workspace),
    )
    assert result.returncode == 0
    test_dest = workspace / "tests" / "test_compile_test.py"
    source = test_dest.read_text()
    # Plain compile: catches any syntax errors in the f-string templating.
    compile(source, str(test_dest), "exec")


def test_add_arm_generated_test_runs_passes(workspace: Path, monkeypatch) -> None:
    """The generated test scaffold actually passes on the real Rizon URDF.

    The scaffold needs the URDF in <repo-root>/tests/fixtures/ to load.
    We copy the test file into the real repo-root (with a unique name to
    avoid collisions) and run pytest against it.
    """
    # Generate into the real repo (so URDF_PATH resolves).
    name = "addarm_runtest"
    result = _run_cli(
        "add-arm",
        "--no-validate",
        str(REPO_ROOT / "tests" / "fixtures" / "rizon4.urdf"),
        "--base",
        "base_link",
        "--ee",
        "flange",
        "--name",
        name,
        "--vendor",
        "flexiv",
        "--repo-root",
        str(REPO_ROOT),
        "--force",
    )
    assert result.returncode == 0, result.stdout

    test_dest = REPO_ROOT / "tests" / f"test_{name}.py"
    urdf_dest = REPO_ROOT / "tests" / "fixtures" / f"{name}.urdf"
    try:
        # Run only the fast tests (URDF load + dispatch routing).
        run = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dest), "-v"],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        # Should pass.
        assert run.returncode == 0, f"pytest failed:\n{run.stdout}\n{run.stderr}"
    finally:
        # Always clean up the real-repo files we generated.
        if test_dest.exists():
            test_dest.unlink()
        if urdf_dest.exists():
            urdf_dest.unlink()


def _seed_manifest(workspace: Path) -> Path:
    """Create a minimal MANIFEST.toml under the workspace repo-root."""
    mf = workspace / "src" / "ssik" / "prebuilt" / "MANIFEST.toml"
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(
        "# manifest\n\n[arms.existing_ik]\ndof = 6\n\n[arms.existing_ik.bench]\nms_mean = 0.1\n",
        encoding="utf-8",
    )
    return mf


def test_add_arm_write_manifest_appends_stanza(workspace: Path) -> None:
    """``--write-manifest`` appends the derived stanza to MANIFEST.toml (valid
    TOML, existing entries preserved)."""
    import tomllib

    mf = _seed_manifest(workspace)
    result = _run_cli(
        "add-arm",
        "--no-validate",
        "--write-manifest",
        str(REPO_ROOT / "tests" / "fixtures" / "ur5.urdf"),
        "--base",
        "base_link",
        "--ee",
        "ee_link",
        "--name",
        "ur5_wm_test",
        "--vendor",
        "universal_robots",
        "--repo-root",
        str(workspace),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Appended [arms.ur5_wm_test]" in result.stdout
    data = tomllib.loads(mf.read_text())
    assert "ur5_wm_test" in data["arms"], "new stanza not written"
    assert "existing_ik" in data["arms"], "existing entry clobbered"
    assert data["arms"]["ur5_wm_test"]["dof"] == 6


def test_add_arm_write_manifest_refuses_duplicate(workspace: Path) -> None:
    """``--write-manifest`` refuses to append a second entry for the same arm
    without ``--force`` (and doesn't duplicate)."""
    import tomllib

    mf = _seed_manifest(workspace)
    args = [
        "add-arm",
        "--no-validate",
        "--write-manifest",
        str(REPO_ROOT / "tests" / "fixtures" / "ur5.urdf"),
        "--base",
        "base_link",
        "--ee",
        "ee_link",
        "--name",
        "ur5_dup_test",
        "--vendor",
        "universal_robots",
        "--repo-root",
        str(workspace),
    ]
    assert _run_cli(*args).returncode == 0
    # Second time: fixture/test exist, so --force is needed for those; but the
    # manifest refusal is the point -- add --force so we reach the manifest step
    # and confirm the entry is replaced (not duplicated).
    second = _run_cli(*args, "--force")
    assert second.returncode == 0
    text = mf.read_text()
    # Exactly one header table (with closing bracket; the .bench sub-table has a dot).
    assert text.count("[arms.ur5_dup_test]") == 1, "force re-add duplicated the entry"
    tomllib.loads(text)  # still valid TOML (raises on duplicate keys)
    # And without --force on an existing manifest entry, it errors.
    third = _run_cli(*args)
    assert third.returncode != 0
    assert "already exists" in third.stdout


def test_add_arm_no_validate_runs_live_smoke(workspace: Path) -> None:
    """``--no-validate`` no longer silently skips onboarding validation (#445):
    it runs a bounded live-solver smoke and prints a loud UNVALIDATED reminder
    with the build + full-gate commands, so a broken slow-build arm can't
    scaffold green unnoticed. Uses UR5 (fast) to exercise the path."""
    result = _run_cli(
        "add-arm",
        "--no-validate",
        str(REPO_ROOT / "tests" / "fixtures" / "ur5.urdf"),
        "--base",
        "base_link",
        "--ee",
        "ee_link",
        "--name",
        "ur5_smoke_test",
        "--vendor",
        "universal_robots",
        "--repo-root",
        str(workspace),
    )
    assert result.returncode == 0, result.stderr
    assert "live-solver smoke" in result.stdout
    assert "8/8 poses solved" in result.stdout  # UR5 is fully covered
    assert "UNVALIDATED" in result.stdout
    assert "regen_artifacts.py --arm ur5_smoke_test" in result.stdout


def test_coverage_band_thresholds() -> None:
    """add-arm classifies in-limits coverage into fail / gap / clean bands so a
    >1% in-limits coverage gap (near-singular swivel-vs-limits, #462) is warned,
    not silently passed as clean."""
    from ssik.cli import _coverage_band

    assert _coverage_band(0.80) == "fail"  # broken arm (0 sols / locks / wrong ee)
    assert _coverage_band(0.95) == "gap"  # ~5% gap -> warn + tracked xfail
    assert _coverage_band(0.985) == "gap"  # >1% gap -> warn
    assert _coverage_band(0.995) == "clean"  # <1% gap
    assert _coverage_band(1.0) == "clean"


def test_add_arm_missing_urdf_errors_cleanly(workspace: Path) -> None:
    """Pointing at a non-existent URDF errors instead of silently writing."""
    result = _run_cli(
        "add-arm",
        "--no-validate",
        str(workspace / "nonexistent.urdf"),
        "--base",
        "base",
        "--ee",
        "ee",
        "--name",
        "ghost_arm",
        "--vendor",
        "flexiv",
        "--repo-root",
        str(workspace),
    )
    assert result.returncode != 0
    assert "not found" in result.stdout.lower() or "not found" in result.stderr.lower()
