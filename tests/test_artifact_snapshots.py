"""Snapshot tests for committed reference artifacts under src/ssik/prebuilt/.

Each fixture arm has a committed ``<arm>_ik.py`` artifact. This test re-emits
each one in-memory and asserts byte-equal against the committed file.

If you change :mod:`ssik.core.codegen` or :mod:`ssik.core.dispatcher` in a
way that affects rendered output (e.g. tweak the dispatch ``reason`` text),
this test fails until you regenerate:

    uv run python scripts/regen_artifacts.py

Then commit the updated ``src/ssik/prebuilt/*.py`` files alongside your
codegen change. The artifact diff is signal, not noise: it shows reviewers
exactly what user-facing output the change produces.

Tier-0 artifacts (UR5, Puma 560) snapshot byte-identically on every platform
because their composers don't run anything through ``sympy.cse`` -- they bake
DH literals into a fixed code template.

JACO 2 (tier-2 RR) is a known platform-specific exception. After
:mod:`ssik.kinematics.poe_to_dh` was made bit-deterministic in #123, the
input DH params to the sympy pipeline are identical across macOS / Linux,
but ``sympy.Poly`` + ``sympy.cse`` + ``sympy.pycode`` together still produce
last-digit-different float literals across platforms in the rendered output
(the underlying float64 values diverge inside sympy's internal arithmetic,
not just at the printing layer). Pinning sympy's bit-level determinism is
deferred to a follow-up; for now the platform-drift artifacts snapshot
enforces byte equality only on macOS (the regen platform), with a structural
smoke check on others.

The parametrisation is driven by ``MANIFEST.toml`` via the loader at
:mod:`ssik.prebuilt._manifest`. ``platform_drift`` and ``drift_markers``
on the manifest entry drive the byte-vs-structural choice.

Slow-build arms (Rizon 4 ~7 min, Kassow KR810 ~20 min, Rizon 10 ~7 min)
are excluded from this snapshot test -- regenerating them in-memory per
test would dominate CI time. They're validated via the uniform fuzz +
sanity tests instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ssik._kinbody import build_kinbody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import dispatch
from ssik.prebuilt._manifest import Arm, load_manifest

FIXTURES = Path(__file__).parent / "fixtures"
ARTIFACTS = Path(__file__).parent.parent / "src" / "ssik" / "prebuilt"

sys.path.insert(0, str(FIXTURES))

_MANIFEST = load_manifest()
# Slow-build arms are excluded from the snapshot test; their artifacts
# take 7-20 minutes to regenerate which would dominate CI time.
_SNAPSHOT_ARMS = [arm.name for arm in _MANIFEST.values() if not arm.slow_build]


def _emit(arm: Arm) -> str:
    """Re-emit the artifact for ``arm`` and return its source code."""
    if arm.fixture_kind == "urdf":
        kb = load_urdf_kinbody_normalized(FIXTURES / arm.fixture, arm.base_link, arm.ee_link)
    else:
        # specs: a Python builder module under tests/fixtures. Pass
        # base_link_name + ee_link_name kwargs so the emitted artifact
        # matches the manifest's declared link names. (Only meaningful
        # when the manifest declares non-default names; harmless when
        # they're the defaults.)
        mod = __import__(arm.fixture)
        specs_fn_name = arm.specs_fn
        assert specs_fn_name is not None  # invariant per manifest schema
        kb = build_kinbody(
            getattr(mod, specs_fn_name)(),
            base_link_name=arm.base_link,
            ee_link_name=arm.ee_link,
        )
    plan = dispatch(kb)
    result = emit_artifact(
        kb=kb,
        plan=plan,
        module_name=arm.name,
        output_path=None,
        arm_label=arm.display_name,
    )
    return result.source


@pytest.mark.parametrize(
    "arm_name",
    _SNAPSHOT_ARMS,
    ids=_SNAPSHOT_ARMS,
)
def test_committed_artifact_matches_regeneration(arm_name: str) -> None:
    """Re-emit + byte-compare. Any drift fails the test with a unified-diff
    of the divergence and a pointer to the regen script.

    Manifest entries with ``platform_drift = true`` (jaco2_ik, gen3_ik,
    xarm7_ik, xarm6_ik, piper_ik) enforce byte-equality only on macOS
    (the regen platform); on other platforms the test runs a structural
    smoke check using the manifest's ``drift_markers``. See module
    docstring; full sympy determinism is a follow-up (#124).
    """
    arm = _MANIFEST[arm_name]
    rendered = _emit(arm)
    committed_path = ARTIFACTS / f"{arm_name}.py"
    assert committed_path.exists(), (
        f"committed artifact {committed_path.relative_to(Path(__file__).parent.parent)} "
        f"is missing -- run `uv run python scripts/regen_artifacts.py` to create it."
    )

    if arm.platform_drift and sys.platform != "darwin":
        # Structural smoke check on platforms whose float repr differs from
        # macOS's. Each ``drift_marker`` must appear verbatim in the
        # regenerated source.
        for marker in arm.drift_markers:
            assert marker in rendered, f"{arm_name}: missing structural marker {marker!r}"
        return

    committed = committed_path.read_text()
    if rendered != committed:
        import difflib

        diff = "".join(
            difflib.unified_diff(
                committed.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=f"committed/{committed_path.name}",
                tofile=f"regenerated/{committed_path.name}",
                n=3,
            )
        )
        # Cap diff length so a runaway divergence doesn't drown the CI log.
        if len(diff) > 4000:
            diff = diff[:4000] + "\n... (truncated)\n"
        pytest.fail(
            f"committed artifact {committed_path.name} differs from regenerated "
            f"output. The codegen module produced different bytes -- this is "
            f"likely an intentional codegen change. Regenerate with:\n"
            f"    uv run python scripts/regen_artifacts.py\n"
            f"and commit the updated src/ssik/prebuilt/*.py alongside your codegen "
            f"change so reviewers can see the user-facing impact.\n\n"
            f"Diff (committed -> regenerated):\n{diff}"
        )
