"""Snapshot tests for committed reference artifacts under tests/artifacts/.

Each fixture arm has a committed ``<arm>_ik.py`` artifact. This test re-emits
each one in-memory and asserts byte-equal against the committed file.

If you change :mod:`ssik.core.codegen` or :mod:`ssik.core.dispatcher` in a
way that affects rendered output (e.g. tweak the dispatch ``reason`` text),
this test fails until you regenerate:

    uv run python scripts/regen_artifacts.py

Then commit the updated ``tests/artifacts/*.py`` files alongside your codegen
change. The artifact diff is signal, not noise: it shows reviewers exactly
what user-facing output the change produces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ssik._kinbody import build_kinbody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import dispatch

FIXTURES = Path(__file__).parent / "fixtures"
ARTIFACTS = Path(__file__).parent / "artifacts"

sys.path.insert(0, str(FIXTURES))
from jaco2 import jaco2_specs  # noqa: E402


def _emit_urdf(urdf: str, base: str, ee: str, module_name: str, arm_label: str) -> str:
    kb = load_urdf_kinbody_normalized(FIXTURES / urdf, base, ee)
    plan = dispatch(kb)
    result = emit_artifact(
        kb=kb,
        plan=plan,
        module_name=module_name,
        output_path=None,
        arm_label=arm_label,
    )
    return result.source


def _emit_jaco2() -> str:
    kb = build_kinbody(jaco2_specs())
    plan = dispatch(kb)
    result = emit_artifact(
        kb=kb,
        plan=plan,
        module_name="jaco2_ik",
        output_path=None,
        arm_label="Kinova JACO 2 (j2n6s200)",
    )
    return result.source


@pytest.mark.parametrize(
    ("module_name", "emit_fn"),
    [
        (
            "ur5_ik",
            lambda: _emit_urdf("ur5.urdf", "base_link", "ee_link", "ur5_ik", "UR5"),
        ),
        (
            "puma560_ik",
            lambda: _emit_urdf(
                "puma560.urdf",
                "base_link",
                "wrist_3_link",
                "puma560_ik",
                "Puma 560",
            ),
        ),
        ("jaco2_ik", _emit_jaco2),
    ],
)
def test_committed_artifact_matches_regeneration(module_name: str, emit_fn: object) -> None:
    """Re-emit + byte-compare. Any drift fails the test with a unified-diff
    of the divergence and a pointer to the regen script."""
    rendered = emit_fn()  # type: ignore[operator]
    committed_path = ARTIFACTS / f"{module_name}.py"
    assert committed_path.exists(), (
        f"committed artifact {committed_path.relative_to(Path(__file__).parent.parent)} "
        f"is missing -- run `uv run python scripts/regen_artifacts.py` to create it."
    )
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
            f"and commit the updated tests/artifacts/*.py alongside your codegen "
            f"change so reviewers can see the user-facing impact.\n\n"
            f"Diff (committed -> regenerated):\n{diff}"
        )
