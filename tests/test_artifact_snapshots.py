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
deferred to a follow-up; for now the JACO 2 snapshot enforces byte equality
only on macOS (the regen platform), with a structural smoke check on others.
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
ARTIFACTS = Path(__file__).parent.parent / "src" / "ssik" / "prebuilt"

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


def _emit_franka_panda() -> str:
    from franka_panda import franka_panda_specs

    kb = build_kinbody(franka_panda_specs())
    plan = dispatch(kb)
    result = emit_artifact(
        kb=kb,
        plan=plan,
        module_name="franka_panda_ik",
        output_path=None,
        arm_label="Franka Emika Panda (no hand)",
    )
    return result.source


def _emit_iiwa14() -> str:
    from kuka_iiwa14 import kuka_iiwa14_specs

    kb = build_kinbody(kuka_iiwa14_specs())
    plan = dispatch(kb)
    result = emit_artifact(
        kb=kb,
        plan=plan,
        module_name="iiwa14_ik",
        output_path=None,
        arm_label="KUKA iiwa LBR 14",
    )
    return result.source


def _emit_xarm7() -> str:
    from xarm7 import xarm7_specs

    kb = build_kinbody(xarm7_specs(), base_link_name="link_base", ee_link_name="link7")
    plan = dispatch(kb)
    result = emit_artifact(
        kb=kb,
        plan=plan,
        module_name="xarm7_ik",
        output_path=None,
        arm_label="UFactory xArm7",
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
        ("franka_panda_ik", _emit_franka_panda),
        ("iiwa14_ik", _emit_iiwa14),
        ("xarm7_ik", _emit_xarm7),
        (
            "gen3_ik",
            lambda: _emit_urdf(
                "gen3.urdf",
                "base_link",
                "end_effector_link",
                "gen3_ik",
                "Kinova Gen3 (7-DOF)",
            ),
        ),
        (
            "xarm6_ik",
            lambda: _emit_urdf(
                "xarm6.urdf",
                "link_base",
                "link_eef",
                "xarm6_ik",
                "UFactory xArm6",
            ),
        ),
        (
            "z1_ik",
            lambda: _emit_urdf(
                "z1.urdf",
                "link00",
                "link06",
                "z1_ik",
                "Unitree Z1",
            ),
        ),
    ],
)
def test_committed_artifact_matches_regeneration(module_name: str, emit_fn: object) -> None:
    """Re-emit + byte-compare. Any drift fails the test with a unified-diff
    of the divergence and a pointer to the regen script.

    JACO 2 byte-equality enforced only on macOS (the regen platform); on
    other platforms a structural smoke check runs instead, because sympy's
    internal arithmetic produces last-digit-different float literals
    across platforms even with deterministic input DH params. See module
    docstring; full sympy determinism is a follow-up.
    """
    rendered = emit_fn()  # type: ignore[operator]
    committed_path = ARTIFACTS / f"{module_name}.py"
    assert committed_path.exists(), (
        f"committed artifact {committed_path.relative_to(Path(__file__).parent.parent)} "
        f"is missing -- run `uv run python scripts/regen_artifacts.py` to create it."
    )

    # Artifacts whose bytes drift across platforms by a last-digit float
    # representation. JACO 2 (tier-2 RR): drift comes from sympy.cse /
    # sympy.pycode internals. Gen3 (SRS-polished thin wrapper): drift
    # comes from the URDF -> KinBody float arithmetic (urchin / numpy).
    # Pinning full bit-determinism is deferred to a follow-up (#124).
    # macOS is the regen platform; enforce byte-equality there, run a
    # structural smoke check elsewhere.
    _PLATFORM_DRIFT_ARTIFACTS = {
        "jaco2_ik": [
            "_solve_algebraic",
            "_build_pq_matrices",
            'SOLVER_NAME = "ikgeo.general_6r"',
        ],
        "gen3_ik": [
            "_solver_solve",
            'SOLVER_NAME = "seven_r.srs_polished"',
            "def fk(q):",
        ],
        # xArm7: same class of drift as gen3 -- the xarm7 fixture's
        # quaternion-to-rotation conversion (np.sqrt + division) produces
        # last-bit-different float64 values on macOS Accelerate vs Linux
        # OpenBLAS, which propagates through poe-normalisation into the
        # baked T_HOME / T_left arrays. Functionally equivalent (FK
        # round-trips at 1e-13 on both platforms); only the float repr
        # differs by one trailing digit.
        "xarm7_ik": [
            'SOLVER_NAME = "jointlock.seven_r"',
            'BASE_LINK = "link_base"',
            'EE_LINK = "link7"',
            "DOF = 7",
            "def solve(",
        ],
        # xArm6: tier-2 RR same as jaco2 -- sympy.cse / sympy.pycode
        # float-repr drift across macOS Accelerate vs Linux OpenBLAS.
        "xarm6_ik": [
            "_solve_algebraic",
            "_build_pq_matrices",
            'SOLVER_NAME = "ikgeo.general_6r"',
        ],
    }
    if module_name in _PLATFORM_DRIFT_ARTIFACTS and sys.platform != "darwin":
        for marker in _PLATFORM_DRIFT_ARTIFACTS[module_name]:
            assert marker in rendered, f"{module_name}: missing structural marker {marker!r}"
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
