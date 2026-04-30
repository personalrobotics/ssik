"""Regenerate the committed reference artifacts under ``tests/artifacts/``.

Run after any change to :mod:`ssik.core.codegen`, :mod:`ssik.core.dispatcher`,
or any solver whose dispatch reasoning text might shift. The companion
snapshot test (``tests/test_artifact_snapshots.py``) re-emits and asserts
byte-equal against the committed file -- if you forget to run this script
the test fails the build and tells you which artifact drifted.

The committed artifacts serve three purposes:

1. **Documentation.** They're the user-facing API surface; reviewers see
   what ``ssik build`` produces without running the CLI.
2. **Regression detection.** Any codegen-touching PR shows an artifact
   diff -- you can scan the diff to confirm the change is intentional.
3. **CI savings (Phase 2).** Once tier-2 RR's symbolic preprocessing is
   baked at build time, the JACO 2 artifact will carry the precompute
   output as data; CI compares against the committed copy instead of
   rerunning the multi-minute sympy step.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ssik._kinbody import build_kinbody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import dispatch

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
ARTIFACTS = REPO_ROOT / "tests" / "artifacts"


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)
    print(f"writing reference artifacts to {ARTIFACTS}/")
    _emit_urdf_artifact(
        urdf=FIXTURES / "ur5.urdf",
        base="base_link",
        ee="ee_link",
        module_name="ur5_ik",
        arm_label="UR5",
    )
    _emit_urdf_artifact(
        urdf=FIXTURES / "puma560.urdf",
        base="base_link",
        ee="wrist_3_link",
        module_name="puma560_ik",
        arm_label="Puma 560",
    )
    _emit_jaco2_artifact()
    _emit_franka_panda_artifact()
    print("done.")
    return 0


def _emit_urdf_artifact(
    *, urdf: Path, base: str, ee: str, module_name: str, arm_label: str
) -> None:
    kb = load_urdf_kinbody_normalized(urdf, base, ee)
    plan = dispatch(kb)
    out = ARTIFACTS / f"{module_name}.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name=module_name,
        output_path=str(out),
        arm_label=arm_label,
    )
    print(f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} (tier {plan.tier})")


def _emit_jaco2_artifact() -> None:
    """JACO 2 fixture: real MJCF transcription. Lives under tests/fixtures
    as a Python builder (jaco2.py) rather than a URDF file."""
    sys.path.insert(0, str(FIXTURES))
    from jaco2 import jaco2_specs

    kb = build_kinbody(jaco2_specs())
    plan = dispatch(kb)
    out = ARTIFACTS / "jaco2_ik.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name="jaco2_ik",
        output_path=str(out),
        arm_label="Kinova JACO 2 (j2n6s200)",
    )
    print(f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} (tier {plan.tier})")


def _emit_franka_panda_artifact() -> None:
    """Franka Panda fixture: real MJCF transcription, 7-DOF. The artifact
    routes through ``jointlock.seven_r`` which auto-picks lock_idx=4
    (matching EAIK) and dispatches to ``reversed:spherical_two_parallel``
    via the chain-reversal pre-pass."""
    sys.path.insert(0, str(FIXTURES))
    from franka_panda import franka_panda_specs

    kb = build_kinbody(franka_panda_specs())
    plan = dispatch(kb)
    out = ARTIFACTS / "franka_panda_ik.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name="franka_panda_ik",
        output_path=str(out),
        arm_label="Franka Emika Panda (no hand)",
    )
    print(f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} (tier {plan.tier})")


if __name__ == "__main__":
    sys.exit(main())
