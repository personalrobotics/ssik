"""Regenerate the committed reference artifacts under ``src/ssik/prebuilt/``.

Run after any change to :mod:`ssik.core.codegen`, :mod:`ssik.core.dispatcher`,
or any solver whose dispatch reasoning text might shift. The companion
snapshot test (``tests/test_artifact_snapshots.py``) re-emits and asserts
byte-equal against the committed file -- if you forget to run this script
the test fails the build and tells you which artifact drifted.

The committed artifacts under ``src/ssik/prebuilt/`` serve three purposes:

1. **User-facing demos.** Users can ``from ssik.prebuilt import ur5_ik``
   and immediately get a working IK solver without running ``ssik build``.
2. **Documentation.** Reviewers see what ``ssik build`` produces without
   running the CLI.
3. **Regression detection.** Any codegen-touching PR shows an artifact
   diff -- you can scan the diff to confirm the change is intentional.

Slow arms (Rizon 4 ~7 min, Kassow KR810 ~20 min) are gated behind
``--include-slow`` so the default regen is fast (<30 s for tier-0 / SRS).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ssik._kinbody import build_kinbody
from ssik._urdf import load_urdf_kinbody_normalized
from ssik.core.codegen import emit_artifact
from ssik.core.dispatcher import dispatch

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
ARTIFACTS = REPO_ROOT / "src" / "ssik" / "prebuilt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Also rebuild Rizon 4 (~7 min) and Kassow KR810 (~20 min).",
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    print(f"writing reference artifacts to {ARTIFACTS}/")

    # Fast tier-0 / tier-0-7R artifacts (<30s each).
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
    _emit_iiwa14_artifact()
    _emit_urdf_artifact(
        urdf=FIXTURES / "gen3.urdf",
        base="base_link",
        ee="end_effector_link",
        module_name="gen3_ik",
        arm_label="Kinova Gen3 (7-DOF)",
    )
    _emit_jaco2_artifact()
    _emit_franka_panda_artifact()

    if args.include_slow:
        # Slow non-SRS 7R artifacts: cached-RR symbolic derivations baked
        # into the artifact (#220). Build cost amortises across the
        # deployment lifetime.
        _emit_urdf_artifact(
            urdf=FIXTURES / "rizon4.urdf",
            base="base_link",
            ee="flange",
            module_name="rizon4_ik",
            arm_label="Flexiv Rizon 4",
        )
        _emit_urdf_artifact(
            urdf=FIXTURES / "kassow_kr810.urdf",
            base="base",
            ee="end_effector",
            module_name="kassow_kr810_ik",
            arm_label="Kassow KR810",
        )

    print("done.")
    return 0


def _emit_urdf_artifact(
    *, urdf: Path, base: str, ee: str, module_name: str, arm_label: str
) -> None:
    t = time.perf_counter()
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
    elapsed = time.perf_counter() - t
    size_kb = out.stat().st_size / 1024
    print(
        f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} "
        f"(tier {plan.tier}, {size_kb:.1f} KB, {elapsed:.1f}s)"
    )


def _emit_iiwa14_artifact() -> None:
    """KUKA iiwa LBR 14 fixture: SRS-class 7R, lives as a Python builder
    (kuka_iiwa14.py) rather than a URDF file."""
    sys.path.insert(0, str(FIXTURES))
    from kuka_iiwa14 import kuka_iiwa14_specs

    t = time.perf_counter()
    kb = build_kinbody(kuka_iiwa14_specs())
    plan = dispatch(kb)
    out = ARTIFACTS / "iiwa14_ik.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name="iiwa14_ik",
        output_path=str(out),
        arm_label="KUKA iiwa LBR 14",
    )
    elapsed = time.perf_counter() - t
    size_kb = out.stat().st_size / 1024
    print(
        f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} "
        f"(tier {plan.tier}, {size_kb:.1f} KB, {elapsed:.1f}s)"
    )


def _emit_jaco2_artifact() -> None:
    """JACO 2 fixture: real MJCF transcription. Lives under tests/fixtures
    as a Python builder (jaco2.py) rather than a URDF file."""
    sys.path.insert(0, str(FIXTURES))
    from jaco2 import jaco2_specs

    t = time.perf_counter()
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
    elapsed = time.perf_counter() - t
    size_kb = out.stat().st_size / 1024
    print(
        f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} "
        f"(tier {plan.tier}, {size_kb:.1f} KB, {elapsed:.1f}s)"
    )


def _emit_franka_panda_artifact() -> None:
    """Franka Panda fixture: real MJCF transcription, 7-DOF. The artifact
    routes through ``jointlock.seven_r`` which auto-picks lock_idx=4
    (matching EAIK) and dispatches to ``reversed:spherical_two_parallel``
    via the chain-reversal pre-pass."""
    sys.path.insert(0, str(FIXTURES))
    from franka_panda import franka_panda_specs

    t = time.perf_counter()
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
    elapsed = time.perf_counter() - t
    size_kb = out.stat().st_size / 1024
    print(
        f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} "
        f"(tier {plan.tier}, {size_kb:.1f} KB, {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    sys.exit(main())
