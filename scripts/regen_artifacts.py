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

The per-arm list is read from ``src/ssik/prebuilt/MANIFEST.toml`` via the
loader at :mod:`ssik.prebuilt._manifest`. Adding a new arm therefore
needs no edits in this script -- populate the manifest (manually or via
``ssik add-arm``) and the regenerator picks it up automatically.

Slow arms (``slow_build = true`` in the manifest: Rizon 4 ~7 min, Kassow
KR810 ~20 min, Rizon 10 ~7 min) are gated behind ``--include-slow`` so
the default regen is fast (<30 s for the fast-build set).
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
from ssik.prebuilt._manifest import Arm, load_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
ARTIFACTS = REPO_ROOT / "src" / "ssik" / "prebuilt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help=(
            "Also rebuild arms with ``slow_build = true`` in the manifest "
            "(Rizon 4 / Rizon 10 ~7 min each, Kassow KR810 ~20 min)."
        ),
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    print(f"writing reference artifacts to {ARTIFACTS}/")

    manifest = load_manifest()
    sys.path.insert(0, str(FIXTURES))

    for arm in manifest.values():
        if arm.slow_build and not args.include_slow:
            continue
        _emit_arm(arm)

    print("done.")
    return 0


def _emit_arm(arm: Arm) -> None:
    """Build + emit one prebuilt artifact from its manifest entry."""
    t = time.perf_counter()
    if arm.fixture_kind == "urdf":
        kb = load_urdf_kinbody_normalized(FIXTURES / arm.fixture, arm.base_link, arm.ee_link)
    else:
        # specs: a Python builder module under tests/fixtures
        mod = __import__(arm.fixture)
        specs_fn_name = arm.specs_fn
        assert specs_fn_name is not None  # invariant per manifest schema
        kb = build_kinbody(
            getattr(mod, specs_fn_name)(),
            base_link_name=arm.base_link,
            ee_link_name=arm.ee_link,
        )
    plan = dispatch(kb)
    out = ARTIFACTS / f"{arm.name}.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name=arm.name,
        output_path=str(out),
        arm_label=arm.display_name,
    )
    elapsed = time.perf_counter() - t
    size_kb = out.stat().st_size / 1024
    print(
        f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} "
        f"(tier {plan.tier}, {size_kb:.1f} KB, {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    sys.exit(main())
