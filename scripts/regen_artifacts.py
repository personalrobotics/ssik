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

from ssik import _formats as formats
from ssik._kinbody import build_kinbody
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
    parser.add_argument(
        "--arm",
        action="append",
        metavar="NAME",
        help=(
            "Build only this arm (e.g. ``--arm ur7e_ik``; repeatable). "
            "Rebuilds just the named artifact(s) instead of the whole "
            "roster -- fast when adding a single arm. The legacy-alias map "
            "is still regenerated from the full manifest so flat imports "
            "and regen_bench keep working."
        ),
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    print(f"writing reference artifacts to {ARTIFACTS}/")

    manifest = load_manifest()
    sys.path.insert(0, str(FIXTURES))

    if args.arm:
        unknown = [name for name in args.arm if name not in manifest]
        if unknown:
            parser.error(f"unknown arm(s): {', '.join(unknown)} (not in MANIFEST.toml)")
        targets = [manifest[name] for name in args.arm]
    else:
        targets = [
            arm for arm in manifest.values() if not (arm.slow_build and not args.include_slow)
        ]

    for arm in targets:
        _emit_arm(arm)

    # Always regenerate the alias map from the FULL manifest -- a partial
    # build (``--arm``) must not leave the flat-import alias map stale, or
    # ``ssik.prebuilt.<arm>_ik`` and regen_bench break with ModuleNotFoundError.
    _regen_prebuilt_init(manifest)
    print("done.")
    return 0


def _regen_prebuilt_init(manifest: dict[str, Arm]) -> None:
    """Regenerate the ``_LEGACY_ALIASES`` map in ``prebuilt/__init__.py`` from
    the full manifest (#421) -- keeps the flat-import alias finder in sync with
    the vendor layout. Runs on every regen, slow arms included, so the map is
    complete even when ``--include-slow`` is off."""
    import re

    init = ARTIFACTS / "__init__.py"
    rows = ",\n".join(
        f'    "{a.name}": "{a.vendor}.{a.module_basename}"' for a in manifest.values()
    )
    new_block = "_LEGACY_ALIASES: dict[str, str] = {\n" + rows + ",\n}"
    text = init.read_text()
    text = re.sub(
        r"_LEGACY_ALIASES: dict\[str, str\] = \{.*?\n\}", new_block, text, count=1, flags=re.S
    )
    init.write_text(text)


def _emit_arm(arm: Arm) -> None:
    """Build + emit one prebuilt artifact from its manifest entry."""
    t = time.perf_counter()
    if arm.fixture_kind == "specs":
        # specs: a Python builder module under tests/fixtures
        mod = __import__(arm.fixture)
        specs_fn_name = arm.specs_fn
        assert specs_fn_name is not None  # invariant per manifest schema
        kb = build_kinbody(
            getattr(mod, specs_fn_name)(),
            base_link_name=arm.base_link,
            ee_link_name=arm.ee_link,
        )
    else:
        # A registered source format (urdf, mjcf, ...): one loader, no branch.
        kb = formats.get(arm.fixture_kind).load(
            FIXTURES / arm.fixture, arm.base_link, arm.ee_link, {}
        )
    plan = dispatch(kb)
    # Vendor subpackage layout (#421): src/ssik/prebuilt/<vendor>/<basename>.py
    vendor_dir = ARTIFACTS / arm.vendor
    vendor_dir.mkdir(exist_ok=True)
    (vendor_dir / "__init__.py").touch()
    out = vendor_dir / f"{arm.module_basename}.py"
    emit_artifact(
        kb=kb,
        plan=plan,
        module_name=arm.name,
        output_path=str(out),
        arm_label=arm.display_name,
    )
    # RR arms ship a sidecar baked-tensor .npz next to the artifact so the shipped
    # native ext covers them (the ~30s sympy derivation runs here, at build time,
    # never per-solve). The emitted _rr_native_geometry() loads it lazily (#555).
    if plan.solver_name == "ikgeo.general_6r":
        from ssik._native import bake_rr_tensor_npz

        npz = vendor_dir / f"{arm.module_basename.removesuffix('_ik')}_rr.npz"
        bake_rr_tensor_npz(kb, str(npz))
    if plan.solver_name == "jointlock.seven_r":
        from ssik._native import _HP_JOINTLOCK_ARMS, bake_jointlock_npz

        npz = vendor_dir / f"{arm.module_basename.removesuffix('_ik')}_jl.npz"
        bake_jointlock_npz(kb, str(npz), use_hp=arm.name in _HP_JOINTLOCK_ARMS)
    elapsed = time.perf_counter() - t
    size_kb = out.stat().st_size / 1024
    print(
        f"  {out.relative_to(REPO_ROOT)}: {plan.solver_name} "
        f"(tier {plan.tier}, {size_kb:.1f} KB, {elapsed:.1f}s)"
    )


if __name__ == "__main__":
    sys.exit(main())
