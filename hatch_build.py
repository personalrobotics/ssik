"""Custom hatchling build hook: cythonize the modules that ship Cython
pure-Python-mode annotations (``@cython.ccall``, ``@cython.locals(...)``)
and force-include the resulting ``.so`` files in the wheel.

The annotated source modules stay valid pure-Python so the library
still imports without compiled extensions (sdist install on an
unsupported platform, dev checkouts that haven't run ``build_ext``);
the decorators are no-ops at the Python level. Compiled wheels load the
``.so`` shim ahead of the ``.py`` source.

Compiled targets (#248): the hot leaf primitives that dominate FK and
LM-polish budgets. Add new files here as their Cython annotations land.
"""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

CYTHON_TARGETS: tuple[str, ...] = (
    # Bench-validated wins (see #248): poe_fk is the inner-loop FK for
    # every SRS-class solve; refinement.kinbody_jacobian + lm_refine_batch
    # dominate Gen3 LM polish. Net: iiwa14 -14%, Gen3 -34%.
    "src/ssik/kinematics/poe_fk.py",
    "src/ssik/refinement/__init__.py",
    # Other modules carry the same ``@cython.ccall`` decorators (sp5,
    # _rotation, _scalar3, ikgeo.spherical) but bench-flat or worse on the
    # canonical fixtures. ``_scalar3`` in particular regresses Rizon 4 /
    # Kassow ~6x because the float(a[0]) array-index pattern can't be
    # unboxed by Cython without a buffer-typed argument annotation, so
    # the compiled C extension dispatch overhead exceeds the pure-Python
    # interpreter cost for these 3-element ops. Re-evaluate when the
    # call sites switch to typed memoryviews or scalar arguments.
)


class CythonBuildHook(BuildHookInterface):  # type: ignore[type-arg]
    PLUGIN_NAME = "cython"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        # sdist must remain platform-independent: ship the annotated .py files
        # and let the install-time path build extensions if Cython is present.
        if self.target_name == "sdist":
            return

        # Lazy imports: Cython + setuptools are build-time dependencies declared
        # in [build-system].requires.
        from Cython.Build import cythonize
        from setuptools import setup  # type: ignore[import-untyped]

        root = Path(self.root)

        # Run setuptools build_ext --inplace. Cython's pure-Python-mode .py
        # files compile to .so beside the source. ``cythonize`` produces
        # Extension objects from the .py files; setuptools then invokes the
        # platform C compiler.
        original_argv = sys.argv[:]
        original_cwd = Path.cwd()
        try:
            sys.argv = ["setup.py", "build_ext", "--inplace"]
            # ``setup`` resolves paths relative to cwd; chdir to project root
            # so the .so files land next to the source .py files.
            import os

            os.chdir(root)
            ext_modules = cythonize(  # type: ignore[no-untyped-call]
                list(CYTHON_TARGETS),
                compiler_directives={"language_level": "3"},
                annotate=False,
            )
            setup(
                name="ssik-cython-ext",
                ext_modules=ext_modules,
                script_args=["build_ext", "--inplace"],
            )
        finally:
            sys.argv = original_argv
            os.chdir(original_cwd)

        # Force-include the compiled .so files in the wheel. Hatchling's
        # default wheel target only picks up .py / .pyi files from the
        # ``packages`` setting; we explicitly map each .so so it ships.
        so_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        force_include: dict[str, str] = build_data.setdefault("force_include", {})
        for src_py in CYTHON_TARGETS:
            src_path = root / src_py
            so_path = src_path.with_name(src_path.stem + so_suffix)
            if not so_path.exists():
                raise RuntimeError(
                    f"CythonBuildHook: expected compiled extension at {so_path} "
                    f"after build_ext --inplace; got nothing."
                )
            # build_data["force_include"] is {source_abs_path: dest_in_wheel}.
            rel_dest = str(so_path.relative_to(root / "src"))
            force_include[str(so_path)] = rel_dest

        # Mark wheel as platform-specific so the .so files (which are arch +
        # python-version + OS specific) only get installed on matching hosts.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        # Clean up build/ tree so we don't leak a stale tree into the next
        # build invocation. The .so files we want are already in src/.
        build_dir = root / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
