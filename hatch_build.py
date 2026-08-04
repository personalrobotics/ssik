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

# Native C++ solver extension (#506): the opt-in native backend (#501). The same
# pybind source that powers the test conformance binding is compiled into the
# wheel as ``ssik._ssik_native``. Header-only Eigen + C++20; shipped on Linux +
# macOS. Windows falls back to the Python path (native deferred), so the wheel
# still builds there without it.
# Top-level Extension name so build_ext --inplace drops the .so at the repo root
# (predictable, unlike a dotted name which setuptools resolves against src/); the
# force_include below then maps it to ssik/ in the wheel, imported as
# ``ssik._ssik_native`` (the pybind init symbol is keyed on the leaf name).
NATIVE_EXT_MODULE = "_ssik_native"
NATIVE_EXT_SOURCE = "cpp/bindings/three_parallel_py.cpp"
NATIVE_SUPPORTED_PLATFORMS = ("linux", "darwin")


def _native_supported() -> bool:
    return sys.platform in NATIVE_SUPPORTED_PLATFORMS


def _eigen_include(root: Path) -> str | None:
    """Locate the Eigen headers: EIGEN_INCLUDE_DIR (set by CIBW_BEFORE_ALL) or
    the standard system paths for local builds. Header-only, so no link step."""
    import os

    env = os.environ.get("EIGEN_INCLUDE_DIR")
    candidates = [env] if env else []
    candidates += [
        "/opt/homebrew/include/eigen3",
        "/usr/local/include/eigen3",
        "/usr/include/eigen3",
    ]
    for c in candidates:
        if c and (Path(c) / "Eigen" / "Dense").exists():
            return c
    return None


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
        from setuptools import Extension, setup  # type: ignore[import-untyped]

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
            # Append the native C++ solver extension on supported platforms when
            # Eigen is available. If Eigen is absent (a dev/editable install that
            # doesn't need native -- the test harness uses cpp/build), skip it
            # rather than failing the build. Shipped wheels are still guaranteed
            # to carry native by the wheel smoke gate ([tool.cibuildwheel]
            # test-command asserts ssik._ssik_native imports) + the native_wheel
            # CI job -- so a native-less wheel can never be published silently.
            native_built = False
            if _native_supported():
                eigen = _eigen_include(root)
                if eigen is None:
                    print(
                        f"[hatch_build] Eigen not found (set EIGEN_INCLUDE_DIR); skipping "
                        f"{NATIVE_EXT_MODULE}. Fine for a dev install; shipped wheels require it "
                        f"(enforced by the wheel smoke gate).",
                        file=sys.stderr,
                    )
                else:
                    import pybind11

                    ext_modules = [
                        *ext_modules,
                        Extension(
                            NATIVE_EXT_MODULE,
                            sources=[str(root / NATIVE_EXT_SOURCE)],
                            include_dirs=[
                                str(root / "cpp" / "include"),
                                pybind11.get_include(),
                                eigen,
                            ],
                            language="c++",
                            extra_compile_args=["-std=c++20", "-O2"],
                        ),
                    ]
                    native_built = True
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

        # Force-include the native C++ extension (#506). setuptools infers
        # package_dir[""]="src" from the Cython packages, so build_ext --inplace
        # drops the top-level _ssik_native module at src/; ship it as
        # ssik/_ssik_native.<abi>.so.
        if native_built:
            native_so = root / "src" / f"_ssik_native{so_suffix}"
            if not native_so.exists():
                raise RuntimeError(
                    f"CythonBuildHook: expected native extension at {native_so} after "
                    f"build_ext --inplace; got nothing."
                )
            force_include[str(native_so)] = f"ssik/_ssik_native{so_suffix}"

        # Mark wheel as platform-specific so the .so files (which are arch +
        # python-version + OS specific) only get installed on matching hosts.
        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        # Clean up build/ tree so we don't leak a stale tree into the next
        # build invocation. The .so files we want are already in src/.
        build_dir = root / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
