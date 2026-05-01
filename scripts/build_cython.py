"""Cython-compile selected ssik modules in-place for benchmarking.

Slice 1 of #137 -- compiles the leaf primitives that everything depends
on. Run this before running ``scripts/track_speedup.py`` to see the
``rung 3`` numbers.

Usage:

    uv run python scripts/build_cython.py

After this, the next ``import ssik.kinematics._scalar3`` (etc.) loads the
compiled ``.so`` instead of the ``.py`` source. Tests, bench, regen
artifacts -- all get the compiled path automatically.

To revert to pure Python: delete the generated ``.so`` files (or
``git clean -fd`` the source tree).

This is a *developer tool*, not part of the wheel build. The wheel
build will integrate Cython via hatchling-build-hook in a later slice
of #137 (cibuildwheel + multi-platform wheels).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Files to compile. Order matters: deeper deps first.
# These targets have explicit Cython type annotations on hot-path locals;
# the aggressive directives (``boundscheck=False, wraparound=False``) are
# safe because indices are typed and bounded.
TARGETS = [
    REPO / "src" / "ssik" / "kinematics" / "_scalar3.py",
    REPO / "src" / "ssik" / "subproblems" / "_rotation.py",
    # Slice 2: sp5._refine_sp5 is 30% of Franka 7R IK time per profile.
    REPO / "src" / "ssik" / "subproblems" / "sp5.py",
]

# Targets compiled with safer directives -- ``wraparound=True`` and
# ``boundscheck=True``. Files here use Python-idiomatic indexing (``cum[-1]``
# negative wrap-around, untyped numpy slicing) that aggressive directives
# would mis-compile or segfault.
SAFE_TARGETS = [
    # Slice 3.1: refinement.dedup_by_wrap_close + _q_close. Inner per-pair
    # check is the dominant cost in 7R jointlock dedup; kinbody_jacobian
    # in the same module uses ``cum[-1]`` so we keep wraparound on.
    REPO / "src" / "ssik" / "refinement" / "__init__.py",
    # Slice 4 step 2 (#147): Franka 7R's actual inner-dispatch hot path
    # is ``ikgeo.spherical`` (most lock samples produce its topology, not
    # the spherical_two_parallel baked into the artifact at codegen time).
    # Per-call profile shows ~1.7 ms in the body; Cython types the loop
    # locals as ``cython.double`` and reduces Python-interpreter overhead.
    REPO / "src" / "ssik" / "solvers" / "ikgeo" / "spherical.py",
]

# Per-arm artifact targets (#137 Slice 3). The orchestrator code emitted
# by ``ssik.core.codegen`` carries pure-Python-mode Cython annotations
# (``@cython.ccall`` on ``_fk`` / ``_spatial_jacobian`` / ``_wrap_to_pi``)
# but the body still uses Python-style numpy indexing. Compiled with the
# same safer directives as ``SAFE_TARGETS``.
ARTIFACT_TARGETS = [
    REPO / "tests" / "artifacts" / "franka_panda_ik.py",
]


def main() -> int:
    try:
        from Cython.Build import cythonize
        from setuptools import setup
    except ImportError as e:
        print(f"missing build dependency: {e}", file=sys.stderr)
        print("  uv pip install cython setuptools", file=sys.stderr)
        return 1

    import numpy as np

    total = len(TARGETS) + len(SAFE_TARGETS) + len(ARTIFACT_TARGETS)
    print(f"compiling {total} ssik modules in-place via Cython:")
    for t in TARGETS:
        print(f"  {t.relative_to(REPO)}")
    for t in SAFE_TARGETS:
        print(f"  {t.relative_to(REPO)}  (safe directives)")
    for t in ARTIFACT_TARGETS:
        print(f"  {t.relative_to(REPO)}  (artifact, safe directives)")

    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    # Aggressive directives for src/ssik targets (manually typed).
    aggressive = cythonize(
        [str(t) for t in TARGETS],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
        annotate=True,  # writes per-file .html annotation reports
    )
    safe_directives = {
        "language_level": "3",
        "boundscheck": True,
        "wraparound": True,
        "cdivision": True,
        "initializedcheck": True,
    }
    safe_src = cythonize(
        [str(t) for t in SAFE_TARGETS],
        compiler_directives=safe_directives,
        annotate=True,
    )
    safe_artifacts = cythonize(
        [str(t) for t in ARTIFACT_TARGETS],
        compiler_directives=safe_directives,
        annotate=True,
    )
    extensions = list(aggressive) + list(safe_src) + list(safe_artifacts)
    # Disable FP contraction (FMA) -- ssik.kinematics._scalar3 uses strict
    # left-to-right IEEE 754 evaluation as the determinism guarantee that
    # makes codegen (poe_to_dh -> sympy.cse) bit-exact across platforms. A
    # compiler that contracts a*b + c*d into fma(a,b,c*d) saves 1 ulp of
    # rounding error but produces a bit-different result, which propagates
    # through cse and breaks artifact byte-equality (test_artifact_snapshots).
    # `-fno-fast-math` is the macOS/clang default but kept explicit as
    # belt-and-braces against future toolchain changes.
    for ext in extensions:
        ext.extra_compile_args = [
            *(ext.extra_compile_args or []),
            "-ffp-contract=off",
            "-fno-fast-math",
        ]
    setup(
        name="ssik_cython_inplace",
        ext_modules=extensions,
        include_dirs=[np.get_include()],
        script_args=sys.argv[1:],
    )
    # ``cythonize`` puts artifact .so files at ``src/<module>.cpython-...so``
    # (because the artifact source lives outside any registered package). Move
    # each one next to its .py source so ``import tests.artifacts.<arm>_ik``
    # picks up the compiled extension.
    for source in ARTIFACT_TARGETS:
        stem = source.stem  # e.g. "franka_panda_ik"
        misplaced = list((REPO / "src").glob(f"{stem}.cpython-*.so"))
        for so in misplaced:
            dest = source.parent / so.name
            shutil.move(str(so), str(dest))
            print(f"  moved {so.relative_to(REPO)} -> {dest.relative_to(REPO)}")
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
