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

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Files to compile. Order matters: deeper deps first.
TARGETS = [
    REPO / "src" / "ssik" / "kinematics" / "_scalar3.py",
    REPO / "src" / "ssik" / "subproblems" / "_rotation.py",
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

    print(f"compiling {len(TARGETS)} ssik modules in-place via Cython:")
    for t in TARGETS:
        print(f"  {t.relative_to(REPO)}")

    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(
        name="ssik_cython_inplace",
        ext_modules=cythonize(
            [str(t) for t in TARGETS],
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
                "initializedcheck": False,
            },
            annotate=True,  # writes per-file .html annotation reports
        ),
        include_dirs=[np.get_include()],
        script_args=sys.argv[1:],
    )
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
