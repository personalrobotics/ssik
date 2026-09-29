"""The header-only C++ solvers that ship with ssik, for native consumers.

Every wheel carries the ``ssik_cpp`` headers (``cpp/include/ssik_cpp`` in the
repository) and a relocatable CMake package for them, so a C++ project, or a
Python extension built with scikit-build-core, can call the family solvers
directly with no Python in the loop::

    cmake -DCMAKE_PREFIX_PATH="$(python -c 'import ssik; print(ssik.get_cmake_dir())')" ...

    find_package(ssik_cpp CONFIG REQUIRED)
    target_link_libraries(my_target PRIVATE ssik::ssik_cpp)

Eigen stays the consumer's responsibility: the package ``find_dependency``s
``Eigen3``, and ssik's own builds use the release pinned in
``scripts/fetch_eigen.py`` (3.4.0). The per-arm generated headers
(``cpp/gen/<arm>_ik.hpp``) are not in the wheel; emit one with
``scripts/cpp_emit.py`` from a source checkout.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["get_cmake_dir", "get_include"]

_HERE = Path(__file__).resolve().parent
_PROBE = Path("ssik_cpp") / "fk.hpp"


def _include_candidates() -> list[Path]:
    # The wheel's copy, then a source checkout's cpp/include (an editable or
    # dev install imports src/ssik/cpp, three levels below the repository root).
    return [_HERE / "include", _HERE.parents[2] / "cpp" / "include"]


def get_include() -> str:
    """Absolute path of the directory that contains ``ssik_cpp/``.

    Add it to the include path to ``#include "ssik_cpp/solvers/three_parallel.hpp"``
    and the other headers. In an installed wheel this is ``ssik/cpp/include``;
    in an editable or development install it is the checkout's ``cpp/include``.

    :raises FileNotFoundError: neither location holds the headers.
    """
    for cand in _include_candidates():
        if (cand / _PROBE).is_file():
            return str(cand)
    looked = ", ".join(str(c) for c in _include_candidates())
    raise FileNotFoundError(
        f"the ssik_cpp headers are not in this ssik install (looked in {looked}). "
        f"They ship in every ssik wheel; reinstall ssik from a wheel or its sdist."
    )


def get_cmake_dir() -> str:
    """Absolute path of the directory that contains ``ssik_cppConfig.cmake``.

    Pass it as ``CMAKE_PREFIX_PATH`` (or ``ssik_cpp_DIR``); then
    ``find_package(ssik_cpp CONFIG REQUIRED)`` defines the ``ssik::ssik_cpp``
    INTERFACE target, the same target ``cmake --install`` of ``cpp/`` exports.
    Its package version is this ssik release, compatible within one major
    version.

    :raises FileNotFoundError: this install has no CMake package. Only wheels
        carry it; in an editable or development install, use
        :func:`get_include` or install ``cpp/`` with ``cmake --install``.
    """
    cmake_dir = _HERE / "cmake"
    if (cmake_dir / "ssik_cppConfig.cmake").is_file():
        get_include()  # the config is useless without the headers beside it
        return str(cmake_dir)
    raise FileNotFoundError(
        f"no ssik_cpp CMake package in this ssik install ({cmake_dir}). It ships in "
        f"ssik wheels only. From a source checkout, use get_include() or "
        f"`cmake --install` of cpp/ (see cpp/README.md)."
    )
