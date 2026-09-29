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
(``cpp/gen/<arm>_ik.hpp``) are not in the wheel. Build the solver inputs at
runtime with :func:`joint_data`, or emit a header with ``scripts/cpp_emit.py``
from a source checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ssik.manipulator import Manipulator

__all__ = ["JointData", "get_cmake_dir", "get_include", "joint_data"]

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


@dataclass(frozen=True)
class JointData:
    """The arrays ``ssik::JointConsts<N>`` and ``ssik::JointLimits<N>`` hold.

    Row ``i`` describes joint ``i`` from the base. Forward kinematics is
    ``T = prod_i t_left[i] @ Joint(axis[i], q[i]) @ t_right[i]``, where
    ``Joint`` rotates about ``axis`` (revolute) or translates along it
    (prismatic); ``ssik::fk`` in ``ssik_cpp/fk.hpp`` evaluates exactly that.

    - ``axis`` (N, 3), ``t_left`` (N, 4, 4), ``t_right`` (N, 4, 4): the
      ``JointConsts`` fields of the same names.
    - ``joint_type`` (N,): ``JointConsts::type``, as the ``ssik::JointType``
      value: 0 ``Revolute``, 1 ``Prismatic``.
    - ``lo``, ``hi`` (N,), ``present`` (N,) bool: the ``JointLimits`` fields.
      A joint without limits has ``present`` false and ``lo = hi = 0``.
    - ``solver``: the solver family these frames are prepared for
      (:attr:`ssik.Manipulator.solver_name`). ``ikgeo.three_parallel`` data
      goes to ``three_parallel_artifact_solve``.
    """

    solver: str
    axis: NDArray[np.float64]
    t_left: NDArray[np.float64]
    t_right: NDArray[np.float64]
    joint_type: NDArray[np.int32]
    lo: NDArray[np.float64]
    hi: NDArray[np.float64]
    present: NDArray[np.bool_]

    @property
    def dof(self) -> int:
        """Number of joints, the ``N`` of ``JointConsts<N>``."""
        return len(self.joint_type)


def joint_data(arm: Manipulator) -> JointData:
    """The joint data to build ``JointConsts<N>`` / ``JointLimits<N>`` from, for ``arm``.

    These are the values the native backend passes to the C++ solvers and the
    values ``scripts/cpp_emit.py`` bakes into ``cpp/gen/<arm>_ik.hpp``. For
    ``ikgeo.spherical_two_parallel`` the frames are in the canonical wrist
    gauge that solver requires (forward kinematics is unchanged); other
    families use the arm's own frames. Limits are always the arm's own.

    >>> import ssik
    >>> d = ssik.cpp.joint_data(ssik.Manipulator.from_prebuilt("ur5"))
    >>> d.solver, d.dof, d.t_left.shape
    ('ikgeo.three_parallel', 6, (6, 4, 4))
    """
    return _joint_data(arm.kinbody, arm.solver_name)


def _joint_data(kb: Any, solver: str) -> JointData:
    """:func:`joint_data` for a KinBody, prepared for ``solver`` (any other
    name, for example ``""``, takes the KinBody's own frames)."""
    geom = kb
    if solver == "ikgeo.spherical_two_parallel":
        from ssik._kinbody import canonicalize_spherical_wrist
        from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY

        geom = canonicalize_spherical_wrist(kb, DEFAULT_TOLERANCE_POLICY)
    gj = geom.joints
    lj = kb.joints  # limits from the arm's own (physical) joints
    return JointData(
        solver=solver,
        axis=np.array([j.axis for j in gj], dtype=np.float64).reshape(-1, 3),
        t_left=np.array([j.T_left for j in gj], dtype=np.float64).reshape(-1, 4, 4),
        t_right=np.array([j.T_right for j in gj], dtype=np.float64).reshape(-1, 4, 4),
        joint_type=np.array([0 if j.joint_type == "revolute" else 1 for j in gj], np.int32),
        lo=np.array([j.limits[0] if j.limits else 0.0 for j in lj], dtype=np.float64),
        hi=np.array([j.limits[1] if j.limits else 0.0 for j in lj], dtype=np.float64),
        present=np.array([bool(j.limits) for j in lj], dtype=np.bool_),
    )
