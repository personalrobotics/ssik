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
(``cpp/gen/<arm>_ik.hpp``) are not in the wheel. Build the solver input at
runtime instead: :func:`solver_data` for ``ssik::Solver``, which serves every
native family, or :func:`joint_data` for ``three_parallel_artifact_solve``.
From a source checkout, ``scripts/cpp_emit.py`` emits a per-arm header.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ssik._native import _NATIVE_SOLVERS, _SOLVER_DATA_SCHEMA

if TYPE_CHECKING:
    from ssik.manipulator import Manipulator

__all__ = [
    "NATIVE_FAMILIES",
    "SOLVER_DATA_SCHEMA",
    "JointData",
    "SolverData",
    "UnsupportedFamilyError",
    "get_cmake_dir",
    "get_include",
    "joint_data",
    "solver_data",
]

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


#: The schema number of the :class:`SolverData` this ssik produces, and the only
#: one its ``ssik::Solver`` accepts (``ssik::kSolverDataSchema``). The arrays of
#: each family at this number are specified in ``docs/specs/solver_data.md``.
SOLVER_DATA_SCHEMA: int = _SOLVER_DATA_SCHEMA

#: The solver families with a native solver: the ones :func:`solver_data` and
#: ``ssik::Solver`` serve (``ssik::native_families()``).
NATIVE_FAMILIES: frozenset[str] = _NATIVE_SOLVERS


class UnsupportedFamilyError(ValueError):
    """:func:`solver_data` cannot describe this arm for ``ssik::Solver``.

    Raised when the arm's solver family has no native solver (it is not in
    :data:`NATIVE_FAMILIES`), for a ``jointlock.seven_r`` arm built at runtime
    (only the prebuilt jointlock arms ship the data its native solver needs),
    and when a family's native geometry check rejects an arm the dispatcher put
    in that family. The message says which. ``family`` is the arm's solver
    family.
    """

    def __init__(self, family: str, message: str) -> None:
        super().__init__(message)
        self.family = family


@dataclass(frozen=True)
class SolverData:
    """The complete input of one arm's native solver, ``ssik::Solver``.

    The same record as ``ssik::SolverData`` (``ssik_cpp/solver_data.hpp``): a
    family name, a schema number and a flat map of named arrays. Every array
    is read-only, C-contiguous and either float64 or int32; a scalar has shape
    ``()``. The names, shapes, dtypes and meaning of each family's arrays are
    specified in ``docs/specs/solver_data.md``; semver covers them through
    ``schema``.

    - ``family``: the solver family, :attr:`ssik.Manipulator.solver_name`.
    - ``schema``: :data:`SOLVER_DATA_SCHEMA` of the ssik that produced it.
    - ``dof``: the number of joints, 6 or 7.
    - ``arrays``: ``{name: ndarray}``, read-only.

    A pybind11 extension converts it with
    ``ssik::pybind11_interop::solver_data_from_python``
    (``ssik_cpp/pybind11/solver_data.hpp``). It maps one-to-one onto an
    ``.npz``: ``np.savez(path, schema=d.schema, family=d.family, **d.arrays)``.
    """

    family: str
    schema: int
    dof: int
    arrays: Mapping[str, NDArray[Any]]


def solver_data(arm: Manipulator) -> SolverData:
    """The input ``ssik::Solver`` (``ssik_cpp/solve.hpp``) needs for ``arm``.

    It is built by the same code that feeds ssik's own native backend, so for
    the options ``ssik::ArtifactParams`` can express, ``ssik::Solver(d).solve<N>``
    returns the list, in the same order, that the native backend returns for
    this data (``docs/api.md``, "The native backend"). For an arm from
    :meth:`ssik.Manipulator.from_prebuilt` that is ``arm.solve(T, native=True)``.

    Works for prebuilt arms and for arms built at runtime (``from_urdf``,
    ``from_mjcf``, ``from_axes``), and needs no native extension. A runtime arm
    on ``ikgeo.general_6r`` needs its numeric coefficient tensor: about 28 s of
    symbolic work the first time on a machine (the JACO 2), then milliseconds
    from the per-arm preparation cache (``docs/api.md``, "Per-arm preparation
    cache"); :meth:`ssik.Manipulator.prepare` pays it ahead of time. Prebuilt
    arms ship it.

    >>> import ssik
    >>> d = ssik.cpp.solver_data(ssik.Manipulator.from_prebuilt("ur5"))
    >>> d.family, d.schema, d.dof, d.arrays["t_left"].shape
    ('ikgeo.three_parallel', 1, 6, (6, 4, 4))

    :raises TypeError: ``arm`` is not a :class:`ssik.Manipulator`.
    :raises UnsupportedFamilyError: the arm's family has no native solver, the
        arm is a ``jointlock.seven_r`` arm built at runtime, or the family's
        native geometry check rejects the arm.
    """
    from ssik import _native
    from ssik.manipulator import Manipulator

    if not isinstance(arm, Manipulator):
        raise TypeError(f"solver_data() takes an ssik.Manipulator, got {type(arm).__name__}")
    family = arm.solver_name
    if family not in NATIVE_FAMILIES:
        raise UnsupportedFamilyError(
            family,
            f"solver family {family!r} has no native solver; ssik::Solver serves "
            f"{', '.join(sorted(NATIVE_FAMILIES))}",
        )
    kb = arm.kinbody
    prebuilt = arm._prebuilt
    geometry: Any = None
    if family == "jointlock.seven_r":
        # The artifact's own baked sweep, the data its native=True path uses.
        geometry = None if prebuilt is None else prebuilt._jointlock_native_geometry()
        if geometry is None:
            raise UnsupportedFamilyError(
                family,
                "a jointlock.seven_r arm built at runtime has no native solver data: its "
                "inner kernel (Raghavan-Roth or Husty-Pfurner) and its 16 lock samples are "
                "fixed when an arm is onboarded, so only the prebuilt jointlock arms "
                "(Manipulator.from_prebuilt) ship them",
            )
    elif family == "ikgeo.general_6r":
        # A prebuilt arm's shipped tensor, the one its native=True path uses.
        if prebuilt is not None:
            geometry = prebuilt._rr_native_geometry()
        if geometry is None:
            geometry = _native.rr_native_geometry(kb)
    arrays = _native._solver_arrays(kb, family, geometry)
    if arrays is None:
        raise UnsupportedFamilyError(
            family,
            f"the dispatcher put this arm in {family!r}, but that family's native "
            f"geometry check rejects it, so there is no native solver data for it",
        )
    frozen: dict[str, NDArray[Any]] = {}
    for name, value in arrays.items():
        a = np.array(value, copy=True, order="C")
        a.flags.writeable = False
        frozen[name] = a
    return SolverData(
        family=family,
        schema=SOLVER_DATA_SCHEMA,
        dof=len(kb.joints),
        arrays=MappingProxyType(frozen),
    )
