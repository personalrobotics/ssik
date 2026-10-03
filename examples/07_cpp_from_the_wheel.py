"""Example 07: the C++ solvers inside the wheel.

Every ssik wheel ships the header-only C++ solvers (``ssik_cpp``) and a CMake
package for them, so a C++ program, or a Python extension of your own, can call
the solver with no Python in the loop. This prints where they are and the joint
data a C++ caller builds its solver input from, for the UR5.

In your ``CMakeLists.txt``::

    find_package(ssik_cpp CONFIG REQUIRED)
    target_link_libraries(my_target PRIVATE ssik::ssik_cpp)

configured with the wheel's package and your Eigen::

    cmake -S . -B build \\
      -DCMAKE_PREFIX_PATH="$(python -c 'import ssik; print(ssik.get_cmake_dir())');<eigen prefix>"

and in C++, with ``consts`` / ``limits`` filled from ``ssik.cpp.joint_data``::

    #include "ssik_cpp/solvers/three_parallel.hpp"
    auto sols = ssik::three_parallel_artifact_solve(consts, limits, T, ssik::ArtifactParams<6>{});

A complete, built-in-CI version is ``cpp/examples/wheel_consumer/`` in the
repository; the guide is ``cpp/README.md`` ("Use it from the Python wheel").

    python examples/07_cpp_from_the_wheel.py

The script checks its own claims at the end and exits non-zero if one fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import ssik


def joint(axis: np.ndarray, q: float) -> np.ndarray:
    """A revolute joint's motion: rotation by ``q`` about the unit ``axis`` (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * (k @ k)
    return out


def main() -> int:
    checks: list[tuple[str, bool]] = []

    include = Path(ssik.get_include())
    print(f"ssik.get_include()   {include}")
    header = include / "ssik_cpp" / "solvers" / "three_parallel.hpp"
    checks.append(("the three_parallel header is in the include dir", header.is_file()))

    try:
        cmake_dir = Path(ssik.get_cmake_dir())
    except FileNotFoundError as exc:
        # Documented: only wheels carry the CMake package. An editable install of
        # a source checkout gets here; use get_include() or `cmake --install cpp`.
        print(f"ssik.get_cmake_dir() not in this install: {exc}")
    else:
        print(f"ssik.get_cmake_dir() {cmake_dir}")
        checks.append(
            (
                "the CMake package config is in the cmake dir",
                (cmake_dir / "ssik_cppConfig.cmake").is_file(),
            )
        )

    arm = ssik.Manipulator.from_prebuilt("ur5")
    d = ssik.cpp.joint_data(arm)
    print(f"\nssik.cpp.joint_data(UR5): solver {d.solver} -> three_parallel_artifact_solve")
    print(f"  JointConsts<{d.dof}>  axis {d.axis.shape}, t_left {d.t_left.shape}, ", end="")
    print(f"t_right {d.t_right.shape}, joint_type {d.joint_type.tolist()} (0 = revolute)")
    print(f"  JointLimits<{d.dof}>  lo      {np.round(d.lo, 4).tolist()}")
    print(f"                  hi      {np.round(d.hi, 4).tolist()}")
    print(f"                  present {d.present.tolist()}")

    # The arrays mean what JointData documents: FK is the product of
    # t_left[i] @ Joint(axis[i], q[i]) @ t_right[i]. Rebuilding it here from the
    # arrays alone must agree with the arm's own fk.
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, size=d.dof)
        T = np.eye(4)
        for i in range(d.dof):
            T = T @ d.t_left[i] @ joint(d.axis[i], q[i]) @ d.t_right[i]
        worst = max(worst, float(np.abs(T - arm.fk(q)).max()))
    print(f"\nFK rebuilt from the arrays vs arm.fk, 20 random q: max difference {worst:.1e}")
    checks.append(("the joint data reproduce the arm's FK to 1e-12", worst < 1e-12))
    checks.append(("UR5 goes to three_parallel_artifact_solve", d.solver == "ikgeo.three_parallel"))
    print()

    for name, ok in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
