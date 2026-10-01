"""ssik -- analytical inverse kinematics for 6R/7R revolute arms.

Public surface (v1.0):

- :class:`Manipulator` -- runtime classifier + dispatcher; load via
  :meth:`Manipulator.from_urdf` (interactive) or build an artifact
  once via ``ssik build`` and ``import <arm>_ik`` (production). For an
  arm ssik already ships, :meth:`Manipulator.from_prebuilt` gives you
  that artifact's own solver plus the geometry it bakes, which is how
  the self-motion chart API is reached on a shipped arm.
- :class:`Solution` -- analytical IK result (``q``, ``fk_residual``,
  ``refinement_used``).
- :class:`TolerancePolicy` / :data:`DEFAULT_TOLERANCE_POLICY` --
  knobs for FK closure thresholds (rarely needed).
- :func:`get_include` / :func:`get_cmake_dir` -- where the wheel's
  header-only C++ solvers and their CMake package are (:mod:`ssik.cpp`).

Quickstart::

    import ssik
    arm = ssik.Manipulator.from_urdf("ur5.urdf", base="base_link", ee="ee_link")
    sols = arm.solve(T_target, max_solutions=1, q_seed=q_current)

For deployment, prefer the build artifact::

    # one-time build, emits my_arm_ik.py
    $ ssik build my_arm.urdf --base base_link --ee tool0

    # then in your code:
    import my_arm_ik
    sols = my_arm_ik.solve(T_target, max_solutions=1, q_seed=q_current)

Contributor / debugging surface (``KinBody``, ``dispatch``,
``describe_topology``, ...) lives under :mod:`ssik.internals`.

Logging: the package emits hierarchical logs under the ``ssik`` namespace.
By default a ``NullHandler`` suppresses all output. To see solver
diagnostics::

    import logging
    logging.getLogger("ssik").setLevel(logging.INFO)
    logging.basicConfig()
"""

import logging as _logging

from ssik._stale_extensions import check_checkout as _check_checkout

# First, before anything imports a compiled module: in a source checkout, refuse
# in-place extensions built from other source than the tree holds (#614).
_check_checkout()

from ssik._version import __version__  # noqa: E402
from ssik.core.diagnostic import Diagnostic  # noqa: E402
from ssik.core.solution import Solution  # noqa: E402
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy  # noqa: E402

# The shipped C++ headers (#641).
from ssik.cpp import get_cmake_dir, get_include  # noqa: E402
from ssik.manipulator import Manipulator  # noqa: E402

# Catalog of shipped arms; imports no artifact (#421).
from ssik.prebuilt import list_arms  # noqa: E402

# Library best practice: prevent "No handlers could be found" warnings and
# avoid emitting any log records unless the consuming application configures
# the ``ssik`` namespace explicitly.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "DEFAULT_TOLERANCE_POLICY",
    "Diagnostic",
    "Manipulator",
    "Solution",
    "TolerancePolicy",
    "__version__",
    "get_cmake_dir",
    "get_include",
    "list_arms",
]
