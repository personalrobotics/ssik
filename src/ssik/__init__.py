"""ssik -- analytical inverse kinematics for 6R/7R revolute arms.

Public surface (v1.0):

- :class:`Manipulator` -- runtime classifier + dispatcher; load via
  :meth:`Manipulator.from_urdf` (interactive) or build an artifact
  once via ``ssik build`` and ``import <arm>_ik`` (production).
- :class:`Solution` -- analytical IK result (``q``, ``fk_residual``,
  ``refinement_used``).
- :class:`TolerancePolicy` / :data:`DEFAULT_TOLERANCE_POLICY` --
  knobs for FK closure thresholds (rarely needed).

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
import sys as _sys
import warnings as _warnings

from ssik._version import __version__
from ssik.core.diagnostic import Diagnostic
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.manipulator import Manipulator

# Library best practice: prevent "No handlers could be found" warnings and
# avoid emitting any log records unless the consuming application configures
# the ``ssik`` namespace explicitly.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Python 3.10 and numpy 1.x install and work, but they sit below the versions
# CI exercises and below the Scientific Python SPEC 0 support window
# (https://scientific-python.org/specs/spec-0000/, which recommends dropping
# Python 3 years and numpy 2 years after release). We allow them on a
# best-effort, untested basis and warn once so users know. See #366.
if _sys.version_info < (3, 11):
    _warnings.warn(
        f"ssik is running on Python {_sys.version_info.major}."
        f"{_sys.version_info.minor}, below the tested minimum (3.11) and the "
        "Scientific Python SPEC 0 support window. It is supported on a "
        "best-effort, untested basis. See "
        "https://github.com/personalrobotics/ssik/issues/366.",
        stacklevel=2,
    )

__all__ = [
    "DEFAULT_TOLERANCE_POLICY",
    "Diagnostic",
    "Manipulator",
    "Solution",
    "TolerancePolicy",
    "__version__",
]
