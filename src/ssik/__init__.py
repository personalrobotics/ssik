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

from ssik._version import __version__
from ssik.core.diagnostic import Diagnostic
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.manipulator import Manipulator

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
]
