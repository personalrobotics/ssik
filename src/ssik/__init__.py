"""ssik -- pluggable analytical inverse-kinematics library for Python.

Quickstart::

    import ssik

    arm = ssik.Manipulator.from_urdf(
        "ur5.urdf", base="base_link", ee="ee_link"
    )
    T = arm.fk([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    sols, is_ls = arm.ik(T)

For trajectory tracking::

    sols, is_ls = arm.ik(T_target, max_solutions=1, q_seed=q_prev)

Logging: the package emits hierarchical logs under the ``ssik`` namespace
(``ssik.solvers.ikgeo.three_parallel``, etc.). By default a ``NullHandler``
suppresses all output, so importing ssik never produces unsolicited stderr
in user applications. To see solver diagnostics, install a handler:

    import logging
    logging.getLogger("ssik").setLevel(logging.INFO)
    logging.basicConfig()  # or supply your own handler

The ``ssik build`` CLI configures this automatically via ``--verbose``.
"""

import logging as _logging

from ssik._kinbody import Joint, JointSpec, KinBody, Link, build_kinbody
from ssik._version import __version__
from ssik.core.dispatcher import DispatchPlan, dispatch
from ssik.core.solution import Solution
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.core.topology import TopologyReport, describe_topology
from ssik.manipulator import Manipulator

# Library best practice: prevent "No handlers could be found" warnings and
# avoid emitting any log records unless the consuming application configures
# the ``ssik`` namespace explicitly.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "DEFAULT_TOLERANCE_POLICY",
    "DispatchPlan",
    "Joint",
    "JointSpec",
    "KinBody",
    "Link",
    "Manipulator",
    "Solution",
    "TolerancePolicy",
    "TopologyReport",
    "__version__",
    "build_kinbody",
    "describe_topology",
    "dispatch",
]
