"""ssik -- pluggable analytical inverse-kinematics library for Python.

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

from ssik._version import __version__
from ssik.core.dispatcher import DispatchPlan, dispatch
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.core.topology import TopologyReport, describe_topology

# Library best practice: prevent "No handlers could be found" warnings and
# avoid emitting any log records unless the consuming application configures
# the ``ssik`` namespace explicitly.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "DEFAULT_TOLERANCE_POLICY",
    "DispatchPlan",
    "TolerancePolicy",
    "TopologyReport",
    "__version__",
    "describe_topology",
    "dispatch",
]
