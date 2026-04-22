"""ssik -- pluggable analytical inverse-kinematics library for Python."""

from ssik._version import __version__
from ssik.core.tolerances import DEFAULT_TOLERANCE_POLICY, TolerancePolicy
from ssik.core.topology import TopologyReport, describe_topology

__all__ = [
    "DEFAULT_TOLERANCE_POLICY",
    "TolerancePolicy",
    "TopologyReport",
    "__version__",
    "describe_topology",
]
