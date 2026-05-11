"""Contributor / debugging surface for ssik.

The names re-exported here are stable enough for advanced users
(``KinBody`` for hand-built fixtures, ``dispatch`` + ``describe_topology``
for inspection, ``build_kinbody`` for the spec-driven fixture path) but
aren't part of the v1.0 user-facing API. Promotion to the top-level
``ssik`` namespace requires the change to land behind a documented
deprecation cycle.

Examples::

    from ssik.internals import KinBody, build_kinbody, dispatch

    # Hand-build a KinBody from per-arm specs (rather than URDF parsing)
    kb = build_kinbody(my_arm_specs())

    # Inspect topology without dispatching IK
    from ssik.internals import describe_topology
    print(describe_topology(kb))

    # Pick the solver explicitly (advanced)
    from ssik.internals import dispatch
    plan = dispatch(kb)
    print(plan.solver_name, plan.tier)
"""

from ssik._kinbody import Joint, JointSpec, KinBody, Link, build_kinbody
from ssik.core.dispatcher import DispatchPlan, dispatch
from ssik.core.topology import TopologyReport, describe_topology

__all__ = [
    "DispatchPlan",
    "Joint",
    "JointSpec",
    "KinBody",
    "Link",
    "TopologyReport",
    "build_kinbody",
    "describe_topology",
    "dispatch",
]
