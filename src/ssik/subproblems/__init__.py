"""Canonical subproblems SP1-SP6 used by closed-form analytical IK.

These are the building blocks of the IK-Geo-style subproblem decomposition
(Elias & Wen, arXiv:2211.05737). Higher-level solvers compose them to invert
the kinematics of specific robot topologies (spherical wrist, three-parallel,
etc.); the subproblems themselves are robot-agnostic.

Every subproblem has **exact** and **least-squares** regimes. Inputs that
satisfy the feasibility conditions (matching magnitudes / axial projections /
etc.) return one or more exact solutions; infeasible inputs return a single
LS solution that continuously extends the exact case near singularities --
critical for numerical robustness. Each ``solve`` function returns an
``is_ls`` flag so downstream solvers can propagate the distinction.

All subproblems are implemented clean-room from the Elias-Wen paper and the
BSD-3 IK-Geo reference at https://github.com/rpiRobotics/ik-geo.
"""

from ssik.subproblems import sp1, sp2, sp3, sp4, sp5, sp6

__all__ = ["sp1", "sp2", "sp3", "sp4", "sp5", "sp6"]
