"""IK-Geo subproblem-composition solvers.

Each module here solves a specific kinematic family by composing the
canonical subproblems from :mod:`ssik.subproblems` (SP1-SP6). Port of the
BSD-3 [ik-geo Rust reference][ikgeo] with the same algorithmic structure;
adapted to our POE-normalized :class:`~ssik._kinbody.KinBody` input
format and the :class:`~ssik.core.tolerances.TolerancePolicy` interface.

[ikgeo]: https://github.com/rpiRobotics/ik-geo/blob/main/rust/src/inverse_kinematics/mod.rs
"""
