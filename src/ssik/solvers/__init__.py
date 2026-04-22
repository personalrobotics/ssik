"""Analytical IK solvers.

Each module consumes a POE-normalized :class:`~ssik._kinbody.KinBody` and a
target pose, returning lists of joint configurations.

Current contents:

- :mod:`ssik.solvers.ur_family_hawkins` -- Hawkins 2013 analytical IK for
  UR3 / UR5 / UR10. **Temporary correctness oracle**: specialized for the
  UR frame conventions; scheduled for deletion once the generic tier-1
  univariate-polynomial solver lands (see umbrella #37).

Future: generic subproblem-composition solvers (IK-Geo style), the tier-1
univariate-polynomial solver, tier-2 fully-general 6R, and Husty-Pfurner as
a universal fallback. The dispatcher that picks which solver to run for a
given chain lands in Phase C.
"""
