"""Native 7R analytical solvers.

- :mod:`ssik.solvers.seven_r.srs` -- Singh-Kreutz strict-SRS solver
  (iiwa14 and other arms with exactly-concurrent shoulder + wrist
  axes).
- :mod:`ssik.solvers.seven_r.srs_polished` -- approximate-SRS variant
  with LM polish, for arms whose URDF axes only nearly meet (Kinova
  Gen3 today; future small-drift arms via the predicate).

Anthropomorphic 7R (Franka), other 7R families, and parametric-
redundancy variants land here as separate modules in future work.
"""
