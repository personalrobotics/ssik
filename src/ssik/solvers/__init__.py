"""Analytical IK solvers.

Each module consumes a POE-normalized :class:`~ssik._kinbody.KinBody` and a
target pose, returning lists of joint configurations.

Current contents:

- :mod:`ssik.solvers.ikgeo.three_parallel` -- generic three-parallel 6R
  solver built on SP1/SP3/SP6 composition. Handles any arm with three
  consecutive parallel axes at joints ``(1, 2, 3)`` -- UR3 / UR5 / UR10,
  and anything else with the same kinematic structure.
- :mod:`ssik.solvers.ikgeo.spherical_two_parallel` -- generic
  spherical-wrist + two-parallel-shoulder 6R solver built on
  SP1/SP3/SP4 composition. Handles any arm with three consecutive
  intersecting axes at joints ``(3, 4, 5)`` and two parallel axes at
  ``(1, 2)`` -- Puma 560, Fanuc, KUKA KR, and anything else with the
  same kinematic structure.
- :mod:`ssik.solvers.ikgeo.spherical_two_intersecting` -- generic
  spherical-wrist + intersecting-shoulder 6R solver built on
  SP1/SP2/SP3/SP4 composition. Handles any arm with three consecutive
  intersecting axes at joints ``(3, 4, 5)`` and joints ``(0, 1)``
  sharing an origin (``p[1] = 0``) -- compact arms where the waist
  and shoulder pivots coincide (Puma 560, ABB IRB smaller variants,
  uFactory lite6/xArm6 family).
- :mod:`ssik.solvers.ikgeo.spherical` -- generic spherical-wrist 6R
  solver built on SP1/SP4/SP5 composition. Fallback for spherical-
  wrist arms that match neither shoulder specialization (rare in
  commercial arms; typically custom / research geometries).

Future: Husty-Pfurner universal fallback, specialist 7R, dispatcher.
"""
