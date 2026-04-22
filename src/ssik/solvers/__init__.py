"""Analytical IK solvers.

Each module consumes a POE-normalized :class:`~ssik._kinbody.KinBody` and a
target pose, returning lists of joint configurations.

Current contents:

- :mod:`ssik.solvers.ikgeo.three_parallel` -- generic three-parallel 6R
  solver built on SP1/SP3/SP6 composition. Handles any arm with three
  consecutive parallel axes at joints ``(1, 2, 3)`` -- UR3 / UR5 / UR10,
  and anything else with the same kinematic structure.

Future: Husty-Pfurner universal fallback, specialist 7R, dispatcher.
"""
