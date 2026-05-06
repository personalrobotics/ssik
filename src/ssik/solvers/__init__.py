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
- :mod:`ssik.solvers.ikgeo.two_intersecting` -- tier-1 univariate-
  search solver for 6R arms where joints ``(4, 5)`` share an origin
  (``p[5] = 0``). Uses 1D ``search_1d`` over ``theta_3`` with an
  inner SP5 shoulder solve per sample. Rare topology in commercial
  arms; dispatcher fallback when no spherical-wrist sibling matches.
- :mod:`ssik.solvers.ikgeo.two_parallel` -- tier-1 univariate-search
  solver for 6R arms where joints ``(1, 2)`` are parallel. Uses 1D
  ``search_1d`` over ``theta_0`` with an inner SP6 coupling (q4, q6)
  per sample. Narrower applicability than ``three_parallel`` and
  fewer returned solutions due to tier-1 search sparsity.
- :mod:`ssik.solvers.ikgeo.general_6r` -- tier-2 general 6R solver via
  Raghavan-Roth m_quad polynomial. Universal-6R coverage with
  ms-scale runtime on well-conditioned arms (JACO 2 ~ 0.6 ms). The
  m_quad matrix conditioning degrades on highly-symmetric DH
  geometries (alpha = pi/2 with a_i = 0 throughout, e.g. KUKA iiwa
  locked sub-chains); when RR stalls, ``husty_pfurner.general_6r``
  is the alternative-algebra fallback.
- :mod:`ssik.solvers.husty_pfurner.general_6r` -- tier-2 universal 6R
  solver via Husty-Pfurner Study quaternion + dual-quaternion
  algebra. Slower than RR on well-conditioned arms (~ 100 ms) but
  robust to ill-conditioning; covers locked-7R sub-chains where
  RR's m_quad blows up. Singular-DH perturbation (#176) handles the
  measure-zero V_L-in-Study-quadric structure that arises in real
  industrial 7R arms.
- :mod:`ssik.solvers.jointlock.seven_r` -- universal 7R wrapper that
  locks one joint (auto-selected by topology) and dispatches the
  resulting 6R sub-chain to the best-matching solver. Covers
  Franka Panda, KUKA iiwa, Flexiv Rizon, Kinova Gen3, uFactory xArm7
  and any other 7R arm. Tier-2 fallback within jointlock is HP (not
  RR) because post-lock geometries hit the symmetric-DH conditioning
  case.

Future: native SRS-class 7R closed-form (#143).
"""
