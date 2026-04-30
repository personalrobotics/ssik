"""Numerical IK solvers (universal correctness backstop).

ssik's analytical solvers (:mod:`ssik.solvers.ikgeo`,
:mod:`ssik.solvers.jointlock`) cover commercial 6R / 7R arms with
recognised kinematic decompositions. For chains that don't match any
analytical pattern -- exotic DH, custom 7R redundancy structures,
arms with no Pieper-class decomposition at any lock joint -- the
numerical solvers in this subpackage give the universal "always works"
guarantee.

Currently exports:

  * :mod:`ssik.solvers.numerical.lm_multi_restart` -- damped
    Levenberg-Marquardt with N random restarts. Builds on the per-arm
    baked FK + spatial Jacobian (#126) so post-Cython compilation
    yields ``~12 ms`` per IK call (8 restarts x 30 iters x ~50 us/iter
    Cython native).

These are NOT the dispatcher's default. Analytical solvers stay the
priority path; numerical sits at the end of the dispatch chain or as
an explicit opt-in for users with chains the analytical pipeline
doesn't recognise.
"""
