"""7R joint-locking solvers.

For 7R (and higher-DOF) redundant arms, ``ssik`` ships a universal
joint-lock wrapper that sweeps one joint across a range of values and
dispatches the locked sub-chain (a 6R arm) to the best-matching 6R
solver in ``ssik.solvers.ikgeo``. One solver covers Franka, iiwa,
Rizon, Gen3, xArm7 (and any other 7R) by topology-driven auto-selection
of the lock joint.

Modules:

- :mod:`ssik.solvers.jointlock.seven_r` -- the universal 7R wrapper.
"""
