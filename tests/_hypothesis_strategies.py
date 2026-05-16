"""Shared Hypothesis strategies for IK property tests.

The flagship :func:`non_singular_q6r` strategy generates 6R joint vectors
whose configurations are far enough from common 6R singularities that
the analytical solver can reliably recover the seeded branch. Property
tests that assert *seeded* ``q*`` recovery (i.e. ``q*`` appears in the
returned solution set, not just that returned solutions FK-close) should
use this strategy. Near-singular behaviour is covered separately by
hand-picked parametrised fixtures in each test file
(``test_near_singular_pose_returned_solutions_fk_match``) which assert
FK closure only, not seed recovery.

Singularities the strategy filters (closes #101, #115, #215):

- ``q[1] ≈ 0 or π`` — shoulder pitch parallel to base.
- ``q[2] ≈ 0 or π`` — upper-arm / elbow alignment.
- ``q[3] ≈ 0 or π`` — elbow alignment. Dominant cause of SP6
  near-double roots on UR-class arms and Puma; the dedup-by-residual
  gate cannot reliably pick the seeded representative when the Bezout
  quartic loses precision.
- ``q[4] ≈ 0 or π`` — wrist 2 alignment (the spherical-wrist gimbal).

The previous (pre-#101/#115/#215) strategy filtered only q[1], q[2],
q[4]. Hypothesis consistently shrank to ``q[3] = 0`` on UR5, Puma 560,
and synthetic-spherical fixtures; adding the q[3] filter closes all
three flakes. Filter rate rises from ~30% to ~40% on uniform input,
well under Hypothesis' ``filter_too_much`` health-check budget.

Other axes (``q[0]`` shoulder yaw, ``q[5]`` flange roll) don't trigger
solver-level degeneracies on the kinematic classes covered by this
strategy. Per-test custom strategies should compose their own
additional ``assume`` filters as needed.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

_ANGLE = st.floats(min_value=-np.pi + 0.3, max_value=np.pi - 0.3, allow_nan=False, width=64)


@st.composite
def non_singular_q6r(draw: st.DrawFn) -> np.ndarray:
    """6R q-vector with the four singularity-prone axes (q[1], q[2],
    q[3], q[4]) at least ``arcsin(0.2) ≈ 11.5°`` away from 0 and π.

    The 0.2 sine threshold matches what each test's local strategy used
    before consolidation; the new piece is the q[3] filter, which is the
    axis Hypothesis shrinks to under the prior strategies' permissive
    bounds.
    """
    q = np.array([draw(_ANGLE) for _ in range(6)])
    assume(abs(np.sin(q[1])) > 0.2)
    assume(abs(np.sin(q[2])) > 0.2)
    assume(abs(np.sin(q[3])) > 0.2)
    assume(abs(np.sin(q[4])) > 0.2)
    return q


@st.composite
def non_singular_q7r(draw: st.DrawFn) -> np.ndarray:
    """7R q-vector with the five singularity-prone middle axes
    (q[1] through q[5]) at least ``arcsin(0.2) ≈ 11.5°`` away from 0
    and π. q[0] (base yaw) and q[6] (flange roll) are left free --
    rotation about the base axis doesn't change reach, and the flange
    roll is a wrist-axis 7R singularity only when *also* aligned with
    the previous joint, which the other filters prevent.

    Applies uniformly to SRS-class (iiwa14, Gen3) and non-SRS 7R
    (Franka, Rizon, Kassow, xArm7). The dominant singularity classes
    -- elbow alignment (q[3] = 0) for SRS, wrist gimbal (q[4]/q[5] = 0)
    for spherical-wrist -- are filtered. Per-arm specialisation can
    compose additional ``assume(...)`` filters on top.
    """
    q = np.array([draw(_ANGLE) for _ in range(7)])
    assume(abs(np.sin(q[1])) > 0.2)
    assume(abs(np.sin(q[2])) > 0.2)
    assume(abs(np.sin(q[3])) > 0.2)
    assume(abs(np.sin(q[4])) > 0.2)
    assume(abs(np.sin(q[5])) > 0.2)
    return q
