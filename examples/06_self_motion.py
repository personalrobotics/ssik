"""Self-motion: the other answers a redundant arm has.

A 7-DOF arm holding a 6-DOF pose is not at a point in configuration space, it
is on a curve. The elbow can swing through a continuous range while the hand
stays exactly where it is. ``solve()`` samples that curve and hands back
points; ``self_motion()`` hands back the curve.

This example needs no display and no extra dependencies. For the animated
version, see ``05_viser_interactive_ik.py --self-motion``.

Run::

    python examples/06_self_motion.py
"""

from __future__ import annotations

import numpy as np

import ssik
from ssik.postprocess import wrap_to_limits
from ssik.prebuilt.franka import panda_ik
from ssik.refinement import kinbody_jacobian

# Charts live on Manipulator, and reaching one from a prebuilt arm currently
# needs its baked KinBody. See issue #587 for the missing constructor.
arm = ssik.Manipulator(panda_ik._KB)
kb = arm.kinbody
lower = np.array([j.limits[0] for j in kb.joints])
upper = np.array([j.limits[1] for j in kb.joints])

rng = np.random.default_rng(17)
T = arm.fk(np.array([rng.uniform(*j.limits) for j in kb.joints]))


def in_the_box(qs: np.ndarray) -> np.ndarray:
    """The representatives of ``qs`` that actually sit inside the joint box.

    ``in_limits`` judges a joint modulo 2*pi, so a posture it admits can come
    back written on a turn outside the box: on the Panda a joint limited to
    ``[-3.07, -0.07]`` can arrive as ``+4.37``. Same posture, one full turn
    away. Take the in-box representative before measuring against limits.
    """
    sols = [ssik.Solution(q=q, fk_residual=0.0) for q in np.atleast_2d(qs)]
    return np.vstack([s.q for s in wrap_to_limits(sols, kb)])


def clearance(q: np.ndarray) -> float:
    """Distance to the nearest joint stop, in radians."""
    return float(np.min(np.minimum(q - lower, upper - q)))


# ---------------------------------------------------------------------------
# The curve.
# ---------------------------------------------------------------------------
manifold = arm.self_motion(T)
print(f"branches at this pose      : {len(manifold.charts)}")
print(f"distinct closed loops      : {len(manifold.sheets())}")

# Each chart is one continuous branch. Pick the one with the most room to move
# inside the joint limits: that is the one worth planning on.
chart = max(manifold.charts, key=lambda c: c.length(limits=True))
print(f"roomiest branch            : {chart.length(limits=True):.3f} rad of arc")
print(f"  exists on                : {len(chart.domain)} interval(s) of t")
print(f"  stays in limits on       : {len(chart.in_limits())} sub-arc(s)")

# ---------------------------------------------------------------------------
# The claim, checked rather than asserted: every posture on the curve holds
# the same pose, and the tangent is a genuine null direction of the Jacobian.
# ---------------------------------------------------------------------------
segments = chart.sample(200, limits=True)
ts = np.concatenate([t for t, _ in segments])
qs = in_the_box(np.vstack([q for _, q in segments]))

drift = np.array([float(np.linalg.norm(arm.fk(q) - T)) for q in qs])

# An arc ends where the branch folds back. A fold is a critical point of the
# parameterization, not of the manifold: the rate of travel diverges there
# while the direction stays well defined, so the invariant is measurable on
# every sample, endpoints included.
null_resid = max(
    float(np.linalg.norm(kinbody_jacobian(kb, chart.q(t)) @ chart.tangent(t)[0])) for t in ts
)
rates = np.array([chart.tangent(t)[1] for t in ts])
print(f"\nover {len(qs)} postures along that branch:")
print(f"  end-effector drift       : {drift[1:-1].max():.2e}  ({len(ts) - 2} interior points)")
print(f"  |J @ tangent direction|  : {null_resid:.2e}  (all {len(ts)} points)")
print(f"  folds (infinite rate)    : {int(np.isinf(rates).sum())} of {len(ts)}")
print(f"  drift at the two folds   : {drift[[0, -1]].max():.2e}  (worse there, see #591)")

# ---------------------------------------------------------------------------
# Why it is useful: the branch is a free choice, so spend it on something.
# Here, stand as far from every joint stop as the pose allows.
# ---------------------------------------------------------------------------
sols = arm.solve(T)
best_sampled = max((s.q for s in sols), key=clearance)

best_on_curve, best_score = best_sampled, clearance(best_sampled)
for c in manifold.charts:
    segs = c.sample(300, limits=True)
    if not segs:
        continue
    for q in in_the_box(np.vstack([q for _, q in segs])):
        if clearance(q) > best_score:
            best_on_curve, best_score = q, clearance(q)

print(f"\nclearance from the nearest joint stop, over {len(sols)} solve() postures")
print("versus the whole self-motion manifold:")
print(f"  best that solve() returned : {clearance(best_sampled):.4f} rad")
print(f"  best on the manifold       : {best_score:.4f} rad")
print(f"  FK check on that posture   : {np.linalg.norm(arm.fk(best_on_curve) - T):.2e}")

print(
    "\nThe gain is small here because solve() already samples the continuum, but it"
    "\nis a sample: the curve is the whole admissible set, so a criterion can be"
    "\noptimized on it rather than picked from whatever came back. arm.solve_path()"
    "\nfollows one branch through a Cartesian path, the same idea applied over time."
)
