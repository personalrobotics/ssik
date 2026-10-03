"""Example 04: redundant (7-DOF) arms -- a curve of answers, not a list.

A 7-DOF arm holding a 6-DOF pose is not at a point in configuration space, it
is on a curve: the elbow can swing through a continuous range while the hand
stays exactly where it is. This example covers both ways ssik exposes that:

1. **The curve itself** (Franka Panda). ``self_motion(T)`` returns the
   self-motion manifold as charts, each one continuous branch ``q(t)``. We check
   that every posture on a branch holds the pose and that the chart's tangent is
   a null direction of the arm's Jacobian, then use the curve to pick the
   posture farthest from every joint stop. The animated version is
   ``05_viser_interactive_ik.py --self-motion``.
2. **Samples, then tracking** (Kinova Gen3). The Gen3's axes miss the exact
   spherical-shoulder/spherical-wrist (SRS) geometry by 12 mm and 0.4 mm, so
   ssik solves a nearby exact SRS arm in closed form and polishes each answer
   against the Gen3's real FK (``seven_r.srs_polished``). ``solve()`` returns
   128 configurations: 16 samples of the self-motion times 8 branches each, a
   sample of the curve rather than "all the solutions". Tracking a path is a
   seeded solve, as on a 6-DOF arm.

Headless, wheel only::

    python examples/04_redundant_arms.py

The script checks its own claims at the end and exits non-zero if one fails.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import ssik
from ssik.postprocess import wrap_to_limits

# Central-difference step for the directional derivative of FK. Its truncation
# error is O(h^2) (~1e-10 here) and its round-off O(eps / h) (~1e-11).
FD_STEP = 1e-5


def fk_rate(arm: ssik.Manipulator, q: np.ndarray, v: np.ndarray) -> float:
    """``||d/ds FK(q + s v)||_F`` at ``s = 0``: the size of ``J(q) @ v`` as a pose rate.

    A central difference of the public ``fk``, so this needs no Jacobian API.
    It is zero exactly when ``v`` is a null direction of the arm's Jacobian.
    """
    d = (arm.fk(q + FD_STEP * v) - arm.fk(q - FD_STEP * v)) / (2 * FD_STEP)
    return float(np.linalg.norm(d))


def wrapped(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi


def panda_self_motion(checks: list[tuple[str, bool]]) -> None:
    # Charts live on Manipulator, so from_prebuilt is how a shipped arm reaches
    # them; solve() on the result is the artifact's own solver.
    arm = ssik.Manipulator.from_prebuilt("panda")
    lower = np.array([lim[0] for lim in arm.joint_limits])
    upper = np.array([lim[1] for lim in arm.joint_limits])
    rng = np.random.default_rng(17)
    T = arm.fk(rng.uniform(lower, upper))

    def in_the_box(qs: np.ndarray) -> np.ndarray:
        """The representatives of ``qs`` that sit inside the joint box.

        A chart reports angles as the geometry gives them, so a posture inside
        the limits can come back one full turn away: on the Panda a joint
        limited to ``[-3.07, -0.07]`` can arrive as ``+4.37``. Take the in-box
        representative before measuring against limits.
        """
        sols = [ssik.Solution(q=q, fk_residual=0.0) for q in np.atleast_2d(qs)]
        return np.vstack([s.q for s in wrap_to_limits(sols, arm.kinbody)])

    def clearance(q: np.ndarray) -> float:
        """Distance to the nearest joint stop, in radians."""
        return float(np.min(np.minimum(q - lower, upper - q)))

    print("== Franka Panda: the self-motion curve ==")
    manifold = arm.self_motion(T)
    print(f"branches at this pose      : {len(manifold.charts)}")
    print(f"distinct closed loops      : {len(manifold.sheets())}")

    # Each chart is one continuous branch. The one with the most room inside the
    # joint limits is the one worth planning on.
    chart = max(manifold.charts, key=lambda c: c.length(limits=True))
    print(f"roomiest branch            : {chart.length(limits=True):.3f} rad of arc")
    print(f"  exists on                : {len(chart.domain)} interval(s) of t")
    print(f"  stays in limits on       : {len(chart.in_limits())} sub-arc(s)")

    # Every posture on the curve holds the pose, and the tangent is a null
    # direction of the Jacobian: moving along it leaves the hand still.
    segments = chart.sample(200, limits=True)
    ts = np.concatenate([t for t, _ in segments])
    qs = in_the_box(np.vstack([q for _, q in segments]))
    drift = np.array([float(np.linalg.norm(arm.fk(q) - T)) for q in qs])

    # An arc ends where the branch folds back. A fold is a critical point of the
    # parameterization, not of the manifold: the rate of travel diverges there
    # while the direction stays well defined, so the null-direction check holds
    # at every sample, endpoints included.
    null_rate = max(fk_rate(arm, chart.q(t), chart.tangent(t)[0]) for t in ts)
    probe = rng.standard_normal(arm.dof)
    other_rate = fk_rate(arm, chart.q(ts[len(ts) // 2]), probe / np.linalg.norm(probe))
    rates = np.array([chart.tangent(t)[1] for t in ts])
    print(f"\nover {len(qs)} postures along that branch:")
    print(f"  end-effector drift       : {drift[1:-1].max():.2e}  ({len(ts) - 2} interior points)")
    print(f"  FK rate along the tangent: {null_rate:.1e}  (largest, all {len(ts)} points)")
    print(f"  FK rate, random direction: {other_rate:.2f}  (for scale)")
    print(f"  folds (infinite rate)    : {int(np.isinf(rates).sum())} of {len(ts)}")
    print(f"  drift at the two folds   : {drift[[0, -1]].max():.2e}  (worse there, see #591)")
    checks.append(("Panda: interior postures hold the pose to 1e-9", drift[1:-1].max() < 1e-9))
    checks.append(("Panda: the tangent is a null direction (FK rate < 1e-7)", null_rate < 1e-7))
    checks.append(("Panda: a random direction is not (FK rate > 1e-2)", other_rate > 1e-2))

    # The branch is a free choice, so spend it on something: here, stand as far
    # from every joint stop as the pose allows.
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
    best_fk = float(np.linalg.norm(arm.fk(best_on_curve) - T))
    print(f"\nclearance from the nearest joint stop, over {len(sols)} solve() postures")
    print("versus the whole self-motion manifold:")
    print(f"  best that solve() returned : {clearance(best_sampled):.4f} rad")
    print(f"  best on the manifold       : {best_score:.4f} rad")
    print(f"  FK check on that posture   : {best_fk:.2e}")
    print(
        "The gain is small here because solve() already samples the continuum, but"
        "\nit is a sample: the curve is the whole admissible set, so a criterion can"
        "\nbe optimized on it rather than picked from whatever came back.\n"
    )
    checks.append(
        (
            "Panda: the manifold's best clearance is at least solve()'s",
            best_score >= clearance(best_sampled),
        )
    )
    checks.append(("Panda: that posture holds the pose to 1e-9", best_fk < 1e-9))


def gen3_samples_and_tracking(checks: list[tuple[str, bool]]) -> None:
    print("== Kinova Gen3: samples of the curve, and seeded tracking ==")
    arm = ssik.Manipulator.from_prebuilt("gen3")
    print(arm)
    q_star = np.array([0.3, -0.4, 0.7, 0.5, 0.6, -0.5, 0.2])
    T = arm.fk(q_star)

    sols = arm.solve(T)
    worst_fk = max(s.fk_residual for s in sols)
    nearest = min(float(np.abs(wrapped(s.q, q_star)).max()) for s in sols)
    print(f"solve(T): {len(sols)} configurations = 16 swivel samples x 8 branches")
    print(f"  worst FK residual (against the Gen3's own FK) {worst_fk:.1e}")
    print(f"  q* itself is not among them: nearest is {nearest:.3f} rad away, because")
    print("  q* sits between samples. The samples are points on the curve, not its")
    print("  every point. The sample count is a solver option:")
    coarse = arm.solve(T, swivel_samples=4)
    print(f"  solve(T, swivel_samples=4): {len(coarse)} configurations")
    (seeded,) = arm.solve(T, q_seed=q_star, max_solutions=1)
    seeded_err = float(np.abs(wrapped(seeded.q, q_star)).max())
    print(f"  seeded with q* itself, max_solutions=1: off by {seeded_err:.1e} rad")
    checks.append(("Gen3: every configuration closes FK below 1e-9", worst_fk < 1e-9))
    checks.append(("Gen3: 128 = 16 x 8 by default", len(sols) == 128))
    checks.append(("Gen3: 4 samples give 4 x 8", len(coarse) == 32))
    checks.append(("Gen3: a seed that solves the pose is returned", seeded_err < 1e-9))

    # A smooth path, tracked with the seeded, capped solve.
    q_b = np.array([0.8, -0.1, 0.4, 0.9, 0.2, -0.1, 0.6])
    s = np.linspace(0.0, 1.0, 61)
    path_q = q_star + np.outer(3 * s**2 - 2 * s**3, q_b - q_star)
    poses = [arm.fk(q) for q in path_q]
    path_step = float(np.abs(np.diff(path_q, axis=0)).max())
    for _ in range(20):  # warm-up: one-time import and setup costs
        arm.solve(poses[1], q_seed=q_star, max_solutions=1)
        arm.solve(poses[1])
    q = q_star
    track, residuals, t_seeded = [q], [], []
    for T_k in poses[1:]:
        t0 = time.perf_counter()
        (sol,) = arm.solve(T_k, q_seed=q, max_solutions=1)
        t_seeded.append(time.perf_counter() - t0)
        q = sol.q
        track.append(q)
        residuals.append(sol.fk_residual)
    t_full = []
    for T_k in poses:
        t0 = time.perf_counter()
        arm.solve(T_k)
        t_full.append(time.perf_counter() - t0)
    track_q = np.array(track)
    step = float(np.abs(np.diff(track_q, axis=0)).max())
    off_path = float(np.abs(track_q - path_q).max())
    print(f"\ntracking {len(poses)} poses: solve(T, q_seed=q_prev, max_solutions=1)")
    print(f"  median time, seeded and capped  {np.median(t_seeded) * 1e3:.3f} ms")
    print(f"  median time, all 128            {np.median(t_full) * 1e3:.3f} ms")
    print(f"  worst FK residual               {max(residuals):.1e}")
    print(f"  largest joint step              {step:.4f} rad (the drawn path's: {path_step:.4f})")
    print(f"  furthest from the drawn path    {off_path:.4f} rad")
    print(
        "  The redundancy is free, so the track need not follow the elbow swivel"
        "\n  that drew the path; it holds every pose and moves continuously.\n"
    )
    checks.append(("Gen3: every tracked pose closes FK below 1e-9", max(residuals) < 1e-9))
    checks.append(("Gen3: no joint step larger than 1.5x the path's", step < 1.5 * path_step))


def main() -> int:
    checks: list[tuple[str, bool]] = []
    panda_self_motion(checks)
    gen3_samples_and_tracking(checks)
    for name, ok in checks:
        print(f"[{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
