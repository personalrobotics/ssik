# 8. Bulletproof validation

!!! warning "Scaffolding"
    Outline below; prose to be filled in.

## What this chapter covers

The validation discipline ssik enforces on every solver PR, why each piece exists, and what failure modes each catches.

### The standard

Memory entry [`feedback_bulletproof_solvers`](https://github.com/siddhss5/ikfastpy/issues/74). Verbatim:

> N-way cross-solver agreement on shared fixtures, 1e-10 FK tolerance, 500+ hypothesis poses; "must be perfect, all the time".

Every solver shipped in ssik passes through five gates before review.

### Gate 1 — Hand-picked generic poses (~5 fixtures × machine-precision FK)

Caught by: typos in the algebraic derivation, wrong indices in branch enumeration, sign flips in atan2, etc. Fast feedback loop in `pytest`.

### Gate 2 — Hand-picked near-singular poses

Wrist-pitch zero ($\sin q_4 = 0$), elbow zero / fully-extended ($\sin q_2 = 0$), shoulder-pan zero ($q_0 = 0$). Caught by: implicit assumptions of nonzero denominators in subproblem decomposition. Solvers must degrade gracefully — return fewer solutions (collapsed branches), all FK-exact, or signal `is_ls=True`.

### Gate 3 — Synthetic alternative-geometry fixture

A second arm with the same topology but different link lengths than the primary fixture. Validates "generic, not geometry-specific". Pieper-class solvers each ship one synthetic alternative; tier-2 solvers ship a randomised alternative.

### Gate 4 — 500-pose hypothesis fuzz with seeded $q^\star$ recovery

[Hypothesis](https://hypothesis.readthedocs.io/) generates 500 random non-singular $q^\star$, the test computes $T = \mathrm{FK}(q^\star)$, calls `solve(kb, T)`, and asserts:

1. `is_ls == False`.
2. Every returned solution FK-closes within 1e-8 to 1e-10 atol (tier-dependent).
3. The seeded $q^\star$ is recoverable mod $2\pi$ within 1e-3 to 1e-4 rad.

This catches: rare branch-enumeration bugs that hand-picked tests miss, completeness regressions where the solver loses one of its 8 solutions, and platform variance (hypothesis runs on every CI matrix entry).

### Gate 5 — N-way cross-solver agreement (where applicable)

Puma 560 satisfies the preconditions of *both* `spherical_two_parallel` and `spherical_two_intersecting`. The two compositions are algebraically distinct (SP4+SP3+SP1+SP4+SP1+SP1 vs SP3+SP2+SP4+SP1+SP1) but **must return the same 8-solution set** on every non-singular pose. The `tests/test_puma_cross_validation.py` file asserts exact set agreement at 1e-6 tolerance over 500 hypothesis poses.

This is the strongest correctness guarantee available without an oracle: each solver acts as the other's reference. If both fail in agreement, both have a parallel bug (vanishingly unlikely); if only one fails, the disagreement immediately identifies which one.

### Real-arm fixtures

Synthetic arms are necessary for completeness across topology classes. Real arms are necessary for hidden bugs that hide behind synthetic-fixture conveniences:

- The [JACO 2 j2n6s200 fixture](https://github.com/siddhss5/ikfastpy/blob/main/tests/fixtures/jaco2.py) was transcribed from the upstream MJCF (`robot-code/ada_assets/.../jaco2.xml`). It exposed the [`poe_to_dh` `T_pre` bug](07_kinbody_bridge.md) — UR5's joint-1 axis happens to align with world +z, so the original code's `T_pre = I` defaulting accidentally worked. JACO 2's `link_1.quat = (0, 0, 1, 0)` flips the axis and broke the bridge.
- Tier-2 RR validation runs against the *actual* JACO 2 chain, not a synthetic 60°-twist DH approximation. Slow tests at [`tests/test_jaco2_general_6r.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/test_jaco2_general_6r.py) cover seeded recovery + 4 keyframe round-trips.
- (#80) tracks adding more real fixtures — Agilex Piper, Flexiv Rizon, KUKA iiwa, Franka Panda — each rebroadcasts the same five gates against a different chain.

### "No papering over"

Memory entry [`feedback_no_papering_over`](https://github.com/siddhss5/ikfastpy/issues/74). When a solver fails a gate, the response is: **investigate the root cause**, fix the underlying bug. Not: clear the hypothesis cache, widen tolerances, mark the test slow and forget it, or scope the workaround without filing the underlying-bug issue.

When the underlying bug is structural and fixing it is out-of-scope for the immediate PR, the failing test gets explicit `xfail(strict=False)` with a reason linking to a tracking issue (e.g. [#82](https://github.com/siddhss5/ikfastpy/issues/82) for the MC Table I coverage gap). The test still **runs** — `xpassed`/`xfailed` counts surface the actual recovery rate per platform — but doesn't block CI on a known cross-platform-flakey case.

### Cumulative test surface

After PR #85, ssik runs:

- ~320 fast-suite tests (single-pose hand-picked + hypothesis fuzz at 500 examples per solver per arm).
- ~10 slow tests (sympy-preprocessing-dominated tier-2 RR round-trips, JACO 2 keyframes + UR5 generic).
- 4 CI jobs per PR (macOS / py3.13 + ubuntu / py3.{11,12,13}).
- Cross-solver Puma agreement (`spherical_two_parallel` ⇿ `spherical_two_intersecting`) at machine precision over 500 random poses.

The discipline is what allows shipping speed work like the Tier 1/2/3 PRs (#88/#89/#90) without precision regressions: every change runs through every gate before merge.
