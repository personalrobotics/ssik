# 8. Bulletproof validation

Algorithms that work on hand-picked test poses are easy. Algorithms that work on **every** target pose, on **every** arm topology, with **machine-precision FK closure**, on **every** LAPACK backend, with **0 failures across 500 random samples** are hard. The discipline ssik enforces on every solver PR is what closes the gap.

This chapter walks through the five gates every solver passes through before review, why each gate exists, and what failure modes each catches. The shorthand we use for the standard is "bulletproof" — short for *must be perfect, all the time, on every reachable pose*.

## The standard

The verbatim direction from memory entry [`feedback_bulletproof_solvers`](https://github.com/siddhss5/ikfastpy/issues/74):

> N-way cross-solver agreement on shared fixtures, 1e-10 FK tolerance, 500+ hypothesis poses; "must be perfect, all the time".

The intent is not perfection-as-aspiration but perfection-as-default. A solver that passes 99% of poses is not bulletproof — the 1% is a debugging black hole, full of cases where your IK silently fails on production data. Better to find and fix the 1% before merge than ship the 99% and fight it later.

ssik passes through five gates, in order of cost.

## Gate 1 — Hand-picked generic poses

**What it is.** A handful of generic non-singular `q*` values (often four or five), each FK-encoded to `T*`, fed into `solve(kb, T*)`. Assertions:

- `is_ls == False`.
- The number of returned solutions matches the algorithm's expected branch count (often 8 for tier-0 closed-form, up to 16 for tier-2 RR).
- Every returned `q` reproduces `T*` under FK at machine precision (1e-10 atol).

**What it catches.** Typos in the algebraic derivation. Wrong indices in branch enumeration (`v_12[5]` vs `v_12[6]`). Sign flips in `atan2` that `±` solutions can't tell apart. Wrong `T_left` / `T_right` interpretation. Off-by-one in the DH parameter order.

**Cost.** Milliseconds per pose. Runs in fast pytest cycles.

## Gate 2 — Hand-picked near-singular poses

Singularities are the failure mode that hides best. The IK pipeline assumes nonzero denominators all over: `atan2(s, c)` doesn't care about magnitude but the SP3 ellipse intersection breaks if the elbow distance is zero, SP4's projection breaks if the projected vector is zero, SP5's quartic develops cluster roots near specific geometries (#55).

ssik's near-singular fixture set covers the classical ones:

- **Wrist-pitch zero**: `sin(q_4) = 0` aligns joints 3 and 5 (wrist roll + tool roll degenerate to a single rotation around a shared axis). Tier-0 spherical-wrist solvers must collapse two wrist branches into one and return the survivors at 1e-6 atol.
- **Wrist-pitch π**: the same degeneracy but at the alternate branch.
- **Elbow zero / fully-extended**: `sin(q_2) = 0` collapses the two shoulder branches. SP3's elbow distance constraint becomes linear instead of quadratic; the solver returns half the usual solutions but they remain exact.
- **Shoulder-pan zero**: `q_0 = 0` doesn't degenerate the algorithm itself but tests that the hand-picked rotation matrices are right.

What we assert: the solver returns at least one solution, no `nan` / `inf` leaks through, every returned `q` is exact under FK at 1e-6 atol. We don't assert solution counts because branch collapse changes them.

## Gate 3 — Synthetic alternative-geometry fixture

Bugs that depend on link-length specific values pass the hand-picked tests by accident. The third gate validates "generic, not geometry-specific" by running the same five gates against a *second* arm with the same topology but deliberately different DH parameters from the first.

For tier-0 solvers, each module ships a synthetic arm with hand-chosen alternative dimensions. For example, [`tests/test_three_parallel.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/test_three_parallel.py) tests against UR5 (the canonical fixture) and against `synthetic_three_parallel_kb` (different `d1, a2, a3, d4, d5, d6`, same `axes` pattern). The fixture builder is in the test file; deliberately not in `tests/fixtures/` to discourage cross-pollination.

For tier-2 solvers, the synthetic alternative is randomised — IK-Geo's `GeneralSetup` pattern: 6 random unit axes + 7 random offsets, no parallel or intersecting constraints. Drawn fresh per test.

What this catches: assumptions hardcoded against UR5's specific 0-radius wrist offset; SP6 cluster-root pathology that triggers only for specific Bezout configurations; numerical underflow when DH parameters span many orders of magnitude.

## Gate 4 — 500-pose hypothesis fuzz

The hand-picked fixtures cover known failure modes. The fuzz catches the *unknown* failure modes.

[Hypothesis](https://hypothesis.readthedocs.io/) draws 500 random non-singular `q*` per test, computes `T*` via FK, calls `solve(kb, T*)`, and asserts:

1. `is_ls == False`.
2. Every returned solution FK-closes within 1e-8 to 1e-10 atol (tier-dependent).
3. The seeded `q*` is recoverable mod $2\pi$ within 1e-3 to 1e-4 rad.

The third assertion is the load-bearing one. It catches **completeness regressions** — cases where the solver finds *some* IK solutions but not the specific seeded one. A solver that finds 7 of 8 IK branches will pass tests 1 and 2 (every solution it returns FK-closes; `is_ls` is False) but fail test 3 if the seeded `q*` was the 8th branch. Real bugs we caught this way: branch tracking errors in tier-1 univariate-search where the (q6, q4) tuples reorder between adjacent samples (memory entry [`project_tier1_search_completeness`](https://github.com/siddhss5/ikfastpy/issues/74)).

The 500-sample size is calibrated against the rare-event frequency. Most regression bugs hit ~5% of poses; 500 samples flags them with high confidence. Some bugs hit 1% of poses; 500 samples still flag those. Bugs at the 0.1% level need more samples — those tend to be near-singularities that the hand-picked Gate 2 catches.

Hypothesis runs on every CI matrix entry (macOS / py3.13 + ubuntu / py3.{11, 12, 13}), so platform variance shows up immediately. The MC Table I `xfail(strict=False)` from PR #89 was a direct consequence: the same 500-pose fuzz produces different recovery rates on different LAPACK backends, and we surface that via the `xpassed` / `xfailed` counts rather than papering over.

## Gate 5 — N-way cross-solver agreement

When two algebraically-distinct solvers cover the same arm family, they become each other's reference. ssik's strongest correctness guarantee is the **Puma 560 cross-validation** in [`tests/test_puma_cross_validation.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/test_puma_cross_validation.py).

Puma 560 satisfies the preconditions of *both*:

- **`spherical_two_parallel`** — three-intersecting-axes wrist + parallel shoulder-elbow (joints 1, 2). Composition: SP4 (projection on shared axis) + SP3 (elbow) + SP1 (shoulder plane) + SP4 (wrist alignment) + SP1 × 2 (wrist roll + tool roll).
- **`spherical_two_intersecting`** — three-intersecting-axes wrist + joints 0, 1 sharing an origin (`p[1] = 0`). Composition: SP3 (elbow distance from wrist center) + SP2 (shoulder) + SP4 + SP1 × 2.

The two compositions chain different subproblems in different orders. They share no algebraic substructure beyond "the answer is some IK solution to the same chain". And yet — by the uniqueness of IK on Puma 560 at non-singular poses — they *must* return the same 8-solution set on every pose.

The cross-validation asserts exact set agreement (every solution from one solver matches some solution from the other within 1e-6 wrap-to-π) over 500 hypothesis poses. If both solvers fail in agreement, both have a parallel bug — vanishingly unlikely given the algebraic distinctness. If only one fails, the disagreement immediately identifies which one.

This was the gate that retired the legacy "Hawkins oracle" cross-check (an external numerical reference): we no longer needed an external oracle once we had two algebraically-distinct in-tree solvers acting as each other's. Memory entry [`reference_ikfast_analytical_tricks`](https://github.com/siddhss5/ikfastpy/issues/81) covers the lineage.

## Real-arm fixtures

Synthetic arms cover topology classes; real arms catch convention bugs.

The **JACO 2 j2n6s200** fixture in [`tests/fixtures/jaco2.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/fixtures/jaco2.py) was transcribed by hand from the upstream MJCF (`robot-code/ada_assets/.../jaco2.xml`). It exposed the [`poe_to_dh` `T_pre` bug from Chapter 7](07_kinbody_bridge.md) — UR5's joint-1 axis happens to align with world +z, so the original code's `T_pre = I` defaulting accidentally worked. JACO 2's `link_1.quat = (0, 0, 1, 0)` flips the local +z axis to world −z and broke the bridge. Synthetic JACO-2-like fixtures (with hand-chosen DH parameters) couldn't have caught this because they would have shared UR5's convenient base-frame alignment.

The real-fixture work is tracked in [#80](https://github.com/siddhss5/ikfastpy/issues/80). Adding more real fixtures — Agilex Piper, Flexiv Rizon, KUKA iiwa, Franka Panda — is on the roadmap. Each one rebroadcasts the same five gates against a different chain.

## "No papering over"

The bulletproof discipline depends on root-cause analysis. When a gate fails, the response is **investigate the underlying bug**, not work around it. Memory entry [`feedback_no_papering_over`](https://github.com/siddhss5/ikfastpy/issues/74) lists the antipatterns:

- Clearing the hypothesis cache to rerandomise away from a failure (the failure tells you something; rerandomising buries it).
- Widening tolerances to make the failure pass (now the test passes but the bug is unfixed).
- Marking the test slow and forgetting it (now it never runs).
- Scoping the workaround without filing the underlying-bug issue (now the bug isn't tracked).

When the underlying bug is *known but structurally out-of-scope* for the immediate PR, the failing case gets explicit `xfail(strict=False)` with a `reason=...` linking to a tracking issue. The MC Table I coverage gap is the canonical example: [#82](https://github.com/siddhss5/ikfastpy/issues/82) is the tracking issue, the four MC Table I seeds are marked `xfail(strict=False)` in [`tests/test_raghavan_roth_pq.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/test_raghavan_roth_pq.py), and the test still runs on every CI run — the `xpassed` / `xfailed` counts surface the actual recovery rate per platform, so when #82 is fixed we'll know to drop the marks.

## Cumulative test surface

After PR #85, ssik's test surface is roughly:

- **~320 fast-suite tests** (single-pose hand-picked + hypothesis fuzz at 500 examples per solver per arm).
- **~10 slow tests** (sympy-preprocessing-dominated tier-2 RR round-trips on JACO 2 keyframes + UR5 generic).
- **4 CI jobs per PR** (macOS / py3.13 + ubuntu / py3.{11, 12, 13}).
- **Cross-solver Puma agreement** at machine precision over 500 random poses.
- **Per-PR slow round-trips** on JACO 2 (real MJCF fixture), UR5 (URDF fixture), Puma 560 (URDF fixture).

The discipline scales with the codebase: every new solver carries its own test file with the same five gates, every new fixture broadens the cross-arm coverage, every new optimisation runs through the same suite before merge. That's how the Tier 1/2/3 speedup PRs (#88, #89, #90) shipped without precision regressions: each one passed the same gates as the original solver. The 32% / 26% / median speed wins came with FK error unchanged at 3.7e-13 and 0 failures across 100 random poses on every PR.

The discipline is the reason the speed work could ship aggressively. Bulletproof gates aren't a slowdown; they're the safety net that lets you sprint.
