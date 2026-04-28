# 2. The Pieper class and subproblem composition

Most commercial 6-DOF arms admit closed-form analytical IK because their kinematic structure has a **special geometric property** — three consecutive intersecting axes (a spherical wrist) or three consecutive parallel axes (a parallel-shoulder configuration). Pieper proved in 1968 that the first of these conditions makes IK a closed-form problem; subsequent work extended the result to other geometric specialisations. This chapter walks through that body of theory — the *Pieper class* — and the modern computational reformulation in terms of six canonical *subproblems* that compose into per-family solvers.

This is necessary background, not novel material. EAIK and IK-Geo already ship excellent implementations of the algorithms in this chapter. ssik bundles the same algorithms (ported from the BSD-3 IK-Geo Rust reference) so that Pieper-class arms — UR5, UR10, Puma 560, most Fanuc / KUKA / ABB industrial arms — solve in 50–200 µs warm-cache without any of the tier-2 numeric machinery from [Chapter 4](04_raghavan_roth.md). The non-Pieper arms — the EAIK gap that motivates ssik's existence — are covered in [Chapter 3](03_eaik_gap.md).

## Pieper's condition

A 6-DOF serial manipulator with three consecutive *intersecting* joint axes admits closed-form IK.

The argument is geometric. If joints 4, 5, 6 share a common point of intersection (a "spherical wrist"), the position of the wrist center is determined by the first three joint angles alone — joints 4, 5, 6 only rotate the EE around the fixed wrist center. So the IK decomposes into:

1. **Inverse position** for $(q_1, q_2, q_3)$: place the wrist center at the target-pose-implied location.
2. **Inverse orientation** for $(q_4, q_5, q_6)$: rotate the EE around the wrist center to match the target rotation.

Each subproblem is a 3-equation 3-unknown closed-form system. The wrist-center position equation reduces to a quartic; the orientation equation reduces to a $\sin$/$\cos$ pair via Euler-angle extraction.

The same decomposition works (with minor adjustments to which subproblem solves which equations) for arms with three *parallel* axes — three-parallel arms like the UR family decouple via projection onto the shared axis.

Pieper's theorem covers virtually every 6-DOF industrial robot built before 2015: anthropomorphic arms with spherical wrists (Puma, Fanuc, KUKA KR, ABB IRB, Stäubli), collaborative arms with three-parallel shoulders (UR3/5/10, Kassow), and most lab manipulators. Modern arms designed for compactness (Kinova JACO 2, Agilex Piper) deliberately *break* the Pieper condition for mechanical-design reasons (compact wrist housings, non-orthogonal twists for cable routing) — those are the EAIK gap.

## The six canonical subproblems

The Paden–Kahan formulation (1986) and its modern Elias–Wen extension (IK-Geo, 2022/2025) decompose Pieper-class IK into six canonical *subproblems*. Each subproblem is a closed-form geometric primitive — circle/sphere/plane intersection — and each per-family solver chains a specific sequence of subproblems to produce 8 IK solutions per pose.

The subproblems, in order of complexity:

**SP1 — single-axis rotation alignment.** Given an axis $\hat{k}$ and two vectors $\mathbf{p}, \mathbf{q}$, find $\theta$ with $R(\hat{k}, \theta) \mathbf{p} = \mathbf{q}$. Closed form: $\theta = \mathrm{atan2}(\hat{k} \cdot (\mathbf{p} \times \mathbf{q}),\ \mathbf{p} \cdot \mathbf{q} - (\hat{k} \cdot \mathbf{p})(\hat{k} \cdot \mathbf{q}))$. Single-valued (or LS-optimal if input is infeasible). Used for "rotate around joint axis to align two vectors."

**SP2 — two sequential rotations.** Given two axes $\hat{k}_1, \hat{k}_2$ and vectors $\mathbf{p}, \mathbf{q}$, find $(\theta_1, \theta_2)$ with $R(\hat{k}_1, \theta_1) R(\hat{k}_2, \theta_2) \mathbf{p} = \mathbf{q}$. Reduces to two SP1 calls plus a magnitude check. Up to 2 solutions. Used for "rotate around two consecutive joints to align two vectors."

**SP3 — distance constraint on a circle.** Given an axis $\hat{k}$, vectors $\mathbf{p}, \mathbf{q}$, and a target distance $d$, find $\theta$ with $\|R(\hat{k}, \theta) \mathbf{p} - \mathbf{q}\| = d$. Reduces to a quadratic in $\sin\theta$ (or $\cos\theta$). Up to 2 solutions. Used for "elbow distance" — given the wrist center is at a known distance from the shoulder, what's the elbow angle?

**SP4 — projection onto an axis.** Given axes $\hat{h}, \hat{k}$, a vector $\mathbf{p}$, and a target $d$, find $\theta$ with $\hat{h} \cdot R(\hat{k}, \theta) \mathbf{p} = d$. Reduces to a linear-in-$\sin$/$\cos$ equation. Up to 2 solutions. Used for "shoulder pan" — project onto the parallel-trio axis to isolate the base joint.

**SP5 — three sequential rotations placing a position.** Given three axes $\hat{k}_1, \hat{k}_2, \hat{k}_3$ and vectors $\mathbf{p}_0, \ldots, \mathbf{p}_3$, find $(\theta_1, \theta_2, \theta_3)$ with $\mathbf{p}_0 + R(\hat{k}_1, \theta_1) (\mathbf{p}_1 + R(\hat{k}_2, \theta_2) (\mathbf{p}_2 + R(\hat{k}_3, \theta_3) \mathbf{p}_3)) = \mathbf{0}$. Reduces to a quartic in $\tan(\theta_3/2)$. Up to 4 triple solutions. Used for "shoulder-elbow-wrist position chain" when no specialisation is available — the heaviest tier-0 subproblem.

**SP6 — Bezout quartic on two SP4 conditions.** Given paired axes $\{(\hat{h}_i, \hat{k}_i, \mathbf{p}_i)\}_{i=1..4}$ and target scalars $d_1, d_2$, find $(\theta_1, \theta_2)$ such that two SP4 conditions hold simultaneously. Reduces to a Bezout resultant — a quartic in $\tan(\theta_2/2)$. Up to 4 pair solutions. Used for "three-parallel base rotation + wrist roll" in the UR family.

ssik's implementations live in [`src/ssik/subproblems/`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/subproblems/) — one module per subproblem (`sp1`, `sp2`, ..., `sp6`), pure-numpy clean-room from the Elias–Wen paper, with hand-computed test vectors for every entry-point case.

## How per-family solvers compose subproblems

For each Pieper-class kinematic family, ssik ships a solver module under [`src/ssik/solvers/ikgeo/`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/) that chains the right sequence of subproblems for that family's geometry.

**`ikgeo.spherical_two_parallel`** — three-intersecting-axes wrist + parallel shoulder-elbow (joints 1, 2). Targets: Puma 560, Fanuc LR, KUKA KR series, ABB IRB. Composition:

1. SP4 on the shoulder projection: project the base-to-wrist vector onto the parallel axis to isolate $q_1$. Up to 2 solutions for $q_1$.
2. For each $q_1$: SP3 on the elbow triangle for $q_3$. Up to 2 solutions.
3. For each $(q_1, q_3)$: SP1 for $q_2$ from the shoulder plane.
4. For each $(q_1, q_2, q_3)$: SP4 for the wrist pitch $q_5$.
5. For each $(q_1, q_2, q_3, q_5)$: SP1 × 2 for $q_4$ and $q_6$.

Total: $2 \times 2 \times 1 \times 2 \times 1 \times 1 = 8$ IK solutions per generic pose.

**`ikgeo.three_parallel`** — three consecutive parallel axes (joints 1, 2, 3). Targets: UR3, UR5, UR10, any other three-parallel-shoulder arm. Composition:

1. SP6 on the parallel-trio + wrist axes for $(q_1, q_5)$ jointly. Up to 4 pair solutions.
2. For each $(q_1, q_5)$: SP1 for $q_3$ + SP1 × 2 for the dependent $\theta_{14} = q_2 + q_3 + q_4$ and $q_6$.
3. SP3 for the elbow.
4. SP1 for $q_2$.
5. $q_4 = \mathrm{wrap\_to\_pi}(\theta_{14} - q_2 - q_3)$.

Total: 8 IK solutions.

**`ikgeo.spherical_two_intersecting`** — three-intersecting-axes wrist + joints 0,1 sharing an origin ($p_1 = 0$). Alternate composition for Puma 560, ABB IRB smaller variants, compact arms. Composition:

1. SP3 on the elbow distance constraint (since $p_1 = 0$ simplifies the wrist-center position equation).
2. For each $q_3$: SP2 on the shoulder for $(q_1, q_2)$. Up to 2 pair solutions per $q_3$.
3. SP4 + SP1 × 2 for the wrist orientation.

Total: 8 IK solutions.

Note Puma 560 satisfies the preconditions of *both* `spherical_two_parallel` and `spherical_two_intersecting` simultaneously. This redundancy is the cross-validation gate from [Chapter 8](08_bulletproof.md): the two algebraically-distinct compositions must return the same 8-solution set on every pose, and they do — over 500 hypothesis poses at 1e-6 wrap-to-π agreement.

**`ikgeo.spherical`** — generic three-intersecting-axes wrist with no additional shoulder specialisation. Falls back to SP5 for the joint shoulder-to-wrist position equation. Up to 8 solutions but slower (~150 µs vs ~50 µs for the specialisations) because SP5's quartic is heavier than SP3 + SP1.

**`ikgeo.two_parallel`** and **`ikgeo.two_intersecting`** — tier-1 univariate-search 6R for arms with two parallel axes (joints 1, 2) but not three, or two intersecting axes at the wrist (joints 4, 5) but not three. SP6 inside a 1D bisection loop over the remaining unconstrained joint. Slower (~1 ms) and partially complete (the 1D search can miss zero crossings); rarely matches commercial arms, but useful for custom geometries.

The dispatcher (in progress; see Chapter 9 for the public API) inspects the kb's topology at registration time and routes to the highest-tier solver that matches.

## Why subproblem composition is so fast

The whole pipeline for `spherical_two_parallel` on Puma 560 involves:

- 1 × SP4 (one 14-element scalar dot product + one quadratic root): ~5 µs.
- 2 × SP3 (each: one quadratic + an `atan2`): ~5 µs each.
- 4 × SP1 (each: one `atan2` + one cross product): ~3 µs each.
- 1 × SP4 (wrist alignment): ~5 µs.
- 4 × FK forward-kinematics validation: ~10 µs each.

Total: ~80 µs warm-cache, allocations included. Sub-millisecond by an order of magnitude. The arithmetic is dominated by `atan2` calls and small dot products; numpy dispatch overhead is the bottleneck, not LAPACK.

By comparison, the tier-2 RR pipeline on a non-Pieper arm involves a 14×8 SVD, an SVD elimination of 14 rows to 6, a 12×12 matrix assembly, a 24×24 eigendecomposition, and 4–16 back-substitution branches each requiring a 14×8 pseudo-inverse multiplication. ~2.25 ms warm-cache (post the [#86](https://github.com/siddhss5/ikfastpy/issues/86) Tier 1/2 work). 30× slower than tier-0, *and* requiring the entire conditioning-fix machinery from [Chapter 5](05_conditioning.md) to be correct in the first place.

The takeaway: **if your arm is Pieper-class, you get 30× speedup and bulletproof precision for free.** The tier-2 RR machinery exists for arms where Pieper doesn't apply.

## Subproblem robustness — cluster roots

The trickiest subproblem in practice is SP5. Its inner quartic has a **cluster-root pathology** ([#55](https://github.com/siddhss5/ikfastpy/issues/55)): for specific geometric configurations, the four roots split into pairs of nearly-equal values, and `numpy.roots` (which uses companion-matrix eigendecomposition) returns slightly-drifted clusters where two of the four reported roots are off the true roots by ~1e-3 to 1e-4.

Symptom: tests that expect 8 IK solutions get 7 or 9; the spurious solution has FK error ~1e-3 instead of ~1e-12. Fix:

1. Extract candidate roots from the quartic.
2. Apply Gauss–Newton refinement to each root against the original SP5 polynomial.
3. Filter imaginary roots with a scale-aware `|imag| < tol * max(|real|, 1)` test.

After SP5's GN polish, cluster-root cases pass at machine precision. The fix lives in `ssik.subproblems.sp5`; the cross-solver Puma agreement gate ([Chapter 8](08_bulletproof.md)) was the test that surfaced the bug. The same pattern applies to SP6's Bezout quartic, with a slightly different invariant (memory entry [`reference_ikfast_analytical_tricks`](https://github.com/siddhss5/ikfastpy/issues/81) covers the lineage).

This is the kind of stability work that doesn't show up in user-facing API documentation but matters enormously for "must be perfect, all the time" — a 1% cluster-root failure rate on SP5 would invalidate every Pieper-class solver in the library.

## References

- **Pieper, D. L.** (1968). *The Kinematics of Manipulators Under Computer Control.* PhD thesis, Stanford. — The original solvability-via-three-intersecting-axes theorem.
- **Paden, B.** (1986). *Kinematics and Control of Robot Manipulators.* PhD thesis, UC Berkeley. — Subproblem decomposition (SP1–SP3 in modern terminology).
- **Elias, A. & Wen, J.** (2022/2025). "IK-Geo: unified robot inverse kinematics using subproblem decomposition." [arXiv:2211.05737](https://arxiv.org/abs/2211.05737). — Modern unified treatment of SP1–SP6 with the per-family compositions ssik ports.
- **Ostermeier, D.** (2024). "EAIK: a Toolbox for Efficient Analytical Inverse Kinematics." [arXiv:2409.14815](https://arxiv.org/abs/2409.14815). — Family-detection-driven dispatch over the same subproblem library.
