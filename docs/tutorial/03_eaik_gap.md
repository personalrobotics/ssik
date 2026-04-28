# 3. The EAIK gap

[Chapter 2](02_pieper_subproblems.md) covered Pieper-class arms — those with three consecutive intersecting wrist axes or three consecutive parallel shoulder axes. EAIK and IK-Geo solve those families in 50–200 µs warm-cache, machine-precision FK closure, all 8 branches at once. For the dominant share of commercial 6R arms (UR3/5/10, Puma 560, Fanuc, KUKA KR, ABB IRB) those tools are excellent and ssik bundles their algorithms via the BSD-3 IK-Geo port.

The arms whose IK is **not** a one-line call against EAIK or IK-Geo — the ones whose geometry deliberately violates Pieper's condition for mechanical-design reasons — are the EAIK gap. ssik exists for them. This chapter catalogues the gap, walks through why the obvious workarounds don't work, and frames the strategic positioning of the library.

## Arms outside the Pieper class

**Kinova JACO 2 (j2n6s200).** Six revolute joints. Joints 1–3 form a conventional shoulder-elbow-elbow configuration. The wrist (joints 4, 5, 6) is **non-orthogonal**: the consecutive twists between joints 4 and 5, and between joints 5 and 6, are 60° rather than the 90° that would put the axes on standard orthogonal planes.

Concretely, the MJCF places `link_5` and `link_6` with `quat = (0, 0, 0.5, 0.866025)` — a 60° rotation around the local z-axis. As a result, no two wrist axes are parallel, no two intersect, and the wrist isn't spherical. Joints 4-5-6 form a **non-Pieper** sub-chain, and the standard subproblem compositions can't apply: SP4 (used for spherical-wrist alignment) requires the three wrist axes to share a common point; SP5 (used for the generic shoulder-elbow-wrist position chain) develops degenerate cluster roots near these geometries. Real fixture in [`tests/fixtures/jaco2.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/fixtures/jaco2.py), transcribed from `robot-code/ada_assets/.../jaco2.xml`.

The 60° twists are intentional — they let Kinova package the wrist's three motors into a more compact volume than an orthogonal wrist would allow, at the cost of giving up Pieper's analytical IK. Modern compact arms make this trade-off frequently: motor packaging volume traded against IK closed-form solvability.

**Agilex Piper.** A 6R arm with similar non-Pieper geometry: the wrist twists aren't orthogonal, and no axis triple is parallel. (Less commercially deployed than JACO 2 but follows the same compactness-driven design philosophy.)

**Flexiv Rizon 4.** A 7-DOF arm. The 7-DOF case is more general than the 6-DOF case — IK on a 7R has a continuous **redundancy manifold** of solutions for any given target pose — but a common reduction is **joint-locking**: fix one joint at a sample value, solve the remaining 6R. ssik's `jointlock.seven_r` does this generically (sweep one configurable joint over N samples; dispatch each lock value to the best-matching 6R inner solver). For Rizon 4 the locked sub-chains are non-Pieper at every lock value because the wrist geometry is non-orthogonal — same gap as JACO 2.

**Custom geometries.** Lab manipulators, prototype arms from kinematic-design exercises, "what if my wrist twist were 47 degrees" experiments. These often deliberately break Pieper because the designer doesn't care about closed-form IK or wants to test the general case. They're the arms ssik users running their own kinematic prototyping bring.

## Why the obvious approaches don't work

**Subproblem composition (EAIK / IK-Geo).** The geometric specialisations don't apply: there's no spherical wrist for SP4 to project onto, no parallel-axis trio for SP6's Bezout setup. The `ikgeo.spherical` solver falls back to SP5 for the shoulder-elbow-wrist position chain when no specialisation matches, but SP5 itself develops structural degeneracy on these chains — its quartic has near-triple roots when the wrist twist isn't orthogonal, which break the cluster-root recovery from [Chapter 2](02_pieper_subproblems.md).

In practice, calling `ikgeo.spherical.solve(jaco2_kb, T_target)` raises `ValueError` at the topology gate: the predicate `three_consecutive_intersecting(joints, policy)` returns `(-1, -1, -1)` instead of `(3, 4, 5)` — the wrist axes don't intersect at any single point.

**IKFast (Diankov 2010, OpenRAVE).** The original IKFast pipeline — symbolic codegen via sympy — was supposed to handle anything. Modern reality:

- The codegen depends on a 2010-vintage `sympy.polys` API that has been substantially rewritten since. Module-level imports fail on `sympy >= 1.5`.
- The `solveLiWoernleHiller` numerical step calls `mpmath.polyroots` on a degree-16 univariate polynomial. On modern `mpmath`, this raises `NoConvergence` after `maxsteps=5000` iterations on real-arm parameter sets. Verified on UR5 (797 s before failure on the cambel URDF), Puma-with-d5=0.01 (617 s, NoConvergence), POE-normalized UR5, and JACO 2.
- Even when the symbolic derivation does complete, it takes hours-to-days per arm. The output is a single C++ file specialised to that one robot's DH parameters — change one number and you re-derive from scratch.
- The vendored copy in this repo's `_legacy/` (slated for deletion in [#84](https://github.com/siddhss5/ikfastpy/issues/84)) is a museum piece. It works on the few specific 2010-era examples that still pass through modern sympy; it doesn't work on JACO 2 or Piper or any of the post-2015 arms ssik users actually need to solve.

The polynomial-coefficient cliff that breaks IKFast on JACO 2 is the *same* conditioning failure that we covered in [Chapter 5](05_conditioning.md) — a singular pencil that the textbook algorithm can't survive. IKFast's response was "wait longer for `mpmath.polyroots`" (i.e. keep iterating with higher precision); ssik's response is the AE-1/3/4 conditioning suite that makes the algebraic problem well-conditioned to begin with.

**Numeric IK (mink, KDL, TRAC-IK, Pinocchio, drake).** These work — every modern numerical IK library handles non-Pieper 6R fine — but pay the iterative cost. On real benchmarks:

| solver | JACO 2 IK time | what it returns |
|---|---|---|
| mink (Python wrapper, JIT-warmed) | ~20 ms | one solution |
| KDL (C++ via ROS) | ~1-5 ms | one solution |
| ssik tier-2 RR (current, post-#86 Tier 2.3) | ~2.25 ms median | all 4-16 branches |

mink and KDL give you one solution. ssik gives you all of them — in a single call, with stable `branch_id` indices that let you track which physical configuration is which across a trajectory. Numeric solvers also have no up-front "unreachable" signal: if the target is unreachable, mink returns the closest reachable approximation rather than telling you it failed. ssik's `is_ls=True` explicitly distinguishes "no candidate met `fk_atol`" (could be unreachable, could be ill-conditioned) from "found N solutions".

## What this means for ssik's design

Three architectural decisions follow from the gap analysis:

**1. ssik bundles tier-0/1 closed-form solvers** (port of IK-Geo) for Pieper-class arms. We don't make the user pay the tier-2 cost when their arm has the Pieper specialisation — they get the same 50-200 µs latency EAIK gives them, with the same `Solution` shape and refinement contract from [Chapter 6](06_refinement.md).

**2. ssik's tier-2 numeric Raghavan–Roth solver** ([Chapter 4](04_raghavan_roth.md)) handles the gap arms. The math has been around since 1990 (Raghavan–Roth) and 1994 (Manocha–Canny); what's new is the **conditioning robustness** ([Chapter 5](05_conditioning.md)) that makes the algebraic pipeline survive on real ill-conditioned arm geometries. Without AE-3 leftvar selection, the textbook RR pipeline fails on JACO 2 — same failure mode as IKFast.

**3. The dispatcher** picks the right tier per kb at registration time (work in progress; see [Chapter 9](09_practical_guide.md)). Users call `solve(kb, T)` and don't think about tiers; the library routes to `ikgeo.three_parallel` for UR5, `ikgeo.spherical_two_parallel` for Puma, and `ikgeo.general_6r` (the tier-2 RR) for JACO 2 — all returning the same `tuple[list[Solution], bool]` shape.

## The strategic frame

EAIK and IK-Geo cover the easy 80% of commercial 6R arms. ssik covers the harder 20% — the EAIK gap. That's where a millisecond-level analytical IK solution didn't exist before.

Memory entry [`project_eaik_gap_strategy`](https://github.com/siddhss5/ikfastpy/issues/78) frames the positioning:

> Non-Pieper 6R (JACO 2, Piper) + non-SRS 7R (Rizon 4) is the wedge. Duplicating EAIK's Pieper/SRS coverage adds nothing.

The library's value isn't in being a faster EAIK — it isn't, on Pieper-class arms (we run IK-Geo's algorithm; can't beat it without codegen). The value is in covering the arms EAIK and IK-Geo *don't* cover, with the same precision contract and the same `Solution` API the rest of the library uses. A user with a fleet of mixed UR5s and JACO 2s shouldn't have to write two different solver paths; ssik gives them one API.

## What the rest of the tutorial covers

Chapters 4 and 5 are the load-bearing technical chapters: the Raghavan–Roth math and the conditioning fixes that make it survive on real arms. Chapters 6, 7, 8 cover the architectural pieces — the `Solution` / `lm_refine` contract, the POE → DH bridge with the JACO 2 `T_pre` bug fix, and the bulletproof validation discipline. Chapters 9 and 10 are practical-guide and roadmap.

If you're here because you have a JACO 2 / Piper / Rizon-style arm and need analytical IK, [Chapter 9](09_practical_guide.md) is the practical entry point and [Chapter 4](04_raghavan_roth.md) is the technical backstop. If you're here because you're porting analytical IK to a new arm family, [Chapter 5](05_conditioning.md) is essential reading: the conditioning intuitions transfer to any algebraic-elimination IK pipeline.

## References

- **Raghavan, M. & Roth, B.** (1990). "Inverse kinematics of the general 6R manipulator and related linkages." *Journal of Mechanical Design*. — The 14-equation algebraic-elimination foundation of the tier-2 pipeline.
- **Manocha, D. & Canny, J. F.** (1994). "Efficient inverse kinematics for general 6R manipulators." *IEEE Transactions on Robotics and Automation* 10(5):648–657. — The companion-matrix eigenvalue route + Möbius reparameterization fallback.
- **Tsai, L.-W.** (1999). *Robot Analysis: The Mechanics of Serial and Parallel Manipulators.* Wiley. Appendix C. — Pedagogical treatment of the RR derivation that ssik's clean-room implementation follows.
- **Ostermeier, D.** (2024). [EAIK](https://github.com/OstermD/EAIK). — Where the gap analysis pivots: EAIK explicitly doesn't cover the cases ssik does.
