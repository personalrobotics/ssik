# 1. The IK problem

A robot arm has $n$ joints. Each joint has a single scalar angle $q_i \in \mathbb{R}$. The configuration $q = (q_1, \ldots, q_n)$ determines, via a deterministic chain of rigid transforms, the pose of the end effector:

$$
T = \mathrm{FK}(q) \in \mathrm{SE}(3).
$$

That is **forward kinematics**: configuration $\to$ pose. It is straightforward, fast, and unique.

The inverse — **inverse kinematics** — asks the dual question. Given a target pose $T^\star$, what configurations $q$ satisfy $\mathrm{FK}(q) = T^\star$?

For a 6-DOF arm reaching a 6-DOF target ($\mathrm{SE}(3)$ has six dimensions: three translation, three rotation), this is generically solvable but typically has **multiple discrete solutions** — up to 16 for a generic 6R chain — corresponding to different elbow / wrist / shoulder branchings. For 7-DOF and higher, there is a continuous redundancy manifold of solutions: every target pose has infinitely many configurations realising it.

## Numeric vs analytic IK

There are two solution strategies, and they differ in what they ask the user to give up.

**Numeric IK** treats $\mathrm{FK}(q) - T^\star$ as a residual to drive to zero with iterative methods — Newton on the spatial Jacobian, damped least squares, gradient descent on a manifold-aware loss. Tools: TRAC-IK, KDL, mink. They handle anything: arbitrary arm topology, joint limits, redundancy. The price is iterative cost (50–200 µs per IK on a tuned implementation; 1–10 ms on Python wrappers), local-minimum risk near singularities, and a single returned solution rather than the full set of branches. They never tell you up front whether the target is reachable.

**Analytic IK** writes the loop-closure equations algebraically and solves them in closed form. Output: every solution at once, with a clear "no real roots ⇒ unreachable" signal. Cost: nanoseconds-to-milliseconds depending on the algorithm. Catch: **the algorithm is arm-specific**. A solver that handles UR5 might not handle JACO 2, and the closed form needs derivation per kinematic family.

This tutorial is about the analytic side. ssik is an analytic IK library.

## The analytical-IK ecosystem

Three families of tools share this corner of the field:

**IKFast** (Diankov 2010, OpenRAVE). Per-robot symbolic codegen. You feed it a kinematic chain, it grinds for hours/days, emits a single C++ file specialised to that arm. Fast at runtime, brittle in derivation: the symbolic pipeline depends on a 2010-vintage sympy and the modern sympy / mpmath stack causes `polyroots NoConvergence` on real arms (UR5, Puma-with-tiny-d, JACO 2). It's not maintained. The vendored copy in this repo's `_legacy/` is an artifact of an earlier resuscitation attempt.

**EAIK** (Ostermeier et al., 2024) and **IK-Geo** (Elias & Wen, 2022). Subproblem-composition closed-form solvers covering Pieper-class arms — those with three consecutive intersecting wrist axes, or three consecutive parallel shoulder axes, or both. Six canonical subproblems (SP1–SP6), each a closed-form circular geometry intersection, compose into per-family analytic IK. Industrial 6R arms — UR3/5/10, Puma 560, Fanuc, KUKA KR, ABB IRB — are all Pieper-class. EAIK and IK-Geo handle them in roughly 0.2 ms with machine-precision closure. **Chapter 2 of this tutorial walks through how this works.**

**Numeric solvers** as above. Strong fallback when no analytic family applies.

ssik bundles tier-0 closed-form solvers ported from IK-Geo (so it covers the same Pieper-class arms EAIK does), and adds a **tier-2 numeric Raghavan–Roth solver** that closes the gap on non-Pieper arms — the EAIK gap. That's the part of the library that didn't already exist.

## What ssik solves that the alternatives don't

The arms whose IK isn't already a one-liner against EAIK or IK-Geo:

- **Kinova JACO 2 (j2n6s200)**: 60° non-orthogonal twists at joints 4–5. No three consecutive intersecting axes. No parallel pair. Subproblem composition's prerequisite geometric specialisations don't apply.
- **Agilex Piper**: similar gap.
- **Flexiv Rizon 4**: a 7-DOF non-SRS arm. Joint-locking turns it into a 6R per lock value, but the sub-chain is still non-Pieper.
- **Custom geometries** that arise from machined-from-scratch arms or kinematic prototyping.

For these, neither closed-form subproblem composition (no specialisation matches) nor IKFast (its symbolic codegen breaks on these chains) gets you analytic IK. Numeric IK works but pays the iterative cost and gives you one solution, not all branches.

ssik's contribution is making the **Raghavan–Roth 1990 / Manocha–Canny 1994** algebraic-elimination pipeline survive on real arms — single-digit-millisecond latency, all branches at once, machine-precision FK closure. Doing this required four independent robustness fixes against ill-conditioning that the textbook algorithm doesn't survive. **Chapter 4** walks the math; **Chapter 5** walks the robustness. Together they describe what's actually shipping.

## What this tutorial assumes

Working knowledge of:

- 4×4 homogeneous transforms and rotation matrices.
- DH parameters or Product-of-Exponentials (POE) — we use POE as the canonical input format and convert to DH internally; either notation is fine for following along.
- Eigenvalue / eigenvector basics. Singular value decomposition shows up briefly.

The math here is approachable from a robotics-aware undergraduate background; the harder material lives in the robustness (Chapter 5) and is presented with the runtime intuition rather than a formal numerical-analysis treatment.
