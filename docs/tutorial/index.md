# Tutorial

A guided tour through the algorithms ssik actually ships, focused on the part of the analytical-IK ecosystem that motivated the library: the **EAIK gap**.

Most analytical IK on commercial 6R arms has been a solved problem for thirty years. Subproblem composition (Paden–Kahan, Pieper, IK-Geo) handles arms with three intersecting axes (spherical wrist) or three consecutive parallel axes; that covers UR5, Puma 560, Fanuc, KUKA KR, ABB IRB, and most industrial manipulators. EAIK ships closed-form solvers for those families in ~0.2 ms.

The arms ssik exists for are the ones that **don't** fit those families: Kinova JACO 2 (60° non-orthogonal twists at joints 4–5), Agilex Piper, Flexiv Rizon 4 (non-SRS 7R), and any custom 6R chain where no axis triple is parallel or intersecting. For those arms ssik runs the **Raghavan–Roth + Manocha–Canny** numeric pipeline — a 14-equation algebraic elimination, a 24×24 companion-matrix eigendecomposition, and a back-substitution — and gets analytical IK at single-digit-millisecond latency with machine-precision FK closure.

This tutorial walks the math step by step, names every robustness trick we needed to make the pipeline bulletproof on real arms, and shows the actual code paths that implement them.

## Structure

1. [The IK problem](01_ik_problem.md) — forward vs inverse, why analytical when you can, the ecosystem ssik plugs into.
2. [The Pieper class](02_pieper_subproblems.md) — three-intersecting-axes, subproblem composition, SP1–SP6, what tier-0 solvers look like.
3. [The EAIK gap](03_eaik_gap.md) — the arms nobody else covers analytically, and why the obvious approaches don't work on them.
4. [Raghavan–Roth in 8 stages](04_raghavan_roth.md) — the math the tier-2 numeric solver runs, with code pointers at each stage. **Load-bearing technical chapter.**
5. [Conditioning is the hard part](05_conditioning.md) — why the textbook RR pipeline blows up on JACO 2, and the four independent attacks (AE-1, AE-3, AE-4, Möbius) we shipped to make it survive. **Load-bearing technical chapter.**
6. [Algebraic-first, refinement-second](06_refinement.md) — the GitHub #74 contract: pure algebraic by default, opt-in Newton polish, transparent diagnostics.
7. [The KinBody-input bridge](07_kinbody_bridge.md) — POE-normalized chains in, DH-form solver under the hood, and the load-bearing bug that hid until JACO 2 exposed it.
8. [Bulletproof validation](08_bulletproof.md) — N-way cross-solver agreement, 500-pose hypothesis fuzz, machine-precision FK on real-MJCF fixtures.
9. [Practical guide](09_practical_guide.md) — installing, picking a solver, reading `Solution.refinement_used`, debugging `is_ls=True`, performance numbers.
10. [Roadmap](10_whats_next.md) — Husty–Pfurner universal fallback, specialist 7R, codegen / Rust runtime.

## Conventions

- Math uses KaTeX inline ($A x^2 + B x + C = 0$) and block:
  $$
  M(x_2) = A x_2^2 + B x_2 + C
  $$
- Code references use clickable file links: [`src/ssik/solvers/ikgeo/_raghavan_roth.py`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py).
- Cited literature is in the [Bibliography](../bibliography.md).
- GitHub issue numbers in parens (e.g. **#82**) link to the canonical tracking thread for ongoing work.

## Status

Chapters 1, 4, 5 are complete. Chapters 2, 3, 6–10 are scaffolded outlines being filled in alongside the public-API stabilisation. See [#87](https://github.com/siddhss5/ikfastpy/issues/87) for the rewrite tracking issue.
