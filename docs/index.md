# ssik

Analytical inverse kinematics for Python, built around the **EAIK gap** — the arms whose IK isn't a one-line call against EAIK or IK-Geo.

For Pieper-class arms (UR5, Puma 560, Fanuc, KUKA KR, ABB IRB) ssik bundles tier-0 closed-form solvers ported from IK-Geo and runs them at ~50–200 µs warm-cache, machine-precision FK closure. For non-Pieper arms (Kinova JACO 2, Agilex Piper, Flexiv Rizon 4, custom geometries with no parallel/intersecting axis triples) ssik runs a tier-2 numeric **Raghavan–Roth + Manocha–Canny** pipeline at ~2.25 ms median warm-cache, FK error 3.7e-13, all branches at once. That tier-2 solver is what ssik adds to the ecosystem — analytical IK for the arms nobody else covers analytically.

## Audience

- **Users** wanting a fast, closed-form IK solver in Python — start with the [Practical guide](tutorial/09_practical_guide.md) or jump to the [Reference](reference/index.md).
- **Learners** wanting to understand how analytic IK actually works — the [Tutorial](tutorial/index.md) walks from the IK problem through the Pieper class, the EAIK gap, the Raghavan–Roth pipeline, and the four robustness techniques (AE-1/3/4 plus Möbius reparameterization) that make the math survive on real arm geometries.
- **Robotics implementers** porting analytical IK to a new arm family — Chapter 5 catalogs the conditioning patterns we ran into and how to recognise them in your own derivation.

## What this is

A Python-native analytical IK framework with a pluggable solver registry. Tier-0 subproblem-composition solvers (port of IK-Geo, BSD-3) ship by default. Tier-2 numeric Raghavan–Roth ships by default for the EAIK gap. Tier-1 univariate-search and tier-2 grid-search solvers are also bundled. Husty–Pfurner universal fallback and specialist 7R solvers (GeoFIK, stereographic-SEW) plug in via Python entry points without core patches.

Unlike prior Python projects sharing the legacy `ikfastpy` name (e.g. [andyzeng/ikfastpy](https://github.com/andyzeng/ikfastpy), [yijiangh/ikfast_pybind](https://github.com/yijiangh/ikfast_pybind)), `ssik` is **not** a runtime wrapper around pre-generated C++. The solvers are pure Python + numpy; sympy is used for offline per-arm preprocessing (cached) and isn't on the runtime hot path.

## Status

Pre-alpha, mid-rebuild. The original `ikfastpy` project (a port of OpenRAVE's IKFast) was renamed to `ssik` and is being rebuilt around the subproblem-decomposition + Raghavan–Roth approach. Implementation is tracked in the [GitHub issues](https://github.com/siddhss5/ikfastpy/issues). Recent landings:

- Real Kinova JACO 2 j2n6s200 fixture transcribed from MJCF (#80).
- `Solution` dataclass + universal `ssik.refinement` opt-in Newton-polish layer (#74, #75).
- POE → DH bridge with the `T_pre` fix that broke on JACO 2 (#79).
- Speed Tier 1/2.x: per-arm `poe_to_dh` cache + drop redundant POE-FK + per-pose pinv cache + vectorised eigenvalue filter (#86 PRs #88, #89, #90). Median JACO 2 IK: 4.5 ms → **2.25 ms**.

## Acknowledgements

The rebuild draws on the subproblem-decomposition approach of Elias & Wen ([IK-Geo](https://arxiv.org/abs/2211.05737), 2022/2025), Ostermeier ([EAIK](https://arxiv.org/abs/2409.14815), 2024), and Husty & Pfurner (2007), and on the algebraic-elimination pipeline of Raghavan & Roth (1990) + Manocha & Canny (1994). The vendored IKFast tree (slated for removal in [#84](https://github.com/siddhss5/ikfastpy/issues/84)) originates from Rosen Diankov's work at Carnegie Mellon. See the [Bibliography](bibliography.md) for primary sources.
