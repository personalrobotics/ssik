# 10. Roadmap

!!! warning "Scaffolding"
    Outline below; prose to be filled in.

## What this chapter covers

The next-order lifts that aren't in ssik today, ranked by user impact.

## Husty–Pfurner universal fallback

[Husty & Pfurner 2007](https://doi.org/10.1115/1.2780537) Study-quaternion method gives a degree-16 univariate polynomial covering **any** 6R arm, even when Raghavan–Roth's $A$ pencil is fundamentally singular (rare but real). Lower registry priority than the tier-2 RR solver; auto-selected on fallback. ~10 ms target latency. Clean-room from the published paper, no LGPL dependency.

## Specialist 7R solvers

For 7-DOF arms with full-redundancy IK (sweeping the 1D redundancy manifold rather than locking one joint to a sample):

- **GeoFIK** for Franka Panda. Specialist solver matched to the SRS topology.
- **Stereographic SEW** (Wenger / NASA 2024) for KUKA iiwa-class arms.
- **Moz1 NonSRS 7R** for Flexiv Rizon and similar non-SRS chains.

Each plugs in as an independent module via `[project.entry-points."ssik.solvers"]`; no core patches needed.

## Codegen / Rust runtime (Phase M)

Per-arm AOT-compile the Raghavan–Roth pipeline to a `.so` extension:

- Replaces the lambdified-sympy `build_pq` callable with a numpy-only or C-array-only callable.
- Replaces the Python loop overhead with a Rust hot loop linking `dgeev` + a hand-vectorised back-substitution.
- Estimated tier-2 latency: ~300–500 µs (vs current 2.25 ms median).
- Estimated tier-0 latency: ~30–50 µs (matches / beats EAIK on the same algorithm).

See [#86](https://github.com/siddhss5/ikfastpy/issues/86) Tier 3 for the speed roadmap.

## Vendored ikfast retirement (Phase K)

[#84](https://github.com/siddhss5/ikfastpy/issues/84). The `_legacy/` tree contains LGPL-tainted code from the abandoned port-and-patch attempt. Once the new pipeline has full conformance against the cases the vendored solver covered (Puma 560 in tier-0, etc.), the legacy tree gets deleted. No code currently imports from it; the cleanup is purely housekeeping + license hygiene.

## IK modes beyond Transform6D

IKFast supported 16 distinct IK modes (Translation3D, Rotation3D, Direction3D, Ray4D, Lookat3D, ...). ssik currently ships Transform6D only. Each non-trivial mode requires its own loop-closure derivation and back-substitution. The reference table is in [`docs/reference/ik-modes/`](../reference/ik-modes/index.md). Tracking issue: [#18](https://github.com/siddhss5/ikfastpy/issues/18).

## MJCF front-end parallel to URDF

Most modern simulation environments use MuJoCo MJCF rather than URDF. Building a `KinBody` from MJCF directly (rather than transcribing joint frames by hand as JACO 2 does) is a separable concern. Tracking: ada-style MJCF parser, parallel to `ssik._urdf`.

## Extensibility — third-party solvers

The Solver / TopologyClaim / SolverRegistry protocols (`src/ssik/core/`) admit external solvers via Python entry points. The aspiration: someone publishes `ssik-geofik` on PyPI, ssik picks it up at registration time, and the dispatcher routes Franka calls to it without core patches. Real example pending; rebuild-plan covers the protocol.

---

This is also the conclusion of the tutorial. The implementation tracking lives on GitHub issues; the math is done; the bulletproof discipline is shipped; the speed work continues. Open questions live in active issues — start there if you want to contribute.
