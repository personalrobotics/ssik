# 10. Roadmap

The previous nine chapters cover what ssik ships today: ten public solvers spanning Pieper-class closed-form (tier-0/1) and non-Pieper numeric Raghavan–Roth (tier-2), all gated by the bulletproof validation discipline of [Chapter 8](08_bulletproof.md), all returning the same `Solution` shape with the algebraic-first / opt-in-Newton contract from [Chapter 6](06_refinement.md). The library closes the EAIK gap for 6R arms and provides joint-locking for 7R arms with one redundant joint.

This chapter walks through what's *not* in ssik today, ranked by user impact. Each item has a tracking issue on GitHub; pick whichever feels closest to your needs and follow the issue thread to see current state.

## Husty–Pfurner universal fallback

[Husty & Pfurner 2007](https://doi.org/10.1115/1.2780537) gives a degree-16 univariate polynomial covering **any** 6R arm via a Study-quaternion parameterisation. It's slower than Raghavan–Roth on well-conditioned arms (the polynomial degree is the same but the back-substitution is more involved) but it has one major advantage: it works on the rare cases where RR's $A$ pencil is *fundamentally singular* — singular pencils that survive even the Möbius reparameterization fallback from [Chapter 5](05_conditioning.md).

The tier-2 RR solver in `ikgeo.general_6r` falls through to a generalised-eigenvalue route (scipy's `linalg.eig` on the matrix pencil $M_1 - x M_2$) when AE-1/3/4 + Möbius all fail. That generalised-eigenvalue route handles most singular-pencil cases. But there exist (rare) 6R configurations where every Möbius transform leaves the pencil singular and the generalised-eigenvalue route still fails. Husty–Pfurner is the universal escape hatch.

Plan: lower registry priority than the tier-2 RR solver; auto-selected on fallback when RR raises `LinAlgError`. Estimated latency ~10 ms (degree-16 polynomial root-finding is cheap; the back-substitution is the cost). Clean-room implementation from the published paper, no LGPL dependency.

Status: not started. Tracking issue: see umbrella rebuild plan.

## Specialist 7R solvers

The joint-locking wrapper (`jointlock.seven_r`) handles any 7R arm by sweeping one redundant joint and dispatching the resulting 6R sub-chain. This works but it doesn't *parametrise* the redundancy — it picks one slice. For applications that need to optimise across the redundancy (collision avoidance, workspace coverage, smooth trajectories), you want a solver that works over the 1D redundancy manifold directly.

Three specialist 7R solvers cover the major commercial families:

- **GeoFIK** for Franka Panda (Emika). Specialist solver matched to Panda's specific SRS topology; closed-form per-redundancy-value.
- **Stereographic SEW** (Wenger / NASA 2024) for KUKA iiwa-class arms. Stereographic projection of the SE(3)-redundancy manifold to a closed-form parametrisation.
- **Moz1 NonSRS 7R** for Flexiv Rizon and similar non-SRS chains. Different algorithmic family for arms whose 7R topology isn't shoulder-elbow-wrist with intersecting axes.

Each plugs in via Python `[project.entry-points."ssik.solvers"]` — independent packages on PyPI that ssik picks up at registration time. Zero core patches needed. The dispatcher routes Franka kinbodies to GeoFIK, iiwa kinbodies to Stereographic SEW, etc., automatically.

Status: not started; registry mechanism exists in the rebuild plan but no specialist solvers ported yet.

## Codegen / Rust runtime (Phase M)

The biggest remaining speed lift on the tier-2 RR path. The current bottleneck is Python-numpy dispatch overhead — every `np.linalg.solve`, every `@`-matmul, every `np.linalg.eig` pays ~1-5 µs of dispatch glue per call, and the RR pipeline has dozens of these per IK. Codegen amortises that overhead.

Two-stage plan:

1. **Per-arm AOT compile of `build_pq`** to a numpy-only callable that doesn't go through sympy's lambdify wrapper. The lambdified callable is effectively a Python function calling numpy ops; replacing it with a single `np.einsum`-based emitter saves ~30-50 µs per IK and removes sympy from the runtime hot path. This is a relatively narrow change inside `_raghavan_roth.py`.

2. **Rust port of the runtime** with Python bindings (`ssik-rust` or similar). The Rust hot path links `dgeev` directly via `lapack-sys`, vectorises the back-substitution branches into a single batched LAPACK call, and skips Python's interpreter-level overhead. Estimated tier-2 latency: ~300-500 µs (vs current 2.25 ms median). Tier-0 closed-form latency: ~30-50 µs, matching or beating EAIK on the same algorithm.

The numbers are bounded by what's amortisable. The 24×24 eigendecomposition itself is ~100-150 µs of pure LAPACK time on standard hardware — that's a hard floor we can't undercut without a fundamentally different algorithm. But the Python interpreter and numpy dispatch overhead is roughly 1-2 ms of the current 2.25 ms median; that's all eliminable with careful codegen.

Status: not started. Tracked under Phase M of the rewrite plan and called out as Tier 3 of [#86](https://github.com/siddhss5/ikfastpy/issues/86).

## Vendored ikfast retirement (Phase K)

The `_legacy/` directory contains the LGPL-licensed vendored OpenRAVE IKFast tree from the abandoned port-and-patch attempt. No code currently imports from it; it exists only because removing it requires a coordinated PR (the legacy conformance tests, the cross-check fixtures, the deprecation shim).

Once the new pipeline has full conformance against the cases the vendored solver covered (Puma 560 in tier-0 — already proven via the cross-validation in [Chapter 8](08_bulletproof.md)), the legacy tree gets deleted. License hygiene only; doesn't affect runtime behaviour.

Status: tracked in [#84](https://github.com/siddhss5/ikfastpy/issues/84). Low-risk, scoped, ready to ship whenever the queue clears.

## IK modes beyond Transform6D

IKFast originally supported 16 distinct IK modes — Translation3D (place a point in space, orientation free), Rotation3D (orient a frame, translation free), Direction3D (point a single axis), Ray4D (drilling / screwing along a line), Lookat3D (camera pointing), and 11 more axis-angle / mixed-DOF combinations. Each is a different loop-closure derivation and a different back-substitution pattern.

ssik currently ships Transform6D only. Adding the other modes requires:

- Per-mode loop-closure derivation (mostly mechanical from the IKFast precedent).
- Per-mode `Solution` semantics (e.g. Translation3D returns up to a 3-DOF redundancy manifold for 6R arms; what does that mean in our `Solution` shape?).
- Per-mode test fixtures (welding seams for TranslationDirection5D, drilling for Ray4D, etc.).

The reference table for the 16 modes lives in [`docs/reference/ik-modes/`](../reference/ik-modes/index.md). Tracking: [#18](https://github.com/siddhss5/ikfastpy/issues/18).

This is a substantial body of work. Realistic timeline depends on demand: most users want Transform6D; the other modes are "specialty". The roadmap is to land them in priority order driven by issue requests.

## MJCF front-end parallel to URDF

ssik currently has a URDF loader (`ssik._urdf.load_urdf_kinbody_normalized`, requires the `[urdf]` extra). For MJCF input, users transcribe joint frames by hand (see [`tests/fixtures/jaco2.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/fixtures/jaco2.py) for the pattern).

A native MJCF loader is a separable concern. The MuJoCo XML schema is well-defined; `mujoco-py` or the lighter `dm_control.mjcf` library can parse it; the conversion to ssik's POE-form `KinBody` is mechanical. Tracking: pending issue.

This matters for the modern simulation ecosystem: `isaac-sim`, `mujoco`, `geodude`, `dm_control` all use MJCF natively. Expecting users to transcribe joint frames or convert MJCF → URDF first is friction.

## Extensibility — third-party solvers

The core architectural commitment is that future analytical-IK algorithms plug in via Python entry points without core patches. The `Solver` / `TopologyClaim` / `SolverRegistry` protocols (in `src/ssik/core/`) admit external modules — someone publishes `ssik-geofik` on PyPI, ssik picks it up at registration time, and the dispatcher routes Franka calls to it.

This isn't aspirational architecture; it's a hard constraint from the rebuild plan. The first specialist 7R solver (GeoFIK or Stereographic SEW) will exercise the protocol end-to-end and prove (or expose problems with) the entry-point boundary. That's where the architecture becomes concrete.

Status: protocols exist; no third-party solvers exercise them yet. The dispatcher is in progress.

## Tutorial completion

This tutorial chapter is the structural conclusion. The previous nine chapters are written in full as of PR #92; the only "to-do" is updates as the library evolves — when the GeoFIK / Stereographic SEW specialists land, [Chapter 10](10_whats_next.md) gets shorter and a new specialist-7R chapter slots between Chapter 6 and Chapter 7. When the codegen / Rust runtime lands, [Chapter 9](09_practical_guide.md)'s performance numbers get updated.

The math (Chapter 4) is stable. The conditioning theory (Chapter 5) is stable. The bulletproof discipline (Chapter 8) is stable. The convention bugs we found in real-arm fixtures (Chapter 7) are stable as cautionary tales. Those are the load-bearing portions of the tutorial; they don't depend on which algorithms ship next.

If you want to contribute, the live issues on GitHub are the entry points. [#82](https://github.com/siddhss5/ikfastpy/issues/82) (MC Table I coverage gap), [#86](https://github.com/siddhss5/ikfastpy/issues/86) (speed Tier 3 / Rust port), [#80](https://github.com/siddhss5/ikfastpy/issues/80) (more real-arm fixtures), [#84](https://github.com/siddhss5/ikfastpy/issues/84) (vendored ikfast retirement) are all in scope. Pick the one you care about; each issue is self-contained enough that you can dive in without reading the entire codebase first.
