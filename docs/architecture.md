# Architecture

## Solver tier catalog

ssik dispatches to one of 11 analytical solvers based on the arm's kinematic topology. The dispatcher (`ssik.core.dispatcher.dispatch`) classifies the chain via topology predicates (`three_consecutive_parallel`, `three_consecutive_intersecting`, `is_srs_7r`, etc.) and picks the highest-tier solver that matches.

| Tier | Solver modules | Typical IK time | Algorithm |
|---|---|---|---|
| 0 — closed-form 6R | `three_parallel`, `spherical_two_parallel`, `spherical_two_intersecting`, `spherical` | ~1 ms | SP1–SP6 composition; one branch per Pieper specialisation |
| 0 — closed-form 7R (SRS) | `seven_r.srs` | ~5 ms full sweep | Singh-Kreutz 1989 parameterised by elbow swivel angle; 8 branches × 16 swivel samples = 128 IKs (vectorised inner loop, [#217](https://github.com/personalrobotics/ssik/issues/217)). Per-candidate FK verify deferred to post-dedup (#246). |
| 0 — approximate SRS + LM polish | `seven_r.srs_polished` | ~40 ms full sweep | Relaxed Singh-Kreutz (small-drift arms) + batched LM polish to machine precision against the original URDF FK |
| 1 — univariate search | `two_parallel`, `two_intersecting` | ~100 ms – 2 s | tan-half-angle reduction + 200-sample search + Newton polish |
| 1 — 7R joint-lock wrapper | `jointlock.seven_r` | ~5-30 ms tier-0 inner; **~17 ms with cached-RR ([#210](https://github.com/personalrobotics/ssik/issues/210))** when artifact-built | lock one joint, dispatch inner 6R, sweep 16 lock samples; Raghavan-Roth pre-baked at codegen for non-Pieper sub-chains |
| 2 — Raghavan–Roth + Manocha–Canny | `ikgeo.general_6r` | ~0.6-5 ms | numeric RR resultant with AE-3 leftvar selection; **production tier-2** |
| 2 — Husty-Pfurner universal fallback | `husty_pfurner.general_6r` | ~25-200 ms | Study-quaternion algebra; perturbation path ([#176](https://github.com/personalrobotics/ssik/issues/176)) handles symmetric-DH singularities; backstops RR on ill-conditioned arms |

## Dispatch flow

1. **Load**: `Manipulator.from_urdf(path, base, ee)` parses the URDF, resolves the kinematic chain between `base` and `ee`, and POE-normalises every joint frame so axes live in the base frame at q=0.

2. **Classify**: `dispatch(kinbody, policy)` evaluates topology predicates in tier order (closed-form first). The first matching predicate determines the solver; ties broken by specialisation ranking (e.g. spherical_two_parallel vs spherical_two_intersecting).

3. **Solve**: `arm.ik(T_target, **kwargs)` looks up the dispatched solver module, filters kwargs by its signature (so `q_seed` is silently ignored on solvers that don't accept it), and calls `solve()`. The result `(list[Solution], is_ls)` flows back unchanged.

4. **(Optional) Refine**: with `allow_refinement=True`, candidates whose algebraic FK closure misses the policy threshold are run through one Levenberg-Marquardt iteration on the spatial Jacobian. Off by default — the analytical path is exact for well-conditioned poses.

## Per-arm artifact pipeline (`ssik build`)

For production deployment, `ssik build` emits a self-contained `.py` file per arm:

```
ssik build my_arm.urdf --base base_link --ee tool0
# → my_arm_ik.py
```

The artifact contains:
- The KinBody constants (joint axes, T_left/T_right transforms) inlined as numpy literals
- The dispatch decision baked at build time (no runtime classification)
- For non-Pieper sub-chains: cached Raghavan-Roth symbolic derivations as base85-encoded zlib-compressed pickle blobs (#210 Phase 2 / #220)
- A `solve(T_target, **kwargs)` function with the same signature as `Manipulator.ik()`

Module-init time loads the cached derivations and primes the runtime cache (~5 seconds for typical 7R arms). Every subsequent `solve()` call hits warm-cache speed.

## Algorithmic lineage

ssik's solver code is clean-room from published math. Algorithmic credits (already cited in module docstrings):

- **IK-Geo** (Elias-Wen 2022, [arXiv:2211.05737](https://arxiv.org/abs/2211.05737)) — BSD-3 Rust reference. Source for the SP1-SP6 subproblem family and the spherical-class composition (`spherical_two_parallel`, `spherical_two_intersecting`, `three_parallel`, `spherical`).
- **Raghavan–Roth 1990** + **Manocha–Canny 1994** — non-Pieper 6R via 24×24 companion matrix eigendecomposition. Foundation of `ikgeo.general_6r`.
- **AE-3 leftvar selection** (ssik-original; [#70](https://github.com/personalrobotics/ssik/issues/70)) — pick the spectral parameter that puts pathological joints out of the linearity variable. Drops cond(m_quad) from 3.75e16 → 127 on JACO 2 (14 orders of magnitude).
- **Singh–Kreutz 1989** — closed-form 7R for SRS-class arms. Foundation of `seven_r.srs`.
- **Husty–Pfurner 2007** + **Capco-Manongsong** giac code (Zenodo 3157441, MIT) — universal 6R fallback via Study-quaternion algebra. Foundation of `husty_pfurner.general_6r`.

## Internal modules

```
src/ssik/
├── __init__.py              # public exports (Manipulator, Solution, KinBody, ...)
├── manipulator.py           # the v1.0 entry point
├── _kinbody.py              # KinBody / Joint / Link dataclasses (private impl)
├── _urdf.py                 # urchin → KinBody bridge (private impl)
├── _pencil.py               # numerical pencil eigsolve (private)
├── cli.py                   # `ssik build`, `ssik add-arm`
├── core/                    # dispatch, tolerances, Solution, codegen
├── kinematics/              # POE-FK, POE→DH, predicates, reverse-chain
├── subproblems/             # SP1-SP6 closed-form primitives + _rotation
├── solvers/
│   ├── ikgeo/               # tier-0/1/2 ikgeo family
│   ├── seven_r/             # SRS strict + polished
│   ├── jointlock/           # 7R joint-locking wrapper
│   └── husty_pfurner/       # universal 6R fallback
├── refinement/              # opt-in Newton polish layer
└── codegen/                 # `ssik build` artifact emitter
```

The leading underscore on `_kinbody`, `_urdf`, `_pencil` signals "implementation detail; use `ssik.X` instead". Internal modules are not part of the public API contract.
