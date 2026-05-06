# ssik

Analytical inverse kinematics for non-Pieper 6R arms — the **EAIK gap**. The arms whose IK isn't a one-line call against EAIK or IK-Geo (Kinova JACO 2, Agilex Piper, Flexiv Rizon, custom geometries with no parallel or intersecting axis triples).

> **Private repository.** This codebase is proprietary. Source distribution is not authorised; see [`LICENSE`](LICENSE). The user-facing artifact is a per-arm compiled wheel built from this codebase; users do not run this source directly. See [#95](https://github.com/siddhss5/ikfastpy/issues/95) for the distribution model.

## What ssik does that the alternatives don't

| arm class | EAIK / IK-Geo | mink / KDL (numeric) | ssik (full sweep, all branches) |
|---|---|---|---|
| Pieper-class (UR5, Puma 560, Fanuc, KUKA KR) | ~0.2 ms | ~20 ms | ~1-2 ms |
| Non-Pieper 6R (JACO 2, Piper) | not supported | ~20 ms | **~0.6 ms median** |
| SRS-class 7R (KUKA iiwa, Rizon 4, Gen3, Sawyer, Kassow) | sub-ms | ~30 ms | **~8.5 ms (128 IKs)** |
| Anthropomorphic 7R (Franka Panda, FR3) | sub-ms | ~30 ms | **~20 ms (48 IKs) via joint-locking** |

Two differentiators:

1. **Non-Pieper 6R analytical solver** (`ikgeo.general_6r`, Raghavan–Roth + Manocha–Canny). No other library in the ecosystem ships analytical IK for arms whose geometry deliberately violates Pieper's condition for mechanical-design reasons (JACO 2 with 60° non-orthogonal twists is the canonical example). ssik does, with all branches recovered at machine precision in sub-ms. See `docs/tutorial/04_raghavan_roth.md` for the math and `docs/tutorial/05_conditioning.md` for the robustness fixes (AE-1, AE-3, AE-4, Möbius reparameterisation) that make the textbook pipeline survive on real ill-conditioned arms.
2. **Native SRS-class 7R solver** (`seven_r.srs`, Singh-Kreutz 1989) — closed-form parameterised by elbow swivel angle. Predicate-driven dispatch (`is_srs_7r`) auto-applies to any 7R arm matching the topology: shoulder axes (0,1,2) concurrent + wrist axes (4,5,6) concurrent. iiwa14 is the canonical fixture. Many arms cited as SRS in the literature (Gen3, Rizon 4, Kassow KR*) actually have non-zero shoulder/wrist offsets in their URDFs and are routed through the universal `jointlock+HP` fallback today; a generalised approximate-SRS + Newton-polish solver (#193) is the planned fast path for the small-drift cases.

## Supported arms & solver coverage

Status legend: ✅ in-repo URDF/MJCF fixture exercised by the test suite — 🔗 external URDF, fixture import pending — 📐 synthetic-only (no canonical commercial arm with this exact topology). Speed is typical median IK-call latency on Apple M3 single-thread, pure Python+numpy.

### 6R industrial arms

| Arm | Solver | Speed | Status |
|-----|--------|:-----:|:-----:|
| Universal Robots UR3 / UR5 / UR10 / UR16 | `ikgeo.three_parallel` | ~1-2 ms | ✅ UR5 in [`tests/fixtures/`](tests/fixtures/), others 🔗 |
| Puma 560 | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | ~1-2 ms | ✅ |
| ABB IRB120 / IRB4600 | `ikgeo.spherical_two_parallel` | ~1-2 ms | 🔗 [ros-industrial/abb](https://github.com/ros-industrial/abb) |
| Fanuc LR Mate / CR | `ikgeo.spherical_two_parallel` | ~1-2 ms | 🔗 (vendor URDF) |
| KUKA KR | `ikgeo.spherical_two_parallel` | ~1-2 ms | 🔗 [ros-industrial/kuka_experimental](https://github.com/ros-industrial/kuka_experimental) |
| uFactory lite6 / xArm6 | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | ~1-2 ms | 🔗 [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros) |

### 6R non-Pieper (the EAIK gap)

| Arm | Solver | Speed | Status |
|-----|--------|:-----:|:-----:|
| **Kinova JACO 2 (j2n6s200, 55° DH)** | `ikgeo.general_6r` (RR + AE-3) | **~5 ms median** | ✅ real MJCF in [`tests/fixtures/`](tests/fixtures/) |
| Agilex Piper | `ikgeo.general_6r` | expected ~5 ms | 🔗 [mujoco_menagerie/agilex_piper](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper) |
| Custom non-Pieper 6R | `ikgeo.general_6r` | expected ~5 ms | URDF intake via [#95](https://github.com/siddhss5/ikfastpy/issues/95) |

### 7R redundant arms — pure SRS (`seven_r.srs`)

Closed-form Singh-Kreutz 1989 algorithm for arms with shoulder-spherical + wrist-spherical topology + elbow roll. Predicate-driven dispatch via `is_srs_7r` (no per-arm hardcoding); auto-applies to any 7R chain whose shoulder axes (0,1,2) and wrist axes (4,5,6) each meet at a common point. Default 16 swivel samples × 8 branches = 128 IK candidates per call.

| Arm | Full-sweep speed | Status |
|-----|:---:|:---:|
| **KUKA iiwa LBR 14** | **~8.5 ms (128 IKs, FK ≤ 1e-13)** | ✅ in [`tests/fixtures/`](tests/fixtures/) |
| KUKA iiwa LBR 7 (R820 / R14) | expected ~8.5 ms | 🔗 [mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14) |

### 7R redundant arms — non-SRS (`jointlock.seven_r`)

For 7R arms whose topology doesn't match strict SRS, `jointlock.seven_r` is the universal fallback: locks one joint (auto-selected by topology rank of the resulting 6R sub-chain) and dispatches the 6R IK to the best-matching tier-0/1 ikgeo solver. Tier-2 fallback inside jointlock is `husty_pfurner.general_6r` for sub-chains with no Pieper / parallel-axis structure. 16-sample lock sweep × inner 6R solver per call.

Includes the "literature-SRS but URDF-non-SRS" arms — those whose published DH classifies as SRS but whose URDFs carry mm- to cm-scale shoulder/wrist offsets. ssik solves their **real URDF** kinematics, not a simplified DH; a generalised approximate-SRS + LM-polish path (#193) is the planned fast solver for the small-drift cases.

| Arm | Drift (shoulder / wrist) | Full-sweep speed | Status |
|-----|---|:---:|:---:|
| Franka Emika Panda / FR3 | non-SRS by design | ~20 ms (48 IKs) | ✅ in [`tests/fixtures/`](tests/fixtures/) |
| uFactory xArm7 | non-SRS by design | ~32 ms (56 IKs) | ✅ in [`tests/fixtures/`](tests/fixtures/) |
| **Kinova Gen3 (7-DOF)** | 12 mm / 0.4 mm | ~1 s (jointlock+HP) | ✅ in [`tests/fixtures/`](tests/fixtures/) — #193 polished-SRS candidate |
| **Flexiv Rizon 4** | 65 mm / 151 mm | ~1.5 s (jointlock+HP) | ✅ in [`tests/fixtures/`](tests/fixtures/) — wrist drift outside Newton basin |
| Kassow KR810 / KR1410 | ~50 mm shoulder | not measured | 🔗 (vendor URDF) |
| Sawyer / Baxter (Rethink Robotics) | not measured | not measured | 🔗 (vendor URDF) |

### Solver tier reference

| Tier | Solver modules | Typical IK time | Algorithm |
|---|---|---|---|
| 0 — closed-form 6R | `three_parallel`, `spherical_two_parallel`, `spherical_two_intersecting`, `spherical` | ~1 ms | SP1–SP6 composition; one branch per Pieper specialisation |
| 0 — closed-form 7R (SRS) | `seven_r.srs` | ~8.5 ms full sweep | Singh-Kreutz 1989 parameterised by elbow swivel angle; 8 branches × 16 swivel samples = 128 IKs |
| 1 — univariate search | `two_parallel`, `two_intersecting` | ~100 ms – 2 s | tan-half-angle reduction + 200-sample search + Newton polish |
| 1 — 7R joint-lock wrapper | `jointlock.seven_r` | ~5-30 ms | lock one joint, dispatch inner 6R, sweep 16 lock samples |
| 2 — Raghavan–Roth + Manocha–Canny | `ikgeo.general_6r` | ~0.6-5 ms | numeric RR resultant with AE-3 leftvar selection; **production tier-2** |
| 2 — Husty-Pfurner universal fallback | `husty_pfurner.general_6r` | ~25-200 ms | Study-quaternion algebra; perturbation path (#176) handles symmetric-DH singularities; backstops RR on ill-conditioned arms |

## Per-arm artifact builder (`ssik build`)

The user-facing flow: hand us a URDF, get back a self-contained Python module.

```bash
$ ssik build path/to/ur5.urdf --base base_link --ee ee_link
[ssik] Loading path/to/ur5.urdf
[ssik]   6 joints, 7 links — POE-normalized OK
[ssik] Classifying topology
[ssik]   → Best solver: ikgeo.three_parallel (tier 0)
[ssik]   → Expected median IK time: ~1.6 ms
[ssik]   → FLOP budget: ~2,519 FLOPs / solve
[ssik]   → Reasoning:
[ssik]       Three consecutive parallel axes at joints (1, 2, 3) — the UR-class structure.
[ssik]       Closed-form via SP6 (joints 0+4) + SP1 + SP3.
[ssik] No build-time precompute needed (tier-0 closed-form)
[ssik] Emitting ./ur5_ik.py
[ssik]   Wrote 5,004 bytes
[ssik] Validating (100 random poses)
[ssik]   ✓ 100 poses, median 0.78 ms, max FK error 6e-09, 0 failures
[ssik] ✓ Done. Try:
[ssik]     >>> import ur5_ik
[ssik]     >>> sols, is_ls = ur5_ik.solve(T_target)
```

For tier-0 arms (Pieper-class: UR, Puma, generic spherical-wrist), the artifact body is **per-arm specialised IKFast-style trig**: explicit `sin`, `cos`, `atan2` of target-pose entries with the arm's geometry constants substituted. For tier-2 (non-Pieper, e.g. JACO 2) the artifact is currently a thin wrapper around the runtime solver; specialising tier-2 with build-time symbolic precompute baking is in progress (#112).

A snippet of the emitted Puma 560 artifact:

```python
# SP4 for q1 (shoulder pan).
q1_x0 = math.atan2(1.0*p_x, -1.0*p_y)
q1_x1 = 0.15005 - 6.12e-17*p_z      # 0.15005 = Puma wrist y-offset
q1_x2 = 1.0*p_x**2 + 1.0*p_y**2
theta_q1_plus = q1_x0 + math.acos(q1_x1/math.sqrt(q1_x2))
theta_q1_minus = q1_x0 - math.acos(q1_x1/math.sqrt(q1_x2))
```

The same for q2/q3/q4/q5/q6 with Puma's link-length constants (0.4318, etc.) substituted throughout.

The emitted artifact wraps the dispatched solver around baked KinBody constants and exposes a rich API:

```python
import ur5_ik
from ssik import TolerancePolicy

# Default: fastest path, no Newton polish.
sols, is_ls = ur5_ik.solve(T_target)

# Tighter FK-closure threshold for high-precision applications.
strict = TolerancePolicy(subproblem_numerical=1e-9)
sols, is_ls = ur5_ik.solve(T_target, policy=strict)

# Newton polish for near-singular poses (off by default).
sols, is_ls = ur5_ik.solve(T_target, allow_refinement=True, refinement_max_iters=8)

# is_ls=True signals the algebraic path didn't close; sols is best-LS or empty.
# Callers wanting only "exact" solutions check is_ls and discard.
if is_ls:
    raise NoExactIK
for sol in sols:
    print(sol.q, sol.fk_residual, sol.refinement_used)
```

The dispatcher picks the best solver from the catalog above based on topology predicates (`three_consecutive_parallel`, `three_consecutive_intersecting`, etc.). Tier-1 univariate-search solvers are not auto-selected: tier-2 Raghavan–Roth (`general_6r`) handles the same chains at ~5 ms vs tier-1's 100 ms–2 s. Tier-1 modules remain importable for users who want them explicitly.

For inspection without emitting an artifact:

```bash
$ ssik classify path/to/your.urdf --base base_link --ee tcp_link
```

Tier-2 (non-Pieper) arms today still trigger the symbolic preprocessing on first `solve()` call (~150–300 s). Phase 2 of [#110](https://github.com/siddhss5/ikfastpy/issues/110) bakes the preprocessing output into the artifact at build time so first-call latency is gone.

## Repository layout

This is the **internal development codebase**. Public users never see this; they receive a per-arm wheel built from this source.

- `src/ssik/` — solver implementations.
  - `core/` — `Solution` dataclass, tolerance policies.
  - `kinematics/` — POE → DH bridge, predicates.
  - `subproblems/` — SP1–SP6 closed-form primitives + `_rotation` helpers.
  - `solvers/ikgeo/` — tier-0/1/2 solver modules.
  - `solvers/jointlock/` — 7R wrapper.
  - `refinement/` — universal opt-in Newton polish layer.
- `tests/` — unit + hypothesis fuzz + cross-solver agreement + slow round-trips.
  - `tests/fixtures/` — UR5, Puma 560, JACO 2 (real MJCF), synthetic arms.
- `scripts/` — bench, profile, diagnostic harnesses.
- `docs/` — internal documentation.
  - `docs/tutorial/` — ten chapters covering the IK problem, the Pieper class, the EAIK gap, the Raghavan–Roth pipeline, conditioning fixes, refinement architecture, KinBody bridge, bulletproof validation, practical guide, roadmap. **Internal only**; the public marketing site (per [#95](https://github.com/siddhss5/ikfastpy/issues/95)) is a stripped subset hosted in a separate public repo.

## Development quick-start

```bash
# Install dev dependencies
uv sync

# Fast tests (excludes slow symbolic-preprocessing tests)
uv run pytest

# Slow tests (sympy preprocessing for tier-2 RR; ~5 min)
uv run pytest -m slow

# Lint, format, typecheck
uv run ruff check
uv run ruff format --check
uv run mypy src tests

# Bench any solver (machine-invariant FLOP budget + wall-clock)
uv run python scripts/bench_three_parallel.py     # UR5
uv run python scripts/bench_real_jaco2.py         # JACO 2, RR pipeline
uv run python scripts/bench_seven_r.py            # synthetic 7R

# Build internal docs
uv run mkdocs serve
```

## Distribution model

Customers submit a URDF / MJCF via [#95](https://github.com/siddhss5/ikfastpy/issues/95)'s intake mechanism. We run `ssik build` (above) which produces a per-arm `.py` artifact today; the [#110](https://github.com/siddhss5/ikfastpy/issues/110) phasing extends this to a Cython `.so` artifact with the same `solve(T, *, policy=..., allow_refinement=..., refinement_max_iters=...)` API. The wheel ships the artifact and nothing else — customer source code never imports ssik internals.

The Cython port (gated on the Python pipeline being stable) closes the ~10000× gap the FLOP budget says is on the table: every solver is currently dispatch-bound at ~1 MFLOP/s achieved, and a native port should hit ~µs IK on Pieper-class arms — the original IKFast promise, this time without IKFast's fragility.

## License

Proprietary. See [`LICENSE`](LICENSE) for full terms; in summary: all rights reserved, no public reproduction or distribution without prior written permission. The library incorporates clean-room reimplementations of algorithms from BSD-3-licensed [IK-Geo](https://github.com/rpiRobotics/ik-geo) (Elias–Wen 2022/2025) and from the academic publications of Raghavan–Roth (1990) and Manocha–Canny (1994); the BSD-3 attribution is preserved in `LICENSE` for the algorithmic lineage.

## Tracking

- Per-arm artifact builder + Cython port: [#110](https://github.com/siddhss5/ikfastpy/issues/110).
- Strategic distribution model: [#95](https://github.com/siddhss5/ikfastpy/issues/95).
- Speed work across all solver pathways: [#93](https://github.com/siddhss5/ikfastpy/issues/93).
- Tier-2 RR speed (already ~2× since baseline): [#86](https://github.com/siddhss5/ikfastpy/issues/86).
- Cold-cache symbolic preprocessing speed: [#97](https://github.com/siddhss5/ikfastpy/issues/97).
- Tutorial / internal docs: [#87](https://github.com/siddhss5/ikfastpy/issues/87).
- Pre-existing hypothesis flake on `test_ikgeo_spherical`: [#101](https://github.com/siddhss5/ikfastpy/issues/101).
- Known coverage gap on synthetic MC Table I: [#82](https://github.com/siddhss5/ikfastpy/issues/82).
