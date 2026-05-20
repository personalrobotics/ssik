# Arm coverage

Per-arm tested fixture tables, dispatched solver, and measured speed. Numbers come from [`examples/04_compare_vs_eaik.py`](https://github.com/personalrobotics/ssik/blob/main/examples/04_compare_vs_eaik.py) — 100 random reachable poses per arm, Apple M3 single-thread, mean ± 95% CI via 1000-resample bootstrap. The README's EAIK comparison table reports the same measurements; this doc breaks them down by kinematic class and points at where each arm's prebuilt lives.

**Status legend**: ✅ shipped in [`ssik.prebuilt`](https://github.com/personalrobotics/ssik/tree/main/src/ssik/prebuilt) and exercised by the test suite — 🔗 external URDF, fixture import pending — 📐 synthetic-only (no canonical commercial arm with this exact topology).

## 6R industrial arms (Pieper-class)

Closed-form IK via subproblem decomposition (SP1–SP6).

| Arm | Solver | Speed | Status |
|-----|--------|:-----:|:-----:|
| **UR5** (also UR3 / UR10 / UR16) | `ikgeo.three_parallel` | 532 ± 10 µs / 2-8 sols | ✅ `ssik.prebuilt.ur5_ik` (UR5 in tests/fixtures); others 🔗 |
| **Unitree Z1** | `ikgeo.three_parallel` | 487 ± 7 µs / 4-8 sols | ✅ `ssik.prebuilt.z1_ik` |
| **Puma 560** | `ikgeo.spherical_two_parallel` | 220 ± 3 µs / 8 sols | ✅ `ssik.prebuilt.puma560_ik` |
| ABB IRB120 / IRB4600 | `ikgeo.spherical_two_parallel` | expected ~1 ms | 🔗 [ros-industrial/abb](https://github.com/ros-industrial/abb) |
| Fanuc LR Mate / CR | `ikgeo.spherical_two_parallel` | expected ~1 ms | 🔗 (vendor URDF) |
| KUKA KR | `ikgeo.spherical_two_parallel` | expected ~1 ms | 🔗 [ros-industrial/kuka_experimental](https://github.com/ros-industrial/kuka_experimental) |
| uFactory lite6 | `ikgeo.spherical_two_parallel` | expected ~1 ms | 🔗 [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros) |

EAIK is ~100× faster on Pieper-class 6R (5 µs vs ssik's ~250-550 µs); that's its native sweet spot and we don't try to compete. See README's comparison table.

## 6R non-Pieper (the EAIK gap)

The arms ssik exists for: deliberate non-orthogonal twists that violate Pieper's condition. Subproblem-decomposition libraries refuse these; ssik solves them analytically via Raghavan–Roth + AE-3 leftvar selection. AE-3 picks the spectral parameter that puts pathological joints out of the linearity variable — on JACO 2 this drops `cond(m_quad)` from 3.75 × 10^16 → 127 (14 orders of magnitude).

| Arm | Solver | Speed | Status |
|-----|--------|:-----:|:-----:|
| **Kinova JACO 2 (j2n6s200, 55° DH)** | `ikgeo.general_6r` (RR + AE-3) | 976 ± 39 µs / 2-12 sols | ✅ `ssik.prebuilt.jaco2_ik` |
| **UFactory xArm6** (joint-6 y-offset breaks spherical wrist) | `ikgeo.general_6r` (RR + AE-3) | 1.06 ± 0.02 ms / 8-12 sols | ✅ `ssik.prebuilt.xarm6_ik` |
| Agilex Piper | `ikgeo.general_6r` | expected ~1-5 ms | 🔗 [mujoco_menagerie/agilex_piper](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper) |
| Custom non-Pieper 6R | `ikgeo.general_6r` | expected ~1-5 ms | use `ssik build` to compile a per-arm artifact |

## 7R redundant — pure SRS (`seven_r.srs`)

Closed-form Singh-Kreutz 1989 algorithm. Predicate-driven dispatch via `is_srs_7r` (no per-arm hardcoding); auto-applies to any 7R chain whose shoulder axes (0,1,2) and wrist axes (4,5,6) each meet at a common point. Default 16 swivel samples × 8 branches = 128 IK candidates per call (URDF joint limits filter most down to ~80-96).

| Arm | Speed | Status |
|-----|:-----:|:-----:|
| **KUKA iiwa LBR 14** | 4.54 ± 0.03 ms / 128 sols / FK 4e-13 | ✅ `ssik.prebuilt.iiwa14_ik` |
| KUKA iiwa LBR 7 (R820 / R14) | expected ~5 ms | 🔗 [mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14) |

## 7R redundant — approximately-SRS with LM polish (`seven_r.srs_polished`)

For arms whose URDF axes only **nearly** meet at common shoulder/wrist points (Kinova Gen3: 12 mm shoulder + 0.4 mm wrist drift), the strict SRS predicate refuses but the relaxed-policy Singh-Kreutz solver still produces good warm-start candidates. Each is LM-polished against the original URDF FK to reach machine-precision closure. Refused (drift exceeds Newton's basin, ~3–5 cm): falls back to `jointlock + HP`.

| Arm | Drift (shoulder / wrist) | Speed | Status |
|-----|---|:-----:|:-----:|
| **Kinova Gen3 (7-DOF)** | 12 mm / 0.4 mm | 41.46 ± 1.25 ms / 10-95 sols / FK 1e-12 | ✅ `ssik.prebuilt.gen3_ik` |

## 7R redundant — non-SRS (`jointlock.seven_r`)

For 7R arms whose topology doesn't match strict or approximate SRS, `jointlock.seven_r` is the universal fallback: locks one joint (auto-selected by topology rank of the resulting 6R sub-chain) and dispatches the 6R IK to the best-matching tier-0/1 ikgeo or HP solver. 16-sample lock sweep × inner 6R solver per call.

For non-Pieper inner sub-chains (Rizon 4, Kassow KR810), `ssik build` bakes the per-(DH, linearity) Raghavan-Roth derivation into the artifact (cached-RR fast path, #210/#220). Module import primes the cache once (~5 seconds); subsequent calls amortise.

| Arm | Drift (shoulder / wrist) | Inner | Speed (full sweep) | Status |
|-----|---|---|:-----:|:-----:|
| **Franka Emika Panda** | non-SRS by design | tier-0 inner (`reversed:spherical_two_parallel`) | 29.27 ± 2.81 ms / 8-124 sols / FK 1e-6 | ✅ `ssik.prebuilt.franka_panda_ik` |
| **uFactory xArm7** | non-SRS by design | tier-0 inner (`reversed:spherical`) | 37.10 ± 0.49 ms / 56-64 sols / FK 4e-11 | ✅ `ssik.prebuilt.xarm7_ik` |
| **Flexiv Rizon 4** | 65 mm / 151 mm | cached-RR (HP otherwise) | 30.58 ± 8.58 ms / 10-60 sols / FK 4e-9 | ✅ `ssik.prebuilt.rizon4_ik` |
| **Kassow KR810** | 86 mm / 111 mm | cached-RR (HP otherwise) | 27.52 ± 10.71 ms / 10-38 sols / FK 7e-8 | ✅ `ssik.prebuilt.kassow_kr810_ik` |
| Sawyer / Baxter (Rethink) | likely non-SRS | TBD | expected ~30-50 ms | 🔗 |

> **Mean vs median:** both 30 ms and ~17 ms are honest measurements of the same prebuilt — the canonical bench reports mean ± 95% CI, an earlier per-pose measurement reported median. Verified 2026-05-13 on Rizon 4 with the canonical pose distribution: mean 28.7 ms / **median 19 ms** / p95 62 ms / min 16.4 ms. The cached-RR fast path is firing on most poses; mean is dragged up by occasional near-singular configurations where the lock-sweep can't short-circuit.

## Worst-case FK floor under adversarial fuzz

The README's EAIK comparison table reports **averaged** max FK across 100 canonical reachable poses. Under 500-pose Hypothesis fuzz (`tests/test_prebuilt_uniform_fuzz.py`), worst-case residuals on jointlock 7R arms are materially worse — but only at the **default tolerance policy**. Investigation (#271) confirmed the floor is set by `subproblem_numerical = 1e-5`, not a solver bug; opt-in to tight policy + LM polish recovers machine precision.

| Arm | Solver path | Default-policy worst | Tight + `allow_refinement` worst |
|-----|---|:---:|:---:|
| UR5 | `ikgeo.three_parallel` | ~2e-8 | (already machine precision) |
| Puma 560 | `ikgeo.spherical_two_parallel` | ~1e-13 | — |
| JACO 2 | `ikgeo.general_6r` (RR + AE-3) | ~1e-5 | ~1e-10 (LM polish) |
| iiwa14 | `seven_r.srs` | ~3e-12 | — |
| Gen3 | `seven_r.srs_polished` | ~1e-10 | — |
| Franka Panda | `jointlock + reversed:spherical_two_parallel` | ~5e-6 | **~3e-10** |
| Rizon 4 | `jointlock + cached-RR` | ~9e-6 | **~3e-10** |
| Kassow KR810 | `jointlock + cached-RR` | ~6e-6 | **~3e-10** |

### Getting machine precision from a jointlock 7R prebuilt

```python
from ssik import TolerancePolicy
from ssik.prebuilt import franka_panda_ik

tight = TolerancePolicy(
    axis_parallel=1e-8,
    axis_intersect=1e-8,
    subproblem_feasibility=1e-9,
    subproblem_numerical=1e-9,         # ← 4 orders tighter than default
    subproblem_degeneracy=1e-12,
    subproblem_dedup=1e-3,
)
sols = franka_panda_ik.solve(T_target, policy=tight, allow_refinement=True)
# every returned IK FK-closes ~1e-10 (machine-precision)
```

The default policy is deliberately loose (1e-5) for throughput — most users don't need 1e-10. Adversarial workloads (RL, sample-based motion planning, learning from demonstration) should opt in.

## Trajectory-tracking speed (`max_solutions=1`)

For real-time control where you only need one IK per waypoint, `max_solutions=1` + `q_seed` short-circuits the lock-sweep on the first in-limits branch closest to seed. Speedup is roughly proportional to lock-sample count (16 by default on 7R jointlock arms): ~5–10× on 7R, sub-ms on 6R and SRS arms. See the README quickstart for the canonical pattern; per-arm trajectory-tracking benches will land alongside the [#236](https://github.com/personalrobotics/ssik/issues/236) MINK / TracIK comparison.
