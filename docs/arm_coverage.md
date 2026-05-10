# Arm coverage

Per-arm tested fixture tables, dispatched solver, and measured speed. All speeds are typical median IK-call latency on Apple M3 single-thread, pure Python + numpy.

**Status legend**: ✅ in-repo URDF/MJCF fixture exercised by the test suite — 🔗 external URDF, fixture import pending — 📐 synthetic-only (no canonical commercial arm with this exact topology).

## 6R industrial arms (Pieper-class)

Closed-form IK in 1-2 ms via subproblem decomposition.

| Arm | Solver | Speed | Status |
|-----|--------|:-----:|:-----:|
| Universal Robots UR3 / UR5 / UR10 / UR16 | `ikgeo.three_parallel` | ~1-2 ms | ✅ UR5 in [`tests/fixtures/`](../tests/fixtures/), others 🔗 |
| Puma 560 | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | ~1-2 ms | ✅ |
| ABB IRB120 / IRB4600 | `ikgeo.spherical_two_parallel` | ~1-2 ms | 🔗 [ros-industrial/abb](https://github.com/ros-industrial/abb) |
| Fanuc LR Mate / CR | `ikgeo.spherical_two_parallel` | ~1-2 ms | 🔗 (vendor URDF) |
| KUKA KR | `ikgeo.spherical_two_parallel` | ~1-2 ms | 🔗 [ros-industrial/kuka_experimental](https://github.com/ros-industrial/kuka_experimental) |
| uFactory lite6 / xArm6 | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | ~1-2 ms | 🔗 [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros) |

## 6R non-Pieper (the EAIK gap)

The arms ssik exists for: deliberate non-orthogonal twists that violate Pieper's condition. Subproblem-decomposition libraries refuse these; ssik solves them analytically via Raghavan–Roth + AE-3 leftvar selection.

| Arm | Solver | Speed | Status |
|-----|--------|:-----:|:-----:|
| **Kinova JACO 2 (j2n6s200, 55° DH)** | `ikgeo.general_6r` (RR + AE-3) | **~5 ms median** | ✅ real MJCF in [`tests/fixtures/`](../tests/fixtures/) |
| Agilex Piper | `ikgeo.general_6r` | expected ~5 ms | 🔗 [mujoco_menagerie/agilex_piper](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper) |
| Custom non-Pieper 6R | `ikgeo.general_6r` | expected ~5 ms | URDF intake via [#95](https://github.com/siddhss5/ikfastpy/issues/95) |

## 7R redundant — pure SRS (`seven_r.srs`)

Closed-form Singh-Kreutz 1989 algorithm. Predicate-driven dispatch via `is_srs_7r` (no per-arm hardcoding); auto-applies to any 7R chain whose shoulder axes (0,1,2) and wrist axes (4,5,6) each meet at a common point. Default 16 swivel samples × 8 branches = 128 IK candidates per call.

| Arm | Full-sweep speed | Status |
|-----|:---:|:---:|
| **KUKA iiwa LBR 14** | **~4.3 ms (128 IKs, FK ≤ 1e-13)** | ✅ in [`tests/fixtures/`](../tests/fixtures/) |
| KUKA iiwa LBR 7 (R820 / R14) | expected ~4.3 ms | 🔗 [mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14) |

## 7R redundant — approximately-SRS with LM polish (`seven_r.srs_polished`)

For arms whose URDF axes only **nearly** meet at common shoulder/wrist points (Kinova Gen3: 12 mm shoulder + 0.4 mm wrist drift), the strict SRS predicate refuses but the relaxed-predicate Singh-Kreutz solver still produces good warm-start candidates. Each is LM-polished against the original URDF FK to reach machine-precision closure (FK ≤ 1e-10). 16-30× faster than `jointlock + HP` on small-drift arms, with no IK error against the real URDF (unlike EAIK's simplified-DH path).

Refused (drift exceeds Newton's basin, ~3-5 cm): falls back to `jointlock + HP`.

## 7R redundant — non-SRS (`jointlock.seven_r`)

For 7R arms whose topology doesn't match strict or approximate SRS, `jointlock.seven_r` is the universal fallback: locks one joint (auto-selected by topology rank of the resulting 6R sub-chain) and dispatches the 6R IK to the best-matching tier-0/1 ikgeo solver. 16-sample lock sweep × inner 6R solver per call.

**Cached-RR fast path** ([#210](https://github.com/siddhss5/ikfastpy/issues/210)): when an arm hits the universal fallback for non-Pieper sub-chains, `ssik build` bakes the per-(DH, linearity) Raghavan-Roth derivation into the artifact. Module import primes the cache once (~5 seconds via #210 Phase 2 / #220); every subsequent `solve(T)` call uses cached RR (~1 ms warm) instead of HP / two_parallel (~13-260 ms). **12-25× speedup post-import** on Rizon 4, Kassow, and other previously-slow non-Pieper 7R arms. The URDF-loaded path (no artifact) keeps the original solver — no cold-cache cost in tests.

Covers Franka Panda (anthropomorphic 7R), uFactory xArm7 (mixed structure), and the literature-SRS-but-URDF-far-from-SRS arms whose drift exceeds the polished-SRS basin (Flexiv Rizon 4: 151 mm wrist drift; Kassow KR810: 111 mm wrist drift).

| Arm | Drift (shoulder / wrist) | URDF path | **Built artifact (cached-RR)** | Status |
|-----|---|:---:|:---:|:---:|
| Franka Emika Panda / FR3 | non-SRS by design | ~42 ms (48 IKs) | same (already tier-0) | ✅ in [`tests/fixtures/`](../tests/fixtures/) |
| uFactory xArm7 | non-SRS by design | ~45 ms (56 IKs) | same (already tier-0) | ✅ in [`tests/fixtures/`](../tests/fixtures/) |
| **Kinova Gen3 (7-DOF)** | 12 mm / 0.4 mm | **~56 ms (`seven_r.srs_polished`)** | n/a (top-level polished-SRS) | ✅ in [`tests/fixtures/`](../tests/fixtures/) |
| **Flexiv Rizon 4** | 65 mm / 151 mm | ~244 ms (jointlock+HP) | **~17 ms (12.8×, 45 IKs)** | ✅ in [`tests/fixtures/`](../tests/fixtures/) |
| **Kassow KR810** | 86 mm / 111 mm | ~444 ms (jointlock+HP) | **~18 ms (16.4×, 30 IKs)** | ✅ in [`tests/fixtures/`](../tests/fixtures/) |
| Sawyer / Baxter (Rethink) | likely non-SRS | expected ~50 ms | expected ~17-20 ms | 🔗 |

## Trajectory-tracking speed (`max_solutions=1`)

For real-time control where you only need one IK per waypoint, `max_solutions=1` short-circuits the redundancy enumeration. Typical numbers on the same hardware:

| Arm | Full sweep | `max_solutions=1` | Speedup |
|-----|:---:|:---:|:---:|
| iiwa14 | 4.3 ms (128 IKs) | ~0.5 ms (1 IK) | ~9× |
| Gen3 | 56 ms (~40 IKs) | ~5 ms (1 IK) | ~11× |
| Rizon 4 | 17 ms (~45 IKs) | ~1.5 ms (1 IK) | ~11× |
| Franka | 42 ms (~64 IKs) | **2.4 ms (1 IK)** | **~17×** |

Combined with `q_seed` (warm-start from the previous waypoint), the dispatcher reorders sample sweeps so the closest-to-seed configuration fires first.
