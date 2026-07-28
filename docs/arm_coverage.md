# Arm coverage

Per-arm tested fixture tables, dispatched solver, and measured speed. Numbers come from [`examples/04_compare_vs_eaik.py`](https://github.com/personalrobotics/ssik/blob/main/examples/04_compare_vs_eaik.py): 100 random reachable poses per arm, Apple M3 single-thread, mean ± 95% CI via 1000-resample bootstrap. The README's EAIK comparison table reports the same measurements; this doc breaks them down by kinematic class and points at where each arm's prebuilt lives.

**Status legend**: ✅ shipped in [`ssik.prebuilt`](https://github.com/personalrobotics/ssik/tree/main/src/ssik/prebuilt) and exercised by the test suite. 🔗 external URDF, fixture import pending. 📐 synthetic-only (no canonical commercial arm with this exact topology).

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

The arms ssik exists for: deliberate non-orthogonal twists that violate Pieper's condition. Subproblem-decomposition libraries refuse these; ssik solves them analytically via Raghavan–Roth + AE-3 leftvar selection. AE-3 picks the spectral parameter that puts pathological joints out of the linearity variable. On JACO 2 this drops `cond(m_quad)` from 3.75 × 10^16 → 127 (14 orders of magnitude).

| Arm | Solver | Speed | Status |
|-----|--------|:-----:|:-----:|
| **Kinova JACO 2 (j2n6s200, 55° DH)** | `ikgeo.general_6r` (RR + AE-3) | 976 ± 39 µs / 2-12 sols | ✅ `ssik.prebuilt.jaco2_ik` |
| **UFactory xArm6** (joint-6 y-offset breaks spherical wrist) | `ikgeo.general_6r` (RR + AE-3) | 1.06 ± 0.02 ms / 8-12 sols | ✅ `ssik.prebuilt.xarm6_ik` |
| **AgileX PiPER** (joints 4 & 6 share a tilted axis) | `ikgeo.general_6r` (RR + AE-3) | 1.17 ± 0.03 ms / 1-8 sols | ✅ `ssik.prebuilt.piper_ik` |
| Custom non-Pieper 6R | `ikgeo.general_6r` | expected ~1-5 ms | use `ssik build` to compile a per-arm artifact |

## 7R redundant: pure SRS (`seven_r.srs`)

Closed-form Singh-Kreutz 1989 algorithm. Predicate-driven dispatch via `is_srs_7r` (no per-arm hardcoding); auto-applies to any 7R chain whose shoulder axes (0,1,2) and wrist axes (4,5,6) each meet at a common point. Default 16 swivel samples × 8 branches = 128 IK candidates per call (URDF joint limits filter most down to ~80-96).

| Arm | Speed | Status |
|-----|:-----:|:-----:|
| **KUKA iiwa LBR 14** | 4.54 ± 0.03 ms / 128 sols / FK 4e-13 | ✅ `ssik.prebuilt.iiwa14_ik` |
| KUKA iiwa LBR 7 (R820 / R14) | expected ~5 ms | 🔗 [mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14) |

## 7R redundant: approximately-SRS with LM polish (`seven_r.srs_polished`)

For arms whose URDF axes only **nearly** meet at common shoulder/wrist points (Kinova Gen3: 12 mm shoulder + 0.4 mm wrist drift), the strict SRS predicate refuses but the relaxed-policy Singh-Kreutz solver still produces good warm-start candidates. Each is LM-polished against the original URDF FK to reach machine-precision closure. Refused (drift exceeds Newton's basin, ~3–5 cm): falls back to `jointlock + HP`.

| Arm | Drift (shoulder / wrist) | Speed | Status |
|-----|---|:-----:|:-----:|
| **Kinova Gen3 (7-DOF)** | 12 mm / 0.4 mm | 41.46 ± 1.25 ms / 10-95 sols / FK 1e-12 | ✅ `ssik.prebuilt.gen3_ik` |

## 7R redundant: non-SRS (`jointlock.seven_r`)

For 7R arms whose topology doesn't match strict or approximate SRS, `jointlock.seven_r` is the universal fallback: locks one joint (auto-selected by topology rank of the resulting 6R sub-chain) and dispatches the 6R IK to the best-matching tier-0/1 ikgeo or HP solver. 16-sample lock sweep × inner 6R solver per call.

For non-Pieper inner sub-chains (Rizon 4, Kassow KR810), `ssik build` bakes the per-(DH, linearity) Raghavan-Roth derivation into the artifact (cached-RR fast path, #210/#220). Module import primes the cache once (~5 seconds); subsequent calls amortise.

The lock-sweep uses cached-RR as the per-sample primary and defers the symbolic Husty-Pfurner fallback to a whole-sweep last resort (fires only if cached-RR finds nothing on every sample). Most lock samples are configurationally unreachable, so this avoids paying the per-call sympy cost just to confirm a 0, cutting full-sweep time 1.6–2.8× on cached-RR arms with no change to the solution set (#326).

| Arm | Drift (shoulder / wrist) | Inner | Speed (full sweep) | Status |
|-----|---|---|:-----:|:-----:|
| **Franka Emika Panda** | non-SRS by design | tier-0 inner (`reversed:spherical_two_parallel`) | 29.27 ± 2.81 ms / 8-124 sols / FK 1e-6 | ✅ `ssik.prebuilt.franka_panda_ik` |
| **uFactory xArm7** | non-SRS by design | tier-0 inner (`reversed:spherical`) | 37.10 ± 0.49 ms / 56-64 sols / FK 4e-11 | ✅ `ssik.prebuilt.xarm7_ik` |
| **Flexiv Rizon 4** | 65 mm / 151 mm | cached-RR (HP otherwise) | 16.57 ± 0.30 ms / 10-60 sols / FK 4e-9 | ✅ `ssik.prebuilt.rizon4_ik` |
| **Flexiv Rizon 10** (~1.4 m reach) | 65 mm / 151 mm | cached-RR (HP otherwise) | 16.26 ± 0.34 ms / 10-64 sols / FK 6e-8 | ✅ `ssik.prebuilt.rizon10_ik` |
| **Kassow KR810** | 86 mm / 111 mm | cached-RR (HP otherwise) | 17.62 ± 0.29 ms / 10-38 sols / FK 7e-8 | ✅ `ssik.prebuilt.kassow_kr810_ik` |
| Sawyer / Baxter (Rethink) | likely non-SRS | TBD | expected ~30-50 ms | 🔗 |

> **Mean vs median:** both 30 ms and ~17 ms are honest measurements of the same prebuilt: the canonical bench reports mean ± 95% CI, an earlier per-pose measurement reported median. Verified 2026-05-13 on Rizon 4 with the canonical pose distribution: mean 28.7 ms / **median 19 ms** / p95 62 ms / min 16.4 ms. The cached-RR fast path is firing on most poses; mean is dragged up by occasional near-singular configurations where the lock-sweep can't short-circuit.

## Worst-case FK floor under adversarial fuzz

The README's EAIK comparison table reports **averaged** max FK across 100 canonical reachable poses. Under 500-pose Hypothesis fuzz (`tests/test_prebuilt_uniform_fuzz.py`), worst-case residuals on jointlock 7R arms are materially worse, but only at the **default tolerance policy**. Investigation (#271) confirmed the floor is set by `subproblem_numerical = 1e-5`, not a solver bug; opt-in to tight policy + LM polish recovers machine precision.

| Arm | Solver path | Default-policy worst | Tight + `allow_refinement` worst |
|-----|---|:---:|:---:|
| UR5 | `ikgeo.three_parallel` | ~2e-8 | (already machine precision) |
| Puma 560 | `ikgeo.spherical_two_parallel` | ~1e-13 | n/a |
| JACO 2 | `ikgeo.general_6r` (RR + AE-3) | ~1e-5 | ~1e-10 (LM polish) |
| iiwa14 | `seven_r.srs` | ~3e-12 | n/a |
| Gen3 | `seven_r.srs_polished` | ~1e-10 | n/a |
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

The default policy is deliberately loose (1e-5) for throughput: most users don't need 1e-10. Adversarial workloads (RL, sample-based motion planning, learning from demonstration) should opt in.

## Trajectory-tracking speed (`max_solutions=1`)

For real-time control where you only need one IK per waypoint, `max_solutions=1` + `q_seed` short-circuits the lock-sweep on the first in-limits branch closest to seed. Speedup is roughly proportional to lock-sample count (16 by default on 7R jointlock arms): ~5–10× on 7R, sub-ms on 6R and SRS arms. See the README quickstart for the canonical pattern; per-arm trajectory-tracking benches will land alongside the [#236](https://github.com/personalrobotics/ssik/issues/236) MINK / TracIK comparison.

## Fixture provenance

Each prebuilt's kinematic chain is sourced from a specific upstream URDF (or, for legacy DH arms, the published parameter set). The IK promise (that the q-vector lands a real arm at the target) only holds when ssik's chain matches the manufacturer's. We lock this in with [`tests/test_prebuilt_fixture_parity.py`](../tests/test_prebuilt_fixture_parity.py): for every arm whose source is reachable via `robot_descriptions`, it asserts `module.fk(q) == upstream.fk(q)` to machine precision.

<!-- AUTOGEN:fixture_source_table -->
<details>
<summary><b>Universal Robots</b>: <code>ssik.prebuilt.universal_robots</code> (11 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `ur5_ik` | robot_descriptions / ur5_description |
| `ur3e_ik` | robot_descriptions / ur3e_description (ur.urdf.xacro, ur_type:=ur3e) |
| `ur5e_ik` | robot_descriptions / ur5e_description (ur.urdf.xacro, ur_type:=ur5e) |
| `ur10e_ik` | robot_descriptions / ur10e_description (ur.urdf.xacro, ur_type:=ur10e) |
| `ur16e_ik` | robot_descriptions / ur16e_description (ur.urdf.xacro, ur_type:=ur16e) |
| `ur20_ik` | robot_descriptions / ur20_description (ur.urdf.xacro, ur_type:=ur20) |
| `ur30_ik` | robot_descriptions / ur30_description (ur.urdf.xacro, ur_type:=ur30) |
| `ur7e_ik` | robot_descriptions / ur7e_description (ur.urdf.xacro, ur_type:=ur7e) |
| `ur12e_ik` | robot_descriptions / ur12e_description (ur.urdf.xacro, ur_type:=ur12e) |
| `ur15_ik` | robot_descriptions / ur15_description (ur.urdf.xacro, ur_type:=ur15) |
| `ur18_ik` | robot_descriptions / ur18_description (ur.urdf.xacro, ur_type:=ur18) |

</details>

<details>
<summary><b>Unimation</b>: <code>ssik.prebuilt.unimation</code> (1 arm)</summary>

| Module | Fixture provenance |
|---|---|
| `puma560_ik` | classical DH (Lee, Asada & Slotine 1986) |

</details>

<details>
<summary><b>Kinova</b>: <code>ssik.prebuilt.kinova</code> (5 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `jaco2_ik` | Kinova j2n6s200 DH (kinova-ros / kinova_description) |
| `gen3_ik` | Kinovarobotics / ros_kortex (kortex_description / gen3.xacro) |
| `gen3_lite_ik` | robot_descriptions / gen3_lite_description |
| `j2s6s300_ik` | robot_descriptions / j2s6s300_description |
| `j2s7s300_ik` | kinova-ros (kinova_description / j2s7s300_standalone.xacro) |

</details>

<details>
<summary><b>KUKA</b>: <code>ssik.prebuilt.kuka</code> (2 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `iiwa14_ik` | robot_descriptions / iiwa14_description |
| `iiwa7_ik` | robot_descriptions / iiwa7_description |

</details>

<details>
<summary><b>Franka</b>: <code>ssik.prebuilt.franka</code> (2 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `panda_ik` | robot_descriptions / panda_description |
| `fr3_ik` | robot_descriptions / fr3_description |

</details>

<details>
<summary><b>UFactory</b>: <code>ssik.prebuilt.ufactory</code> (2 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `xarm7_ik` | robot_descriptions / xarm7_description |
| `xarm6_ik` | robot_descriptions / xarm6_description |

</details>

<details>
<summary><b>Unitree</b>: <code>ssik.prebuilt.unitree</code> (1 arm)</summary>

| Module | Fixture provenance |
|---|---|
| `z1_ik` | robot_descriptions / z1_description |

</details>

<details>
<summary><b>AgileX</b>: <code>ssik.prebuilt.agilex</code> (1 arm)</summary>

| Module | Fixture provenance |
|---|---|
| `piper_ik` | robot_descriptions / piper_description |

</details>

<details>
<summary><b>Flexiv</b>: <code>ssik.prebuilt.flexiv</code> (2 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `rizon4_ik` | robot_descriptions / rizon4_description |
| `rizon10_ik` | Flexiv Rizon 10 URDF (vendor-supplied) |

</details>

<details>
<summary><b>Kassow</b>: <code>ssik.prebuilt.kassow</code> (1 arm)</summary>

| Module | Fixture provenance |
|---|---|
| `kr810_ik` | Kassow KR810 URDF (vendor-supplied) |

</details>

<details>
<summary><b>FANUC</b>: <code>ssik.prebuilt.fanuc</code> (8 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `crx3ia_ik` | FANUC-CORPORATION/fanuc_description (Apache-2.0) |
| `crx5ia_ik` | FANUC-CORPORATION/fanuc_description (Apache-2.0) |
| `crx10ia_ik` | FANUC-CORPORATION/fanuc_description (Apache-2.0) |
| `crx10ialp_ik` | FANUC-CORPORATION/fanuc_description (Apache-2.0) |
| `crx20ial_ik` | FANUC-CORPORATION/fanuc_description (Apache-2.0) |
| `crx30ia_ik` | FANUC-CORPORATION/fanuc_description (Apache-2.0) |
| `crx10ial_ik` | ros-industrial / fanuc_crx10ia_support |
| `m710ic_ik` | robot_descriptions / fanuc_m710ic_description |

</details>

<details>
<summary><b>I2RT</b>: <code>ssik.prebuilt.i2rt</code> (2 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `yam_ik` | robot_descriptions / yam_description |
| `big_yam_ik` | i2rt-robotics / i2rt |

</details>

<details>
<summary><b>Enactic OpenArm</b>: <code>ssik.prebuilt.openarm</code> (2 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `left_ik` | enactic / openarm_description |
| `right_ik` | enactic / openarm_description |

</details>

<details>
<summary><b>Galaxea</b>: <code>ssik.prebuilt.galaxea</code> (2 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `r1pro_left_ik` | OpenGalaxea / GalaxeaManipSim (Apache-2.0) |
| `r1pro_right_ik` | OpenGalaxea / GalaxeaManipSim (Apache-2.0) |

</details>

<details>
<summary><b>Standard Bots</b>: <code>ssik.prebuilt.standard_bots</code> (3 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `thor_ik` | standardbots / ros2-realtime-api (robot_urdfs/thor.urdf) |
| `core_ik` | standardbots / ros2-realtime-api (robot_urdfs/core.urdf) |
| `spark_ik` | standardbots / ros2-realtime-api (robot_urdfs/spark.urdf) |

</details>

<details>
<summary><b>Abb</b>: <code>ssik.prebuilt.abb</code> (2 arms)</summary>

| Module | Fixture provenance |
|---|---|
| `yumi_left_ik` | robot_descriptions / yumi_description |
| `yumi_right_ik` | robot_descriptions / yumi_description |

</details>
<!-- /AUTOGEN -->
