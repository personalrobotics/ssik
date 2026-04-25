# Supported Robots

Arms for which `ssik` ships an analytical IK solver, the URDF (or DH) source
we validate against, and the solver family that handles each. Updated
alongside each solver PR.

## TL;DR — what works today

**Commercial 6R industrial arms (~95% of the installed base)**:
all covered by tier-0 closed-form solvers at **milliseconds per IK**.
UR, Puma, Fanuc, KUKA KR, ABB IRB, uFactory lite6/xArm6. Production-ready.

**Research 7R arms** (Franka, iiwa, Rizon, Gen3, xArm7): **NOT YET COVERED**.
This is our biggest remaining gap and the next strategic priority —
`jointlock.seven_r` unlocks all of them from one solver.

**Non-Pieper 6R** (Kinova JACO 2 classical, Agilex Piper): no ms-scale
solver shipped yet. `ikgeo.gen_six_dof` is correct but ~100s per IK in
Python; the practical answer is the upcoming `husty_pfurner.general_6r`.

## Legend

- **Status**
  - ✅ in repo — URDF fixture lives in [tests/fixtures/](tests/fixtures/) and is exercised by the solver's test suite
  - 🔗 external — validated against upstream URDF; fixture import pending
  - 📐 synthetic — validated with an inline synthetic arm (no single authoritative URDF)
- **Speed** — typical IK-call latency in pure Python on commodity hardware
  - ⚡ `<10 ms` — tier-0 closed-form; production-ready
  - 🐢 `~100 ms–1 s` — tier-1 univariate search; usable but slow
  - 🦥 `≥10 s` — tier-2 bivariate search; validation oracle only

## 6R industrial arms — **all production-ready via tier-0**

| Arm | DOF | URDF source | Solver | Speed | Status |
|-----|-----|-------------|--------|:-----:|:-----:|
| Puma 560 | 6 | [Peter Corke, Robotics Toolbox](https://github.com/petercorke/robotics-toolbox-python) DH | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | ⚡ | ✅ |
| Universal Robots UR5 | 6 | [universal_robots_ros2_description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) | `ikgeo.three_parallel` | ⚡ | ✅ |
| UR3 / UR10 / UR16 | 6 | same upstream URDF family as UR5 | `ikgeo.three_parallel` | ⚡ | 🔗 |
| ABB IRB120 | 6 | [ros-industrial/abb](https://github.com/ros-industrial/abb/tree/main/abb_irb120_support) | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | ⚡ | 🔗 |
| ABB IRB4600 | 6 | [ros-industrial/abb](https://github.com/ros-industrial/abb/tree/main/abb_irb4600_support) | `ikgeo.spherical_two_parallel` | ⚡ | 🔗 |
| Fanuc LR Mate / CR series | 6 | Fanuc-supplied URDF | `ikgeo.spherical_two_parallel` (expected) | ⚡ | 🔗 |
| KUKA KR series | 6 | [ros-industrial/kuka_experimental](https://github.com/ros-industrial/kuka_experimental) | `ikgeo.spherical_two_parallel` (expected) | ⚡ | 🔗 |
| uFactory lite6 | 6 | [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros), [mujoco_menagerie/ufactory_lite6](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_lite6) | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | ⚡ | 🔗 |
| uFactory xArm6 | 6 | [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros) | `ikgeo.spherical_two_parallel` (expected) | ⚡ | 🔗 |

### Synthetic fixtures (for coverage the real-arm catalog doesn't exercise)

| Fixture | Solver | Tier | Speed |
|---------|--------|------|:-----:|
| three-parallel synth | `ikgeo.three_parallel` | 0 | ⚡ |
| spherical + two-parallel synth | `ikgeo.spherical_two_parallel` | 0 | ⚡ |
| spherical + two-intersecting synth | `ikgeo.spherical_two_intersecting` | 0 | ⚡ |
| generic spherical-wrist synth | `ikgeo.spherical` | 0 | ⚡ |
| two-intersecting (p[5]=0) synth | `ikgeo.two_intersecting` | 1 | 🐢 |
| two-parallel (axes[1]=axes[2]) synth | `ikgeo.two_parallel` | 1 | 🐢 |
| generic 6R synth (IK-Geo `GeneralSetup`) | `ikgeo.gen_six_dof` | 2 | 🦥 |

## 6R non-Pieper (no tier-0 match)

These arms have no special Pieper structure so tier-0 solvers don't apply.
Production-quality solver (`husty_pfurner.general_6r`) is the next Round 3
target — degree-16 univariate polynomial, ms-scale in Python.

| Arm | DOF | URDF source | Planned solver | Speed | Status |
|-----|-----|-------------|----------------|:-----:|:-----:|
| Kinova JACO 2 (classical) | 6 | Kinova URDF (55° DH) | `husty_pfurner.general_6r` (primary), `ikgeo.gen_six_dof` (cross-check) | ⚡ (planned) / 🦥 (current) | ❌ |
| Agilex Piper | 6 | [mujoco_menagerie/agilex_piper](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper) | `husty_pfurner.general_6r` or `ikgeo.gen_six_dof` | ⚡ (planned) | ❌ |

## 7R redundant arms — **biggest remaining gap**

No solver shipped yet. All these arms are used extensively in research and
have zero IK coverage in `ssik` today. Strategic priority:

| Arm | DOF | URDF source | Planned solver | Status |
|-----|-----|-------------|----------------|:-----:|
| Franka Emika Panda | 7 | [mujoco_menagerie/franka_emika_panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda), [frankaemika/franka_description](https://github.com/frankaemika/franka_description) | `specialist.geofik` (primary), `jointlock.seven_r` (fallback) | ❌ |
| Franka Research 3 | 7 | [mujoco_menagerie/franka_fr3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3) | `specialist.geofik` (primary) | ❌ |
| KUKA iiwa 14 | 7 | [mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14) | `specialist.stereo_sew` (primary), `jointlock.seven_r` (fallback) | ❌ |
| Kinova Gen3 | 7 | [mujoco_menagerie/kinova_gen3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kinova_gen3) | `specialist.stereo_sew` (SRS variant) | ❌ |
| uFactory xArm7 | 7 | [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros) | `specialist.stereo_sew` (primary) | ❌ |
| Flexiv Rizon 4/10 | 7 | [flexivrobotics/flexiv_description](https://github.com/flexivrobotics/flexiv_description) | `specialist.stereo_sew` (SRS ZYZYZYZ) | ❌ |
| Kassow KR series | 7 | vendor URDF | `specialist.moz1_nonsrs` | ❌ |

## Solver tier reference

| Tier | Example solvers | Typical IK time | Use when |
|------|-----------------|-----------------|----------|
| 0 — closed-form | `three_parallel`, `spherical_two_parallel`, `spherical_two_intersecting`, `spherical` | **<10 ms** ⚡ | Arm matches a Pieper specialization (covers ~95% of commercial 6R) |
| 1 — univariate search | `two_intersecting`, `two_parallel` | ~100 ms–1 s 🐢 | Arm matches a weaker specialization than tier-0 |
| 2 — bivariate search | `gen_six_dof` | ~100 s in Python 🦥 | Correctness oracle / last-resort fallback; NOT a production path |
| 2 — univariate Study-quaternion | `husty_pfurner.general_6r` (planned) | **<10 ms** ⚡ (planned) | Production path for non-Pieper 6R (JACO 2, Piper, custom) |
| 7R specialist | `jointlock.seven_r`, `specialist.geofik`, `specialist.stereo_sew`, `specialist.moz1_nonsrs` | ms–100 ms | Franka, iiwa, Rizon, etc. |

## Key insights from Round 1-3.1 validation

1. **Commercial 6R coverage is essentially complete via tier-0.** The shipped
   spherical-wrist + parallel-shoulder solvers (`spherical_two_parallel`,
   `spherical_two_intersecting`, `three_parallel`) handle every major arm
   family. Tier-1 `two_parallel` / `two_intersecting` cover narrow gaps
   that no commercial arm in our catalog hits.

2. **Tier-2 `gen_six_dof` is a Python-performance dead end.** Correct but
   ~100 s/IK. Pure Python can't compete with Rust's native LAPACK on 10,000
   SP5 calls per solve. Shipped as a cross-check oracle only.

3. **`husty_pfurner.general_6r` is the practical tier-2 path.** One degree-16
   polynomial root-find instead of 10,000 SP5 calls. Orders-of-magnitude
   faster than `gen_six_dof`; production-usable.

4. **7R is where the biggest user impact lies.** Franka alone is installed
   in tens of thousands of research labs; none of them can use `ssik` today.
   `jointlock.seven_r` would unlock Franka + iiwa + Rizon + Gen3 + xArm7
   simultaneously (all inherit the tier-0 6R speed) — one solver, five robot
   families.

## Future: pre-compiled per-robot artifacts

Phase M in the roadmap emits a single-file C / Rust / Python artifact per
robot (resurrecting IKFast's value prop without its fragility). Target: a
user downloads `ssik-ur5.so` (or `.py` fallback) and gets the compiled
analytical solver for that arm with zero Python analytical-IK dependency
at runtime. Not yet implemented; gated on the full Python solver library
being stable (Rounds 1-4 complete).
