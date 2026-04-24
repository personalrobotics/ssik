# Supported Robots

Arms for which `ssik` ships an analytical IK solver, the URDF (or DH) source
we validate against, and the solver family that handles each. Updated
alongside each solver PR.

## Legend

- **Status**
  - ✅ in repo — URDF fixture lives in [tests/fixtures/](tests/fixtures/) and
    is exercised by the solver's test suite
  - 🔗 external — validated against upstream URDF; fixture import pending
  - 📐 synthetic — validated with an inline synthetic arm (no single
    authoritative URDF); common for testing the "generic" claim of a solver
    family
- **DOF** — active revolute joints
- **Solver** — module under [src/ssik/solvers/](src/ssik/solvers/)
- **Topology family** — structural class the arm belongs to (multiple may
  apply; the dispatcher in Phase C will pick by specialization rank)

## 6R industrial arms

| Arm | DOF | Source | Solver | Topology | Status |
|-----|-----|--------|--------|----------|:-----:|
| Puma 560 | 6 | [Peter Corke, Robotics Toolbox](https://github.com/petercorke/robotics-toolbox-python) DH | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | spherical wrist + parallel shoulder AND intersecting shoulder | ✅ |
| Universal Robots UR5 | 6 | [universal_robots_ros2_description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) | `ikgeo.three_parallel` | three parallel shoulder/elbow | ✅ |
| Universal Robots UR3 / UR10 / UR16 | 6 | same upstream URDF family as UR5 | `ikgeo.three_parallel` | three parallel | 🔗 |
| ABB IRB120 | 6 | [ros-industrial/abb](https://github.com/ros-industrial/abb/tree/main/abb_irb120_support) | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | spherical wrist + parallel AND intersecting shoulder | 🔗 |
| ABB IRB4600 | 6 | [ros-industrial/abb](https://github.com/ros-industrial/abb/tree/main/abb_irb4600_support) | `ikgeo.spherical_two_parallel` | spherical wrist + parallel shoulder | 🔗 |
| Fanuc LR Mate / CR series | 6 | Fanuc-supplied URDF | `ikgeo.spherical_two_parallel` (expected) | spherical wrist + parallel shoulder | 🔗 |
| KUKA KR series | 6 | [ros-industrial/kuka_experimental](https://github.com/ros-industrial/kuka_experimental) | `ikgeo.spherical_two_parallel` (expected) | spherical wrist + parallel shoulder | 🔗 |
| uFactory lite6 | 6 | [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros/tree/master/xarm_description/urdf/lite6), [mujoco_menagerie/ufactory_lite6](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_lite6) | `ikgeo.spherical_two_parallel`, `ikgeo.spherical_two_intersecting` | both shoulder specializations | 🔗 |
| uFactory xArm6 | 6 | [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros) | `ikgeo.spherical_two_parallel` (expected) | similar to lite6 | 🔗 |
| (synthetic three-parallel) | 6 | inline in `tests/test_three_parallel.py` | `ikgeo.three_parallel` | three parallel | 📐 |
| (synthetic spherical + two-parallel) | 6 | inline in `tests/test_spherical_two_parallel.py` | `ikgeo.spherical_two_parallel` | spherical wrist + parallel shoulder | 📐 |
| (synthetic spherical + intersecting) | 6 | inline in `tests/test_spherical_two_intersecting.py` | `ikgeo.spherical_two_intersecting` | spherical wrist + intersecting shoulder | 📐 |
| (synthetic generic spherical) | 6 | inline in `tests/test_ikgeo_spherical.py` (two variants) | `ikgeo.spherical` | spherical wrist, shoulder neither parallel nor intersecting | 📐 |

## 6R non-Pieper (tier-1 / tier-2, future)

| Arm | DOF | Source | Planned solver | Notes |
|-----|-----|--------|----------------|-------|
| Agilex Piper | 6 | [mujoco_menagerie/agilex_piper](https://github.com/google-deepmind/mujoco_menagerie/tree/main/agilex_piper) | `ikgeo.gen_six_dof` (tier-2) | axes 4,6 tilted x-axis, parallel but not coincident → no spherical wrist |
| Kinova JACO 2 | 6 | Kinova URDF | `ikgeo.gen_six_dof` + `husty_pfurner.general_6r` fallback | classical non-orthogonal 55° DH |

## 7R redundant arms (Round 4, future)

| Arm | DOF | Source | Planned solver | Topology |
|-----|-----|--------|----------------|----------|
| Franka Emika Panda | 7 | [mujoco_menagerie/franka_emika_panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda), [frankaemika/franka_description](https://github.com/frankaemika/franka_description) | `specialist.geofik` | SRS 7R |
| Franka Research 3 | 7 | [mujoco_menagerie/franka_fr3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_fr3) | `specialist.geofik` | SRS 7R |
| KUKA iiwa 14 | 7 | [mujoco_menagerie/kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kuka_iiwa_14), [ros-industrial/kuka_experimental](https://github.com/ros-industrial/kuka_experimental) | `specialist.stereo_sew` | SRS 7R |
| Kinova Gen3 | 7 | [mujoco_menagerie/kinova_gen3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kinova_gen3), [Kinovarobotics/ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex) | `specialist.stereo_sew` | SRS 7R variant |
| uFactory xArm7 | 7 | [xArm-Developer/xarm_ros](https://github.com/xArm-Developer/xarm_ros) | `specialist.stereo_sew` | SRS 7R |
| Flexiv Rizon 4 | 7 | [flexivrobotics/flexiv_description](https://github.com/flexivrobotics/flexiv_description) | `specialist.stereo_sew` | SRS 7R (ZYZYZYZ) |
| Kassow KR series | 7 | vendor URDF | `specialist.moz1_nonsrs` | NonSRS 7R |

## Fallbacks

- `husty_pfurner.general_6r` — universal 6R fallback; degree-16 Study-quaternion
  polynomial. Covers any 6R arm (even those numerically ill-conditioned for
  `ikgeo.gen_six_dof`).
- `jointlock.seven_r` — generic 7R wrapper; locks one joint, sweeps samples,
  dispatches the 6R slice to whichever 6R solver applies.

## Future: pre-compiled per-robot artifacts

Phase M in the roadmap emits a single-file C / Rust / Python artifact per
robot (resurrecting IKFast's value prop without its fragility). Target: a
user downloads `ssik-ur5.so` (or `.py` fallback) and gets the compiled
analytical solver for that arm with zero Python analytical-IK dependency
at runtime. Not yet implemented; gated on the full Python solver library
being stable (Rounds 1-4 complete).
