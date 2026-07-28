# Quickstart

## Install

```bash
pip install ssik                  # core: library + 13 prebuilt arms + CLI
pip install ssik[urdf]            # adds urchin + sympy for ssik build / Manipulator.from_urdf
```

Python 3.11+. Wheels for Linux x86_64, macOS arm64, macOS x86_64, Windows x86_64.

## Use a prebuilt arm

```python
from ssik.prebuilt import franka_panda_ik
import numpy as np

T_target = np.eye(4)
T_target[:3, 3] = [0.5, 0.1, 0.3]
sols = franka_panda_ik.solve(T_target)        # every analytical IK branch
```

`sols` is a `list[Solution]`. Each `Solution` carries `q` (the joint vector), `fk_residual` (`‖FK(q) − T‖_F`), and which polish path fired. Empty list = pose is unreachable.

### Shipped prebuilts

<!-- AUTOGEN:quickstart_prebuilt_table -->
<details>
<summary><b>Universal Robots</b>: <code>ssik.prebuilt.universal_robots</code> (11 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `ur5_ik` | Universal Robots UR5 | three-parallel 6R | `base_link` | `ee_link` |
| `ur3e_ik` | Universal Robots UR3e | three-parallel 6R | `base_link` | `tool0` |
| `ur5e_ik` | Universal Robots UR5e | three-parallel 6R | `base_link` | `tool0` |
| `ur10e_ik` | Universal Robots UR10e | three-parallel 6R | `base_link` | `tool0` |
| `ur16e_ik` | Universal Robots UR16e | three-parallel 6R | `base_link` | `tool0` |
| `ur20_ik` | Universal Robots UR20 | three-parallel 6R | `base_link` | `tool0` |
| `ur30_ik` | Universal Robots UR30 | three-parallel 6R | `base_link` | `tool0` |
| `ur7e_ik` | Universal Robots UR7E | three-parallel 6R | `base_link` | `tool0` |
| `ur12e_ik` | Universal Robots UR12E | three-parallel 6R | `base_link` | `tool0` |
| `ur15_ik` | Universal Robots UR15 | three-parallel 6R | `base_link` | `tool0` |
| `ur18_ik` | Universal Robots UR18 | three-parallel 6R | `base_link` | `tool0` |

</details>

<details>
<summary><b>Unimation</b>: <code>ssik.prebuilt.unimation</code> (1 arm)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `puma560_ik` | KUKA Puma 560 | Pieper 6R (spherical wrist) | `base_link` | `wrist_3_link` |

</details>

<details>
<summary><b>Kinova</b>: <code>ssik.prebuilt.kinova</code> (5 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `jaco2_ik` | Kinova JACO 2 | non-Pieper 6R | `base_link` | `ee_link` |
| `gen3_ik` | Kinova Gen3 7-DOF | approximate-SRS 7R | `base_link` | `end_effector_link` |
| `gen3_lite_ik` | Kinova Gen3 Lite | non-Pieper 6R | `base_link` | `end_effector_link` |
| `j2s6s300_ik` | Kinova JACO j2s6s300 | Pieper 6R (spherical wrist) | `j2s6s300_link_base` | `j2s6s300_end_effector` |
| `j2s7s300_ik` | Kinova JACO j2s7s300 | approximate-SRS 7R (spherical wrist) | `j2s7s300_link_base` | `j2s7s300_link_7` |

</details>

<details>
<summary><b>KUKA</b>: <code>ssik.prebuilt.kuka</code> (2 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `iiwa14_ik` | KUKA iiwa LBR 14 | SRS 7R | `base` | `iiwa_link_ee_kuka` |
| `iiwa7_ik` | KUKA iiwa LBR 7 | SRS 7R (offset wrist) | `iiwa_link_0` | `iiwa_link_ee` |

</details>

<details>
<summary><b>Franka</b>: <code>ssik.prebuilt.franka</code> (2 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `panda_ik` | Franka Panda | spherical-shoulder + offset-wrist 7R | `panda_link0` | `panda_link8` |
| `fr3_ik` | Franka Research 3 | spherical-shoulder + offset-wrist 7R (Panda successor) | `fr3_link0` | `fr3_link8` |

</details>

<details>
<summary><b>UFactory</b>: <code>ssik.prebuilt.ufactory</code> (2 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `xarm7_ik` | UFactory xArm7 | approximately-spherical-shoulder 7R | `link_base` | `link7` |
| `xarm6_ik` | UFactory xArm6 | non-Pieper 6R (joint 6 y-offset) | `link_base` | `link_eef` |

</details>

<details>
<summary><b>Unitree</b>: <code>ssik.prebuilt.unitree</code> (1 arm)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `z1_ik` | Unitree Z1 | three-parallel 6R (UR-class) | `link00` | `link06` |

</details>

<details>
<summary><b>AgileX</b>: <code>ssik.prebuilt.agilex</code> (1 arm)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `piper_ik` | AgileX PiPER | non-Pieper 6R (joints 4 & 6 tilted axis) | `base_link` | `link6` |

</details>

<details>
<summary><b>Flexiv</b>: <code>ssik.prebuilt.flexiv</code> (2 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `rizon4_ik` | Flexiv Rizon 4 | non-SRS 7R | `base_link` | `flange` |
| `rizon10_ik` | Flexiv Rizon 10 | non-SRS 7R (~1.4 m reach) | `base_link` | `flange` |

</details>

<details>
<summary><b>Kassow</b>: <code>ssik.prebuilt.kassow</code> (1 arm)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `kr810_ik` | Kassow KR810 | non-SRS 7R | `base` | `end_effector` |

</details>

<details>
<summary><b>FANUC</b>: <code>ssik.prebuilt.fanuc</code> (8 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `crx3ia_ik` | FANUC CRX-3iA | non-Pieper 6R (non-spherical wrist) | `base_link` | `tool0` |
| `crx5ia_ik` | FANUC CRX-5iA | non-Pieper 6R (non-spherical wrist) | `base_link` | `tool0` |
| `crx10ia_ik` | FANUC CRX-10iA | non-Pieper 6R (non-spherical wrist) | `base_link` | `tool0` |
| `crx10ialp_ik` | FANUC CRX-10iA/LP | non-Pieper 6R (non-spherical wrist) | `base_link` | `tool0` |
| `crx20ial_ik` | FANUC CRX-20iA/L | non-Pieper 6R (non-spherical wrist) | `base_link` | `tool0` |
| `crx30ia_ik` | FANUC CRX-30iA | non-Pieper 6R (non-spherical wrist) | `base_link` | `tool0` |
| `crx10ial_ik` | FANUC CRX-10iA/L | non-Pieper 6R (non-spherical wrist, 150 mm y-offset) | `base_link` | `tool0` |
| `m710ic_ik` | FANUC M-710iC/70 | Pieper 6R (spherical wrist) | `base_link` | `link_6` |

</details>

<details>
<summary><b>I2RT</b>: <code>ssik.prebuilt.i2rt</code> (2 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `yam_ik` | I2RT YAM | non-Pieper 6R | `base_link` | `link_6` |
| `big_yam_ik` | I2RT big_yam | non-Pieper 6R | `base` | `gripper` |

</details>

<details>
<summary><b>Enactic OpenArm</b>: <code>ssik.prebuilt.openarm</code> (2 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `left_ik` | Enactic OpenArm v2.0 (left) | SRS 7R (non-Z*Z) | `openarm_left_base_link` | `openarm_left_ee_base_link` |
| `right_ik` | Enactic OpenArm v2.0 (right) | SRS 7R (non-Z*Z) | `openarm_right_base_link` | `openarm_right_ee_base_link` |

</details>

<details>
<summary><b>Galaxea</b>: <code>ssik.prebuilt.galaxea</code> (2 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `r1pro_left_ik` | Galaxea R1 Pro (left) | SRS 7R (non-Z*Z) | `left_arm_base_link` | `left_arm_link7` |
| `r1pro_right_ik` | Galaxea R1 Pro (right) | SRS 7R (non-Z*Z) | `right_arm_base_link` | `right_arm_link7` |

</details>

<details>
<summary><b>Standard Bots</b>: <code>ssik.prebuilt.standard_bots</code> (3 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `thor_ik` | Standard Bots Thor | three-parallel 6R | `base_link` | `tool0` |
| `core_ik` | Standard Bots Core | three-parallel 6R | `base_link` | `tool0` |
| `spark_ik` | Standard Bots Spark | three-parallel 6R | `base_link` | `tool0` |

</details>

<details>
<summary><b>Abb</b>: <code>ssik.prebuilt.abb</code> (2 arms)</summary>

| Module | Arm | Class | base_link | ee_link |
|---|---|---|---|---|
| `yumi_left_ik` | ABB YuMi (IRB 14000) left | approximate-SRS 7R | `yumi_body` | `yumi_link_7_l` |
| `yumi_right_ik` | ABB YuMi (IRB 14000) right | approximate-SRS 7R | `yumi_body` | `yumi_link_7_r` |

</details>
<!-- /AUTOGEN -->

Each prebuilt exposes `BASE_LINK`, `EE_LINK`, `DOF`, `T_HOME` constants so you can verify the baked geometry matches your robot:

```python
from ssik.prebuilt import franka_panda_ik
print(franka_panda_ik.BASE_LINK, "→", franka_panda_ik.EE_LINK)
# base_link → ee_link
print(franka_panda_ik.T_HOME[:3, 3])
# array([0.088, 0., 0.926])    ← matches Franka's documented home
```

## Trajectory tracking pattern

For real-time control / teleop, "give me the IK closest to where I am now":

```python
q_current = np.array([0.0, -0.5, 0.0, 0.7, 0.0, 1.2, 0.0])

# max_solutions=1 + q_seed: returns the single solution nearest q_current.
# On 7R jointlock arms the seed also drives the lock-outward fast path
# (~20x faster than the full sweep).
sols = franka_panda_ik.solve(T_target, max_solutions=1, q_seed=q_current)
q_command = sols[0].q if sols else q_current
```

Two knobs refine what "nearest" means when a seed is given:

```python
# seed_metric (default "wrap_linf"): rank by the LARGEST single-joint move,
# so the arm holds its branch instead of flipping. Use "wrap_l2" for
# summed-distance ranking.
sols = franka_panda_ik.solve(T_target, q_seed=q_current, seed_metric="wrap_linf")

# seed_tolerance (radians): a HARD bound -- only return solutions where every
# joint is within the tolerance of the seed. The list may come back EMPTY,
# which is the signal that smooth continuation isn't possible at this pose
# (replan / accept a larger jump). Best-effort behaviour when omitted.
sols = franka_panda_ik.solve(
    T_target, q_seed=q_current, max_solutions=1, seed_tolerance=np.deg2rad(6)
)
if not sols:
    ...  # no config within 6 deg of q_current -- handle the discontinuity
```

## When `solve()` returns an empty list

Use `explain=True` to attribute the failure:

```python
import ssik
arm = ssik.Manipulator.from_urdf("my_arm.urdf", base="base_link", ee="tool0")

sols, diag = arm.solve(T_target, explain=True)
if not sols:
    print(diag.summary())
    # solver: ikgeo.three_parallel (tier 0)
    # dispatch: Three consecutive parallel axes at joints (1, 2, 3) ...
    #   -> 0 raw candidates: pose appears unreachable
```

Distinguishes **unreachable** (zero raw candidates) from **all-filtered** (out-of-limits or below FK threshold) from **capped** (truncated by `max_solutions`).

## Build an artifact for your own arm

```bash
ssik build my_arm.urdf --base base_link --ee tool0
# → my_arm_ik.py
```

Build time:
- **<1 s** for tier-0 closed-form (UR-class, Pieper, SRS-class 7R)
- **~30 s** for non-Pieper 6R (Raghavan–Roth symbolic derivation)
- **7–20 min** for non-SRS 7R (cached HP per lock sample)

Then `import my_arm_ik` and use exactly like a prebuilt. See [Setting up your robot](setting_up_your_robot.md) for the full URDF-to-artifact workflow.
