# Setting up your robot

Guide for getting your specific arm working with ssik — beyond the <!-- AUTOGEN:arm_count -->25<!-- /AUTOGEN --> prebuilts. Three real-world friction points: **URDF readiness**, **link selection** (base + EE + tool), and **verification** (does the baked geometry match your robot?).

## When a prebuilt is enough vs when to `ssik build`

The <!-- AUTOGEN:arm_count -->25<!-- /AUTOGEN --> prebuilts cover **nominal manufacturer geometry with a bare flange**. They work when:

- You're using the same URDF source we built against (ros-industrial, vendor reference, etc.)
- Your robot's calibration matches the nominal kinematic parameters
- Your end-effector IS the flange — no gripper / suction cup / custom tool past it
- Your URDF link names match what we baked (see the [quickstart table](quickstart.md#shipped-prebuilts))

If **any** is false, especially on a 7R arm with any tool attached, `ssik build` your own.

## Step 1: get the URDF ready

ssik's URDF loader (`ssik._urdf`, backed by `urchin`) is strict about valid URDFs but tolerant of common omissions:

- ✅ Missing inertias and dynamics — ssik is kinematic-only
- ✅ Missing collision meshes — not used by ssik
- ✅ Missing visual meshes — not used by ssik
- ❌ Non-orthonormal `rpy` quaternions — Xacro sometimes produces near-orthonormal that errors. Re-export from the source if you hit a `urchin` parse error.
- ❌ Mid-chain fixed joints — must be either welded into adjacent revolute frames or expanded. Ssik treats the chain as 6R or 7R revolute; fixed joints break the chain.
- ❌ Continuous joints without limits — ssik treats unset limits as `±2π`, which may produce out-of-range IKs. Set explicit `limit` tags.

### Xacro and MJCF descriptions

You don't need to pre-convert other formats — every loader entry point (`Manipulator.from_urdf`, `ssik build`, `ssik classify`, `ssik add-arm`) accepts them directly:

- **Xacro** (`.xacro` / `*.urdf.xacro`, or a `.urdf` with a xacro namespace) — expanded via `xacrodoc` (`pip install ssik[xacro]`). This resolves `<xacro:include>`, macros, and substitution args, so multi-file descriptions like the Universal Robots family just work. Pass parametrized args with `--xacro-arg name:=value` (repeatable), e.g.:
  ```bash
  ssik build ur.xacro --base base_link --ee tool0 --xacro-arg ur_type:=ur10e
  ```
- **MJCF** (MuJoCo `.xml`) — load via `ssik._mjcf.load_mjcf_kinbody_normalized(path, base_body, ee_body)` (`pip install ssik[mjcf]`). Parsed through `mujoco` itself, so `<default>` classes, `<compiler>` settings, and `<include>` are all honored.

`ssik add-arm` vendors a **plain, expanded, kinematics-only URDF** into `tests/fixtures/` regardless of the input format.

### Calibrated URDFs (UR's `.calibrated_urdf`, etc.)

UR ships per-arm calibration offsets that nudge the nominal DH parameters by ~mm. Two paths:

**Use the calibrated file with `ssik build`:**
```bash
ssik build my_ur5_calibrated.urdf --base base_link --ee tool0
```
The emitted artifact bakes the calibration. Specific to that physical robot.

**Use the prebuilt with `respect_limits=False` + post-multiply:** if the calibration is small enough that the IK still converges, you can apply a per-call correction in T-space. Brittle; prefer the build-from-calibrated path.

## Step 2: pick `--base` and `--ee`

This is the most common gotcha. ssik solves IK from `--base` to `--ee` — meaning **`T_target` is the pose of `--ee` expressed in `--base`'s frame**.

Convention varies by source URDF. Common patterns:

| URDF source | Typical `--base` | Typical `--ee` |
|---|---|---|
| ros-industrial UR | `base_link` | `ee_link` (UR's `tool0` is past a fixed `tool0_joint`) |
| Vendor UR | `base_link` | `tool0` |
| Franka (`franka_description`) | `panda_link0` | `panda_hand` or `panda_link8` |
| KUKA iiwa (mujoco_menagerie) | `world` | `tool0` |
| MoveIt configs | usually `<arm>_base_link` | end of the kinematic group |

**Verify your choice**:

```python
import ssik
arm = ssik.Manipulator.from_urdf("your.urdf", base="base_link", ee="tool0")
print(arm.solver_name, arm.dof)
import numpy as np
T_home = arm.fk(np.zeros(arm.dof))
print(T_home[:3, 3])    # position at q=0; does it match your robot's documented home?
```

If `T_home` doesn't match what your robot's data sheet says about the q=0 pose, your `--base` / `--ee` is wrong (or your URDF is wrong).

## Step 3: tool / gripper attachment

If your end-effector isn't the bare flange — gripper, suction cup, weld torch, lidar — you have two options.

### Option A (recommended): bake the tool into the URDF

Add a fixed joint + tool link to your URDF before `ssik build`:

```xml
<link name="tool_tip"/>
<joint name="tool_attachment" type="fixed">
  <parent link="tool0"/>
  <child link="tool_tip"/>
  <origin xyz="0 0 0.15" rpy="0 0 0"/>    <!-- 15 cm gripper offset along z -->
</joint>
```

Then `ssik build my_arm.urdf --base base_link --ee tool_tip`. The baked artifact's `T_target` is now the pose of `tool_tip` in `base_link` — no per-call math.

### Option B: post-multiply T_target in Python

If you can't edit the URDF, build the artifact against the flange and apply the tool offset in your own code:

```python
import numpy as np
from ssik.prebuilt import ur5_ik

T_tool_in_flange = np.eye(4)
T_tool_in_flange[2, 3] = 0.15        # 15 cm along the flange's z-axis

# We want: T_tool_in_base = arm.fk(q) @ T_tool_in_flange
# Inverse direction for IK: arm.solve gets T_flange_in_base
T_flange_in_base = T_tool_target_in_base @ np.linalg.inv(T_tool_in_flange)
sols = ur5_ik.solve(T_flange_in_base)
```

Brittle if you ever change the tool — prefer Option A.

## Step 4: verify the baked geometry

Once you have `<your_arm>_ik.py`, sanity-check that:

```python
import my_arm_ik
import numpy as np

# 1. DOF matches your hardware
assert my_arm_ik.DOF == 6   # or 7 for redundant arms

# 2. Frame conventions match what you asked for
print(my_arm_ik.BASE_LINK, "->", my_arm_ik.EE_LINK)

# 3. Home pose matches the manufacturer spec
print(my_arm_ik.T_HOME)

# 4. FK and IK round-trip at a non-singular config
q_test = np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6])    # or DOF=7 equivalent
T = my_arm_ik.fk(q_test)
sols = my_arm_ik.solve(T)
assert sols, "round-trip failed -- baked geometry is likely wrong"
max_fk = max(s.fk_residual for s in sols)
print(f"{len(sols)} sols, max FK residual = {max_fk:.2e}")
# Should print ~10^-12 to 10^-5 depending on solver class (see arm_coverage)
```

If step 4 fails on a generic non-singular `q`, the URDF and ssik disagree on geometry. Common causes: wrong `--base` / `--ee`, missing fixed-joint expansion, non-orthonormal frames.

## Step 5: optional — re-run `ssik build` after upgrades

Old artifacts keep working when you `pip install -U ssik`, but they're frozen against the ssik version that built them. To pick up later solver fixes, re-run `ssik build`. Behaviour is byte-stable across same-ssik-version regenerations (verified by `tests/test_artifact_snapshots.py`).

## Common failure modes

### `ssik build` says "Best solver: husty_pfurner.general_6r"

You have a non-Pieper 6R with no specialised solver. The HP universal fallback works but is slow (~25-200 ms per IK vs ~1 ms for tier-0). Two options:

- Accept the slower solve time
- File an issue with your DH parameters; if your arm matches one of the un-implemented specialist classes (e.g. a different non-Pieper structure than JACO 2), we can add support

### Tighter FK closure than the default

The default `subproblem_numerical = 1e-5` is appropriate for control (10 µm position error on a 1 m arm, well below typical robot repeatability). For machine precision (RL training, differentiable IK, sample-based planning), opt in:

```python
from ssik import TolerancePolicy
tight = TolerancePolicy(
    axis_parallel=1e-8, axis_intersect=1e-8,
    subproblem_feasibility=1e-9, subproblem_numerical=1e-9,
    subproblem_degeneracy=1e-12, subproblem_dedup=1e-3,
)
sols = my_arm_ik.solve(T_target, policy=tight, allow_refinement=True)
# every returned IK FK-closes ~1e-10 (0.1 nm position scale)
```

See [Arm coverage → worst-case FK floor](arm_coverage.md#worst-case-fk-floor-under-adversarial-fuzz) for per-arm behaviour under each policy.

### Joint limits cause `solve()` to return `[]`

By default `solve()` runs `respect_limits=True`. If your URDF limits are tighter than the analytical IK can produce, the postprocess pass drops them all. Two diagnostics:

```python
sols = arm.solve(T, respect_limits=False)            # raw geometric set
sols, diag = arm.solve(T, explain=True)              # attribution
print(diag.dropped_by_limits, "filtered by limits")
```

If `dropped_by_limits == raw_candidates`, your limits are too tight or the pose is genuinely unreachable within them.
