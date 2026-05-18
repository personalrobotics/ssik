# Draft: ROS Discourse post

**Site:** https://discourse.ros.org
**Category:** General → "Release announcements" (or "Packages" if that fits better; the General/Release-announcements subcategory is the standard)
**Suggested title:** `ssik 1.1 — analytical inverse kinematics for 6R / 7R arms (PyPI, Python)`

---

## Title

`ssik 1.1 — analytical inverse kinematics for 6R / 7R arms (PyPI, Python)`

## Body

I'd like to introduce **ssik**, a Python library for analytical inverse kinematics on 6R and 7R revolute manipulators. Version 1.1 just shipped to PyPI.

```bash
pip install ssik
```

### What it does

ssik returns **every analytical IK branch** for a target pose — not one converged config like a numerical IK does. Each branch is a complete `q` plus the FK residual against the original target.

```python
from ssik.prebuilt import ur5_ik
import numpy as np

T_target = np.eye(4); T_target[:3, 3] = [0.5, 0.1, 0.3]
sols = ur5_ik.solve(T_target)    # up to 8 branches for a Pieper-class arm
```

For trajectory tracking / teleop, the standard pattern is "give me the IK closest to where the robot is now":

```python
q_current = np.array([...])    # current joint state from /joint_states
sols = ur5_ik.solve(T_target, max_solutions=1, q_seed=q_current)
q_command = sols[0].q if sols else q_current
```

### Arms supported

Eight prebuilt arms ship with the wheel:

| Arm | Class |
|---|---|
| Universal Robots UR5 | three-parallel 6R |
| KUKA Puma 560 | Pieper 6R |
| Kinova JACO 2 | non-Pieper 6R |
| KUKA iiwa LBR 14 | SRS 7R |
| Kinova Gen3 | approximate-SRS 7R |
| Franka Panda | anthropomorphic 7R |
| Flexiv Rizon 4 | non-SRS 7R |
| Kassow KR810 | non-SRS 7R |

For any other arm: `ssik build my_arm.urdf --base base_link --ee tool0` emits a single-file Python artifact that imports just like the prebuilts.

### Where this fits in the ROS ecosystem

ssik is a **standalone Python library**, not a ROS package — there's no `package.xml` and no `rclpy` dependency. It's intended to be imported from Python nodes / planners / control code where a closed-form per-arm IK is more useful than a generic numerical solver.

Comparison to what's commonly used in ROS:

- **MoveIt's KDL plugin (default):** numerical seed-and-solve. Returns one branch. ssik returns all of them and is faster on every arm we've measured for the comparable "pick one nearest the seed" pattern.
- **TracIK:** SQP + Jacobian, numerical. Same one-branch semantics. The maintained Python binding (`pytracik`) currently ships a broken arm64 wheel; the ROS-native binding works fine inside ROS.
- **MoveIt's IKFast plugin:** same per-arm-codegen idea as ssik. IKFast is the closest analog; ssik solves arms IKFast cannot (non-Pieper 6R, non-SRS 7R) and ships as `pip install` instead of a C++ codegen pipeline.

A MoveIt plugin that uses ssik as the backend is feasible but not yet built; if anyone wants to take that on, please file an issue and I'll review the PR.

### Calibration / non-nominal URDFs

The 8 prebuilts are built against nominal manufacturer geometry. For UR's `.calibrated_urdf` per-arm offsets, attached grippers / suction cups, or any non-nominal kinematic chain, run `ssik build` against your URDF — the emitted artifact bakes the exact geometry.

### Links

- **Repo:** https://github.com/personalrobotics/ssik
- **Docs:** https://personalrobotics.github.io/ssik/
- **PyPI:** https://pypi.org/project/ssik/
- **License:** BSD-3-Clause

Issues / questions welcome on the repo or in this thread.

---

## Notes for posting

- ROS Discourse expects a measured tone. The "Release announcements" subcategory has a regular cadence of similar posts; format matches.
- Specifically *do not* claim ssik replaces TracIK or KDL. It complements them; the analytical-branch enumeration is the differentiator.
- Be ready to answer: "MoveIt plugin?", "ROS 2 launch file example?", "BSD vs LGPL?", "calibration support?". All covered in the body but expect to repeat in replies.
- Tag the post with `inverse-kinematics`, `motion-planning`, `python` if the category allows.
