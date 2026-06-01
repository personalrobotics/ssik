# Draft: ROS Discourse post

**Site:** https://discourse.ros.org
**Category:** General → "Release announcements" (or "Packages" if that fits better; the General/Release-announcements subcategory is the standard)
**Suggested title:** `ssik 1.1 — analytical inverse kinematics for 6R / 7R arms (PyPI, Python)`

---

## Title

`ssik 1.1 — analytical inverse kinematics for 6R / 7R arms (PyPI, Python)`

## Body

Hello ROS community,

I'd like to share **ssik**, a Python library for analytical inverse kinematics on 6R and 7R revolute manipulators. Version 1.1 just shipped to PyPI, and I'm posting in the hope it's useful to folks here — happy to answer questions, take requests for specific arms, or help anyone integrate it into their stack.

```bash
pip install ssik
```

### A bit of history

ssik grew out of work the Personal Robotics Lab has been doing on inverse kinematics for a long time. Rosen Diankov's original **IKFast** shaped how a generation of roboticists thought about analytical IK. ssik is the successor: pure-Python at runtime, BSD-3-Clause, and reaching kinematic classes that the original IKFast pipeline never quite handled cleanly (non-Pieper 6R like Kinova JACO 2; non-SRS 7R like Flexiv Rizon 4).

It's very much in the same spirit: per-arm specialised code, analytical branches, every solution enumerated — and I hope it carries some of that lineage forward in a way that's useful to the community.

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

This branch-closest-to-current pattern is, in my experience, where ssik earns its keep most clearly: **teleoperation and demonstration-collection pipelines** where jump-free joint trajectories at controller rate matter more than any single-shot accuracy number. Analytical IK gives you deterministic, continuous output — no seed-dependent jumps, no convergence stalls mid-demo — which is exactly the property that imitation-learning, behaviour-cloning, and VLA training pipelines need from their data-collection rigs. If you are building one of those rigs, I would love to hear what you need.

When a pose returns no solutions, `solve(T, explain=True)` reports *why* — which dispatch tier was used, which subproblem failed, and the residuals along the way. This was added to make debugging unreachable targets much less mysterious; teaching contexts especially seem to find it useful.

### Arms supported

Thirteen prebuilt arms ship with the wheel:

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
| UFactory xArm7 | 7R Pieper-wedge |
| UFactory xArm6 | non-Pieper 6R |
| Unitree Z1 | three-parallel 6R |
| AgileX PiPER | non-Pieper 6R |
| Flexiv Rizon 10 | non-SRS 7R |

Every prebuilt is exercised by a 500-pose Hypothesis fuzz sweep on every PR, with FK closure ‖FK(q) − T‖_F asserted below per-class tolerance. The goal is "correct, all the time". If you find a pose ssik gets wrong, that's a bug I'd very much like to hear about.

For any other arm: `ssik build my_arm.urdf --base base_link --ee tool0` emits a single-file Python artifact that imports just like the prebuilts. Adding a new prebuilt to the repo is now driven by a single [`MANIFEST.toml`](https://github.com/personalrobotics/ssik/blob/main/src/ssik/prebuilt/MANIFEST.toml) entry plus the URDF; the regen scripts and docs follow automatically. If your favourite arm isn't here, I'd be glad to help add it — open an issue with the URDF and we can take it from there.

### Where this fits in the ROS ecosystem

ssik is a **standalone Python library**, not a ROS package — there's no `package.xml` and no `rclpy` dependency. It's intended to be imported from Python nodes / planners / control code where a closed-form per-arm IK is more useful than a generic numerical solver. It's meant to complement, not replace, the IK tools you already use:

- **MoveIt's KDL plugin (default):** numerical seed-and-solve, returns one branch. ssik returns all of them, and on the comparable "pick one nearest the seed" pattern is faster on every arm we've measured. Both have their place.
- **TracIK:** SQP + Jacobian, numerical, same one-branch semantics. Excellent tool. The maintained Python binding (`pytracik`) currently ships a broken arm64 wheel; the ROS-native binding works fine inside ROS.
- **MoveIt's IKFast plugin:** same per-arm-codegen idea as ssik — and the direct predecessor in spirit. ssik adds coverage for non-Pieper 6R and non-SRS 7R, and ships as `pip install` rather than a C++ codegen pipeline. For the classes IKFast handles well, IKFast is still a fine choice.

A MoveIt plugin that uses ssik as the backend would be a lovely contribution — feasible but not yet built. If anyone wants to take that on, please file an issue and I'll happily review the PR.

### Calibration / non-nominal URDFs

The 13 prebuilts are built against nominal manufacturer geometry. For UR's `.calibrated_urdf` per-arm offsets, attached grippers / suction cups, or any non-nominal kinematic chain, run `ssik build` against your URDF — the emitted artifact bakes the exact geometry.

### Links

- **Repo:** https://github.com/personalrobotics/ssik
- **Docs:** https://personalrobotics.github.io/ssik/
- **PyPI:** https://pypi.org/project/ssik/
- **DOI:** https://doi.org/10.5281/zenodo.20278005
- **License:** BSD-3-Clause

Thank you to everyone in this community who has built and maintained the IK tools we've all relied on for years. Issues, questions, and requests for arms not yet covered are warmly welcomed, either on the repo or in this thread.

— Siddhartha Srinivasa, Personal Robotics Lab

---

## Notes for posting

- ROS Discourse expects a measured tone. The "Release announcements" subcategory has a regular cadence of similar posts; format matches.
- Specifically *do not* claim ssik replaces TracIK or KDL. It complements them; the analytical-branch enumeration is the differentiator.
- Be ready to answer: "MoveIt plugin?", "ROS 2 launch file example?", "BSD vs LGPL?", "calibration support?". All covered in the body but expect to repeat in replies.
- Tag the post with `inverse-kinematics`, `motion-planning`, `python` if the category allows.
