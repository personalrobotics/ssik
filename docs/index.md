# ssik

[![PyPI](https://img.shields.io/pypi/v/ssik.svg?v=1)](https://pypi.org/project/ssik/)
[![Python](https://img.shields.io/pypi/pyversions/ssik.svg?v=1)](https://pypi.org/project/ssik/)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/personalrobotics/ssik/blob/main/LICENSE)

Analytical inverse kinematics for 6R and 7R revolute robot arms. Each arm becomes a single self-contained Python module that returns **every IK branch** with FK closure well below typical robot repeatability — and tightenable to machine precision when needed.

```bash
pip install ssik
```

## Two-minute quickstart

```python
from ssik.prebuilt import franka_panda_ik
import numpy as np

T_target = np.eye(4)
T_target[:3, 3] = [0.5, 0.1, 0.3]
sols = franka_panda_ik.solve(T_target)        # every analytical IK branch
```

11 prebuilt arms ship with the wheel: UR5, Puma 560, JACO 2, iiwa14, Gen3, Franka Panda, Rizon 4, Kassow KR810, UFactory xArm7, UFactory xArm6, Unitree Z1. For other arms, run `ssik build <your.urdf>` once and import the emitted module.

## Where to go next

- **Just want to use it?** → [Quickstart](quickstart.md)
- **Adapting to your robot?** → [Setting up your robot](setting_up_your_robot.md) — calibration, custom tools, link conventions
- **Need the API surface?** → [API reference](api.md) — `Manipulator`, `Solution`, `Diagnostic`, `TolerancePolicy`, postprocess helpers
- **Want to understand the dispatch?** → [Architecture](architecture.md) — solver tier catalog + algorithmic lineage
- **Checking arm coverage?** → [Arm coverage](arm_coverage.md) — per-arm fixtures, speeds, FK floors

## Why ssik exists

Numerical-IK libraries take a seed, run damped least-squares to a **single** converged configuration, and stop. ssik returns **every analytical branch** at near-machine precision. Branch enumeration matters for motion planning (try every branch, pick the one with best clearance), for dexterity analysis (the manipulability ellipsoid is per-branch), and for trajectory continuation across kinematic singularities.

ssik covers the kinematic classes that the subproblem-decomposition libraries (EAIK, IK-Geo) refuse: **non-Pieper 6R** (JACO 2's 55° twists), **non-SRS 7R** (Flexiv Rizon 4, Kassow KR810), **approximate-SRS** (Kinova Gen3, 12 mm offset). See the [README comparison table](https://github.com/personalrobotics/ssik#measured-comparison-vs-eaik) for measured numbers.

## License

[BSD-3-Clause](https://github.com/personalrobotics/ssik/blob/main/LICENSE). Clean-room reimplementations of algorithms from BSD-3-licensed IK-Geo and the academic publications of Raghavan–Roth (1990), Manocha–Canny (1994), Singh–Kreutz (1989), and Husty–Pfurner (2007). Algorithmic lineage documented in module docstrings.
