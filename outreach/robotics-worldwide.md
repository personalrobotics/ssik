# Draft: robotics-worldwide mailing list post

**List:** robotics-worldwide@usc.edu (subscribe / archives at https://duerer.usc.edu/mailman/listinfo.cgi/robotics-worldwide)
**Suggested subject:** `[Software] ssik: open-source analytical inverse kinematics for 6R and 7R revolute manipulators`

---

## Subject

`[Software] ssik: open-source analytical inverse kinematics for 6R and 7R revolute manipulators`

## Body

Dear robotics-worldwide,

I'd like to announce **ssik**, an open-source Python library for analytical inverse kinematics on 6R and 7R revolute manipulators.

  Repository:  https://github.com/personalrobotics/ssik
  Docs:        https://personalrobotics.github.io/ssik/
  Install:     pip install ssik
  DOI:         https://doi.org/10.5281/zenodo.20278005
  License:     BSD-3-Clause

ssik returns every analytical IK branch for a target end-effector pose. Twelve commercial arms ship as prebuilt artifacts in the wheel (UR5, Puma 560, Kinova JACO 2, KUKA iiwa LBR 14, Kinova Gen3, Franka Panda, Flexiv Rizon 4, Kassow KR810, UFactory xArm7, UFactory xArm6, Unitree Z1, AgileX PiPER); any other arm is supported via "ssik build my_arm.urdf", which emits a single-file Python artifact specialised to the URDF.

The library is intended for tasks where branch enumeration matters: motion planning (search over branches for clearance / manipulability), dexterity analysis, trajectory continuation across kinematic singularities, and reinforcement-learning environments where IK ground truth is part of the reward / observation. Numerical IK libraries that converge to a single configuration are not substitutes for this use case.

### Algorithmic coverage

ssik dispatches by kinematic class. The novel coverage relative to the subproblem-decomposition libraries (IK-Geo, EAIK) is the kinematic classes those libraries refuse:

  - non-Pieper 6R, e.g. Kinova JACO 2 with 55° DH twists between joints 4 and 5;
  - non-SRS 7R, e.g. Flexiv Rizon 4 and Kassow KR810 with offset wrists;
  - approximate-SRS 7R, e.g. Kinova Gen3's 12 mm offset.

The implementation is a clean-room reimplementation of algorithms from:

  - Elias & Wen (2022/2025), "IK-Geo: Unified Robot Inverse Kinematics Using Subproblem Decomposition" — six canonical subproblems + composition rules, the closed-form path for Pieper / SRS classes.
  - Husty, Pfurner, Schröcker (2007), "A new and efficient algorithm for the inverse kinematics of a general serial 6R manipulator", Mechanism and Machine Theory — Study-quaternion degree-16 univariate, the universal 6R fallback.
  - Raghavan & Roth (1993), "Inverse Kinematics of the General 6R Manipulator and Related Linkages", J. Mech. Des. — Sylvester resultant + numerical eigenvalue path.
  - Singh & Kreutz-Delgado (1989) — closed-form 7R analytical IK for the SRS class.

Algorithmic lineage is documented in each module's docstring. ssik does not vendor LGPL code from OpenRAVE / IKFast.

### Numerical behaviour

Forward-kinematics closure ‖FK(q) − T‖_F is below 1e-5 by default (10 µm position error on a 1 m arm, well below typical commercial-robot repeatability) and tightenable to ~1e-12 (0.1 nm scale) via the public TolerancePolicy with opt-in Levenberg-Marquardt polish. Hypothesis-driven fuzz tests at 500+ random poses per arm document the worst-case FK floor per kinematic class.

### Architecture

The runtime is pure NumPy + (for non-Pieper 6R and non-SRS 7R) SciPy. Per-arm symbolic preprocessing runs once at "ssik build" time and produces numeric coefficients that are baked into the emitted Python module; sympy is not on the runtime import path. Two Cython hot-loops (POE forward kinematics; LM refinement) are compiled to native extensions in the published wheels.

### Use cases I'd value feedback on

  - Motion-planning groups doing branch enumeration today via numerical IK + restart-from-perturbed-seeds — ssik enumerates the analytical set directly.
  - Researchers reporting "IKFast cannot solve our arm" results — ssik may; happy to investigate specific URDFs.
  - Teaching contexts (kinematics, mechanisms): the per-arm artifact is a single readable Python file, suitable for assignments / labs.

Issues, PRs, and questions welcome on the repo. I'd particularly value feedback from anyone who has tried IKFast or the subproblem libraries on an arm those did not cover.

Best,
Siddhartha Srinivasa
Personal Robotics Laboratory, University of Washington
https://goodrobot.ai

---

## Notes for posting

- robotics-worldwide is a moderated, low-volume academic list. The subject-line prefix "[Software]" is the convention for tool announcements.
- One post per major release is well within community norms; do not post for every PATCH.
- Expected replies: comparison questions (IKFast, IK-Geo, EAIK, OPW); requests for specific arms; bug reports. Be ready to handle each with a specific reference back to the docs.
