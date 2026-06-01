# Draft: robotics-worldwide mailing list post

**List:** robotics-worldwide@usc.edu (subscribe / archives at https://duerer.usc.edu/mailman/listinfo.cgi/robotics-worldwide)
**Suggested subject:** `[Software] ssik: open-source analytical inverse kinematics for 6R and 7R revolute manipulators`

---

## Subject

`[Software] ssik: open-source analytical inverse kinematics for 6R and 7R revolute manipulators`

## Body

Dear robotics-worldwide,

I'm delighted to share **ssik**, an open-source Python library for analytical inverse kinematics on 6R and 7R revolute manipulators. It's a project from the Personal Robotics Lab, released in the hope that it's useful to the community — happy to help anyone get it running, and happy to take requests for additional arms.

  Repository:  https://github.com/personalrobotics/ssik
  Docs:        https://personalrobotics.github.io/ssik/
  Install:     pip install ssik
  DOI:         https://doi.org/10.5281/zenodo.20278005
  License:     BSD-3-Clause

### Lineage

ssik is the successor to my Personal Robotics Lab's earlier IKFast authored by Rosen Diankov. IKFast has, for fifteen years, set the standard for what analytical IK should look like: per-arm specialised code, every branch enumerated, near-instant solves on the kinematic classes it covered. ssik is an attempt to carry that legacy forward into a moment when researchers and practitioners increasingly need IK for arms outside IKFast's original tier list: collaborative 7R manipulators with offset wrists, modular 6R cobots with non-standard joint twists, redundant arms where the analytical branch structure is the whole point of the analysis.

### What it does

ssik returns every analytical IK branch for a target end-effector pose. Thirteen commercial arms ship as prebuilt artifacts in the wheel (UR5, Puma 560, Kinova JACO 2, KUKA iiwa LBR 14, Kinova Gen3, Franka Panda, Flexiv Rizon 4, Kassow KR810, UFactory xArm7, UFactory xArm6, Unitree Z1, AgileX PiPER, Flexiv Rizon 10); any other arm is supported via `ssik build my_arm.urdf`, which emits a single-file Python artifact specialised to the URDF. Adding a new prebuilt to the repo is now a single MANIFEST.toml entry plus the URDF — a workflow we designed to make community contributions easy.

The library is intended for tasks where branch enumeration matters: motion planning (search over branches for clearance / manipulability), dexterity analysis, trajectory continuation across kinematic singularities, and reinforcement-learning environments where IK ground truth is part of the reward / observation. Numerical IK libraries that converge to a single configuration are not substitutes for this use case — but they remain excellent tools for the cases they were designed for.

### Algorithmic coverage

ssik dispatches by kinematic class. The novel coverage relative to the subproblem-decomposition libraries (IK-Geo, EAIK) is in the kinematic classes those libraries refuse:

  - non-Pieper 6R, e.g. Kinova JACO 2 with 60° DH twists between joints 4 and 5;
  - non-SRS 7R, e.g. Flexiv Rizon 4 and Kassow KR810 with offset wrists;
  - approximate-SRS 7R, e.g. Kinova Gen3's 12 mm offset.

The implementation is a clean-room reimplementation of algorithms from:

  - Elias & Wen (2022/2025), "IK-Geo: Unified Robot Inverse Kinematics Using Subproblem Decomposition" — six canonical subproblems + composition rules, the closed-form path for Pieper / SRS classes.
  - Husty, Pfurner, Schröcker (2007), "A new and efficient algorithm for the inverse kinematics of a general serial 6R manipulator", Mechanism and Machine Theory — Study-quaternion degree-16 univariate, the universal 6R fallback.
  - Raghavan & Roth (1993), "Inverse Kinematics of the General 6R Manipulator and Related Linkages", J. Mech. Des. — Sylvester resultant + numerical eigenvalue path.
  - Singh & Kreutz-Delgado (1989) — closed-form 7R analytical IK for the SRS class.

Algorithmic lineage is documented in each module's docstring, with citations back to the original publications. ssik does not vendor LGPL code from OpenRAVE / IKFast — it is a fresh implementation built on the open literature.

### Numerical behaviour and correctness

Forward-kinematics closure ‖FK(q) − T‖_F is below 1e-5 by default (10 µm position error on a 1 m arm, well below typical commercial-robot repeatability) and tightenable to ~1e-12 (0.1 nm scale) via the public TolerancePolicy with opt-in Levenberg-Marquardt polish.

Every prebuilt arm is exercised on every PR by a 500-pose Hypothesis fuzz sweep, with per-class FK-floor tolerances asserted on every solution. The intent is empirical: "bulletproof for every arm, on every pose, every time". Worst-case FK floors per kinematic class are characterised and documented; if a pose ssik returns ever fails this bound, we treat it as a release-blocking bug rather than a tolerance to widen.

For debugging unreachable targets, `solve(T, explain=True)` returns a structured trace of which dispatch tier was used, which subproblem(s) executed, and the residuals along the way. This has been particularly valuable in teaching contexts and when narrowing down "is it the arm, the URDF, or the target?" questions.

### Architecture

The runtime is pure NumPy + (for non-Pieper 6R and non-SRS 7R) SciPy. Per-arm symbolic preprocessing runs once at `ssik build` time and produces numeric coefficients that are baked into the emitted Python module; sympy is not on the runtime import path. Two Cython hot-loops (POE forward kinematics; LM refinement) are compiled to native extensions in the published wheels.

### Use cases I'd value feedback on

  - Teleoperation pipelines for demonstration collection. The `solve(T, max_solutions=1, q_seed=q_current)` pattern returns the analytical branch nearest the current joint state — deterministic, continuous, and fast enough for controller-rate IK with no seed-dependent jumps or convergence stalls mid-demonstration. This is the natural primitive for jump-free joint-space trajectories during data collection for imitation learning, behaviour cloning, and VLA training. It is also, for what it's worth, the use case I personally find most exciting; if you are building demonstration-collection infrastructure I would love to hear what you need.
  - Motion-planning groups doing branch enumeration today via numerical IK + restart-from-perturbed-seeds — ssik enumerates the analytical set directly.
  - Researchers reporting "IKFast cannot solve our arm" results — ssik may; I would be very happy to investigate specific URDFs.
  - Teaching contexts (kinematics, mechanisms): the per-arm artifact is a single readable Python file, suitable for assignments and labs. The diagnostic `explain=True` mode is designed with student debugging in mind.

Issues, PRs, and questions are warmly welcomed on the repo. I would especially value feedback from colleagues who have tried IKFast or the subproblem libraries on an arm those did not cover — those edge cases are exactly what ssik was built for, and hearing where it works (or doesn't) is the most useful signal we can get.

Thank you to the many people across this community whose work on IK over the decades has made this possible.

Best,
Siddhartha Srinivasa
Professor, Paul G. Allen School of Computer Science and Engineering
Personal Robotics Laboratory, University of Washington
https://goodrobot.ai

---

## Notes for posting

- robotics-worldwide is a moderated, low-volume academic list. The subject-line prefix "[Software]" is the convention for tool announcements.
- One post per major release is well within community norms; do not post for every PATCH.
- Expected replies: comparison questions (IKFast, IK-Geo, EAIK, OPW); requests for specific arms; bug reports. Be ready to handle each with a specific reference back to the docs.
