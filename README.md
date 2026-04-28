# ssik

Analytical inverse kinematics for Python, built around the **EAIK gap** — the arms whose IK isn't a one-line call against EAIK or IK-Geo.

> **Status: pre-alpha, mid-rebuild.** See the [umbrella rebuild issue](https://github.com/siddhss5/ikfastpy/issues/37) for the current architecture.

> **Renamed from `ikfastpy`.** The package was originally a port of OpenRAVE's [IKFast](https://www.openrave.org/docs/0.8.2/openravepy/ikfast/) symbolic IK generator. The IKFast general-solver path turned out unfixable on modern sympy for non-Pieper 6R arms, so the project was rebuilt around the **subproblem-decomposition approach** ([IK-Geo](https://github.com/rpiRobotics/ik-geo), [EAIK](https://github.com/OstermD/EAIK)) for Pieper-class arms, plus a **numeric Raghavan–Roth + Manocha–Canny** pipeline for the EAIK gap (non-Pieper arms like Kinova JACO 2, Agilex Piper, Flexiv Rizon). The PyPI name change reflects the algorithmic rewrite.

## What this is

For Pieper-class arms (UR5, Puma 560, Fanuc, KUKA KR, ABB IRB) ssik bundles tier-0 closed-form solvers ported from IK-Geo and runs them at ~50–200 µs warm-cache, machine-precision FK closure.

For non-Pieper arms (JACO 2, Piper, Rizon 4, custom geometries with no parallel/intersecting axis triples) ssik runs a tier-2 numeric Raghavan–Roth pipeline at ~2.25 ms median warm-cache, FK error 3.7e-13, **all branches at once**.

```python
import ssik

arm = ssik.Manipulator.from_urdf("ur5.urdf", base_link="base", ee_link="tool0")
T = arm.fk(q)              # forward kinematics: (4, 4) ndarray
solutions = arm.ik(T)      # inverse kinematics: list of Solutions with provenance
```

The dispatcher routes each kb to the best-matching solver:

- **Tier 0** (closed-form): spherical wrist (Pieper), three-parallel axes (UR-style). Puma 560, UR5, UR10, KUKA KR.
- **Tier 1** (1D search): any chain with one intersecting or parallel axis pair.
- **Tier 2** (numeric Raghavan–Roth + AE-3 leftvar selection): non-Pieper 6R. JACO 2, Piper.
- **Universal fallback**: Husty–Pfurner degree-16 univariate (planned).
- **7R via joint-locking**: Franka Panda, KUKA iiwa, Flexiv Rizon.

Third-party packages register new solvers via the `ssik.solvers` entry-point group; no core patching required.

Per-robot support (URDF source + which solver handles each arm) is tracked in [SUPPORTED_ROBOTS.md](SUPPORTED_ROBOTS.md).

## Relation to prior work

Unlike the other Python packages named `ikfastpy` (e.g. [andyzeng/ikfastpy](https://github.com/andyzeng/ikfastpy), [yijiangh/ikfast_pybind](https://github.com/yijiangh/ikfast_pybind)), `ssik` is **not** a runtime wrapper around pre-generated C++. It is a Python-native analytical IK framework that combines subproblem decomposition with the Raghavan–Roth numeric pipeline under one public API.

[EAIK](https://github.com/OstermD/EAIK) is a closely related project — C++/pybind11, built directly on [IK-Geo](https://github.com/rpiRobotics/ik-geo). EAIK's subproblem catalog is the initial bundled algorithm in `ssik`'s registry; what `ssik` adds is the tier-2 RR numeric solver for non-Pieper arms (the EAIK gap), the `Solution` dataclass with transparent refinement diagnostics, and a plugin surface for future specialist solvers.

## License

LGPL-3.0-or-later ([`LICENSE`](LICENSE)). The license stems from the original `ikfastpy` port; the codebase no longer contains the vendored OpenRAVE IKFast tree (removed in [#84](https://github.com/siddhss5/ikfastpy/issues/84)). Re-licensing to a more permissive option (BSD-3 to match IK-Geo's upstream license) is on the roadmap pending audit.
