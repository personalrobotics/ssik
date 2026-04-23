# ssik

Pluggable analytical inverse-kinematics library for Python. Subproblem decomposition, extensible solver registry, URDF-native.

> **Status: pre-alpha, mid-rebuild.** See the [umbrella rebuild issue](https://github.com/siddhss5/ikfastpy/issues/37) for the current architecture and phase tracking.

> **Renamed from `ikfastpy`.** The package was originally a port of OpenRAVE's [IKFast](https://www.openrave.org/docs/0.8.2/openravepy/ikfast/) symbolic IK generator. The IKFast general-solver path turned out unfixable on modern sympy for non-Pieper 6R arms, so the project is being rebuilt around the subproblem-decomposition approach ([IK-Geo](https://github.com/rpiRobotics/ik-geo), [EAIK](https://github.com/OstermD/EAIK)) with a pluggable registry for specialist solvers (Husty-Pfurner, GeoFIK, stereographic-SEW, future algorithms). The vendored IKFast tree remains available for Pieper-class arms during the transition, quarantined under `ssik._vendor`. See #37.

## What this will be

```python
import ssik

arm = ssik.Manipulator.from_urdf("ur5.urdf", base_link="base", ee_link="tool0")
T = arm.fk(q)              # forward kinematics: (4, 4) ndarray
solutions = arm.ik(T)      # inverse kinematics: list of solutions with provenance
```

The dispatcher routes each chain to the best-matching analytical solver:

- **Tier 0** (closed-form): spherical wrist (Pieper), three-parallel axes (UR-style). Puma, UR5, UR10.
- **Tier 1** (1D search): any chain with one intersecting or parallel axis pair.
- **Tier 2** (2D search + numeric polish): fully general 6R. JACO 2.
- **Universal fallback**: Husty-Pfurner Study-quaternion degree-16 univariate for any 6R.
- **7R via joint-locking**: Franka Panda, KUKA iiwa.

Third-party packages register new solvers via the `ssik.solvers` entry-point group; no core patching required.

Per-robot support (URDF source + which solver handles each arm) is tracked in
[SUPPORTED_ROBOTS.md](SUPPORTED_ROBOTS.md).

## Relation to prior work

Unlike the other Python packages named `ikfastpy` (e.g. [andyzeng/ikfastpy](https://github.com/andyzeng/ikfastpy), [yijiangh/ikfast_pybind](https://github.com/yijiangh/ikfast_pybind)), `ssik` is not a runtime wrapper around pre-generated C++. It is a Python-native analytical IK framework that combines subproblem decomposition with specialist solvers under one public API.

[EAIK](https://github.com/OstermD/EAIK) is a closely related project — C++/pybind11, built directly on [IK-Geo](https://github.com/rpiRobotics/ik-geo). EAIK's subproblem catalog is the initial bundled algorithm in `ssik`'s registry. Both projects are Apache/BSD-licensed analytical-IK tooling; `ssik` additionally exposes a plugin surface so future algorithms and universal fallbacks can coexist.

## License

This project is dual-licensed, matching its upstream sources:

- The vendored **IKFast generator** (`ikfast.py`, `ikfast_generator_cpp.py` under `src/ssik/_vendor/`) is **LGPL-3.0-or-later** ([`LICENSE`](LICENSE)) from [rdiankov/openrave](https://github.com/rdiankov/openrave); see [`src/ssik/_vendor/UPSTREAM.md`](src/ssik/_vendor/UPSTREAM.md) for the pinned upstream commit.
- The vendored **runtime header** (`ikfast.h`) is **Apache-2.0** ([`LICENSE.apache`](LICENSE.apache)).
- New code written for the rebuild is Apache-2.0, clean-room from [Elias & Wen (IK-Geo)](https://arxiv.org/abs/2211.05737) and the BSD-3 IK-Geo reference.

Generated solvers bind only the Apache-2.0 runtime header; LGPL stays with the generator itself.
