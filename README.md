# ikfastpy

Standalone Python implementation of the [IKFast](https://www.openrave.org/docs/0.8.2/openravepy/ikfast/) analytic inverse kinematics solver, with no [OpenRAVE](https://github.com/rdiankov/openrave) dependency.

> **Status: pre-alpha.** Under active construction. Tracking work in [issues](../../issues).

## What this is

IKFast symbolically derives a closed-form inverse kinematics solver for a given kinematic chain, emits C++ source, and compiles it into a fast runtime library. Upstream, this lives inside OpenRAVE and is awkward to use as a Python module.

`ikfastpy` extracts the IKFast generator (`ikfast.py`, ~10k lines of sympy) into a standalone Python package, so you can do:

```python
import ikfastpy

arm = ikfastpy.Manipulator.from_urdf("ur5.urdf", base_link="base", ee_link="tool0")
T = arm.fk(q)               # forward kinematics: (4, 4) ndarray
solutions = arm.ik(T)       # inverse kinematics: list of joint configs
```

## What this is not

Unlike prior projects sharing this name (e.g. [andyzeng/ikfastpy](https://github.com/andyzeng/ikfastpy), [yijiangh/ikfast_pybind](https://github.com/yijiangh/ikfast_pybind)), `ikfastpy` is **not** a runtime wrapper around pre-generated C++. It contains the full symbolic generator and produces solvers on demand.

## Relationship to EAIK

[EAIK](https://github.com/OstermD/EAIK) is a complementary project that detects robots belonging to known closed-form kinematic families and runs hand-coded analytic solvers. It is fast at construction time but limited to recognised families. `ikfastpy` derives a custom solver per robot, supporting arbitrary 6-DOF chains and (via joint-locking) many redundant ones, at the cost of a slow first-run code-generation step. The two are useful in different regimes.

## License

LGPL-3.0, matching the upstream OpenRAVE IKFast sources from which this work is derived.
