# ikfastpy

A standalone Python implementation of the [IKFast](https://www.openrave.org/docs/0.8.2/openravepy/ikfast/) analytic inverse kinematics solver, with no [OpenRAVE](https://github.com/rdiankov/openrave) dependency.

This site serves two audiences:

- **Users** who want a fast, closed-form IK solver in Python — start with the *Tutorial* (forthcoming) or jump to the *Reference*.
- **Learners** who want to understand how analytic inverse kinematics actually works — the tutorial walks from first principles through the symbolic resolution algorithm IKFast uses.

## Status

Pre-alpha. Implementation is being tracked in the [GitHub issues](https://github.com/siddhss5/ikfastpy/issues). The docs site you're reading is the framework — chapters land as the implementation lands.

## What this is not

Unlike prior Python projects sharing this name (e.g. [andyzeng/ikfastpy](https://github.com/andyzeng/ikfastpy), [yijiangh/ikfast_pybind](https://github.com/yijiangh/ikfast_pybind)), `ikfastpy` is **not** a runtime wrapper around pre-generated C++. It contains the full symbolic generator and produces solvers on demand from URDF, DH parameters, or raw kinematic specifications.

## Acknowledgements

The IKFast algorithm and reference implementation are the work of Rosen Diankov, developed during his PhD at Carnegie Mellon University. See the [Bibliography](bibliography.md) for the primary source.
