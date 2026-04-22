# ssik

A pluggable analytical inverse-kinematics library for Python.

This site serves two audiences:

- **Users** who want a fast, closed-form IK solver in Python — start with the *Tutorial* (forthcoming) or jump to the *Reference*.
- **Learners** who want to understand how analytic inverse kinematics actually works — the tutorial walks from first principles through the subproblem-decomposition and resultant-elimination algorithms the library uses under the hood.

## Status

Pre-alpha, mid-rebuild. The original `ikfastpy` project (a port of OpenRAVE's IKFast) was renamed to `ssik` and is being rebuilt around the subproblem-decomposition approach. Implementation is tracked in the [GitHub issues](https://github.com/siddhss5/ikfastpy/issues); see the [umbrella rebuild issue](https://github.com/siddhss5/ikfastpy/issues/37) for the current architecture. The docs site you're reading is the framework — chapters land as the implementation lands.

## What this is not

Unlike prior Python projects sharing the legacy name (e.g. [andyzeng/ikfastpy](https://github.com/andyzeng/ikfastpy), [yijiangh/ikfast_pybind](https://github.com/yijiangh/ikfast_pybind)), `ssik` is **not** a runtime wrapper around pre-generated C++. It is a Python-native analytical IK framework with a pluggable solver registry: subproblem decomposition as the primary tier, Husty-Pfurner as a universal analytical fallback, and specialist solvers (GeoFIK, stereographic-SEW, and future algorithms) plug in via entry-points without core patches.

## Acknowledgements

The rebuild draws on the subproblem-decomposition approach of Elias & Wen ([IK-Geo](https://arxiv.org/abs/2211.05737), 2024), Ostermeier ([EAIK](https://arxiv.org/abs/2409.14815), 2024), and Husty & Pfurner (2007). The vendored IKFast tree originates from Rosen Diankov's work at Carnegie Mellon University; see the [Bibliography](bibliography.md) for the primary sources.
