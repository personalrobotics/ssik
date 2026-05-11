# ssik

Analytical inverse kinematics for 6R and 7R revolute robot arms. Returns **every IK branch** at machine-precision FK closure, dispatches the right solver automatically, and ships a per-arm build artifact that contains the full IK pipeline as a self-contained Python module.

## Quickstart — use the build artifact

The intended deployment path is one `ssik build` per arm, then `import <arm>_ik` everywhere. The artifact is a single self-contained `.py` file with the per-arm KinBody constants, the dispatched solver, and any cached symbolic preprocessing already baked in — no URDF parsing, no `urchin`, no `sympy` on the import path.

```python
import franka_panda_ik           # prebuilt; see prebuilt/ for 8 ready-to-use arms
import numpy as np

T_target = np.eye(4); T_target[:3, 3] = [0.5, 0.1, 0.3]
sols = franka_panda_ik.solve(T_target)                   # all IK branches; empty list = unreachable
```

### Trajectory tracking / IK-based teleop

The canonical pattern for teleop and tracking is "give me the IK closest to where the robot is now":

```python
# Robot's current configuration (from joint sensors, last command, etc.).
q_current = np.array([0.0, -0.5, 0.0, 0.7, 0.0, 1.2, 0.0])

# Target pose updates every control tick (VR controller, planner, etc.).
T_target = ...

# max_solutions=1 + q_seed: solver searches the 1-parameter redundancy
# sweep starting from the lock-sample closest to q_current and short-
# circuits on the first valid in-limits branch (~5-6x faster than
# the full sweep on 7R jointlock arms; sub-ms on 6R / SRS arms).
sols = franka_panda_ik.solve(T_target, max_solutions=1, q_seed=q_current)
q_command = sols[0].q if sols else q_current             # empty list = unreachable
```

The same kwarg shape works on every prebuilt artifact (UR5, Puma 560, JACO 2, iiwa14, Gen3, Franka, Rizon 4, Kassow). By default `solve()` runs **`respect_limits=True`**: out-of-URDF-limit branches are dropped (with a `q ± 2π` rescue pass first). On 7R jointlock arms (Franka / Rizon / Kassow) the limits filter runs *during* the lock-sweep so `max_solutions=1` short-circuits on the first in-limits candidate rather than wasting samples on branches the postprocess would discard. Pass `respect_limits=False` for the raw geometric set when you want to filter yourself.

## Returns all solutions, not one

A single 6-DOF target pose admits up to **16 analytical IK branches** (8 typical for a Pieper-class arm: 4 shoulder × 2 elbow, with the wrist deterministic). For a 7R redundant arm the IK is a 1-parameter family; ssik discretises it into 32–256 branches per pose depending on the swivel-sample count. Every returned `Solution` carries `q`, the per-IK FK residual, and whether the Newton polish path fired (`refinement_used`).

Numerical IK libraries take a seed, run damped least-squares to a single converged configuration, and stop. Branch enumeration matters for motion planning (try every branch, pick the one with best clearance), for dexterity analysis (the manipulability ellipsoid is per-branch), and for trajectory continuation across kinematic singularities.

## Measured comparison vs EAIK

EAIK (Ostermeier 2024) is the canonical Python wrapper around C++ subproblem-decomposition solvers. It is analytical on the kinematic families it recognises and refuses everything else. The numbers below come from [`examples/04_compare_vs_eaik.py`](examples/04_compare_vs_eaik.py) over 100 random reachable poses per arm, Apple M3 single-thread, mean ± 95% CI via bootstrap (1000 resamples). FK residual is the Frobenius norm `‖FK(q) − T‖` against the original URDF / spec FK.

| Arm (class) | EAIK | ssik |
|---|---|---|
| UR5 (Pieper 6R, three-parallel) | 5 ± 0 µs / FK 2e-15 / 4 sols | 556 ± 12 µs / FK 2e-9 / 4 sols |
| Puma 560 (Pieper 6R, spherical wrist) | 5 ± 1 µs / FK 3e-14 / 8 sols | 245 ± 3 µs / FK 2e-14 / 8 sols |
| JACO 2 (**non-Pieper 6R**) | **refuses** ("6R-Unknown Kinematic Class") | 1.02 ± 0.04 ms / FK 3e-6 / 2 sols |
| iiwa14 (SRS 7R) | **refuses** ("only 1-6R robots are solvable") | 6.56 ± 0.49 ms / FK 4e-13 / 96 sols |
| Gen3 (**approximate-SRS 7R**, 12 mm offset) | **refuses** ("only 1-6R") | 69.37 ± 4.32 ms / FK 1e-12 / 47 sols |
| Franka Panda (anthropomorphic 7R) | **refuses** ("only 1-6R") | 28.03 ± 2.57 ms / FK 7e-13 / 9 sols |
| Rizon 4 (**non-SRS 7R**) | **refuses** ("only 1-6R") | 30.88 ± 8.80 ms / FK 4e-9 / 35 sols |
| Kassow KR810 (**non-SRS 7R**) | **refuses** ("only 1-6R") | 28.06 ± 10.63 ms / FK 7e-8 / 24 sols |

EAIK is ~100× faster than ssik on Pieper-class 6R — that is its native sweet spot, and ssik does not try to compete there. The interesting cells are the **refuses** ones: non-Pieper 6R (JACO 2) and every 7R arm. Those are the geometries ssik exists for. The "refuses (...)" strings are EAIK's actual error messages, captured verbatim by the bench harness. A numerical-IK comparison (MINK) is tracked separately in [#236](https://github.com/personalrobotics/ssik/issues/236).

ssik FK residuals above are the algebraic candidates returned by `solve()` with default tolerance policy and `respect_limits=True`. Branch counts on 7R arms (iiwa14: 96, Franka: 9, Rizon: 35) reflect URDF-joint-limit filtering; pass `respect_limits=False` to get the full geometric set. The `allow_refinement=True` opt-in runs Levenberg–Marquardt polish per algebraic candidate at a few hundred microseconds per branch — useful when an algebraic candidate lands just above `fk_atol` near a kinematic singularity; it does not always reach machine precision on tier-2 RR arms whose polish basin is narrow.

The algorithmic ingredients are not novel — Raghavan–Roth (1990), Manocha–Canny (1994), Singh–Kreutz (1989), Husty–Pfurner (2007). What's new is making the textbook pipelines survive on real ill-conditioned arms (AE-3 leftvar selection on JACO 2 drops `cond(m_quad)` from 3.75 × 10^16 to 127), composing them with a uniform dispatch layer, and packaging the whole thing as a deployable artifact.

## Build an artifact for your own arm

```bash
ssik build my_arm.urdf --base base_link --ee tool0
# → my_arm_ik.py    (~1-5 min build, all symbolic preprocessing baked in)
```

The artifact is a single `.py` file containing:

- the per-arm KinBody constants inlined as numpy literals,
- the dispatch choice baked at build time (no runtime classification),
- for non-Pieper sub-chains, the cached Raghavan–Roth symbolic derivations as base85-encoded zlib-compressed pickle blobs.

Module-init takes ~5 s (deserialise + re-`lambdify` the cached blobs); every subsequent `solve()` call hits warm-cache speed. **No URDF parsing, no `urchin` dependency, no cold-cache symbolic preprocessing at runtime, no sympy on the import path of the deployed artifact.** A robot stack that imports `my_arm_ik.py` carries no algorithmic complexity beyond what the build pipeline already resolved.

This is the same idea OpenRAVE's IKFast had — generate per-arm specialised IK code at design time, run pure numeric at deployment — but without IKFast's brittleness on non-Pieper geometries.

Cython hot loops cover the leaf primitives (Rodrigues rotations, POE forward kinematics, SP1–SP6 subproblems); the rest is pure Python so it stays inspectable.

## Tuning knobs

### `TolerancePolicy` — six thresholds, one object

`solve()` accepts an optional `policy=` kwarg. The default `ssik.DEFAULT_TOLERANCE_POLICY` works for every shipped fixture; reach for a custom policy when a real arm's URDF has structural near-degeneracies (axes that *almost* but not exactly meet) or when you want tighter / looser FK closure than the defaults provide.

```python
from ssik import TolerancePolicy, DEFAULT_TOLERANCE_POLICY

policy = TolerancePolicy(
    axis_parallel=1e-8,         # ||a × b||: when two axes are "parallel"
    axis_intersect=1e-8,        # perpendicular distance: when two lines "meet"
    subproblem_feasibility=1e-9,# is_ls boundary inside SP1-SP6
    subproblem_numerical=1e-5,  # FK-closure filter on algebraic candidates
    subproblem_degeneracy=1e-12,# rank-drop threshold; below this, return []
    subproblem_dedup=1e-3,      # angle-space tolerance for collapsing duplicates
)
sols = my_arm_ik.solve(T_target, policy=policy)
```

The fields are named for *why* they exist so log messages can say `"SP6 sign branch rejected: closure 1.2e-4 > subproblem_numerical 1e-5"` instead of citing magic numbers. Tighten `subproblem_numerical` when you want stricter FK closure (at the cost of dropping more borderline branches); loosen `axis_parallel` / `axis_intersect` when an Xacro-derived URDF has axes mis-aligned by a few microradians and the dispatcher refuses to classify it.

### `ssik.postprocess` — composable filters

`solve()` returns the geometric IK set. For application-specific filtering, four helpers in `ssik.postprocess` compose into the typical "robot-aware IK" pipeline:

```python
from ssik.postprocess import (
    respect_limits, wrap_to_limits, nearest_to_seed, take_first,
)

sols = my_arm_ik.solve(T_target, respect_limits=False)  # raw geometric set
sols = wrap_to_limits(sols, my_arm_ik._KB)              # try q ± 2π to bring in
sols = respect_limits(sols, my_arm_ik._KB)              # drop anything still outside
sols = nearest_to_seed(sols, q_current)                 # sort by wrap-to-pi distance
sols = take_first(sols, k=4)                            # top-k after ranking
```

By default `solve()` already runs `wrap_to_limits` + `respect_limits` (see [Trajectory tracking](#trajectory-tracking--ik-based-teleop) above); the standalone helpers exist for callers who want a different order, a different metric, or to add their own filters (collision-aware filtering, dexterity scoring, custom seed metrics) between the layers.

Out of scope: collision filtering (use FCL or similar at the application layer) and continuous-trajectory smoothness (typically a separate planner concern). The four shipped filters cover the standard "single-pose, robot-aware IK" case.

## Bulletproof discipline

Every solver lands with: N-way cross-solver agreement on shared fixtures, FK closure ≤ 1e-10 on every retained IK, 500+ Hypothesis-fuzzed random poses per fixture, and an explicit speed bench that has to clear a regression gate. The current suite has **1300+ tests across 11 fixture arms**. Negative-result spikes (a Cython estimate that misses by 2-5×, a codegen-bake on a part that's 0.3% of runtime) are published as closed issues with profile data so the next contributor doesn't repeat the path.

## Install

```bash
pip install ssik              # core + analytical solvers (everything you need at deployment)
pip install ssik[urdf]        # adds urchin + sympy for `ssik build` and the dev/exploration path
```

Python 3.11+. Wheels for Linux x86_64 and macOS arm64.

## Development & exploration: `Manipulator.from_urdf`

For one-off experiments or fuzzing during solver development, the runtime classifier is also exposed as a Python class. It parses a URDF, dispatches a solver at construction time, and exposes the same `solve()` / `fk()` API as the artifact. Every fresh process re-runs URDF parsing, topology classification, and (for non-Pieper sub-chains) first-call sympy preprocessing — so it is strictly slower than the build-artifact path in production and requires `urchin` + `sympy` on the runtime path:

```python
import ssik

arm = ssik.Manipulator.from_urdf("my_arm.urdf", base="base_link", ee="tool0")
sols = arm.solve(T_target, max_solutions=1, q_seed=q_current)
```

Once the dispatch is settled, switch to `ssik build my_arm.urdf` and import the artifact. Contributors extending ssik's test suite (vs deploying for their own arm) use `ssik add-arm`; see [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-arm-fixture).

## Documentation

- [docs/arm_coverage.md](docs/arm_coverage.md) — per-arm fixture tables, tested speeds, source URDFs
- [docs/architecture.md](docs/architecture.md) — solver tier catalog, dispatch flow, build artifact internals, algorithmic lineage
- [CONTRIBUTING.md](CONTRIBUTING.md) — repo layout, dev setup, testing discipline

## Related libraries

ssik does not compete with these on the arms they cover. Pick the right tool for your geometry.

- [**EAIK**](https://github.com/OstermD/EAIK) (Ostermeier 2024) — Python wrapper around C++ subproblem-decomposition solvers. Analytical, returns all branches on Pieper-class 6R and canonical SRS 7R (with a manual joint lock). Refuses arms outside its recognised kinematic families. Directly benchmarked in the table above.
- [**IK-Geo**](https://github.com/rpiRobotics/ik-geo) (Elias–Wen 2022/2025) — the reference C++/Rust implementation of subproblem decomposition. Same coverage profile as EAIK. Has Python bindings (`ik-geo` on PyPI); currently pins `pyo3==0.20.3` so the wheel is incompatible with Python 3.13 — track upstream for an update.
- [**IKFast**](http://openrave.org/docs/latest_stable/openravepy/ikfast/) (Diankov 2010, part of OpenRAVE) — the original analytical-IK codegen tool. Symbolic preprocessing in sympy → per-arm C++. Works well on the kinematic families it was tuned for (Pieper-class 6R, spherical-wrist 7R via joint lock); the symbolic pipeline fails on modern sympy for non-Pieper geometries (`mpmath.polyroots` NoConvergence, `Matrix.inv` / `Matrix.det` stalls). LGPL-licensed.
- [**MINK**](https://github.com/kevinzakka/mink) (Zakka) — Mujoco-native numerical IK via damped least-squares. Iterative, takes a seed, converges to a single configuration. Handles any kinematic geometry but returns one IK, not all branches, and FK closure is proportional to the convergence tolerance (typically 1e-3 to 1e-6 rather than machine precision).
- [**TracIK**](https://traclabs.com/projects/trac-ik/) (Beeson & Ames 2015) — combined SQP / pseudoinverse Jacobian solver; the ROS Industrial default numerical IK. URDF-native. Same one-branch-per-seed semantics as MINK. The maintained Python binding (`pytracik`) ships a broken arm64 wheel; the ROS-native binding works fine inside ROS.
- [**KDL-LMA**](https://github.com/orocos/orocos_kinematics_dynamics) — OROCOS KDL's Levenberg-Marquardt numerical IK. Older and less robust than TracIK or MINK on the same problem class.

## License

[BSD-3-Clause](LICENSE). The library incorporates clean-room reimplementations of algorithms from BSD-3-licensed IK-Geo (Elias–Wen 2022/2025) and from the academic publications of Raghavan–Roth (1990), Manocha–Canny (1994), Singh–Kreutz (1989), and Husty–Pfurner (2007). Algorithmic lineage is documented in module docstrings.

## Citation

```bibtex
@software{ssik,
  author = {Srinivasa, Siddhartha},
  title  = {ssik: analytical inverse kinematics for 6R and 7R revolute arms},
  url    = {https://github.com/personalrobotics/ssik},
  year   = {2026},
}
```
