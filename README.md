# ssik

Analytical inverse kinematics for 6R and 7R revolute robot arms. Returns **every IK branch** at machine-precision FK closure, dispatches the right solver automatically, and ships a per-arm build artifact that contains the full IK pipeline as a self-contained Python module.

```python
import ssik

arm = ssik.Manipulator.from_urdf("ur5.urdf", base="base_link", ee="ee_link")
T = arm.fk([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
sols, is_ls = arm.ik(T)                                   # all 8 IK branches
sols, _    = arm.ik(T, max_solutions=1, q_seed=q_prev)    # closest branch only
```

## Returns all solutions, not one

A single 6-DOF target pose admits up to **16 analytical IK branches** (8 typical for a Pieper-class arm: 4 shoulder × 2 elbow, with the wrist deterministic). For a 7R redundant arm the IK is a 1-parameter family; ssik discretises it into 32–256 branches per pose depending on the swivel-sample count. Every returned `Solution` carries `q`, the per-IK FK residual, and which solver branch produced it.

Numerical IK libraries take a seed, run damped least-squares to a single converged configuration, and stop. Branch enumeration matters for motion planning (try every branch, pick the one with best clearance), for dexterity analysis (the manipulability ellipsoid is per-branch), and for trajectory continuation across kinematic singularities.

## Measured comparison vs EAIK

EAIK (Ostermeier 2024) is the canonical Python wrapper around C++ subproblem-decomposition solvers. It is analytical on the kinematic families it recognises and refuses everything else. The numbers below come from [`examples/04_compare_vs_eaik.py`](examples/04_compare_vs_eaik.py) over 100 random reachable poses per arm, Apple M3 single-thread, mean ± 95% CI via bootstrap (1000 resamples). FK residual is the Frobenius norm `‖FK(q) − T‖` against the original URDF / spec FK.

| Arm (class) | EAIK | ssik |
|---|---|---|
| UR5 (Pieper 6R, three-parallel) | 5 ± 0 µs / FK 2e-15 / 4 sols | 549 ± 14 µs / FK 2e-9 / 4 sols |
| Puma 560 (Pieper 6R, spherical wrist) | 6 ± 0 µs / FK 3e-14 / 8 sols | 233 ± 5 µs / FK 2e-14 / 8 sols |
| JACO 2 (**non-Pieper 6R**) | **refuses** ("6R-Unknown Kinematic Class") | 1.04 ± 0.04 ms / FK 5e-6 / 8 sols |
| iiwa14 (SRS 7R) | **refuses** ("only 1-6R robots are solvable") | 4.57 ± 0.03 ms / FK 4e-13 / 128 sols |
| Gen3 (**approximate-SRS 7R**, 12 mm offset) | **refuses** ("only 1-6R") | 41.48 ± 1.18 ms / FK 1e-12 / 47 sols |
| Franka Panda (anthropomorphic 7R) | **refuses** ("only 1-6R") | 28.42 ± 2.65 ms / FK 1e-6 / 64 sols |
| Rizon 4 (**non-SRS 7R**) | **refuses** ("only 1-6R") | 33.18 ± 8.58 ms / FK 4e-9 / 42 sols |
| Kassow KR810 (**non-SRS 7R**) | **refuses** ("only 1-6R") | 27.03 ± 10.40 ms / FK 7e-8 / 24 sols |

EAIK is ~100× faster than ssik on Pieper-class 6R — that is its native sweet spot, and ssik does not try to compete there. The interesting cells are the **refuses** ones: non-Pieper 6R (JACO 2) and every 7R arm. Those are the geometries ssik exists for. The "refuses (...)" strings are EAIK's actual error messages, captured verbatim by the bench harness. A numerical-IK comparison (MINK) is tracked separately in [#236](https://github.com/personalrobotics/ssik/issues/236).

ssik FK residuals above are the algebraic candidates returned by `solve()` with default tolerance policy. Passing `allow_refinement=True` runs an opt-in Levenberg–Marquardt polish per candidate and tightens the residual to machine precision (~1e-14) at a few hundred microseconds per branch.

The algorithmic ingredients are not novel — Raghavan–Roth (1990), Manocha–Canny (1994), Singh–Kreutz (1989), Husty–Pfurner (2007). What's new is making the textbook pipelines survive on real ill-conditioned arms (AE-3 leftvar selection on JACO 2 drops `cond(m_quad)` from 3.75 × 10^16 to 127), composing them with a uniform dispatch layer, and packaging the whole thing as a deployable artifact.

## The build artifact is the deployment story

ssik supports two execution paths. They share the same public API but solve different problems.

**Development / interactive** — `ssik.Manipulator.from_urdf(...)`:

```python
arm = ssik.Manipulator.from_urdf("my_arm.urdf", base="base_link", ee="ee_link")
sols, _ = arm.ik(T)
```

The runtime parses the URDF, classifies the topology, dispatches a solver, and runs IK. First-call symbolic preprocessing (when the dispatch lands on the universal Raghavan-Roth fallback) is paid lazily — fine for tests and one-off use. Requires `urchin` for URDF parsing and `sympy` for any symbolic preprocessing on first call.

**Production / embedded** — `ssik build` artifact:

```bash
ssik build my_arm.urdf --base base_link --ee tool0
# → my_arm_ik.py    (~1-5 min build, all symbolic preprocessing baked in)
```

```python
import my_arm_ik
sols, is_ls = my_arm_ik.solve(T_target)         # same signature as Manipulator.ik()
```

The artifact is a single `.py` file that contains:
- the per-arm KinBody constants inlined as numpy literals,
- the dispatch choice baked at build time (no runtime classification),
- for non-Pieper sub-chains, the cached Raghavan–Roth symbolic derivations as base85-encoded zlib-compressed pickle blobs.

Module-init takes ~5 seconds (deserialise + re-`lambdify` the cached blobs); every subsequent `solve()` call hits warm-cache speed. **No URDF parsing, no `urchin` dependency, no cold-cache symbolic preprocessing at runtime, no sympy on the import path of the deployed artifact.** A robot stack that imports `my_arm_ik.py` carries no algorithmic complexity beyond what the build pipeline already resolved.

This is the same idea OpenRAVE's IKFast had — generate per-arm specialised IK code at design time, run pure numeric at deployment — but without IKFast's brittleness on non-Pieper geometries.

Cython hot loops cover the leaf primitives (Rodrigues rotations, POE forward kinematics, SP1–SP6 subproblems); the rest is pure Python so it stays inspectable. The speed numbers in the comparison table are post-artifact-load.

## Bulletproof discipline

Every solver lands with: N-way cross-solver agreement on shared fixtures, FK closure ≤ 1e-10 on every retained IK, 500+ Hypothesis-fuzzed random poses per fixture, and an explicit speed bench that has to clear a regression gate. The current suite has **1284 tests across 11 fixture arms**. Negative-result spikes (a Cython estimate that misses by 2-5×, a codegen-bake on a part that's 0.3% of runtime) are published as closed issues with profile data so the next contributor doesn't repeat the path.

## Install

```bash
pip install ssik              # core + analytical solvers
pip install ssik[urdf]        # adds URDF loader (urchin) for from_urdf
```

Python 3.11+. Wheels for Linux x86_64 and macOS arm64.

## Onboarding a new arm

```bash
ssik add-arm my_arm.urdf --base base_link --ee flange --name my_arm
# → tests/fixtures/my_arm.urdf and tests/test_my_arm.py with FK-closure assertions
uv run pytest tests/test_my_arm.py -v
```

The generated test scaffold checks dispatcher routing and FK closure on hand-picked + Hypothesis-fuzzed reachable poses. Catches regressions before they land.

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
