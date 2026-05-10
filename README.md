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

A single 6-DOF target pose admits up to **16 analytical IK branches** (8 typical for a Pieper-class arm: 4 shoulder × 2 elbow, with the wrist deterministic). For a 7R redundant arm, the IK is a 1-parameter family; ssik discretises it into 32–256 branches per pose depending on the swivel-sample count. Every returned `Solution` carries `q`, the per-IK FK residual, and which solver branch produced it.

This is what numerical IK libraries (MINK, TracIK, KDL-LMA) cannot give you: they take a seed, run damped least-squares to a single converged configuration, and stop. Branch enumeration matters for motion planning (try every branch, pick the one with best clearance), for dexterity analysis (the manipulability ellipsoid is per-branch), and for trajectory continuation across kinematic singularities.

## Coverage relative to existing libraries

|   | EAIK | IK-Geo | MINK / TracIK (numeric) | ssik |
|---|:---:|:---:|:---:|:---:|
| Pieper-class 6R (UR5, Puma 560, KUKA KR) | analytical | analytical | iterative | analytical |
| **Non-Pieper 6R** (JACO 2 j2n6s200, Agilex Piper) | refuses | refuses | iterative, single solution | analytical (`ikgeo.general_6r`) |
| SRS-class 7R (KUKA iiwa14) | analytical | analytical | iterative | analytical (`seven_r.srs`) |
| **Approximate-SRS 7R** (Kinova Gen3, ~12 mm offset) | analytical against simplified DH (mm IK error) | similar | iterative | analytical against original URDF FK |
| **Non-SRS 7R** (Flexiv Rizon, Kassow KR810) | analytical against simplified DH (cm IK error) | refuses | iterative | analytical (`jointlock.seven_r` + cached Raghavan–Roth) |
| Anthropomorphic 7R (Franka, FR3, xArm7) | analytical | analytical | iterative | analytical (joint-locking) |
| FK closure on returned IK | varies | machine precision | 1e-3 to 1e-6 | ≤ 1e-10 every retained IK |
| Returns all branches | yes (where supported) | yes (where supported) | no | yes |

The arms ssik exists for are the ones where EAIK and IK-Geo refuse: non-Pieper 6R chains where the geometry deliberately violates Pieper's three-axes-intersect condition (the JACO 2's 60° non-orthogonal twists are the canonical example), and 7R arms whose URDF axes don't quite meet at the canonical SRS shoulder/wrist points. Numeric solvers handle these but at 100× the cost, without redundancy enumeration, and with FK error proportional to the convergence tolerance rather than machine precision.

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

## Speed (typical median, Apple M3 single-thread, pure Python + numpy)

| Arm | Full sweep (all branches) | `max_solutions=1` (single closest branch) |
|-----|:---:|:---:|
| UR5 (6R, three-parallel) | 1.6 ms (8 IKs) | 0.2 ms |
| Puma 560 (6R, spherical wrist) | 1.2 ms (8 IKs) | 0.2 ms |
| JACO 2 (non-Pieper 6R) | 5 ms (8 IKs) | 0.6 ms |
| iiwa14 (SRS 7R) | 4.3 ms (128 IKs) | 0.5 ms |
| Gen3 (approximate-SRS 7R) | 56 ms (40 IKs, machine-precision LM polish) | 5 ms |
| Rizon 4 (non-SRS 7R, built artifact) | 17 ms (45 IKs) | 1.5 ms |
| Kassow KR810 (non-SRS 7R, built artifact) | 18 ms (30 IKs) | 1.6 ms |
| Franka Panda (anthropomorphic 7R) | 42 ms (64 IKs) | 2.4 ms |

Cython hot loops cover the leaf primitives (Rodrigues rotations, POE forward kinematics, SP1–SP6 subproblems); the rest is pure Python so it stays inspectable. Numbers above are post-`ssik build` artifact load.

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

## License

[BSD-3-Clause](LICENSE). The library incorporates clean-room reimplementations of algorithms from BSD-3-licensed [IK-Geo](https://github.com/rpiRobotics/ik-geo) (Elias–Wen 2022/2025) and from the academic publications of Raghavan–Roth (1990), Manocha–Canny (1994), Singh–Kreutz (1989), and Husty–Pfurner (2007). Algorithmic lineage is documented in module docstrings.

## Citation

```bibtex
@software{ssik,
  author = {Srinivasa, Siddhartha},
  title  = {ssik: analytical inverse kinematics for 6R and 7R revolute arms},
  url    = {https://github.com/siddhss5/ikfastpy},
  year   = {2026},
}
```
