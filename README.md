# ssik

Analytical inverse kinematics for 6R and 7R revolute robot arms. Returns **every IK branch** at machine-precision FK closure. The canonical workflow is `ssik build` once per arm, then `import <arm>_ik` everywhere — a self-contained Python module with the per-arm KinBody constants, the dispatched solver, and any cached symbolic preprocessing already inside it.

## Quickstart — use a prebuilt arm

```python
import sys; sys.path.insert(0, "prebuilt")
import ur5_ik                # or puma560_ik, iiwa14_ik, gen3_ik, jaco2_ik,
                             # franka_panda_ik, rizon4_ik, kassow_kr810_ik
import numpy as np

T_target = np.eye(4); T_target[:3, 3] = [0.5, 0.1, 0.3]
sols, is_ls = ur5_ik.solve(T_target)             # all 8 IK branches
sols, _    = ur5_ik.solve(T_target, max_solutions=1, q_seed=q_prev)
```

That's the whole API. `sols` is a list of `Solution` objects (each with `.q`, `.fk_residual`, `.solver_name`); `is_ls=True` signals no candidate FK-closed within the tolerance policy.

The `prebuilt/` directory ships ready-to-use artifacts for **UR5, Puma 560, KUKA iiwa14, Kinova Gen3, Kinova JACO 2, Franka Panda, Flexiv Rizon 4, and Kassow KR810**. See [`prebuilt/README.md`](prebuilt/README.md) for the full list and what's inside each one.

## Your own arm

```bash
ssik build my_arm.urdf --base base_link --ee tool0
# → my_arm_ik.py (one-time, ~1-7 minutes depending on solver tier)
```

```python
import my_arm_ik
sols, is_ls = my_arm_ik.solve(T_target)             # same signature as the prebuilt arms
```

The artifact:

- bakes the per-arm KinBody constants as numpy literals,
- bakes the dispatched solver choice (no runtime classification),
- for non-Pieper sub-chains, embeds the cached Raghavan–Roth symbolic derivations as base85-encoded zlib-compressed pickle blobs.

Module-init takes ~5 seconds (deserialise + re-`lambdify` the cached blobs). Every subsequent `solve()` call runs at warm-cache speed. **No URDF parsing, no `urchin` dependency, no `sympy` on the deployed import path, no cold-cache symbolic preprocessing at runtime.** A robot stack that imports `my_arm_ik.py` carries no algorithmic complexity beyond what `ssik build` already resolved.

This is the same model OpenRAVE's IKFast had — generate per-arm specialised IK code at design time, run pure numeric at deployment — but without IKFast's brittleness on non-Pieper geometries.

## Returns all solutions, not one

A single 6-DOF target pose admits up to **16 analytical IK branches** (8 typical for a Pieper-class arm: 4 shoulder × 2 elbow, with the wrist deterministic). For a 7R redundant arm the IK is a 1-parameter family; ssik discretises it into 32–256 branches per pose depending on the swivel-sample count. Every returned `Solution` carries `q`, the per-IK FK residual, and which solver branch produced it.

This is the structural distinction from numerical IK libraries (MINK, TracIK, KDL-LMA): those take a seed, run damped least-squares to a single converged configuration, and stop. Branch enumeration matters for motion planning (try every branch, pick the one with best clearance), for dexterity analysis (the manipulability ellipsoid is per-branch), and for trajectory continuation across kinematic singularities.

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

The arms ssik exists for are the ones where EAIK and IK-Geo refuse: non-Pieper 6R chains whose geometry deliberately violates Pieper's three-axes-intersect condition (the JACO 2's 60° non-orthogonal twists are the canonical example), and 7R arms whose URDF axes don't quite meet at the canonical SRS shoulder/wrist points. Numeric solvers handle these but at 100× the cost, without redundancy enumeration, and with FK error proportional to the convergence tolerance rather than machine precision.

The algorithmic ingredients are not novel — Raghavan–Roth (1990), Manocha–Canny (1994), Singh–Kreutz (1989), Husty–Pfurner (2007). What's new is making the textbook pipelines survive on real ill-conditioned arms (AE-3 leftvar selection on JACO 2 drops `cond(m_quad)` from 3.75 × 10^16 to 127), composing them with a uniform dispatch layer, and packaging the whole thing as a deployable artifact.

## Speed (typical median, Apple M3 single-thread, post-artifact-load)

| Arm | Full sweep (all branches) | `max_solutions=1` |
|-----|:---:|:---:|
| UR5 (6R, three-parallel) | 1.6 ms (8 IKs) | 0.2 ms |
| Puma 560 (6R, spherical wrist) | 1.2 ms (8 IKs) | 0.2 ms |
| JACO 2 (non-Pieper 6R) | 5 ms (8 IKs) | 0.6 ms |
| KUKA iiwa14 (SRS 7R) | 4.3 ms (128 IKs) | 0.5 ms |
| Kinova Gen3 (approximate-SRS 7R) | 56 ms (40 IKs, machine-precision LM polish) | 5 ms |
| Flexiv Rizon 4 (non-SRS 7R) | 17 ms (45 IKs) | 1.5 ms |
| Kassow KR810 (non-SRS 7R) | 18 ms (30 IKs) | 1.6 ms |
| Franka Panda (anthropomorphic 7R) | 42 ms (64 IKs) | 2.4 ms |

Cython hot loops cover the leaf primitives (Rodrigues rotations, POE forward kinematics, SP1–SP6 subproblems); the rest is pure Python so it stays inspectable.

## Install

```bash
pip install ssik              # core + analytical solvers
pip install ssik[urdf]        # adds urchin, needed for `ssik build` and Manipulator.from_urdf
```

Python 3.11+. Wheels for Linux x86_64 and macOS arm64.

## Interactive exploration (without building)

For first-day experimentation, prototyping, or running tests against many arms, the `Manipulator` class is also available. It loads a URDF directly, classifies the topology at construction time, and dispatches to the right solver — same `solve()` semantics, but with the URDF parsing + classification cost paid on every fresh process:

```python
import ssik
arm = ssik.Manipulator.from_urdf("my_arm.urdf", base="base_link", ee="tool0")
T = arm.fk([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
sols, is_ls = arm.ik(T)
```

This is **fine for tests and exploration**. For production it's almost always worse than building an artifact: every fresh process re-classifies the URDF, re-extracts joint axes, and (for non-Pieper sub-chains) re-runs the sympy preprocessing on the first IK call. `ssik build` pays each of those costs once.

## Adding a new arm

```bash
ssik add-arm my_arm.urdf --base base_link --ee flange --name my_arm
# → tests/fixtures/my_arm.urdf and tests/test_my_arm.py with FK-closure assertions
uv run pytest tests/test_my_arm.py -v
```

The generated test scaffold checks dispatcher routing and FK closure on hand-picked + Hypothesis-fuzzed reachable poses. Catches regressions before they land.

## Bulletproof testing discipline

Every solver lands with: N-way cross-solver agreement on shared fixtures, FK closure ≤ 1e-10 on every retained IK, 500+ Hypothesis-fuzzed random poses per fixture, and an explicit speed bench that has to clear a regression gate. The current suite has **1300+ tests across 11 fixture arms**. Negative-result spikes (a Cython estimate that misses by 2-5×, a codegen-bake on a part that's 0.3% of runtime) are published as closed issues with profile data so the next contributor doesn't repeat the path.

## Documentation

- [`prebuilt/README.md`](prebuilt/README.md) — full list of prebuilt artifacts, sizes, build times
- [`examples/`](examples/) — runnable scripts: UR5 quickstart, JACO 2 (non-Pieper), Gen3 (approximate-SRS), comparison vs EAIK/MINK
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
  url    = {https://github.com/personalrobotics/ssik},
  year   = {2026},
}
```
