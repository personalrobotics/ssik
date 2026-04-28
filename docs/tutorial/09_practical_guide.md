# 9. Practical guide

!!! warning "Scaffolding"
    Outline below; prose to be filled in.

## What this chapter covers

How to use ssik day-to-day: install, build a `KinBody`, pick a solver, interpret the returned `Solution`, debug failures.

## Install

```
pip install ssik          # core
pip install ssik[urdf]    # + URDF loader (urchin)
```

(Pre-alpha; package name reservation in progress as of writing. See [#83](https://github.com/siddhss5/ikfastpy/issues/83).)

## Build a `KinBody`

From a URDF:

```python
from ssik._urdf import load_urdf_kinbody_normalized

kb = load_urdf_kinbody_normalized("robots/ur5.urdf", "base_link", "ee_link")
```

From an MJCF (transcribe joint frames; see [`tests/fixtures/jaco2.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/fixtures/jaco2.py) for the pattern):

```python
from ssik._kinbody import build_kinbody
from fixtures.jaco2 import jaco2_specs  # transcribed once from MJCF

kb = build_kinbody(jaco2_specs())
```

From scratch: build a list of [`JointSpec`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/_kinbody.py) and pass to [`build_kinbody`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/_kinbody.py).

## Pick a solver

For 6R arms, the dispatcher picks the right tier per kb at registration time:

```python
from ssik.solvers.ikgeo import general_6r

solutions, is_ls = general_6r.solve(kb, T_target)
for sol in solutions:
    print(sol.q, sol.fk_residual, sol.refinement_used)
```

(The unified `Manipulator.from_urdf(...).ik(T)` API of the rewrite plan is in progress; until it lands, importing the per-tier solver directly is the way.)

For 7R arms with one redundant joint (Franka, KUKA iiwa, Flexiv Rizon):

```python
from ssik.solvers.jointlock import seven_r

solutions, is_ls = seven_r.solve(kb, T_target, lock_samples=16)
```

`lock_samples=16` sweeps the chosen redundant joint over 16 lock values; or pass an explicit list `lock_samples=[0.0, 0.5, 1.1, -0.5]` to control which lock values get sampled.

## Read the returned `Solution`

```python
@dataclass(frozen=True)
class Solution:
    q: NDArray[np.float64]
    fk_residual: float
    refinement_used: Literal["none", "lm"]
    refinement_iters: int
    branch_id: int | None = None
    solver_name: str = ""
```

- **`fk_residual`** — `||FK(q) - T_target||_F` at return time. Compare against your application's tolerance; the solver's own `fk_atol` was a *filter*, not a contract on the value.
- **`refinement_used`** — `"none"` if the solution came directly from algebra; `"lm"` if Newton polished it. Surfaces the algebraic-vs-iterative line.
- **`refinement_iters`** — how many Newton iterations consumed (0 if `refinement_used == "none"`).
- **`branch_id`** — IK branch index where applicable (0..15 for the 16-root RR route, 0..7 for spherical+parallel-shoulder, etc.). Stable across calls for the same target pose; useful for branch-tracking across a trajectory.
- **`solver_name`** — dotted module path. Useful when results pass through a dispatcher that routes across multiple solver candidates.

## Debug `is_ls=True`

`is_ls=True` means **no candidate survived FK validation**. Common causes:

1. **Target unreachable.** The arm physically can't get to `T_target`. Check your target pose against the workspace.
2. **Singular pose.** The arm *can* reach `T_target` but does so at a singularity (wrist pitch zero, elbow extended). Some branches collapse; the solver may return fewer solutions but they should still be exact. If `is_ls=True` at a singularity, the geometry is on the edge of the singular set.
3. **Tier-2 RR with `allow_refinement=False`** on an ill-conditioned pose. The algebraic candidates miss `fk_atol` and get dropped. Try `allow_refinement=True` and inspect the returned `refinement_iters` — high iteration counts (>5) suggest the solver is struggling and the pose is genuinely difficult.
4. **A topology-precondition mismatch.** You called `spherical_two_parallel.solve` on an arm that isn't actually spherical-wrist-with-parallel-shoulder. Tier-0 solvers raise `ValueError` on topology mismatch up front; tier-2 solvers accept any 6R but may struggle if the arm is at a degenerate pose for that algorithm.

## Performance numbers (current at writing)

Real JACO 2 j2n6s200 (60° twists, the hard case), warm cache, 100 random poses:

| metric | value |
|---|---|
| min | 0.83 ms |
| **median** | **2.25 ms** |
| mean | 3.49 ms |
| p95 | 10.24 ms |
| max | 29.61 ms |
| solutions / pose | median 6, range 1–12 |
| FK error | median 3.7e-13, max 9.6e-08 |
| failures (`is_ls=True`) | 0 / 100 |

Cold-cache cost (first IK on a new arm): ~150-300 s of one-time sympy preprocessing for the AE-3 leftvar selection. Cached on disk after that.

For Pieper-class arms (UR5, Puma), the tier-0 solvers run in ~50–200 µs warm-cache, comparable to EAIK.

See [#86](https://github.com/siddhss5/ikfastpy/issues/86) for ongoing speed work; the Phase M codegen / Rust runtime is the next order-of-magnitude lift.

## Reading test output

When a CI job shows xpassed/xfailed counts on test_solve_all_ik_recovers_q_star_and_alternatives, that's the platform-sensitive [#82](https://github.com/siddhss5/ikfastpy/issues/82) coverage gap on MC Table I (a synthetic benchmark). Different LAPACK backends pick different IK branches; the test passes either way (xfail with `strict=False`) but the counts surface the actual recovery rate per platform.
