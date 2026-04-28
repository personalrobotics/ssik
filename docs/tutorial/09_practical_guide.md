# 9. Practical guide

This chapter is the user-facing entry point. It covers installation, building a `KinBody`, picking a solver, interpreting the returned `Solution`, debugging failures, and the performance numbers you should actually expect on the hardware in front of you.

## Install

ssik is pre-alpha; package-name reservation in progress as of writing (see [#83](https://github.com/siddhss5/ikfastpy/issues/83)). Until the PyPI release lands:

```
pip install git+https://github.com/siddhss5/ikfastpy.git
```

For URDF loading you also need the `urdf` extra:

```
pip install "git+https://github.com/siddhss5/ikfastpy.git#egg=ssik[urdf]"
```

This pulls in [urchin](https://github.com/fishbotics/urchin) for URDF parsing. Without the extra, the `ssik._urdf` module raises `ImportError`; the rest of the library works fine.

Runtime dependencies are kept narrow: numpy ≥ 2.0 and sympy ≥ 1.10 (sympy is used at module-import time for offline per-arm symbolic preprocessing; not on the hot IK path). scipy is *not* required — the few places that benefit from `scipy.linalg.eig` lazy-import it inside an `except ImportError` branch and degrade gracefully.

## Build a `KinBody`

ssik's canonical input is a POE-normalized `KinBody`. There are three ways to get one.

### From a URDF

Most users start here. URDF is the dominant interchange format in ROS / MoveIt / mink ecosystems.

```python
from ssik._urdf import load_urdf_kinbody_normalized

kb = load_urdf_kinbody_normalized("robots/ur5.urdf", "base_link", "ee_link")
```

`load_urdf_kinbody_normalized` reads the URDF, walks the chain from `base_link` to `ee_link`, normalizes per-joint origins into POE form (`T_left @ R_axis(joint.axis, q) @ T_right` per joint), and returns the assembled `KinBody`. The function handles non-orthogonal RPY in URDF `<origin>` tags, mid-chain fixed joints (their offsets fold into the next non-fixed joint), and continuous joints (treated as revolute without limits).

### From an MJCF

MuJoCo MJCF is increasingly common in modern simulation environments (`isaac-sim`, `mujoco`, `geodude`). ssik doesn't currently parse MJCF directly (tracked as a follow-up); the recommended pattern is to **transcribe the joint frames once by hand** into a `JointSpec` list. See [`tests/fixtures/jaco2.py`](https://github.com/siddhss5/ikfastpy/blob/main/tests/fixtures/jaco2.py) for the canonical example — six `<body pos quat>` extractions, one tool offset, one `JointSpec` per joint:

```python
from ssik._kinbody import build_kinbody
from fixtures.jaco2 import jaco2_specs  # transcribed once from MJCF

kb = build_kinbody(jaco2_specs())
```

The transcription is one-time per arm. Once written, the fixture is a Python module that imports cheaply. Real-fixture transcription was the last thing that needed to happen before ssik could close the JACO 2 loop end-to-end (see [Chapter 7](07_kinbody_bridge.md) for the convention bug it surfaced).

### From scratch

For prototyping kinematics or testing the solver against a synthetic arm, build a list of [`JointSpec`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/_kinbody.py) directly:

```python
from ssik._kinbody import JointSpec, build_kinbody
import numpy as np

specs = [
    JointSpec(
        parent_link_T=np.eye(4),
        axis=np.array([0., 0., 1.]),
        joint_type="revolute",
        child_link_T=np.eye(4),
        name="joint_0",
    ),
    # ... five more JointSpec entries
]
kb = build_kinbody(specs)
```

Each `JointSpec` carries the per-joint `T_left` (parent frame to joint frame), the joint axis in the joint frame, the joint type (`"revolute"` or `"prismatic"` — though prismatic is currently out-of-scope, see Chapter 10 roadmap), and an optional `T_right` (joint frame to child link frame; defaults to identity).

## Pick a solver

ssik ships ten solver modules. The dispatcher (in progress; see Chapter 10) will pick the right one for your kb at registration time. Until that lands, the recommended pattern is to import the per-tier solver directly.

### 6R arms

For most arms, start with the tier-2 numeric Raghavan–Roth solver. It works on every 6R chain — Pieper-class or non-Pieper — at machine precision. It's slower than the tier-0 specialisations but it never refuses a topology.

```python
from ssik.solvers.ikgeo import general_6r

solutions, is_ls = general_6r.solve(kb, T_target)
for sol in solutions:
    print(f"q = {sol.q}, fk_residual = {sol.fk_residual:.2e}, "
          f"refinement = {sol.refinement_used}")
```

If your arm is Pieper-class — three intersecting wrist axes or three parallel shoulder axes — the tier-0 solvers are 30× faster:

```python
# Three-parallel arms (UR5, UR10):
from ssik.solvers.ikgeo import three_parallel
solutions, is_ls = three_parallel.solve(kb, T_target)

# Spherical wrist + parallel shoulder (Puma 560, Fanuc, KUKA KR):
from ssik.solvers.ikgeo import spherical_two_parallel
solutions, is_ls = spherical_two_parallel.solve(kb, T_target)

# Spherical wrist + intersecting shoulder (Puma 560 alternate, compact arms):
from ssik.solvers.ikgeo import spherical_two_intersecting
solutions, is_ls = spherical_two_intersecting.solve(kb, T_target)

# Generic spherical wrist with no shoulder specialisation:
from ssik.solvers.ikgeo import spherical
solutions, is_ls = spherical.solve(kb, T_target)
```

These tier-0 solvers raise `ValueError` at the topology gate if your kb doesn't match their preconditions. That's the right signal — try a different solver.

### 7R arms with one redundant joint

For Franka Panda, KUKA iiwa, Flexiv Rizon — anything with one redundant degree of freedom — the joint-locking wrapper is the generic solution:

```python
from ssik.solvers.jointlock import seven_r

solutions, is_ls = seven_r.solve(kb, T_target, lock_samples=16)
```

The `lock_samples=16` parameter sweeps the auto-selected redundant joint over 16 evenly-spaced lock values; for each, the wrapper dispatches the resulting 6R sub-chain to the best-matching `ikgeo.*` solver. You can also pass an explicit list of lock values:

```python
solutions, is_ls = seven_r.solve(kb, T_target, lock_samples=[0.0, 0.5, 1.1, -0.5])
```

This is useful when you want to track a specific configuration's redundancy slice (e.g. "lock joint 3 at zero, find every IK there"). The joint-locking convention picks one slice of the 2D redundancy manifold rather than parametrising it; if you need exact redundancy parametrisation you'll want to wait for the specialist 7R solvers (`specialist.geofik` for Franka, `specialist.stereo_sew` for iiwa-class arms) on the roadmap.

## Read the returned `Solution`

Every solver returns `tuple[list[Solution], bool]`. The `Solution` dataclass:

```python
@dataclass(frozen=True)
class Solution:
    q: NDArray[np.float64]                       # joint vector
    fk_residual: float                            # ||FK(q) - T_target||_F
    refinement_used: Literal["none", "lm"]        # "none" or "lm"
    refinement_iters: int                         # iter count if "lm" fired
    branch_id: int | None = None                  # IK branch index
    solver_name: str = ""                         # e.g. "ikgeo.general_6r"
```

The semantic guarantees:

- **`q`**: the joint vector. Always in the user's POE frame (matches `FK_POE(q)`, not `FK_DH(theta)`); the bridge from [Chapter 7](07_kinbody_bridge.md) is applied internally.
- **`fk_residual`**: what `np.linalg.norm(FK(q) - T_target)` would measure. Compare against your application's tolerance; the solver's own `policy.subproblem_numerical` was a *filter*, not a contract on this value.
- **`refinement_used`**: `"none"` if the solution came directly from algebra; `"lm"` if [`lm_refine`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/refinement/__init__.py) polished it. Surfaces the algebraic-vs-iterative line — see [Chapter 6](06_refinement.md).
- **`refinement_iters`**: number of Newton iterations consumed (0 if `refinement_used == "none"`). Values >5 suggest the algebraic path is struggling on this pose.
- **`branch_id`**: stable IK-branch index (0..15 for the 16-root RR route, 0..7 for spherical-wrist + parallel-shoulder, etc.). Useful for branch-tracking across a trajectory — pick a `q` from `solutions[i].q` where `solutions[i].branch_id` is consistent across calls.
- **`solver_name`**: dotted module path (`"ikgeo.general_6r"`, `"ikgeo.spherical_two_parallel"`, ...). Useful when results pass through a dispatcher that routes across multiple solver candidates.

## Debug `is_ls=True`

`is_ls=True` means **no candidate survived FK validation** — the returned solution list is empty. Common causes:

**1. Target unreachable.** The arm physically can't get to `T_target`. Check your target pose against the reachable workspace. If you computed `T_target = FK(q)` for some known `q`, this can't happen — the target is reachable by definition. If `T_target` came from elsewhere (a planner, a desired pose), it might be outside the workspace.

**2. Singular pose.** The arm *can* reach `T_target` but does so at a kinematic singularity (wrist pitch zero, elbow fully extended, shoulder-pan zero). At a singularity, two or more IK branches collapse into one and the solver may return fewer solutions than the generic case (4 instead of 8, for example). If `is_ls=True` at a singularity, you're on the very edge of the singular set — usually the corresponding `q` is at a value like exactly `0` or exactly `π`.

**3. Tier-2 RR with `allow_refinement=False`** on an ill-conditioned pose. The algebraic candidates miss `policy.subproblem_numerical` and get dropped per the [#74 contract](06_refinement.md). Try `allow_refinement=True`:

```python
solutions, is_ls = general_6r.solve(kb, T_target, allow_refinement=True)
```

If that returns solutions with `refinement_iters > 5`, your arm or pose is in a difficult conditioning regime — see [Chapter 5](05_conditioning.md) for the conditioning theory and [`pick_best_leftvar`](https://github.com/siddhss5/ikfastpy/blob/main/src/ssik/solvers/ikgeo/_raghavan_roth.py) for the AE-3 leftvar selection mechanism (which usually fixes this; if it didn't, you've found a case worth filing as a bug).

**4. Topology mismatch.** You called `spherical_two_parallel.solve` on an arm that isn't actually spherical-wrist-with-parallel-shoulder. Tier-0 solvers raise `ValueError` on topology mismatch up front, with a message identifying which precondition failed. Tier-2 solvers (`general_6r`) accept any 6R but may struggle if the arm is at a degenerate pose for that algorithm — usually that shows up as `is_ls=True` rather than a raised error.

## Performance numbers

Real Kinova JACO 2 (j2n6s200), warm cache, 100 random poses, post-Tier-2.3 of [#86](https://github.com/siddhss5/ikfastpy/pull/90):

| metric | value |
|---|---|
| min | 0.83 ms |
| **median** | **2.25 ms** |
| mean | 3.49 ms |
| p95 | 10.24 ms |
| max | 29.61 ms |
| solutions / pose | median 6, range 1-12 |
| FK error | median 3.7e-13, max 9.6e-08 |
| failures (`is_ls=True`) | 0 / 100 |

Cumulative since the original PR #85 baseline (4.5 ms median): roughly **2× speedup** on the JACO 2 hot path while keeping FK error at machine precision and 0 failures across the whole sequence.

For Pieper-class arms (UR5, Puma 560, etc.), the tier-0 solvers run in ~50–200 µs warm-cache, comparable to EAIK. The cold-cache cost (first IK on a new arm) is tier-dependent:

- Tier-0 closed-form: instant; no symbolic preprocessing needed.
- Tier-1 univariate-search: instant.
- Tier-2 grid-search (`gen_six_dof`): instant.
- **Tier-2 numeric RR (`general_6r`)**: ~150-300 s of one-time sympy preprocessing per arm. Dominated by the AE-3 leftvar selection — the library tries all three leftvar choices, builds (P, Q) symbolically for each, and picks the lowest-condition option. Cached on disk via `lru_cache`; subsequent imports pay zero cost.

The 150-300s is the price of admission for non-Pieper arms. If you're loading the library once per ROS node startup and serving IK calls on a long-running service, this is paid once at startup. If you're running ssik in a short-lived script that processes a few poses and exits, you'll feel it.

## Reading test output

When a CI job shows `xpassed` / `xfailed` counts on `test_solve_all_ik_recovers_q_star_and_alternatives`, that's the platform-sensitive [#82](https://github.com/siddhss5/ikfastpy/issues/82) coverage gap on MC Table I (a synthetic 1994-Manocha–Canny benchmark). Different LAPACK backends pick different IK branches due to floating-point variance in the eigenvalue / dedup path; the test passes either way (`xfail` with `strict=False`) but the counts surface the actual recovery rate per platform. When #82 closes, all four MC seeds will `xpass` and we'll drop the marks.

This is *not* a correctness bug on real arms. The MC Table I synthetic was deliberately constructed by Manocha and Canny in 1994 to have unusual algebraic properties; it's a stress test, not a typical use case. Real arms (JACO 2, UR5, Puma) round-trip at machine precision on every CI matrix entry.

## Common pitfalls

A few things first-time users sometimes hit:

**Building the kb inside an IK loop.** The `poe_to_dh(kb)` conversion is cached on the kb instance; if you rebuild kb every IK call, the cache misses every time and you pay the conversion cost (~1.6 ms on a typical chain). Build kb once outside the loop, reuse.

**Confusing `policy.subproblem_numerical` with `Solution.fk_residual`.** The policy field is the *filter* — solutions with `fk_residual > policy.subproblem_numerical` are dropped. The reported `Solution.fk_residual` is the actual measured value, which can be much smaller than the filter (typical: 1e-13 vs filter at 1e-5).

**Expecting `ikgeo.spherical` to handle JACO 2.** It can't — the topology gate refuses non-spherical wrists. Use `general_6r` instead. ssik will emit a `ValueError` with the exact reason if you call the wrong solver; pay attention to the error message, it identifies the topology mismatch.

**Treating `is_ls=True` as an error.** It's a *signal* — the algebraic pipeline didn't find a solution within tolerance. The right response depends on your application: try `allow_refinement=True`, try a different tolerance, or check the target reachability. Don't assume the kb is broken; the pipeline tells you when it can't solve.
