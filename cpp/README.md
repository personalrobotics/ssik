# ssik native C++ IK artifacts

Self-contained, header-only inverse-kinematics solvers generated from the ssik
Python "compiler". Each `<arm>_ik.hpp` bakes one robot's geometry and exposes a
`solve(T)` that returns all IK solutions for a target pose — **zero runtime
Python, zero ssik build step** on the consumer side. Intended for C++ / MoveIt /
real-time use where the Python library isn't an option.

```cpp
#include "iiwa14_ik.hpp"

using namespace ssik;
namespace arm = ssik::iiwa14_ik;

Pose T = fk<arm::DOF>(arm::consts(), q);          // 4x4 target
std::vector<Solution<arm::DOF>> sols = arm::solve(T);  // all in-limits IK
// sols[i].q is the joint vector; sols[i].fk_residual its FK closure.
```

`Pose`, `Solution`, and `fk<DOF>` live in namespace `ssik`; the per-arm
`consts()`, `solve()`, and `DOF` live in `ssik::<arm>_ik`. `solve` takes an
optional `ArtifactParams<DOF>` for limits / seed ranking / `max_solutions`.
Its defaults match the Python API exactly, so the same call returns the same
set on either side.

### Joints that can turn more than once

A joint whose limits span more than a full turn (the UR family's `[-2π, 2π]`)
reaches the same pose at several different joint coordinates, and since v6.0
`solve()` returns all of them — 256 for a UR pose rather than 8. They are
in-limit **lifts** of the same geometric branch, not extra IK branches: same
end-effector pose, different admissible configuration, different distance from
wherever the robot is now.

```cpp
ssik::ArtifactParams<arm::DOF> p;
p.enumerate_windings = false;         // one representative per geometric branch
auto sols = arm::solve(T, p);

p.enumerate_windings = true;          // the default
p.has_seed = true; p.q_seed = q_now;
p.max_solutions = 1;                  // nearest configuration; skips building the rest
auto tracked = arm::solve(T, p);
```

A seeded, capped solve costs the same as the un-enumerated one — it takes the
globally nearest solution directly instead of materialising what it would
discard.

## Use it (CMake)

Install the package, then `find_package` it:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --install build --prefix /path/to/install
```

```cmake
find_package(ssik_cpp REQUIRED)
target_link_libraries(my_app PRIVATE ssik::ssik_cpp)
```

That puts the primitives (`ssik_cpp/…`) and every committed `<arm>_ik.hpp` on the
include path. The only dependency is **Eigen** (header-only) — the exported
package `find_dependency()`s it, so Eigen must be findable
(`brew install eigen` / `apt install libeigen3-dev`).

Or bare, without CMake:

```bash
c++ -std=c++20 -I<install>/include -I<eigen-include> my_app.cpp
```

`examples/solve_arm.cpp` is a complete, runnable example;
`examples/consumer/` is a standalone downstream project that consumes the
**installed** package via `find_package` (the "C++ consumer" CI smoke builds it).

## Self-motion charts (redundant 7R)

`ssik_cpp/chart.hpp` exposes the self-motion manifold of a 7R pose as charts, the
C++ counterpart of `ssik.chart`: one continuous branch `q(t)` per chart with a
stable label and its domain, plus the inverse map `locate(q)`. Build a family once
per target pose (microseconds: the elbow-reachability arcs are closed form) and
evaluate per tick; `q(t)` and `locate(q)` cost about a microsecond. A chart's
`domain(i)` is computed on first request per reachable arc and cached.

```cpp
#include "ssik_cpp/chart.hpp"
// Franka Panda / FR3: t = q6. `coef` is the arm's baked (3,48) geometry
// (SphericalShoulderConsts::coef in the generated <arm>_ik.hpp).
auto family = ssik::chart::SphericalShoulderCharts::build(coef, T_target);
double t, mismatch;
int chart = family.locate(q_now, 1e-6, t, mismatch);   // -1 if q_now is not on FK^-1(T)
std::array<double, 7> q;
family.q(chart, t + 0.01, q);                           // a step along the arm's own branch
const auto& dom = family.domain(chart);                 // its q6 intervals, computed on first request
std::array<double, 7> dq;
family.tangent(chart, t, dq);                           // dq/dt along the branch (srs.tangent(i, psi) is closed form)
const auto arcs = family.in_limits(chart, joint_limits);  // the branch under joint limits, exact

// KUKA iiwa and other exact SRS arms: t = elbow swivel, charts are full circles.
ssik::chart::SrsCharts srs;
srs.init(joint_consts, srs_consts, T_target);
```

## Which arms

The committed `gen/<arm>_ik.hpp` are the shippable artifacts. Generate any native
arm on demand from the Python catalog:

```bash
python scripts/cpp_emit.py <arm>_ik      # e.g. franka_panda_ik
python scripts/cpp_emit.py --all         # every already-emitted arm
```

Native families today: `three_parallel` (UR-class 6R), `spherical_two_parallel`
(Pieper 6R), and `seven_r.srs` (iiwa/Rizon-class 7R, canonical + general). Each
artifact is validated Python-free against the Python oracle by the data-driven
gate (`tests/test_artifacts.cpp`).

## Completeness

`solve()` is the full contract, not just the analytical sweep: joint-limit
filtering, an exact in-limits resolver for redundant 7R, and a T-perturbation
rescue that recovers reachable rank-deficient (near-singular) poses instead of
returning empty. Every returned solution FK-closes and respects limits.
