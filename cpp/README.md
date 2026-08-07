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
