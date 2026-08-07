// Minimal external-consumer example for the self-contained ssik C++ IK artifacts.
//
// Everything a consumer needs is in one header: `#include "<arm>_ik.hpp"` and
// call `ssik::<arm>_ik::solve(T)`. Zero runtime Python, no ssik build step --
// the geometry is baked into the header. `Pose`, `Solution`, `fk<DOF>` live in
// the `ssik` namespace; the per-arm `consts()`, `solve()`, `DOF` live in
// `ssik::<arm>_ik`.
//
// Build (against an installed package):
//   find_package(ssik_cpp REQUIRED)
//   target_link_libraries(app PRIVATE ssik::ssik_cpp)
// or bare:
//   c++ -std=c++20 -I<prefix>/include -I<eigen-include> solve_arm.cpp

#include <array>
#include <cstdio>

#include "iiwa14_ik.hpp"

int main() {
  using namespace ssik;
  namespace arm = ssik::iiwa14_ik;

  // A reachable target: the forward kinematics of a joint configuration.
  const std::array<double, arm::DOF> q0 = {0.3, 0.4, -0.5, 0.6, 0.2, -0.3, 0.4};
  const Pose target = fk<arm::DOF>(arm::consts(), q0);

  // Inverse kinematics: all joint solutions reaching that target, in-limits.
  const std::vector<Solution<arm::DOF>> solutions = arm::solve(target);

  double worst_fk = 0.0;
  for (const auto& s : solutions)
    worst_fk = std::max(worst_fk, (fk<arm::DOF>(arm::consts(), s.q) - target).norm());

  std::printf("iiwa14: %zu IK solutions, worst FK residual %.2e\n", solutions.size(), worst_fk);
  return solutions.empty() ? 1 : 0;
}
