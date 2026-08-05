// Self-contained artifact gate for irb6700 (spherical_two_parallel, offset wrist).
// Proves the baked-canonicalization path: the artifact carries CANONICAL constants
// (the wrist gauge moved from runtime to build time), so it solves standalone --
// no pybind, no Python, no runtime canonicalize.
#include "artifact_conformance.hpp"
#include "irb6700_ik.hpp"
#include "irb6700_ik_solve_parity.hpp"

int main() {
  return ssik::artifact_test::run<ssik::irb6700_ik::DOF>(
      "IRB6700", ssik::irb6700_ik::consts(), ssik::irb6700_ik::solve_parity_cases(),
      [](const ssik::Pose& T) { return ssik::irb6700_ik::solve(T); });
}
