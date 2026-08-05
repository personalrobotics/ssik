// Self-contained artifact gate for ur5 (three_parallel). Includes ONLY the
// generated artifact + its oracle golden -- no pybind, no Python.
#include "artifact_conformance.hpp"
#include "ur5_ik.hpp"
#include "ur5_ik_solve_parity.hpp"

int main() {
  return ssik::artifact_test::run<ssik::ur5_ik::DOF>(
      "UR5", ssik::ur5_ik::consts(), ssik::ur5_ik::solve_parity_cases(),
      [](const ssik::Pose& T) { return ssik::ur5_ik::solve(T); });
}
