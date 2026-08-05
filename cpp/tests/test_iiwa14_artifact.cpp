// Self-contained artifact gate for iiwa14 (seven_r.srs). Includes ONLY the
// generated artifact + its oracle golden -- no pybind, no Python. Exercises the
// full SRS artifact pipeline (seeded_track / srs_canonical_solve / finalize with
// the resolve_in_limits in-limits fallback) behind ssik::iiwa14_ik::solve(T).
#include "artifact_conformance.hpp"
#include "iiwa14_ik.hpp"
#include "iiwa14_ik_solve_parity.hpp"

int main() {
  return ssik::artifact_test::run<ssik::iiwa14_ik::DOF>(
      "iiwa14", ssik::iiwa14_ik::consts(), ssik::iiwa14_ik::solve_parity_cases(),
      [](const ssik::Pose& T) { return ssik::iiwa14_ik::solve(T); });
}
