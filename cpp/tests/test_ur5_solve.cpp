// UR5 solve conformance (#487): the native three_parallel solver must (a) close
// FK on every returned solution (soundness) and (b) reproduce the Python
// reference's analytical solution set under wrap-to-pi dedup (behavioral parity),
// on poses emitted by scripts/cpp_emit.py from three_parallel.solve(
// allow_refinement=False).
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/solvers/three_parallel.hpp"
#include "ur5_ik.hpp"
#include "ur5_ik_solve_parity.hpp"

namespace {

constexpr int DOF = ssik::ur5_ik::DOF;
constexpr double kDedup = 1e-3;  // Tolerances::dedup / policy.subproblem_dedup

double wrap_pi(double a) {
  const double two_pi = 2.0 * M_PI;
  return std::fmod(a + M_PI + 2.0 * two_pi, two_pi) - M_PI;
}

bool wrap_close(const std::array<double, DOF>& a, const std::array<double, DOF>& b) {
  for (int i = 0; i < DOF; ++i) {
    if (std::abs(wrap_pi(a[i] - b[i])) > kDedup) return false;
  }
  return true;
}

}  // namespace

int main() {
  const auto c = ssik::ur5_ik::consts();
  const auto& cases = ssik::ur5_ik::solve_parity_cases();

  double worst_fk = 0.0;
  int total_expected = 0, total_native = 0;
  int mismatched_cases = 0;

  for (std::size_t ci = 0; ci < cases.size(); ++ci) {
    const auto& tc = cases[ci];
    ssik::Pose T;
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col) T(r, col) = tc.target[r * 4 + col];

    const auto sols = ssik::three_parallel_solve(c, T);
    total_native += static_cast<int>(sols.size());
    total_expected += static_cast<int>(tc.solutions.size());

    // (a) Soundness: every native solution FK-closes to the target.
    for (const auto& s : sols) {
      const ssik::Pose fk_q = ssik::fk<DOF>(c, s.q);
      worst_fk = std::max(worst_fk, (fk_q - T).norm());
    }

    // (b) Set agreement (both directions) under wrap-to-pi dedup.
    bool ok = tc.solutions.size() == sols.size();
    for (const auto& e : tc.solutions) {
      bool found = false;
      for (const auto& s : sols)
        if (wrap_close(e, s.q)) {
          found = true;
          break;
        }
      if (!found) ok = false;
    }
    for (const auto& s : sols) {
      bool found = false;
      for (const auto& e : tc.solutions)
        if (wrap_close(e, s.q)) {
          found = true;
          break;
        }
      if (!found) ok = false;
    }
    if (!ok) {
      ++mismatched_cases;
      if (mismatched_cases <= 5) {
        std::printf("  case %zu: native=%zu expected=%zu\n", ci, sols.size(),
                    tc.solutions.size());
      }
    }
  }

  std::printf("UR5 solve conformance: %zu poses, expected=%d native=%d\n", cases.size(),
              total_expected, total_native);
  std::printf("  worst FK closure = %.3e (gate %.0e)\n", worst_fk, ssik::kThreeParallelFkAtol);
  std::printf("  set-agreement mismatches = %d\n", mismatched_cases);

  if (worst_fk < ssik::kThreeParallelFkAtol && mismatched_cases == 0) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL\n");
  return 1;
}
