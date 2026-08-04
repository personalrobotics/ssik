// Self-contained artifact conformance -- THE GATE (pure-C++ deployable).
//
// This program includes ONLY the generated artifact (ur5_ik.hpp) -- no pybind,
// no Python, no _native.py -- and calls ssik::ur5_ik::solve(T) directly. It is
// the proof that a C++ consumer (MoveIt, OMPL, embedded) can deploy the artifact
// with zero Python runtime. It checks the artifact's solutions against the
// Python oracle (solve_parity golden): every returned IK FK-closes, and the set
// agrees with Python under wrap-to-pi dedup.
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "ur5_ik.hpp"             // the self-contained artifact (baked consts + solve)
#include "ur5_ik_solve_parity.hpp"  // Python oracle golden (git-ignored, regenerated)

namespace {

constexpr int DOF = ssik::ur5_ik::DOF;
constexpr double kDedup = 1e-3;

double wrap_pi(double a) {
  const double two_pi = 2.0 * M_PI;
  return std::fmod(a + M_PI + 2.0 * two_pi, two_pi) - M_PI;
}

bool wrap_close(const std::array<double, DOF>& a, const std::array<double, DOF>& b) {
  for (int i = 0; i < DOF; ++i)
    if (std::abs(wrap_pi(a[i] - b[i])) > kDedup) return false;
  return true;
}

}  // namespace

int main() {
  const auto c = ssik::ur5_ik::consts();
  const auto& cases = ssik::ur5_ik::solve_parity_cases();

  double worst_fk = 0.0;
  int mismatched = 0;

  for (std::size_t ci = 0; ci < cases.size(); ++ci) {
    const auto& tc = cases[ci];
    ssik::Pose T;
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col) T(r, col) = tc.target[r * 4 + col];

    const auto sols = ssik::ur5_ik::solve(T);  // <-- the self-contained C++ solve

    for (const auto& s : sols)
      worst_fk = std::max(worst_fk, (ssik::fk<DOF>(c, s.q) - T).norm());

    bool ok = sols.size() == tc.solutions.size();
    for (const auto& e : tc.solutions) {
      bool found = false;
      for (const auto& s : sols)
        if (wrap_close(e, s.q)) { found = true; break; }
      if (!found) ok = false;
    }
    for (const auto& s : sols) {
      bool found = false;
      for (const auto& e : tc.solutions)
        if (wrap_close(e, s.q)) { found = true; break; }
      if (!found) ok = false;
    }
    if (!ok) {
      ++mismatched;
      if (mismatched <= 5)
        std::printf("  case %zu: artifact=%zu oracle=%zu\n", ci, sols.size(), tc.solutions.size());
    }
  }

  std::printf("UR5 self-contained artifact: %zu poses, worst FK = %.3e, mismatches = %d\n",
              cases.size(), worst_fk, mismatched);
  if (worst_fk < 1e-7 && mismatched == 0) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL\n");
  return 1;
}
