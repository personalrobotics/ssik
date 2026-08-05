// Shared self-contained-artifact conformance check (THE GATE). Each per-arm test
// is a 3-line main() over this: it includes ONLY the generated artifact + its
// oracle golden (no pybind, no Python) and asserts ssik::<arm>::solve(T) closes
// FK and agrees with the Python oracle set under wrap-to-pi dedup.
#pragma once

#include <array>
#include <cmath>
#include <cstdio>

#include "ssik_cpp/fk.hpp"

namespace ssik::artifact_test {

inline double wrap_pi(double a) {
  const double t = 2.0 * M_PI;
  return std::fmod(a + M_PI + 2.0 * t, t) - M_PI;
}

template <int DOF>
bool wrap_close(const std::array<double, DOF>& a, const std::array<double, DOF>& b, double tol) {
  for (int i = 0; i < DOF; ++i)
    if (std::abs(wrap_pi(a[i] - b[i])) > tol) return false;
  return true;
}

// Cases: a container of {std::array<double,16> target; vector<array<double,DOF>>
// solutions;}. SolveFn: Pose -> vector<Solution<DOF>>.
template <int DOF, typename Cases, typename SolveFn>
int run(const char* name, const JointConsts<DOF>& c, const Cases& cases, SolveFn solve_fn) {
  double worst_fk = 0.0;
  int mismatched = 0;
  for (std::size_t ci = 0; ci < cases.size(); ++ci) {
    const auto& tc = cases[ci];
    Pose T;
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col) T(r, col) = tc.target[r * 4 + col];

    const auto sols = solve_fn(T);
    for (const auto& s : sols) worst_fk = std::max(worst_fk, (fk<DOF>(c, s.q) - T).norm());

    bool ok = sols.size() == tc.solutions.size();
    for (const auto& e : tc.solutions) {
      bool found = false;
      for (const auto& s : sols)
        if (wrap_close<DOF>(e, s.q, 1e-3)) { found = true; break; }
      if (!found) ok = false;
    }
    for (const auto& s : sols) {
      bool found = false;
      for (const auto& e : tc.solutions)
        if (wrap_close<DOF>(e, s.q, 1e-3)) { found = true; break; }
      if (!found) ok = false;
    }
    if (!ok) {
      ++mismatched;
      if (mismatched <= 5)
        std::printf("  case %zu: artifact=%zu oracle=%zu\n", ci, sols.size(), tc.solutions.size());
    }
  }
  std::printf("%s self-contained artifact: %zu poses, worst FK = %.3e, mismatches = %d\n", name,
              cases.size(), worst_fk, mismatched);
  if (worst_fk < 1e-7 && mismatched == 0) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL\n");
  return 1;
}

}  // namespace ssik::artifact_test
