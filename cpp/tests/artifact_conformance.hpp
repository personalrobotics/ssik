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
// fk_ceiling: the arm's own FK-closure tolerance (its solver's fk_atol). A
// solution that closes within its solver's tolerance is valid; a global 1e-7 is
// wrong for families whose gate is looser (RR / general_6r at 1e-5, where force-
// refined near-double-root solutions settle ~1e-6). The emitter passes it.
template <int DOF, typename Cases, typename SolveFn>
int run(const char* name, const JointConsts<DOF>& c, const Cases& cases, SolveFn solve_fn,
        double fk_ceiling = 1e-7) {
  double worst_fk = 0.0;
  int mismatched = 0;
  int extensions = 0;  // C++ solutions beyond the oracle (valid, sound + distinct)
  for (std::size_t ci = 0; ci < cases.size(); ++ci) {
    const auto& tc = cases[ci];
    Pose T;
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col) T(r, col) = tc.target[r * 4 + col];

    const auto sols = solve_fn(T);
    for (const auto& s : sols) worst_fk = std::max(worst_fk, (fk<DOF>(c, s.q) - T).norm());

    // #487 conformance: relative completeness + soundness, NOT bit-exact match.
    // C++ (Eigen JacobiSVD / RealQZ) is legitimately more complete than the
    // numpy oracle at degenerate poses, and numpy's completeness is LAPACK-
    // backend-dependent (OpenBLAS finds fewer than Accelerate). So a C++ EXTRA
    // that FK-closes + is distinct is a valid extension, not a mismatch. A real
    // failure is: C++ MISSING an oracle solution, or a duplicate C++ branch.
    bool ok = true;
    for (const auto& e : tc.solutions) {
      bool found = false;
      for (const auto& s : sols)
        if (wrap_close<DOF>(e, s.q, 1e-3)) { found = true; break; }
      if (!found) ok = false;  // C++ dropped a solution the oracle found
    }
    for (std::size_t i = 0; i < sols.size(); ++i)
      for (std::size_t j = i + 1; j < sols.size(); ++j)
        if (wrap_close<DOF>(sols[i].q, sols[j].q, 1e-3)) ok = false;  // duplicate branch
    if (sols.size() > tc.solutions.size()) extensions += sols.size() - tc.solutions.size();
    if (!ok) {
      ++mismatched;
      if (mismatched <= 5)
        std::printf("  case %zu: artifact=%zu oracle=%zu (C++ missing an oracle sol or dup)\n", ci,
                    sols.size(), tc.solutions.size());
    }
  }
  std::printf(
      "%s self-contained artifact: %zu poses, worst FK = %.3e, incomplete/dup = %d, extensions = %d\n",
      name, cases.size(), worst_fk, mismatched, extensions);
  if (worst_fk <= fk_ceiling && mismatched == 0) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL\n");
  return 1;
}

}  // namespace ssik::artifact_test
