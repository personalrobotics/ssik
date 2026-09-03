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
//
// max_incomplete: per-arm allowance for poses where C++ MISSES an oracle solution.
// Default 0 (strict: C++ must cover the whole oracle). Set >0 ONLY for a solver
// with a documented, bounded completeness gap vs the Python oracle -- e.g. kassow
// (HP jointlock), whose monic-companion eigensolve recovers fewer roots than
// LAPACK dggev at the degenerate lock samples (axes aligned at multiples of pi/2);
// tracked in #544. Duplicate C++ branches are ALWAYS a hard failure (never
// allowed), independent of this knob.
template <int DOF, typename Cases, typename SolveFn>
int run(const char* name, const JointConsts<DOF>& c, const Cases& cases, SolveFn solve_fn,
        double fk_ceiling = 1e-7, int max_incomplete = 0) {
  double worst_fk = 0.0;
  int incomplete = 0;  // poses where C++ dropped an oracle solution
  int dup = 0;         // poses with a duplicate C++ branch (always fatal)
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
    bool miss = false;
    for (const auto& e : tc.solutions) {
      bool found = false;
      for (const auto& s : sols)
        if (wrap_close<DOF>(e, s.q, 1e-3)) { found = true; break; }
      if (!found) miss = true;  // C++ dropped a solution the oracle found
    }
    bool has_dup = false;
    for (std::size_t i = 0; i < sols.size(); ++i)
      for (std::size_t j = i + 1; j < sols.size(); ++j)
        if (wrap_close<DOF>(sols[i].q, sols[j].q, 1e-3)) has_dup = true;  // duplicate branch
    if (sols.size() > tc.solutions.size()) extensions += sols.size() - tc.solutions.size();
    if (miss) {
      ++incomplete;
      if (incomplete <= 5)
        std::printf("  case %zu: artifact=%zu oracle=%zu (C++ missing an oracle sol)\n", ci,
                    sols.size(), tc.solutions.size());
    }
    if (has_dup) {
      ++dup;
      std::printf("  case %zu: artifact=%zu (duplicate C++ branch)\n", ci, sols.size());
    }
  }
  std::printf(
      "%s self-contained artifact: %zu poses, worst FK = %.3e, incomplete = %d (allow %d), dup = %d, "
      "extensions = %d\n",
      name, cases.size(), worst_fk, incomplete, max_incomplete, dup, extensions);
  if (worst_fk <= fk_ceiling && incomplete <= max_incomplete && dup == 0) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL\n");
  return 1;
}

}  // namespace ssik::artifact_test
