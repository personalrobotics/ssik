// Shared self-contained-artifact conformance check (THE GATE). Each per-arm test
// is a 3-line main() over this: it includes ONLY the generated artifact + its
// oracle golden (no pybind, no Python) and asserts ssik::<arm>::solve(T) closes
// FK and agrees with the Python oracle set under same-root dedup (#600).
#pragma once

#include <array>
#include <cmath>
#include <cstdio>

#include "ssik_cpp/dedup.hpp"
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
// match_tol: the L-infinity radius (every joint wrapped) within which a C++
// solution covers an oracle one; the emitter passes 1e-3 for 6R and 1e-2 for
// 7R, whose solutions sample a one-dimensional self-motion (#550).
//
// known_incomplete / known_gap: a native completeness gap tied to its issue
// (the emitter's _KNOWN_INCOMPLETE). Default 0: C++ must cover the whole
// oracle. With an entry, at most known_incomplete poses may miss, and a run
// that misses on none fails too: the gap is closed, so its entry must go.
// Duplicate C++ branches are ALWAYS a hard failure, independent of this knob.
template <int DOF, typename Cases, typename SolveFn>
int run(const char* name, const JointConsts<DOF>& c, const Cases& cases, SolveFn solve_fn,
        double fk_ceiling = 1e-7, double match_tol = 1e-3, int known_incomplete = 0,
        const char* known_gap = "") {
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
        if (wrap_close<DOF>(e, s.q, match_tol)) { found = true; break; }
      if (!found) miss = true;  // C++ dropped a solution the oracle found
    }
    // A duplicate is two returned solutions that are the same root (#600).
    // Two within 1e-3 are not enough: near a fold, twin roots a fraction of a
    // milliradian apart are distinct branches the golden also contains.
    bool has_dup = false;
    const double floor = same_root_floor(T);
    for (std::size_t i = 0; i < sols.size(); ++i)
      for (std::size_t j = i + 1; j < sols.size(); ++j)
        if (wrap_close<DOF>(sols[i].q, sols[j].q, 1e-3) &&
            is_same_root<DOF>(c, T, sols[i], sols[j], floor))
          has_dup = true;  // duplicate branch
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
      "%s self-contained artifact: %zu poses, worst FK = %.3e, incomplete = %d (known %d%s%s), "
      "dup = %d, extensions = %d\n",
      name, cases.size(), worst_fk, incomplete, known_incomplete, known_incomplete ? ": " : "",
      known_gap, dup, extensions);
  const bool stale = known_incomplete > 0 && incomplete == 0;
  if (stale)
    std::printf("  no pose misses any more: %s is closed, delete its _KNOWN_INCOMPLETE entry\n",
                known_gap);
  if (worst_fk <= fk_ceiling && incomplete <= known_incomplete && !stale && dup == 0) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL\n");
  return 1;
}

}  // namespace ssik::artifact_test
