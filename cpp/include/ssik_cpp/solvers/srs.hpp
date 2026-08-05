// Self-contained SRS-class 7R artifact solve (#515): the C++ replica of the
// generated <arm>_ik.solve() for the seven_r.srs family. Composes the pieces a
// pure-C++ consumer needs with ZERO runtime Python -- geometry constants are
// baked at emit time (Python is the compiler). The pipeline mirrors
// codegen.py's thin-wrapper solve() exactly:
//
//   1. seeded_track fast path (q_seed + max_solutions == 1): Newton-continue
//      from the seed, finalize, return if non-empty.
//   2. analytical solve (srs_canonical_solve, the Singh-Kreutz swivel sweep).
//   3. T-perturbation rescue gate: PROVEN DORMANT for the canonical SRS arms
//      (well-conditioned; the analytical path is never empty on a reachable
//      pose) and guarded red-on-fire by a Python dormancy fuzz, so the body is
//      omitted -- matching three_parallel_artifact_solve. The rescue uses a
//      numpy RNG that cannot be reproduced bit-for-bit in C++ anyway; it lands
//      with the RR/HP families where it is load-bearing.
//   4. finalize (limits -> seed -> truncate) with the #359 in-limits fallback
//      wired to resolve_in_limits (the exact feasible-swivel resolver).
#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/newton.hpp"  // seeded_track
#include "ssik_cpp/seven_r/srs_swivel_limits.hpp"
#include "ssik_cpp/solvers/srs_canonical.hpp"

namespace ssik {

// Reach-sphere upper bound (sum of all link translation norms), the rescue
// gate. Triangle-inequality bound, so it never rejects a reachable pose.
inline double reach_radius(const JointConsts<7>& c) {
  double r = 0.0;
  for (int i = 0; i < 7; ++i) {
    r += c.t_left[i].block<3, 1>(0, 3).norm();
    r += c.t_right[i].block<3, 1>(0, 3).norm();
  }
  return r;
}

// Full artifact-contract solve for a canonical-ZYZ offset-free SRS arm. All
// geometry is baked (JointConsts + SrsConsts + JointLimits); no Python.
inline std::vector<Solution<7>> srs_artifact_solve(const JointConsts<7>& c, const SrsConsts& s,
                                                   const JointLimits<7>& lim, const Pose& T,
                                                   const ArtifactParams<7>& p) {
  // _joint_limits(kb): baked limits, defaulting to [-pi, pi] where absent --
  // the exact box the Python in-limits fallback resolves against.
  std::array<std::array<double, 2>, 7> limits;
  for (int i = 0; i < 7; ++i)
    limits[i] = lim.present[i] ? std::array<double, 2>{lim.lo[i], lim.hi[i]}
                               : std::array<double, 2>{-M_PI, M_PI};

  // 1. Seeded numerical-tracking fast path (#380).
  if (p.has_seed && p.max_solutions == 1) {
    const auto tracked = seeded_track<7>(c, p.q_seed, T);
    if (tracked) {
      // Same postprocess as the full path, but no in-limits fallback: a tracked
      // seed that fails limits/tolerance falls through to the analytical solve.
      const auto fast = finalize_solutions<7>({*tracked}, c, lim, p);
      if (!fast.empty()) return fast;
    }
  }

  // 2. Analytical swivel sweep.
  std::vector<Solution<7>> sols = srs_canonical_solve(c, s, T);

  // 3. Rescue gate (dormant for this family; see the file comment). If ported,
  //    the rescue would run here when sols.empty() && p.allow_rescue &&
  //    T.p.norm() <= reach.
  const double reach = reach_radius(c);
  (void)reach;

  // 4. Finalize with the #359 exact in-limits fallback.
  return finalize_solutions<7>(std::move(sols), c, lim, p, [&]() {
    return srs_swivel::resolve_in_limits(c, s, T, limits, kSrsFkThreshold);
  });
}

}  // namespace ssik
