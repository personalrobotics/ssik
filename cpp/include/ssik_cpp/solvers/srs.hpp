// Self-contained SRS-class 7R artifact solve (#515): the C++ replica of the
// generated <arm>_ik.solve() for the seven_r.srs family. Composes the pieces a
// pure-C++ consumer needs with ZERO runtime Python -- geometry constants are
// baked at emit time (Python is the compiler). The pipeline mirrors
// codegen.py's thin-wrapper solve() exactly:
//
//   1. seeded_track fast path (q_seed + max_solutions == 1): Newton-continue
//      from the seed, finalize, return if non-empty.
//   2. analytical solve (canonical Singh-Kreutz or general Davenport sweep).
//   3. T-perturbation rescue (#319): when the analytical path is empty at a
//      reachable target -- a measure-zero kinematic singularity where the
//      closed-form solve degenerates (within ~1e-9 rad of an exact singularity;
//      the exact-SRS analytical is complete everywhere else) -- recover via the
//      shared, family-agnostic rescue. This is what makes the standalone artifact
//      genuinely complete: no Python fallback, so it must handle the singular
//      poses Python's solve() does.
//   4. finalize (limits -> seed -> truncate) with the #359 in-limits fallback
//      wired to resolve_in_limits (the exact feasible-swivel resolver).
#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/newton.hpp"  // seeded_track
#include "ssik_cpp/rescue.hpp"
#include "ssik_cpp/seven_r/srs_swivel_limits.hpp"
#include "ssik_cpp/solvers/srs_canonical.hpp"
#include "ssik_cpp/solvers/srs_general.hpp"

namespace ssik {

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

  // 2. Analytical swivel sweep -- canonical ZYZ fast-path or the general
  //    Davenport path, per the baked dispatch flag (mirrors use_canonical).
  const auto core = [&](const Pose& Tp) {
    return s.general_path ? srs_general_solve(c, s, Tp) : srs_canonical_solve(c, s, Tp);
  };
  std::vector<Solution<7>> sols = core(T);

  // 3. Rescue gate (#319): the analytical extraction found nothing. If the
  //    target is within reach it is a measure-zero rank-deficient pose (a
  //    kinematic singularity where the closed-form solve degenerates), not an
  //    unreachable target -- recover via the T-perturbation rescue. The
  //    reach-sphere (triangle-inequality upper bound) is checked only in this
  //    rare empty branch, so far-field targets stay cheap.
  if (sols.empty() && p.allow_rescue &&
      T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    sols = rescue_via_T_perturbation<7>(core, c, T);
  }

  // 4. Finalize with the #359 exact in-limits fallback.
  return finalize_solutions<7>(std::move(sols), c, lim, p, [&]() {
    return srs_swivel::resolve_in_limits(c, s, T, limits, kSrsFkThreshold);
  });
}

}  // namespace ssik
