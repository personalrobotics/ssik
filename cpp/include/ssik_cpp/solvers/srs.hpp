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

  // 3. Limit pass + #359 in-limits fallback ONLY (no seed/tolerance/truncate
  //    yet): the rescue gate is "no in-limits solution exists", so it must not
  //    depend on the seed-tolerance / max_solutions filters -- a seed filter
  //    emptying a non-empty in-limits set is a user preference, not a missing
  //    solution, and must not trigger a rescue (#524).
  ArtifactParams<7> p_limits;
  p_limits.respect_limits = p.respect_limits;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  std::vector<Solution<7>> in_limits = finalize_solutions<7>(core(T), c, lim, p_limits, [&]() {
    return srs_swivel::resolve_in_limits(c, s, T, limits, kSrsFkThreshold);
  });

  // 4. Rescue gate (#319/#524): nothing survives the limit filter. If the target
  //    is within reach it is a measure-zero rank-deficient pose (a kinematic
  //    singularity where the closed-form degenerates), not an unreachable target
  //    -- recover via the T-perturbation rescue, then re-apply the limit filter.
  if (in_limits.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    in_limits = finalize_solutions<7>(rescue_via_T_perturbation<7>(core, c, T), c, lim, p_limits);
  }

  // 5. Seed tolerance / ranking / truncate over the in-limits set.
  ArtifactParams<7> p_seed = p;
  p_seed.respect_limits = false;
  return finalize_solutions<7>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
