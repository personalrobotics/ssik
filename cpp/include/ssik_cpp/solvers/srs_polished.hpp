// Self-contained approximate-SRS 7R artifact solve (#550): the C++ replica of the
// generated <arm>_ik.solve() for the seven_r.srs_polished family (gen3, JACO
// j2s7s300, rm75, yumi L/R). These arms are SRS up to a small axis drift (<= 4cm),
// so the exact SRS core returns cm-off algebraic candidates that an LM polish
// against the TRUE FK corrects. Mirrors ssik.solvers.seven_r.srs_polished.solve:
//
//   1. raw candidates from the canonical SRS core with reach_slack = 2*max_drift
//      and a keep-all FK threshold (srs.solve(reach_slack=..., fk_atol=10.0)).
//   2. LM-polish every candidate against the real JointConsts FK, keep those that
//      close to polish_fk_atol, cluster-merge.
//   3. finalize (limits -> in-limits fallback -> rescue -> seed/truncate).
//
// The SrsConsts here are baked under a RELAXED classifier (axis_intersect =
// max_drift) so the approximate pivots pass -- see ssik._native.
// srs_polished_native_geometry. The geometry is otherwise identical to SRS.
#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/newton.hpp"  // lm_refine
#include "ssik_cpp/rescue.hpp"
#include "ssik_cpp/seven_r/srs_swivel_limits.hpp"
#include "ssik_cpp/solvers/srs_canonical.hpp"

namespace ssik {

// srs_polished tuning (ssik.solvers.seven_r.srs_polished defaults).
inline constexpr double kSrsPolishedMaxDrift = 0.04;      // _DEFAULT_MAX_DRIFT_M
inline constexpr double kSrsPolishedReachSlack = 0.08;    // 2 * max_drift
inline constexpr double kSrsPolishedFkAtol = 1e-12;       // polish_fk_atol accept
inline constexpr double kSrsPolishedKeepAll = 1e9;        // fk_atol=10.0 -> keep every candidate
inline constexpr int kSrsPolishedMaxIters = 30;          // polish_max_iters (cm-off seeds need it)

namespace srs_polished_detail {

// LM-polish every raw candidate against the true FK, keep residual <= atol, then
// wrap-to-pi cluster-merge (mirrors _polish.polish_candidates, which uses
// lm_refine_batch's tighter divergence guard 2.0/2 -- passed explicitly so a seed
// stays on its local branch instead of wandering across the redundant manifold to
// a different solution, which would both miss the oracle's branch and add a dup).
inline std::vector<Solution<7>> polish(const JointConsts<7>& c,
                                       const std::vector<Solution<7>>& raw, const Pose& T,
                                       int max_iters) {
  std::vector<Solution<7>> polished;
  for (const auto& cand : raw) {
    auto r = lm_refine<7>(c, cand.q, T, kSrsPolishedFkAtol, max_iters, /*divergence_factor=*/2.0,
                          /*divergence_min_iters=*/2, /*fixed_damping=*/1e-9);
    if (r && r->second <= kSrsPolishedFkAtol)
      polished.push_back(Solution<7>{r->first, r->second, Refinement::Lm});
  }
  // Cluster-merge in wrap-to-pi max-joint distance (keep lower-residual dup).
  std::vector<Solution<7>> out;
  for (const auto& cand : polished) {
    int dup = -1;
    for (std::size_t j = 0; j < out.size() && dup < 0; ++j) {
      bool close = true;
      for (int i = 0; i < 7; ++i)
        if (std::abs(detail::wrap_pi(cand.q[i] - out[j].q[i])) > kSrsDedupTol) {
          close = false;
          break;
        }
      if (close) dup = static_cast<int>(j);
    }
    if (dup < 0)
      out.push_back(cand);
    else if (cand.fk_residual < out[dup].fk_residual)
      out[dup] = cand;
  }
  return out;
}

}  // namespace srs_polished_detail

// Full artifact-contract solve for an approximate-SRS (srs_polished) arm.
inline std::vector<Solution<7>> srs_polished_artifact_solve(const JointConsts<7>& c,
                                                            const SrsConsts& s,
                                                            const JointLimits<7>& lim, const Pose& T,
                                                            const ArtifactParams<7>& p) {
  std::array<std::array<double, 2>, 7> limits;
  for (int i = 0; i < 7; ++i)
    limits[i] = lim.present[i] ? std::array<double, 2>{lim.lo[i], lim.hi[i]}
                               : std::array<double, 2>{-M_PI, M_PI};

  // core: exact SRS candidates (reach-slackened, keep-all) -> LM-polish -> dedup.
  const auto core = [&](const Pose& Tp) {
    const auto raw =
        srs_canonical_solve(c, s, Tp, kSrsPolishedReachSlack, kSrsPolishedKeepAll);
    return srs_polished_detail::polish(c, raw, Tp, kSrsPolishedMaxIters);
  };

  // Limit pass + #359 in-limits fallback (the SRS swivel resolver, wired for
  // srs_polished by codegen). Its exact-geometry solutions are cm-off for these
  // approximate arms, so polish them too before the limit filter.
  ArtifactParams<7> p_limits;
  p_limits.respect_limits = p.respect_limits;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  std::vector<Solution<7>> in_limits = finalize_solutions<7>(core(T), c, lim, p_limits, [&]() {
    return srs_polished_detail::polish(
        c, srs_swivel::resolve_in_limits(c, s, T, limits, kSrsPolishedKeepAll), T,
        kSrsPolishedMaxIters);
  });

  // Rescue gate: nothing in-limits at a reachable target -> singular pose.
  if (in_limits.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    in_limits = finalize_solutions<7>(rescue_via_T_perturbation<7>(core, c, T), c, lim, p_limits);
  }

  ArtifactParams<7> p_seed = p;
  p_seed.respect_limits = false;
  return finalize_solutions<7>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
