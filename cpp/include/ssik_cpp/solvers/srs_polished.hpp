// Self-contained approximate-SRS 7R artifact solve (#550): the C++ replica of the
// generated <arm>_ik.solve() for the seven_r.srs_polished family (gen3, JACO
// j2s7s300, rm75, yumi L/R). These arms are SRS up to a small axis drift (<= 4cm),
// so the exact SRS core returns cm-off algebraic candidates that an LM polish
// against the TRUE FK corrects. Mirrors ssik.solvers.seven_r.srs_polished.solve:
//
//   1. raw candidates from the SRS core that matches the arm's axes (general
//      Davenport for every shipped arm), with reach_slack = 2*max_drift and a
//      keep-all FK threshold (srs.solve(reach_slack=..., fk_atol=10.0)); the
//      canonical sweep's candidates only as a fallback when those polish to
//      nothing (#598).
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
#include "ssik_cpp/solvers/srs_general.hpp"

namespace ssik {

// srs_polished tuning (ssik.solvers.seven_r.srs_polished defaults).
inline constexpr double kSrsPolishedMaxDrift = 0.04;      // _DEFAULT_MAX_DRIFT_M
inline constexpr double kSrsPolishedReachSlack = 0.08;    // 2 * max_drift
inline constexpr double kSrsPolishedFkAtol = 1e-12;       // polish_fk_atol accept
inline constexpr double kSrsPolishedKeepAll = 1e9;        // fk_atol=10.0 -> keep every candidate
inline constexpr int kSrsPolishedMaxIters = 30;          // polish_max_iters (cm-off seeds need it)

namespace srs_polished_detail {

// In-limits acceptance slack for polished candidates (_polish._LIMIT_SLACK).
inline constexpr double kPolishLimitSlack = 1e-9;

// LM-polish every raw candidate against the true FK, keep residual <= atol, then
// wrap-to-pi cluster-merge (mirrors _polish.polish_candidates, which uses
// lm_refine_batch's tighter divergence guard 2.0/2 -- passed explicitly so a seed
// stays on its local branch instead of wandering across the redundant manifold to
// a different solution, which would both miss the oracle's branch and add a dup).
// With `limits`, a polished candidate outside them is dropped before the merge
// (_polish._within_limits): the polish moves a seed, so an in-limits seed can
// land out of limits (#621).
inline std::vector<Solution<7>> polish(
    const JointConsts<7>& c, const std::vector<Solution<7>>& raw, const Pose& T, int max_iters,
    const std::array<std::array<double, 2>, 7>* limits = nullptr) {
  std::vector<Solution<7>> polished;
  for (const auto& cand : raw) {
    auto r = lm_refine<7>(c, cand.q, T, kSrsPolishedFkAtol, max_iters, /*divergence_factor=*/2.0,
                          /*divergence_min_iters=*/2, /*fixed_damping=*/1e-9);
    if (!r || r->second > kSrsPolishedFkAtol) continue;
    if (limits) {
      bool within = true;
      for (int i = 0; i < 7 && within; ++i)
        within = (*limits)[i][0] - kPolishLimitSlack <= r->first[i] &&
                 r->first[i] <= (*limits)[i][1] + kPolishLimitSlack;
      if (!within) continue;
    }
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

  // Seeded numerical-tracking fast path (#380), as in srs.hpp: Newton-continue
  // from q_seed and keep it only if it converges AND stays near the seed
  // (seeded_track's max_dist guard rejects a branch jump). Without this the
  // approximate-SRS arms fall back to the sampled swivel manifold, whose nearest
  // sample can sit radians from the seed -- bad for trajectory tracking (#562).
  // No in-limits fallback: a tracked seed failing limits/tolerance falls through.
  if (p.has_seed && p.max_solutions == 1) {
    const auto tracked = seeded_track<7>(c, p.q_seed, T);
    if (tracked) {
      const auto fast = finalize_solutions<7>({*tracked}, c, lim, p);
      if (!fast.empty()) return fast;
    }
  }

  // core: exact SRS candidates (reach-slackened, keep-all) -> LM-polish -> dedup.
  // The extraction follows the arm's axes (s.general_path, #598); when those
  // geometric seeds polish to nothing, the canonical sweep's candidates are a
  // fixed multistart fallback (mirrors srs_polished.solve).
  const auto core = [&](const Pose& Tp) {
    if (s.general_path) {
      auto sols = srs_polished_detail::polish(
          c, srs_general_solve(c, s, Tp, kSrsPolishedReachSlack, kSrsPolishedKeepAll), Tp,
          kSrsPolishedMaxIters);
      if (!sols.empty()) return sols;
    }
    const auto raw =
        srs_canonical_solve(c, s, Tp, kSrsPolishedReachSlack, kSrsPolishedKeepAll);
    return srs_polished_detail::polish(c, raw, Tp, kSrsPolishedMaxIters);
  };

  // Limit pass + #359 in-limits fallback (the SRS swivel resolver, wired for
  // srs_polished by codegen). Its exact-geometry solutions are cm-off for these
  // approximate arms, so they are polished, and the polished vectors are then
  // held to the limits: finalize trusts the fallback's output to be in limits
  // and does not filter it again (_swivel_limits.resolve_in_limits, #621).
  ArtifactParams<7> p_limits;
  // Intermediate pass: never lift here, or the lifts would be produced
  // twice (see ArtifactParams::enumerate_windings).
  p_limits.enumerate_windings = false;
  p_limits.respect_limits = p.respect_limits;
  p_limits.wrap_only = p.wrap_only;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  std::vector<Solution<7>> in_limits = finalize_solutions<7>(core(T), c, lim, p_limits, [&]() {
    return srs_polished_detail::polish(
        c, srs_swivel::resolve_in_limits(c, s, T, limits, kSrsPolishedKeepAll), T,
        kSrsPolishedMaxIters, &limits);
  });

  // Rescue gate: nothing in-limits at a reachable target -> singular pose.
  if (in_limits.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    in_limits = finalize_solutions<7>(rescue_via_T_perturbation<7>(core, c, T), c, lim, p_limits);
  }

  ArtifactParams<7> p_seed = p;
  p_seed.respect_limits = false;
  // The single lifting stage (#562): this is the call that yields the returned
  // set. Skipped when the caller wanted the raw geometric set.
  p_seed.enumerate_windings = p.enumerate_windings && p.respect_limits;
  return finalize_solutions<7>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
