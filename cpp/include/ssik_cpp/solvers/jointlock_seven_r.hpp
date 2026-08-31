// jointlock.seven_r native artifact solve (#491): universal 7R IK by locking one
// redundant joint and sweeping it over a fixed sample schedule, solving the
// resulting 6R sub-chain per sample. For the RR-covered arms (rizon4, rizon10)
// every locked sub-chain is a general_6r (Raghavan-Roth) problem, so this reuses
// the RR runtime (general_6r_core) per sample -- no Study-quaternion kernel.
//
// Everything per-sample is baked at emit time: the lock similarity transform
// depends only on the (fixed) sample angle, so each of the N samples has a fixed
// 6R sub-chain whose DH bridge + RR coefficients are emitted like any RR arm.
// The runtime is a thin sweep: solve each sample's sub-chain for T, pad the 6-vec
// with the locked angle, re-verify against the 7R target, dedup, finalize. No
// rescue: an arm is only emitted when its RR-only sweep provably covers the
// oracle, so the rescue would only mask a gap (#535).
#pragma once

#include <array>
#include <vector>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/solvers/general_6r.hpp"

namespace ssik {

// Baked sweep geometry: which joint is locked and the sample schedule.
template <int NSamples>
struct JointlockConsts {
  int lock_idx = 0;
  std::array<double, NSamples> q_lock{};
};

namespace jointlock_detail {

// Wrap-to-pi max-joint dedup for 7-DOF (mirrors general_6r's 6-DOF version),
// keeping the lower-residual duplicate.
inline std::vector<Solution<7>> dedup7(const std::vector<Solution<7>>& cands, double atol) {
  constexpr double kPi = 3.14159265358979323846;
  std::vector<Solution<7>> out;
  for (const auto& cand : cands) {
    int dup = -1;
    for (std::size_t j = 0; j < out.size() && dup < 0; ++j) {
      double worst = 0.0;
      for (int i = 0; i < 7; ++i) {
        double d = std::fmod(cand.q[i] - out[j].q[i] + kPi, 2.0 * kPi);
        if (d < 0) d += 2.0 * kPi;
        worst = std::max(worst, std::abs(d - kPi));
      }
      if (worst < atol) dup = static_cast<int>(j);
    }
    if (dup < 0)
      out.push_back(cand);
    else if (cand.fk_residual < out[dup].fk_residual)
      out[dup] = cand;
  }
  return out;
}

}  // namespace jointlock_detail

// Full artifact-contract solve for an RR-covered jointlock.seven_r arm. All
// geometry baked: JointConsts<7> (POE FK for finalize/rescue) + JointlockConsts
// + per-sample RrConsts + coeff fns + JointLimits<7>. No Python, no HP kernel.
template <int NSamples, class CoeffFns>
std::vector<Solution<7>> jointlock_artifact_solve(const JointConsts<7>& c,
                                                  const JointlockConsts<NSamples>& jl,
                                                  const std::array<RrConsts, NSamples>& rr,
                                                  const CoeffFns& coeffs, const JointLimits<7>& lim,
                                                  const Pose& T, const ArtifactParams<7>& p) {
  // core: sweep the locked joint, RR-solve each 6R sub-chain, pad to 7-DOF.
  const auto core = [&](const Pose& tp) {
    std::vector<Solution<7>> all;
    // Sub-chain JointConsts (POE) is only touched by the 6R refiner, which is off
    // here (the locked sub-chains solve cleanly for the RR-covered arms); a
    // default suffices. If a degenerate sub-chain ever needs the 6R polish, the
    // emitter bakes the 16 sub-chain JointConsts and this passes them.
    static const JointConsts<6> kNoRefineConsts{};
    for (int i = 0; i < NSamples; ++i) {
      const auto sub = general_6r_core(kNoRefineConsts, rr[i], coeffs[i], tp, kGeneral6rFkAtol,
                                       kGeneral6rDedupAtol, /*allow_refinement=*/false, 15);
      for (const auto& s6 : sub) {
        Solution<7> s7;
        int j = 0;
        for (int k = 0; k < 7; ++k) s7.q[k] = (k == jl.lock_idx) ? jl.q_lock[i] : s6.q[j++];
        // Re-verify against the 7R target (not the sub-chain's DH residual): at
        // a degenerate lock sample (q_lock making adjacent axes parallel, e.g.
        // the identity-rotation sample) poe_to_dh of the sub-chain is slightly
        // inconsistent, so the sub-chain DH-FK closes but the padded 7R FK does
        // not. Python's jointlock drops these via the same re-verify (#491).
        s7.fk_residual = (fk<7>(c, s7.q) - tp).norm();
        if (s7.fk_residual > kGeneral6rFkAtol) continue;
        s7.refinement = s6.refinement;
        all.push_back(s7);
      }
    }
    return jointlock_detail::dedup7(all, kGeneral6rDedupAtol);
  };

  // NO rescue (#535): jointlock is a tier-1 SAMPLING solver -- its completeness
  // model is "the 16 sampled lock slices", and the emitter only ships an arm
  // whose RR-only sweep provably covers the full oracle (direct-completeness
  // check). The T-perturbation rescue finds OFF-sample solutions, which both blur
  // that model and (as seen on kassow) silently MASK a genuinely-incomplete sweep
  // by doing the HP kernel's job. Omitting it means any incompleteness fails
  // loudly at the gate instead of being hidden. HP-needing arms (kassow) are
  // deferred to the Study-quaternion kernel, not rescued into looking complete.
  ArtifactParams<7> p_limits;
  p_limits.respect_limits = p.respect_limits;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  std::vector<Solution<7>> in_limits = finalize_solutions<7>(core(T), c, lim, p_limits);
  ArtifactParams<7> p_seed = p;
  p_seed.respect_limits = false;
  return finalize_solutions<7>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
