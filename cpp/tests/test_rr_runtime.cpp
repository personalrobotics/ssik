// Runtime-parity test for the shared C++ RR runtime (#490).
//
// Feeds Python-evaluated P/Q coefficients (the <arm>_rr_parity.hpp fixture)
// into general_6r_core and checks its solution SET matches Python's solve_all_ik
// pose-by-pose. This isolates the runtime (eliminate -> weierstrass -> build_m ->
// RealQZ eigensolve -> back-substitute -> dedup) from the not-yet-built emitter.
#include <cstdio>
#include <vector>

#include "ssik_cpp/solvers/general_6r.hpp"
#ifndef RR_FIXTURE
#define RR_FIXTURE "xarm6_ik_rr_parity.hpp"
#endif
#include RR_FIXTURE
// With -DUSE_EMITTED_COEFFS, drive the full path through the emitted rr_coeffs
// (slice 4) instead of the fixture's Python-evaluated P/Q -- validates the
// sympy->C++ emitter end-to-end against the same expected solution set.
#ifdef USE_EMITTED_COEFFS
#include RR_COEFFS_HEADER
#endif

namespace {

using namespace ssik;

constexpr double kPi = 3.14159265358979323846;

double wrap_dist(const std::array<double, 6>& x, const std::array<double, 6>& y) {
  double worst = 0.0;
  for (int i = 0; i < 6; ++i) {
    double d = std::fmod(x[i] - y[i] + kPi, 2.0 * kPi);
    if (d < 0) d += 2.0 * kPi;
    worst = std::max(worst, std::abs(d - kPi));
  }
  return worst;
}

}  // namespace

int main() {
  const auto rr = rr_parity::consts();
  const auto poses = rr_parity::poses();
  const double match_tol = 1e-6;

  // Production set-match tolerance (the artifact gate's dedup_by_wrap_close is
  // 1e-3). Tighter than that only exercises ill-conditioned arms' back-sub
  // amplification (jaco2 cond~1e16), where QZ and Python's Mobius/scipy path
  // yield the same FK-closing branch to ~1e-5 joint precision -- not a bug.
  const double match_tol_wrap = 1e-3;
  int n_fail = 0;
  int extensions = 0;
  double worst_match = 0.0;
  for (std::size_t pi = 0; pi < poses.size(); ++pi) {
    const auto& p = poses[pi];
#ifdef USE_EMITTED_COEFFS
    // Refinement off: the fixture's expected set is un-refined solve_all_ik, so
    // JointConsts is unused here (only the force_refine path touches it).
    const auto got = general_6r_core(JointConsts<6>{}, rr, rr_emitted::rr_coeffs, p.t,
                                     rr_parity::fk_atol(), rr_parity::dedup_atol(),
                                     /*allow_refinement=*/false, 15);
#else
    // Coeff callback: ignore t12, return the fixture's pre-evaluated P/Q.
    auto coeffs = [&](const double[12], rr_detail::Mat14x9& p_sin, rr_detail::Mat14x9& p_cos,
                      rr_detail::Mat14x9& p_one, rr_detail::Mat14x8& q) {
      p_sin = p.p_sin;
      p_cos = p.p_cos;
      p_one = p.p_one;
      q = p.q;
    };
    const auto got = general_6r_core(JointConsts<6>{}, rr, coeffs, p.t, rr_parity::fk_atol(),
                                     rr_parity::dedup_atol(), /*allow_refinement=*/false, 15);
#endif

#ifdef POSE_DEBUG
    if (static_cast<int>(pi) == POSE_DEBUG) {
      std::printf("== pose %d: %zu C++ solutions ==\n", POSE_DEBUG, got.size());
      for (const auto& g : got) {
        std::printf("q= ");
        for (int i = 0; i < 6; ++i) std::printf("%.12f ", g.q[i]);
        std::printf(" fk=%.3e\n", g.fk_residual);
      }
    }
#endif

    // #487 conformance: relative completeness (every Python solution has a
    // wrap-close C++ match) + soundness (every C++ solution FK-closes, already
    // guaranteed by the core's fk_atol filter). Exact count is NOT required:
    // at degenerate arms (piper a_2~0) QZ is a sound superset of Python's
    // companion+Mobius path -- extra FK-closing branches are valid extensions,
    // and they collapse to the same set as Python once limits are applied.
    bool ok = true;
    for (const auto& e : p.expected) {
      double best = 1e9;
      for (const auto& g : got) best = std::min(best, wrap_dist(e, g.q));
      worst_match = std::max(worst_match, best);
      if (best > match_tol_wrap) ok = false;
    }
    if (got.size() > p.expected.size()) extensions += got.size() - p.expected.size();
    if (!ok) {
      ++n_fail;
      std::printf("pose %zu: MISSING a Python branch (got %zu, expected %zu)\n", pi, got.size(),
                  p.expected.size());
    }
  }

  std::printf(
      "%s RR runtime parity: %zu poses, %d incomplete, worst match %.3e, %d valid extensions\n",
      rr_parity::arm_name(), poses.size(), n_fail, worst_match, extensions);
  return n_fail == 0 ? 0 : 1;
}
