// Artifact-layer post-processing (#503), ported bit-for-bit from
// ssik.postprocess.finalize_solutions + the <arm>_ik.solve() wrapper. This is
// the shared tail every shipped Python artifact runs after the core solve:
// limits (wrap-into-range then drop) -> seed (tolerance filter then rank) ->
// truncate. The C++ replica must match the artifact, so every ordering rule,
// inclusive/strict inequality, and wrap convention is reproduced exactly.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/ik_types.hpp"

namespace ssik {

// Per-joint limits alongside JointConsts (JointConsts carries axes/frames/type;
// limits live here since only the artifact layer needs them).
template <int N>
struct JointLimits {
  std::array<double, N> lo{};
  std::array<double, N> hi{};
  std::array<bool, N> present{};  // false => joint has no limits (never rejects)
};

// Seed-ranking metric selector, matching seed_metric="wrap_linf"|"wrap_l2".
enum class SeedMetric { WrapLinf, WrapL2 };

// Full artifact solve() parameters (defaults match codegen.py:507-670).
template <int N>
struct ArtifactParams {
  bool respect_limits = true;
  bool has_seed = false;
  std::array<double, N> q_seed{};
  SeedMetric seed_metric = SeedMetric::WrapLinf;
  bool has_seed_tolerance = false;
  double seed_tolerance = 0.0;
  int max_solutions = -1;  // -1 => None (no cap)
  bool allow_rescue = true;
  int refinement_max_iters = 15;
};

namespace finalize_detail {

// ((x + pi) mod 2pi) - pi, matching ssik.postprocess._wrap_to_pi exactly:
// Python's % (floored) yields [0, 2pi), so the result is [-pi, pi) -- +pi maps
// to -pi. std::fmod is truncated, so we correct negatives to reproduce it.
inline double wrap_to_pi(double x) {
  double m = std::fmod(x + M_PI, 2.0 * M_PI);
  if (m < 0.0) m += 2.0 * M_PI;
  return m - M_PI;
}

}  // namespace finalize_detail

// wrap_to_limits: for each revolute joint with limits, try wraps k in
// (1,-1,2,-2) to bring q_i into [lo,hi] inclusive; first hit wins. Prismatic /
// no-limits / already-in-range joints untouched. (postprocess.py:101-151)
template <int N>
std::vector<Solution<N>> wrap_to_limits(const std::vector<Solution<N>>& sols,
                                        const JointConsts<N>& consts,
                                        const JointLimits<N>& lim) {
  static constexpr int kWrapOrder[4] = {1, -1, 2, -2};
  std::vector<Solution<N>> out;
  out.reserve(sols.size());
  for (const auto& sol : sols) {
    Solution<N> s = sol;  // copy; q gets adjusted in place
    for (int i = 0; i < N; ++i) {
      if (!lim.present[i] || consts.type[i] != JointType::Revolute) continue;
      const double lo = lim.lo[i], hi = lim.hi[i];
      const double qi = sol.q[i];
      if (lo <= qi && qi <= hi) continue;
      for (int k : kWrapOrder) {
        const double cand = qi + 2.0 * M_PI * k;
        if (lo <= cand && cand <= hi) {
          s.q[i] = cand;
          break;
        }
      }
    }
    out.push_back(s);
  }
  return out;
}

// respect_limits: drop a solution if any limited joint has q_i < lo or q_i > hi
// (strict -> boundary values accepted); no wrapping. (postprocess.py:69-98)
template <int N>
std::vector<Solution<N>> apply_respect_limits(const std::vector<Solution<N>>& sols,
                                              const JointLimits<N>& lim) {
  std::vector<Solution<N>> out;
  for (const auto& sol : sols) {
    bool within = true;
    for (int i = 0; i < N; ++i) {
      if (!lim.present[i]) continue;
      if (sol.q[i] < lim.lo[i] || sol.q[i] > lim.hi[i]) {
        within = false;
        break;
      }
    }
    if (within) out.push_back(sol);
  }
  return out;
}

// within_seed_tolerance: keep iff max_i |wrap(q_i - seed_i)| <= tol (Linf,
// inclusive). (postprocess.py:198-228)
template <int N>
std::vector<Solution<N>> within_seed_tolerance(const std::vector<Solution<N>>& sols,
                                               const std::array<double, N>& seed, double tol) {
  std::vector<Solution<N>> out;
  for (const auto& sol : sols) {
    bool ok = true;
    for (int i = 0; i < N; ++i) {
      if (std::abs(finalize_detail::wrap_to_pi(sol.q[i] - seed[i])) > tol) {
        ok = false;
        break;
      }
    }
    if (ok) out.push_back(sol);
  }
  return out;
}

// nearest_to_seed: STABLE sort by distance to seed. wrap_l2 = sqrt(sum d_i^2),
// wrap_linf = max |d_i|, d_i = wrap(q_i - seed_i). (postprocess.py:159-195)
template <int N>
std::vector<Solution<N>> nearest_to_seed(std::vector<Solution<N>> sols,
                                         const std::array<double, N>& seed, SeedMetric metric) {
  auto dist = [&](const Solution<N>& sol) {
    if (metric == SeedMetric::WrapL2) {
      double acc = 0.0;
      for (int i = 0; i < N; ++i) {
        const double d = finalize_detail::wrap_to_pi(sol.q[i] - seed[i]);
        acc += d * d;
      }
      return std::sqrt(acc);
    }
    double m = 0.0;
    for (int i = 0; i < N; ++i)
      m = std::max(m, std::abs(finalize_detail::wrap_to_pi(sol.q[i] - seed[i])));
    return m;
  };
  std::stable_sort(sols.begin(), sols.end(),
                   [&](const Solution<N>& a, const Solution<N>& b) { return dist(a) < dist(b); });
  return sols;
}

// The finalize_solutions pipeline: limits -> seed(tolerance then rank) ->
// truncate, in that fixed order. (postprocess.py:255-301)
template <int N>
std::vector<Solution<N>> finalize_solutions(std::vector<Solution<N>> sols,
                                            const JointConsts<N>& consts,
                                            const JointLimits<N>& lim,
                                            const ArtifactParams<N>& p) {
  if (p.respect_limits) {
    sols = wrap_to_limits<N>(sols, consts, lim);
    sols = apply_respect_limits<N>(sols, lim);
  }
  if (p.has_seed) {
    if (p.has_seed_tolerance) sols = within_seed_tolerance<N>(sols, p.q_seed, p.seed_tolerance);
    sols = nearest_to_seed<N>(std::move(sols), p.q_seed, p.seed_metric);
  }
  if (p.max_solutions >= 0 && static_cast<int>(sols.size()) > p.max_solutions) {
    sols.resize(p.max_solutions);
  }
  return sols;
}

}  // namespace ssik
