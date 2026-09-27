// Same-root dedup (#600), the C++ twin of ssik.refinement.dedup_same_root.
//
// A solution is a distinct exact root. Two FK-closing candidates merge only
// when they approximate the SAME root, never merely because they are close:
// near a fold, twin roots a fraction of a milliradian apart are both kept.
//
// The test evaluates FK at the pair's wrap-aware midpoint. Two approximations
// of one regular root see FK as affine across the segment, so the midpoint
// residual is at most the mean endpoint residual plus a second-order term
// (r_a + r_b bounds both). Two distinct roots each close FK, but the segment
// between them leaves the solution set: the midpoint residual grows with the
// separation squared (~2e-8 for Puma's fold twins 7.6e-4 rad apart, #597).
//
// The gate (policy.subproblem_dedup) is only a pre-filter: farther pairs are
// distinct without an FK evaluation. The survivor of a merge is the first-seen
// candidate unless a later one closes FK better by more than round-off.
#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/ik_types.hpp"

namespace ssik {

// FK round-off level for residuals against T: 64 eps (1 + ||T||_F), matching
// ssik.refinement.same_root_floor.
inline double same_root_floor(const Pose& T) {
  return 64.0 * std::numeric_limits<double>::epsilon() * (1.0 + T.norm());
}

template <int N>
inline bool is_same_root(const JointConsts<N>& c, const Pose& T, const Solution<N>& a,
                         const Solution<N>& b, double floor) {
  constexpr double kPi = 3.14159265358979323846;
  std::array<double, N> mid;
  for (int i = 0; i < N; ++i) {
    // Python's ((b - a + pi) % 2pi) - pi: floored modulo, result in [-pi, pi).
    double d = std::fmod(b.q[i] - a.q[i] + kPi, 2.0 * kPi);
    if (d < 0) d += 2.0 * kPi;
    mid[i] = a.q[i] + 0.5 * (d - kPi);
  }
  const double r_mid = (fk<N>(c, mid) - T).norm();
  return r_mid <= a.fk_residual + b.fk_residual + floor;
}

template <int N>
inline std::vector<Solution<N>> dedup_same_root(const std::vector<Solution<N>>& cands,
                                                const JointConsts<N>& c, const Pose& T,
                                                double gate) {
  constexpr double kPi = 3.14159265358979323846;
  const double floor = same_root_floor(T);
  std::vector<Solution<N>> out;
  for (const auto& cand : cands) {
    int match = -1;
    for (std::size_t j = 0; j < out.size() && match < 0; ++j) {
      bool close = true;
      for (int i = 0; i < N; ++i) {
        double d = std::fmod(cand.q[i] - out[j].q[i] + kPi, 2.0 * kPi);
        if (d < 0) d += 2.0 * kPi;
        if (std::abs(d - kPi) > gate) {
          close = false;
          break;
        }
      }
      if (close && is_same_root<N>(c, T, cand, out[j], floor)) match = static_cast<int>(j);
    }
    if (match < 0)
      out.push_back(cand);
    else if (cand.fk_residual < out[match].fk_residual - floor)
      out[match] = cand;
  }
  return out;
}

}  // namespace ssik
