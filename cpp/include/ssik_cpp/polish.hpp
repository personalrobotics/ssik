// Machine-precision polish of an accepted candidate (#632): the native port of
// ssik.refinement.polish.polish_accepted, whose module docstring owns the
// definition ("One definition, two backends"). A change on one side must be
// mirrored on the other.
//
// A candidate that already passed the acceptance gate is Newton-polished on the
// true FK. The polished point replaces it only if the residual strictly improved
// to kPolishTarget and the point stayed within twice the first step of the
// candidate (the Newton-Kantorovich ball, so it cannot have crossed to another
// branch) and kept every in-limit winding of the candidate (#644); otherwise the
// candidate is left exactly as it was. The step is the
// rescue's (lm_refine_batch's): clip((J^T J + 1e-9 I)^-1 J^T log(T FK(q)^-1),
// +-0.5), solved by partial-pivot LU as numpy.linalg.solve does.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Dense>

#include "ssik_cpp/dedup.hpp"     // same_root_floor
#include "ssik_cpp/finalize.hpp"  // JointLimits, kLimitBand
#include "ssik_cpp/newton.hpp"    // se3_log_residual

namespace ssik {

// ssik.refinement.polish.POLISH_TARGET / POLISH_MAX_ITERS.
inline constexpr double kPolishTarget = 1e-12;
inline constexpr int kPolishMaxIters = 4;

// Whether x1 has every in-limit winding that x0 has: on every joint with finite
// limits, x0_i + 2 pi k within kLimitBand of [lo_i, hi_i] implies the same for
// x1_i. `lim` is in the polished chain's coordinates. Mirrors
// ssik.refinement.polish.Chain.keeps_limit_windings.
template <int N>
bool keeps_limit_windings(const std::array<double, N>& x0, const std::array<double, N>& x1,
                          const JointLimits<N>& lim) {
  constexpr double kTwoPi = 2.0 * M_PI;
  for (int i = 0; i < N; ++i) {
    if (!lim.present[i] || !std::isfinite(lim.lo[i]) || !std::isfinite(lim.hi[i])) continue;
    const double lo = lim.lo[i] - kLimitBand;
    const double hi = lim.hi[i] + kLimitBand;
    // Every k with x0 + 2 pi k in [lo, hi] lies in [k_lo, k_hi]; one extra k on
    // each side absorbs the rounding of the division.
    const int k_lo = static_cast<int>(std::ceil((lo - x0[i]) / kTwoPi)) - 1;
    const int k_hi = static_cast<int>(std::floor((hi - x0[i]) / kTwoPi)) + 1;
    for (int k = k_lo; k <= k_hi; ++k) {
      const double v0 = x0[i] + kTwoPi * k;
      const double v1 = x1[i] + kTwoPi * k;
      if (v0 >= lo && v0 <= hi && !(v1 >= lo && v1 <= hi)) return false;
    }
  }
  return true;
}

// fk: (const std::array<double, N>&) -> Pose; jac: same -> Eigen::Matrix<double, 6, N>
// (the spatial Jacobian, columns (p_i x z_i ; z_i)), both in the frame of t_target;
// lim: the joint limits in the same coordinates, which the polish must not cross.
// On acceptance overwrites q and r with the polished point and its Frobenius
// residual and returns true; otherwise leaves both untouched and returns false.
template <int N, class FkFn, class JacFn>
bool polish_accepted(FkFn&& fk, JacFn&& jac, const Pose& t_target, const JointLimits<N>& lim,
                     std::array<double, N>& q, double& r) {
  constexpr double kStepClip = 0.5;
  constexpr double kDamping = 1e-9;
  const double floor = same_root_floor(t_target);

  Pose t_q = fk(q);
  const double r0 = (t_q - t_target).norm();
  if (!(r0 > floor)) return false;
  std::array<double, N> cur = q, best_q = q;
  double best_r = r0, eta = 0.0;
  for (int k = 0; k < kPolishMaxIters; ++k) {
    const Eigen::Matrix<double, 6, N> js = jac(cur);
    const Eigen::Matrix<double, 6, 1> res = se3_log_residual(t_target * t_q.inverse());
    const Eigen::Matrix<double, N, N> jtj =
        js.transpose() * js + kDamping * Eigen::Matrix<double, N, N>::Identity();
    const Eigen::Matrix<double, N, 1> dq = jtj.partialPivLu().solve(js.transpose() * res);
    double step2 = 0.0;
    for (int i = 0; i < N; ++i) {
      const double d = std::max(-kStepClip, std::min(kStepClip, dq[i]));
      step2 += d * d;
      cur[i] += d;
    }
    if (k == 0) eta = std::sqrt(step2);
    t_q = fk(cur);
    const double rk = (t_q - t_target).norm();
    if (!(rk < best_r)) break;  // stopped improving
    best_q = cur;
    best_r = rk;
    if (rk <= floor) break;  // at round-off
  }
  if (!(best_r <= kPolishTarget && best_r < r0)) return false;
  double moved2 = 0.0;
  for (int i = 0; i < N; ++i) moved2 += (best_q[i] - q[i]) * (best_q[i] - q[i]);
  if (!(std::sqrt(moved2) <= 2.0 * eta)) return false;
  if (!keeps_limit_windings<N>(q, best_q, lim)) return false;
  q = best_q;
  r = best_r;
  return true;
}

}  // namespace ssik
