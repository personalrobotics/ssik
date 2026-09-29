// Machine-precision polish of an accepted candidate (#632): the native port of
// ssik.refinement.polish.polish_accepted, whose module docstring owns the
// definition ("One definition, two backends"). A change on one side must be
// mirrored on the other.
//
// A candidate that already passed the acceptance gate is Newton-polished on the
// true FK. The polished point replaces it only if the residual strictly improved
// to kPolishTarget and the point stayed within twice the first step of the
// candidate (the Newton-Kantorovich ball, so it cannot have crossed to another
// branch); otherwise the candidate is left exactly as it was. The step is the
// rescue's (lm_refine_batch's): clip((J^T J + 1e-9 I)^-1 J^T log(T FK(q)^-1),
// +-0.5), solved by partial-pivot LU as numpy.linalg.solve does.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <Eigen/Dense>

#include "ssik_cpp/dedup.hpp"   // same_root_floor
#include "ssik_cpp/newton.hpp"  // se3_log_residual

namespace ssik {

// ssik.refinement.polish.POLISH_TARGET / POLISH_MAX_ITERS.
inline constexpr double kPolishTarget = 1e-12;
inline constexpr int kPolishMaxIters = 4;

// fk: (const std::array<double, N>&) -> Pose; jac: same -> Eigen::Matrix<double, 6, N>
// (the spatial Jacobian, columns (p_i x z_i ; z_i)), both in the frame of t_target.
// On acceptance overwrites q and r with the polished point and its Frobenius
// residual and returns true; otherwise leaves both untouched and returns false.
template <int N, class FkFn, class JacFn>
bool polish_accepted(FkFn&& fk, JacFn&& jac, const Pose& t_target, std::array<double, N>& q,
                     double& r) {
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
  q = best_q;
  r = best_r;
  return true;
}

}  // namespace ssik
