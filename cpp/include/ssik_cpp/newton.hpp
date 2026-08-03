// Shared Newton-Raphson polish (#496 Phase 1), ported verbatim from
// ssik.refinement.lm_refine + se3_log_residual + kinbody_jacobian. SE(3)-log
// Newton steps with a Frobenius convergence gate; divergence/stall guards drop
// extraneous algebraic roots. Arm-agnostic primitive (ADR-0001's "one Newton"):
// every geometric solver's opt-in refinement routes through here.
#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"

namespace ssik {

// 6-vector SE(3) log residual (translation, rotation axis-angle) for the error
// matrix T_err = T_target * FK(q)^-1. Matches ssik.refinement.se3_log_residual
// (the #199 antisymmetric-vee + atan2 formulation, precise at every scale).
inline Eigen::Matrix<double, 6, 1> se3_log_residual(const Pose& t_err) {
  const Eigen::Vector3d trans = t_err.block<3, 1>(0, 3);
  const Eigen::Matrix3d r = t_err.block<3, 3>(0, 0);
  const Eigen::Vector3d skew(0.5 * (r(2, 1) - r(1, 2)), 0.5 * (r(0, 2) - r(2, 0)),
                             0.5 * (r(1, 0) - r(0, 1)));
  const double sin_angle = skew.norm();
  const double cos_angle = std::max(-1.0, std::min(1.0, 0.5 * (r.trace() - 1.0)));
  const double angle = std::atan2(sin_angle, cos_angle);

  Eigen::Vector3d rot;
  if (sin_angle > 1e-9) {
    rot = (angle / sin_angle) * skew;
  } else if (cos_angle > 0.0) {
    rot = skew;  // near identity: skew ~ angle * axis to leading order
  } else {
    // Near pi: recover axis from the dominant column of R + I.
    const Eigen::Matrix3d rpi = r + Eigen::Matrix3d::Identity();
    int idx = 0;
    double best = -1.0;
    for (int cc = 0; cc < 3; ++cc) {
      const double nc = rpi.col(cc).norm();
      if (nc > best) {
        best = nc;
        idx = cc;
      }
    }
    rot = (angle / best) * rpi.col(idx);
  }
  Eigen::Matrix<double, 6, 1> out;
  out.head<3>() = trans;
  out.tail<3>() = rot;
  return out;
}

// Closed-form 6xN spatial Jacobian for a POE JointConsts, matching
// ssik.refinement.kinbody_jacobian: column i is (p_i x z_i ; z_i) in the world
// frame at q. Eigen 4x4 matmuls (not the scalar-inlined Python body) -- equal to
// ~1e-15, far below any Newton tolerance.
template <int N>
Eigen::Matrix<double, 6, N> spatial_jacobian(const JointConsts<N>& c,
                                             const std::array<double, N>& q) {
  Eigen::Matrix<double, 6, N> jac;
  Pose t_acc = Pose::Identity();
  for (int i = 0; i < N; ++i) {
    const Eigen::Vector3d axis = c.axis[i].normalized();
    const Pose p = t_acc * c.t_left[i];
    const Eigen::Vector3d z = p.block<3, 3>(0, 0) * axis;
    const Eigen::Vector3d origin = p.block<3, 1>(0, 3);
    jac.col(i).template head<3>() = origin.cross(z);
    jac.col(i).template tail<3>() = z;

    Pose joint = Pose::Identity();
    if (c.type[i] == JointType::Prismatic) {
      joint.block<3, 1>(0, 3) = q[i] * axis;
    } else {
      joint.block<3, 3>(0, 0) = Eigen::AngleAxisd(q[i], axis).toRotationMatrix();
    }
    t_acc = p * joint * c.t_right[i];
  }
  return jac;
}

// Newton polish. Returns (q_refined, fk_residual) on convergence, nullopt if
// max_iters was hit without ||fk(q) - T||_F < fk_atol or a guard fired. Mirrors
// ssik.refinement.lm_refine (step clip, divergence + stall guards, Tikhonov
// fallback only on an exactly-singular Jacobian -- the parity of np.linalg.solve
// raising LinAlgError).
template <int N>
std::optional<std::pair<std::array<double, N>, double>> lm_refine(
    const JointConsts<N>& c, const std::array<double, N>& q_seed, const Pose& t_target,
    double fk_atol = 1e-9, int max_iters = 15) {
  constexpr double step_clip = 0.5;
  constexpr double divergence_factor = 5.0;
  constexpr int divergence_min_iters = 4;
  constexpr int stall_patience = 5;

  std::array<double, N> q = q_seed;
  double r_best = std::numeric_limits<double>::infinity();
  int stalled = 0;

  for (int it = 0; it < max_iters; ++it) {
    const Pose t_q = fk<N>(c, q);
    const double fro = (t_q - t_target).norm();
    if (fro < fk_atol) return std::make_pair(q, fro);
    if (fro < r_best) {
      r_best = fro;
      stalled = 0;
    } else {
      ++stalled;
      if (it >= divergence_min_iters && fro > divergence_factor * r_best) return std::nullopt;
      if (stalled >= stall_patience) return std::nullopt;
    }

    const Eigen::Matrix<double, 6, 1> res = se3_log_residual(t_target * t_q.inverse());
    const Eigen::Matrix<double, 6, N> js = spatial_jacobian<N>(c, q);

    Eigen::Matrix<double, N, 1> dq;
    bool use_tikhonov = true;
    if constexpr (N == 6) {
      const Eigen::Matrix<double, 6, 6> jsq = js;
      // np.linalg.solve raises only on an exactly-singular matrix; a nonzero
      // determinant takes the direct (LU) path, matching Python.
      if (jsq.determinant() != 0.0) {
        dq = jsq.partialPivLu().solve(res);
        use_tikhonov = false;
      }
    }
    if (use_tikhonov) {
      const double damping = std::max(1e-9, 1e-6 * fro);
      const Eigen::Matrix<double, N, N> jtj =
          js.transpose() * js + damping * Eigen::Matrix<double, N, N>::Identity();
      dq = jtj.ldlt().solve(js.transpose() * res);
    }

    for (int i = 0; i < N; ++i) {
      const double d = std::max(-step_clip, std::min(step_clip, dq[i]));
      q[i] += d;
    }
  }

  // Final convergence check after max_iters.
  const Pose t_check = fk<N>(c, q);
  const double final_fro = (t_check - t_target).norm();
  if (final_fro > fk_atol) return std::nullopt;
  return std::make_pair(q, final_fro);
}

}  // namespace ssik
