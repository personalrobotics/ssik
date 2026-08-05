// Canonical-ZYZ SRS-class 7R analytical IK (#512), ported from the canonical
// path of ssik.solvers.seven_r.srs.solve (iiwa-class: z-y-z shoulder + z-y-z
// wrist, offset-free wrist). Singh-Kreutz swivel parameterization: sweep the
// elbow swivel, place the elbow on the reach circle (cosine rule), SP1 for the
// upper-arm roll q2, ZYZ-Euler the residual for the wrist triple.
//
// The general Davenport path (tilted/offset elbow, non-ZYZ axes) is deferred;
// those arms fall back to Python. SRS geometric constants are precomputed in the
// Python dispatch (classify + _arm_constants) and passed in via SrsConsts.
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/sp6.hpp"  // detail::wrap_pi
#include "ssik_cpp/subproblems.hpp"

namespace ssik {

// Precomputed SRS geometry (from the Python classifier + _arm_constants). The
// canonical solve uses l_se..r_post_wrist; the swivel-limits resolver (#515)
// additionally needs the elbow index + home upper/forearm vectors for branch
// enumeration. All baked at emit time.
struct SrsConsts {
  double l_se = 0.0;
  double l_ew = 0.0;
  Eigen::Vector3d ee_offset_local{0, 0, 0};  // ee_home - wrist_pivot (local)
  Eigen::Vector3d shoulder_pivot{0, 0, 0};
  Eigen::Matrix3d r_post_wrist = Eigen::Matrix3d::Identity();  // joints[6].T_right rotation
  int elbow_index = 3;
  Eigen::Vector3d upper_home{0, 0, 0};    // origins[elbow] - shoulder_pivot
  Eigen::Vector3d forearm_home{0, 0, 0};  // wrist_pivot - origins[elbow]
};

inline constexpr int kSrsSwivelSamples = 16;
inline constexpr double kSrsFkThreshold = 1e-5;   // policy.subproblem_numerical
inline constexpr double kSrsDedupTol = 1e-3;      // policy.subproblem_dedup

namespace srs_detail {

// Two orthonormal vectors spanning the plane perpendicular to u_sw
// (ssik.solvers.seven_r.srs._swivel_basis).
inline void swivel_basis(const Eigen::Vector3d& u_sw, Eigen::Vector3d& u_perp1,
                         Eigen::Vector3d& u_perp2) {
  const Eigen::Vector3d ref =
      std::abs(u_sw[2]) < 0.99 ? Eigen::Vector3d(0, 0, 1) : Eigen::Vector3d(1, 0, 0);
  u_perp1 = ref - ref.dot(u_sw) * u_sw;
  u_perp1.normalize();
  u_perp2 = u_sw.cross(u_perp1);
}

// Canonical-ZYZ shoulder: (q0, q1) aiming home upper arm (+z) onto elbow dir d
// (ssik.solvers.seven_r.srs._shoulder_angles_zyz).
inline void shoulder_angles_zyz(const Eigen::Vector3d& d, int q1_sign, double& q0, double& q1) {
  const double cos_q1 = std::max(-1.0, std::min(1.0, d[2]));
  q1 = q1_sign * std::acos(cos_q1);
  const double sin_q1 = std::sin(q1);
  q0 = 0.0;
  if (std::abs(sin_q1) > 1e-9) q0 = std::atan2(d[1] / sin_q1, d[0] / sin_q1);
}

}  // namespace srs_detail

// Canonical-path SRS solve. Returns FK-verified, deduped 7-DOF solutions.
// Assumes a canonical-ZYZ, offset-free-wrist SRS arm (the dispatch gates this).
inline std::vector<Solution<7>> srs_canonical_solve(const JointConsts<7>& c, const SrsConsts& s,
                                                    const Pose& T) {
  const Eigen::Matrix3d R_target = T.block<3, 3>(0, 0);
  const Eigen::Vector3d p_target = T.block<3, 1>(0, 3);
  const Eigen::Vector3d W_t = p_target - R_target * s.ee_offset_local;

  const Eigen::Vector3d SW = W_t - s.shoulder_pivot;
  const double d_sw = SW.norm();
  if (d_sw > s.l_se + s.l_ew || d_sw < std::abs(s.l_se - s.l_ew) || d_sw < 1e-12) return {};
  const Eigen::Vector3d u_sw = SW / d_sw;

  const double cos_int = std::max(
      -1.0, std::min(1.0, (s.l_se * s.l_se + s.l_ew * s.l_ew - d_sw * d_sw) / (2.0 * s.l_se * s.l_ew)));
  const double base_q3 = M_PI - std::acos(cos_int);
  const std::array<double, 2> q3_branches = {base_q3, -base_q3};

  const double x_c = (s.l_se * s.l_se - s.l_ew * s.l_ew + d_sw * d_sw) / (2.0 * d_sw);
  const double r_circle = std::sqrt(std::max(s.l_se * s.l_se - x_c * x_c, 0.0));
  Eigen::Vector3d u_perp1, u_perp2;
  srs_detail::swivel_basis(u_sw, u_perp1, u_perp2);

  const Eigen::Matrix3d R_post = s.r_post_wrist;
  const Eigen::Vector3d S = s.shoulder_pivot;

  // Per-swivel elbow placement.
  std::array<Eigen::Vector3d, kSrsSwivelSamples> E_t, d_dir;
  for (int n = 0; n < kSrsSwivelSamples; ++n) {
    const double sw = -M_PI + (2.0 * M_PI) * n / kSrsSwivelSamples;  // linspace endpoint=False
    E_t[n] = S + x_c * u_sw + r_circle * (std::cos(sw) * u_perp1 + std::sin(sw) * u_perp2);
    d_dir[n] = (E_t[n] - S) / s.l_se;
  }

  // Candidate order matches Python: q3 -> q1_sign -> q5_sign -> swivel.
  std::vector<Solution<7>> candidates;
  for (double q3 : q3_branches) {
    for (int q1_sign : {+1, -1}) {
      // Per-swivel (q0, q1, q2, R_res), independent of q5_sign.
      std::array<double, kSrsSwivelSamples> q0s, q1s, q2s;
      std::array<Eigen::Matrix3d, kSrsSwivelSamples> R_res;
      for (int n = 0; n < kSrsSwivelSamples; ++n) {
        const Eigen::Vector3d& d = d_dir[n];
        double q0, q1;
        srs_detail::shoulder_angles_zyz(d, q1_sign, q0, q1);

        std::array<double, 7> q_partial = {q0, q1, 0.0, q3, 0.0, 0.0, 0.0};
        const auto [R5, W_at_q2_zero] = frame_at_joint<7>(c, q_partial, 5);

        // SP1 for q2 (upper-arm roll mapping the q2=0 wrist pivot onto W_t).
        const Eigen::Vector3d p_from = W_at_q2_zero - E_t[n];
        const Eigen::Vector3d p_to = W_t - E_t[n];
        const double up_pf = d.dot(p_from);
        const double up_pt = d.dot(p_to);
        const double num = d.dot(p_from.cross(p_to));
        const double den = p_from.dot(p_to) - up_pf * up_pt;
        const double q2 = std::atan2(num, den);

        std::array<double, 7> q_post = {q0, q1, q2, q3, 0.0, 0.0, 0.0};
        const auto [R_pre_wrist, p4] = frame_at_joint<7>(c, q_post, 4);
        (void)p4;
        q0s[n] = q0;
        q1s[n] = q1;
        q2s[n] = q2;
        R_res[n] = R_pre_wrist.transpose() * R_target * R_post.transpose();
      }

      for (int q5_sign : {+1, -1}) {
        for (int n = 0; n < kSrsSwivelSamples; ++n) {
          const Eigen::Matrix3d& r = R_res[n];
          const double cos_q5 = std::max(-1.0, std::min(1.0, r(2, 2)));
          const double q5 = q5_sign * std::acos(cos_q5);
          const double sin_q5 = std::sin(q5);
          double q4, q6;
          if (std::abs(sin_q5) > 1e-9) {
            q4 = std::atan2(q5_sign * r(1, 2), q5_sign * r(0, 2));
            q6 = std::atan2(q5_sign * r(2, 1), q5_sign * -r(2, 0));
          } else {
            q4 = 0.0;  // gimbal lock
            q6 = cos_q5 > 0.0 ? std::atan2(-r(0, 1), r(0, 0)) : std::atan2(r(0, 1), -r(0, 0));
          }
          const std::array<double, 7> q = {q0s[n], q1s[n], q2s[n], q3, q4, q5, q6};
          bool finite = true;
          for (double v : q)
            if (!std::isfinite(v)) finite = false;
          if (finite) candidates.push_back(Solution<7>{q, 0.0, Refinement::None});
        }
      }
    }
  }

  if (candidates.empty()) return {};

  // dedup_by_wrap_close (all fk_residual == 0 pre-verify -> first-seen wins).
  std::vector<Solution<7>> deduped;
  for (const auto& cand : candidates) {
    bool dup = false;
    for (const auto& u : deduped) {
      bool close = true;
      for (int i = 0; i < 7; ++i) {
        if (std::abs(detail::wrap_pi(cand.q[i] - u.q[i])) > kSrsDedupTol) {
          close = false;
          break;
        }
      }
      if (close) {
        dup = true;
        break;
      }
    }
    if (!dup) deduped.push_back(cand);
  }

  // FK verify: fill residual, drop past threshold.
  std::vector<Solution<7>> verified;
  for (const auto& cand : deduped) {
    const double resid = (fk<7>(c, cand.q) - T).norm();
    if (resid <= kSrsFkThreshold) verified.push_back(Solution<7>{cand.q, resid, Refinement::None});
  }
  return verified;
}

}  // namespace ssik
