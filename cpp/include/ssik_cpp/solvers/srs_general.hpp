// General concurrent-axis SRS-class 7R solve (#354), ported from the general
// (non-canonical) branch of ssik.solvers.seven_r.srs.solve. Covers any spherical
// 7R whose shoulder/wrist triples are not z-y-z and/or whose wrist is laterally
// offset (R1 Pro, OpenArm, iiwa7-class) -- the arms the canonical ZYZ fast-path
// (srs_canonical.hpp) cannot solve. Pure rotation algebra: sweep the elbow
// swivel; per swivel, SP4 fixes the elbow q3, an SP1 roll aligns the forearm, and
// the generalized-Euler decomposition extracts the shoulder + wrist triples. Up
// to 2 (q3) x 2 (shoulder) x 2 (wrist) = 8 candidates per swivel, FK-verified +
// deduped. Strict path only (reach_slack == 0); the singular/clamp handling is
// srs_polished's and is not needed here.
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/generalized_euler.hpp"
#include "ssik_cpp/rotation.hpp"
#include "ssik_cpp/seven_r/srs_swivel_limits.hpp"  // srs_swivel::min_rotation
#include "ssik_cpp/solvers/srs_canonical.hpp"      // SrsConsts, srs_detail::swivel_basis, kSrs*

namespace ssik {

namespace srs_general_detail {

// SP4 (h . Rot(k,q) p == delta) in the always-two-branches form matching
// _sp4_branches_batch: returns both roots (base+off, base-off) and a validity
// flag; the duplicate at off->0 is collapsed downstream by dedup.
struct Sp4Two {
  double qa, qb;
  bool valid;
};
inline Sp4Two sp4_two(const Eigen::Vector3d& h, const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                      double delta) {
  const double hk = h.dot(k), kp = k.dot(p);
  const double a = h.dot(p) - hk * kp;
  const double b = h.dot(k.cross(p));
  const double cc = delta - hk * kp;
  const double amp = std::hypot(a, b);
  bool valid = amp >= 1e-12;
  const double ratio = valid ? cc / amp : 2.0;
  valid = valid && std::abs(ratio) <= 1.0 + 1e-9;
  const double base = std::atan2(b, a);
  const double off = std::acos(std::max(-1.0, std::min(1.0, ratio)));
  return {base + off, base - off, valid};
}

}  // namespace srs_general_detail

// General-path SRS solve. Returns FK-verified, deduped 7-DOF solutions. Assumes a
// concurrent-axis (spherical shoulder + spherical wrist) SRS arm; geometry comes
// from the baked SrsConsts (classifier + _arm_constants).
inline std::vector<Solution<7>> srs_general_solve(const JointConsts<7>& c, const SrsConsts& s,
                                                  const Pose& T) {
  const Eigen::Matrix3d R_target = T.block<3, 3>(0, 0);
  const Eigen::Vector3d p_target = T.block<3, 1>(0, 3);
  const Eigen::Vector3d W_t = p_target - R_target * s.ee_offset_local;

  const Eigen::Vector3d SW = W_t - s.shoulder_pivot;
  const double d_sw = SW.norm();
  if (d_sw > s.l_se + s.l_ew || d_sw < std::abs(s.l_se - s.l_ew) || d_sw < 1e-12) return {};
  const Eigen::Vector3d u_sw = SW / d_sw;

  const double x_c =
      (s.l_se * s.l_se - s.l_ew * s.l_ew + d_sw * d_sw) / (2.0 * d_sw);
  const double r_circle = std::sqrt(std::max(s.l_se * s.l_se - x_c * x_c, 0.0));
  Eigen::Vector3d u_p1, u_p2;
  srs_detail::swivel_basis(u_sw, u_p1, u_p2);

  std::array<Eigen::Vector3d, 7> n;
  for (int i = 0; i < 7; ++i) n[i] = c.axis[i].normalized();
  const Eigen::Vector3d& n3 = n[s.elbow_index];
  const Eigen::Matrix3d M_wrist = R_target * s.r_post_wrist.transpose();
  const Eigen::Vector3d& S = s.shoulder_pivot;

  std::vector<Solution<7>> candidates;
  for (int m = 0; m < kSrsSwivelSamples; ++m) {
    const double sw = -M_PI + (2.0 * M_PI) * m / kSrsSwivelSamples;  // linspace endpoint=False
    const Eigen::Vector3d E_t = S + x_c * u_sw + r_circle * (std::cos(sw) * u_p1 + std::sin(sw) * u_p2);
    const Eigen::Vector3d d = (E_t - S) / s.l_se;
    const Eigen::Vector3d upper = E_t - S;
    const Eigen::Vector3d wrist_vec = W_t - E_t;

    const Eigen::Matrix3d r0 = srs_swivel::min_rotation(s.upper_home, upper);
    const Eigen::Vector3d k_elbow = r0 * n3;
    const Eigen::Vector3d v_forearm0 = r0 * s.forearm_home;
    const auto sp4 = srs_general_detail::sp4_two(d, k_elbow, v_forearm0, wrist_vec.dot(d));
    if (!sp4.valid) continue;
    const double dw = d.dot(wrist_vec);

    for (double q3 : {sp4.qa, sp4.qb}) {
      const Eigen::Vector3d g = r0 * (rotation_matrix(n3, q3) * s.forearm_home);
      const Eigen::Vector3d g_perp = g - d * d.dot(g);
      const Eigen::Vector3d w_perp = wrist_vec - d * dw;
      const double phi = g_perp.norm() < 1e-9
                             ? 0.0
                             : std::atan2(d.dot(g_perp.cross(w_perp)), g_perp.dot(w_perp));
      const Eigen::Matrix3d r_sh = rotation_matrix(d, phi) * r0;
      const Eigen::Matrix3d r_pre_elbow = r_sh * rotation_matrix(n3, q3);
      const Eigen::Matrix3d r_res = r_pre_elbow.transpose() * M_wrist;

      const auto sh = geuler::decompose_3axis(r_sh, n[0], n[1], n[2]);
      const auto wr = geuler::decompose_3axis(r_res, n[4], n[5], n[6]);
      if (sh.empty() || wr.empty()) continue;
      for (const auto& b_sh : sh) {
        for (const auto& b_wr : wr) {
          const std::array<double, 7> q = {b_sh.a, b_sh.b, b_sh.c, q3, b_wr.a, b_wr.b, b_wr.c};
          bool finite = true;
          for (double v : q)
            if (!std::isfinite(v)) finite = false;
          if (finite) candidates.push_back(Solution<7>{q, 0.0, Refinement::None});
        }
      }
    }
  }

  if (candidates.empty()) return {};

  // dedup_by_wrap_close (fk_residual == 0 pre-verify -> first-seen wins).
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
