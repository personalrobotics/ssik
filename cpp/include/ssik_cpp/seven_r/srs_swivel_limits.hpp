// SRS-class 7R exact feasible-swivel joint-limit resolution (#359/#515), ported
// verbatim from ssik.solvers.seven_r._swivel_limits (exact path -- the
// approximate/polish path is for srs_polished, a different solver). Recovers the
// in-limits solution(s) the coarse swivel sweep misses: enumerate the <=8 IK
// branches, take each branch's closed-form q(psi), compute its feasible-swivel
// arcs (feasible_arcs on the 6 non-elbow joints), and return the arc-centre
// solution (max joint-limit margin), FK-verified. _Branch.q(psi) is the general
// concurrent-axis SRS solve (Davenport triple-phase), so this also handles the
// tilted/offset arms.
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/rotation.hpp"  // rotation_matrix
#include "ssik_cpp/seven_r/feasible_arcs.hpp"
#include "ssik_cpp/solvers/srs_canonical.hpp"  // SrsConsts, srs_detail::swivel_basis

namespace ssik {
namespace srs_swivel {

// Angle about unit k carrying a's perp component onto b's (_signed_angle).
inline double signed_angle(const Eigen::Vector3d& k, const Eigen::Vector3d& a,
                           const Eigen::Vector3d& b) {
  const Eigen::Vector3d ap = a - k * (k.dot(a));
  const Eigen::Vector3d bp = b - k * (k.dot(b));
  return std::atan2(k.dot(ap.cross(bp)), ap.dot(bp));
}

// (rho, delta, gamma) with m0 . Rot(m1,q) m2 == rho cos(q - delta) + gamma.
struct TriplePhase {
  double rho, delta, gamma;
};
inline TriplePhase triple_phase(const Eigen::Vector3d& m0, const Eigen::Vector3d& m1,
                                const Eigen::Vector3d& m2) {
  const double a = m0.dot(m2) - m0.dot(m1) * m1.dot(m2);
  const double b = m0.dot(m1.cross(m2));
  return {std::hypot(a, b), std::atan2(b, a), m0.dot(m1) * m1.dot(m2)};
}

// Minimal rotation carrying dir(u) onto dir(v) (_min_rotation).
inline Eigen::Matrix3d min_rotation(Eigen::Vector3d u, Eigen::Vector3d v) {
  u.normalize();
  v.normalize();
  const double c = u.dot(v);
  if (c > 1.0 - 1e-12) return Eigen::Matrix3d::Identity();
  if (c < -1.0 + 1e-12) {
    Eigen::Vector3d perp = u.cross(Eigen::Vector3d(1, 0, 0));
    if (perp.norm() < 1e-6) perp = u.cross(Eigen::Vector3d(0, 1, 0));
    return rotation_matrix(perp.normalized(), M_PI);
  }
  return rotation_matrix(u.cross(v).normalized(), std::acos(c));
}

// SP4: h . (Rot(k,q) p) == delta -> up to 2 roots (_sp4_branches).
inline std::vector<double> sp4_branches(const Eigen::Vector3d& h, const Eigen::Vector3d& k,
                                        const Eigen::Vector3d& p, double delta) {
  const double hk = h.dot(k), kp = k.dot(p);
  const double a = h.dot(p) - hk * kp;
  const double b = h.dot(k.cross(p));
  const double cc = delta - hk * kp;
  const double amp = std::hypot(a, b);
  if (amp < 1e-12) return {};
  const double ratio = cc / amp;
  if (std::abs(ratio) > 1.0 + 1e-9) return {};
  const double base = std::atan2(b, a);
  const double off = std::acos(std::max(-1.0, std::min(1.0, ratio)));
  if (off < 1e-12) return {base + off};
  return {base + off, base - off};
}

// One discrete IK branch; q(psi) is the closed-form joint vector along the swivel.
struct Branch {
  const std::array<Eigen::Vector3d, 7>* n;  // unit joint axes
  Eigen::Vector3d u_sw;
  Eigen::Matrix3d R_sh0;
  double q3;
  Eigen::Matrix3d R_t;
  Eigen::Matrix3d R_post;
  int s_sgn, w_sgn;
  TriplePhase ps, pw;

  std::array<double, 7> q(double psi) const {
    const auto& N = *n;
    const Eigen::Matrix3d R_sh = rotation_matrix(u_sw, psi) * R_sh0;
    const double q1 =
        ps.delta + s_sgn * std::acos(std::max(-1.0, std::min(1.0, (N[0].dot(R_sh * N[2]) - ps.gamma) / ps.rho)));
    const double q0 = signed_angle(N[0], rotation_matrix(N[1], q1) * N[2], R_sh * N[2]);
    const double q2 = -signed_angle(N[2], rotation_matrix(N[1], -q1) * N[0], R_sh.transpose() * N[0]);
    const Eigen::Matrix3d R_res =
        (R_sh * rotation_matrix(N[3], q3)).transpose() * R_t * R_post.transpose();
    const double q5 =
        pw.delta + w_sgn * std::acos(std::max(-1.0, std::min(1.0, (N[4].dot(R_res * N[6]) - pw.gamma) / pw.rho)));
    const double q4 = signed_angle(N[4], rotation_matrix(N[5], q5) * N[6], R_res * N[6]);
    const double q6 = -signed_angle(N[6], rotation_matrix(N[5], -q5) * N[4], R_res.transpose() * N[4]);
    return {q0, q1, q2, q3, q4, q5, q6};
  }
};

// All <=8 branches at psi=0 (mirrors _enumerate_branches).
inline std::vector<Branch> enumerate_branches(const std::array<Eigen::Vector3d, 7>& n,
                                              const SrsConsts& s, const Pose& T) {
  const Eigen::Matrix3d R_t = T.block<3, 3>(0, 0);
  const Eigen::Vector3d W_t = T.block<3, 1>(0, 3) - R_t * s.ee_offset_local;
  const Eigen::Vector3d& S = s.shoulder_pivot;
  const Eigen::Vector3d& n3 = n[s.elbow_index];

  const double d_sw = (W_t - S).norm();
  if (d_sw < feasible::kEps || !(std::abs(s.l_se - s.l_ew) < d_sw && d_sw < s.l_se + s.l_ew))
    return {};
  const Eigen::Vector3d u_sw = (W_t - S) / d_sw;
  const double x_c = (s.l_se * s.l_se - s.l_ew * s.l_ew + d_sw * d_sw) / (2.0 * d_sw);
  const double r_circle = std::sqrt(std::max(s.l_se * s.l_se - x_c * x_c, 0.0));
  Eigen::Vector3d u_p1, u_p2;
  srs_detail::swivel_basis(u_sw, u_p1, u_p2);

  const Eigen::Vector3d elbow0 = S + x_c * u_sw + r_circle * u_p1;
  const Eigen::Vector3d upper0 = elbow0 - S;
  const Eigen::Vector3d d0 = upper0 / s.l_se;
  const Eigen::Matrix3d r0 = min_rotation(s.upper_home, upper0);
  const Eigen::Vector3d wrist_vec0 = W_t - elbow0;

  const TriplePhase ps = triple_phase(n[0], n[1], n[2]);
  const TriplePhase pw = triple_phase(n[4], n[5], n[6]);

  std::vector<Branch> branches;
  for (double q3 : sp4_branches(d0, r0 * n3, r0 * s.forearm_home, wrist_vec0.dot(d0))) {
    const Eigen::Vector3d g = r0 * (rotation_matrix(n3, q3) * s.forearm_home);
    const Eigen::Vector3d g_perp = g - d0 * d0.dot(g);
    const Eigen::Vector3d w_perp = wrist_vec0 - d0 * d0.dot(wrist_vec0);
    const double phi = g_perp.norm() < feasible::kEps
                           ? 0.0
                           : std::atan2(d0.dot(g_perp.cross(w_perp)), g_perp.dot(w_perp));
    const Eigen::Matrix3d R_sh0 = rotation_matrix(d0, phi) * r0;
    for (int s_sgn : {+1, -1})
      for (int w_sgn : {+1, -1})
        branches.push_back(Branch{&n, u_sw, R_sh0, q3, R_t, s.r_post_wrist, s_sgn, w_sgn, ps, pw});
  }
  return branches;
}

// Feasible-swivel arcs for a branch: pre-check the fixed elbow q3, then intersect
// the 6 non-elbow joints' feasible arcs (mirrors _branch_arcs).
inline feasible::Arcs branch_arcs(const Branch& br,
                                  const std::array<std::array<double, 2>, 7>& limits) {
  if (!(limits[3][0] <= br.q3 && br.q3 <= limits[3][1])) return {};
  static const std::vector<double> grid = feasible::param_grid();
  auto q_scalar = [&](double psi) {
    const std::array<double, 7> qv = br.q(psi);
    return std::vector<double>(qv.begin(), qv.end());
  };
  std::vector<feasible::Arc> lim(7);
  for (int i = 0; i < 7; ++i) lim[i] = {limits[i][0], limits[i][1]};
  return feasible::feasible_arcs(q_scalar, {0, 1, 2, 4, 5, 6}, lim, grid);
}

// In-limits solutions via feasible-swivel resolution: arc-centre q per branch
// arc, wrapped to each joint's limit box, FK-verified (mirrors
// feasible_in_limits_solutions + resolve_in_limits' exact path).
inline std::vector<Solution<7>> resolve_in_limits(const JointConsts<7>& c, const SrsConsts& s,
                                                  const Pose& T,
                                                  const std::array<std::array<double, 2>, 7>& limits,
                                                  double fk_atol) {
  std::array<Eigen::Vector3d, 7> n;
  for (int i = 0; i < 7; ++i) n[i] = c.axis[i].normalized();

  std::vector<Solution<7>> out;
  for (const Branch& br : enumerate_branches(n, s, T)) {
    for (const auto& [a, b] : branch_arcs(br, limits)) {
      std::array<double, 7> q = br.q(0.5 * (a + b));
      for (int i = 0; i < 7; ++i) q[i] = feasible::to_limits(q[i], limits[i][0], limits[i][1]);
      const double resid = (fk<7>(c, q) - T).norm();
      if (resid <= fk_atol) out.push_back(Solution<7>{q, resid, Refinement::None});
    }
  }
  return out;
}

}  // namespace srs_swivel
}  // namespace ssik
