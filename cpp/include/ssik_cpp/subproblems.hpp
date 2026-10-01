// IK-Geo canonical subproblems (#486), ported from ssik's Python reference.
//
// Phase-0a scope: SP1, SP3, SP4 -- the closed-form ones `ikgeo.three_parallel`
// composes. SP6 (QR + two-ellipse Bezout resultant + companion-eigenvalue
// quartic) is the numeric subproblem and lands in its own chunk. Each function
// mirrors the corresponding `ssik.subproblems.spN.solve` exactly, including the
// `is_ls` feasibility flag and the scale-aware degeneracy thresholds, so the
// conformance harness can compare like with like.
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace ssik {

// Subproblem tolerances -- the DEFAULT_TOLERANCE_POLICY values, baked so the
// native side gates feasibility/degeneracy identically to Python.
struct Tolerances {
  double feasibility = 1e-9;
  double degeneracy = 1e-12;
  double numerical = 1e-5;
  double dedup = 1e-3;
};

// The SP1 angle: theta rotating p about unit axis k toward q, the one rule for
// every SP1 site (ssik.subproblems.sp1.angle, whose comment derives it). Both
// terms come from k x p and k x q: for unit k, (k x p) . (k x q) equals
// p.q - (k.p)(k.q) and k . ((k x p) x (k x q)) equals (k x p) . q, but without
// their cancellation near a singularity (#661).
// kxp is k x p, for a caller that has it already.
inline double sp1_angle_kxp(const Eigen::Vector3d& k, const Eigen::Vector3d& kxp,
                            const Eigen::Vector3d& q) {
  const Eigen::Vector3d kxq = k.cross(q);
  return std::atan2(k.dot(kxp.cross(kxq)), kxp.dot(kxq));
}

inline double sp1_angle(const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                        const Eigen::Vector3d& q) {
  return sp1_angle_kxp(k, k.cross(p), q);
}

// SP1: the angle theta rotating p about unit axis k toward q.
// Returns {theta, is_ls}; is_ls is true when the exact feasibility conditions
// (|p_perp| == |q_perp| and k.p == k.q) do not hold, so theta is the LS optimum.
inline std::pair<double, bool> sp1(const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& q, const Tolerances& tol = {}) {
  const double kp = k.dot(p);
  const double kq = k.dot(q);
  const double theta = sp1_angle(k, p, q);
  const double p_perp_sq = p.dot(p) - kp * kp;
  const double q_perp_sq = q.dot(q) - kq * kq;
  const bool is_ls =
      std::abs(p_perp_sq - q_perp_sq) > tol.feasibility || std::abs(kp - kq) > tol.feasibility;
  return {theta, is_ls};
}

// Whether A cos(theta) + B sin(theta) = rhs has a real (double) root, with
// amplitude = |A, B| (ssik.subproblems.sp4.within_tangent_band). Exact inputs
// are feasible when |rhs| <= amplitude up to relative round-off. A caller whose
// rhs carries a known approximation error of at most `drift` also accepts
// |rhs| up to amplitude + drift: a double root pushed past tangency by that
// error, recovered by clamping rhs / amplitude to +-1 (#598). drift == 0 leaves
// exact callers unchanged.
inline bool within_tangent_band(double rhs, double amplitude, double drift = 0.0) {
  constexpr double kTangentRelTol = 1e-9;
  return std::abs(rhs) <= amplitude * (1.0 + kTangentRelTol) + drift;
}

// SP4: the angles theta rotating p about k such that h . (rot(k, theta) p) = d.
// Returns {thetas (0/1/2), is_ls}.
inline std::pair<std::vector<double>, bool> sp4(const Eigen::Vector3d& h, const Eigen::Vector3d& k,
                                                const Eigen::Vector3d& p, double d,
                                                const Tolerances& tol = {}) {
  const double hp = h.dot(p);
  const double kp = k.dot(p);
  const double hk = h.dot(k);
  const double coef_a = hp - kp * hk;
  const double coef_b = h.dot(k.cross(p));
  const double coef_c = kp * hk;
  const double r_sq = coef_a * coef_a + coef_b * coef_b;

  const double deg_sq = tol.degeneracy * tol.degeneracy;
  if (r_sq < deg_sq) {
    // p collinear with k: rotation doesn't change the projection.
    return {{0.0}, std::abs(coef_c - d) > tol.feasibility};
  }

  const double r = std::sqrt(r_sq);
  const double rhs = d - coef_c;
  const double phi = std::atan2(coef_b, coef_a);

  if (std::abs(rhs) - r > tol.feasibility) {
    const double theta = rhs > 0.0 ? phi : phi + M_PI;
    return {{theta}, true};
  }

  const double rhs_clipped = std::clamp(rhs / r, -1.0, 1.0);
  const double delta = std::acos(rhs_clipped);
  if (delta < tol.feasibility) {
    return {{phi}, false};  // tangent: single solution
  }
  return {{phi + delta, phi - delta}, false};
}

// SP3: the angles theta rotating p about k such that |rot(k, theta) p - q| = d.
// Thin wrapper over SP4 (identical to the Python reference), returns {thetas, is_ls}.
inline std::pair<std::vector<double>, bool> sp3(const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                                                const Eigen::Vector3d& q, double d,
                                                const Tolerances& tol = {}) {
  const double target = 0.5 * (p.dot(p) + q.dot(q) - d * d);
  return sp4(q, k, p, target, tol);
}

// SP2 (Paden-Kahan 2): angles (theta1, theta2) with
// rot(k1, theta1) p == rot(k2, theta2) q. Returns {solutions (1 or 2), is_ls}.
// Mirrors ssik.subproblems.sp2.solve (parallel-axis fallback, tangent/LS single
// representative, else two intersection branches via sp1 to the two z points).
inline std::pair<std::vector<std::pair<double, double>>, bool> sp2(
    const Eigen::Vector3d& k1, const Eigen::Vector3d& k2, const Eigen::Vector3d& p,
    const Eigen::Vector3d& q, const Tolerances& tol = {}) {
  const double c = k1.dot(k2);
  const double s_sq = 1.0 - c * c;
  if (s_sq < tol.degeneracy) {  // parallel axes: canonical single choice
    const double th1 = sp1(k1, p, q, tol).first;
    return {{{th1, 0.0}}, true};
  }
  const double d1 = k1.dot(p), d2 = k2.dot(q);
  const double alpha = (d1 - c * d2) / s_sq;
  const double beta = (d2 - c * d1) / s_sq;
  const Eigen::Vector3d kxk = k1.cross(k2);  // |kxk|^2 = s_sq
  const double pp = p.dot(p), qq = q.dot(q);
  const double gamma_sq_scaled =
      0.5 * (pp + qq) - alpha * alpha - beta * beta - 2.0 * alpha * beta * c;
  const bool is_ls =
      std::abs(pp - qq) > tol.feasibility || gamma_sq_scaled < -tol.feasibility;

  if (gamma_sq_scaled <= 0.0) {  // tangent / LS: single representative z (gamma 0)
    const Eigen::Vector3d z = alpha * k1 + beta * k2;
    return {{{sp1(k1, p, z, tol).first, sp1(k2, q, z, tol).first}}, is_ls};
  }
  const double gamma = std::sqrt(gamma_sq_scaled / s_sq);
  const Eigen::Vector3d z_a = alpha * k1 + beta * k2 + gamma * kxk;
  const Eigen::Vector3d z_b = alpha * k1 + beta * k2 - gamma * kxk;
  return {{{sp1(k1, p, z_a, tol).first, sp1(k2, q, z_a, tol).first},
           {sp1(k1, p, z_b, tol).first, sp1(k2, q, z_b, tol).first}},
          is_ls};
}

}  // namespace ssik
