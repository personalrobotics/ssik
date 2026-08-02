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

// SP1: the angle theta rotating p about unit axis k toward q.
// Returns {theta, is_ls}; is_ls is true when the exact feasibility conditions
// (|p_perp| == |q_perp| and k.p == k.q) do not hold, so theta is the LS optimum.
inline std::pair<double, bool> sp1(const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& q, const Tolerances& tol = {}) {
  const Eigen::Vector3d kxp = k.cross(p);
  const double kp = k.dot(p);
  const double kq = k.dot(q);
  const double theta = std::atan2(kxp.dot(q), p.dot(q) - kp * kq);
  const double p_perp_sq = p.dot(p) - kp * kp;
  const double q_perp_sq = q.dot(q) - kq * kq;
  const bool is_ls =
      std::abs(p_perp_sq - q_perp_sq) > tol.feasibility || std::abs(kp - kq) > tol.feasibility;
  return {theta, is_ls};
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

}  // namespace ssik
