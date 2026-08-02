// Subproblem 6 (#486): two coupled SP4-like equations in two unknowns, ported
// verbatim from ssik.subproblems.sp6.solve. After Rodrigues expansion each
// equation is linear in (cos,sin) of the two angles; stacking gives a 2x4
// A x = b whose null space (from the QR of A^T) parameterises the solution set
// as x = x_min + xi1 n1 + xi2 n2. Enforcing |x[:2]| = |x[2:]| = 1 reduces to
// two conics in (xi1, xi2), solved by the Bezout resultant. Candidates are
// sorted by pre-GN residual, Gauss-Newton refined, and returned; caller-level
// FK-residual dedup merges physically-equivalent candidates.
#pragma once

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/rotation.hpp"
#include "ssik_cpp/subproblems.hpp"
#include "ssik_cpp/two_ellipse.hpp"

namespace ssik {

using Vec3x4 = std::array<Eigen::Vector3d, 4>;

namespace detail {

inline double wrap_pi(double a) {
  const double two_pi = 2.0 * M_PI;
  return std::fmod(a + M_PI + 2.0 * two_pi, two_pi) - M_PI;
}

inline double sp6_residual(double t1, double t2, const Vec3x4& h, const Vec3x4& k, const Vec3x4& p,
                           double d1, double d2) {
  const double lhs1 = h[0].dot(rotate(k[0], t1, p[0])) + h[1].dot(rotate(k[1], t2, p[1]));
  const double lhs2 = h[2].dot(rotate(k[2], t1, p[2])) + h[3].dot(rotate(k[3], t2, p[3]));
  return std::max(std::abs(lhs1 - d1), std::abs(lhs2 - d2));
}

// Gauss-Newton refinement of an SP6 angle pair on the 2D residual.
inline std::pair<double, double> refine_sp6(double t1, double t2, const Vec3x4& h, const Vec3x4& k,
                                            const Vec3x4& p, double d1, double d2) {
  const int max_iter = 20;
  const double step_tol = 1e-15;
  const double max_step = M_PI / 4.0;
  for (int it = 0; it < max_iter; ++it) {
    const Eigen::Vector3d r0 = rotate(k[0], t1, p[0]);
    const Eigen::Vector3d r1 = rotate(k[1], t2, p[1]);
    const Eigen::Vector3d r2 = rotate(k[2], t1, p[2]);
    const Eigen::Vector3d r3 = rotate(k[3], t2, p[3]);

    const double f0 = h[0].dot(r0) + h[1].dot(r1) - d1;
    const double f1 = h[2].dot(r2) + h[3].dot(r3) - d2;

    const double j00 = h[0].dot(k[0].cross(r0));
    const double j01 = h[1].dot(k[1].cross(r1));
    const double j10 = h[2].dot(k[2].cross(r2));
    const double j11 = h[3].dot(k[3].cross(r3));

    const double det = j00 * j11 - j01 * j10;
    if (std::abs(det) < 1e-15) break;
    const double inv_det = 1.0 / det;
    double delta0 = -(j11 * f0 - j01 * f1) * inv_det;
    double delta1 = -(-j10 * f0 + j00 * f1) * inv_det;

    const double step_norm = std::sqrt(delta0 * delta0 + delta1 * delta1);
    if (step_norm > max_step) {
      const double scale = max_step / step_norm;
      delta0 *= scale;
      delta1 *= scale;
    }

    t1 = wrap_pi(t1 + delta0);
    t2 = wrap_pi(t2 + delta1);

    if (step_norm < step_tol) break;
  }
  return {t1, t2};
}

inline bool all_p_collinear_with_k(const Vec3x4& k, const Vec3x4& p, double deg_tol) {
  for (int i = 0; i < 4; ++i) {
    const double kp = k[i].dot(p[i]);
    const double p_perp_sq = p[i].dot(p[i]) - kp * kp;
    if (p_perp_sq >= deg_tol) return false;
  }
  return true;
}

}  // namespace detail

// SP6 solve. Returns {solutions (theta1, theta2), is_ls}; is_ls semantics match
// the Python reference (exact deduped set, or single best-LS on infeasibility).
inline std::pair<std::vector<std::pair<double, double>>, bool> sp6(
    const Vec3x4& h, const Vec3x4& k, const Vec3x4& p, double d1, double d2,
    const Tolerances& tol = {}) {
  const double deg_tol = tol.degeneracy;
  const double num_tol = tol.numerical;

  if (detail::all_p_collinear_with_k(k, p, deg_tol)) return {{}, true};

  // A columns: for each i, [k x p | -k x (k x p)] (3x2). h_i . (that) = row pair.
  std::array<Eigen::Vector2d, 4> ha;
  for (int i = 0; i < 4; ++i) {
    const Eigen::Vector3d kxp = k[i].cross(p[i]);
    const Eigen::Vector3d col1 = kxp;
    const Eigen::Vector3d col2 = -k[i].cross(kxp);
    ha[i] = Eigen::Vector2d(h[i].dot(col1), h[i].dot(col2));
  }

  Eigen::Matrix<double, 2, 4> a_mat;
  a_mat << ha[0][0], ha[0][1], ha[1][0], ha[1][1], ha[2][0], ha[2][1], ha[3][0], ha[3][1];

  Eigen::Vector2d b;
  b[0] = d1 - h[0].dot(k[0]) * k[0].dot(p[0]) - h[1].dot(k[1]) * k[1].dot(p[1]);
  b[1] = d2 - h[2].dot(k[2]) * k[2].dot(p[2]) - h[3].dot(k[3]) * k[3].dot(p[3]);

  // Complete QR of A^T (4x2): Q is 4x4, R is 4x2. numpy qr(mode="complete").
  const Eigen::Matrix<double, 4, 2> at = a_mat.transpose();
  Eigen::HouseholderQR<Eigen::Matrix<double, 4, 2>> qr(at);
  const Eigen::Matrix4d q_full = qr.householderQ();
  const Eigen::Matrix<double, 4, 2> r_full = qr.matrixQR().triangularView<Eigen::Upper>();

  const Eigen::Vector4d x_null_1 = q_full.col(2);
  const Eigen::Vector4d x_null_2 = q_full.col(3);
  const Eigen::Matrix<double, 4, 2> q_range = q_full.leftCols<2>();
  const Eigen::Matrix2d r_upper = r_full.topRows<2>();
  const Eigen::Matrix2d r_lower = r_upper.transpose();

  if (std::abs(r_upper(0, 0)) < deg_tol || std::abs(r_upper(1, 1)) < deg_tol) return {{}, true};

  // Solve lower-triangular r_lower @ coefs = b.
  const double la = r_lower(0, 0), lb0 = r_lower(1, 0), lc = r_lower(1, 1);
  const double c0 = b[0] / la;
  const double c1 = (b[1] - c0 * lb0) / lc;
  const Eigen::Vector2d x_min_coefs(c0, c1);
  const Eigen::Vector4d x_min = q_range * x_min_coefs;

  Eigen::Matrix2d xn1;
  xn1 << x_null_1[0], x_null_2[0], x_null_1[1], x_null_2[1];
  Eigen::Matrix2d xn2;
  xn2 << x_null_1[2], x_null_2[2], x_null_1[3], x_null_2[3];

  const auto xi_solutions = solve_two_ellipse_numeric(
      x_min.head<2>(), xn1, x_min.tail<2>(), xn2, deg_tol);
  if (xi_solutions.empty()) return {{}, true};

  std::vector<std::pair<double, double>> candidates;
  for (const auto& [xi0, xi1] : xi_solutions) {
    const Eigen::Vector4d x = x_min + xi0 * x_null_1 + xi1 * x_null_2;
    const double n1 = x.head<2>().norm();
    const double n2 = x.tail<2>().norm();
    if (std::abs(n1 - 1.0) > num_tol || std::abs(n2 - 1.0) > num_tol) continue;
    candidates.emplace_back(std::atan2(x[0], x[1]), std::atan2(x[2], x[3]));
  }

  // Sort by pre-GN residual (stable ordering across LAPACK backends).
  std::stable_sort(candidates.begin(), candidates.end(),
                   [&](const std::pair<double, double>& u, const std::pair<double, double>& v) {
                     return detail::sp6_residual(u.first, u.second, h, k, p, d1, d2) <
                            detail::sp6_residual(v.first, v.second, h, k, p, d1, d2);
                   });

  std::vector<std::pair<double, double>> refined;
  for (const auto& cand : candidates) {
    refined.push_back(detail::refine_sp6(cand.first, cand.second, h, k, p, d1, d2));
  }

  std::vector<std::pair<double, double>> exact;
  for (const auto& cand : refined) {
    if (detail::sp6_residual(cand.first, cand.second, h, k, p, d1, d2) < num_tol) exact.push_back(cand);
  }
  if (!exact.empty()) return {exact, false};
  if (refined.empty()) return {{}, true};

  auto best = *std::min_element(
      refined.begin(), refined.end(),
      [&](const std::pair<double, double>& u, const std::pair<double, double>& v) {
        return detail::sp6_residual(u.first, u.second, h, k, p, d1, d2) <
               detail::sp6_residual(v.first, v.second, h, k, p, d1, d2);
      });
  return {{best}, true};
}

}  // namespace ssik
