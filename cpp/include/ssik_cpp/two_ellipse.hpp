// Intersection of two conics |xm_i + xn_i [xi, eta]^T| = 1 (#486), ported
// verbatim from ssik.subproblems._aux.solve_two_ellipse_numeric: the Bezout
// resultant (a quartic in y) via np_roots, plus ssik's degenerate-conic
// fallback (where stock IK-Geo produces NaN). Conforms to the oracle.
#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/quartic.hpp"

namespace ssik {

// Real roots of a x^2 + b x + c = 0 (linear/empty if degenerate), matching
// ssik._aux._quadratic_real_roots.
inline std::vector<double> quadratic_real_roots(double a, double b, double c, double tol) {
  if (std::abs(a) < tol) {
    if (std::abs(b) < tol) return {};
    return {-c / b};
  }
  const double disc = b * b - 4.0 * a * c;
  if (disc < -tol) return {};
  const double rd = std::sqrt(std::max(0.0, disc));
  return {(-b + rd) / (2.0 * a), (-b - rd) / (2.0 * a)};
}

using Conic = std::array<double, 6>;  // (a, b, c, d, e, f): a x^2 + b xy + c y^2 + d x + e y + f

// One conic decouples to a 1-variable quadratic; solve it then substitute into
// the other. primary_y: coef1 is in y alone; else coef1 is in x alone.
inline std::vector<std::pair<double, double>> solve_two_ellipse_degenerate(
    const Conic& coef1, const Conic& coef2, double tol, bool primary_y) {
  std::vector<std::pair<double, double>> out;
  const auto& [a2, b2, c2, d2, e2, f2] = coef2;
  if (primary_y) {
    const double c = coef1[2], e = coef1[4], f = coef1[5];
    for (double y : quadratic_real_roots(c, e, f, tol)) {
      for (double x : quadratic_real_roots(a2, b2 * y + d2, c2 * y * y + e2 * y + f2, tol)) {
        out.emplace_back(x, y);
      }
    }
  } else {
    const double a = coef1[0], d = coef1[3], f = coef1[5];
    for (double x : quadratic_real_roots(a, d, f, tol)) {
      for (double y : quadratic_real_roots(c2, b2 * x + e2, a2 * x * x + d2 * x + f2, tol)) {
        out.emplace_back(x, y);
      }
    }
  }
  return out;
}

inline std::vector<std::pair<double, double>> solve_two_ellipse_numeric(
    const Eigen::Vector2d& xm1, const Eigen::Matrix2d& xn1, const Eigen::Vector2d& xm2,
    const Eigen::Matrix2d& xn2, double epsilon) {
  const Eigen::Matrix2d a_1 = xn1.transpose() * xn1;
  const double a = a_1(0, 0), b = 2.0 * a_1(1, 0), c = a_1(1, 1);
  const Eigen::Vector2d b_1 = 2.0 * (xn1.transpose() * xm1);
  const double d = b_1(0), e = b_1(1), f = xm1.dot(xm1) - 1.0;

  const Eigen::Matrix2d a_2 = xn2.transpose() * xn2;
  const double a1 = a_2(0, 0), b1 = 2.0 * a_2(1, 0), c1 = a_2(1, 1);
  const Eigen::Vector2d b_2 = 2.0 * (xn2.transpose() * xm2);
  const double d1 = b_2(0), e1 = b_2(1), fq = xm2.dot(xm2) - 1.0;

  const Conic coef1{a, b, c, d, e, f};
  const Conic coef2{a1, b1, c1, d1, e1, fq};
  if (std::abs(a) < epsilon && std::abs(b) < epsilon && std::abs(d) < epsilon)
    return solve_two_ellipse_degenerate(coef1, coef2, epsilon, /*primary_y=*/true);
  if (std::abs(c) < epsilon && std::abs(b) < epsilon && std::abs(e) < epsilon)
    return solve_two_ellipse_degenerate(coef1, coef2, epsilon, /*primary_y=*/false);
  if (std::abs(a1) < epsilon && std::abs(b1) < epsilon && std::abs(d1) < epsilon)
    return solve_two_ellipse_degenerate(coef2, coef1, epsilon, /*primary_y=*/true);
  if (std::abs(c1) < epsilon && std::abs(b1) < epsilon && std::abs(e1) < epsilon) {
    // coef2 is x-only; solve for x, substitute into coef1 for y. Output stays (x, y).
    return solve_two_ellipse_degenerate(coef2, coef1, epsilon, /*primary_y=*/false);
  }

  // Bezout resultant: quartic in y (z4 y^4 + ... + z0).
  const double z0 = f * a * d1 * d1 + a * a * fq * fq - d * a * d1 * fq + a1 * a1 * f * f -
                    2.0 * a * fq * a1 * f - d * d1 * a1 * f + a1 * d * d * fq;
  const double z1 = e1 * d * d * a1 - fq * d1 * a * b - 2.0 * a * fq * a1 * e - f * a1 * b1 * d +
                    2.0 * d1 * b1 * a * f + 2.0 * e1 * fq * a * a + d1 * d1 * a * e -
                    e1 * d1 * a * d - 2.0 * a * e1 * a1 * f - f * a1 * d1 * b +
                    2.0 * f * e * a1 * a1 - fq * b1 * a * d - e * a1 * d1 * d + 2.0 * fq * b * a1 * d;
  const double z2 = e1 * e1 * a * a + 2.0 * c1 * fq * a * a - e * a1 * d1 * b + fq * a1 * b * b -
                    e * a1 * b1 * d - fq * b1 * a * b - 2.0 * a * e1 * a1 * e +
                    2.0 * d1 * b1 * a * e - c1 * d1 * a * d - 2.0 * a * c1 * a1 * f +
                    b1 * b1 * a * f + 2.0 * e1 * b * a1 * d + e * e * a1 * a1 - c * a1 * d1 * d -
                    e1 * b1 * a * d + 2.0 * f * c * a1 * a1 - f * a1 * b1 * b + c1 * d * d * a1 +
                    d1 * d1 * a * c - e1 * d1 * a * b - 2.0 * a * fq * a1 * c;
  const double z3 = -2.0 * a * a1 * c * e1 + e1 * a1 * b * b + 2.0 * c1 * b * a1 * d -
                    c * a1 * b1 * d + b1 * b1 * a * e - e1 * b1 * a * b - 2.0 * a * c1 * a1 * e -
                    e * a1 * b1 * b - c1 * b1 * a * d + 2.0 * e1 * c1 * a * a +
                    2.0 * e * c * a1 * a1 - c * a1 * d1 * b + 2.0 * d1 * b1 * a * c -
                    c1 * d1 * a * b;
  const double z4 = a * a * c1 * c1 - 2.0 * a * c1 * a1 * c + a1 * a1 * c * c - b * a * b1 * c1 -
                    b * b1 * a1 * c + b * b * a1 * c1 + c * a * b1 * b1;

  const auto y_all = np_roots({z4, z3, z2, z1, z0});
  std::vector<std::pair<double, double>> solutions;
  for (const auto& y_c : y_all) {
    if (std::abs(y_c.imag()) >= epsilon) continue;
    const double y = y_c.real();
    const double num =
        -((a * c1 * y * y + a * fq) - a1 * c * y * y + a * e1 * y - a1 * e * y - a1 * f);
    const double den = (a * b1 * y + a * d1) - a1 * b * y - a1 * d;
    if (std::abs(den) < epsilon) continue;
    solutions.emplace_back(num / den, y);
  }
  return solutions;
}

}  // namespace ssik
