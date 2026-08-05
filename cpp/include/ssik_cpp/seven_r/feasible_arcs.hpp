// Exact feasible-parameter arcs for redundant 7R in-limits resolution (#515),
// ported verbatim from ssik.solvers.seven_r._feasible_param. SHARED primitive:
// SRS swivel-limits (periodic feasible_arcs) AND spherical_shoulder q6-redundancy
// (bounded feasible_arcs_bounded) both build on this. Dependency-free -- numpy
// -> std/Eigen, same algebra + branch structure so the port is auditable.
//
// A "joint family" is a callable q_scalar(t) -> per-joint values at parameter t;
// arcs_for_joint brackets the sign-zeros of cos(q_i(t) - c) - cos(half) (i.e.
// q_i(t) in [lo,hi]) on a grid, refines by bisection, and intersects across the
// swept joints. Callers precompute nothing; the grid values are evaluated here.
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace ssik::feasible {

using Arc = std::pair<double, double>;
using Arcs = std::vector<Arc>;

inline constexpr double kEps = 1e-9;
inline constexpr double kTwoPi = 2.0 * M_PI;
inline constexpr int kArcGrid = 180;  // PARAM_GRID resolution

inline double wrap(double a) {
  double m = std::fmod(a + M_PI, kTwoPi);
  if (m < 0.0) m += kTwoPi;
  return m - M_PI;
}

// PARAM_GRID = linspace(-pi, pi, 180, endpoint=False).
inline std::vector<double> param_grid() {
  std::vector<double> g(kArcGrid);
  for (int k = 0; k < kArcGrid; ++k) g[k] = -M_PI + kTwoPi * k / kArcGrid;
  return g;
}

// Root of a monotone-crossing f in [a, b] by bisection (tol on bracket width).
template <typename F>
double bisect(F&& f, double a, double b, double tol = 1e-8) {
  double fa = f(a);
  while (b - a > tol) {
    const double m = 0.5 * (a + b);
    const double fm = f(m);
    if (fm == 0.0) return m;
    if (fa * fm < 0.0) {
      b = m;
    } else {
      a = m;
      fa = fm;
    }
  }
  return 0.5 * (a + b);
}

inline Arcs merge(Arcs arcs) {
  if (arcs.empty()) return {};
  std::sort(arcs.begin(), arcs.end());
  Arcs out;
  out.push_back(arcs[0]);
  for (std::size_t i = 1; i < arcs.size(); ++i) {
    if (arcs[i].first <= out.back().second + kEps) {
      out.back().second = std::max(out.back().second, arcs[i].second);
    } else {
      out.push_back(arcs[i]);
    }
  }
  return out;
}

inline Arcs intersect(const Arcs& a_arcs, const Arcs& b_arcs) {
  Arcs out;
  for (const auto& [a0, a1] : a_arcs)
    for (const auto& [b0, b1] : b_arcs) {
      const double lo = std::max(a0, b0), hi = std::min(a1, b1);
      if (hi - lo > kEps) out.emplace_back(lo, hi);
    }
  return merge(std::move(out));
}

// The 2*pi-equivalent of v nearest the limit centre.
inline double to_limits(double v, double lo, double hi) {
  const double k = std::round((0.5 * (lo + hi) - v) / kTwoPi);
  return v + kTwoPi * k;
}

// Feasible-t arcs for a single periodic joint q_of(t) in [lo, hi]. q_col is q_of
// on `grid` (passed in so a batched family evaluates once).
template <typename QOf>
Arcs arcs_for_joint(QOf&& q_of, double lo, double hi, const std::vector<double>& grid,
                    const std::vector<double>& q_col) {
  const double c = 0.5 * (lo + hi);
  const double half = 0.5 * (hi - lo);
  if (half >= M_PI) return {{-M_PI, M_PI}};  // unconstrained
  const double thr = std::cos(half);
  auto phi = [&](double p) { return std::cos(q_of(p) - c) - thr; };

  const int n = static_cast<int>(grid.size());
  std::vector<double> val(n);
  for (int k = 0; k < n; ++k) val[k] = std::cos(q_col[k] - c) - thr;

  std::vector<double> roots;
  for (int k = 0; k < n; ++k) {
    const double a = grid[k];
    const double b = (k + 1 < n) ? grid[k + 1] : M_PI;
    if (val[k] * val[(k + 1) % n] < 0.0) roots.push_back(bisect(phi, a, b));
  }
  if (roots.empty()) return val[0] >= 0.0 ? Arcs{{-M_PI, M_PI}} : Arcs{};
  std::sort(roots.begin(), roots.end());

  Arcs arcs;
  std::vector<double> ext = roots;
  ext.push_back(roots[0] + kTwoPi);
  for (std::size_t i = 0; i + 1 < ext.size(); ++i) {
    const double u = ext[i], w = ext[i + 1];
    if (phi(wrap(0.5 * (u + w))) >= 0.0) {
      if (w <= M_PI) {
        arcs.emplace_back(u, w);
      } else {  // arc straddles +pi: split
        arcs.emplace_back(u, M_PI);
        arcs.emplace_back(-M_PI, wrap(w));
      }
    }
  }
  return merge(std::move(arcs));
}

// Feasible sub-intervals of the bounded domain [grid.front(), grid.back()] for a
// single joint q_of(t) in [lo, hi] -- non-periodic analogue (no wrap).
template <typename QOf>
Arcs arcs_for_joint_bounded(QOf&& q_of, double lo, double hi, const std::vector<double>& grid,
                            const std::vector<double>& q_col) {
  const double a0 = grid.front(), b0 = grid.back();
  const double c = 0.5 * (lo + hi);
  const double half = 0.5 * (hi - lo);
  if (half >= M_PI) return {{a0, b0}};
  const double thr = std::cos(half);
  auto phi = [&](double p) { return std::cos(q_of(p) - c) - thr; };

  const int n = static_cast<int>(grid.size());
  std::vector<double> val(n);
  for (int k = 0; k < n; ++k) val[k] = std::cos(q_col[k] - c) - thr;

  std::vector<double> roots;
  for (int k = 0; k + 1 < n; ++k)
    if (val[k] * val[k + 1] < 0.0) roots.push_back(bisect(phi, grid[k], grid[k + 1]));
  std::sort(roots.begin(), roots.end());

  std::vector<double> pts;
  pts.push_back(a0);
  pts.insert(pts.end(), roots.begin(), roots.end());
  pts.push_back(b0);
  Arcs arcs;
  for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
    const double u = pts[i], w = pts[i + 1];
    if (phi(0.5 * (u + w)) >= 0.0) arcs.emplace_back(u, w);
  }
  return merge(std::move(arcs));
}

// Exact periodic feasible-t set: intersection of every swept joint's arcs.
// q_scalar(t) -> per-joint values. Empty iff no t keeps all swept joints in range.
template <typename QScalar>
Arcs feasible_arcs(QScalar&& q_scalar, const std::vector<int>& swept_joints,
                   const std::vector<Arc>& limits, const std::vector<double>& grid) {
  // Evaluate the joint family once on the grid (mirrors the Python q_grid).
  std::vector<std::vector<double>> q_grid(grid.size());
  for (std::size_t k = 0; k < grid.size(); ++k) q_grid[k] = q_scalar(grid[k]);

  Arcs arcs = {{-M_PI, M_PI}};
  for (int i : swept_joints) {
    std::vector<double> q_col(grid.size());
    for (std::size_t k = 0; k < grid.size(); ++k) q_col[k] = q_grid[k][i];
    auto q_of = [&, i](double t) { return q_scalar(t)[i]; };
    arcs = intersect(arcs, arcs_for_joint(q_of, limits[i].first, limits[i].second, grid, q_col));
    if (arcs.empty()) return {};
  }
  return arcs;
}

// Bounded-domain analogue of feasible_arcs (non-periodic).
template <typename QScalar>
Arcs feasible_arcs_bounded(QScalar&& q_scalar, const std::vector<int>& swept_joints,
                           const std::vector<Arc>& limits, const std::vector<double>& grid) {
  std::vector<std::vector<double>> q_grid(grid.size());
  for (std::size_t k = 0; k < grid.size(); ++k) q_grid[k] = q_scalar(grid[k]);

  Arcs arcs = {{grid.front(), grid.back()}};
  for (int i : swept_joints) {
    std::vector<double> q_col(grid.size());
    for (std::size_t k = 0; k < grid.size(); ++k) q_col[k] = q_grid[k][i];
    auto q_of = [&, i](double t) { return q_scalar(t)[i]; };
    arcs = intersect(
        arcs, arcs_for_joint_bounded(q_of, limits[i].first, limits[i].second, grid, q_col));
    if (arcs.empty()) return {};
  }
  return arcs;
}

}  // namespace ssik::feasible
