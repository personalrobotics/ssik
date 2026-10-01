// Limit contacts of a redundant 7R self-motion: the minimax-margin point (#662),
// ported from ssik.solvers.seven_r._minimax, where the method is argued in full.
//
// When the in-limit part of a chart is a single point or a sliver below the
// bisection tolerance (two or more joints at their limits), the per-joint margins
// touch zero without changing sign and the sign-change arcs (feasible_arcs.hpp)
// are empty. chart_minima finds the local minima of the worst-case violation
//   V(t) = max_i ( |wrap(q_i(t) - c_i)| - h_i )
// along the chart instead, refined by golden section; on a closed-form chart that
// is the contact. On an approximate chart (the polished families) it seeds walk,
// which minimises V along the true self-motion curve: each step solves the 1-D
// linear minimax of the linearised violations along the Jacobian's null
// direction, then projects back onto fk(q) = T. place accepts a point whose
// violation is within its own error band (#651) through onto_limits, clamps it
// onto the limit and re-measures fk_residual (#649).
//
// Only the in-limits fallbacks call this, when their own result is empty, and a
// chart's in_limits only when it has no arc.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/finalize.hpp"  // onto_limits, Remeasure, LazyBand, kBandCap
#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/newton.hpp"  // se3_log_residual, spatial_jacobian
#include "ssik_cpp/seven_r/feasible_arcs.hpp"

namespace ssik::minimax {

using Limits7 = std::array<std::array<double, 2>, 7>;
using Joints7 = std::array<bool, 7>;  // which joints count in V
inline constexpr Joints7 kAllJoints = {true, true, true, true, true, true, true};
inline constexpr Joints7 kSrsSwept = {true, true, true, false, true, true, true};

inline constexpr double kScan = 0.05;  // _minimax.SCAN
// Deepest margin a contact may have (_minimax.SLIVER, where it is measured): this
// owns the in-limit sets that are a point or a sliver; a deeper minimum with no
// arc is an arc another gate lost, left to the rescue as before.
inline constexpr double kSliver = 1e-4;
inline constexpr double kGoldenRel = 4.0 * std::numeric_limits<double>::epsilon();
inline constexpr int kGoldenIters = 100;
inline constexpr double kWalkStep = 0.05;
inline constexpr int kWalkIters = 100;
inline constexpr double kWalkStop = 1e-15;
inline constexpr int kProjectIters = 30;
inline constexpr double kProjectClip = 0.5;

// Worst angular limit violation of q over the limited joints in `use`; negative
// is the smallest margin; inf for a non-finite q. (_minimax.limit_violation)
template <class Q>
double limit_violation(const Q& q, const Limits7& lim, const Joints7& use = kAllJoints) {
  double worst = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < 7; ++i)
    if (!std::isfinite(q[i])) return std::numeric_limits<double>::infinity();
  for (int i = 0; i < 7; ++i) {
    const double c = 0.5 * (lim[i][0] + lim[i][1]);
    const double h = 0.5 * (lim[i][1] - lim[i][0]);
    if (h >= M_PI || !use[i]) continue;
    worst = std::max(worst, std::abs(feasible::wrap(q[i] - c)) - h);
  }
  return worst;
}

// Minimum of f on [a, b] by golden section: the best point evaluated. (_golden)
template <typename F>
std::pair<double, double> golden(F&& f, double a, double b) {
  const double g = 0.5 * (std::sqrt(5.0) - 1.0);
  double x1 = b - g * (b - a), x2 = a + g * (b - a);
  double f1 = f(x1), f2 = f(x2);
  for (int it = 0; it < kGoldenIters; ++it) {
    if (b - a <= kGoldenRel * std::max({1.0, std::abs(a), std::abs(b)})) break;
    if (f1 <= f2) {
      b = x2;
      x2 = x1;
      f2 = f1;
      x1 = b - g * (b - a);
      f1 = f(x1);
    } else {
      a = x1;
      x1 = x2;
      f1 = f2;
      x2 = a + g * (b - a);
      f2 = f(x2);
    }
  }
  return f1 <= f2 ? std::make_pair(x1, f1) : std::make_pair(x2, f2);
}

// Local minima (t, V) of the worst limit violation along a chart q_scalar(t)
// (NaN off it), sorted by V then t. The grid is refined as for the arcs (over all
// seven joints); every local minimum at or below kScan (and the lowest grid
// point) is refined by golden section between its neighbours. (chart_minima)
template <typename QScalar>
std::vector<std::pair<double, double>> chart_minima(QScalar&& q_scalar,
                                                    const std::vector<double>& base_grid,
                                                    const Limits7& lim, bool periodic,
                                                    const Joints7& use = kAllJoints) {
  std::vector<double> ts = base_grid;
  std::vector<std::vector<double>> qs(ts.size());
  for (std::size_t k = 0; k < ts.size(); ++k) qs[k] = q_scalar(ts[k]);
  feasible::refine_grid(q_scalar, ts, qs, {0, 1, 2, 3, 4, 5, 6}, periodic);
  const int n = static_cast<int>(ts.size());
  std::vector<double> v(n);
  bool any = false;
  for (int k = 0; k < n; ++k) {
    v[k] = limit_violation(qs[k], lim, use);
    any = any || std::isfinite(v[k]);
  }
  std::vector<std::pair<double, double>> out;
  if (n == 0 || !any) return out;
  const int best = static_cast<int>(std::min_element(v.begin(), v.end()) - v.begin());
  auto f = [&](double t) {
    return limit_violation(q_scalar(periodic ? feasible::wrap(t) : t), lim, use);
  };
  for (int k = 0; k < n; ++k) {
    const double vk = v[k];
    if (!(vk <= kScan)) continue;
    const int left = periodic ? (k - 1 + n) % n : std::max(k - 1, 0);
    const int right = periodic ? (k + 1) % n : std::min(k + 1, n - 1);
    // The first point of a plateau: strictly below the left, not above the right.
    if (k != best && ((left != k && !(vk < v[left])) || (right != k && v[right] < vk))) continue;
    const double a = ts[left] - (periodic && left > k ? feasible::kTwoPi : 0.0);
    const double b = ts[right] + (periodic && right < k ? feasible::kTwoPi : 0.0);
    auto [t, ft] = golden(f, a, b);
    if (!(ft < vk)) {  // never worse than the grid point itself
      t = ts[k];
      ft = vk;
    }
    out.emplace_back(ft, periodic ? feasible::wrap(t) : t);
  }
  std::sort(out.begin(), out.end());
  for (auto& p : out) std::swap(p.first, p.second);  // -> (t, V)
  return out;
}

// Unit null vector of a 6x7 Jacobian by cofactors; nullopt where it drops rank.
inline std::optional<Eigen::Matrix<double, 7, 1>> null_direction(const Eigen::Matrix<double, 6, 7>& j) {
  Eigen::Matrix<double, 7, 1> n;
  for (int i = 0; i < 7; ++i) {
    Eigen::Matrix<double, 6, 6> m;
    for (int col = 0, k = 0; col < 7; ++col)
      if (col != i) m.col(k++) = j.col(col);
    n[i] = (i % 2 == 0 ? 1.0 : -1.0) * m.determinant();
  }
  const double norm = n.norm();
  if (!(norm > 0.0)) return std::nullopt;
  return Eigen::Matrix<double, 7, 1>(n / norm);
}

// q moved onto fk(q) = T by clipped minimum-norm Gauss-Newton steps. (_project)
inline std::optional<std::array<double, 7>> project(const JointConsts<7>& c,
                                                    std::array<double, 7> q, const Pose& T) {
  for (int it = 0; it < kProjectIters; ++it) {
    const Eigen::Matrix<double, 6, 1> e = se3_log_residual(T * fk<7>(c, q).inverse());
    if (e.cwiseAbs().maxCoeff() <= kWalkStop) break;
    const Eigen::Matrix<double, 6, 7> jac = spatial_jacobian<7>(c, q);
    const Eigen::Matrix<double, 6, 6> jjt = jac * jac.transpose();
    if (jjt.determinant() == 0.0) return std::nullopt;
    const Eigen::Matrix<double, 7, 1> dq = jac.transpose() * jjt.partialPivLu().solve(e);
    for (int i = 0; i < 7; ++i) q[i] += std::max(-kProjectClip, std::min(kProjectClip, dq[i]));
    if (dq.cwiseAbs().maxCoeff() <= kWalkStop) break;
  }
  return q;
}

// The s in [-rho, rho] minimising max_i(v_i + g_i s); ties to the shorter step.
inline double lp_step(const std::vector<double>& v, const std::vector<double>& g, double rho) {
  std::vector<double> cands = {-rho, rho};
  const std::size_t k = v.size();
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = i + 1; j < k; ++j)
      if (g[i] != g[j]) {
        const double s = (v[j] - v[i]) / (g[i] - g[j]);
        if (-rho < s && s < rho) cands.push_back(s);
      }
  double best_s = 0.0, best_v = std::numeric_limits<double>::infinity();
  for (double s : cands) {
    double val = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < k; ++i) val = std::max(val, v[i] + g[i] * s);
    if (val < best_v || (val == best_v && std::abs(s) < std::abs(best_s))) {
      best_s = s;
      best_v = val;
    }
  }
  return best_s;
}

// Minimise the worst limit violation along the true self-motion curve through q.
// (_minimax.walk)
inline std::optional<std::array<double, 7>> walk(const JointConsts<7>& c,
                                                 const std::array<double, 7>& q0, const Pose& T,
                                                 const Limits7& lim) {
  auto cur = project(c, q0, T);
  if (!cur) return std::nullopt;
  double v_cur = limit_violation(*cur, lim);
  double rho = kWalkStep;
  for (int it = 0; it < kWalkIters; ++it) {
    const auto n = null_direction(spatial_jacobian<7>(c, *cur));
    if (!n) break;
    std::vector<double> v, g;
    for (int i = 0; i < 7; ++i) {
      const double ci = 0.5 * (lim[i][0] + lim[i][1]);
      const double h = 0.5 * (lim[i][1] - lim[i][0]);
      if (!(h < M_PI)) continue;
      const double d = feasible::wrap((*cur)[i] - ci);
      v.push_back(std::abs(d) - h);
      g.push_back((d > 0.0 ? 1.0 : (d < 0.0 ? -1.0 : 0.0)) * (*n)[i]);
    }
    const double s = lp_step(v, g, rho);
    if (std::abs(s) <= kWalkStop) break;
    std::array<double, 7> step = *cur;
    for (int i = 0; i < 7; ++i) step[i] += s * (*n)[i];
    const auto nxt = project(c, step, T);
    const double v_nxt = nxt ? limit_violation(*nxt, lim) : std::numeric_limits<double>::infinity();
    if (v_nxt < v_cur) {
      cur = nxt;
      v_cur = v_nxt;
    } else {
      rho = 0.5 * std::abs(s);
      if (rho <= kWalkStop) break;
    }
  }
  return cur;
}

// A contact as a Solution, or nullopt: each joint at its representative nearest
// its range's centre, its worst violation in [-kSliver, kBandCap], closing FK
// within fk_atol there, then every limited joint
// through onto_limits with the point's own error band (#651); a clamped point's
// fk_residual is re-measured (#649). (_minimax.place)
inline std::optional<Solution<7>> place(const JointConsts<7>& c, const JointLimits<7>& present,
                                        const std::array<double, 7>& q, const Pose& T,
                                        const Limits7& lim, double fk_atol, Refinement ref) {
  std::array<double, 7> qw;
  for (int i = 0; i < 7; ++i) qw[i] = feasible::to_limits(q[i], lim[i][0], lim[i][1]);
  const double v = limit_violation(qw, lim);
  if (!(-kSliver <= v && v <= kBandCap)) return std::nullopt;
  double resid = (fk<7>(c, qw) - T).norm();
  if (!(resid <= fk_atol)) return std::nullopt;
  const Remeasure<7> rm{&c, &T};
  const LazyBand<7> band(rm, qw);
  std::array<double, 7> out = qw;
  for (int i = 0; i < 7; ++i) {
    if (!present.present[i]) continue;
    const auto v = onto_limits(qw[i], lim[i][0], lim[i][1], band);
    if (!v) return std::nullopt;
    out[i] = *v;
  }
  if (out != qw) resid = (fk<7>(c, out) - T).norm();
  return Solution<7>{out, resid, ref};
}

// Whether every joint of q is in limits up to the point's own error band (place
// without the FK gate; nor deeper than kSliver), for a chart's in-limits contact.
// (_minimax.within_band)
inline bool within_band(const JointConsts<7>& c, const std::array<double, 7>& q, const Pose& T,
                        const Limits7& lim) {
  const double v = limit_violation(q, lim);
  if (!(-kSliver <= v && v <= kBandCap)) return false;
  const Remeasure<7> rm{&c, &T};
  const LazyBand<7> band(rm, q);
  for (int i = 0; i < 7; ++i) {
    if (lim[i][1] - lim[i][0] >= feasible::kTwoPi) continue;
    if (!onto_limits(feasible::to_limits(q[i], lim[i][0], lim[i][1]), lim[i][0], lim[i][1], band))
      return false;
  }
  return true;
}

// Cluster-merge in wrap-to-pi max-joint distance, keeping the lower residual
// (refinement.dedup_by_wrap_close).
inline std::vector<Solution<7>> dedup(const std::vector<Solution<7>>& sols, double tol) {
  std::vector<Solution<7>> out;
  for (const auto& cand : sols) {
    int dup = -1;
    for (std::size_t j = 0; j < out.size() && dup < 0; ++j) {
      bool close = true;
      for (int i = 0; i < 7; ++i)
        if (std::abs(feasible::wrap(cand.q[i] - out[j].q[i])) > tol) {
          close = false;
          break;
        }
      if (close) dup = static_cast<int>(j);
    }
    if (dup < 0)
      out.push_back(cand);
    else if (cand.fk_residual < out[dup].fk_residual)
      out[dup] = cand;
  }
  return out;
}

}  // namespace ssik::minimax
