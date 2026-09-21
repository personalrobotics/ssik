// Charts of the self-motion manifold for the two closed-form 7R families --
// the C++ replica of ssik.chart (Python is the reference, ADR-0001).
//
// A chart is one continuous branch q(t) of FK^-1(T) at one pose, with a stable
// discrete label, its domain in the redundancy coordinate t, and the inverse
// map locate(q) -> (chart, t). Everything here is header-only Eigen so a C++
// control loop can hold a *Charts object per target pose and evaluate q(t) /
// locate(q) per tick in microseconds.
//
//   seven_r.spherical_shoulder (franka/fr3):  t = q6, labels (e, sh, w, interval)
//   seven_r.srs (iiwa and other exact SRS):   t = swivel psi, labels (e, s, w)
//
// Mirrors ssik/chart.py: same slot indexing (4e + 2sh + w), same feasibility
// gates as the Python batched subproblems (_sp4_batch / _sp2_batch), same
// reachable-interval bracket, domain boundaries refined by bisection on branch
// validity. Parity is asserted in tests/test_chart_native.py.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/newton.hpp"  // spatial_jacobian
#include "ssik_cpp/seven_r/srs_swivel_limits.hpp"
#include "ssik_cpp/solvers/spherical_shoulder.hpp"
#include "ssik_cpp/subproblems.hpp"

namespace ssik::chart {

inline constexpr int kSlots = 8;
inline constexpr int kDomainGrid = 180;           // ssik.chart._Q6_DOMAIN_GRID
inline constexpr double kDomainTol = 1e-10;       // bisection width on a boundary
inline constexpr double kContainsTol = 1e-9;      // Chart.contains default tol
inline constexpr double kSeedEps = 1e-7;          // ssik.chart._Q6_SEED_EPS
inline constexpr double kStencilH = 1e-4;         // ssik.chart._STENCIL_H
inline constexpr double kLockTol = 1e-9;          // spherical_shoulder._LOCK_TOL
inline constexpr double kTangentSnap = 1e-15;     // spherical_shoulder._TANGENT_SNAP

struct Interval {
  double lo, hi;
};

inline double wrap_pi(double a) { return std::fmod(a + M_PI, 2.0 * M_PI) < 0
                                         ? std::fmod(a + M_PI, 2.0 * M_PI) + 2.0 * M_PI - M_PI
                                         : std::fmod(a + M_PI, 2.0 * M_PI) - M_PI; }

inline double wrap_dist(const std::array<double, 7>& a, const std::array<double, 7>& b) {
  double d = 0.0;
  for (int i = 0; i < 7; ++i) d = std::max(d, std::abs(wrap_pi(a[i] - b[i])));
  return d;
}

inline constexpr double kFrameDegenerate = 1e-12;  // ssik.chart.Chart.frame

// Split a raw tangent into direction and rate: returns rate = |dq| and writes
// d_out = dq / rate. Plain IEEE throughout, so the degenerate cases carry the
// same values numpy's `d / norm(d)` gives and ssik.chart.Chart.tangent
// documents: at a fold rate diverges while the direction stays unit; a
// vanishing dq gives rate 0 and a NaN direction (0/0); off the branch dq is
// NaN and both are NaN.
inline double unit_tangent(const std::array<double, 7>& dq, std::array<double, 7>& d_out) {
  const Eigen::Map<const Eigen::Matrix<double, 7, 1>> raw(dq.data());
  const double rate = raw.norm();
  Eigen::Map<Eigen::Matrix<double, 7, 1>>(d_out.data()) = raw / rate;
  return rate;
}

// The chart frame from a raw tangent (request D1): the unit direction
// d = dq/|dq| and a (7, 6) basis V of its complement, M-orthogonal to it,
// d^T M V = 0. `metric` is a row-major 7x7 SPD matrix, or nullptr for the
// Euclidean case (the kinematic split ker(J)^perp). The columns of V are
// Euclidean-orthonormal, built by the Householder reflection carrying e_0 onto
// M d / |M d|, so V varies continuously along the arc except where M d passes
// through -e_0. False off the branch or at a fold (|dq| not finite).
// Mirrors ssik.chart.Chart.frame.
inline bool frame_from_tangent(const std::array<double, 7>& dq, const double* metric,
                               std::array<double, 7>& d_out,
                               Eigen::Matrix<double, 7, 6>& v_out) {
  const double rate = unit_tangent(dq, d_out);
  if (!std::isfinite(rate) || rate == 0.0) return false;
  const Eigen::Map<const Eigen::Matrix<double, 7, 1>> d(d_out.data());

  Eigen::Matrix<double, 7, 1> w =
      metric ? Eigen::Matrix<double, 7, 1>(
                   Eigen::Map<const Eigen::Matrix<double, 7, 7, Eigen::RowMajor>>(metric) * d)
             : d;
  const double nw = w.norm();
  if (!std::isfinite(nw) || nw == 0.0) return false;
  w /= nw;

  // Householder H = I - 2 u u^T with H e0 = w  (u along e0 - w); V = H[:, 1:].
  Eigen::Matrix<double, 7, 1> u = -w;
  u(0) += 1.0;
  const double nu = u.norm();
  if (nu < kFrameDegenerate) {  // w == e0: H = I
    v_out = Eigen::Matrix<double, 7, 7>::Identity().rightCols<6>();
    return true;
  }
  u /= nu;
  v_out = (Eigen::Matrix<double, 7, 7>::Identity() - 2.0 * u * u.transpose()).rightCols<6>();
  return true;
}

// ---------------------------------------------------------------------------
// spherical_shoulder: slot-indexed closed form (mirrors _slot_grid)
// ---------------------------------------------------------------------------

namespace sh {

// Batched-subproblem replicas: both roots + feasibility, never fewer roots, so
// the slot index is a stable label (Python _sp1_batch / _sp4_batch / _sp2_batch).
// Angle 0 where p lies on the axis (the angle is free; atan2 of rounding noise
// would be arbitrary) -- the canonical representative, as _sp1_batch.
inline double sp1_both(const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                       const Eigen::Vector3d& q) {
  const Eigen::Vector3d kxp = k.cross(p);
  if (kxp.norm() <= kLockTol * p.norm()) return 0.0;
  return std::atan2(kxp.dot(q), p.dot(q) - k.dot(p) * k.dot(q));
}

inline bool sp4_both(const Eigen::Vector3d& h, const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                     double d, const Tolerances& tol, std::array<double, 2>& out) {
  const double a = h.dot(p) - k.dot(p) * h.dot(k);
  const double b = h.dot(k.cross(p));
  const double cc = k.dot(p) * h.dot(k);
  const double r = std::hypot(a, b);
  const double rhs = d - cc;
  double ratio = std::clamp(r > 1e-12 ? rhs / r : 0.0, -1.0, 1.0);
  // Snap the tangent case to an exact double root (as _sp4_batch).
  if (ratio >= 1.0 - kTangentSnap) ratio = 1.0;
  if (ratio <= -1.0 + kTangentSnap) ratio = -1.0;
  const double delta = std::acos(ratio);
  const double phi = std::atan2(b, a);
  out = {phi + delta, phi - delta};
  return (std::abs(rhs) - r <= tol.feasibility) && (r * r >= tol.degeneracy * tol.degeneracy);
}

inline bool sp2_both(const Eigen::Vector3d& k1, const Eigen::Vector3d& k2, const Eigen::Vector3d& p,
                     const Eigen::Vector3d& q, const Tolerances& tol, std::array<double, 2>& t1,
                     std::array<double, 2>& t2) {
  const double c = k1.dot(k2);
  const double s_sq = 1.0 - c * c;
  const double safe = s_sq > tol.degeneracy ? s_sq : 1.0;
  const double d1 = k1.dot(p), d2 = k2.dot(q);
  const double alpha = (d1 - c * d2) / safe;
  const double beta = (d2 - c * d1) / safe;
  const Eigen::Vector3d kxk = k1.cross(k2);
  const double pp = p.dot(p), qq = q.dot(q);
  double gss = 0.5 * (pp + qq) - alpha * alpha - beta * beta - 2.0 * alpha * beta * c;
  if (std::abs(gss) <= kTangentSnap * std::max(pp, qq)) gss = 0.0;  // tangent case (as _sp2_batch)
  const bool feas = (std::abs(pp - qq) <= tol.feasibility) && (gss >= -tol.feasibility) &&
                    (s_sq >= tol.degeneracy);
  const double gamma = std::sqrt(std::max(gss, 0.0) / safe);
  const Eigen::Vector3d base = alpha * k1 + beta * k2;
  const Eigen::Vector3d za = base + gamma * kxk;
  const Eigen::Vector3d zb = base - gamma * kxk;
  t1 = {sp1_both(k1, p, za), sp1_both(k1, p, zb)};
  t2 = {sp1_both(k2, q, za), sp1_both(k2, q, zb)};
  return feas;
}

struct SlotEval {
  std::array<std::array<double, 7>, kSlots> q{};
  std::array<bool, kSlots> valid{};
};

inline Eigen::Matrix3d rot(const Eigen::Vector3d& k, double th) {
  return Eigen::AngleAxisd(th, k).toRotationMatrix();
}

// Every slot at one q6 (mirrors _slot_grid at a single grid point). With
// valid_only the two SP1 stages are skipped (q is left unset).
inline SlotEval slot_eval(const Eigen::Matrix<double, 3, 48>& coef, const Pose& t_rev, double q6,
                          const Tolerances& tol, bool valid_only = false) {
  const sh_detail::Geom g = sh_detail::eval_geom(coef, q6);
  const auto& a = g.axes;
  const Eigen::Vector3d p0 = g.our_p[0], p2 = g.our_p[2];
  const Eigen::Vector3d p3 = g.our_p[3] + g.our_p[4] + g.our_p[5];
  const Eigen::Matrix3d r_06 = t_rev.block<3, 3>(0, 0) * g.r_home.transpose();
  const Eigen::Vector3d p_16 = t_rev.block<3, 1>(0, 3) - r_06 * g.tool - p0;

  SlotEval out;
  const double d3 = 0.5 * (p3.dot(p3) + p2.dot(p2) - p_16.dot(p_16));
  std::array<double, 2> q3_both;
  const bool feas3 = sp4_both(-p2, a[2], p3, d3, tol, q3_both);  // SP3 via SP4
  for (int e = 0; e < 2; ++e) {
    const double q3 = q3_both[e];
    const Eigen::Vector3d sp2_arg = p2 + rot(a[2], q3) * p3;
    std::array<double, 2> t1, t2;
    const bool feas2 = sp2_both(-a[0], a[1], p_16, sp2_arg, tol, t1, t2);
    for (int s = 0; s < 2; ++s) {
      const double q1 = t1[s], q2 = t2[s];
      const Eigen::Matrix3d r36 = rot(-a[2], q3) * rot(-a[1], q2) * rot(-a[0], q1) * r_06;
      std::array<double, 2> q5_both;
      const bool feas4 = sp4_both(a[3], a[4], a[5], a[3].dot(r36 * a[5]), tol, q5_both);
      const bool ok = feas3 && feas2 && feas4;
      for (int w = 0; w < 2; ++w) {
        const int slot = 4 * e + 2 * s + w;
        out.valid[slot] = ok;
        if (valid_only) continue;
        const double q5 = q5_both[w];
        const Eigen::Matrix3d r45 = rot(a[4], q5);
        const Eigen::Vector3d a5_mid = r45 * a[5];
        double q4 = sp1_both(a[3], a5_mid, r36 * a[5]);
        double q6i = sp1_both(-a[5], rot(-a[4], q5) * a[3], r36.transpose() * a[3]);
        // Gimbal lock of the wrist triple (a3 || R45 a5): only q4 + q6i is
        // determined. Canonical split as _slot_grid: q6i = 0, q4 the angle of
        // r36 R45^T about a3 read off a4.
        if (a[3].cross(a5_mid).norm() <= kLockTol) {
          const Eigen::Matrix3d m = r36 * r45.transpose();
          q4 = sp1_both(a[3], a[4], m * a[4]);
          q6i = 0.0;
        }
        out.q[slot] = {q6i, q5, q4, q3, q2, q1, q6};
      }
    }
  }
  return out;
}

}  // namespace sh

// Exact reachable q6 arcs of the elbow (SP3) gate -- spherical_shoulder.elbow_arcs.
// The reversed lock-6 chain is one rigid chain rotated about the joint-6 axis, so
// every chain-fixed scalar product is constant in q6 and the only varying SP3
// quantity is |p_16|^2 = A + B cos q6 + C sin q6 (three samples determine it).
// SP3 closes iff lo <= rho cos(q6 - phi) <= hi: at most two arcs, ends at
// phi +- acos(.), which are the exact elbow folds. Sorted, split at the seam.
inline std::vector<Interval> elbow_arcs(const Eigen::Matrix<double, 3, 48>& coef,
                                        const Pose& t_rev) {
  const double samples[3] = {0.0, 2.0 * M_PI / 3.0, 4.0 * M_PI / 3.0};
  Eigen::Vector3d d2;
  double r = 0.0, center = 0.0, s_const = 0.0;
  for (int i = 0; i < 3; ++i) {
    const sh_detail::Geom g = sh_detail::eval_geom(coef, samples[i]);
    const Eigen::Vector3d p2 = g.our_p[2];
    const Eigen::Vector3d p3 = g.our_p[3] + g.our_p[4] + g.our_p[5];
    const Eigen::Matrix3d r_06 = t_rev.block<3, 3>(0, 0) * g.r_home.transpose();
    const Eigen::Vector3d p_16 = t_rev.block<3, 1>(0, 3) - r_06 * g.tool - g.our_p[0];
    d2[i] = p_16.dot(p_16);
    if (i == 0) {  // the q6-invariant SP3 scalars, as sp4_both(-p2, a2, p3, .) forms them
      const Eigen::Vector3d k = g.axes[2], h = -p2, pp = p3;
      const double a = h.dot(pp) - k.dot(pp) * h.dot(k);
      const double b = h.dot(k.cross(pp));
      r = std::hypot(a, b);
      center = k.dot(pp) * h.dot(k);
      s_const = pp.dot(pp) + p2.dot(p2);
    }
  }
  Eigen::Matrix3d basis;
  for (int i = 0; i < 3; ++i) basis.row(i) << 1.0, std::cos(samples[i]), std::sin(samples[i]);
  const Eigen::Vector3d abc = basis.colPivHouseholderQr().solve(d2);
  const double rho = std::hypot(abc[1], abc[2]), phi = std::atan2(abc[2], abc[1]);
  const double lo = s_const - abc[0] - 2.0 * (center + r);
  const double hi = s_const - abc[0] - 2.0 * (center - r);
  if (rho < 1e-15) return (lo <= 0.0 && 0.0 <= hi) ? std::vector<Interval>{{-M_PI, M_PI}} : std::vector<Interval>{};
  const double c_lo = std::max(lo / rho, -1.0), c_hi = std::min(hi / rho, 1.0);
  if (c_lo > c_hi) return {};
  if (c_lo <= -1.0 && c_hi >= 1.0) return {{-M_PI, M_PI}};
  const double a_hi = std::acos(c_lo), a_lo = std::acos(c_hi);
  std::vector<Interval> raw;
  if (a_lo <= 0.0)
    raw = {{phi - a_hi, phi + a_hi}};
  else if (a_hi >= M_PI - 1e-15)
    raw = {{phi + a_lo, phi + 2.0 * M_PI - a_lo}};
  else
    raw = {{phi + a_lo, phi + a_hi}, {phi - a_hi, phi - a_lo}};
  std::vector<Interval> out;
  for (const auto& iv : raw) {
    const double wa = wrap_pi(iv.lo), wb = wrap_pi(iv.hi);
    if (wa <= wb) {
      out.push_back({wa, wb});
    } else {
      out.push_back({wa, M_PI});
      out.push_back({-M_PI, wb});
    }
  }
  std::sort(out.begin(), out.end(), [](const Interval& p, const Interval& q) { return p.lo < q.lo; });
  return out;
}

// Every chart at one pose for a spherical-shoulder + offset-wrist 7R arm. Chart
// i = 8*k + slot lives on reachable arc k. Building costs microseconds (the arcs
// are closed form); q(t) and locate(q) never need a domain; a slot's domain
// within an arc is computed on first request, per arc, and cached.
struct SphericalShoulderCharts {
  JointConsts<7> c;  // the chain itself (for the Jacobian tangent)
  Eigen::Matrix<double, 3, 48> coef;
  Pose t_rev;
  Tolerances tol;
  std::vector<Interval> arcs;  // exact elbow arcs
  using SlotDomains = std::array<std::vector<Interval>, kSlots>;
  mutable std::vector<std::optional<SlotDomains>> cache;  // per arc

  static SphericalShoulderCharts build(const JointConsts<7>& c,
                                       const Eigen::Matrix<double, 3, 48>& coef, const Pose& T,
                                       const Tolerances& tol = {}) {
    SphericalShoulderCharts f;
    f.c = c;
    f.coef = coef;
    f.t_rev = T.inverse();
    f.tol = tol;
    f.arcs = elbow_arcs(coef, f.t_rev);
    f.cache.resize(f.arcs.size());
    return f;
  }

  int n_charts() const { return kSlots * static_cast<int>(arcs.size()); }
  int slot_of(int chart) const { return chart % kSlots; }
  int arc_of(int chart) const { return chart / kSlots; }
  std::array<int, 4> label(int chart) const {
    const int s = slot_of(chart);
    return {(s >> 2) & 1, (s >> 1) & 1, s & 1, arc_of(chart)};
  }
  static double param_of(const std::array<double, 7>& q) { return wrap_pi(q[6]); }

  int arc_containing(double t) const {
    for (int k = 0; k < static_cast<int>(arcs.size()); ++k)
      if (arcs[k].lo - kContainsTol <= t && t <= arcs[k].hi + kContainsTol) return k;
    return -1;
  }

  // Per-slot domains within arc k: grid scan, bisection of every interior
  // boundary, fold-seeded slivers, merge. Mirrors ssik.chart._arc_domains.
  const SlotDomains& arc_domains(int k) const {
    if (cache[k]) return *cache[k];
    const double a = arcs[k].lo, b = arcs[k].hi;
    auto valid_at = [&](int slot, double t) {
      return sh::slot_eval(coef, t_rev, t, tol, true).valid[slot];
    };
    auto refine = [&](int slot, double t_in, double t_out) {
      while (std::abs(t_out - t_in) > kDomainTol) {
        const double m = 0.5 * (t_in + t_out);
        (valid_at(slot, m) ? t_in : t_out) = m;
      }
      return t_in;
    };
    std::array<double, kDomainGrid> grid;
    std::array<std::array<bool, kSlots>, kDomainGrid> valid;
    for (int i = 0; i < kDomainGrid; ++i) {
      grid[i] = a + (b - a) * i / (kDomainGrid - 1);
      valid[i] = sh::slot_eval(coef, t_rev, grid[i], tol, true).valid;
    }
    SlotDomains domains;
    for (int slot = 0; slot < kSlots; ++slot) {
      int i = 0;
      while (i < kDomainGrid) {
        if (!valid[i][slot]) { ++i; continue; }
        int j = i;
        while (j + 1 < kDomainGrid && valid[j + 1][slot]) ++j;
        const double lo = i == 0 ? grid[0] : refine(slot, grid[i], grid[i - 1]);
        const double hi = j == kDomainGrid - 1 ? grid[j] : refine(slot, grid[j], grid[j + 1]);
        domains[slot].push_back({lo, hi});
        i = j + 1;
      }
    }
    // Slivers thinner than a grid cell, seeded from every refined fold boundary.
    std::vector<double> bounds;
    for (const auto& d : domains)
      for (const auto& iv : d) {
        bounds.push_back(iv.lo);
        bounds.push_back(iv.hi);
      }
    const double step = (b - a) / (kDomainGrid - 1);
    for (int slot = 0; slot < kSlots; ++slot) {
      for (double x : bounds)
        for (double t : {x - kSeedEps, x + kSeedEps}) {
          if (!(a < t && t < b)) continue;
          bool covered = false;
          for (const auto& iv : domains[slot])
            if (iv.lo - 2 * kSeedEps <= t && t <= iv.hi + 2 * kSeedEps) covered = true;
          if (covered || !valid_at(slot, t)) continue;
          const int cell = std::min(static_cast<int>((t - a) / step), kDomainGrid - 2);
          if (valid[cell][slot] || valid[cell + 1][slot]) continue;
          domains[slot].push_back({refine(slot, t, grid[cell]), refine(slot, t, grid[cell + 1])});
        }
      std::sort(domains[slot].begin(), domains[slot].end(),
                [](const Interval& p, const Interval& q) { return p.lo < q.lo; });
      std::vector<Interval> merged;
      for (const auto& iv : domains[slot]) {
        if (!merged.empty() && iv.lo <= merged.back().hi + 2 * kSeedEps)
          merged.back().hi = std::max(merged.back().hi, iv.hi);
        else
          merged.push_back(iv);
      }
      domains[slot] = std::move(merged);
    }
    cache[k] = std::move(domains);
    return *cache[k];
  }

  const std::vector<Interval>& domain(int chart) const { return arc_domains(arc_of(chart))[slot_of(chart)]; }
  bool nonempty(int chart) const { return !domain(chart).empty(); }

  // q(t) on one chart; false (q untouched) outside its arc or where the slot is
  // infeasible at t. No domain needed.
  bool q(int chart, double t, std::array<double, 7>& out) const {
    const auto& arc = arcs[arc_of(chart)];
    if (!(arc.lo - kContainsTol <= t && t <= arc.hi + kContainsTol)) return false;
    const sh::SlotEval ev = sh::slot_eval(coef, t_rev, t, tol);
    if (!ev.valid[slot_of(chart)]) return false;
    out = ev.q[slot_of(chart)];
    return true;
  }

  // dq/dt on one chart by implicit differentiation of FK(q(t)) = T with q6 = t:
  // J[:, :6] dq' = -J[:, 6] on the spatial Jacobian (ssik.chart._jacobian_tangent).
  // false off the branch or at an exact fold (singular 6x6).
  bool tangent(int chart, double t, std::array<double, 7>& out) const {
    std::array<double, 7> q0;
    if (!q(chart, t, q0)) return false;
    const Eigen::Matrix<double, 6, 7> jac = spatial_jacobian<7>(c, q0);
    const Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> lu(jac.template leftCols<6>());
    if (!lu.isInvertible()) return false;
    const Eigen::Matrix<double, 6, 1> dq = lu.solve(-jac.col(6));
    for (int j = 0; j < 6; ++j) out[j] = dq[j];
    out[6] = 1.0;
    return true;
  }

  // In-limits arcs of one chart (request A3): per domain interval the exact
  // feasible sub-intervals of joints 0..5 (feasible_arcs_bounded), intersected
  // with joint 6's own range (t is q6). Mirrors ssik.chart._q6_limit_arcs.
  std::vector<Interval> in_limits(int chart, const std::array<std::array<double, 2>, 7>& limits) const {
    std::vector<feasible::Arc> lim(7);
    for (int i = 0; i < 7; ++i) lim[i] = {limits[i][0], limits[i][1]};
    auto q_scalar = [&](double t) {
      std::array<double, 7> qv;
      if (!q(chart, t, qv)) qv.fill(std::numeric_limits<double>::quiet_NaN());
      return std::vector<double>(qv.begin(), qv.end());
    };
    std::vector<Interval> out;
    for (const auto& iv : domain(chart)) {
      std::vector<double> grid(kDomainGrid);
      for (int i = 0; i < kDomainGrid; ++i) grid[i] = iv.lo + (iv.hi - iv.lo) * i / (kDomainGrid - 1);
      // t is q6 on [-pi, pi]; joint 6's range may sit on another turn of it.
      feasible::Arcs own;
      for (int k = -1; k <= 1; ++k) own.emplace_back(limits[6][0] + k * 2.0 * M_PI, limits[6][1] + k * 2.0 * M_PI);
      const feasible::Arcs arcs = feasible::intersect(
          feasible::feasible_arcs_bounded(q_scalar, {0, 1, 2, 3, 4, 5}, lim, grid), own);
      for (const auto& a : arcs) out.push_back({a.first, a.second});
    }
    return out;
  }

  // Inverse chart map: index of the chart q lies on (or -1), with t and the
  // wrap-Linf mismatch there. One slot evaluation.
  int locate(const std::array<double, 7>& qv, double match_tol, double& t, double& dist) const {
    t = param_of(qv);
    dist = std::numeric_limits<double>::quiet_NaN();
    const int k = arc_containing(t);
    if (k < 0) return -1;
    const sh::SlotEval ev = sh::slot_eval(coef, t_rev, t, tol);
    for (int s = 0; s < kSlots; ++s) {
      if (!ev.valid[s]) continue;
      const double d = wrap_dist(ev.q[s], qv);
      if (d <= match_tol) {
        dist = d;
        return kSlots * k + s;
      }
    }
    return -1;
  }
};

// ---------------------------------------------------------------------------
// srs: charts are the <=8 closed-form branches over the swivel circle
// ---------------------------------------------------------------------------

struct SrsCharts {
  JointConsts<7> c;
  std::array<Eigen::Vector3d, 7> n;
  SrsConsts s;
  Pose T;
  Eigen::Vector3d u_p1, u_p2;
  std::vector<srs_swivel::Branch> branches;  // Branch::n points at this->n

  SrsCharts() = default;
  SrsCharts(const SrsCharts&) = delete;
  SrsCharts& operator=(const SrsCharts&) = delete;

  // Builds in place (the branches hold a pointer to n, so the object must not move).
  void init(const JointConsts<7>& consts, const SrsConsts& srs, const Pose& pose) {
    c = consts;
    s = srs;
    T = pose;
    for (int i = 0; i < 7; ++i) n[i] = c.axis[i].normalized();
    const Eigen::Vector3d W_t = T.block<3, 1>(0, 3) - T.block<3, 3>(0, 0) * s.ee_offset_local;
    const Eigen::Vector3d SW = W_t - s.shoulder_pivot;
    const double d_sw = SW.norm();
    Eigen::Vector3d u_sw = d_sw < 1e-12 ? Eigen::Vector3d(0, 0, 1) : Eigen::Vector3d(SW / d_sw);
    srs_detail::swivel_basis(u_sw, u_p1, u_p2);
    branches = srs_swivel::enumerate_branches(n, s, T);
  }

  int size() const { return static_cast<int>(branches.size()); }
  // (elbow root index, shoulder sign, wrist sign): enumerate_branches emits 4 per root.
  std::array<int, 3> label(int chart) const {
    return {chart / 4, branches[chart].s_sgn, branches[chart].w_sgn};
  }
  std::array<double, 7> q(int chart, double psi) const { return branches[chart].q(psi); }

  // Joint rates of R = Rot(n0,q0) Rot(n1,q1) Rot(n2,q2) with spatial angular
  // velocity omega: solve [n0, R0 n1, R0 R1 n2] rates = omega (singular at gimbal lock).
  static Eigen::Vector3d rates_3axis(const Eigen::Vector3d& n0, const Eigen::Vector3d& n1,
                                     const Eigen::Vector3d& n2, double q0, double q1,
                                     const Eigen::Vector3d& omega) {
    const Eigen::Matrix3d r0 = rotation_matrix(n0, q0);
    Eigen::Matrix3d cols;
    cols.col(0) = n0;
    cols.col(1) = r0 * n1;
    cols.col(2) = r0 * rotation_matrix(n1, q1) * n2;
    return cols.colPivHouseholderQr().solve(omega);
  }

  // Closed-form dq/dpsi (request B2): the shoulder's angular velocity in psi is
  // u_sw (rotor derivative of the swivel orbit), q3 is fixed, and the wrist
  // residual R_res = A^T R_t R_post^T with A = R_sh Rot(n3, q3) has angular
  // velocity -A^T u_sw. Mirrors ssik.chart._srs_tangent.
  std::array<double, 7> tangent(int chart, double psi) const {
    const auto& br = branches[chart];
    const std::array<double, 7> qv = br.q(psi);
    const Eigen::Vector3d sh = rates_3axis(n[0], n[1], n[2], qv[0], qv[1], br.u_sw);
    const Eigen::Matrix3d a = rotation_matrix(br.u_sw, psi) * br.R_sh0 * rotation_matrix(n[3], br.q3);
    const Eigen::Vector3d wr = rates_3axis(n[4], n[5], n[6], qv[4], qv[5], Eigen::Vector3d(-a.transpose() * br.u_sw));
    return {sh[0], sh[1], sh[2], 0.0, wr[0], wr[1], wr[2]};
  }

  // In-limits arcs of one swivel chart (request A3): the elbow is constant along
  // the swivel and checked once; the other six joints give exact periodic arcs.
  std::vector<Interval> in_limits(int chart, const std::array<std::array<double, 2>, 7>& limits) const {
    const auto& br = branches[chart];
    const double q3 = feasible::to_limits(br.q3, limits[3][0], limits[3][1]);
    if (!(limits[3][0] <= q3 && q3 <= limits[3][1])) return {};
    std::vector<feasible::Arc> lim(7);
    for (int i = 0; i < 7; ++i) lim[i] = {limits[i][0], limits[i][1]};
    static const std::vector<double> grid = feasible::param_grid();
    auto q_scalar = [&](double psi) {
      const std::array<double, 7> qv = br.q(psi);
      return std::vector<double>(qv.begin(), qv.end());
    };
    std::vector<Interval> out;
    for (const auto& a : feasible::feasible_arcs(q_scalar, {0, 1, 2, 4, 5, 6}, lim, grid))
      out.push_back({a.first, a.second});
    return out;
  }

  double param_of(const std::array<double, 7>& qv) const {
    const auto [R, p] = frame_at_joint<7>(c, qv, s.elbow_index);
    const Eigen::Vector3d e = p - s.shoulder_pivot;
    return std::atan2(e.dot(u_p2), e.dot(u_p1));
  }

  int locate(const std::array<double, 7>& qv, double match_tol, double& psi, double& dist) const {
    psi = param_of(qv);
    for (int i = 0; i < size(); ++i) {
      const double d = wrap_dist(branches[i].q(psi), qv);
      if (d <= match_tol) {
        dist = d;
        return i;
      }
    }
    dist = std::numeric_limits<double>::quiet_NaN();
    return -1;
  }
};

}  // namespace ssik::chart
