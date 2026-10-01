// Self-contained spherical-shoulder + offset-wrist 7R artifact solve (#551): the
// C++ replica of seven_r.spherical_shoulder{,_polished}.solve (franka/fr3 exact;
// xarm7/gen72 approximate -> LM-polish). Redundancy is the last joint q6 (He &
// Liu 2021): lock q6, reverse the chain -> a tier-0 spherical-wrist 6R solved
// closed-form by SP3->SP2->SP4->SP1x2. The reversed geometry is exactly affine in
// {cos q6, sin q6, 1}, so a (3,48) coef matrix is baked once (emit time) and any
// q6 is {cos,sin,1} @ coef -- no chain rebuild at runtime.
//
// Redundancy resolution (default path): an SP3-margin reachability bracket over
// q6 in [-pi, pi], each reachable interval grid-sampled (16 pts), every closed
// branch FK-gated (base) or LM-polished (polished), deduped. When no sampled
// solution survives the limit filter, the in-limits fallback runs before any
// rescue, as in Python: the exact q6-tracking resolver
// (spherical_shoulder.resolve_in_limits) for the base class, and the sweep
// polished then held to the limits (spherical_shoulder_polished.resolve_in_limits)
// for the approximate one.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/newton.hpp"  // lm_refine
#include "ssik_cpp/rescue.hpp"
#include "ssik_cpp/seven_r/feasible_arcs.hpp"
#include "ssik_cpp/seven_r/minimax.hpp"
#include "ssik_cpp/sp6.hpp"  // detail::wrap_pi
#include "ssik_cpp/subproblems.hpp"

namespace ssik {

// Baked reversed-lock-6 geometry: (3,48) affine coefficients in {cos q6, sin q6,
// 1}. Rows are the basis; the 48 cols are axes(18) + offsets(18) + tool(3) +
// r_home(9), matching spherical_shoulder._bake.
struct SphericalShoulderConsts {
  Eigen::Matrix<double, 3, 48> coef = Eigen::Matrix<double, 3, 48>::Zero();
};

inline constexpr int kShBracketGrid = 90;   // _BRACKET_GRID
inline constexpr int kShSampleGrid = 16;    // _SAMPLE_GRID
inline constexpr double kShFkAtol = 1e-10;  // _FK_ATOL (base FK gate + polish accept)
inline constexpr double kShDedupAtol = 1e-3;

namespace sh_detail {

struct Geom {
  std::array<Eigen::Vector3d, 6> axes;   // unit
  std::array<Eigen::Vector3d, 6> our_p;  // joint origins
  Eigen::Vector3d tool;
  Eigen::Matrix3d r_home;
};

// {cos q6, sin q6, 1} @ coef, unpacked (axes normalized) -- _eval_geom.
inline Geom eval_geom(const Eigen::Matrix<double, 3, 48>& coef, double q6) {
  Eigen::RowVector3d b(std::cos(q6), std::sin(q6), 1.0);
  const Eigen::Matrix<double, 1, 48> v = b * coef;
  Geom g;
  for (int i = 0; i < 6; ++i) {
    g.axes[i] = v.segment<3>(3 * i).transpose();
    g.axes[i].normalize();
    g.our_p[i] = v.segment<3>(18 + 3 * i).transpose();
  }
  g.tool = v.segment<3>(36).transpose();
  for (int r = 0; r < 3; ++r)
    for (int col = 0; col < 3; ++col) g.r_home(r, col) = v(39 + 3 * r + col);
  return g;
}

inline Eigen::Matrix3d rot(const Eigen::Vector3d& k, double th) {
  return Eigen::AngleAxisd(th, k).toRotationMatrix();
}

// SP3 elbow-solvability margin at q6 (>= 0 on a superset of the reachable set) --
// _sp3_reach_margins, scalar form.
inline double reach_margin(const Eigen::Matrix<double, 3, 48>& coef, const Pose& t_rev, double q6) {
  const Geom g = eval_geom(coef, q6);
  const Eigen::Vector3d p2 = g.our_p[2];
  const Eigen::Vector3d p3 = g.our_p[3] + g.our_p[4] + g.our_p[5];
  const Eigen::Matrix3d r_06 = t_rev.block<3, 3>(0, 0) * g.r_home.transpose();
  const Eigen::Vector3d p_16 = t_rev.block<3, 1>(0, 3) - r_06 * g.tool - g.our_p[0];
  const Eigen::Vector3d k = g.axes[2], pp = p3, qq = -p2;
  const double target = 0.5 * (pp.dot(pp) + qq.dot(qq) - p_16.dot(p_16));
  const double center = qq.dot(k) * pp.dot(k);
  const double qperp = (qq - k * qq.dot(k)).norm();
  const double pperp = (pp - k * pp.dot(k)).norm();
  return qperp * pperp - std::abs(target - center);
}

// Reachable q6 sub-intervals of [lo, hi] (default [-pi, pi]): margin >= 0
// brackets on a 90-grid, padded one step, merged (_reachable_intervals + merge).
inline std::vector<std::array<double, 2>> reachable_intervals(
    const Eigen::Matrix<double, 3, 48>& coef, const Pose& t_rev, double lo = -M_PI,
    double hi = M_PI) {
  std::array<double, kShBracketGrid> grid;
  std::array<bool, kShBracketGrid> m;
  for (int i = 0; i < kShBracketGrid; ++i) {
    grid[i] = lo + (hi - lo) * i / (kShBracketGrid - 1);  // linspace endpoint=True
    m[i] = reach_margin(coef, t_rev, grid[i]) >= 0.0;
  }
  std::vector<std::array<double, 2>> raw;
  int k = 0;
  while (k < kShBracketGrid) {
    if (m[k]) {
      int j = k;
      while (j + 1 < kShBracketGrid && m[j + 1]) ++j;
      raw.push_back({grid[std::max(k - 1, 0)], grid[std::min(j + 1, kShBracketGrid - 1)]});
      k = j + 1;
    } else {
      ++k;
    }
  }
  // merge overlapping/adjacent intervals.
  std::vector<std::array<double, 2>> out;
  for (const auto& iv : raw) {
    if (!out.empty() && iv[0] <= out.back()[1]) {
      out.back()[1] = std::max(out.back()[1], iv[1]);
    } else {
      out.push_back(iv);
    }
  }
  return out;
}

// All q0..q6 IK branches at a fixed q6 (reversed spherical_two_intersecting recipe
// SP3->SP2->SP4->SP1x2, flipped back + q6 appended) -- _closed_branches.
inline std::vector<std::array<double, 7>> closed_branches(
    const Eigen::Matrix<double, 3, 48>& coef, const Pose& t_rev, double q6, const Tolerances& tol) {
  const Geom g = eval_geom(coef, q6);
  const Eigen::Vector3d p2 = g.our_p[2];
  const Eigen::Vector3d p3 = g.our_p[3] + g.our_p[4] + g.our_p[5];
  const Eigen::Matrix3d r_06 = t_rev.block<3, 3>(0, 0) * g.r_home.transpose();
  const Eigen::Vector3d p_16 = t_rev.block<3, 1>(0, 3) - r_06 * g.tool - g.our_p[0];

  std::vector<std::array<double, 7>> out;
  const auto t3 = sp3(g.axes[2], p3, -p2, p_16.norm(), tol).first;
  for (double q3 : t3) {
    const auto t12 = sp2(-g.axes[0], g.axes[1], p_16, p2 + rot(g.axes[2], q3) * p3, tol).first;
    for (const auto& [q1, q2] : t12) {
      const Eigen::Matrix3d r_36 =
          rot(-g.axes[2], q3) * rot(-g.axes[1], q2) * rot(-g.axes[0], q1) * r_06;
      const auto t5 = sp4(g.axes[3], g.axes[4], g.axes[5], g.axes[3].dot(r_36 * g.axes[5]), tol).first;
      for (double q5 : t5) {
        const double q4 = sp1(g.axes[3], rot(g.axes[4], q5) * g.axes[5], r_36 * g.axes[5], tol).first;
        const double q6i =
            sp1(-g.axes[5], rot(-g.axes[4], q5) * g.axes[3], r_36.transpose() * g.axes[3], tol).first;
        // reversed q = [q1,q2,q3,q4,q5,q6i]; map back = flip; append q6.
        const std::array<double, 6> qr = {q1, q2, q3, q4, q5, q6i};
        std::array<double, 7> q;
        for (int i = 0; i < 6; ++i) q[i] = qr[5 - i];
        q[6] = q6;
        out.push_back(q);
      }
    }
  }
  return out;
}

// Round-to-6-decimals dedup key equality (_MERGE_KEY), mirroring the Python
// seen-set on np.round(q, 6).
inline bool round6_equal(const std::array<double, 7>& a, const std::array<double, 7>& b) {
  for (int i = 0; i < 7; ++i)
    if (std::llround(a[i] * 1e6) != std::llround(b[i] * 1e6)) return false;
  return true;
}

// --- slot-indexed closed form (mirrors _slot_grid) ---------------------------

inline constexpr int kShSlots = 8;                // N_SLOTS
inline constexpr double kShLockTol = 1e-9;        // _LOCK_TOL
inline constexpr double kShTangentSnap = 1e-15;   // _TANGENT_SNAP


// Batched-subproblem replicas: both roots + feasibility, never fewer roots, so
// the slot index is a stable label (Python _sp1_batch / _sp4_batch / _sp2_batch).
// Angle 0 where p lies on the axis (the angle is free; atan2 of rounding noise
// would be arbitrary) -- the canonical representative, as _sp1_batch.
inline double sp1_both(const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                       const Eigen::Vector3d& q) {
  const Eigen::Vector3d kxp = k.cross(p);
  if (kxp.norm() <= kShLockTol * p.norm()) return 0.0;
  return sp1_angle_kxp(k, kxp, q);
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
  if (ratio >= 1.0 - kShTangentSnap) ratio = 1.0;
  if (ratio <= -1.0 + kShTangentSnap) ratio = -1.0;
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
  if (std::abs(gss) <= kShTangentSnap * std::max(pp, qq)) gss = 0.0;  // tangent case (as _sp2_batch)
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
  std::array<std::array<double, 7>, kShSlots> q{};
  std::array<bool, kShSlots> valid{};
};

// Every slot at one q6 (mirrors _slot_grid at a single grid point). With
// valid_only the two SP1 stages are skipped (q is left unset).
inline SlotEval slot_eval(const Eigen::Matrix<double, 3, 48>& coef, const Pose& t_rev, double q6,
                          const Tolerances& tol, bool valid_only = false) {
  const Geom g = eval_geom(coef, q6);
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
        if (a[3].cross(a5_mid).norm() <= kShLockTol) {
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

}  // namespace sh_detail

// Default-path candidates for one target: reachable-interval q6 sweep -> closed
// branches -> FK gate (base) or LM-polish vs true FK (polished) -> round-6 dedup.
inline std::vector<Solution<7>> spherical_shoulder_core(const JointConsts<7>& c,
                                                        const SphericalShoulderConsts& sh,
                                                        const Pose& T, bool polished,
                                                        int refinement_max_iters) {
  const Tolerances tol;
  const Pose t_rev = T.inverse();
  std::vector<Solution<7>> out;
  std::vector<std::array<double, 7>> seen;
  for (const auto& iv : sh_detail::reachable_intervals(sh.coef, t_rev)) {
    for (int gi = 0; gi < kShSampleGrid; ++gi) {
      const double q6 = iv[0] + (iv[1] - iv[0]) * gi / (kShSampleGrid - 1);  // linspace
      for (const auto& q : sh_detail::closed_branches(sh.coef, t_rev, q6, tol)) {
        std::array<double, 7> qc = q;
        double resid = (fk<7>(c, qc) - T).norm();
        if (polished && resid > kShFkAtol) {
          // approximate arm: LM-polish the closed-form seed to machine precision
          // (lm_refine_batch semantics: fixed 1e-9 damping, no stall guard).
          auto r = lm_refine<7>(c, qc, T, kShFkAtol, refinement_max_iters, 2.0, 2, 1e-9);
          if (!r) continue;
          qc = r->first;
          resid = r->second;
        }
        if (resid > kShFkAtol) continue;
        // Base path dedups exact solutions by the round-6 merge key (_MERGE_KEY);
        // the polished path clusters LM-landed points in wrap-to-pi (1e-3), like
        // polish_candidates (round-6 would leak near-duplicates the LM scatters).
        bool dup = false;
        for (const auto& s : seen) {
          if (polished) {
            bool close = true;
            for (int i = 0; i < 7; ++i)
              if (std::abs(detail::wrap_pi(qc[i] - s[i])) > kShDedupAtol) {
                close = false;
                break;
              }
            if (close) {
              dup = true;
              break;
            }
          } else if (sh_detail::round6_equal(qc, s)) {
            dup = true;
            break;
          }
        }
        if (dup) continue;
        seen.push_back(qc);
        out.push_back(Solution<7>{qc, resid, polished ? Refinement::Lm : Refinement::None});
      }
    }
  }
  return out;
}

// --- exact in-limits resolver (the #359 in-limits fallback) -------------------

inline constexpr int kShTrackGrid = 180;         // _TRACK_GRID
inline constexpr double kShTrackBreak = 0.4;     // _track_branches continuity break
inline constexpr int kShTrackMinPoints = 4;      // _track_branches minimum curve length
inline constexpr double kShLimitSlack = kLimitBand;  // in-limits acceptance (_polish._LIMIT_SLACK)
inline constexpr int kShPolishMaxIters = 30;     // spherical_shoulder_polished._POLISH_MAX_ITERS

using ShLimits = std::array<std::array<double, 2>, 7>;

namespace sh_detail {

// np.linspace(a, b, n): start + k * step, last point exactly b.
inline std::vector<double> linspace(double a, double b, int n) {
  std::vector<double> g(n);
  const double step = (b - a) / (n - 1);
  for (int k = 0; k < n; ++k) g[k] = k * step + a;
  g[n - 1] = b;
  return g;
}

// Every valid branch at each q6 of `grid`, in slot order (_closed_branches_grid).
inline std::vector<std::vector<std::array<double, 7>>> branches_grid(
    const Eigen::Matrix<double, 3, 48>& coef, const Pose& t_rev, const std::vector<double>& grid,
    const Tolerances& tol) {
  std::vector<std::vector<std::array<double, 7>>> per(grid.size());
  for (std::size_t k = 0; k < grid.size(); ++k) {
    const SlotEval ev = slot_eval(coef, t_rev, grid[k], tol);
    for (int s = 0; s < kShSlots; ++s)
      if (ev.valid[s]) per[k].push_back(ev.q[s]);
  }
  return per;
}

// One tracked branch: its values on the contiguous grid run it covers.
struct TrackedCurve {
  std::vector<double> g;
  std::vector<std::array<double, 7>> q;
};

// Link the per-grid-point branches into continuous curves by greedy nearest
// neighbour (_track_branches): a curve stops where the grid has no branch, where
// the nearest one is already taken, or where it is more than 0.4 away; curves
// shorter than 4 points are dropped.
inline std::vector<TrackedCurve> track_branches(
    const std::vector<std::vector<std::array<double, 7>>>& per, const std::vector<double>& grid) {
  const std::size_t n = grid.size();
  std::vector<std::vector<bool>> used(n);
  for (std::size_t k = 0; k < n; ++k) used[k].assign(per[k].size(), false);
  auto dist = [](const std::array<double, 7>& a, const std::array<double, 7>& b) {
    double s = 0.0;
    for (int i = 0; i < 7; ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(s);
  };
  std::vector<TrackedCurve> curves;
  for (std::size_t k0 = 0; k0 < n; ++k0) {
    for (std::size_t b0 = 0; b0 < per[k0].size(); ++b0) {
      if (used[k0][b0]) continue;
      TrackedCurve curve;
      curve.g.push_back(grid[k0]);
      curve.q.push_back(per[k0][b0]);
      used[k0][b0] = true;
      std::array<double, 7> prev = per[k0][b0];
      for (std::size_t k = k0 + 1; k < n; ++k) {
        if (per[k].empty()) break;
        std::size_t j = 0;
        double dj = dist(per[k][0], prev);
        for (std::size_t m = 1; m < per[k].size(); ++m) {
          const double dm = dist(per[k][m], prev);
          if (dm < dj) {  // first minimum, as np.argmin
            dj = dm;
            j = m;
          }
        }
        if (used[k][j] || dj > kShTrackBreak) break;
        curve.g.push_back(grid[k]);
        curve.q.push_back(per[k][j]);
        used[k][j] = true;
        prev = per[k][j];
      }
      if (static_cast<int>(curve.g.size()) >= kShTrackMinPoints) curves.push_back(std::move(curve));
    }
  }
  return curves;
}

// np.interp of one tracked curve at t (clamped to its end values outside it).
inline std::vector<double> interp_curve(const TrackedCurve& cv, double t) {
  const auto& g = cv.g;
  std::vector<double> out(7);
  if (t < g.front() || t >= g.back()) {
    const auto& q = t < g.front() ? cv.q.front() : cv.q.back();
    for (int i = 0; i < 7; ++i) out[i] = q[i];
    return out;
  }
  const std::size_t j =
      static_cast<std::size_t>(std::upper_bound(g.begin(), g.end(), t) - g.begin()) - 1;
  for (int i = 0; i < 7; ++i) {
    const double slope = (cv.q[j + 1][i] - cv.q[j][i]) / (g[j + 1] - g[j]);
    out[i] = slope * (t - g[j]) + cv.q[j][i];
  }
  return out;
}

// In-limits q vectors for one reachable interval (_solutions_in_interval): track
// each branch, take its exact in-limits q6 arcs, and at each arc centre emit
// every closed branch that lands in limits (wrapped to them) and closes FK.
inline std::vector<std::array<double, 7>> solutions_in_interval(
    const JointConsts<7>& c, const Eigen::Matrix<double, 3, 48>& coef, const Pose& t_rev,
    const Pose& T, double a, double b, const ShLimits& limits, const Tolerances& tol) {
  static const std::vector<int> kSwept = {0, 1, 2, 3, 4, 5};  // _SWEPT
  feasible::Arcs lim_arcs(7);
  for (int i = 0; i < 7; ++i) lim_arcs[i] = {limits[i][0], limits[i][1]};

  const std::vector<double> grid = linspace(a, b, kShTrackGrid);
  std::vector<std::array<double, 7>> out;
  for (const TrackedCurve& cv : track_branches(branches_grid(coef, t_rev, grid, tol), grid)) {
    const auto q_scalar = [&cv](double t) { return interp_curve(cv, t); };
    for (const auto& [u, w] : feasible::feasible_arcs_bounded(q_scalar, kSwept, lim_arcs, cv.g)) {
      const double q6c = 0.5 * (u + w);
      for (const auto& q : closed_branches(coef, t_rev, q6c, tol)) {
        std::array<double, 7> qw;
        bool in_lim = true;
        for (int i = 0; i < 7; ++i) {
          qw[i] = feasible::to_limits(q[i], limits[i][0], limits[i][1]);
          in_lim = in_lim && limits[i][0] - kShLimitSlack <= qw[i] &&
                   qw[i] <= limits[i][1] + kShLimitSlack;
        }
        if (in_lim && (fk<7>(c, qw) - T).norm() <= kShFkAtol) out.push_back(qw);
      }
    }
  }
  return out;
}

// In-limits contacts when no branch has an in-limits arc (#662): the
// minimax-margin points of every slot chart over the reachable q6 intervals of
// [lo, hi] (the whole range when the SP3 margin rules out every q6 at an elbow
// fold whose slot gates still pass). On the exact class a slot's minimum is the
// contact itself; on the approximate class it seeds a walk along the true
// self-motion curve from each minimum. Mirrors spherical_shoulder.contacts.
inline std::vector<Solution<7>> contacts(const JointConsts<7>& c, const JointLimits<7>& lim,
                                         const Eigen::Matrix<double, 3, 48>& coef, const Pose& T,
                                         double lo, double hi, const ShLimits& limits,
                                         double fk_atol, bool exact_chart) {
  const Tolerances tol;
  const Pose t_rev = T.inverse();
  auto ivs = reachable_intervals(coef, t_rev, lo, hi);
  if (ivs.empty()) ivs.push_back({lo, hi});
  auto slot_q = [&](int slot, double t) {
    const SlotEval ev = slot_eval(coef, t_rev, t, tol);
    std::vector<double> out(7, std::numeric_limits<double>::quiet_NaN());
    if (ev.valid[slot]) out.assign(ev.q[slot].begin(), ev.q[slot].end());
    return out;
  };
  struct Minimum {
    double v;
    int slot;
    double t;
  };
  std::vector<Minimum> minima;
  for (const auto& iv : ivs) {
    const std::vector<double> grid = linspace(iv[0], iv[1], kShTrackGrid);
    for (int slot = 0; slot < kShSlots; ++slot) {
      auto q_scalar = [&, slot](double t) { return slot_q(slot, t); };
      for (const auto& [t, v] : minimax::chart_minima(q_scalar, grid, limits, /*periodic=*/false))
        minima.push_back({v, slot, t});
    }
  }
  std::sort(minima.begin(), minima.end(), [](const Minimum& x, const Minimum& y) {
    if (x.v != y.v) return x.v < y.v;
    if (x.slot != y.slot) return x.slot < y.slot;
    return x.t < y.t;
  });
  std::vector<Solution<7>> out;
  for (const Minimum& m : minima) {
    const std::vector<double> qv = slot_q(m.slot, m.t);
    std::array<double, 7> q;
    std::copy(qv.begin(), qv.end(), q.begin());
    if (exact_chart) {
      if (m.v > kBandCap) break;
      if (auto sol = minimax::place(c, lim, q, T, limits, fk_atol, Refinement::None))
        out.push_back(*sol);
    } else if (const auto qw = minimax::walk(c, q, T, limits)) {
      if (auto sol = minimax::place(c, lim, *qw, T, limits, fk_atol, Refinement::Lm))
        out.push_back(*sol);
    }
  }
  return minimax::dedup(out, kShDedupAtol);
}

}  // namespace sh_detail

// Exact in-limits IK for the exact class (spherical_shoulder.resolve_in_limits):
// the q6 redundancy resolved over the reachable intervals of joint 6's own range
// intersected with every joint's in-limits arcs, so a reachable in-limits target
// the coarse 16-sample sweep misses still gets an in-limits, FK-verified solution.
inline std::vector<Solution<7>> spherical_shoulder_resolve_in_limits(
    const JointConsts<7>& c, const SphericalShoulderConsts& sh, const Pose& T,
    const ShLimits& limits) {
  const Tolerances tol;
  const Pose t_rev = T.inverse();
  std::vector<Solution<7>> out;
  std::vector<std::array<double, 7>> seen;
  for (const auto& iv : sh_detail::reachable_intervals(sh.coef, t_rev, limits[6][0], limits[6][1])) {
    for (const auto& q :
         sh_detail::solutions_in_interval(c, sh.coef, t_rev, T, iv[0], iv[1], limits, tol)) {
      bool dup = false;
      for (const auto& s : seen)
        if (sh_detail::round6_equal(q, s)) {
          dup = true;
          break;
        }
      if (dup) continue;
      seen.push_back(q);
      out.push_back(Solution<7>{q, (fk<7>(c, q) - T).norm(), Refinement::None});
    }
  }
  return out;
}

// In-limits IK for the approximate class
// (spherical_shoulder_polished.resolve_in_limits): the default sweep's closed-form
// seeds over [-pi, pi], LM-polished against the true FK, held to the limits AFTER
// the polish (a polish moves a seed, so it can leave the limits -- #621), then
// cluster-merged keeping the lower residual. finalize does not re-filter a
// fallback's output, so nothing out of limits may leave here.
inline std::vector<Solution<7>> spherical_shoulder_polished_resolve_in_limits(
    const JointConsts<7>& c, const SphericalShoulderConsts& sh, const Pose& T,
    const ShLimits& limits) {
  const Tolerances tol;
  const Pose t_rev = T.inverse();
  std::vector<Solution<7>> kept;
  for (const auto& iv : sh_detail::reachable_intervals(sh.coef, t_rev)) {
    const std::vector<double> grid = sh_detail::linspace(iv[0], iv[1], kShSampleGrid);
    for (const auto& branch_list : sh_detail::branches_grid(sh.coef, t_rev, grid, tol)) {
      for (const auto& q : branch_list) {
        std::array<double, 7> qc = q;
        double resid = (fk<7>(c, qc) - T).norm();
        if (resid > kShFkAtol) {
          // lm_refine_batch semantics, as spherical_shoulder_core's polish.
          auto r = lm_refine<7>(c, qc, T, kShFkAtol, kShPolishMaxIters, 2.0, 2, 1e-9);
          if (!r) continue;
          qc = r->first;
          resid = r->second;
        }
        if (resid > kShFkAtol) continue;
        bool within = true;
        for (int i = 0; i < 7 && within; ++i)
          within = limits[i][0] - kShLimitSlack <= qc[i] && qc[i] <= limits[i][1] + kShLimitSlack;
        if (!within) continue;
        kept.push_back(Solution<7>{qc, resid, Refinement::Lm});
      }
    }
  }
  // dedup_by_wrap_close: first-match clusters, keeping the lower residual.
  std::vector<Solution<7>> out;
  for (const auto& cand : kept) {
    int dup = -1;
    for (std::size_t j = 0; j < out.size() && dup < 0; ++j) {
      bool close = true;
      for (int i = 0; i < 7; ++i)
        if (std::abs(detail::wrap_pi(cand.q[i] - out[j].q[i])) > kShDedupAtol) {
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

// Full artifact-contract solve for a spherical-shoulder arm (base or polished).
inline std::vector<Solution<7>> spherical_shoulder_artifact_solve(
    const JointConsts<7>& c, const SphericalShoulderConsts& sh, const JointLimits<7>& lim,
    const Pose& T, const ArtifactParams<7>& p, bool polished) {
  // Seeded numerical-tracking fast path (#380), as in srs.hpp: Newton-continue
  // from q_seed, kept only if it converges AND stays near the seed (seeded_track's
  // max_dist guard rejects a branch jump). Without it, seeded tracking falls back
  // to the sampled arm-angle manifold, whose nearest sample can sit radians from
  // the seed (#562). No in-limits fallback: a failing track falls through below.
  if (p.has_seed && p.max_solutions == 1) {
    const auto tracked = seeded_track<7>(c, p.q_seed, T);
    if (tracked) {
      const auto fast = finalize_solutions<7>({*tracked}, c, lim, T, p);
      if (!fast.empty()) return fast;
    }
  }

  const auto core = [&](const Pose& Tp) {
    return spherical_shoulder_core(c, sh, Tp, polished, p.refinement_max_iters);
  };
  // _joint_limits(kb): baked limits, [-pi, pi] where absent -- the box the
  // Python in-limits fallback resolves against.
  ShLimits limits;
  for (int i = 0; i < 7; ++i)
    limits[i] = lim.present[i] ? std::array<double, 2>{lim.lo[i], lim.hi[i]}
                               : std::array<double, 2>{-M_PI, M_PI};
  ArtifactParams<7> p_limits;
  // Intermediate pass: never lift here, or the lifts would be produced
  // twice (see ArtifactParams::enumerate_windings).
  p_limits.enumerate_windings = false;
  p_limits.respect_limits = p.respect_limits;
  p_limits.wrap_only = p.wrap_only;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  // Limit pass + #359 in-limits fallback (#615): the coarse sweep can miss a
  // thin in-limits q6 arc, so an empty limit-filtered set is resolved exactly
  // before the rescue gate below sees it.
  std::vector<Solution<7>> in_limits = finalize_solutions<7>(core(T), c, lim, T, p_limits, [&]() {
    auto sols = polished ? spherical_shoulder_polished_resolve_in_limits(c, sh, T, limits)
                         : spherical_shoulder_resolve_in_limits(c, sh, T, limits);
    if (sols.empty()) {  // a point or sliver no arc brackets (#662)
      const double lo = polished ? -M_PI : limits[6][0], hi = polished ? M_PI : limits[6][1];
      sols = sh_detail::contacts(c, lim, sh.coef, T, lo, hi, limits, kShFkAtol, !polished);
    }
    return sols;
  });
  if (in_limits.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    in_limits = finalize_solutions<7>(rescue_via_T_perturbation<7>(core, c, T), c, lim, T, p_limits);
  }
  ArtifactParams<7> p_seed = p;
  p_seed.respect_limits = false;
  // The single lifting stage (#562): this is the call that yields the returned
  // set. Skipped when the caller wanted the raw geometric set.
  p_seed.enumerate_windings = p.enumerate_windings && p.respect_limits;
  return finalize_solutions<7>(std::move(in_limits), c, lim, T, p_seed);
}

}  // namespace ssik
