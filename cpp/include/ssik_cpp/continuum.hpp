// Singular continua of a 6R solution set (#662): the native port of
// ssik.continuum, whose module docstring owns the definition (the cores'
// detection rule, the representative a locked wrist emits, and the slide every
// flagged solution takes before finalize). A change on one side must be
// mirrored on the other.
//
// The contract (docs/api.md, "Singular continua"): a seeded solve returns the
// point of the seed's continuum nearest the seed, within limits; an unseeded
// solve returns one representative per continuum, the point whose free joint is
// at 0, else the in-limit point nearest it along the continuum.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/dedup.hpp"
#include "ssik_cpp/finalize.hpp"  // kLimitBand, finalize_detail::wrap_to_pi
#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/ik_types.hpp"
#include "ssik_cpp/newton.hpp"  // spatial_jacobian, se3_log_residual
#include "ssik_cpp/rotation.hpp"
#include "ssik_cpp/subproblems.hpp"  // sp1_angle

namespace ssik {

// ssik.continuum.LOCK_TOL / RANK_TOL / NOT_FLAGGED / FREE_FROM_KERNEL.
inline constexpr double kLockTol = 1e-9;
inline constexpr double kAxisTol = 1e-12;  // ssik.continuum.AXIS_TOL
inline constexpr double kRankTol = 1e-4;
inline constexpr int kNotFlagged = -1;
inline constexpr int kFreeFromKernel = -2;

namespace continuum_detail {

// ssik.continuum's walk constants (STEP, STOP, MAX_STEPS, WALK_ARC,
// BISECT_STEPS, CORRECT_ITERS, WALK_FK, SLIDE_FK, FREE_MIN).
inline constexpr double kStep = 0.25;
inline constexpr double kStop = 1e-12;
inline constexpr int kMaxSteps = 64;
inline constexpr double kWalkArc = 2.0 * M_PI * 1.4142135623730951;
inline constexpr int kBisectSteps = 40;
inline constexpr int kCorrectIters = 8;
inline constexpr double kWalkFk = 1e-6;
inline constexpr double kSlideFk = 1e-10;
inline constexpr double kFreeMin = 0.1;

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Q6 = std::array<double, 6>;

inline Vec6 to_vec(const Q6& q) {
  Vec6 v;
  for (int i = 0; i < 6; ++i) v[i] = q[i];
  return v;
}

inline Q6 to_q(const Vec6& v) {
  Q6 q;
  for (int i = 0; i < 6; ++i) q[i] = v[i];
  return q;
}

// FK, Jacobian and the in-limits test the slide works with (ssik.continuum._Chain).
struct Chain {
  const JointConsts<6>& c;
  const JointLimits<6>& lim;
  const Pose& t;
  double floor;

  Chain(const JointConsts<6>& c_, const JointLimits<6>& lim_, const Pose& t_)
      : c(c_), lim(lim_), t(t_), floor(same_root_floor(t_)) {}

  double residual(const Vec6& x) const { return (fk<6>(c, to_q(x)) - t).norm(); }

  // Truncated Gauss-Newton onto the solution set; the best point seen.
  std::pair<Vec6, double> correct(Vec6 x) const {
    Pose f = fk<6>(c, to_q(x));
    double r = (f - t).norm();
    Vec6 best_x = x;
    double best_r = r;
    for (int it = 0; it < kCorrectIters; ++it) {
      if (r <= floor) break;
      const Mat6 jac = spatial_jacobian<6>(c, to_q(x));
      Eigen::JacobiSVD<Mat6> svd(jac, Eigen::ComputeFullU | Eigen::ComputeFullV);
      const Vec6 s = svd.singularValues();
      const Vec6 e = se3_log_residual(t * f.inverse());
      const Vec6 ue = svd.matrixU().transpose() * e;
      Vec6 dq = Vec6::Zero();
      for (int k = 0; k < 6; ++k)
        if (s[k] > kRankTol * s[0]) dq += svd.matrixV().col(k) * (ue[k] / s[k]);
      x = x + dq;
      f = fk<6>(c, to_q(x));
      r = (f - t).norm();
      if (!(r < best_r)) break;
      best_x = x;
      best_r = r;
    }
    return {best_x, best_r};
  }

  // The oriented tangent (last right singular vector), the null directions
  // (columns), and whether there is one.
  struct Tangent {
    Vec6 v;
    Eigen::Matrix<double, 6, Eigen::Dynamic> null;
    bool ok;
  };

  Tangent tangent(const Vec6& x) const {
    const Mat6 jac = spatial_jacobian<6>(c, to_q(x));
    Eigen::JacobiSVD<Mat6> svd(jac, Eigen::ComputeFullV);
    const Vec6 s = svd.singularValues();
    Vec6 v = svd.matrixV().col(5);
    int big = -1;
    for (int i = 0; i < 6; ++i)
      if (std::abs(v[i]) >= kFreeMin) big = i;
    if (big >= 0 && v[big] < 0.0) v = -v;
    int k = 0;
    for (int i = 0; i < 6; ++i)
      if (s[i] <= kRankTol * s[0]) ++k;
    Eigen::Matrix<double, 6, Eigen::Dynamic> null(6, k);
    for (int j = 0; j < k; ++j) null.col(j) = svd.matrixV().col(6 - k + j);
    return {v, null, k > 0};
  }

  // Whether every joint has a winding inside its limits (within kLimitBand).
  bool in_limits(const Vec6& x) const {
    constexpr double kTwoPi = 2.0 * M_PI;
    for (int i = 0; i < 6; ++i) {
      if (!lim.present[i] || c.type[i] != JointType::Revolute) continue;
      const double lo = lim.lo[i] - kLimitBand, hi = lim.hi[i] + kLimitBand;
      const double xi = x[i];
      const double k = std::ceil((lo - xi) / kTwoPi);
      bool any = false;
      for (double j = k - 1.0; j <= k + 1.0; j += 1.0) {
        const double v = xi + kTwoPi * j;
        if (lo <= v && v <= hi) any = true;
      }
      if (!any) return false;
    }
    return true;
  }
};

inline Vec6 wrap_all(const Vec6& a) {
  Vec6 out;
  for (int i = 0; i < 6; ++i) out[i] = finalize_detail::wrap_to_pi(a[i]);
  return out;
}

// The core's free joint, or the highest-index joint the null directions move
// by at least kFreeMin (sqrt(P_ii)).
inline int free_joint(int free, const Eigen::Matrix<double, 6, Eigen::Dynamic>& null) {
  if (free >= 0) return free;
  int out = -1;
  for (int i = 0; i < 6; ++i)
    if (std::sqrt(null.row(i).squaredNorm()) >= kFreeMin) out = i;
  return out;
}

// Step 3: from a corrected point of the continuum to the rule's point.
inline Vec6 walk_to_rule(const Chain& ch, Vec6 x, int free, const Vec6* seed) {
  bool jumped = false;
  for (int it = 0; it < kMaxSteps; ++it) {
    const auto tg = ch.tangent(x);
    if (!tg.ok) break;
    Vec6 d;
    if (seed != nullptr) {
      d = tg.null * (tg.null.transpose() * wrap_all(*seed - x));
    } else {
      const int f = free_joint(free, tg.null);
      if (f < 0) break;
      const Eigen::VectorXd pf = tg.null.row(f).transpose();
      const double g = pf.dot(pf);
      if (g < kFreeMin * kFreeMin) break;
      d = (-finalize_detail::wrap_to_pi(x[f]) / g) * (tg.null * pf);
    }
    const double n = d.norm();
    if (n <= kStop) break;
    std::pair<Vec6, double> y = ch.correct(x + (n > kStep ? Vec6(d * (kStep / n)) : d));
    if (y.second > kSlideFk) {
      if (seed == nullptr || jumped || n <= kStep) break;
      // Only near a continuum, where the seed may be an exact solution: take
      // the whole step once, as the core would set the free joint.
      jumped = true;
      y = ch.correct(x + d);
      if (y.second > kWalkFk) break;
    }
    x = y.first;
  }
  return x;
}

// Step 4, one direction: the first in-limit point and the arc to it.
inline std::optional<std::pair<Vec6, double>> walk_into_limits(const Chain& ch, const Vec6& x,
                                                                double sign) {
  const auto t0 = ch.tangent(x);
  if (!t0.ok) return std::nullopt;
  Vec6 v_prev = sign * t0.v;
  Vec6 y = x;
  double arc = 0.0;
  for (int k = 0; k < static_cast<int>(std::ceil(kWalkArc / kStep)); ++k) {
    const auto tg = ch.tangent(y);
    if (!tg.ok) return std::nullopt;
    Vec6 v = tg.v;
    if (v.dot(v_prev) < 0.0) v = -v;
    const auto [z, rz] = ch.correct(y + kStep * v);
    if (rz > kSlideFk) return std::nullopt;
    if (ch.in_limits(z)) {
      Vec6 out = y, inside = z;
      for (int b = 0; b < kBisectSteps; ++b) {
        const auto [m, rm] = ch.correct(out + 0.5 * (inside - out));
        if (rm > kSlideFk) break;
        if (ch.in_limits(m))
          inside = m;
        else
          out = m;
      }
      return std::make_pair(inside, arc + (inside - y).norm());
    }
    arc += (z - y).norm();
    y = z;
    v_prev = v;
  }
  return std::nullopt;
}

inline Solution<6> slide_one(const Chain& ch, const Solution<6>& sol, int free, const Vec6* seed,
                             bool respect_limits) {
  const Vec6 x0 = to_vec(sol.q);
  if (!ch.tangent(x0).ok) return sol;
  Vec6 x = ch.correct(x0).first;
  if (!ch.tangent(x).ok) return sol;
  x = walk_to_rule(ch, x, free, seed);
  if (respect_limits && !ch.in_limits(x)) {
    std::optional<std::pair<Vec6, double>> best;
    for (double sign : {1.0, -1.0}) {
      const auto hit = walk_into_limits(ch, x, sign);
      if (hit && (!best || hit->second < best->second)) best = hit;
    }
    if (best) x = best->first;
  }
  const double r = ch.residual(x);
  if (!(r <= kSlideFk)) return sol;
  return Solution<6>{to_q(x), r, sol.refinement};
}

}  // namespace continuum_detail


// |k x p| / (|k| |p|): at most kRankTol the wrist candidate is flagged for the
// slide, at most kLockTol the core splits the lock (ssik.continuum.lock_sine).
inline double lock_sine(const Eigen::Vector3d& k, const Eigen::Vector3d& p) {
  return std::sqrt(k.cross(p).squaredNorm() / (k.squaredNorm() * p.squaredNorm()));
}

// The SP1 angle about k toward q of Rot(a4, t) p0 as t -> 0 from side `sign`,
// p0 on k (ssik.continuum._one_sided).
inline double one_sided(const Eigen::Vector3d& k, const Eigen::Vector3d& a4,
                        const Eigen::Vector3d& p0, const Eigen::Vector3d& q, double sign) {
  return sp1_angle(k, sign * a4.cross(p0), q);
}

using WristPair = std::pair<double, double>;

// Wrist angles of a degenerate wrist SP1 pair: one pair (on the lock, the free
// joint at q_free) or two (near it, the branches either side).
struct WristSplit {
  std::array<WristPair, 2> pairs;
  int count;
  double q1, q5;  // three_parallel: the branch's (q1, q5), made exact on a lock
  bool reached = false;  // three_parallel: some free value tried reaches the elbow
};

// ssik.continuum.LOCK_SEARCH.
inline constexpr int kLockSearch = 64;

// Whether SP3 |Rot(k, t) p - q| = d is strictly feasible (ssik.continuum._sp3_feasible).
inline bool sp3_feasible(const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                         const Eigen::Vector3d& q, double d) {
  const double target = 0.5 * (p.dot(p) + q.dot(q) - d * d);
  const double axial = k.dot(q) * k.dot(p);
  const double amp = k.cross(p).norm() * k.cross(q).norm();
  return std::abs(target - axial) <= amp * (1.0 - kLockTol);
}

// How far d lies outside the distances |Rot(k, t) p - q| reaches, 0 within them
// (ssik.continuum._sp3_miss).
inline double sp3_miss(const Eigen::Vector3d& k, const Eigen::Vector3d& p,
                       const Eigen::Vector3d& q, double d) {
  const double s = p.dot(p) + q.dot(q) - 2.0 * k.dot(q) * k.dot(p);
  const double amp = 2.0 * k.cross(p).norm() * k.cross(q).norm();
  const double lo = std::sqrt(std::max(s - amp, 0.0));
  const double hi = std::sqrt(std::max(s + amp, 0.0));
  return std::max({lo - d, d - hi, 0.0});
}

// The representatives of a three_parallel branch within kRankTol of a lock
// (ssik.continuum.three_parallel_lock): (q1, q5) made exact on the lock, and
// (theta14, q6) with q6 at q_free and q_free + pi, each moved to the nearest of
// kLockSearch values around the circle at which the elbow reaches. `p` is the
// chain's p[0..6].
inline WristSplit three_parallel_lock(const std::array<Eigen::Vector3d, 6>& axes,
                                      const Eigen::Matrix3d& r_06, const Eigen::Vector3d& p_0t,
                                      const std::array<Eigen::Vector3d, 7>& p, double q1,
                                      double q5, double q_free) {
  const Eigen::Vector3d p_16 = p_0t - p[0] - r_06 * p[6];
  const Eigen::Vector3d a5_mid = rotation_matrix(axes[4], q5) * axes[5];
  q5 += sp1_angle(axes[4], a5_mid, std::copysign(1.0, a5_mid.dot(axes[1])) * axes[1]);
  const Eigen::Matrix3d r_45 = rotation_matrix(axes[4], q5);
  const double d1 = axes[1].dot(p[1] + p[2] + p[3] + p[4]) + axes[1].dot(r_45 * p[5]);
  const auto roots = sp4(axes[1], -axes[0], p_16, d1).first;
  if (!roots.empty()) {
    double best = roots[0];
    for (double t : roots)
      if (std::abs(finalize_detail::wrap_to_pi(t - q1)) <
          std::abs(finalize_detail::wrap_to_pi(best - q1)))
        best = t;
    q1 = best;
  }
  const Eigen::Matrix3d r_01 = rotation_matrix(axes[0], q1);
  WristSplit out{};
  out.q1 = q1;
  out.q5 = q5;
  out.count = 2;
  for (int side = 0; side < 2; ++side) {
    const double start = q_free + (side == 0 ? 0.0 : M_PI);
    bool have = false;
    for (int j = 0; j <= kLockSearch; ++j) {
      // k = 0, 1, -1, 2, -2, ..., kLockSearch / 2.
      const int k = j == 0 ? 0 : ((j + 1) / 2) * (j % 2 == 1 ? 1 : -1);
      if (std::abs(k) > kLockSearch / 2) break;
      const double v = start + 2.0 * M_PI * k / kLockSearch;
      const Eigen::Matrix3d m = r_01.transpose() * r_06 *
                                rotation_matrix(axes[5], v).transpose() * r_45.transpose();
      const double th = sp1_angle(axes[1], axes[4], m * axes[4]);
      if (!have) {
        out.pairs[side] = {th, v};
        have = true;
      }
      const Eigen::Matrix3d r_14 = rotation_matrix(axes[1], th);
      const Eigen::Vector3d d_inner =
          r_01.transpose() * p_16 - p[1] - r_14 * r_45 * p[5] - r_14 * p[4];
      if (sp3_feasible(axes[1], -p[3], p[2], d_inner.norm())) {
        out.pairs[side] = {th, v};
        out.reached = true;
        break;
      }
    }
  }
  return out;
}

// (q4, q6) of a degenerate spherical wrist (ssik.continuum.spherical_wrist_lock).
inline WristSplit spherical_wrist_lock(const std::array<Eigen::Vector3d, 6>& axes,
                                       const Eigen::Matrix3d& r_36, double q5, double q_free) {
  const Eigen::Vector3d q_q4 = r_36 * axes[5];
  WristSplit out{};
  if (lock_sine(axes[3], q_q4) <= kAxisTol) {
    const Eigen::Matrix3d m = r_36 * rotation_matrix(axes[5], q_free).transpose() *
                              rotation_matrix(axes[4], q5).transpose();
    out.pairs[0] = {sp1_angle(axes[3], axes[4], m * axes[4]), q_free};
    out.count = 1;
    return out;
  }
  const Eigen::Vector3d p_q4 = rotation_matrix(axes[4], q5) * axes[5];
  const Eigen::Vector3d p_q6 = rotation_matrix(axes[4], -q5) * axes[3];
  for (int n = 0; n < 2; ++n) {
    const double sign = n == 0 ? 1.0 : -1.0;
    out.pairs[n] = {one_sided(axes[3], axes[4], p_q4, q_q4, sign),
                    one_sided(-axes[5], -axes[4], p_q6, r_36.transpose() * axes[3], sign)};
  }
  out.count = 2;
  return out;
}

// How far three_parallel's wrist angles miss the target's wrist rotation
// (ssik.continuum.three_parallel_wrist_error).
inline double three_parallel_wrist_error(const std::array<Eigen::Vector3d, 6>& axes,
                                         const Eigen::Matrix3d& r_06, double q1, double q5,
                                         double theta14, double q6) {
  const Eigen::Matrix3d want = rotation_matrix(axes[0], q1).transpose() * r_06;
  const Eigen::Matrix3d got = rotation_matrix(axes[1], theta14) * rotation_matrix(axes[4], q5) *
                              rotation_matrix(axes[5], q6);
  return (want - got).norm();
}

// Whether a flagged three_parallel branch is split because its SP1 wrist
// angles cannot give a candidate within the FK `gate`: they miss the wrist
// rotation by more than it, or their theta14 leaves the elbow farther than it
// out of reach while a free value the split tries reaches it
// (ssik.continuum.three_parallel_wrist_misses). `p` is the chain's p[0..6].
inline bool three_parallel_wrist_misses(const std::array<Eigen::Vector3d, 6>& axes,
                                        const Eigen::Matrix3d& r_06, const Eigen::Vector3d& p_0t,
                                        const std::array<Eigen::Vector3d, 7>& p, double q1,
                                        double q5, double theta14, double q6, double q_free,
                                        double gate) {
  if (three_parallel_wrist_error(axes, r_06, q1, q5, theta14, q6) > gate) return true;
  const Eigen::Vector3d p_16 = p_0t - p[0] - r_06 * p[6];
  const Eigen::Matrix3d r_14 = rotation_matrix(axes[1], theta14);
  const Eigen::Vector3d d_inner = rotation_matrix(axes[0], q1).transpose() * p_16 - p[1] -
                                  r_14 * rotation_matrix(axes[4], q5) * p[5] - r_14 * p[4];
  if (sp3_miss(axes[1], -p[3], p[2], d_inner.norm()) <= gate) return false;
  return three_parallel_lock(axes, r_06, p_0t, p, q1, q5, q_free).reached;
}

// How far a spherical wrist's angles miss R_36 (ssik.continuum.spherical_wrist_error).
inline double spherical_wrist_error(const std::array<Eigen::Vector3d, 6>& axes,
                                    const Eigen::Matrix3d& r_36, double q4, double q5, double q6) {
  const Eigen::Matrix3d got =
      rotation_matrix(axes[3], q4) * rotation_matrix(axes[4], q5) * rotation_matrix(axes[5], q6);
  return (r_36 - got).norm();
}

// Whether a 6x6 Jacobian may be rank deficient: the Frobenius condition number
// of A = J^T J + 1e-9 I is at least 0.5 / kRankTol^2 (ssik.continuum.
// rank_deficient; no false negatives). Singular in floating point: flagged.
inline bool rank_deficient(const Eigen::Matrix<double, 6, 6>& jac) {
  constexpr double kRankDamping = 1e-9;  // ssik.continuum.RANK_DAMPING
  const Eigen::Matrix<double, 6, 6> normal =
      jac.transpose() * jac + kRankDamping * Eigen::Matrix<double, 6, 6>::Identity();
  const Eigen::Matrix<double, 6, 6> inv = normal.partialPivLu().inverse();
  const double kappa = std::sqrt(normal.squaredNorm() * inv.squaredNorm());
  return !(kappa * kRankTol * kRankTol < 0.5);
}

// Whether one Jacobian has a null direction, sigma_min <= kRankTol sigma_max
// (ssik.continuum.has_null).
inline bool has_null(const Eigen::Matrix<double, 6, 6>& jac) {
  const Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(jac);
  const auto s = svd.singularValues();
  return s[5] <= kRankTol * s[0];
}

// Same-root dedup (#600, dedup_same_root) that keeps each survivor's continuum
// flag, a merge keeping the flag either side had: the artifact orchestrator's
// merge in ssik.core.codegen._with_continuum.
template <int N>
void dedup_same_root_flagged(std::vector<Solution<N>>& sols, std::vector<int>& free,
                             const JointConsts<N>& c, const Pose& T, double gate) {
  constexpr double kPi = 3.14159265358979323846;
  const double floor = same_root_floor(T);
  std::vector<Solution<N>> out;
  std::vector<int> out_free;
  for (std::size_t n = 0; n < sols.size(); ++n) {
    const auto& cand = sols[n];
    int match = -1;
    for (std::size_t j = 0; j < out.size() && match < 0; ++j) {
      bool close = true;
      for (int i = 0; i < N; ++i) {
        double d = std::fmod(cand.q[i] - out[j].q[i] + kPi, 2.0 * kPi);
        if (d < 0) d += 2.0 * kPi;
        if (std::abs(d - kPi) > gate) {
          close = false;
          break;
        }
      }
      if (close && is_same_root<N>(c, T, cand, out[j], floor)) match = static_cast<int>(j);
    }
    if (match < 0) {
      out.push_back(cand);
      out_free.push_back(free[n]);
      continue;
    }
    const int f = out_free[match] != kNotFlagged ? out_free[match] : free[n];
    if (cand.fk_residual < out[match].fk_residual - floor) out[match] = cand;
    out_free[match] = f;
  }
  sols = std::move(out);
  free = std::move(out_free);
}

// The slide (ssik.continuum.slide_continua): every flagged solution moves to its
// continuum's point under the rule, then the samples that met are merged by the
// same-root rule. `free[i]` is kNotFlagged, kFreeFromKernel or the free joint;
// an empty `free` flags nothing.
// `seed` null for the unseeded rule; only `respect_limits` brings a point into
// limits. Nothing flagged returns `sols` untouched.
inline std::vector<Solution<6>> slide_continua(std::vector<Solution<6>> sols,
                                               const std::vector<int>& free,
                                               const JointConsts<6>& c, const JointLimits<6>& lim,
                                               const Pose& T, const std::array<double, 6>* seed,
                                               bool respect_limits, double dedup_atol) {
  using namespace continuum_detail;
  bool any = false;  // an empty `free` is none flagged
  for (int f : free) any = any || f != kNotFlagged;
  if (!any) return sols;
  const Chain ch(c, lim, T);
  Vec6 seed_v;
  if (seed != nullptr) seed_v = to_vec(*seed);
  for (std::size_t i = 0; i < sols.size(); ++i) {
    if (free[i] == kNotFlagged) continue;
    sols[i] = slide_one(ch, sols[i], free[i], seed != nullptr ? &seed_v : nullptr, respect_limits);
  }
  return dedup_same_root<6>(sols, c, T, dedup_atol);
}

}  // namespace ssik
