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
// branch FK-gated (base) or LM-polished (polished), deduped. The exact in-limits
// q6-tracking resolver (resolve_in_limits) is a coarse-sweep refinement that
// rarely fires (only when no sampled q6 is in-limits); it is deferred (bounded,
// documented gap -- see the per-arm gate allowance in cpp_emit).
#pragma once

#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/newton.hpp"  // lm_refine
#include "ssik_cpp/rescue.hpp"
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

// Reachable q6 sub-intervals of [-pi, pi]: margin >= 0 brackets on a 90-grid,
// padded one step, merged (_reachable_intervals + merge).
inline std::vector<std::array<double, 2>> reachable_intervals(
    const Eigen::Matrix<double, 3, 48>& coef, const Pose& t_rev) {
  std::array<double, kShBracketGrid> grid;
  std::array<bool, kShBracketGrid> m;
  for (int i = 0; i < kShBracketGrid; ++i) {
    grid[i] = -M_PI + (2.0 * M_PI) * i / (kShBracketGrid - 1);  // linspace endpoint=True
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

// Full artifact-contract solve for a spherical-shoulder arm (base or polished).
inline std::vector<Solution<7>> spherical_shoulder_artifact_solve(
    const JointConsts<7>& c, const SphericalShoulderConsts& sh, const JointLimits<7>& lim,
    const Pose& T, const ArtifactParams<7>& p, bool polished) {
  const auto core = [&](const Pose& Tp) {
    return spherical_shoulder_core(c, sh, Tp, polished, p.refinement_max_iters);
  };
  ArtifactParams<7> p_limits;
  p_limits.respect_limits = p.respect_limits;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  std::vector<Solution<7>> in_limits = finalize_solutions<7>(core(T), c, lim, p_limits);
  if (in_limits.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    in_limits = finalize_solutions<7>(rescue_via_T_perturbation<7>(core, c, T), c, lim, p_limits);
  }
  ArtifactParams<7> p_seed = p;
  p_seed.respect_limits = false;
  return finalize_solutions<7>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
