// Generic three-parallel 6R analytical IK (#486), ported verbatim from
// ssik.solvers.ikgeo.three_parallel.solve. Covers the UR family (three
// consecutive parallel axes at joints 1,2,3). SP6 solves (q1,q5) jointly;
// per branch SP1/SP3/SP1 recover the remaining joints; candidates are
// FK-verified (Frobenius closure) and deduped by wrap-to-pi closeness.
//
// Phase-0a scope: analytical parity (allow_refinement=False, FK gate 1e-7).
// The shipped standalone artifact forces Newton polish (SolverSpec.force_refine);
// that shared Eigen Newton lands in Phase 1.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/newton.hpp"
#include "ssik_cpp/rescue.hpp"
#include "ssik_cpp/rotation.hpp"
#include "ssik_cpp/sp6.hpp"
#include "ssik_cpp/subproblems.hpp"

namespace ssik {

// FK-closure gate for accepting a composed candidate (three_parallel._FK_VERIFY_ATOL).
inline constexpr double kThreeParallelFkAtol = 1e-7;

namespace detail {

inline double wrap_to_pi(double a) { return wrap_pi(a); }

// Sign of trio joints 2,3 relative to reference axes[1] (trio_reference_signs).
inline std::pair<int, int> trio_reference_signs(const std::array<Eigen::Vector3d, 6>& axes) {
  return {axes[2].dot(axes[1]) >= 0.0 ? 1 : -1, axes[3].dot(axes[1]) >= 0.0 ? 1 : -1};
}

}  // namespace detail

// Solve three-parallel 6R IK for target pose T. Returns FK-verified, deduped
// joint solutions. `tol` mirrors the Python policy. With `allow_refinement`,
// near-misses (FK > gate) are Newton-polished (the shipped artifact's
// force_refine path); off by default, matching ikgeo.three_parallel.solve.
inline std::vector<Solution<6>> three_parallel_solve(const JointConsts<6>& c, const Pose& T,
                                                     const Tolerances& tol = {},
                                                     bool allow_refinement = false,
                                                     int refinement_max_iters = 15) {
  std::array<Eigen::Vector3d, 6> axes;
  for (int i = 0; i < 6; ++i) axes[i] = c.axis[i];

  std::array<Eigen::Vector3d, 7> p;
  for (int i = 0; i < 6; ++i) p[i] = c.t_left[i].block<3, 1>(0, 3);
  p[6] = c.t_right[5].block<3, 1>(0, 3);

  const auto [s2, s3] = detail::trio_reference_signs(axes);
  const std::array<double, 6> trio_flip = {1.0, 1.0, double(s2), double(s3), 1.0, 1.0};

  const Eigen::Matrix3d r_home = c.t_right[5].block<3, 3>(0, 0);
  const Eigen::Matrix3d r_06 = T.block<3, 3>(0, 0) * r_home.transpose();
  const Eigen::Vector3d p_0t = T.block<3, 1>(0, 3);
  const Eigen::Vector3d p_16 = p_0t - p[0] - r_06 * p[6];

  // SP6 for (theta1, theta5).
  Vec3x4 h_sp = {axes[1], axes[1], axes[1], axes[1]};
  Vec3x4 k_sp = {-axes[0], axes[4], -axes[0], axes[4]};
  Vec3x4 p_sp = {p_16, -p[5], r_06 * axes[5], -axes[5]};
  const double d1 = axes[1].dot(p[2] + p[3] + p[4] + p[1]);
  const double d2 = 0.0;

  const auto [theta15_solutions, sp6_ls] = sp6(h_sp, k_sp, p_sp, d1, d2, tol);
  (void)sp6_ls;

  std::vector<std::array<double, 6>> candidates;
  for (const auto& [q1, q5] : theta15_solutions) {
    const Eigen::Matrix3d r_01 = rotation_matrix(axes[0], q1);
    const Eigen::Matrix3d r_45 = rotation_matrix(axes[4], q5);

    const auto [theta14, ls_a] =
        sp1(axes[1], r_45 * axes[5], r_01.transpose() * r_06 * axes[5], tol);
    const auto [q6, ls_b] = sp1(-axes[5], r_45.transpose() * axes[1], r_06.transpose() * r_01 * axes[1], tol);
    (void)ls_a;
    (void)ls_b;

    const Eigen::Matrix3d r_14 = rotation_matrix(axes[1], theta14);
    const Eigen::Vector3d d_inner =
        r_01.transpose() * p_16 - p[1] - r_14 * r_45 * p[5] - r_14 * p[4];
    const double d_elbow = d_inner.norm();

    const auto [theta3_solutions, ls_c] = sp3(axes[1], -p[3], p[2], d_elbow, tol);
    (void)ls_c;

    for (double q3 : theta3_solutions) {
      const Eigen::Vector3d p2_rotated = p[2] + rotate(axes[1], q3, p[3]);
      const auto [q2, ls_d] = sp1(axes[1], p2_rotated, d_inner, tol);
      (void)ls_d;
      const double q4 = detail::wrap_to_pi(theta14 - q2 - q3);
      const std::array<double, 6> raw = {q1, q2, q3, q4, q5, q6};
      std::array<double, 6> q;
      for (int i = 0; i < 6; ++i) q[i] = raw[i] * trio_flip[i];
      candidates.push_back(q);
    }
  }

  // verify_candidates tail: FK-closure gate, then dedup_by_wrap_close (keep
  // lowest fk_residual per wrap-close cluster, first-match-wins order).
  std::vector<Solution<6>> verified;
  for (const auto& q : candidates) {
    const Pose fk_q = fk<6>(c, q);
    const double resid = (fk_q - T).norm();  // Frobenius
    if (resid <= kThreeParallelFkAtol) {
      verified.push_back(Solution<6>{q, resid, Refinement::None});
    } else if (allow_refinement && resid < 0.1) {
      // Refine only near-misses; skip candidates clearly not near a solution
      // (matches the codegen refine pre-filter, #490).
      const auto refined = lm_refine<6>(c, q, T, kThreeParallelFkAtol, refinement_max_iters);
      if (refined) {
        verified.push_back(Solution<6>{refined->first, refined->second, Refinement::Lm});
      }
    }
  }

  std::vector<Solution<6>> deduped;
  for (const auto& cand : verified) {
    int match = -1;
    for (int j = 0; j < static_cast<int>(deduped.size()); ++j) {
      bool close = true;
      for (int i = 0; i < 6; ++i) {
        const double diff = detail::wrap_to_pi(cand.q[i] - deduped[j].q[i]);
        if (std::abs(diff) > tol.dedup) {
          close = false;
          break;
        }
      }
      if (close) {
        match = j;
        break;
      }
    }
    if (match == -1) {
      deduped.push_back(cand);
    } else if (cand.fk_residual < deduped[match].fk_residual) {
      deduped[match] = cand;
    }
  }
  return deduped;
}

// Full artifact-contract solve -- the C++ replica of <arm>_ik.solve() (#503):
// core solve (force-refined, the artifact always polishes near-misses) ->
// rescue gate -> finalize (limits -> seed -> truncate). The rescue (#319) makes
// the standalone artifact complete: at a reachable but rank-deficient pose (a
// kinematic singularity where the geometric extraction degenerates) it recovers
// solutions instead of returning empty, matching Python's solve(). Dormant on
// non-singular poses (the vast majority), so the artifact stays bit-for-bit with
// Python there.
inline std::vector<Solution<6>> three_parallel_artifact_solve(const JointConsts<6>& c,
                                                              const JointLimits<6>& lim,
                                                              const Pose& T,
                                                              const ArtifactParams<6>& p,
                                                              const Tolerances& tol = {}) {
  const auto core = [&](const Pose& Tp) {
    return three_parallel_solve(c, Tp, tol, /*allow_refinement=*/true, p.refinement_max_iters);
  };
  // Limit pass only, then rescue on LIMIT-empty (#524): the gate is "no in-limits
  // solution", so it must not depend on the seed-tolerance / max_solutions
  // filters. Seed/truncate is applied afterwards over the in-limits set.
  ArtifactParams<6> p_limits;
  p_limits.respect_limits = p.respect_limits;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  std::vector<Solution<6>> in_limits = finalize_solutions<6>(core(T), c, lim, p_limits);
  if (in_limits.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    in_limits = finalize_solutions<6>(rescue_via_T_perturbation<6>(core, c, T), c, lim, p_limits);
  }
  ArtifactParams<6> p_seed = p;
  p_seed.respect_limits = false;
  return finalize_solutions<6>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
