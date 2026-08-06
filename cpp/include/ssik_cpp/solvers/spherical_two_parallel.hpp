// Spherical-wrist two-parallel 6R analytical IK (#510), ported verbatim from
// ssik.solvers.ikgeo.spherical_two_parallel.solve. Pieper 6R: a spherical wrist
// (intersecting axes at joints 3,4,5) plus two parallel shoulder axes (1,2).
// SP4 isolates q1, SP3/SP1 solve the shoulder, SP4/SP1/SP1 solve the wrist.
//
// The caller must pass CANONICAL constants (canonicalize_spherical_wrist applied
// in the Python dispatch, cached per-arm) -- the p[3] wrist-collapse assumes the
// canonical form. Canonicalization is FK-identical, so verification here is valid
// with the canonical constants and the returned q are physical joint values.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/newton.hpp"
#include "ssik_cpp/rescue.hpp"
#include "ssik_cpp/rotation.hpp"
#include "ssik_cpp/sp6.hpp"  // detail::wrap_pi
#include "ssik_cpp/subproblems.hpp"

namespace ssik {

inline constexpr double kSphericalTwoParallelFkAtol = 1e-7;

// Core analytical solve. Returns FK-verified, deduped joint solutions. With
// allow_refinement, near-misses are Newton-polished (the artifact's force_refine
// path). `tol` mirrors the Python policy.
inline std::vector<Solution<6>> spherical_two_parallel_solve(const JointConsts<6>& c, const Pose& T,
                                                             const Tolerances& tol = {},
                                                             bool allow_refinement = false,
                                                             int refinement_max_iters = 15) {
  std::array<Eigen::Vector3d, 6> axes;
  for (int i = 0; i < 6; ++i) axes[i] = c.axis[i];

  // p[]: consolidated shoulder-to-wrist offset at p[3] (sum of T_left[3..5]);
  // p[4],p[5] unused (wrist-collapse). p[6] = tool translation.
  std::array<Eigen::Vector3d, 7> p;
  for (int i = 0; i < 6; ++i) p[i] = c.t_left[i].block<3, 1>(0, 3);
  p[3] = c.t_left[3].block<3, 1>(0, 3) + c.t_left[4].block<3, 1>(0, 3) +
         c.t_left[5].block<3, 1>(0, 3);
  p[4].setZero();
  p[5].setZero();
  p[6] = c.t_right[5].block<3, 1>(0, 3);

  const Eigen::Matrix3d r_home = c.t_right[5].block<3, 3>(0, 0);
  const Eigen::Matrix3d r_06 = T.block<3, 3>(0, 0) * r_home.transpose();
  const Eigen::Vector3d p_0t = T.block<3, 1>(0, 3);

  // SP4 isolates q1: axes[1] . Rot(-axes[0], q1) (p_0t - r_06 p[6] - p[0]) =
  // axes[1] . (p[1]+p[2]+p[3]).
  const auto [t1_solutions, ls1] =
      sp4(axes[1], -axes[0], p_0t - r_06 * p[6] - p[0], axes[1].dot(p[1] + p[2] + p[3]), tol);
  (void)ls1;

  std::vector<std::array<double, 6>> candidates;
  for (double q1 : t1_solutions) {
    const Eigen::Vector3d shoulder =
        rotation_matrix(-axes[0], q1) * (-p_0t + r_06 * p[6] + p[0]) + p[1];

    const auto [t3_solutions, ls3] = sp3(axes[2], -p[3], p[2], shoulder.norm(), tol);
    (void)ls3;

    for (double q3 : t3_solutions) {
      const auto [q2, ls2] =
          sp1(axes[1], -p[2] - rotation_matrix(axes[2], q3) * p[3], shoulder, tol);
      (void)ls2;

      const Eigen::Matrix3d r_36 = rotation_matrix(-axes[2], q3) * rotation_matrix(-axes[1], q2) *
                                   rotation_matrix(-axes[0], q1) * r_06;

      const auto [t5_solutions, ls5] =
          sp4(axes[3], axes[4], axes[5], axes[3].dot(r_36 * axes[5]), tol);
      (void)ls5;

      for (double q5 : t5_solutions) {
        const auto [q4, ls4] =
            sp1(axes[3], rotation_matrix(axes[4], q5) * axes[5], r_36 * axes[5], tol);
        const auto [q6, ls6] =
            sp1(-axes[5], rotation_matrix(-axes[4], q5) * axes[3], r_36.transpose() * axes[3], tol);
        (void)ls4;
        (void)ls6;
        candidates.push_back({q1, q2, q3, q4, q5, q6});
      }
    }
  }

  // verify_candidates tail: FK-closure gate (+ optional Newton polish), then
  // dedup_by_wrap_close (lowest fk_residual per cluster, first-seen order).
  std::vector<Solution<6>> verified;
  for (const auto& q : candidates) {
    const Pose fk_q = fk<6>(c, q);
    const double resid = (fk_q - T).norm();
    if (resid <= kSphericalTwoParallelFkAtol) {
      verified.push_back(Solution<6>{q, resid, Refinement::None});
    } else if (allow_refinement) {
      const auto refined = lm_refine<6>(c, q, T, kSphericalTwoParallelFkAtol, refinement_max_iters);
      if (refined) verified.push_back(Solution<6>{refined->first, refined->second, Refinement::Lm});
    }
  }

  std::vector<Solution<6>> deduped;
  for (const auto& cand : verified) {
    int match = -1;
    for (int j = 0; j < static_cast<int>(deduped.size()); ++j) {
      bool close = true;
      for (int i = 0; i < 6; ++i) {
        if (std::abs(detail::wrap_pi(cand.q[i] - deduped[j].q[i])) > tol.dedup) {
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

// Full artifact-contract solve (#513): core solve (force-refined, as the artifact
// always polishes) -> rescue gate (#319) -> finalize (limits -> seed ->
// truncate). `c` must be the CANONICAL constants (canonicalize_spherical_wrist
// applied at emit time and baked). The rescue recovers solutions at reachable
// rank-deficient (singular) poses where the analytical returns empty, so the
// standalone artifact matches Python's solve() there; dormant otherwise.
inline std::vector<Solution<6>> spherical_two_parallel_artifact_solve(
    const JointConsts<6>& c, const JointLimits<6>& lim, const Pose& T, const ArtifactParams<6>& p,
    const Tolerances& tol = {}) {
  const auto core = [&](const Pose& Tp) {
    return spherical_two_parallel_solve(c, Tp, tol, /*allow_refinement=*/true,
                                        p.refinement_max_iters);
  };
  std::vector<Solution<6>> sols = core(T);

  if (sols.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    sols = rescue_via_T_perturbation<6>(core, c, T);
  }

  return finalize_solutions<6>(std::move(sols), c, lim, p);
}

}  // namespace ssik
