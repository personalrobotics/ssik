// Family-agnostic T-perturbation rescue (#319), ported from
// ssik.refinement.rescue.rescue_via_T_perturbation. Recovers IK at reachable
// but rank-deficient poses (kinematic singularities) where the closed-form
// analytical extraction returns nothing: perturb T_target by small random SE(3)
// increments (off the rank-deficient ridge), re-solve the analytical core at
// each perturbed pose, then Newton-polish every candidate back to the original
// T_target. Returns the unique FK-closing solutions.
//
// Like the Python original this is SOLVER-AGNOSTIC: the analytical core is a
// callable, so one implementation serves every family (three_parallel /
// spherical / SRS now; RR / HP when they gain native cores). Each
// <family>_artifact_solve wires it in by passing its own core.
//
// RNG note: this uses a portable std PRNG, NOT numpy's PCG64, so at a singular
// pose (where the solution set is an ill-posed continuum sampled via the
// perturbations) the recovered SET differs from Python's -- both are sound
// (every solution FK-closes), which is the only meaningful contract there
// (#56). On the non-singular poses that dominate, rescue never fires, so the
// artifact stays bit-for-bit with Python.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/newton.hpp"  // lm_refine
#include "ssik_cpp/rotation.hpp"

namespace ssik {

// Reach-sphere upper bound (sum of all link translation norms) -- the rescue
// gate. A triangle-inequality bound, so it never rejects a reachable pose; it
// only keeps far-field unreachable targets from paying for a rescue attempt.
template <int N>
double reach_radius(const JointConsts<N>& c) {
  double r = 0.0;
  for (int i = 0; i < N; ++i) {
    r += c.t_left[i].template block<3, 1>(0, 3).norm();
    r += c.t_right[i].template block<3, 1>(0, 3).norm();
  }
  return r;
}

struct RescueParams {
  int n_perturbations = 16;
  double perturbation_scale_m = 5e-3;
  double perturbation_scale_rad = 5e-3;
  std::array<double, 4> scale_multipliers = {1.0, 2.0, 4.0, 10.0};
  double fk_atol = 1e-8;
  int refinement_max_iters = 20;
  double dedup_atol = 1e-4;
  std::uint64_t seed = 20260608;
};

namespace rescue_detail {

// ||wrap_to_pi(a - b)||_2, the dedup metric.
template <int N>
double wrap_dist(const std::array<double, N>& a, const std::array<double, N>& b) {
  double s = 0.0;
  for (int i = 0; i < N; ++i) {
    double d = std::fmod(a[i] - b[i] + M_PI, 2.0 * M_PI);
    if (d < 0.0) d += 2.0 * M_PI;
    d -= M_PI;
    s += d * d;
  }
  return std::sqrt(s);
}

}  // namespace rescue_detail

// solve_fn: (const Pose&) -> vector<Solution<N>>, the arm's analytical core
// evaluated at a perturbed pose (identical role to Python's solve_fn). Returns
// the FK-closing, deduped solutions rescued back to T_target; empty if none.
template <int N, typename SolveFn>
std::vector<Solution<N>> rescue_via_T_perturbation(SolveFn&& solve_fn, const JointConsts<N>& c,
                                                   const Pose& T_target,
                                                   const RescueParams& p = {}) {
  std::mt19937_64 rng(p.seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  const double tight = std::min(1e-12, p.fk_atol);

  std::vector<Solution<N>> refined;
  std::vector<std::array<double, N>> refined_qs;

  for (int i = 0; i < p.n_perturbations; ++i) {
    const double mult = p.scale_multipliers[i % p.scale_multipliers.size()];
    // Draw dx (translation) then w (so3), matching the Python draw order.
    Eigen::Vector3d dx, w;
    for (int k = 0; k < 3; ++k) dx[k] = normal(rng) * p.perturbation_scale_m * mult;
    for (int k = 0; k < 3; ++k) w[k] = normal(rng) * p.perturbation_scale_rad * mult;
    const double angle = w.norm();
    Eigen::Matrix3d R_delta = Eigen::Matrix3d::Identity();
    if (angle > 0.0) R_delta = rotation_matrix(w / angle, angle);
    Pose dT = Pose::Identity();
    dT.block<3, 3>(0, 0) = R_delta;
    dT.block<3, 1>(0, 3) = dx;
    const Pose T_pert = T_target * dT;

    const std::vector<Solution<N>> pert = solve_fn(T_pert);
    for (const auto& sol : pert) {
      // Polish back to the ORIGINAL T_target: tight (machine precision) then a
      // loose retry for candidates that only reach fk_atol on a genuine ridge.
      auto r = lm_refine<N>(c, sol.q, T_target, tight, p.refinement_max_iters);
      if (!r) r = lm_refine<N>(c, sol.q, T_target, p.fk_atol, p.refinement_max_iters);
      if (!r || r->second > p.fk_atol) continue;

      bool dup = false;
      for (const auto& e : refined_qs)
        if (rescue_detail::wrap_dist<N>(r->first, e) < p.dedup_atol) {
          dup = true;
          break;
        }
      if (dup) continue;
      refined.push_back(Solution<N>{r->first, r->second, Refinement::Rescue});
      refined_qs.push_back(r->first);
    }
  }
  return refined;
}

}  // namespace ssik
