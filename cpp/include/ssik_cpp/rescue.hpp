// Family-agnostic T-perturbation rescue (#319): the native port of
// ssik.refinement.rescue.rescue_via_T_perturbation. Recovers IK at reachable
// but rank-deficient poses (kinematic singularities) where the closed-form
// analytical extraction returns nothing: perturb T_target by small SE(3)
// increments (off the rank-deficient ridge), re-solve the analytical core at
// each perturbed pose, then Newton-polish every candidate back to the original
// T_target. Returns the unique FK-closing solutions.
//
// Like the Python original this is SOLVER-AGNOSTIC: the analytical core is a
// callable, so one implementation serves every family. Each
// <family>_artifact_solve wires it in by passing its own core.
//
// ONE DEFINITION (#622): this implements exactly the algorithm that the module
// docstring of src/ssik/refinement/rescue.py ("The definition") owns -- the
// same deterministic perturbation sequence (an R_6 Kronecker sequence built from
// correctly rounded IEEE operations, so bit-identical in both languages and on
// every platform; no PRNG) and the same polish (lm_refine_batch's fixed-damping
// Newton with its 5.0 / 4 divergence guard, aiming for 1e-12 and accepting the
// end point at fk_atol, then the residual-scaled-damping retry of a candidate it
// does not accept). Native and Python therefore recover the same set at a
// singular pose, up to float round-off in the polish. A change on one side must
// be mirrored on the other.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/newton.hpp"  // se3_log_residual, spatial_jacobian
#include "ssik_cpp/parallel.hpp"
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

// The rescue's parameters: rescue_via_T_perturbation's keyword defaults, which
// every Python call site uses. A call site that changes one leaves the shared
// definition (#622).
struct RescueParams {
  int n_perturbations = 16;
  double perturbation_scale_m = 5e-3;
  double perturbation_scale_rad = 5e-3;
  std::array<double, 4> scale_multipliers = {1.0, 2.0, 4.0, 10.0};
  double fk_atol = 1e-8;
  int refinement_max_iters = 20;
  // Match the solver's dedup tolerance (subproblem_dedup, 1e-3): a tighter rescue
  // dedup leaked near-duplicate pairs in (1e-4, 1e-3) into the returned set, since
  // finalize does not re-dedup (#534). Two solutions the solver considers the same
  // (< 1e-3 wrap-to-pi) must not both survive the rescue.
  double dedup_atol = 1e-3;
};

namespace rescue_detail {

// frac(phi^-(k+1)), k = 0..5, for the real root phi of x^7 = x + 1: the R_6
// Kronecker sequence's increments. Mirrors rescue.py's _KRONECKER_ALPHA.
inline constexpr std::array<double, 6> kKroneckerAlpha = {
    0.8986537126286993, 0.8075784952213448, 0.7257334129697598,
    0.6521830259439717, 0.5860866975779695, 0.5266889867007359};
inline constexpr double kSqrt3 = 1.7320508075688772;

// Perturbation i's unscaled SE(3) direction u_i, translation first:
// sqrt(3) * (2 * frac((i + 1) * alpha_k) - 1). rescue.py's perturbation_direction.
inline std::array<double, 6> perturbation_direction(int i) {
  std::array<double, 6> u{};
  for (int k = 0; k < 6; ++k) {
    const double x = static_cast<double>(i + 1) * kKroneckerAlpha[k];
    u[k] = kSqrt3 * (2.0 * (x - std::floor(x)) - 1.0);
  }
  return u;
}

// The polish: ssik.refinement.lm_refine_batch for one candidate, with the
// arguments rescue.py passes (target `tight`, 5.0 / 4 divergence guard). Step
// dq = clip((J^T J + lambda I)^-1 J^T r, +-0.5) with r = log(T_target FK(q)^-1)
// and lambda = 1e-9, or 1e-9 * clip(|r|, 1e-5, 1) when `scaled_damping`
// (lm_refine_batch's residual_scaled_damping). The normal equations are solved
// by partial-pivot LU as numpy.linalg.solve does: at a singular pose J^T J is
// nearly singular and the step's null-space part, which decides the candidate's
// fate, depends on the factorization (with LDLT, twice as many parity-gate
// poses had a rescue empty on one backend only). Returns (q, residual) when the
// Frobenius residual drops below `tight`, or when all max_iters steps ran and
// the end point is within `accept`; nullopt when the divergence guard fired or
// the end point misses `accept`. No stall guard: lm_refine_batch has none.
template <int N>
std::optional<std::pair<std::array<double, N>, double>> polish(const JointConsts<N>& c,
                                                               const std::array<double, N>& seed,
                                                               const Pose& t_target, double tight,
                                                               double accept, int max_iters,
                                                               bool scaled_damping) {
  constexpr double kStepClip = 0.5;
  constexpr double kDamping = 1e-9;
  constexpr double kDampingResidualFloor = 1e-5;
  constexpr double kDivergenceFactor = 5.0;
  constexpr int kDivergenceMinIters = 4;
  std::array<double, N> q = seed;
  double r_best = std::numeric_limits<double>::infinity();
  for (int it = 0; it < max_iters; ++it) {
    const Pose t_q = fk<N>(c, q);
    const double fro = (t_q - t_target).norm();
    if (fro < tight) return std::make_pair(q, fro);
    if (fro < r_best)
      r_best = fro;
    else if (it >= kDivergenceMinIters && fro > kDivergenceFactor * r_best)
      return std::nullopt;
    const Eigen::Matrix<double, 6, 1> res = se3_log_residual(t_target * t_q.inverse());
    const Eigen::Matrix<double, 6, N> js = spatial_jacobian<N>(c, q);
    const double lambda =
        scaled_damping ? kDamping * std::clamp(res.norm(), kDampingResidualFloor, 1.0) : kDamping;
    const Eigen::Matrix<double, N, N> jtj =
        js.transpose() * js + lambda * Eigen::Matrix<double, N, N>::Identity();
    const Eigen::Matrix<double, N, 1> dq = jtj.partialPivLu().solve(js.transpose() * res);
    for (int i = 0; i < N; ++i) q[i] += std::max(-kStepClip, std::min(kStepClip, dq[i]));
  }
  const double final_fro = (fk<N>(c, q) - t_target).norm();
  if (final_fro > accept) return std::nullopt;
  return std::make_pair(q, final_fro);
}

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
  const double tight = std::min(1e-12, p.fk_atol);
  const int n = p.n_perturbations;

  // Phase 1 (serial, cheap): the perturbed poses, in the operation order of
  // rescue.py's _perturbation, so the two agree to the last bit up to libm's
  // sin/cos.
  std::vector<Pose> T_perts(n);
  for (int i = 0; i < n; ++i) {
    const double mult = p.scale_multipliers[i % p.scale_multipliers.size()];
    const std::array<double, 6> u = rescue_detail::perturbation_direction(i);
    const double w0 = u[3] * p.perturbation_scale_rad * mult;
    const double w1 = u[4] * p.perturbation_scale_rad * mult;
    const double w2 = u[5] * p.perturbation_scale_rad * mult;
    const double angle = std::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
    Pose dT = Pose::Identity();
    if (angle > 0.0)
      dT.block<3, 3>(0, 0) =
          rotation_matrix(Eigen::Vector3d(w0 / angle, w1 / angle, w2 / angle), angle);
    dT(0, 3) = u[0] * p.perturbation_scale_m * mult;
    dT(1, 3) = u[1] * p.perturbation_scale_m * mult;
    dT(2, 3) = u[2] * p.perturbation_scale_m * mult;
    T_perts[i] = T_target * dT;
  }

  // Phase 2 (parallel): the expensive part -- re-solve each perturbed pose and
  // polish every candidate back to T_target. Perturbations are independent
  // (solve_fn is pure; each writes its own slot), so fan out. When solve_fn is
  // itself a parallel sweep (jointlock), parallel_for's nesting guard runs that
  // inner sweep serially -- only this outer level threads.
  std::vector<std::vector<Solution<N>>> per(n);
  parallel_for(static_cast<std::size_t>(n), [&](std::size_t idx) {
    for (const auto& sol : solve_fn(T_perts[idx])) {
      // Polish back to the ORIGINAL T_target: aim for machine precision and
      // accept the end point at fk_atol (a genuine ridge stalls in between). A
      // candidate the fixed-damping polish does not accept is polished again
      // from its seed with residual-scaled damping, which does not stall near
      // a singular solution (#646; rescue.py's module docstring).
      auto r = rescue_detail::polish<N>(c, sol.q, T_target, tight, p.fk_atol,
                                        p.refinement_max_iters, /*scaled_damping=*/false);
      if (!r)
        r = rescue_detail::polish<N>(c, sol.q, T_target, tight, p.fk_atol, p.refinement_max_iters,
                                     /*scaled_damping=*/true);
      if (!r) continue;
      per[idx].push_back(Solution<N>{r->first, r->second, Refinement::Rescue});
    }
  });

  // Phase 3 (serial): merge in perturbation order + dedup. Same order as the
  // single-threaded loop, so the surviving set + representatives are identical.
  std::vector<Solution<N>> refined;
  std::vector<std::array<double, N>> refined_qs;
  for (int i = 0; i < n; ++i) {
    for (const auto& sol : per[i]) {
      bool dup = false;
      for (const auto& e : refined_qs)
        if (rescue_detail::wrap_dist<N>(sol.q, e) < p.dedup_atol) {
          dup = true;
          break;
        }
      if (dup) continue;
      refined.push_back(sol);
      refined_qs.push_back(sol.q);
    }
  }
  return refined;
}

}  // namespace ssik
