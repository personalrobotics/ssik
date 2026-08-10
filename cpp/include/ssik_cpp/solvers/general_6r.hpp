// Raghavan-Roth general 6R analytical IK -- the shared native runtime (#490).
//
// Ports the numeric half of ssik.solvers.ikgeo._raghavan_roth: everything from
// the (already-evaluated) elimination coefficient matrices P/Q down to the joint
// solutions. The per-arm coefficient matrices P_sin/P_cos/P_one (14x9) and Q
// (14x8) -- symbolic functions of the 12 DH-target entries -- are emitted per
// arm (the sympy->C++ CSE piece, RrCoeffs below); this header is the arm-
// agnostic pipeline they feed.
//
// Deliberately NOT a transliteration of the Python (#feedback: re-derive for
// C++). Python builds a 24x24 *companion* matrix, which needs A^{-1} and so
// carries an equilibration + random-Mobius reconditioning search + a scipy
// generalized-eigenvalue last-resort fallback -- a ladder that exists only
// because np.linalg.eig does standard eigenvalues and A^{-1} blows up when
// m_quad is singular. Eigen's RealQZ (GeneralizedEigenSolver) solves the
// Manocha-Canny pencil M1 - x*M2 directly, never inverting A: the well-
// conditioned case and the singular-A fallback collapse into ONE call, with
// the singular-A roots surfacing as QZ infinite eigenvalues (beta ~ 0) that we
// skip. And rather than thread the 24-vector eigenvector's block structure
// through de-equilibration, we recover v_12 fresh as the right null-vector of
// the 12x12 M(x_k) per real root (a trivial SVD, <=16 roots).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/ik_types.hpp"
#include "ssik_cpp/newton.hpp"  // lm_refine (force_refine path)
#include "ssik_cpp/rescue.hpp"

namespace ssik {

// FK-closure gate + dedup tolerance, matching the general_6r SolverSpec
// (force_refine, #528; default subproblem_numerical gate) and subproblem_dedup,
// so the native core keeps exactly the Python artifact's solution set. Marginal
// near-double-root candidates refine to well under 1e-5 (they settle ~1e-6);
// the artifact gate ceiling is per-arm (this fk_atol), not a global 1e-7 (which
// refined too many candidates and put solutions on a cross-backend-fragile
// boundary, #490 CI).
inline constexpr double kGeneral6rFkAtol = 1e-5;
inline constexpr double kGeneral6rDedupAtol = 1e-3;

// Per-arm baked constants for the RR bridge (everything except the coefficient
// matrices). poe_to_dh gives (alpha, a, d, theta_offset, T_pre, T_post) with
// FK_POE(q) = T_pre @ FK_DH(q + theta_offset) @ T_post; the pre/post inverses
// are baked so the runtime does no per-solve inversion. The four leftvar-role
// fields come from the AE-3 derivation metadata.
struct RrConsts {
  std::array<double, 6> alpha{};
  std::array<double, 6> a{};
  std::array<double, 6> d{};
  std::array<double, 6> theta_offset{};
  Eigen::Matrix4d t_pre_inv = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d t_post_inv = Eigen::Matrix4d::Identity();
  int linearity_joint = 2;
  std::array<int, 2> left_bilinear{3, 4};
  std::array<int, 2> right_bilinear{0, 1};
  int drop_joint = 5;
};

namespace rr_detail {

using Mat14x9 = Eigen::Matrix<double, 14, 9>;
using Mat14x8 = Eigen::Matrix<double, 14, 8>;
using Mat6x9 = Eigen::Matrix<double, 6, 9>;
using Mat12 = Eigen::Matrix<double, 12, 12>;
using Vec12 = Eigen::Matrix<double, 12, 1>;

// The bundle of evaluated elimination coefficients for one target pose.
struct PqCoeffs {
  Mat14x9 p_sin;
  Mat14x9 p_cos;
  Mat14x9 p_one;
  Mat14x8 q;
};

// Constant Weierstrass transform W (v_left_trig*(1+x3^2)(1+x4^2) = W @ v_left_x),
// _W_TRIG_TO_X from the Python. Rows: s3s4, s3c4, c3s4, c3c4, s3, c3, s4, c4, 1.
inline const Eigen::Matrix<double, 9, 9>& weierstrass_w() {
  static const Eigen::Matrix<double, 9, 9> W = [] {
    Eigen::Matrix<double, 9, 9> m;
    m << 0, 0, 0, 0, 4, 0, 0, 0, 0,   //
        0, 0, 0, -2, 0, 2, 0, 0, 0,   //
        0, -2, 0, 0, 0, 0, 0, 2, 0,   //
        1, 0, -1, 0, 0, 0, -1, 0, 1,  //
        0, 0, 0, 2, 0, 2, 0, 0, 0,    //
        -1, 0, -1, 0, 0, 0, 1, 0, 1,  //
        0, 2, 0, 0, 0, 0, 0, 2, 0,    //
        -1, 0, 1, 0, 0, 0, -1, 0, 1,  //
        1, 0, 1, 0, 0, 0, 1, 0, 1;
    return m;
  }();
  return W;
}

// Standard distal DH transform at a joint angle (mirrors _dh_matrix_num).
inline Eigen::Matrix4d dh_matrix(double theta, double alpha, double a, double d) {
  const double ct = std::cos(theta), st = std::sin(theta);
  const double ca = std::cos(alpha), sa = std::sin(alpha);
  Eigen::Matrix4d m;
  m << ct, -st * ca, st * sa, a * ct,  //
      st, ct * ca, -ct * sa, a * st,   //
      0.0, sa, ca, d,                  //
      0.0, 0.0, 0.0, 1.0;
  return m;
}

// Eliminate the v_right(q0,q1) monomials via the left null space of Q (14x8):
// N = last 6 columns of U in SVD(Q); E = N^T P. The particular orthonormal null
// basis is irrelevant -- a different basis N' = N R gives E' = R^T E, which
// left-multiplies M(x) by a constant invertible blkdiag(R^T,R^T), leaving both
// det M(x)=0 roots and the right null-vector v_12 unchanged.
inline void eliminate_q0_q1(const PqCoeffs& pq, Mat6x9& e_sin, Mat6x9& e_cos, Mat6x9& e_one) {
  Eigen::JacobiSVD<Mat14x8> svd(pq.q, Eigen::ComputeFullU);
  const Eigen::Matrix<double, 14, 6> n = svd.matrixU().rightCols(6);
  e_sin = n.transpose() * pq.p_sin;
  e_cos = n.transpose() * pq.p_cos;
  e_one = n.transpose() * pq.p_one;
}

// Weierstrass half-angle for q2 (quadratic in x2) + basis change for (q3,q4).
inline void weierstrass(const Mat6x9& e_sin, const Mat6x9& e_cos, const Mat6x9& e_one,
                        Mat6x9& e_quad, Mat6x9& e_lin, Mat6x9& e_const) {
  const auto& w = weierstrass_w();
  e_quad = (e_one - e_cos) * w;
  e_lin = (2.0 * e_sin) * w;
  e_const = (e_one + e_cos) * w;
}

// Embed a 6x9 E into the 12x12 doubled block (base eqs + x3-shifted eqs).
inline Mat12 embed_e(const Mat6x9& e) {
  Mat12 m = Mat12::Zero();
  m.block<6, 9>(0, 0) = e;                  // top: E in cols 0-8
  m.block<6, 6>(6, 0) = e.block<6, 6>(0, 3);  // bottom cols 0-5 <- E[:,3:9]
  m.block<6, 3>(6, 9) = e.block<6, 3>(0, 0);  // bottom cols 9-11 <- E[:,0:3]
  return m;
}

// Real tan(q2/2) roots of det M(x)=0 with their v_12 null-vectors, via the
// Manocha-Canny pencil M1 - x*M2 solved by RealQZ. Filters spurious roots near
// +/-i (the (1+x^2)^4 factor) and non-real eigenvalues, exactly as the Python
// companion route does. spurious_tol / imag_rel_tol match solve_x2_roots.
inline void solve_x2_roots(const Mat12& a_mat, const Mat12& b_mat, const Mat12& c_mat,
                           std::vector<double>& roots, std::vector<Vec12>& vecs,
                           double spurious_tol = 0.1, double imag_rel_tol = 1e-3) {
  Eigen::Matrix<double, 24, 24> m1 = Eigen::Matrix<double, 24, 24>::Zero();
  Eigen::Matrix<double, 24, 24> m2 = Eigen::Matrix<double, 24, 24>::Zero();
  m1.block<12, 12>(0, 0).setIdentity();
  m1.block<12, 12>(12, 12) = c_mat;
  m2.block<12, 12>(0, 12).setIdentity();
  m2.block<12, 12>(12, 0) = -a_mat;
  m2.block<12, 12>(12, 12) = -b_mat;

  // det(M1 - x*M2) = 0. GeneralizedEigenSolver on (M1, M2): eigenvalue = x.
  Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(Eigen::MatrixXd(m1), Eigen::MatrixXd(m2), /*computeEigenvectors=*/false);
  const auto alphas = ges.alphas();
  const auto betas = ges.betas();

  for (int i = 0; i < 24; ++i) {
    const double beta = betas(i);
    if (std::abs(beta) < 1e-12) continue;  // QZ infinite eigenvalue (singular A)
    const std::complex<double> lambda = alphas(i) / beta;
    const double re = lambda.real(), im = std::abs(lambda.imag());
    if (std::abs(im - 1.0) < spurious_tol && std::abs(re) < spurious_tol) continue;  // near +/-i
    if (im > imag_rel_tol * std::max(std::abs(re), 1.0)) continue;                    // non-real
    // v_12 = right null-vector of the real 12x12 M(re) = A re^2 + B re + C.
    const Mat12 m_x = a_mat * (re * re) + b_mat * re + c_mat;
    Eigen::JacobiSVD<Mat12> svd(m_x, Eigen::ComputeFullV);
    roots.push_back(re);
    vecs.push_back(svd.matrixV().col(11));
  }
}

// eigenvector -> (q0..q5) in DH frame + FK-closure residual. Mirrors
// _back_substitute_inner; takes the real v_12 directly. Returns false on a
// numerically degenerate branch (caller skips it).
inline bool back_substitute(double x_lin, const Vec12& v12, const PqCoeffs& pq, const RrConsts& rr,
                            const Eigen::Matrix4d& t_dh, std::array<double, 6>& q_out,
                            double& fk_err) {
  // Robust ratio selection: value == x_lb0 (or x_lb1); pick the pair whose
  // denominator entry has the largest magnitude to minimise amplified noise.
  static const int x0c[7][2] = {{5, 8}, {2, 5}, {11, 2}, {4, 7}, {10, 1}, {3, 6}, {9, 0}};
  static const int x1c[5][2] = {{7, 8}, {6, 7}, {1, 2}, {4, 5}, {10, 11}};

  auto pick = [&](const int (*cands)[2], int n, double& x) -> bool {
    int best = 0;
    for (int k = 1; k < n; ++k)
      if (std::abs(v12(cands[k][1])) > std::abs(v12(cands[best][1]))) best = k;
    const double den = v12(cands[best][1]);
    if (std::abs(den) < 1e-12) return false;
    x = v12(cands[best][0]) / den;
    return true;
  };

  double x_l0, x_l1;
  if (!pick(x0c, 7, x_l0) || !pick(x1c, 5, x_l1)) return false;

  const double q_lin = 2.0 * std::atan(x_lin);
  const double q_l0 = 2.0 * std::atan(x_l0);
  const double q_l1 = 2.0 * std::atan(x_l1);

  const double s_lin = std::sin(q_lin), c_lin = std::cos(q_lin);
  const double s_l0 = std::sin(q_l0), c_l0 = std::cos(q_l0);
  const double s_l1 = std::sin(q_l1), c_l1 = std::cos(q_l1);
  Eigen::Matrix<double, 9, 1> v_left;
  v_left << s_l0 * s_l1, s_l0 * c_l1, c_l0 * s_l1, c_l0 * c_l1, s_l0, c_l0, s_l1, c_l1, 1.0;

  const Eigen::Matrix<double, 14, 1> rhs =
      (pq.p_sin * s_lin + pq.p_cos * c_lin + pq.p_one) * v_left;
  // v_right = min-norm least-squares solution of Q v_right = rhs (== pinv(Q) @ rhs).
  // FullV (not ThinV): Eigen forbids ThinV on a fixed-size tall matrix, and the
  // SVD solve matches numpy.pinv's min-norm behaviour on rank-deficient Q.
  Eigen::JacobiSVD<Mat14x8> q_svd(pq.q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix<double, 8, 1> v_right = q_svd.solve(rhs);

  const double q_r0 = std::atan2(v_right(4), v_right(5));
  const double q_r1 = std::atan2(v_right(6), v_right(7));

  std::array<double, 6> q{};
  q[rr.linearity_joint] = q_lin;
  q[rr.left_bilinear[0]] = q_l0;
  q[rr.left_bilinear[1]] = q_l1;
  q[rr.right_bilinear[0]] = q_r0;
  q[rr.right_bilinear[1]] = q_r1;

  // Recover the drop joint from the FK residual: A_drop = chain_before^{-1} T chain_after^{-1}.
  const int drop = rr.drop_joint;
  Eigen::Matrix4d chain_before = Eigen::Matrix4d::Identity();
  for (int i = 0; i < drop; ++i) chain_before *= dh_matrix(q[i], rr.alpha[i], rr.a[i], rr.d[i]);
  Eigen::Matrix4d chain_after = Eigen::Matrix4d::Identity();
  for (int i = drop + 1; i < 6; ++i) chain_after *= dh_matrix(q[i], rr.alpha[i], rr.a[i], rr.d[i]);

  const Eigen::Matrix4d a_drop_res =
      chain_before.colPivHouseholderQr().solve(t_dh) * chain_after.inverse();
  const double q_drop = std::atan2(a_drop_res(1, 0), a_drop_res(0, 0));
  q[drop] = q_drop;

  const Eigen::Matrix4d fk =
      chain_before * dh_matrix(q_drop, rr.alpha[drop], rr.a[drop], rr.d[drop]) * chain_after;
  fk_err = (fk - t_dh).norm();
  q_out = q;
  return true;
}

// Wrap-to-pi max-joint-distance dedup, keeping the lower-residual duplicate
// (mirrors solve_all_ik's dedup loop).
inline std::vector<Solution<6>> dedup_wrap_close(const std::vector<Solution<6>>& cands,
                                                 double dedup_atol) {
  constexpr double kPi = 3.14159265358979323846;
  std::vector<Solution<6>> out;
  for (const auto& cand : cands) {
    int dup = -1;
    for (std::size_t j = 0; j < out.size() && dup < 0; ++j) {
      double worst = 0.0;
      for (int i = 0; i < 6; ++i) {
        double dd = std::fmod(cand.q[i] - out[j].q[i] + kPi, 2.0 * kPi);
        if (dd < 0) dd += 2.0 * kPi;
        worst = std::max(worst, std::abs(dd - kPi));
      }
      if (worst < dedup_atol) dup = static_cast<int>(j);
    }
    if (dup < 0)
      out.push_back(cand);
    else if (cand.fk_residual < out[dup].fk_residual)
      out[dup] = cand;
  }
  return out;
}

}  // namespace rr_detail

// The per-arm emitted coefficient evaluator: fills the four elimination
// matrices from the 12 DH-target entries (t[0,0..3], t[1,0..3], t[2,0..3]).
// Emitted as a CSE'd function in <arm>.hpp.
using RrCoeffFn = void (*)(const double t12[12], rr_detail::Mat14x9& p_sin,
                           rr_detail::Mat14x9& p_cos, rr_detail::Mat14x9& p_one,
                           rr_detail::Mat14x8& q);

// The RR analytical core: every in-frame algebraic solution for the POE target,
// FK-filtered and deduplicated. q is in the POE frame (q_dh - theta_offset);
// fk_residual is the DH-frame Frobenius residual (== POE residual under the
// rigid bridge). No limit/seed/refine/rescue logic -- that is the artifact layer.
template <class CoeffFn>
std::vector<Solution<6>> general_6r_core(const JointConsts<6>& c, const RrConsts& rr,
                                         CoeffFn&& coeffs, const Pose& t_poe, double fk_atol,
                                         double dedup_atol, bool allow_refinement,
                                         int refinement_max_iters) {
  using namespace rr_detail;
  const Eigen::Matrix4d t_dh = rr.t_pre_inv * t_poe * rr.t_post_inv;

  double t12[12];
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c) t12[r * 4 + c] = t_dh(r, c);

  PqCoeffs pq;
  coeffs(t12, pq.p_sin, pq.p_cos, pq.p_one, pq.q);

  Mat6x9 e_sin, e_cos, e_one;
  eliminate_q0_q1(pq, e_sin, e_cos, e_one);
  Mat6x9 e_quad, e_lin, e_const;
  weierstrass(e_sin, e_cos, e_one, e_quad, e_lin, e_const);

  std::vector<double> roots;
  std::vector<Vec12> vecs;
  solve_x2_roots(embed_e(e_quad), embed_e(e_lin), embed_e(e_const), roots, vecs);

  std::vector<Solution<6>> cands;
  for (std::size_t k = 0; k < roots.size(); ++k) {
    std::array<double, 6> q_dh{};
    double fk_err = 0.0;
    if (!back_substitute(roots[k], vecs[k], pq, rr, t_dh, q_dh, fk_err)) continue;
    std::array<double, 6> q_poe;
    for (int i = 0; i < 6; ++i) q_poe[i] = q_dh[i] - rr.theta_offset[i];
    if (fk_err <= fk_atol) {
      cands.push_back(Solution<6>{q_poe, fk_err, Refinement::None});
    } else if (allow_refinement && fk_err < 0.1) {
      // Refine only near-misses (fk < 0.1): a candidate already >0.1 off is an
      // eigensolve root with no real IK here, and polishing it just stalls
      // (matches the codegen refine pre-filter, #490).
      // Marginal algebraic candidate: at near-double roots the back-sub v_12 is
      // numerically delicate, leaving a genuine solution above the gate. Polish
      // it in POE frame (keep iff it converges), mirroring solve_all_ik's
      // force_refine path (#528). q_poe is exact-bridge-equivalent to q_dh.
      const auto refined = lm_refine<6>(c, q_poe, t_poe, fk_atol, refinement_max_iters);
      if (refined) cands.push_back(Solution<6>{refined->first, refined->second, Refinement::Lm});
    }
  }
  return dedup_wrap_close(cands, dedup_atol);
}

// Full artifact-contract solve for a general 6R (RR) arm. All geometry is baked
// (JointConsts for the POE FK/rescue + RrConsts for the DH solve + coeffs +
// JointLimits); no Python. Mirrors the codegen thin-wrapper solve() and the
// SRS artifact structure: limit-pass gate -> T-perturbation rescue when no
// in-limits solution exists at a reachable target (#524) -> seed/truncate.
// No seeded-track or in-limits swivel fallback: those are redundant-7R specific;
// a 6R has a discrete solution set.
template <class CoeffFn>
std::vector<Solution<6>> general_6r_artifact_solve(const JointConsts<6>& c, const RrConsts& rr,
                                                   CoeffFn&& coeffs, const JointLimits<6>& lim,
                                                   const Pose& T, const ArtifactParams<6>& p) {
  // force_refine=True on the general_6r SolverSpec (#528): the artifact always
  // polishes marginal near-double-root candidates, so the native set matches the
  // Python oracle.
  const auto core = [&](const Pose& tp) {
    return general_6r_core(c, rr, coeffs, tp, kGeneral6rFkAtol, kGeneral6rDedupAtol,
                           /*allow_refinement=*/true, p.refinement_max_iters);
  };

  // Limit pass only (no seed/tolerance/truncate): the rescue gate is "no
  // in-limits solution exists", so it must not depend on the seed filters (#524).
  ArtifactParams<6> p_limits;
  p_limits.respect_limits = p.respect_limits;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  std::vector<Solution<6>> in_limits = finalize_solutions<6>(core(T), c, lim, p_limits);

  // Rescue gate: nothing in-limits + target within reach => a measure-zero
  // rank-deficient pose where the closed form degenerates; recover via the
  // shared T-perturbation rescue, then re-apply the limit filter.
  if (in_limits.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    in_limits = finalize_solutions<6>(rescue_via_T_perturbation<6>(core, c, T), c, lim, p_limits);
  }

  ArtifactParams<6> p_seed = p;
  p_seed.respect_limits = false;
  return finalize_solutions<6>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
