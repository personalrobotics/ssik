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
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "ssik_cpp/dedup.hpp"
#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/ik_types.hpp"
#include "ssik_cpp/newton.hpp"  // lm_refine (force_refine path)
#include "ssik_cpp/polish.hpp"
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

// Baked RR elimination coefficients as a NUMERIC TENSOR -- the HP-style
// alternative to a per-arm emitted rr_coeffs() function (#555). p_sin / p_cos are
// constant in the target; p_one (14x9) and q (14x8) are degree-<=3 polynomials in
// the 12 target entries, baked SPARSE over a shared monomial basis (~800-2400
// nonzero coeffs/arm). rr_eval_coeffs (below) evaluates them at solve time, so a
// SINGLE compiled binary covers every RR arm with zero per-arm code -- unblocking
// native=True through the one shipped ext (#554), and removing the per-arm CSE
// (and its cross-BLAS drift, #536). Baked by ssik._native.rr_native_geometry;
// mathematically identical to the emitted fn (parity-gated).
struct RrCoeffTensor {
  Mat14x9 p_sin = Mat14x9::Zero();
  Mat14x9 p_cos = Mat14x9::Zero();
  // Shared monomial basis: each monomial is a product of up to 3 target-entry
  // factors (variable indices 0..11, -1 = padding); the empty product is the
  // constant 1.
  std::vector<std::array<int, 3>> mono_factors;
  // COO sparse entries for p_one and q: parallel (row, col, monomial, coeff).
  std::vector<int> po_row, po_col, po_mono;
  std::vector<double> po_coeff;
  std::vector<int> q_row, q_col, q_mono;
  std::vector<double> q_coeff;
};

// Generic RR coefficient evaluator: fill a PqCoeffs from a baked RrCoeffTensor at
// target t12. Replaces the per-arm emitted rr_coeffs() with one hand-written
// routine over baked numbers (the HP kernel pattern).
inline void rr_eval_coeffs(const RrCoeffTensor& t, const double t12[12], PqCoeffs& pq) {
  pq.p_sin = t.p_sin;
  pq.p_cos = t.p_cos;
  pq.p_one.setZero();
  pq.q.setZero();
  std::vector<double> mv(t.mono_factors.size());
  for (std::size_t m = 0; m < t.mono_factors.size(); ++m) {
    double v = 1.0;
    for (int f : t.mono_factors[m])
      if (f >= 0) v *= t12[f];
    mv[m] = v;
  }
  for (std::size_t i = 0; i < t.po_coeff.size(); ++i)
    pq.p_one(t.po_row[i], t.po_col[i]) += t.po_coeff[i] * mv[t.po_mono[i]];
  for (std::size_t i = 0; i < t.q_coeff.size(); ++i)
    pq.q(t.q_row[i], t.q_col[i]) += t.q_coeff[i] * mv[t.q_mono[i]];
}

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

// Repeated roots (#595). When k real branches share the linearity-joint value
// (finite, or pi), M(x) has a k-dimensional null space there, and any single
// null vector is an arbitrary mix of the k branches' monomial vectors: back-
// substitution reads nonsense and FK certification drops the branch. Close
// roots do the same more softly (a lone null vector's error is ~eps/delta).
// The monomial structure resolves it: every genuine v_12 satisfies
// v[hi] = x_lb0 * v[lo] over the index pairs below, so inside the null space N
// (12 x k) the branch vectors are exactly the eigenvectors c of the k x k
// pencil N[hi] c = w N[lo] c. When the branches share x_lb0 too, the x_lb1
// shift separates them. Mirrors _raghavan_roth.split_repeated_roots.

// Numerical rank tolerance for M(x), relative to its largest singular value:
// sqrt(eps). A lone null vector read from a space whose next singular value is
// s has error ~eps/s, which passes sqrt(eps) exactly when s drops below it; the
// split reads each branch from its own root's space at ~eps however close the
// roots are.
inline constexpr double kNullRankRtol = 1.4901161193847656e-08;  // sqrt(DBL_EPSILON)

inline constexpr int kShiftLb0Lo[9] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
inline constexpr int kShiftLb0Hi[9] = {5, 4, 3, 2, 1, 0, 11, 10, 9};
inline constexpr int kShiftLb1Lo[8] = {8, 7, 5, 4, 2, 1, 11, 10};
inline constexpr int kShiftLb1Hi[8] = {7, 6, 4, 3, 1, 0, 10, 9};

using NullBasis = Eigen::Matrix<double, 12, Eigen::Dynamic>;

// Branch vectors in span(n_basis) from one shift; false when the shift does not
// give k distinct real values.
template <int R>
inline bool shift_split_one(const NullBasis& n_basis, const int (&lo)[R], const int (&hi)[R],
                            double imag_tol, std::vector<Vec12>& out) {
  const int k = static_cast<int>(n_basis.cols());
  if (k > R) return false;
  Eigen::MatrixXd low(R, k), high(R, k);
  for (int r = 0; r < R; ++r) {
    low.row(r) = n_basis.row(lo[r]);
    high.row(r) = n_basis.row(hi[r]);
  }
  // Both sides map into the same k-dim span, even where w is infinite and the
  // low side vanishes; square the pencil up on that span.
  Eigen::MatrixXd both(R, 2 * k);
  both << low, high;
  Eigen::JacobiSVD<Eigen::MatrixXd> span(both, Eigen::ComputeThinU);
  const Eigen::MatrixXd u = span.matrixU().leftCols(k);
  const Eigen::MatrixXd low_k = u.transpose() * low, high_k = u.transpose() * high;

  Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
  ges.compute(high_k, low_k, /*computeEigenvectors=*/false);
  if (ges.info() != Eigen::Success) return false;
  std::vector<Vec12> parts;
  for (int i = 0; i < k; ++i) {
    const std::complex<double> alpha = ges.alphas()(i);
    const double beta = ges.betas()(i);
    const double scale = std::hypot(std::abs(alpha), std::abs(beta));
    if (!(scale > 0.0) || !std::isfinite(scale)) return false;
    const std::complex<double> a = alpha / scale;
    if (std::abs(a.imag()) > imag_tol) return false;  // complex pair: not real branches
    // c spans the null space of beta*H - alpha*L; read it by SVD, not from a QZ
    // eigenvector, which is ill-defined at beta = 0.
    const Eigen::MatrixXd p = (beta / scale) * high_k - a.real() * low_k;
    Eigen::JacobiSVD<Eigen::MatrixXd> null(p, Eigen::ComputeFullV);
    const auto& s = null.singularValues();
    if (k > 1 && s(k - 2) <= kNullRankRtol * s(0)) return false;  // w repeats
    const Vec12 v = n_basis * null.matrixV().col(k - 1);
    parts.push_back(v.normalized());
  }
  out = std::move(parts);
  return true;
}

inline bool shift_split(const NullBasis& n_basis, double imag_tol, std::vector<Vec12>& out) {
  return shift_split_one(n_basis, kShiftLb0Lo, kShiftLb0Hi, imag_tol, out) ||
         shift_split_one(n_basis, kShiftLb1Lo, kShiftLb1Hi, imag_tol, out);
}

// Distance of two points of the projective line (bounded, inf-safe).
inline double chordal(double x, double y) {
  if (!std::isfinite(x)) return std::isfinite(y) ? 1.0 / std::hypot(1.0, y) : 0.0;
  if (!std::isfinite(y)) return 1.0 / std::hypot(1.0, x);
  return std::abs(x - y) / (std::hypot(1.0, x) * std::hypot(1.0, y));
}

// An accepted root with M(x) (A at infinity) and its SVD.
struct RootSvd {
  double x;
  Mat12 m;
  Vec12 sv;  // singular values, descending
  Mat12 v;   // right singular vectors
};

// Emit (root, v_12) pairs, splitting every repeated root into its branches.
// Multiplicity is read from M(x)'s singular values rather than from how close
// the eigenvalues are: a defective double root has two equal eigenvalues and a
// one-dimensional null space, and needs no split. A root whose null space is
// one-dimensional keeps its null vector as before. A root with a k-dim null
// space (k >= 2) is grouped with up to k-1 other roots for which that space is
// also null (QZ's copies of a k-fold root, or the close roots of a near
// repeat); the space is split into its k branches; each branch goes to the
// member it fits best, one each; and a member reads its branch from the split
// of its OWN null space, exact at its own root rather than only to within the
// roots' separation. Branches no member took are still emitted, so a missing
// eigenvalue copy cannot lose one. A group whose space does not split into k
// distinct real branches is emitted exactly as before.
inline void emit_split_roots(const std::vector<RootSvd>& acc, double imag_tol,
                             std::vector<double>& roots, std::vector<Vec12>& vecs) {
  const int n = static_cast<int>(acc.size());
  std::vector<int> ks(n);
  for (int j = 0; j < n; ++j) {
    int k = 0;
    for (int r = 0; r < 12; ++r)
      if (acc[j].sv(r) <= kNullRankRtol * acc[j].sv(0)) ++k;
    ks[j] = k;
  }
  auto basis = [&](int j) -> NullBasis { return acc[j].v.rightCols(ks[j]); };
  auto residual = [&](int j, const Vec12& w) { return (acc[j].m * w).norm() / acc[j].sv(0); };
  auto emit = [&](int j, const Vec12& w) {
    roots.push_back(acc[j].x);
    vecs.push_back(w);
  };

  std::vector<bool> used(n, false);
  for (int i = 0; i < n; ++i) {
    if (used[i]) continue;
    const int k = ks[i];
    if (k < 2) {
      used[i] = true;
      emit(i, acc[i].v.col(11));
      continue;
    }
    const NullBasis nb = basis(i);
    std::vector<int> members{i};
    std::vector<std::pair<double, int>> partners;
    for (int j = 0; j < n; ++j)
      if (j != i && !used[j] && ks[j] >= 2) partners.emplace_back(chordal(acc[i].x, acc[j].x), j);
    std::sort(partners.begin(), partners.end());
    for (const auto& pj : partners) {
      if (static_cast<int>(members.size()) == k) break;
      const int j = pj.second;
      if ((acc[j].m * nb).norm() <= kNullRankRtol * acc[j].sv(0)) members.push_back(j);
    }
    for (int j : members) used[j] = true;

    std::vector<Vec12> branches;
    if (!shift_split(nb, imag_tol, branches)) {
      for (int j : members) emit(j, acc[j].v.col(11));
      continue;
    }
    // One branch per member, cheapest fit first.
    std::vector<std::tuple<double, int, int>> costs;
    for (int j : members)
      for (int b = 0; b < static_cast<int>(branches.size()); ++b)
        costs.emplace_back(residual(j, branches[b]), j, b);
    std::sort(costs.begin(), costs.end());
    std::vector<int> taken_members;
    std::vector<bool> taken_branch(branches.size(), false);
    for (const auto& [cost, j, b] : costs) {
      (void)cost;
      if (taken_branch[b] ||
          std::find(taken_members.begin(), taken_members.end(), j) != taken_members.end())
        continue;
      taken_members.push_back(j);
      taken_branch[b] = true;
      Vec12 w = branches[b];
      if (j != i && ks[j] == k) {
        std::vector<Vec12> own;
        if (shift_split(basis(j), imag_tol, own)) {
          double best = -1.0;
          for (const auto& o : own) {
            const double overlap = std::abs(o.dot(branches[b]));
            if (overlap > best) {
              best = overlap;
              w = o;
            }
          }
        }
      }
      emit(j, w);
    }
    for (int b = 0; b < static_cast<int>(branches.size()); ++b) {
      if (taken_branch[b]) continue;
      int best_j = members[0];
      for (int j : members)
        if (residual(j, branches[b]) < residual(best_j, branches[b])) best_j = j;
      emit(best_j, branches[b]);
    }
  }
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
  // RealQZ's default cap is 400 iterations without a deflation. A pencil that
  // is numerically singular (det M(x) ~ 0 for every x, as JACO 2 gives under
  // linearity 0 or 2) can exhaust it on the rounding-level indeterminate
  // eigenvalues before the regular ones deflate; Eigen 3.5 does on #599's
  // pose, 3.4 does not. Same generous budget the Husty-Pfurner QZ uses.
  ges.setMaxIterations(400 * 24);
  ges.compute(Eigen::MatrixXd(m1), Eigen::MatrixXd(m2), /*computeEigenvectors=*/false);
  // A QZ that still fails leaves alphas/betas unsized, so reading them indexes
  // past an empty buffer (#599's segfault). No converged Schur form means no
  // eigenvalues to trust: report no roots, which the FK-certified caller
  // treats like any other pose with nothing found (rescue, then empty).
  if (ges.info() != Eigen::Success) return;
  const auto alphas = ges.alphas();
  const auto betas = ges.betas();

  std::vector<RootSvd> accepted;
  auto accept = [&](double x, const Mat12& m) {
    Eigen::JacobiSVD<Mat12> svd(m, Eigen::ComputeFullV);
    accepted.push_back(RootSvd{x, m, svd.singularValues(), svd.matrixV()});
  };
  for (Eigen::Index i = 0; i < alphas.size(); ++i) {
    // Work the pair (alpha, beta) projectively rather than forming alpha/beta
    // straight away (#571). beta == 0 is a QZ eigenvalue at infinity, which in
    // this coordinate is x = tan(q/2) -> infinity, i.e. the joint at exactly
    // pi: a real root of the pencil, not a failure. Normalising the pair puts
    // the "is beta zero" test on a scale-free footing.
    const std::complex<double> alpha = alphas(i);
    const double beta = betas(i);
    const double pair_norm = std::hypot(std::abs(alpha), std::abs(beta));
    if (!(pair_norm > 0.0) || !std::isfinite(pair_norm)) continue;  // degenerate pencil row
    const std::complex<double> a_n = alpha / pair_norm;
    const double b_n = beta / pair_norm;

    if (std::abs(b_n) < 1e-12) {
      // Root at infinity. Non-real alpha here is not a real branch; a real
      // alpha gives q = pi, and M(alpha, 0) = alpha^2 * A, so the null
      // vector is A's -- no division by the vanishing beta anywhere. The
      // projective line has one point at infinity, so it is always +inf (the
      // sign of the pair is QZ's choice); Python's Mobius path does the same.
      if (std::abs(a_n.imag()) > imag_rel_tol) continue;
      accept(std::numeric_limits<double>::infinity(), a_mat);
      continue;
    }

    const std::complex<double> lambda = a_n / b_n;
    const double re = lambda.real(), im = std::abs(lambda.imag());
    if (std::abs(im - 1.0) < spurious_tol && std::abs(re) < spurious_tol) continue;  // near +/-i
    if (im > imag_rel_tol * std::max(std::abs(re), 1.0)) continue;                    // non-real
    // v_12 = right null-vector of the real 12x12 M(re) = A re^2 + B re + C.
    accept(re, a_mat * (re * re) + b_mat * re + c_mat);
  }
  emit_split_roots(accepted, imag_rel_tol, roots, vecs);
}

// eigenvector -> (q0..q5) in DH frame + FK-closure residual. Mirrors
// _back_substitute_inner; takes the real v_12 directly. Returns false on a
// numerically degenerate branch (caller skips it).
inline bool back_substitute(double x_lin, const Vec12& v12, const PqCoeffs& pq, const RrConsts& rr,
                            const Eigen::Matrix4d& t_dh, std::array<double, 6>& q_out,
                            double& fk_err) {
  // Each pair differs by one degree in its variable, so it *is* that
  // variable's homogeneous coordinate [num : den] and 2*atan2 reads the angle
  // off it. Dividing first loses the pole (#571): a joint at pi sends its
  // variable to infinity, v12 is unit-normalised, and the whole low-degree
  // block drops to ~1e-16, so every ratio that divides by it divides noise.
  // Hence: select on the entry carrying signal, not on the denominator.
  static const int x0c[7][2] = {{5, 8}, {2, 5}, {11, 2}, {4, 7}, {10, 1}, {3, 6}, {9, 0}};
  static const int x1c[5][2] = {{7, 8}, {6, 7}, {1, 2}, {4, 5}, {10, 11}};

  const double v_scale = v12.cwiseAbs().maxCoeff();
  if (!(v_scale > 0.0) || !std::isfinite(v_scale)) return false;
  const double floor = 1e-12 * v_scale;

  auto pick = [&](const int (*cands)[2], int n, double& q) -> bool {
    auto signal = [&](int k) {
      return std::max(std::abs(v12(cands[k][0])), std::abs(v12(cands[k][1])));
    };
    int best = 0;
    for (int k = 1; k < n; ++k)
      if (signal(k) > signal(best)) best = k;
    if (signal(best) < floor) return false;  // both entries are noise
    // [num : den] == [-num : -den] but 2*atan2 differs by 2*pi, and the
    // eigenvector sign is the eigensolver's choice; canonicalise so q is in
    // (-pi, pi] on both backends (Python mirrors this).
    double num = v12(cands[best][0]), den = v12(cands[best][1]);
    if (den < 0.0 || (den == 0.0 && num < 0.0)) { num = -num; den = -den; }
    q = 2.0 * std::atan2(num, den);
    return true;
  };

  double q_l0, q_l1;
  if (!pick(x0c, 7, q_l0) || !pick(x1c, 5, q_l1)) return false;

  // atan(inf) is exactly pi/2, so a root at infinity gives q_lin = pi.
  const double q_lin = 2.0 * std::atan(x_lin);

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

// FK and spatial Jacobian of the standard-DH chain at q (DH frame), columns
// (p_i x z_i ; z_i): the convention of spatial_jacobian<N> and of the twist
// se3_log_residual measures, so a Newton step on it converges quadratically.
// Mirrors ssik.refinement.polish.Chain.from_dh.
inline Eigen::Matrix4d dh_fk(const RrConsts& rr, const std::array<double, 6>& q) {
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 6; ++i) t = t * dh_matrix(q[i], rr.alpha[i], rr.a[i], rr.d[i]);
  return t;
}

inline Eigen::Matrix<double, 6, 6> dh_spatial_jacobian(const RrConsts& rr,
                                                       const std::array<double, 6>& q) {
  Eigen::Matrix<double, 6, 6> jac;
  Eigen::Matrix4d t = Eigen::Matrix4d::Identity();
  for (int i = 0; i < 6; ++i) {
    const Eigen::Vector3d z = t.block<3, 1>(0, 2);
    const Eigen::Vector3d p = t.block<3, 1>(0, 3);
    jac.col(i).head<3>() = p.cross(z);
    jac.col(i).tail<3>() = z;
    t = t * dh_matrix(q[i], rr.alpha[i], rr.a[i], rr.d[i]);
  }
  return jac;
}

// Same-root dedup (#600): two eigenvalues can back-substitute to one
// configuration, but twin roots near a fold are distinct however close.
// dedup_atol is the pre-filter radius (mirrors solve_all_ik's dedup).
inline std::vector<Solution<6>> dedup_wrap_close(const std::vector<Solution<6>>& cands,
                                                 const JointConsts<6>& c, const Pose& t_poe,
                                                 double dedup_atol) {
  return dedup_same_root<6>(cands, c, t_poe, dedup_atol);
}

}  // namespace rr_detail

// The per-arm emitted coefficient evaluator: fills the four elimination
// matrices from the 12 DH-target entries (t[0,0..3], t[1,0..3], t[2,0..3]).
// Emitted as a CSE'd function in <arm>.hpp.
using RrCoeffFn = void (*)(const double t12[12], rr_detail::Mat14x9& p_sin,
                           rr_detail::Mat14x9& p_cos, rr_detail::Mat14x9& p_one,
                           rr_detail::Mat14x8& q);

// Which chain the core polishes accepted candidates on (polish.hpp), or none.
// The 6R artifact polishes on its POE chain `c`, as the Python artifact does;
// the jointlock sweep has no POE sub-chain here, so it polishes on the DH chain,
// as Python's solve_all_ik does.
enum class RrPolish : std::uint8_t { Off, Poe, Dh };

// The RR analytical core: every in-frame algebraic solution for the POE target,
// FK-filtered, polished (`polish`) and deduplicated. q is in the POE frame
// (q_dh - theta_offset); fk_residual is the DH-frame Frobenius residual (== POE
// residual under the rigid bridge), or the polished chain's. `lim` (POE frame)
// only keeps the polish from crossing a limit (polish.hpp); no candidate is
// dropped for its limits, and there is no seed/rescue logic -- that is the
// artifact layer.
template <class CoeffFn>
std::vector<Solution<6>> general_6r_core(const JointConsts<6>& c, const RrConsts& rr,
                                         CoeffFn&& coeffs, const Pose& t_poe, double fk_atol,
                                         double dedup_atol, bool allow_refinement,
                                         int refinement_max_iters,
                                         RrPolish polish = RrPolish::Off,
                                         const JointLimits<6>& lim = {}) {
  using namespace rr_detail;
  const Eigen::Matrix4d t_dh = rr.t_pre_inv * t_poe * rr.t_post_inv;
  // The limits in DH coordinates (q_dh = q_poe + theta_offset), for the DH polish.
  JointLimits<6> lim_dh = lim;
  for (int i = 0; i < 6; ++i) {
    lim_dh.lo[i] += rr.theta_offset[i];
    lim_dh.hi[i] += rr.theta_offset[i];
  }

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
    // Accepted: polish to machine precision on the same branch before the
    // same-root dedup below sees it (polish.hpp; Python polishes the same chain).
    const bool accepted = fk_err <= fk_atol;
    if (accepted && polish == RrPolish::Dh)
      polish_accepted<6>([&](const std::array<double, 6>& x) { return dh_fk(rr, x); },
                         [&](const std::array<double, 6>& x) { return dh_spatial_jacobian(rr, x); },
                         t_dh, lim_dh, q_dh, fk_err);
    std::array<double, 6> q_poe;
    for (int i = 0; i < 6; ++i) q_poe[i] = q_dh[i] - rr.theta_offset[i];
    if (accepted) {
      if (polish == RrPolish::Poe)
        polish_accepted<6>([&](const std::array<double, 6>& x) { return fk<6>(c, x); },
                           [&](const std::array<double, 6>& x) { return spatial_jacobian<6>(c, x); },
                           t_poe, lim, q_poe, fk_err);
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
  return dedup_wrap_close(cands, c, t_poe, dedup_atol);
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
    auto sols = general_6r_core(c, rr, coeffs, tp, kGeneral6rFkAtol, kGeneral6rDedupAtol,
                               /*allow_refinement=*/true, p.refinement_max_iters,
                               RrPolish::Poe, lim);
    // POE-FK re-verify (#533): general_6r_core filters the DH-frame residual, but
    // the rigid poe_to_dh bridge is slightly inconsistent at degenerate geometry
    // (DH-FK closes, POE-FK does not). Re-verify against the actual POE target,
    // as every other family does; drop candidates that miss it.
    std::vector<Solution<6>> verified;
    for (auto& s : sols) {
      const double poe_fk = (fk<6>(c, s.q) - tp).norm();
      if (poe_fk <= kGeneral6rFkAtol) {
        s.fk_residual = poe_fk;
        verified.push_back(s);
      }
    }
    return verified;
  };

  // Limit pass only (no seed/tolerance/truncate): the rescue gate is "no
  // in-limits solution exists", so it must not depend on the seed filters (#524).
  ArtifactParams<6> p_limits;
  // Intermediate pass: never lift here, or the lifts would be produced
  // twice (see ArtifactParams::enumerate_windings).
  p_limits.enumerate_windings = false;
  p_limits.respect_limits = p.respect_limits;
  p_limits.wrap_only = p.wrap_only;
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
  // The single lifting stage (#562): this is the call that yields the returned
  // set. Skipped when the caller wanted the raw geometric set.
  p_seed.enumerate_windings = p.enumerate_windings && p.respect_limits;
  return finalize_solutions<6>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
