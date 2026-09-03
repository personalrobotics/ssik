// Husty-Pfurner universal-6R analytical IK -- shared native runtime (#491).
//
// Ports the numeric HP elimination kernel from ssik.solvers.husty_pfurner
// (_eliminate.py / _back_substitute.py / _pencil.py). Unlike RR there is no
// per-arm sympy->C coefficient function: precompute_rrr_chain runs the whole
// symbolic build at BUILD time and emits numeric tensors T_u / T_w_pre (baked in
// HpConsts). Only sigma_E (the Study DQ of the target) enters at solve time, so
// everything here is a fixed numeric pipeline parameterized by baked constants.
//
// Slice 1 (#537): the f/g stage -- sigma_E injection + Cramer 5x4
// evaluation-interpolation + 2-D convolution -> the Study quadric f(u,w) (9x7)
// and dropped-row g(u,w) (6x5). Parity-gated vs _eliminate.compute_fg_numeric.
// Slice 2 (#538): Sylvester pencil + 80x80 eigensolve -> candidate u roots.
// Slice 3 (#539): (u,w) Newton refinement + back-substitution -> joint angles.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/newton.hpp"   // lm_refine
#include "ssik_cpp/quartic.hpp"  // np_roots (companion-matrix roots) for _initial_w_for
#include "ssik_cpp/rescue.hpp"
#include "ssik_cpp/study_dq.hpp"

namespace ssik {

// Per-arm baked HP constants. The (4,8,2) tensors T_u / T_w_pre are stored as
// their two degree slices [...,0] and [...,1] (each 4x8). Emitted from
// EliminatePrecompute (_eliminate.py:163). Slice 3 adds the target->sigma_E
// bridge transforms, the dispatch flags, and the perturbed DH.
struct HpConsts {
  std::array<Eigen::Matrix<double, 4, 8>, 2> t_u{};
  std::array<Eigen::Matrix<double, 4, 8>, 2> t_w_pre{};
  int drop_idx = 7;
  // Back-substitution DH (the possibly-perturbed chain), 1-indexed to match the
  // HP naming: a[1..5] link lengths, l[1..5]=tan(alpha/2) twists, d[2..5] offsets
  // (a[0]/l[0]/d[0]/d[1] unused; d_1 = 0 per HP convention).
  std::array<double, 6> a{};
  std::array<double, 6> l{};
  std::array<double, 6> d{};
  // Dispatch (baked at emit time from the singular-DH predicates):
  int parametric_var = 0;        // u = 0:v_1 (Tv1) | 1:v_2 (Tv2, not yet native)
  int right_parametric_var = 0;  // w = 0:v_6 (Tv6) | 1:v_4 (Tv4)
  // Target -> sigma_E bridge (all baked): t_hp = t_z_neg_d1 (T_pre^-1 . T .
  // T_post^-1) t_joint6_offset^-1, then sigma_E = dq_from_se3(t_hp). And the DH
  // theta_offset to land recovered q back in the POE frame (q = 2 atan(v) -
  // theta_offset).
  Eigen::Matrix4d t_pre_inv = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d t_post_inv = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d t_z_neg_d1 = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d t_joint6_offset_inv = Eigen::Matrix4d::Identity();
  std::array<double, 6> theta_offset{};
};

namespace hp_detail {

using Mat4x8 = Eigen::Matrix<double, 4, 8>;
using Mat5x4 = Eigen::Matrix<double, 5, 4>;
using Mat9x7 = Eigen::Matrix<double, 9, 7>;
using Mat6x5 = Eigen::Matrix<double, 6, 5>;
// The 8 Cramer-cofactor bivariate polynomials, each a (5,4) = (u-power, w-power)
// coefficient block. Mirrors P_coef (8,5,4) in _cramer_8vec_via_interp.
using PCoef = std::array<Mat5x4, 8>;

// Fixed Cramer evaluation grid (_eliminate.py:157-160). U has 5 points, W has 4.
inline const std::array<double, 5>& cramer_u_grid() {
  static const std::array<double, 5> g = {-2.0, -1.0, 0.0, 1.0, 2.0};
  return g;
}
inline const std::array<double, 4>& cramer_w_grid() {
  static const std::array<double, 4> g = {-1.5, -0.5, 0.5, 1.5};
  return g;
}
// Inverse Vandermonde (increasing powers) of each grid: V[i,j] = grid[i]^j.
inline const Eigen::Matrix<double, 5, 5>& cramer_vu_inv() {
  static const Eigen::Matrix<double, 5, 5> m = [] {
    Eigen::Matrix<double, 5, 5> v;
    const auto& g = cramer_u_grid();
    for (int i = 0; i < 5; ++i) {
      double p = 1.0;
      for (int j = 0; j < 5; ++j) {
        v(i, j) = p;
        p *= g[i];
      }
    }
    return Eigen::Matrix<double, 5, 5>(v.inverse());
  }();
  return m;
}
inline const Eigen::Matrix<double, 4, 4>& cramer_vw_inv() {
  static const Eigen::Matrix<double, 4, 4> m = [] {
    Eigen::Matrix<double, 4, 4> v;
    const auto& g = cramer_w_grid();
    for (int i = 0; i < 4; ++i) {
      double p = 1.0;
      for (int j = 0; j < 4; ++j) {
        v(i, j) = p;
        p *= g[i];
      }
    }
    return Eigen::Matrix<double, 4, 4>(v.inverse());
  }();
  return m;
}

// Full 2-D linear convolution of coefficient blocks (numpy/scipy convolve2d,
// mode='full'): out has shape (A.rows+B.rows-1, A.cols+B.cols-1). Accumulates
// into `out` (caller zeroes / sizes it), so a summed convolution is one call
// per term.
template <class MatA, class MatB, class MatOut>
inline void convolve2d_add(const MatA& a, const MatB& b, MatOut& out) {
  for (int ar = 0; ar < a.rows(); ++ar)
    for (int ac = 0; ac < a.cols(); ++ac) {
      const double av = a(ar, ac);
      if (av == 0.0) continue;
      for (int br = 0; br < b.rows(); ++br)
        for (int bc = 0; bc < b.cols(); ++bc)
          out(ar + br, ac + bc) += av * b(br, bc);
    }
}

// T_w = T_w_pre @ M(sigma_E^*), the pose-dependent right chain
// (_apply_sigma_e_to_tw_pre). Returns the two 4x8 degree slices.
inline std::array<Mat4x8, 2> apply_sigma_e(const std::array<Mat4x8, 2>& t_w_pre,
                                           const Vec8& sigma_E) {
  Vec8 conj;
  conj << sigma_E[0], -sigma_E[1], -sigma_E[2], -sigma_E[3],  //
      sigma_E[4], -sigma_E[5], -sigma_E[6], -sigma_E[7];
  const Mat8 M_e = study::dq_left_mult_matrix(conj);
  return {Mat4x8(t_w_pre[0] * M_e), Mat4x8(t_w_pre[1] * M_e)};
}

// Cramer 8-vec cofactors as a (8,5,4) coefficient tensor
// (_cramer_8vec_via_interp): evaluate the 7x7 system on the 5x4 grid, one LU
// (solve + det) per point, then interpolate via the Vandermonde inverses.
inline PCoef cramer_8vec_via_interp(const std::array<Mat4x8, 2>& t_u,
                                    const std::array<Mat4x8, 2>& t_w, int drop_idx) {
  // P_grid[component](u-index, w-index).
  std::array<Mat5x4, 8> p_grid;
  for (auto& m : p_grid) m.setZero();
  const auto& ug = cramer_u_grid();
  const auto& wg = cramer_w_grid();
  for (int iu = 0; iu < 5; ++iu) {
    for (int jw = 0; jw < 4; ++jw) {
      Mat8 full;
      full.block<4, 8>(0, 0) = t_u[0] + ug[iu] * t_u[1];
      full.block<4, 8>(4, 0) = t_w[0] + wg[jw] * t_w[1];
      // Drop row drop_idx -> 7x8, split A = cols[1:] (7x7), rhs = -col[0].
      Eigen::Matrix<double, 7, 8> m7;
      int r = 0;
      for (int k = 0; k < 8; ++k)
        if (k != drop_idx) m7.row(r++) = full.row(k);
      const Eigen::Matrix<double, 7, 7> A = m7.block<7, 7>(0, 1);
      const Eigen::Matrix<double, 7, 1> rhs = -m7.block<7, 1>(0, 0);
      const Eigen::Matrix<double, 7, 1> x = A.lu().solve(rhs);
      const double d = A.determinant();
      p_grid[0](iu, jw) = d;
      for (int c = 0; c < 7; ++c) p_grid[c + 1](iu, jw) = x[c] * d;
    }
  }
  // P_coef[k] = VU_INV @ P_grid[k] @ VW_INV^T (the two interpolation einsums).
  const auto& vu = cramer_vu_inv();
  const auto& vw = cramer_vw_inv();
  PCoef p_coef;
  for (int k = 0; k < 8; ++k) p_coef[k] = vu * p_grid[k] * vw.transpose();
  return p_coef;
}

// f(u,w) = sum_{i=0..3} P_i * P_{i+4}, the Study quadric (_study_quadric_f) (9x7).
inline Mat9x7 study_quadric_f(const PCoef& p) {
  Mat9x7 f = Mat9x7::Zero();
  for (int i = 0; i < 4; ++i) convolve2d_add(p[i], p[i + 4], f);
  return f;
}

// g(u,w) = c_dropped(u,w) . P(u,w), the dropped-row residual (_dropped_row_g)
// (6x5). c is a T(u) row (degree (1,0)) for drop<4, else a T(w) row ((0,1)).
inline Mat6x5 dropped_row_g(const std::array<Mat4x8, 2>& t_u, const std::array<Mat4x8, 2>& t_w,
                            const PCoef& p, int drop_idx) {
  Mat6x5 g = Mat6x5::Zero();
  for (int comp = 0; comp < 8; ++comp) {
    if (drop_idx < 4) {
      Eigen::Matrix<double, 2, 1> c;
      c << t_u[0](drop_idx, comp), t_u[1](drop_idx, comp);
      convolve2d_add(c, p[comp], g);
    } else {
      Eigen::Matrix<double, 1, 2> c;
      c << t_w[0](drop_idx - 4, comp), t_w[1](drop_idx - 4, comp);
      convolve2d_add(c, p[comp], g);
    }
  }
  return g;
}

// The full f/g stage: sigma_E injection -> Cramer -> quadric + dropped row.
// Mirrors _eliminate.compute_fg_numeric. `f` is (9,7), `g` is (6,5), indexed
// [u-power, w-power].
// f/g from the (already sigma_E-injected) T_w tensor -- Cramer -> quadric +
// dropped row (_eliminate._compute_fg_with_tw). Split out so eliminate_uw_pairs
// hoists the drop-independent apply_sigma_e once across the drop-index loop.
inline void compute_fg_with_tw(const std::array<Mat4x8, 2>& t_u, const std::array<Mat4x8, 2>& t_w,
                               int drop_idx, Mat9x7& f, Mat6x5& g) {
  const PCoef p = cramer_8vec_via_interp(t_u, t_w, drop_idx);
  f = study_quadric_f(p);
  g = dropped_row_g(t_u, t_w, p, drop_idx);
}

inline void compute_fg(const HpConsts& hp, const Vec8& sigma_E, int drop_idx, Mat9x7& f,
                       Mat6x5& g) {
  compute_fg_with_tw(hp.t_u, apply_sigma_e(hp.t_w_pre, sigma_E), drop_idx, f, g);
}

// =============================================================================
// Sylvester pencil + 80x80 generalized eigensolve (#538). From f (9x7), g (6x5)
// to the candidate real u roots. For the 6R HP chain the degrees are fixed:
// deg_w(f)=6, deg_w(g)=4 -> n=10; deg_u(f)=8, deg_u(g)=5 -> max_d=8; so the
// Sylvester tensor is S (9,10,10) and its Frobenius companion pencil is 80x80.
// =============================================================================

using Mat10 = Eigen::Matrix<double, 10, 10>;
using Vec10 = Eigen::Matrix<double, 10, 1>;
using Pencil = std::array<Mat10, 9>;  // S[d], d = 0..8

// Sylvester matrix pencil S(u) = sum_d S[d] u^d (_eliminate.build_pencil_tensor).
// Top deg_w(g)=4 rows are shifted f-rows (descending in w); bottom deg_w(f)=6
// rows are shifted g-rows.
inline Pencil build_pencil_tensor(const Mat9x7& f, const Mat6x5& g) {
  constexpr int deg_w_f = 6, deg_w_g = 4;
  Pencil S;
  for (auto& m : S) m.setZero();
  for (int shift = 0; shift < deg_w_g; ++shift)
    for (int d = 0; d < 9; ++d)
      for (int k = 0; k < 7; ++k) S[d](shift, shift + (deg_w_f - k)) += f(d, k);
  for (int shift = 0; shift < deg_w_f; ++shift)
    for (int d = 0; d < 6; ++d)
      for (int k = 0; k < 5; ++k) S[d](deg_w_g + shift, shift + (deg_w_g - k)) += g(d, k);
  return S;
}

// Row + column equilibration with variable rescale (_pencil.equilibrate_
// polynomial_matrix, rescale_variable=True). Returns S_eq; scale_c is written
// out (eigenvalues of S_eq(x_internal)=0 times scale_c give those of S(x)=0).
// d_l/d_r are not needed for eigenvalues, so they are not returned.
inline Pencil equilibrate(const Pencil& S, double& scale_c) {
  const double norm_0 = S[0].cwiseAbs().maxCoeff();
  const double norm_d = S[8].cwiseAbs().maxCoeff();
  scale_c = (norm_0 > 0.0 && norm_d > 0.0) ? std::pow(norm_0 / norm_d, 1.0 / 8.0) : 1.0;

  Pencil ss;
  if (scale_c != 1.0)
    for (int k = 0; k < 9; ++k) ss[k] = S[k] * std::pow(scale_c, k);
  else
    ss = S;

  Vec10 row_max = Vec10::Zero();
  for (int k = 0; k < 9; ++k) row_max = row_max.cwiseMax(ss[k].cwiseAbs().rowwise().maxCoeff());
  for (int i = 0; i < 10; ++i)
    if (row_max[i] <= 0.0) row_max[i] = 1.0;
  const Vec10 d_l = row_max.cwiseInverse();

  Pencil after_row;
  for (int k = 0; k < 9; ++k) after_row[k] = d_l.asDiagonal() * ss[k];

  Vec10 col_max = Vec10::Zero();
  for (int k = 0; k < 9; ++k)
    col_max = col_max.cwiseMax(after_row[k].cwiseAbs().colwise().maxCoeff().transpose());
  for (int j = 0; j < 10; ++j)
    if (col_max[j] <= 0.0) col_max[j] = 1.0;
  const Vec10 d_r = col_max.cwiseInverse();

  Pencil s_eq;
  for (int k = 0; k < 9; ++k) s_eq[k] = after_row[k] * d_r.asDiagonal();
  return s_eq;
}

// Frobenius companion linearization of S(x) into (A, B) with A v = x B v
// (_pencil.build_frobenius_pencil_pair). d=8, n=10 -> 80x80. B = block_diag(I x
// 7, S[8]); A has identity superdiagonal blocks and bottom row [-S[0]..-S[7]].
inline void build_frobenius(const Pencil& s, Eigen::MatrixXd& A, Eigen::MatrixXd& B) {
  constexpr int n = 10, d = 8, nd = 80;
  A = Eigen::MatrixXd::Zero(nd, nd);
  B = Eigen::MatrixXd::Identity(nd, nd);
  for (int k = 0; k < d - 1; ++k) A.block<n, n>(k * n, (k + 1) * n).setIdentity();
  for (int k = 0; k < d; ++k) A.block<n, n>((d - 1) * n, k * n) = -s[k];
  B.block<n, n>((d - 1) * n, (d - 1) * n) = s[d];
}

// End-to-end det S(u)=0 solve (solve_pencil_eigenvalues -> _pencil.solve_
// polynomial_matrix_eigenvalues): equilibrate -> Frobenius -> generalized
// eigensolve -> filter finite / |re|<=max_magnitude / near-real. Returns the
// sorted finite real candidate u values.
inline std::vector<double> solve_pencil_eigenvalues(const Mat9x7& f, const Mat6x5& g,
                                                    double real_tol = 1e-3,
                                                    double max_magnitude = 1e10) {
  constexpr int n = 10, d = 8, nd = 80;
  double scale_c = 1.0;
  const Pencil s_eq = equilibrate(build_pencil_tensor(f, g), scale_c);

  std::vector<double> out;
  auto keep = [&](std::complex<double> e) {
    e *= scale_c;
    if (!std::isfinite(e.real()) || !std::isfinite(e.imag())) return;
    const double re = e.real(), im = e.imag();
    if (std::abs(re) > max_magnitude) return;
    if (std::abs(im) / (1.0 + std::abs(re)) > real_tol) return;
    out.push_back(re);
  };

  // Primary path -- reduce the matrix-polynomial eigenproblem to a STANDARD one.
  // The monic Frobenius companion of a matrix polynomial with invertible leading
  // coefficient is a standard 80x80 matrix whose eigenvalues are the roots of
  // det S(u)=0; Eigen's EigenSolver (RealSchur: balancing + Hessenberg + Francis
  // double-shift QR) solves it far more robustly than Eigen's RealQZ, which fails
  // to converge on ~38% of these pencils. The catch is that forming Sd^-1
  // amplifies error when the leading coefficient Sd=S[d] is ill-conditioned (the
  // extreme-config case where roots spread toward infinity). So pick the STABLE
  // orientation: if S[0] (constant coeff) is better-conditioned than S[d], solve
  // the reversed polynomial Q(y) = y^d S(1/y) (whose leading coeff is S[0]) and
  // invert the roots u = 1/y. RealQZ remains the fallback when the chosen leading
  // coefficient is singular (a genuine degree drop / infinite eigenvalues).
  // NB: on well-conditioned HP inputs (the locked-7R sub-chains HP is dispatched
  // for) this recovers every real root -- verified oracle-complete on clean Tv6
  // and Tv4 DH. On pathologically ill-conditioned general-6R DH (which HP is
  // never dispatched for) it can recover fewer roots than LAPACK's generalized
  // eigensolver; see #544 (monic-vs-dggev completeness gap).
  auto rcond = [](const Mat10& m) {
    Eigen::JacobiSVD<Mat10> svd(m);
    const double smax = svd.singularValues()(0), smin = svd.singularValues()(n - 1);
    return smax > 0.0 ? smin / smax : 0.0;  // reciprocal condition (1 well, 0 singular)
  };
  const bool reversed = rcond(s_eq[0]) > rcond(s_eq[d]);
  std::array<Mat10, 9> coeff;
  for (int k = 0; k <= d; ++k) coeff[k] = reversed ? s_eq[d - k] : s_eq[k];

  bool solved = false;
  Eigen::FullPivLU<Mat10> lu(coeff[d]);
  if (lu.isInvertible()) {
    const Mat10 lead_inv = lu.inverse();
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nd, nd);
    for (int k = 0; k < d - 1; ++k) C.block<n, n>(k * n, (k + 1) * n).setIdentity();
    for (int k = 0; k < d; ++k) C.block<n, n>((d - 1) * n, k * n) = -(lead_inv * coeff[k]);
    Eigen::EigenSolver<Eigen::MatrixXd> es;
    es.compute(C, /*computeEigenvectors=*/false);
    if (es.info() == Eigen::Success) {
      const auto eig = es.eigenvalues();
      for (int i = 0; i < eig.size(); ++i) {
        if (reversed) {
          const std::complex<double> y = eig(i);
          if (std::abs(y) > 1e-300) keep(std::complex<double>(1.0, 0.0) / y);
        } else {
          keep(eig(i));
        }
      }
      solved = true;
    }
  }
  if (!solved) {
    // Fallback: the generalized pencil via RealQZ. Read the Schur factors S, T
    // directly (never the asserting eigenvalues()/alphas() accessors, which abort
    // on a non-converged QZ) and extract eigenvalues from the 1x1/2x2 blocks: a
    // 1x1 gives real S(i,i)/T(i,i); a 2x2 gives the roots of det(S_b - x T_b)=0.
    Eigen::MatrixXd A, B;
    build_frobenius(s_eq, A, B);
    Eigen::RealQZ<Eigen::MatrixXd> qz(nd);
    qz.setMaxIterations(400 * nd);
    qz.compute(A, B, /*computeQZ=*/false);
    const Eigen::MatrixXd& S = qz.matrixS();
    const Eigen::MatrixXd& T = qz.matrixT();
    for (int i = 0; i < nd;) {
      if (i + 1 < nd && S(i + 1, i) != 0.0) {
        const double s00 = S(i, i), s01 = S(i, i + 1), s10 = S(i + 1, i), s11 = S(i + 1, i + 1);
        const double t00 = T(i, i), t01 = T(i, i + 1), t11 = T(i + 1, i + 1);
        const double qa = t00 * t11;
        const double qb = -(s00 * t11 + s11 * t00 - s10 * t01);
        const double qc = s00 * s11 - s10 * s01;
        if (qa == 0.0) {
          if (qb != 0.0) keep(std::complex<double>(-qc / qb, 0.0));
        } else {
          const double disc = qb * qb - 4.0 * qa * qc;
          if (disc >= 0.0) {
            const double sq = std::sqrt(disc);
            keep(std::complex<double>((-qb + sq) / (2.0 * qa), 0.0));
            keep(std::complex<double>((-qb - sq) / (2.0 * qa), 0.0));
          } else {
            const double sq = std::sqrt(-disc);
            keep(std::complex<double>(-qb / (2.0 * qa), sq / (2.0 * qa)));
            keep(std::complex<double>(-qb / (2.0 * qa), -sq / (2.0 * qa)));
          }
        }
        i += 2;
      } else {
        const double beta = T(i, i);
        if (beta != 0.0) keep(std::complex<double>(S(i, i) / beta, 0.0));
        i += 1;
      }
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// =============================================================================
// (u,w) refinement (#539): each pencil u-root is seeded with a w and polished by
// a 2x2 Newton on [f(u,w); g(u,w)] = 0. Mirrors _eliminate._initial_w_for +
// _refine_uw_inline + eliminate_uw_pairs.
// =============================================================================

inline constexpr int kHpNewtonMaxIter = 5;
inline constexpr double kHpNewtonResidueTol = 1e-12;
inline constexpr double kHpClusterTol = 1e-7;

// Evaluate a bivariate-polynomial block p (rows = u-power, cols = w-power) and
// its partials at (u,w); `scale` is the abs-coefficient polyval floored at the
// block's max |coeff| (the residue normaliser, matching _refine_uw_inline).
template <int R, int C>
inline void eval_poly2d(const Eigen::Matrix<double, R, C>& p, double u, double w, double& val,
                        double& du, double& dw, double& scale) {
  double up[R], wp[C], uap[R], wap[C];
  up[0] = wp[0] = uap[0] = wap[0] = 1.0;
  for (int i = 1; i < R; ++i) {
    up[i] = up[i - 1] * u;
    uap[i] = uap[i - 1] * std::abs(u);
  }
  for (int j = 1; j < C; ++j) {
    wp[j] = wp[j - 1] * w;
    wap[j] = wap[j - 1] * std::abs(w);
  }
  val = du = dw = 0.0;
  double sc = 0.0, maxc = 0.0;
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) {
      const double c = p(i, j);
      val += c * up[i] * wp[j];
      if (i >= 1) du += i * c * up[i - 1] * wp[j];
      if (j >= 1) dw += j * c * up[i] * wp[j - 1];
      sc += std::abs(c) * uap[i] * wap[j];
      maxc = std::max(maxc, std::abs(c));
    }
  scale = std::max(sc, maxc);
}

// Seed w at u=u0: cross-validate the real roots of f(u0,.) and g(u0,.), take the
// closest f/g pair's midpoint (_eliminate._initial_w_for). nullopt if neither has
// a real root.
inline std::optional<double> initial_w_for(const Mat9x7& f, const Mat6x5& g, double u0) {
  Eigen::Matrix<double, 1, 7> fu = Eigen::Matrix<double, 1, 7>::Zero();
  Eigen::Matrix<double, 1, 5> gu = Eigen::Matrix<double, 1, 5>::Zero();
  double up = 1.0;
  for (int i = 0; i < 9; ++i) {
    fu += up * f.row(i);
    if (i < 6) gu += up * g.row(i);
    up *= u0;
  }
  if (fu.cwiseAbs().maxCoeff() == 0.0 || gu.cwiseAbs().maxCoeff() == 0.0) return std::nullopt;
  // np_roots takes highest-degree-first; our row is lowest-first -> reverse.
  auto real_roots = [](auto row, int n) {
    std::vector<double> coeffs(n);
    for (int k = 0; k < n; ++k) coeffs[k] = row(n - 1 - k);
    std::vector<double> reals;
    for (const auto& r : np_roots(coeffs))
      if (std::abs(r.imag()) <= 1e-6 * (1.0 + std::abs(r.real()))) reals.push_back(r.real());
    return reals;
  };
  const std::vector<double> rf = real_roots(fu, 7), rg = real_roots(gu, 5);
  if (rf.empty() || rg.empty()) return std::nullopt;
  double best_w = 0.0, best_gap = std::numeric_limits<double>::infinity();
  for (double wf : rf)
    for (double wg : rg)
      if (std::abs(wf - wg) < best_gap) {
        best_gap = std::abs(wf - wg);
        best_w = 0.5 * (wf + wg);
      }
  return best_w;
}

// 2x2 Newton on [f;g]=0 with monotone-best tracking (_eliminate._refine_uw_inline).
// Returns (u, w, best_residue).
inline std::array<double, 3> refine_uw_inline(const Mat9x7& f, const Mat6x5& g, double u0,
                                              double w0) {
  double fv, gv, fdu, fdw, gdu, gdw, sf, sg;
  double u = u0, w = w0;
  eval_poly2d(f, u, w, fv, fdu, fdw, sf);
  eval_poly2d(g, u, w, gv, gdu, gdw, sg);
  double best_u = u, best_w = w;
  double best_res = std::max(std::abs(fv) / std::max(sf, 1e-300), std::abs(gv) / std::max(sg, 1e-300));
  for (int it = 0; it < kHpNewtonMaxIter; ++it) {
    if (best_res < kHpNewtonResidueTol) break;
    eval_poly2d(f, u, w, fv, fdu, fdw, sf);
    eval_poly2d(g, u, w, gv, gdu, gdw, sg);
    const double det = fdu * gdw - fdw * gdu;
    if (det == 0.0 || !std::isfinite(det)) break;
    const double inv = 1.0 / det;
    const double d_u = -inv * (gdw * fv - fdw * gv);
    const double d_w = -inv * (-gdu * fv + fdu * gv);
    if (!std::isfinite(d_u) || !std::isfinite(d_w)) break;
    u += d_u;
    w += d_w;
    double fn, gn, s1, s2, s3, s4, sfn, sgn;
    eval_poly2d(f, u, w, fn, s1, s2, sfn);
    eval_poly2d(g, u, w, gn, s3, s4, sgn);
    const double res =
        std::max(std::abs(fn) / std::max(sfn, 1e-300), std::abs(gn) / std::max(sgn, 1e-300));
    if (res < best_res) {
      best_u = u;
      best_w = w;
      best_res = res;
    }
  }
  return {best_u, best_w, best_res};
}

// Full refinement pipeline: for each drop index, f/g -> pencil roots -> seed w ->
// Newton, collect accepted (u,w), then 2-D cluster-merge (_eliminate.eliminate_
// uw_pairs). drop_indices {7,4,0} cover both Tv6 and Tv4 right paths.
// accept_residue_tol loosens the keep threshold (HP general_6r polishes downstream
// in 6-D via lm_refine, so multi-root candidates that Newton can't push to 1e-12
// are still valid IK).
inline std::vector<std::array<double, 2>> eliminate_uw_pairs(
    const HpConsts& hp, const Vec8& sigma_E, const std::vector<int>& drop_indices = {7, 4, 0},
    double accept_residue_tol = 1e-3) {
  const std::array<Mat4x8, 2> t_w = apply_sigma_e(hp.t_w_pre, sigma_E);
  std::vector<std::array<double, 2>> refined;
  for (int di : drop_indices) {
    Mat9x7 f;
    Mat6x5 g;
    compute_fg_with_tw(hp.t_u, t_w, di, f, g);
    for (double u0 : solve_pencil_eigenvalues(f, g)) {
      const std::optional<double> w0 = initial_w_for(f, g, u0);
      if (!w0) continue;
      const std::array<double, 3> r = refine_uw_inline(f, g, u0, *w0);
      if (r[2] < accept_residue_tol) refined.push_back({r[0], r[1]});
    }
  }
  // 2-D cluster-merge: sort, then greedily fold points within kHpClusterTol
  // (scaled) into the running cluster centroid.
  std::sort(refined.begin(), refined.end());
  std::vector<std::array<double, 2>> out;
  for (const auto& uw : refined) {
    bool merged = false;
    for (auto& ref : out) {
      const double scale = std::max({1.0, std::abs(ref[0]), std::abs(ref[1])});
      const double d2 = (ref[0] - uw[0]) * (ref[0] - uw[0]) + (ref[1] - uw[1]) * (ref[1] - uw[1]);
      if (d2 <= (kHpClusterTol * scale) * (kHpClusterTol * scale)) {
        ref[0] = 0.5 * (ref[0] + uw[0]);
        ref[1] = 0.5 * (ref[1] + uw[1]);
        merged = true;
        break;
      }
    }
    if (!merged) out.push_back(uw);
  }
  return out;
}

// =============================================================================
// Back-substitution (#539): from a refined (u,w) to the joint tan-half-angles
// (v_1..v_6). Mirrors _back_substitute. Tv1 left + Tv4/Tv6 right; Tv2 left (the
// case-keyed hyperplanes) is not yet native -- rizon/kassow reach Tv1 via the
// build-time perturbation, so parametric_var is always 0 here.
// =============================================================================

// Elementary projective Study DQs (_back_substitute._sigma_*).
inline Vec8 sigma_z(double v) { return (Vec8() << 1, 0, 0, v, 0, 0, 0, 0).finished(); }
inline Vec8 sigma_tz(double d) { return (Vec8() << 1, 0, 0, 0, 0, 0, 0, 0.5 * d).finished(); }
inline Vec8 sigma_tx(double a) { return (Vec8() << 1, 0, 0, 0, 0, 0.5 * a, 0, 0).finished(); }
inline Vec8 sigma_rx(double ls) { return (Vec8() << 1, ls, 0, 0, 0, 0, 0, 0).finished(); }

// R_z(v) T_z(d) T_x(a) R_x(ls) as a projective DQ (_sigma_joint_full).
inline Vec8 sigma_joint_full(double v, double a, double ls, double d) {
  using study::dq_mul;
  return dq_mul(sigma_z(v), dq_mul(sigma_tz(d), dq_mul(sigma_tx(a), sigma_rx(ls))));
}

// Cramer cofactor 8-vec P at a single (u,w) (_back_substitute._cramer_P_at):
// build the 8x8, drop row drop_idx, P = [det(A), x_1 det, ..] with x = A^-1(-col0).
inline Vec8 cramer_P_at(const std::array<Mat4x8, 2>& t_u, const std::array<Mat4x8, 2>& t_w,
                        double u, double w, int drop_idx) {
  Mat8 full;
  full.block<4, 8>(0, 0) = t_u[0] + u * t_u[1];
  full.block<4, 8>(4, 0) = t_w[0] + w * t_w[1];
  Eigen::Matrix<double, 7, 8> m7;
  int r = 0;
  for (int k = 0; k < 8; ++k)
    if (k != drop_idx) m7.row(r++) = full.row(k);
  const Eigen::Matrix<double, 7, 7> A = m7.block<7, 7>(0, 1);
  const Eigen::Matrix<double, 7, 1> x = A.lu().solve(-m7.block<7, 1>(0, 0));
  const double det = A.determinant();
  Vec8 P;
  P[0] = det;
  for (int i = 0; i < 7; ++i) P[i + 1] = x[i] * det;
  return P;
}

// Recover (v_a, v_b) from a 2R-chain Study DQ target by a closed-form ZXZ-like
// rotation decomposition (_back_substitute._solve_2r_chain). d_a/d_b are
// translation, unused by the rotation extraction. Single (v_a, v_b) solution.
inline std::pair<double, double> solve_2r_chain(const Vec8& sigma_target, double a_a, double ls_a,
                                                double a_b, double ls_b) {
  (void)a_a;
  (void)a_b;
  const Eigen::Matrix3d R = study::rot_from_dq(sigma_target);

  const double den_b = 1.0 + ls_b * ls_b;
  const double sa_b = 2.0 * ls_b / den_b, ca_b = (1.0 - ls_b * ls_b) / den_b;
  Eigen::Matrix3d rx_neg_b;
  rx_neg_b << 1, 0, 0, 0, ca_b, sa_b, 0, -sa_b, ca_b;
  const Eigen::Matrix3d rp = R * rx_neg_b;

  const double den_a = 1.0 + ls_a * ls_a;
  const double sa_a = 2.0 * ls_a / den_a, ca_a = (1.0 - ls_a * ls_a) / den_a;
  const double rxz = rp(0, 2), ryz = rp(1, 2);

  double v_a;
  Eigen::Matrix3d r_double;
  if (std::abs(sa_a) < 1e-12) {
    v_a = 0.0;
    r_double = rp;
  } else {
    const double sgn = sa_a > 0.0 ? 1.0 : -1.0;
    const double th_a = std::atan2(rxz * sgn, -ryz * sgn);
    v_a = std::tan(0.5 * th_a);
    const double c = std::cos(th_a), s = std::sin(th_a);
    Eigen::Matrix3d rz_neg;
    rz_neg << c, s, 0, -s, c, 0, 0, 0, 1;
    Eigen::Matrix3d rx_neg_a;
    rx_neg_a << 1, 0, 0, 0, ca_a, sa_a, 0, -sa_a, ca_a;
    r_double = rx_neg_a * rz_neg * rp;
  }
  const double th_b = std::atan2(r_double(1, 0), r_double(0, 0));
  return {v_a, std::tan(0.5 * th_b)};
}

using JointTuple = std::array<double, 6>;  // (v_1..v_6) tan-half-angles

// Full (v_1..v_6) for one (u,w) (_back_substitute.back_substitute_one), Tv1 left.
inline JointTuple back_substitute_one(const HpConsts& hp, const Vec8& sigma_E, double u, double w) {
  using study::dq_conj;
  using study::dq_mul;
  const std::array<Mat4x8, 2> t_w = apply_sigma_e(hp.t_w_pre, sigma_E);
  const Vec8 P = cramer_P_at(hp.t_u, t_w, u, w, hp.drop_idx);

  double v_4, v_5, v_6;
  if (hp.right_parametric_var == 1) {  // Tv4: w = v_4
    v_4 = w;
    const Vec8 s4 = sigma_joint_full(v_4, hp.a[4], hp.l[4], hp.d[4]);
    const Vec8 rhs = dq_mul(dq_conj(s4), dq_mul(dq_conj(P), sigma_E));
    const auto [a, b] = solve_2r_chain(rhs, hp.a[5], hp.l[5], 0.0, 0.0);
    v_5 = a;
    v_6 = b;
  } else {  // Tv6: w = v_6
    v_6 = w;
    const Vec8 rhs = dq_mul(dq_conj(P), dq_mul(sigma_E, dq_conj(sigma_z(v_6))));
    const auto [a, b] = solve_2r_chain(rhs, hp.a[4], hp.l[4], hp.a[5], hp.l[5]);
    v_4 = a;
    v_5 = b;
  }

  const double v_1 = u;  // Tv1 left
  const Vec8 s1 = sigma_joint_full(v_1, hp.a[1], hp.l[1], 0.0);
  const Vec8 left = dq_mul(dq_conj(s1), P);
  const auto [v_2, v_3] = solve_2r_chain(left, hp.a[2], hp.l[2], hp.a[3], hp.l[3]);
  return {v_1, v_2, v_3, v_4, v_5, v_6};
}

// Full HP IK kernel: refined (u,w) pairs -> back-substitute each -> the
// (v_1..v_6) tan-half-angle candidates (_back_substitute.solve_ik). The
// projective Study-DQ FK-closure filter is intentionally omitted: HP general_6r
// runs the FK check downstream in POE space (verify_candidates), matching the
// Python solve_ik(fk_tol=0.5) fast path.
inline std::vector<JointTuple> solve_ik(const HpConsts& hp, const Vec8& sigma_E) {
  std::vector<JointTuple> out;
  for (const auto& uw : eliminate_uw_pairs(hp, sigma_E))
    out.push_back(back_substitute_one(hp, sigma_E, uw[0], uw[1]));
  return out;
}

// Wrap-to-pi max-joint dedup for 6-DOF, keeping the lower-FK duplicate.
inline std::vector<Solution<6>> dedup6(const std::vector<Solution<6>>& cands, double atol) {
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
      if (worst < atol) dup = static_cast<int>(j);
    }
    if (dup < 0)
      out.push_back(cand);
    else if (cand.fk_residual < out[dup].fk_residual)
      out[dup] = cand;
  }
  return out;
}

}  // namespace hp_detail

// FK-closure accept gate + dedup tolerance for the HP artifact. Algebraic /
// back-sub seeds that already close under kHpFkAtol are accepted as-is; the rest
// (perturbed O(epsilon) seeds, multiplicity-k roots) are lm_refined in POE space.
inline constexpr double kHpFkAtol = 1e-7;
inline constexpr double kHpDedupAtol = 1e-3;

// Raw HP 6R candidates for one target: bridge the POE pose into the HP frame (all
// transforms baked), run the numeric kernel, convert tan-half-angles to POE q,
// FK-verify + lm_refine (perturbed O(epsilon) / multi-root seeds), dedup. No
// limit-gate / rescue / seed finalize -- that is the artifact-solve wrapper's job.
// Shared by hp_artifact_solve (below) and the jointlock HP sweep
// (jointlock_hp_artifact_solve), which needs the raw per-sub-chain candidates to
// pad to 7-DOF and re-verify against the 7R target.
inline std::vector<Solution<6>> hp_core(const JointConsts<6>& c, const HpConsts& hp, const Pose& tp,
                                        int refinement_max_iters) {
  const Pose t_dh = hp.t_pre_inv * tp * hp.t_post_inv;
  const Pose t_hp = hp.t_z_neg_d1 * t_dh * hp.t_joint6_offset_inv;
  const Vec8 sigma_E = study::dq_from_se3(t_hp);
  std::vector<Solution<6>> sols;
  for (const auto& v : hp_detail::solve_ik(hp, sigma_E)) {
    std::array<double, 6> q;
    for (int i = 0; i < 6; ++i) q[i] = 2.0 * std::atan(v[i]) - hp.theta_offset[i];
    const double fk_err = (fk<6>(c, q) - tp).norm();
    if (fk_err <= kHpFkAtol) {
      sols.push_back(Solution<6>{q, fk_err, Refinement::None});
    } else if (fk_err < 0.1) {  // seed near the basin -> polish (perturbed / multi-root)
      auto r = lm_refine<6>(c, q, tp, kHpFkAtol, refinement_max_iters);
      if (r && r->second <= kHpFkAtol)
        sols.push_back(Solution<6>{r->first, r->second, Refinement::Lm});
    }
  }
  return hp_detail::dedup6(sols, kHpDedupAtol);
}

// Full self-contained HP universal-6R solve (#539). hp_core + the limit-gate ->
// rescue -> seed finalize skeleton of general_6r_artifact_solve. Mirrors
// husty_pfurner.general_6r.solve + verify_candidates.
inline std::vector<Solution<6>> hp_artifact_solve(const JointConsts<6>& c, const HpConsts& hp,
                                                  const JointLimits<6>& lim, const Pose& T,
                                                  const ArtifactParams<6>& p) {
  const auto core = [&](const Pose& tp) -> std::vector<Solution<6>> {
    return hp_core(c, hp, tp, p.refinement_max_iters);
  };

  ArtifactParams<6> p_limits;
  p_limits.respect_limits = p.respect_limits;
  p_limits.refinement_max_iters = p.refinement_max_iters;
  std::vector<Solution<6>> in_limits = finalize_solutions<6>(core(T), c, lim, p_limits);
  if (in_limits.empty() && p.allow_rescue && T.block<3, 1>(0, 3).norm() <= reach_radius(c)) {
    RescueParams rp;
    rp.dedup_atol = kHpDedupAtol;
    in_limits = finalize_solutions<6>(rescue_via_T_perturbation<6>(core, c, T, rp), c, lim,
                                      p_limits);
  }
  ArtifactParams<6> p_seed = p;
  p_seed.respect_limits = false;
  return finalize_solutions<6>(std::move(in_limits), c, lim, p_seed);
}

}  // namespace ssik
