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
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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
inline void compute_fg(const HpConsts& hp, const Vec8& sigma_E, int drop_idx, Mat9x7& f,
                       Mat6x5& g) {
  const std::array<Mat4x8, 2> t_w = apply_sigma_e(hp.t_w_pre, sigma_E);
  const PCoef p = cramer_8vec_via_interp(hp.t_u, t_w, drop_idx);
  f = study_quadric_f(p);
  g = dropped_row_g(hp.t_u, t_w, p, drop_idx);
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
  auto rcond = [](const Mat10& m) {
    Eigen::JacobiSVD<Mat10> svd(m);
    const double smax = svd.singularValues()(0), smin = svd.singularValues()(n - 1);
    return smax > 0.0 ? smin / smax : 0.0;  // reciprocal condition (1 well, 0 singular)
  };
  const bool reversed = rcond(s_eq[0]) > rcond(s_eq[d]);
  // Coefficient list for the chosen orientation, leading = coeff[d].
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
        // Reversed solves for y = 1/u; invert (skip y ~ 0, i.e. u -> infinity).
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

}  // namespace hp_detail
}  // namespace ssik
