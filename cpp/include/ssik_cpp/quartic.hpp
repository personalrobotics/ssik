// Polynomial roots via the companion-matrix eigenvalues (#486) -- the C++
// equivalent of numpy.roots, matching ssik's solve_quartic_roots (which is
// np.roots, chosen over closed-form Ferrari for robustness on ill-conditioned
// quartics). Trims leading/trailing zeros exactly like numpy so degree
// degradation matches the oracle.
#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>

namespace ssik {

// Complex roots of coeffs[0] x^n + ... + coeffs[n] (decreasing power), matching
// numpy.roots: strip leading zeros (lower the degree), strip trailing zeros
// (roots at 0), companion-matrix eigenvalues for the rest.
inline std::vector<std::complex<double>> np_roots(const std::vector<double>& coeffs) {
  // First and last non-zero indices.
  int lo = 0;
  const int n = static_cast<int>(coeffs.size());
  while (lo < n && coeffs[lo] == 0.0) ++lo;
  int hi = n - 1;
  while (hi >= lo && coeffs[hi] == 0.0) --hi;
  if (lo > hi) return {};  // all zero

  const int trailing = (n - 1) - hi;  // roots at 0
  std::vector<std::complex<double>> roots;
  const int deg = hi - lo;  // polynomial degree after trimming
  if (deg >= 1) {
    // Companion matrix (numpy convention): first row = -c[1..]/c[0], unit
    // subdiagonal. Eigenvalues are the roots.
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(deg, deg);
    const double lead = coeffs[lo];
    for (int j = 0; j < deg; ++j) a(0, j) = -coeffs[lo + 1 + j] / lead;
    for (int j = 0; j < deg - 1; ++j) a(j + 1, j) = 1.0;
    Eigen::EigenSolver<Eigen::MatrixXd> es(a, /*computeEigenvectors=*/false);
    const auto ev = es.eigenvalues();
    for (int j = 0; j < ev.size(); ++j) roots.emplace_back(ev(j));
  }
  for (int j = 0; j < trailing; ++j) roots.emplace_back(0.0, 0.0);
  return roots;
}

}  // namespace ssik
