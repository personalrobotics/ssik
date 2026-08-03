// Newton-polish conformance (#496): the native lm_refine must reproduce the
// Python reference's outcome on perturbed seeds -- same converged/diverged
// decision, and where it converges, the same joint vector at machine precision.
#include <array>
#include <cmath>
#include <cstdio>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/newton.hpp"
#include "ur5_ik.hpp"
#include "ur5_ik_newton_parity.hpp"

int main() {
  const auto c = ssik::ur5_ik::consts();
  const auto& cases = ssik::ur5_ik::newton_parity_cases();
  constexpr int DOF = ssik::ur5_ik::DOF;
  constexpr double kQMatch = 1e-6;  // both converge to the same q* to well under this

  int decision_mismatch = 0;
  double worst_q = 0.0;
  double worst_resid = 0.0;
  int converged = 0;

  for (std::size_t ci = 0; ci < cases.size(); ++ci) {
    const auto& tc = cases[ci];
    ssik::Pose T;
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col) T(r, col) = tc.target[r * 4 + col];

    const auto refined =
        ssik::lm_refine<DOF>(c, tc.q_seed, T, ssik::ur5_ik::NEWTON_FK_ATOL,
                             ssik::ur5_ik::NEWTON_MAX_ITERS);

    const bool native_conv = refined.has_value();
    if (native_conv != tc.converged) {
      ++decision_mismatch;
      if (decision_mismatch <= 5)
        std::printf("  case %zu: native converged=%d, python=%d\n", ci, native_conv, tc.converged);
      continue;
    }
    if (!native_conv) continue;
    ++converged;

    for (int i = 0; i < DOF; ++i)
      worst_q = std::max(worst_q, std::abs(refined->first[i] - tc.q_refined[i]));
    // Independent check: the native refined q actually closes FK.
    const double resid = (ssik::fk<DOF>(c, refined->first) - T).norm();
    worst_resid = std::max(worst_resid, resid);
  }

  std::printf("UR5 Newton conformance: %zu cases, %d converged (both), %d decision mismatches\n",
              cases.size(), converged, decision_mismatch);
  std::printf("  worst |q_native - q_python| = %.3e (match gate %.0e)\n", worst_q, kQMatch);
  std::printf("  worst native FK residual = %.3e (gate %.0e)\n", worst_resid,
              ssik::ur5_ik::NEWTON_FK_ATOL);

  if (decision_mismatch == 0 && worst_q < kQMatch && worst_resid < ssik::ur5_ik::NEWTON_FK_ATOL) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL\n");
  return 1;
}
