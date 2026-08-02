// FK conformance (soundness foundation, #487): the native FK must match the
// Python reference at machine precision on the emitted parity cases.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "ssik_cpp/fk.hpp"
#include "ur5_ik.hpp"
#include "ur5_ik_fk_parity.hpp"

int main() {
  const auto c = ssik::ur5_ik::consts();
  const auto& cases = ssik::ur5_ik::fk_parity_cases();

  double worst = 0.0;
  for (const auto& tc : cases) {
    const ssik::Pose t = ssik::fk<ssik::ur5_ik::DOF>(c, tc.q);
    for (int r = 0; r < 4; ++r) {
      for (int col = 0; col < 4; ++col) {
        worst = std::max(worst, std::abs(t(r, col) - tc.fk[r * 4 + col]));
      }
    }
  }

  std::printf("UR5 native FK vs Python: %zu cases, worst |diff| = %.3e\n", cases.size(), worst);
  if (worst < 1e-12) {
    std::printf("PASS\n");
    return 0;
  }
  std::printf("FAIL (>= 1e-12)\n");
  return 1;
}
