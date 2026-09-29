// The general_6r polish never takes an accepted candidate across a joint limit
// (#644): the native mirror of tests/test_polish_limits.py, which explains the
// construction and the oracle. A regular root of the FANUC CRX-20iA/L is placed
// 5e-9 rad past joint 3's lower limit (+-3 pi / 2), directly or through its
// +2 pi winding; the accepted candidate sits on the limit. The polish must keep
// every in-limit winding of the candidate, returning it unchanged when the
// polished point would lose one; a root inside the band is still polished.
#include <array>
#include <cmath>
#include <cstdio>
#include <set>

#include "fanuc_crx20ial_ik.hpp"
#include "ssik_cpp/polish.hpp"

namespace {

constexpr int kDof = 6;
constexpr int kJ = 2;  // joint 3

std::set<int> in_band_windings(double x, double lo, double hi) {
  constexpr double kTwoPi = 2.0 * M_PI;
  std::set<int> ks;
  for (int k = static_cast<int>(std::floor((lo - x) / kTwoPi)) - 1;
       k <= static_cast<int>(std::ceil((hi - x) / kTwoPi)) + 1; ++k) {
    const double v = x + kTwoPi * k;
    if (v >= lo - ssik::kLimitBand && v <= hi + ssik::kLimitBand) ks.insert(k);
  }
  return ks;
}

// Returns the number of failed checks.
int run_case(const char* name, double root_offset, int winding, bool expect_polished) {
  namespace arm = ssik::fanuc_crx20ial_ik;
  const auto c = arm::consts();
  const auto lim = arm::limits();
  const double lo = lim.lo[kJ];
  std::array<double, kDof> q_star = {1.5310050577906944, 2.169188913994496,  lo,
                                     -1.363346478405587, 2.7887203364781827, 3.6332152303383065};
  q_star[kJ] = lo + root_offset + 2.0 * M_PI * winding;
  std::array<double, kDof> q0 = q_star;
  q0[kJ] = lo + 2.0 * M_PI * winding;
  const ssik::Pose t = ssik::fk<kDof>(c, q_star);
  double r = (ssik::fk<kDof>(c, q0) - t).norm();
  const double r0 = r;

  std::array<double, kDof> q = q0;
  const bool polished = ssik::polish_accepted<kDof>(
      [&](const std::array<double, kDof>& x) { return ssik::fk<kDof>(c, x); },
      [&](const std::array<double, kDof>& x) { return ssik::spatial_jacobian<kDof>(c, x); }, t,
      lim, q, r);

  int fails = 0;
  if (!(r0 > 1e-12 && r0 <= 1e-5)) {
    std::printf("  %s: candidate residual %.3e is not an accepted, improvable one\n", name, r0);
    ++fails;
  }
  for (int i = 0; i < kDof; ++i) {
    const auto before = in_band_windings(q0[i], lim.lo[i], lim.hi[i]);
    const auto after = in_band_windings(q[i], lim.lo[i], lim.hi[i]);
    for (int k : before)
      if (!after.count(k)) {
        std::printf("  %s: joint %d lost its in-limit winding k=%d\n", name, i, k);
        ++fails;
      }
  }
  if (polished != expect_polished) {
    std::printf("  %s: polished=%d, expected %d\n", name, polished, expect_polished);
    ++fails;
  }
  if (polished && !(r <= ssik::kPolishTarget)) {
    std::printf("  %s: polished residual %.3e above target\n", name, r);
    ++fails;
  }
  if (!polished && (q != q0 || r != r0)) {
    std::printf("  %s: a rejected polish changed the candidate\n", name);
    ++fails;
  }
  return fails;
}

}  // namespace

int main() {
  int fails = 0;
  fails += run_case("past-limit", -5e-9, 0, false);
  fails += run_case("lost-winding", -5e-9, 1, false);
  fails += run_case("inside-control", +5e-9, 0, true);
  if (fails != 0) {
    std::printf("FAIL (%d)\n", fails);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
