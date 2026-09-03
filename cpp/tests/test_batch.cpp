// solve_batch equivalence (#546): the parallel batch solve must return, for
// every pose, the same solution set (same count, same solutions in order) as the
// serial per-pose solve(). Covers a jointlock arm (kassow -- nested: batch outer,
// sweep inner serial), a 6R arm (jaco2), and an SRS 7R arm (iiwa14).
//
// Compared with a wrap-to-pi tolerance, NOT bit-exact: on a heterogeneous CPU
// (Apple Silicon P/E cores) libm transcendentals (atan/sqrt/trig) differ by ~1
// ULP between core types, and an lm-refined solution amplifies that seed noise to
// ~1e-13 as work migrates across cores under threading. Both results are equally
// valid (identical fk_residual, far inside the FK tolerance); only lm-refined
// solutions carry it, and the artifact gate's own wrap_close(1e-3) is unaffected.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "iiwa14_ik.hpp"
#include "jaco2_ik.hpp"
#include "kassow_kr810_ik.hpp"

using namespace ssik;

static int failures = 0;

template <int D, class NS_consts, class NS_limits, class NS_solve, class NS_batch>
void check_arm(const char* name, NS_consts consts, NS_limits limits, NS_solve solve,
               NS_batch batch) {
  auto c = consts();
  auto lim = limits();
  std::mt19937 rng(7);
  std::vector<Pose> Ts;
  for (int t = 0; t < 60; ++t) {
    std::array<double, D> q;
    for (int i = 0; i < D; ++i) {
      std::uniform_real_distribution<double> u(lim.lo[i], lim.hi[i]);
      q[i] = u(rng);
    }
    Ts.push_back(fk<D>(c, q));
  }
  const auto b = batch(Ts, ArtifactParams<D>{});
  if (b.size() != Ts.size()) {
    std::printf("FAIL %s: batch size %zu != %zu\n", name, b.size(), Ts.size());
    ++failures;
    return;
  }
  // wrap-to-pi Linf distance; 1e-9 is far above the ~1e-13 heterogeneous-core FP
  // noise but far below any distinct IK branch (dedup is 1e-3).
  const auto wrap_close = [](const std::array<double, D>& a, const std::array<double, D>& b_) {
    double m = 0.0;
    for (int j = 0; j < D; ++j) {
      double d = std::fmod(a[j] - b_[j] + M_PI, 2.0 * M_PI);
      if (d < 0.0) d += 2.0 * M_PI;
      m = std::max(m, std::abs(d - M_PI));
    }
    return m < 1e-9;
  };
  int mism = 0;
  for (std::size_t i = 0; i < Ts.size(); ++i) {
    const auto s = solve(Ts[i], ArtifactParams<D>{});
    bool ok = s.size() == b[i].size();
    for (std::size_t k = 0; ok && k < s.size(); ++k) ok = ok && wrap_close(s[k].q, b[i][k].q);
    if (!ok) ++mism;
  }
  if (mism) {
    std::printf("FAIL %s: %d/%zu poses differ batch vs serial\n", name, mism, Ts.size());
    ++failures;
  } else {
    std::printf("%s: solve_batch == serial solve over %zu poses\n", name, Ts.size());
  }
}

int main() {
  check_arm<6>(
      "jaco2", [] { return jaco2_ik::consts(); }, [] { return jaco2_ik::limits(); },
      [](const Pose& T, const ArtifactParams<6>& p) { return jaco2_ik::solve(T, p); },
      [](const std::vector<Pose>& Ts, const ArtifactParams<6>& p) {
        return jaco2_ik::solve_batch(Ts, p);
      });
  check_arm<7>(
      "iiwa14", [] { return iiwa14_ik::consts(); }, [] { return iiwa14_ik::limits(); },
      [](const Pose& T, const ArtifactParams<7>& p) { return iiwa14_ik::solve(T, p); },
      [](const std::vector<Pose>& Ts, const ArtifactParams<7>& p) {
        return iiwa14_ik::solve_batch(Ts, p);
      });
  check_arm<7>(
      "kassow", [] { return kassow_kr810_ik::consts(); },
      [] { return kassow_kr810_ik::limits(); },
      [](const Pose& T, const ArtifactParams<7>& p) { return kassow_kr810_ik::solve(T, p); },
      [](const std::vector<Pose>& Ts, const ArtifactParams<7>& p) {
        return kassow_kr810_ik::solve_batch(Ts, p);
      });
  if (failures == 0) {
    std::printf("test_batch: all arms match\n");
    return 0;
  }
  return 1;
}
