// Unit test for the parallel_for primitive (#546): every index runs exactly once
// with no data race, results are independent of worker count, and the serial
// fallback (min_n gate, cap=1) matches.
#include <atomic>
#include <cstdio>
#include <numeric>
#include <vector>

#include "ssik_cpp/parallel.hpp"

static int failures = 0;
static void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

int main() {
  // 1. Full coverage + correct per-index writes (disjoint slots => no race).
  for (unsigned cap : {0u, 1u, 2u, 4u, 8u}) {
    ssik::set_max_threads(cap);
    const std::size_t n = 10000;
    std::vector<long> out(n, -1);
    ssik::parallel_for(n, [&](std::size_t i) { out[i] = static_cast<long>(i) * 2; });
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok = ok && (out[i] == static_cast<long>(i) * 2);
    check(ok, "every index written exactly once with the right value");
  }

  // 2. Exactly-once: an atomic counter must equal n (no double-run, no skip).
  ssik::set_max_threads(8);
  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{97},
                        std::size_t{1000}}) {
    std::atomic<std::size_t> runs{0};
    ssik::parallel_for(n, [&](std::size_t) { runs.fetch_add(1, std::memory_order_relaxed); });
    check(runs.load() == n, "iteration count equals n");
  }

  // 3. min_n gate forces serial (single-threaded) below the threshold.
  ssik::set_max_threads(8);
  {
    std::vector<std::size_t> order;
    ssik::parallel_for(
        5, [&](std::size_t i) { order.push_back(i); }, /*min_n=*/16);
    std::vector<std::size_t> expect(5);
    std::iota(expect.begin(), expect.end(), 0);
    check(order == expect, "min_n gate runs serially in-order");
  }

  // 4. Result independent of worker count (sum over a parallel reduction pattern).
  {
    const std::size_t n = 5000;
    long ref = 0;
    for (unsigned cap : {1u, 3u, 8u}) {
      ssik::set_max_threads(cap);
      std::vector<long> part(n);
      ssik::parallel_for(n, [&](std::size_t i) { part[i] = static_cast<long>(i * i % 7); });
      const long s = std::accumulate(part.begin(), part.end(), 0L);
      if (cap == 1u) ref = s;
      check(s == ref, "reduction result independent of worker count");
    }
  }

  // 5. Nested parallel_for: inner loops run serial (region flag set) but still
  //    cover every (outer, inner) pair exactly once.
  ssik::set_max_threads(8);
  {
    const std::size_t no = 12, ni = 50;
    std::vector<std::atomic<int>> hit(no * ni);
    for (auto& h : hit) h.store(0);
    std::atomic<int> inner_saw_region{0};
    ssik::parallel_for(no, [&](std::size_t o) {
      ssik::parallel_for(ni, [&](std::size_t i) {
        if (ssik::parallel_in_region()) inner_saw_region.fetch_add(1, std::memory_order_relaxed);
        hit[o * ni + i].fetch_add(1, std::memory_order_relaxed);
      });
    });
    bool once = true;
    for (auto& h : hit) once = once && (h.load() == 1);
    check(once, "nested parallel_for covers every pair exactly once");
    check(inner_saw_region.load() == static_cast<int>(no * ni),
          "inner parallel_for runs inside the region flag (serial nesting)");
  }

  ssik::set_max_threads(0);  // restore auto
  if (failures == 0) {
    std::printf("test_parallel: all checks passed\n");
    return 0;
  }
  std::printf("test_parallel: %d failures\n", failures);
  return 1;
}
