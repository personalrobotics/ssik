// Backend-wide parallelism primitive for the native artifacts (#546).
//
// The native solvers have several embarrassingly-parallel loops with no shared
// state: the jointlock lock-sweep (N independent 6R sub-chain solves), the
// T-perturbation rescue (N independent perturbed solves), and batch solving (one
// independent solve per target pose). All of them fan out through this one
// `parallel_for`, so the threading policy lives in a single place and stays
// dependency-free (std::thread only -- no OpenMP flag, no TBB, no std::execution
// backend), which keeps every shipped `<arm>.hpp` self-contained.
//
// Parallelism is a runtime backend property, not per-solve geometry, so the
// thread cap is a process-global (set once at startup) rather than a field on
// every ArtifactParams. Default 0 = auto (hardware_concurrency); set 1 to force
// serial (e.g. inside an already-threaded caller, or for deterministic tests).
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace ssik {

// Process-global worker-thread cap for parallel_for. 0 = auto
// (std::thread::hardware_concurrency); 1 = force serial; N = at most N workers.
inline std::atomic<unsigned>& parallel_thread_cap() {
  static std::atomic<unsigned> cap{0};
  return cap;
}

// Set the backend-wide worker cap (0 = auto, 1 = serial). Intended to be called
// once at startup; a concurrent solve may still observe the old value.
inline void set_max_threads(unsigned n) {
  parallel_thread_cap().store(n, std::memory_order_relaxed);
}

// Effective worker count for a loop of `n` iterations: min(cap-or-hardware, n),
// floored at 1.
inline unsigned parallel_worker_count(std::size_t n) {
  const unsigned cap = parallel_thread_cap().load(std::memory_order_relaxed);
  unsigned hw = cap != 0 ? cap : std::thread::hardware_concurrency();
  if (hw == 0) hw = 1;
  return static_cast<unsigned>(std::min<std::size_t>(hw, n == 0 ? 1 : n));
}

// Per-thread flag: are we already inside a parallel_for? Nested parallel loops
// (e.g. the rescue fans out perturbations, and each perturbation runs the
// jointlock sweep, which itself fans out) would oversubscribe the cores, so only
// the OUTERMOST parallel_for threads -- inner ones detect the flag and run serial.
inline bool& parallel_in_region() {
  thread_local bool in = false;
  return in;
}

// Run fn(i) for i in [0, n). Iterations MUST be independent: fn must only write
// to storage private to its own i (e.g. a pre-sized per-index output slot), so
// there is no data race and no ordering dependence. Runs serially when n < min_n
// or only one worker is available; otherwise partitions [0, n) into contiguous
// chunks, one per worker, with the calling thread taking chunk 0 (so `workers`
// total threads of execution spawn `workers - 1` std::threads). fn must not throw
// (an escaped exception in a worker terminates via std::thread's contract).
//
// min_n gates out the fixed thread-spawn cost (~tens of us/worker): keep serial
// unless the loop is either long or its bodies are individually expensive. The
// coarse native loops (16-sample sweep, 16-perturbation rescue, >=~0.1ms/pose
// batch) clear this easily; a single fast 6R/SRS solve does not fan out here.
template <class Fn>
void parallel_for(std::size_t n, Fn&& fn, std::size_t min_n = 2) {
  if (n == 0) return;
  const unsigned workers = (n < min_n || parallel_in_region()) ? 1u : parallel_worker_count(n);
  if (workers <= 1) {
    for (std::size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  const std::size_t chunk = (n + workers - 1) / workers;
  // Each thread of execution marks itself in-region for the duration so a nested
  // parallel_for on it runs serial. Spawned workers start with a fresh (false)
  // thread_local and die after join; the calling thread saves/restores its flag.
  const auto run_range = [&fn](std::size_t lo, std::size_t hi) {
    parallel_in_region() = true;
    for (std::size_t i = lo; i < hi; ++i) fn(i);
  };
  std::vector<std::thread> pool;
  pool.reserve(workers - 1);
  for (unsigned w = 1; w < workers; ++w) {
    const std::size_t lo = std::min(n, static_cast<std::size_t>(w) * chunk);
    const std::size_t hi = std::min(n, lo + chunk);
    if (lo < hi) pool.emplace_back(run_range, lo, hi);
  }
  const bool outer = parallel_in_region();
  run_range(0, std::min(n, chunk));  // calling thread runs chunk 0
  parallel_in_region() = outer;      // restore (spawned workers' flags die with them)
  for (auto& t : pool) t.join();
}

}  // namespace ssik
