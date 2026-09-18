// Artifact-layer post-processing (#503), ported bit-for-bit from
// ssik.postprocess.finalize_solutions + the <arm>_ik.solve() wrapper. This is
// the shared tail every shipped Python artifact runs after the core solve:
// limits (wrap-into-range then drop) -> seed (tolerance filter then rank) ->
// truncate. The C++ replica must match the artifact, so every ordering rule,
// inclusive/strict inequality, and wrap convention is reproduced exactly.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <utility>
#include <vector>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/ik_types.hpp"

namespace ssik {

// Per-joint limits alongside JointConsts (JointConsts carries axes/frames/type;
// limits live here since only the artifact layer needs them).
template <int N>
struct JointLimits {
  std::array<double, N> lo{};
  std::array<double, N> hi{};
  std::array<bool, N> present{};  // false => joint has no limits (never rejects)
};

// Seed-ranking metric selector, matching seed_metric="wrap_linf"|"wrap_l2".
enum class SeedMetric { WrapLinf, WrapL2 };

// Full artifact solve() parameters (defaults match codegen.py:507-670).
template <int N>
struct ArtifactParams {
  bool respect_limits = true;
  bool has_seed = false;
  std::array<double, N> q_seed{};
  SeedMetric seed_metric = SeedMetric::WrapLinf;
  bool has_seed_tolerance = false;
  double seed_tolerance = 0.0;
  int max_solutions = -1;  // -1 => None (no cap)
  bool allow_rescue = true;
  int refinement_max_iters = 15;
  // #562 step 2. Defaults FALSE so that params a solver constructs itself --
  // the limit pass, the rescue pass -- never expand: a solve runs finalize
  // several times and the lifts must be produced exactly once, by the call that
  // yields the returned set. The binding sets it from the caller's argument.
  bool enumerate_windings = false;
};

namespace finalize_detail {

// ((x + pi) mod 2pi) - pi, matching ssik.postprocess._wrap_to_pi exactly:
// Python's % (floored) yields [0, 2pi), so the result is [-pi, pi) -- +pi maps
// to -pi. std::fmod is truncated, so we correct negatives to reproduce it.
inline double wrap_to_pi(double x) {
  double m = std::fmod(x + M_PI, 2.0 * M_PI);
  if (m < 0.0) m += 2.0 * M_PI;
  return m - M_PI;
}

// A joint is enumerable only when its limits span strictly more than one turn.
// The tolerance keeps a joint whose span is 2pi to within round-off out of
// enumeration (URDFs write +/-3.14159265359, which is 4e-10 wider than pi):
// such a span admits a second representative only at the exact boundary, which
// is the same physical configuration, not a lift. (postprocess._SPAN_TOL)
inline constexpr double kSpanTol = 1e-9;

// Per-joint seed difference, wrapped modulo 2pi only for continuous joints.
// A finite revolute joint's configuration space is an interval, not a circle:
// it cannot rotate through a limit, so measuring it modulo 2pi understates real
// motion and makes distinct windings tie. (postprocess._seed_deltas)
template <int N>
std::array<double, N> seed_deltas(const std::array<double, N>& q, const JointConsts<N>& consts,
                                  const JointLimits<N>& lim, const std::array<double, N>& seed) {
  std::array<double, N> d{};
  for (int i = 0; i < N; ++i) {
    const bool circular = consts.type[i] == JointType::Revolute && !lim.present[i];
    d[i] = circular ? wrap_to_pi(q[i] - seed[i]) : q[i] - seed[i];
  }
  return d;
}

template <int N>
double aggregate(const std::array<double, N>& deltas, SeedMetric metric) {
  if (metric == SeedMetric::WrapL2) {
    double acc = 0.0;
    for (int i = 0; i < N; ++i) acc += deltas[i] * deltas[i];
    return std::sqrt(acc);
  }
  double m = 0.0;
  for (int i = 0; i < N; ++i) m = std::max(m, std::abs(deltas[i]));
  return m;
}

// The total, deterministic seeded ordering (postprocess._rank_key): aggregate,
// then the per-joint deviations sorted descending (the leximax refinement),
// then the joint vector. Ranking on the aggregate alone leaves ties that are
// neither rare nor harmless once windings are enumerated -- under WrapLinf one
// distant joint fixes the max for every winding of a branch -- and the Python
// and native backends do not generate candidates in the same order, so the
// order must depend only on the solutions.
template <int N>
struct RankKey {
  double agg = 0.0;
  std::array<double, N> desc{};  // |deltas| sorted descending
  std::array<double, N> q{};

  bool operator<(const RankKey& o) const {
    if (agg != o.agg) return agg < o.agg;
    for (int i = 0; i < N; ++i)
      if (desc[i] != o.desc[i]) return desc[i] < o.desc[i];
    for (int i = 0; i < N; ++i)
      if (q[i] != o.q[i]) return q[i] < o.q[i];
    return false;
  }
};

// The monotone prefix of RankKey (aggregate + descending deviations), which is
// what a best-first walk over a winding lattice can be ordered by: both parts
// rise whenever any per-joint deviation rises.
template <int N>
struct LatticeKey {
  double agg = 0.0;
  std::array<double, N> desc{};

  bool operator<(const LatticeKey& o) const {
    if (agg != o.agg) return agg < o.agg;
    for (int i = 0; i < N; ++i)
      if (desc[i] != o.desc[i]) return desc[i] < o.desc[i];
    return false;
  }
  bool operator>(const LatticeKey& o) const { return o < *this; }
};

// Seeded ordering compares on this grid rather than on raw doubles. Exact ties
// are the norm once windings are enumerated -- under WrapLinf one dominant joint
// fixes the max for every winding of a branch -- and the tie-break then turns on
// the leading element, which the two backends compute to within about 1e-15 of
// each other. Comparing raw doubles let that noise decide the order. A grid far
// below any meaningful joint difference (1e-9 rad is a nanoradian) makes equal
// things compare equal on both backends. (postprocess._RANK_QUANTUM)
inline constexpr double kRankQuantum = 1e-9;
inline double snap(double x) { return std::round(x / kRankQuantum); }

template <int N>
RankKey<N> rank_key(const std::array<double, N>& q, const std::array<double, N>& deltas,
                    SeedMetric metric) {
  RankKey<N> key;
  key.agg = snap(aggregate<N>(deltas, metric));
  for (int i = 0; i < N; ++i) key.desc[i] = std::abs(deltas[i]);
  std::sort(key.desc.begin(), key.desc.end(), std::greater<double>());
  for (int i = 0; i < N; ++i) key.desc[i] = snap(key.desc[i]);
  for (int i = 0; i < N; ++i) key.q[i] = snap(q[i]);
  return key;
}

}  // namespace finalize_detail

// The joints admitting more than one in-limit winding: revolute, finitely
// limited, spanning strictly more than one turn. (postprocess.winding_joints)
struct WindingJoint {
  int idx;
  double lo, hi;
};

template <int N>
std::vector<WindingJoint> winding_joints(const JointConsts<N>& consts, const JointLimits<N>& lim) {
  std::vector<WindingJoint> out;
  for (int i = 0; i < N; ++i) {
    if (consts.type[i] != JointType::Revolute || !lim.present[i]) continue;
    if (lim.hi[i] - lim.lo[i] > 2.0 * M_PI + finalize_detail::kSpanTol)
      out.push_back(WindingJoint{i, lim.lo[i], lim.hi[i]});
  }
  return out;
}

// Every q + 2pi*k inside [lo, hi], ascending. The k range comes from the limits
// (so a boundary value like 0 under [-2pi, 2pi] yields {-2pi, 0, 2pi}), then
// each candidate is re-checked so round-off cannot emit an out-of-limit value.
// (postprocess._reps)
inline std::vector<double> winding_reps(double q, double lo, double hi) {
  constexpr double kTwoPi = 2.0 * M_PI;
  const int k_lo = static_cast<int>(std::ceil((lo - q) / kTwoPi)) - 1;
  const int k_hi = static_cast<int>(std::floor((hi - q) / kTwoPi)) + 1;
  std::vector<double> out;
  for (int k = k_lo; k <= k_hi; ++k) {
    const double v = q + kTwoPi * k;
    if (lo <= v && v <= hi) out.push_back(v);
  }
  // A value with no in-limit representative keeps its own, so expansion can
  // only ever add configurations. Expansion runs after the limit filter, so
  // this is unreachable in the wired paths; it is here so that a mistake in
  // that wiring could never silently delete solutions.
  if (out.empty()) out.push_back(q);
  return out;
}

// How many configurations expand_windings would produce, without building them.
// Keeps diagnostics truthful when a cap means the set is never materialized.
template <int N>
long long count_windings(const std::vector<Solution<N>>& sols,
                         const std::vector<WindingJoint>& wind) {
  if (wind.empty()) return static_cast<long long>(sols.size());
  long long total = 0;
  for (const auto& sol : sols) {
    long long n = 1;
    for (const auto& w : wind) n *= static_cast<long long>(winding_reps(sol.q[w.idx], w.lo, w.hi).size());
    total += n;
  }
  return total;
}

// Expand each solution into every in-limit winding representative: the
// Cartesian product across winding joints, branch-major, and within a branch in
// ascending value order. That is the stable order the ranking and truncation
// rules are defined against. `limit` stops early, and is only sound where the
// caller keeps a prefix (unseeded truncation). (postprocess.expand_windings)
template <int N>
std::vector<Solution<N>> expand_windings(const std::vector<Solution<N>>& sols,
                                         const std::vector<WindingJoint>& wind, int limit = -1) {
  if (wind.empty()) return sols;
  const int m = static_cast<int>(wind.size());
  std::vector<Solution<N>> out;
  for (const auto& sol : sols) {
    std::vector<std::vector<double>> opts;
    opts.reserve(m);
    for (const auto& w : wind) opts.push_back(winding_reps(sol.q[w.idx], w.lo, w.hi));
    std::vector<int> pos(m, 0);
    while (true) {
      Solution<N> s = sol;
      for (int t = 0; t < m; ++t) s.q[wind[t].idx] = opts[t][pos[t]];
      out.push_back(s);
      if (limit >= 0 && static_cast<int>(out.size()) >= limit) return out;
      int t = m - 1;  // odometer, last axis fastest (matches itertools.product)
      while (t >= 0 && ++pos[t] == static_cast<int>(opts[t].size())) pos[t--] = 0;
      if (t < 0) break;
    }
  }
  return out;
}

// wrap_to_limits: for each revolute joint with limits, try wraps k in
// (1,-1,2,-2) to bring q_i into [lo,hi] inclusive; first hit wins. Prismatic /
// no-limits / already-in-range joints untouched. (postprocess.py:101-151)
template <int N>
std::vector<Solution<N>> wrap_to_limits(const std::vector<Solution<N>>& sols,
                                        const JointConsts<N>& consts,
                                        const JointLimits<N>& lim) {
  static constexpr int kWrapOrder[4] = {1, -1, 2, -2};
  std::vector<Solution<N>> out;
  out.reserve(sols.size());
  for (const auto& sol : sols) {
    Solution<N> s = sol;  // copy; q gets adjusted in place
    for (int i = 0; i < N; ++i) {
      if (!lim.present[i] || consts.type[i] != JointType::Revolute) continue;
      const double lo = lim.lo[i], hi = lim.hi[i];
      const double qi = sol.q[i];
      if (lo <= qi && qi <= hi) continue;
      for (int k : kWrapOrder) {
        const double cand = qi + 2.0 * M_PI * k;
        if (lo <= cand && cand <= hi) {
          s.q[i] = cand;
          break;
        }
      }
    }
    out.push_back(s);
  }
  return out;
}

// respect_limits: drop a solution if any limited joint has q_i < lo or q_i > hi
// (strict -> boundary values accepted); no wrapping. (postprocess.py:69-98)
template <int N>
std::vector<Solution<N>> apply_respect_limits(const std::vector<Solution<N>>& sols,
                                              const JointLimits<N>& lim) {
  std::vector<Solution<N>> out;
  for (const auto& sol : sols) {
    bool within = true;
    for (int i = 0; i < N; ++i) {
      if (!lim.present[i]) continue;
      if (sol.q[i] < lim.lo[i] || sol.q[i] > lim.hi[i]) {
        within = false;
        break;
      }
    }
    if (within) out.push_back(sol);
  }
  return out;
}

// within_seed_tolerance: keep iff max_i |wrap(q_i - seed_i)| <= tol (Linf,
// inclusive). (postprocess.py:198-228)
// Measured the way the joint actually moves: modulo 2pi for continuous joints
// only, ordinary distance for finite revolute and prismatic (#562). A finite
// joint that must travel 6 radians is no longer admitted by a 0.5-radian bound
// just because its endpoints happen to wrap close.
template <int N>
std::vector<Solution<N>> within_seed_tolerance(const std::vector<Solution<N>>& sols,
                                               const JointConsts<N>& consts,
                                               const JointLimits<N>& lim,
                                               const std::array<double, N>& seed, double tol) {
  std::vector<Solution<N>> out;
  for (const auto& sol : sols) {
    const auto d = finalize_detail::seed_deltas<N>(sol.q, consts, lim, seed);
    bool ok = true;
    for (int i = 0; i < N; ++i) {
      if (std::abs(d[i]) > tol) {
        ok = false;
        break;
      }
    }
    if (ok) out.push_back(sol);
  }
  return out;
}

// nearest_to_seed: STABLE sort by distance to seed. wrap_l2 = sqrt(sum d_i^2),
// wrap_linf = max |d_i|, d_i = wrap(q_i - seed_i). (postprocess.py:159-195)
template <int N>
std::vector<Solution<N>> nearest_to_seed(std::vector<Solution<N>> sols,
                                         const JointConsts<N>& consts, const JointLimits<N>& lim,
                                         const std::array<double, N>& seed, SeedMetric metric) {
  auto key = [&](const Solution<N>& sol) {
    return finalize_detail::rank_key<N>(
        sol.q, finalize_detail::seed_deltas<N>(sol.q, consts, lim, seed), metric);
  };
  std::stable_sort(sols.begin(), sols.end(),
                   [&](const Solution<N>& a, const Solution<N>& b) { return key(a) < key(b); });
  return sols;
}

// rewrap_to_seed (#562 step 1): rewrap each revolute joint to the q_i + 2pi*k
// representative nearest the seed, staying within finite limits. Finite limits:
// among all q_i + 2pi*k in [lo,hi], pick the one nearest seed_i (ties -> smaller
// value). Continuous (no limits): nearest turn (round). Prismatic: untouched.
// Only the coordinate changes; FK is identical. (postprocess.py rewrap_to_seed)
// continuous_only: enumeration already emits every in-limit representative of a
//   finite joint, so collapsing them onto the seed-nearest one would destroy the
//   set being returned. Continuous joints are never enumerated, so they still
//   need the nearest-turn choice.
//
// A value already outside its limits (only reachable via respect_limits=false,
// where the caller wants the raw geometric set) takes the nearest turn,
// unclamped: dragging it into limits the caller waived returned a
// representative a full turn from the seed. An in-limit value never leaves its
// limits. The choice is per value rather than per call, because a call-level
// flag would have to mean "the user wanted limits", which is not what the
// pipeline's own respect_limits says by the time the ranking pass runs.
template <int N>
std::vector<Solution<N>> rewrap_to_seed(std::vector<Solution<N>> sols,
                                        const JointConsts<N>& consts, const JointLimits<N>& lim,
                                        const std::array<double, N>& seed,
                                        bool continuous_only = false) {
  constexpr double kTwoPi = 2.0 * M_PI;
  for (auto& sol : sols) {
    for (int i = 0; i < N; ++i) {
      if (consts.type[i] != JointType::Revolute) continue;
      const double qi = sol.q[i], si = seed[i];
      if (!lim.present[i] || qi < lim.lo[i] || qi > lim.hi[i]) {  // nearest turn, no clamp
        sol.q[i] = qi + kTwoPi * std::round((si - qi) / kTwoPi);
        continue;
      }
      if (continuous_only) continue;  // enumeration emits this joint's reps
      // Same candidate set expansion uses, so the two agree exactly.
      double best = qi;
      double best_d = std::numeric_limits<double>::infinity();
      for (double cand : winding_reps(qi, lim.lo[i], lim.hi[i])) {
        const double d = std::abs(cand - si);
        if (d < best_d || (d == best_d && cand < best)) {  // nearest, tie -> smaller value
          best = cand;
          best_d = d;
        }
      }
      sol.q[i] = best;
    }
  }
  return sols;
}

// The globally nearest k winding representatives, without materializing the
// complete expansion (postprocess._windings_topk).
//
// Exactly equivalent to expand -> rank -> truncate, ties included, but it never
// builds the discarded configurations: a UR lifts 8 branches to 256 and a
// Doosan to 1944, while the tracking idiom asks for one. Two facts make the
// pruning exact. A configuration in the global top-k is in its own branch's
// top-k, so per-branch top-k then a global merge loses nothing. And within a
// branch the per-joint choices are independent while both metrics are
// non-decreasing in every deviation, so the representatives can be walked in
// ascending order by best-first search over the product lattice.
template <int N>
std::vector<Solution<N>> windings_topk(const std::vector<Solution<N>>& sols,
                                       const JointConsts<N>& consts, const JointLimits<N>& lim,
                                       const std::array<double, N>& seed, SeedMetric metric,
                                       int k) {
  if (k <= 1) {
    // The tracking idiom. Choosing each joint's seed-nearest representative
    // minimizes every per-joint deviation at once, so it minimizes both the
    // aggregate and the leximax refinement: the branch's best winding is
    // exactly its seed-rewrap, and no lattice search is needed.
    auto best = nearest_to_seed<N>(rewrap_to_seed<N>(sols, consts, lim, seed), consts, lim, seed,
                                   metric);
    if (static_cast<int>(best.size()) > k) best.resize(std::max(k, 0));
    return best;
  }

  const auto wind = winding_joints<N>(consts, lim);
  const int m = static_cast<int>(wind.size());
  using Key = finalize_detail::LatticeKey<N>;
  std::vector<Solution<N>> picked;

  for (const auto& sol : sols) {
    const auto base = finalize_detail::seed_deltas<N>(sol.q, consts, lim, seed);
    // Per winding joint: the in-limit representatives ordered by distance to the
    // seed, so rank 0 is the nearest and each step out costs more. Ties resolve
    // to the smaller value, matching Python's sort of (distance, value) pairs.
    std::vector<std::vector<std::pair<double, double>>> ladders(m);
    for (int t = 0; t < m; ++t) {
      for (double v : winding_reps(sol.q[wind[t].idx], wind[t].lo, wind[t].hi))
        ladders[t].emplace_back(std::abs(v - seed[wind[t].idx]), v);
      std::sort(ladders[t].begin(), ladders[t].end());
    }

    auto key_of = [&](const std::vector<int>& ranks) {
      std::array<double, N> d = base;
      for (int t = 0; t < m; ++t)
        d[wind[t].idx] = ladders[t][ranks[t]].second - seed[wind[t].idx];
      Key key;
      key.agg = finalize_detail::snap(finalize_detail::aggregate<N>(d, metric));
      for (int i = 0; i < N; ++i) key.desc[i] = std::abs(d[i]);
      std::sort(key.desc.begin(), key.desc.end(), std::greater<double>());
      for (int i = 0; i < N; ++i) key.desc[i] = finalize_detail::snap(key.desc[i]);
      return key;
    };

    using Entry = std::pair<Key, std::vector<int>>;
    auto worse = [](const Entry& a, const Entry& b) { return b.first < a.first; };
    std::priority_queue<Entry, std::vector<Entry>, decltype(worse)> heap(worse);
    std::set<std::vector<int>> seen;
    std::vector<int> start(m, 0);
    heap.push({key_of(start), start});
    seen.insert(start);

    int taken = 0;
    bool have_boundary = false;
    Key boundary;
    while (!heap.empty()) {
      // Stop once k are taken and the frontier has moved strictly past the k-th
      // key. Popping through an exact tie keeps the result identical to ranking
      // the complete expansion, whose tie order this cannot see.
      if (taken >= k && have_boundary && boundary < heap.top().first) break;
      const Entry top = heap.top();
      heap.pop();
      ++taken;
      if (taken == k) {
        boundary = top.first;
        have_boundary = true;
      }
      Solution<N> s = sol;
      for (int t = 0; t < m; ++t) s.q[wind[t].idx] = ladders[t][top.second[t]].second;
      picked.push_back(s);
      for (int t = 0; t < m; ++t) {
        std::vector<int> nxt = top.second;
        ++nxt[t];
        if (nxt[t] < static_cast<int>(ladders[t].size()) && seen.insert(nxt).second)
          heap.push({key_of(nxt), nxt});
      }
    }
  }

  auto ranked = nearest_to_seed<N>(std::move(picked), consts, lim, seed, metric);
  if (static_cast<int>(ranked.size()) > k) ranked.resize(k);
  return ranked;
}

// The finalize_solutions pipeline: limits -> seed(tolerance then rank) ->
// truncate, in that fixed order. (postprocess.py:255-301)
//
// in_limits_fallback (optional): a zero-arg callable invoked ONLY when
// respect_limits empties the set (redundant-7R exact resolver, #359). Its
// solutions are already in-limits, so they skip the wrap/drop pass and flow
// straight into seed -> truncate -- matching the Python hook exactly.
template <int N>
std::vector<Solution<N>> finalize_solutions(
    std::vector<Solution<N>> sols, const JointConsts<N>& consts, const JointLimits<N>& lim,
    const ArtifactParams<N>& p,
    const std::function<std::vector<Solution<N>>()>& in_limits_fallback = nullptr) {
  if (p.respect_limits) {
    sols = wrap_to_limits<N>(sols, consts, lim);
    sols = apply_respect_limits<N>(sols, lim);
    if (sols.empty() && in_limits_fallback) sols = in_limits_fallback();
  }

  // Whether this call is the one that lifts is the caller's decision (see the
  // field comment), not something inferred from respect_limits: the final
  // ranking pass runs with respect_limits=false because the limit filter
  // already happened, either here or inside the solver.
  const auto wind =
      p.enumerate_windings ? winding_joints<N>(consts, lim) : std::vector<WindingJoint>{};
  const bool expanding = !wind.empty();

  if (!p.has_seed) {
    // Unseeded output keeps expansion order, so a cap is a prefix and the
    // discarded representatives need never be built.
    if (expanding) sols = expand_windings<N>(sols, wind, p.max_solutions);
  } else {
    // #562 step 1: a seeded solve returns the representative nearest the seed
    // rather than the principal value, so it never commands a gratuitous 2pi
    // turn. Under enumeration the finite joints are covered by the expansion
    // itself (collapsing them here would destroy the set), leaving only the
    // continuous joints, whose lift family is infinite.
    sols = rewrap_to_seed<N>(std::move(sols), consts, lim, p.q_seed,
                             /*continuous_only=*/expanding);
    if (expanding) {
      if (!p.has_seed_tolerance && p.max_solutions >= 0) {
        // Ranked truncation: take the globally nearest max_solutions directly.
        sols = windings_topk<N>(sols, consts, lim, p.q_seed, p.seed_metric, p.max_solutions);
      } else {
        sols = expand_windings<N>(sols, wind);
      }
    }
    if (p.has_seed_tolerance)
      sols = within_seed_tolerance<N>(sols, consts, lim, p.q_seed, p.seed_tolerance);
    sols = nearest_to_seed<N>(std::move(sols), consts, lim, p.q_seed, p.seed_metric);
  }
  if (p.max_solutions >= 0 && static_cast<int>(sols.size()) > p.max_solutions) {
    sols.resize(p.max_solutions);
  }
  return sols;
}

}  // namespace ssik
