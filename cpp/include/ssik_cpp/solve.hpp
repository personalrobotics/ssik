// One entry point for every native solver family (#707).
//
// ssik::Solver takes a SolverData (ssik_cpp/solver_data.hpp), checks it once
// against the schema and prepares the family's constants once (the
// Raghavan-Roth coefficient tensors included). Then solve<N>(T, p) runs
// exactly the family's *_artifact_solve, and fk<N>(q) the arm's forward
// kinematics. ssik's own Python extension calls this class, so the native
// backend that ssik tests is the one a C++ consumer gets.
//
// Unlike the *_artifact_solve functions, which assume well-formed input, a
// Solver checks everything it is given: a malformed SolverData throws
// SolverDataError (UnsupportedSolverData for an unknown family or schema), and
// a malformed pose, seed or parameter throws std::invalid_argument. Every
// method is const and a Solver may be shared across threads.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "ssik_cpp/finalize.hpp"
#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/solver_data.hpp"
#include "ssik_cpp/solvers/general_6r.hpp"
#include "ssik_cpp/solvers/husty_pfurner.hpp"
#include "ssik_cpp/solvers/jointlock_seven_r.hpp"
#include "ssik_cpp/solvers/spherical_shoulder.hpp"
#include "ssik_cpp/solvers/spherical_two_parallel.hpp"
#include "ssik_cpp/solvers/srs.hpp"
#include "ssik_cpp/solvers/srs_polished.hpp"
#include "ssik_cpp/solvers/three_parallel.hpp"

namespace ssik {

// A SolverData that does not match the schema: a missing, unknown, mis-shaped
// or mis-typed array, a non-finite value or an out-of-range index.
class SolverDataError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

// A SolverData this build cannot read at all: another schema number, or a
// family with no native solver.
class UnsupportedSolverData : public SolverDataError {
 public:
  using SolverDataError::SolverDataError;
};

// The families Solver accepts.
inline const std::vector<std::string>& native_families() {
  static const std::vector<std::string> families{
      "ikgeo.three_parallel",          "ikgeo.spherical_two_parallel",
      "ikgeo.general_6r",              "seven_r.srs",
      "seven_r.srs_polished",          "seven_r.spherical_shoulder",
      "seven_r.spherical_shoulder_polished", "jointlock.seven_r"};
  return families;
}

namespace solve_detail {

// The jointlock lock-sample count every shipped jointlock arm bakes.
inline constexpr int kJointlockSamples = 16;
// HpConsts::drop_idx names one of the eight Study rows.
inline constexpr int kHpStudyRows = 8;
// The rigid-target check (docs/api.md "Input validation") at the default
// policy, subproblem_numerical = 1e-5, the only policy native implements.
inline constexpr double kPoseTol = 1e-5;

[[noreturn]] inline void fail(const std::string& family, const std::string& what) {
  throw SolverDataError("SolverData (" + family + "): " + what);
}

inline std::string shape_str(const std::vector<std::int64_t>& shape) {
  std::string s = "(";
  for (std::size_t k = 0; k < shape.size(); ++k) {
    if (k) s += ", ";
    s += shape[k] < 0 ? std::string("any") : std::to_string(shape[k]);
  }
  return s + (shape.size() == 1 ? ",)" : ")");
}

// Checked access to a SolverData's arrays. Every array is self-consistent
// (non-negative extents, exactly prod(shape) values in its dtype's vector,
// finite float64 values) before the family reads it; each read names the
// shape it requires; finish() rejects every array the family did not read.
class Reader {
 public:
  explicit Reader(const SolverData& d) : d_(d) {
    for (const auto& [name, a] : d.arrays) {
      std::uint64_t n = 1;
      for (std::int64_t e : a.shape) {
        if (e < 0) fail(d.family, "array '" + name + "' has a negative extent");
        if (e > 0 && n > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(e))
          fail(d.family, "array '" + name + "' has more elements than fit in memory");
        n *= static_cast<std::uint64_t>(e);
      }
      const bool f = a.dtype == DType::Float64;
      const std::size_t have = f ? a.f64.size() : a.i32.size();
      const std::size_t other = f ? a.i32.size() : a.f64.size();
      if (static_cast<std::uint64_t>(have) != n || other != 0)
        fail(d.family, "array '" + name + "' of shape " + shape_str(a.shape) + " must hold " +
                           std::to_string(n) + " values of its dtype and none of the other");
      if (f)
        for (double v : a.f64)
          if (!std::isfinite(v)) fail(d.family, "array '" + name + "' has a NaN or inf value");
    }
  }

  // A float64 array of exactly `shape` (an extent of -1 matches any).
  const std::vector<double>& f64(const std::string& name, std::vector<std::int64_t> shape) {
    return get(name, DType::Float64, shape).f64;
  }
  const std::vector<std::int32_t>& i32(const std::string& name, std::vector<std::int64_t> shape) {
    return get(name, DType::Int32, shape).i32;
  }
  double f64_scalar(const std::string& name) { return f64(name, {})[0]; }
  // An int32 scalar in [lo, hi).
  int index(const std::string& name, int lo, int hi) {
    const int v = i32(name, {})[0];
    require_range(name, v, lo, hi);
    return v;
  }
  // Every element of an int32 array in [lo, hi).
  void require_all(const std::string& name, const std::vector<std::int32_t>& v, int lo, int hi) {
    for (std::int32_t x : v) require_range(name, x, lo, hi);
  }
  void require_range(const std::string& name, std::int64_t v, std::int64_t lo, std::int64_t hi) {
    if (v < lo || v >= hi)
      fail(d_.family, "array '" + name + "' has " + std::to_string(v) + ", outside [" +
                          std::to_string(lo) + ", " + std::to_string(hi) + ")");
  }
  std::int64_t extent(const std::string& name, std::size_t axis) const {
    return d_.arrays.at(name).shape[axis];
  }
  bool has(const std::string& name) const { return d_.arrays.count(name) != 0; }
  const std::string& family() const { return d_.family; }

  void finish() const {
    for (const auto& [name, a] : d_.arrays)
      if (used_.count(name) == 0)
        fail(d_.family, "unknown array '" + name + "' (not in schema " +
                            std::to_string(kSolverDataSchema) + " for this family)");
  }

 private:
  const DataArray& get(const std::string& name, DType dtype, const std::vector<std::int64_t>& shape) {
    const auto it = d_.arrays.find(name);
    if (it == d_.arrays.end()) fail(d_.family, "missing array '" + name + "'");
    const DataArray& a = it->second;
    if (a.dtype != dtype)
      fail(d_.family, "array '" + name + "' must be " +
                          (dtype == DType::Float64 ? "float64" : "int32"));
    bool ok = a.shape.size() == shape.size();
    for (std::size_t k = 0; ok && k < shape.size(); ++k) ok = shape[k] < 0 || a.shape[k] == shape[k];
    if (!ok)
      fail(d_.family, "array '" + name + "' must have shape " + shape_str(shape) + ", got " +
                          shape_str(a.shape));
    used_.insert(name);
    return a;
  }

  const SolverData& d_;
  std::set<std::string> used_;
};

template <int N>
JointConsts<N> read_joints(Reader& r) {
  const auto& axis = r.f64("axis", {N, 3});
  const auto& tl = r.f64("t_left", {N, 4, 4});
  const auto& tr = r.f64("t_right", {N, 4, 4});
  const auto& ty = r.i32("joint_type", {N});
  r.require_all("joint_type", ty, 0, 2);
  JointConsts<N> c;
  for (int i = 0; i < N; ++i) {
    c.axis[i] = Eigen::Vector3d(axis[3 * i], axis[3 * i + 1], axis[3 * i + 2]);
    for (int row = 0; row < 4; ++row)
      for (int col = 0; col < 4; ++col) {
        c.t_left[i](row, col) = tl[16 * i + 4 * row + col];
        c.t_right[i](row, col) = tr[16 * i + 4 * row + col];
      }
    c.type[i] = ty[i] == 0 ? JointType::Revolute : JointType::Prismatic;
  }
  return c;
}

template <int N>
JointLimits<N> read_limits(Reader& r) {
  const auto& lo = r.f64("lo", {N});
  const auto& hi = r.f64("hi", {N});
  const auto& present = r.i32("present", {N});
  r.require_all("present", present, 0, 2);
  JointLimits<N> lim;
  for (int i = 0; i < N; ++i) {
    lim.lo[i] = lo[i];
    lim.hi[i] = hi[i];
    lim.present[i] = present[i] != 0;
  }
  return lim;
}

inline Eigen::Matrix4d mat4(const std::vector<double>& v, std::size_t offset = 0) {
  Eigen::Matrix4d m;
  for (int row = 0; row < 4; ++row)
    for (int col = 0; col < 4; ++col) m(row, col) = v[offset + 4 * row + col];
  return m;
}

// linearity, the two left- and two right-bilinear joints and the dropped joint
// name each of the six joints exactly once.
inline void require_joint_roles(Reader& r, const RrConsts& rr) {
  const int roles[6] = {rr.linearity_joint,  rr.left_bilinear[0], rr.left_bilinear[1],
                        rr.right_bilinear[0], rr.right_bilinear[1], rr.drop_joint};
  bool seen[6] = {false, false, false, false, false, false};
  for (int role : roles) {
    r.require_range("RR joint roles", role, 0, 6);
    if (seen[role]) fail(r.family(), "the RR joint roles must be a permutation of 0..5");
    seen[role] = true;
  }
}

// The RR constants of sample `s` of arrays carrying `lead` leading extents
// ({} for general_6r, {16} for jointlock).
inline RrConsts read_rr_consts(Reader& r, const std::vector<std::int64_t>& lead, int s) {
  auto shape = [&](std::vector<std::int64_t> tail) {
    tail.insert(tail.begin(), lead.begin(), lead.end());
    return tail;
  };
  RrConsts rr;
  const auto& alpha = r.f64("alpha", shape({6}));
  const auto& a = r.f64("a", shape({6}));
  const auto& d = r.f64("d", shape({6}));
  const auto& to = r.f64("theta_offset", shape({6}));
  for (int k = 0; k < 6; ++k) {
    rr.alpha[k] = alpha[6 * s + k];
    rr.a[k] = a[6 * s + k];
    rr.d[k] = d[6 * s + k];
    rr.theta_offset[k] = to[6 * s + k];
  }
  rr.t_pre_inv = mat4(r.f64("t_pre_inv", shape({4, 4})), 16 * s);
  rr.t_post_inv = mat4(r.f64("t_post_inv", shape({4, 4})), 16 * s);
  rr.linearity_joint = r.i32("linearity_joint", shape({}))[s];
  rr.drop_joint = r.i32("drop_joint", shape({}))[s];
  const auto& lb = r.i32("left_bilinear", shape({2}));
  const auto& rb = r.i32("right_bilinear", shape({2}));
  rr.left_bilinear = {lb[2 * s], lb[2 * s + 1]};
  rr.right_bilinear = {rb[2 * s], rb[2 * s + 1]};
  require_joint_roles(r, rr);
  return rr;
}

// Row ranges [begin[s], begin[s + 1]) of a ragged per-sample table.
struct Rows {
  std::int64_t begin = 0;
  std::int64_t end = 0;
};

// The RR coefficient tensor over monomial rows `mono` and COO rows `po` and
// `q` of the concatenated arrays. Monomial indices are local to `mono`.
inline rr_detail::RrCoeffTensor read_rr_tensor(Reader& r, const std::vector<std::int64_t>& lead,
                                               int s, Rows mono, Rows po, Rows q) {
  std::vector<std::int64_t> pshape(lead);
  pshape.push_back(14);
  pshape.push_back(9);
  const auto& p_sin = r.f64("p_sin", pshape);
  const auto& p_cos = r.f64("p_cos", pshape);
  rr_detail::RrCoeffTensor t;
  for (int row = 0; row < 14; ++row)
    for (int col = 0; col < 9; ++col) {
      t.p_sin(row, col) = p_sin[126 * s + 9 * row + col];
      t.p_cos(row, col) = p_cos[126 * s + 9 * row + col];
    }
  const auto& mf = r.i32("mono_factors", {-1, 3});
  for (std::int64_t m = mono.begin; m < mono.end; ++m) {
    // -1 pads a monomial of degree < 3; otherwise a factor is a target-entry index.
    for (int f = 0; f < 3; ++f) r.require_range("mono_factors", mf[3 * m + f], -1, 12);
    t.mono_factors.push_back({mf[3 * m], mf[3 * m + 1], mf[3 * m + 2]});
  }
  const std::int64_t n_mono = mono.end - mono.begin;
  auto load_coo = [&](const char* prefix, Rows rows, int n_cols, std::vector<int>& out_row,
                      std::vector<int>& out_col, std::vector<int>& out_mono,
                      std::vector<double>& out_coeff) {
    const std::string p(prefix);
    const auto& rc = r.i32(p + "_rc", {-1, 2});
    const std::int64_t n = r.extent(p + "_rc", 0);
    const auto& mo = r.i32(p + "_mono", {n});
    const auto& co = r.f64(p + "_coeff", {n});
    for (std::int64_t i = rows.begin; i < rows.end; ++i) {
      r.require_range(p + "_rc row", rc[2 * i], 0, 14);
      r.require_range(p + "_rc col", rc[2 * i + 1], 0, n_cols);
      r.require_range(p + "_mono", mo[i], 0, n_mono);
      out_row.push_back(rc[2 * i]);
      out_col.push_back(rc[2 * i + 1]);
      out_mono.push_back(mo[i]);
      out_coeff.push_back(co[i]);
    }
  };
  load_coo("po", po, 9, t.po_row, t.po_col, t.po_mono, t.po_coeff);
  load_coo("q", q, 8, t.q_row, t.q_col, t.q_mono, t.q_coeff);
  return t;
}

// A CSR offsets array (S + 1,) over `rows` rows: starts at 0, never
// decreases, ends at `rows`.
inline std::vector<std::int32_t> read_offsets(Reader& r, const std::string& name, int samples,
                                              std::int64_t rows) {
  const auto& off = r.i32(name, {samples + 1});
  if (off[0] != 0) fail(r.family(), "array '" + name + "' must start at 0");
  for (int s = 0; s < samples; ++s)
    if (off[s + 1] < off[s]) fail(r.family(), "array '" + name + "' must not decrease");
  if (off[samples] != rows)
    fail(r.family(), "array '" + name + "' must end at the row count " + std::to_string(rows) +
                         ", got " + std::to_string(off[samples]));
  return off;
}

// The RR coefficient evaluator a family solver calls, bound to a prepared
// tensor.
struct TensorCoeffs {
  const rr_detail::RrCoeffTensor* tensor = nullptr;
  void operator()(const double t12[12], rr_detail::Mat14x9& ps, rr_detail::Mat14x9& pc,
                  rr_detail::Mat14x9& po, rr_detail::Mat14x8& q) const {
    rr_detail::PqCoeffs pq;
    rr_detail::rr_eval_coeffs(*tensor, t12, pq);
    ps = pq.p_sin;
    pc = pq.p_cos;
    po = pq.p_one;
    q = pq.q;
  }
};

struct ThreeParallel {
  static constexpr int kDof = 6;
  JointConsts<6> c;
  JointLimits<6> lim;
  bool spherical_two_parallel = false;
  std::vector<Solution<6>> solve(const Pose& T, const ArtifactParams<6>& p) const {
    return spherical_two_parallel ? spherical_two_parallel_artifact_solve(c, lim, T, p)
                                  : three_parallel_artifact_solve(c, lim, T, p);
  }
};

struct General6r {
  static constexpr int kDof = 6;
  JointConsts<6> c;
  JointLimits<6> lim;
  RrConsts rr;
  rr_detail::RrCoeffTensor tensor;
  std::vector<Solution<6>> solve(const Pose& T, const ArtifactParams<6>& p) const {
    return general_6r_artifact_solve(c, rr, TensorCoeffs{&tensor}, lim, T, p);
  }
};

struct Srs {
  static constexpr int kDof = 7;
  JointConsts<7> c;
  JointLimits<7> lim;
  SrsConsts s;
  bool polished = false;
  std::vector<Solution<7>> solve(const Pose& T, const ArtifactParams<7>& p) const {
    return polished ? srs_polished_artifact_solve(c, s, lim, T, p)
                    : srs_artifact_solve(c, s, lim, T, p);
  }
};

struct SphericalShoulder {
  static constexpr int kDof = 7;
  JointConsts<7> c;
  JointLimits<7> lim;
  SphericalShoulderConsts sh;
  bool polished = false;
  std::vector<Solution<7>> solve(const Pose& T, const ArtifactParams<7>& p) const {
    return spherical_shoulder_artifact_solve(c, sh, lim, T, p, polished);
  }
};

struct JointlockRr {
  static constexpr int kDof = 7;
  static constexpr int S = kJointlockSamples;
  JointConsts<7> c;
  JointLimits<7> lim;
  JointlockConsts<S> jl;
  std::array<RrConsts, S> rr;
  std::array<rr_detail::RrCoeffTensor, S> tensors;
  std::vector<Solution<7>> solve(const Pose& T, const ArtifactParams<7>& p) const {
    std::array<TensorCoeffs, S> coeffs;
    for (int i = 0; i < S; ++i) coeffs[i].tensor = &tensors[i];
    return jointlock_artifact_solve<S>(c, jl, rr, coeffs, lim, T, p);
  }
};

struct JointlockHp {
  static constexpr int kDof = 7;
  static constexpr int S = kJointlockSamples;
  JointConsts<7> c;
  JointLimits<7> lim;
  JointlockConsts<S> jl;
  std::array<HpConsts, S> hp;
  std::array<JointConsts<6>, S> sub;
  std::vector<Solution<7>> solve(const Pose& T, const ArtifactParams<7>& p) const {
    return jointlock_hp_artifact_solve<S>(c, jl, hp, sub, lim, T, p);
  }
};

using Impl =
    std::variant<ThreeParallel, General6r, Srs, SphericalShoulder, JointlockRr, JointlockHp>;

inline Impl read_three_parallel(Reader& r, bool spherical_two_parallel) {
  ThreeParallel f;
  f.c = read_joints<6>(r);
  f.lim = read_limits<6>(r);
  f.spherical_two_parallel = spherical_two_parallel;
  return f;
}

inline Impl read_general_6r(Reader& r) {
  General6r f;
  f.c = read_joints<6>(r);
  f.lim = read_limits<6>(r);
  f.rr = read_rr_consts(r, {}, 0);
  r.i32("mono_factors", {-1, 3});
  r.i32("po_rc", {-1, 2});
  r.i32("q_rc", {-1, 2});
  f.tensor = read_rr_tensor(r, {}, 0, {0, r.extent("mono_factors", 0)}, {0, r.extent("po_rc", 0)},
                            {0, r.extent("q_rc", 0)});
  return f;
}

inline Eigen::Vector3d vec3(Reader& r, const std::string& name) {
  const auto& v = r.f64(name, {3});
  return Eigen::Vector3d(v[0], v[1], v[2]);
}

inline Impl read_srs(Reader& r, bool polished) {
  Srs f;
  f.c = read_joints<7>(r);
  f.lim = read_limits<7>(r);
  f.polished = polished;
  f.s.l_se = r.f64_scalar("l_se");
  f.s.l_ew = r.f64_scalar("l_ew");
  f.s.ee_offset_local = vec3(r, "ee_offset_local");
  f.s.shoulder_pivot = vec3(r, "shoulder_pivot");
  f.s.upper_home = vec3(r, "upper_home");
  f.s.forearm_home = vec3(r, "forearm_home");
  const auto& rp = r.f64("r_post_wrist", {3, 3});
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) f.s.r_post_wrist(i, j) = rp[3 * i + j];
  f.s.elbow_index = r.index("elbow_index", 0, 7);
  f.s.general_path = r.index("general_path", 0, 2) != 0;
  return f;
}

inline Impl read_spherical_shoulder(Reader& r, bool polished) {
  SphericalShoulder f;
  f.c = read_joints<7>(r);
  f.lim = read_limits<7>(r);
  f.polished = polished;
  const auto& coef = r.f64("coef", {3, 48});
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 48; ++j) f.sh.coef(i, j) = coef[48 * i + j];
  return f;
}

template <class F>
void read_jointlock_head(Reader& r, F& f) {
  f.c = read_joints<7>(r);
  f.lim = read_limits<7>(r);
  // lock_idx picks the locked joint of seven; the other six form the sub-chain.
  f.jl.lock_idx = r.index("lock_idx", 0, 7);
  const auto& q_lock = r.f64("q_lock", {kJointlockSamples});
  for (int i = 0; i < kJointlockSamples; ++i) f.jl.q_lock[i] = q_lock[i];
}

inline Impl read_jointlock_rr(Reader& r) {
  constexpr int S = kJointlockSamples;
  JointlockRr f;
  read_jointlock_head(r, f);
  r.i32("mono_factors", {-1, 3});
  r.i32("po_rc", {-1, 2});
  r.i32("q_rc", {-1, 2});
  const auto mono = read_offsets(r, "mono_offsets", S, r.extent("mono_factors", 0));
  const auto po = read_offsets(r, "po_offsets", S, r.extent("po_rc", 0));
  const auto q = read_offsets(r, "q_offsets", S, r.extent("q_rc", 0));
  for (int s = 0; s < S; ++s) {
    f.rr[s] = read_rr_consts(r, {S}, s);
    f.tensors[s] = read_rr_tensor(r, {S}, s, {mono[s], mono[s + 1]}, {po[s], po[s + 1]},
                                  {q[s], q[s + 1]});
  }
  return f;
}

inline Impl read_jointlock_hp(Reader& r) {
  constexpr int S = kJointlockSamples;
  JointlockHp f;
  read_jointlock_head(r, f);
  const auto& t_u = r.f64("t_u", {S, 4, 8, 2});
  const auto& t_w_pre = r.f64("t_w_pre", {S, 4, 8, 2});
  const auto& dh_a = r.f64("dh_a", {S, 5});
  const auto& dh_l = r.f64("dh_l", {S, 5});
  const auto& dh_d = r.f64("dh_d", {S, 4});
  const auto& to = r.f64("theta_offset", {S, 6});
  const auto& t_pre_inv = r.f64("t_pre_inv", {S, 4, 4});
  const auto& t_post_inv = r.f64("t_post_inv", {S, 4, 4});
  const auto& t_z = r.f64("t_z_neg_d1", {S, 4, 4});
  const auto& t_j6 = r.f64("t_joint6_offset_inv", {S, 4, 4});
  const auto& right_pv = r.i32("right_pv", {S});
  const auto& drop_idx = r.i32("drop_idx", {S});
  r.require_all("right_pv", right_pv, 0, 2);
  r.require_all("drop_idx", drop_idx, 0, kHpStudyRows);
  for (int i = 0; i < S; ++i) {
    HpConsts& hp = f.hp[i];
    for (int row = 0; row < 4; ++row)
      for (int col = 0; col < 8; ++col)
        for (int k = 0; k < 2; ++k) {
          hp.t_u[k](row, col) = t_u[64 * i + 16 * row + 2 * col + k];
          hp.t_w_pre[k](row, col) = t_w_pre[64 * i + 16 * row + 2 * col + k];
        }
    hp.t_pre_inv = mat4(t_pre_inv, 16 * i);
    hp.t_post_inv = mat4(t_post_inv, 16 * i);
    hp.t_z_neg_d1 = mat4(t_z, 16 * i);
    hp.t_joint6_offset_inv = mat4(t_j6, 16 * i);
    // dh_a / dh_l = [a_1..a_5] -> a[1..5]; dh_d = [d_2..d_5] -> d[2..5].
    for (int k = 0; k < 5; ++k) {
      hp.a[k + 1] = dh_a[5 * i + k];
      hp.l[k + 1] = dh_l[5 * i + k];
    }
    for (int k = 0; k < 4; ++k) hp.d[k + 2] = dh_d[4 * i + k];
    for (int k = 0; k < 6; ++k) hp.theta_offset[k] = to[6 * i + k];
    hp.right_parametric_var = right_pv[i];
    hp.drop_idx = drop_idx[i];
  }
  const auto& sub_axis = r.f64("sub_axis", {S, 6, 3});
  const auto& sub_tl = r.f64("sub_t_left", {S, 6, 4, 4});
  const auto& sub_tr = r.f64("sub_t_right", {S, 6, 4, 4});
  const auto& sub_ty = r.i32("sub_joint_type", {S, 6});
  r.require_all("sub_joint_type", sub_ty, 0, 2);
  for (int i = 0; i < S; ++i)
    for (int j = 0; j < 6; ++j) {
      const int ij = 6 * i + j;
      for (int k = 0; k < 3; ++k) f.sub[i].axis[j][k] = sub_axis[3 * ij + k];
      f.sub[i].t_left[j] = mat4(sub_tl, 16 * ij);
      f.sub[i].t_right[j] = mat4(sub_tr, 16 * ij);
      f.sub[i].type[j] = sub_ty[ij] == 0 ? JointType::Revolute : JointType::Prismatic;
    }
  return f;
}

inline Impl read(const SolverData& data) {
  if (data.schema != kSolverDataSchema)
    throw UnsupportedSolverData("SolverData schema " + std::to_string(data.schema) +
                                " is not the schema this ssik build reads (" +
                                std::to_string(kSolverDataSchema) + ")");
  const auto& fams = native_families();
  if (std::find(fams.begin(), fams.end(), data.family) == fams.end()) {
    std::string list;
    for (const auto& f : fams) list += (list.empty() ? "" : ", ") + f;
    throw UnsupportedSolverData("SolverData family '" + data.family +
                                "' has no native solver; the native families are " + list);
  }
  Reader r(data);
  const std::string& f = data.family;
  Impl impl = [&]() -> Impl {
    if (f == "ikgeo.three_parallel") return read_three_parallel(r, false);
    if (f == "ikgeo.spherical_two_parallel") return read_three_parallel(r, true);
    if (f == "ikgeo.general_6r") return read_general_6r(r);
    if (f == "seven_r.srs") return read_srs(r, false);
    if (f == "seven_r.srs_polished") return read_srs(r, true);
    if (f == "seven_r.spherical_shoulder") return read_spherical_shoulder(r, false);
    if (f == "seven_r.spherical_shoulder_polished") return read_spherical_shoulder(r, true);
    // jointlock.seven_r: the inner kernel, 0 Raghavan-Roth, 1 Husty-Pfurner.
    return r.index("inner_kernel", 0, 2) == 0 ? read_jointlock_rr(r) : read_jointlock_hp(r);
  }();
  r.finish();
  return impl;
}

// The solve() input checks: docs/api.md "Input validation" at the default
// policy (the same quantities, in the same order, as ssik._solve_inputs).
template <int N>
void check_input(const Pose& T, const ArtifactParams<N>& p) {
  if (!T.allFinite()) throw std::invalid_argument("target must be finite (no NaN or inf)");
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  if ((R.transpose() * R - Eigen::Matrix3d::Identity()).norm() > 3.0 * kPoseTol)
    throw std::invalid_argument("target is not a rigid transform: its rotation block is not orthonormal");
  if (R.determinant() <= 0.0)
    throw std::invalid_argument("target is not a rigid transform: its rotation block is a reflection");
  const Eigen::Vector4d bottom(T(3, 0), T(3, 1), T(3, 2), T(3, 3) - 1.0);
  if (bottom.norm() > kPoseTol) throw std::invalid_argument("target must have bottom row [0, 0, 0, 1]");
  if (p.has_seed)
    for (double v : p.q_seed)
      if (!std::isfinite(v)) throw std::invalid_argument("q_seed must be finite (no NaN or inf)");
  if (p.max_solutions < -1)
    throw std::invalid_argument("max_solutions must be -1 (no cap) or >= 0, got " +
                                std::to_string(p.max_solutions));
  if (p.refinement_max_iters < 0)
    throw std::invalid_argument("refinement_max_iters must be >= 0, got " +
                                std::to_string(p.refinement_max_iters));
}

}  // namespace solve_detail

class Solver {
 public:
  // Checks `data` and prepares the family's solver. Throws
  // UnsupportedSolverData (another schema, a family with no native solver) or
  // SolverDataError (any array missing, unknown, mis-shaped or mis-typed, a
  // non-finite value, an out-of-range index).
  explicit Solver(const SolverData& data)
      : family_(data.family),
        impl_(std::make_shared<const solve_detail::Impl>(solve_detail::read(data))) {
    dof_ = std::visit([](const auto& f) { return std::decay_t<decltype(f)>::kDof; }, *impl_);
  }

  int dof() const noexcept { return dof_; }
  const std::string& family() const noexcept { return family_; }

  // The family's full artifact solve. N must equal dof(). Throws
  // std::invalid_argument for a target that is not finite and rigid, a
  // non-finite seed (when p.has_seed), p.max_solutions < -1 or
  // p.refinement_max_iters < 0. p.max_solutions == 0 returns nothing.
  template <int N>
  std::vector<Solution<N>> solve(const Pose& T, const ArtifactParams<N>& p = {}) const {
    static_assert(N == 6 || N == 7, "ssik::Solver solves 6- and 7-joint arms");
    require_dof(N);
    solve_detail::check_input<N>(T, p);
    if (p.max_solutions == 0) return {};
    return std::visit(
        [&](const auto& f) -> std::vector<Solution<N>> {
          if constexpr (std::decay_t<decltype(f)>::kDof == N)
            return f.solve(T, p);
          else
            throw std::logic_error("ssik::Solver: unreachable dof mismatch");
        },
        *impl_);
  }

  // Forward kinematics of the arm's own joints (ssik::fk). N must equal dof().
  template <int N>
  Pose fk(const std::array<double, N>& q) const {
    static_assert(N == 6 || N == 7, "ssik::Solver solves 6- and 7-joint arms");
    require_dof(N);
    return std::visit(
        [&](const auto& f) -> Pose {
          if constexpr (std::decay_t<decltype(f)>::kDof == N)
            return ssik::fk<N>(f.c, q);
          else
            throw std::logic_error("ssik::Solver: unreachable dof mismatch");
        },
        *impl_);
  }

 private:
  void require_dof(int n) const {
    if (n != dof_)
      throw std::invalid_argument("ssik::Solver for " + family_ + " has " + std::to_string(dof_) +
                                  " joints, called with " + std::to_string(n));
  }

  std::string family_;
  std::shared_ptr<const solve_detail::Impl> impl_;
  int dof_ = 0;
};

}  // namespace ssik
