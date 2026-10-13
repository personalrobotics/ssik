// ssik::Solver at its trust boundary (#707). A SolverData comes from outside
// the library (a consumer's own conversion, a file), and its integer arrays are
// used as indices into fixed-size matrices and per-sample tables. Every
// malformed record must throw SolverDataError (or UnsupportedSolverData) from
// the constructor, and every malformed solve() input std::invalid_argument,
// before any array is read out of bounds. The end-to-end suites only ever feed
// well-formed data, so they cannot see this. Build with
// -DSSIK_SANITIZE=ON to run it under AddressSanitizer and UBSan.
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "ssik_cpp/solve.hpp"
#include "ur5_ik.hpp"

namespace {

using ssik::DataArray;
using ssik::DType;
using ssik::SolverData;

int g_failures = 0;
int g_cases = 0;

DataArray f64(std::vector<std::int64_t> shape, double fill = 0.0) {
  DataArray a;
  a.dtype = DType::Float64;
  std::int64_t n = 1;
  for (auto s : shape) n *= s;
  a.shape = std::move(shape);
  a.f64.assign(static_cast<std::size_t>(n), fill);
  return a;
}

DataArray i32(std::vector<std::int64_t> shape, std::vector<std::int32_t> values) {
  DataArray a;
  a.dtype = DType::Int32;
  a.shape = std::move(shape);
  a.i32 = std::move(values);
  return a;
}

DataArray i32_scalar(std::int32_t v) { return i32({}, {v}); }

DataArray f64_scalar(double v) { return f64({}, v); }

// Identity frames, z axes, revolute joints with limits: well-formed, if not a
// meaningful arm.
void add_joints(SolverData& d, int n) {
  d.arrays["axis"] = f64({n, 3});
  for (int i = 0; i < n; ++i) d.arrays["axis"].f64[3 * i + 2] = 1.0;
  d.arrays["t_left"] = f64({n, 4, 4});
  d.arrays["t_right"] = f64({n, 4, 4});
  for (const char* k : {"t_left", "t_right"})
    for (int i = 0; i < n; ++i)
      for (int r = 0; r < 4; ++r) d.arrays[k].f64[16 * i + 5 * r] = 1.0;
  d.arrays["joint_type"] = i32({n}, std::vector<std::int32_t>(n, 0));
  d.arrays["lo"] = f64({n}, -3.0);
  d.arrays["hi"] = f64({n}, 3.0);
  d.arrays["present"] = i32({n}, std::vector<std::int32_t>(n, 1));
}

// UR5 (ikgeo.three_parallel), from the committed artifact header: real data,
// so solve() has something to find.
SolverData ur5() {
  SolverData d;
  d.family = "ikgeo.three_parallel";
  add_joints(d, 6);
  const auto c = ssik::ur5_ik::consts();
  const auto lim = ssik::ur5_ik::limits();
  for (int i = 0; i < 6; ++i) {
    for (int k = 0; k < 3; ++k) d.arrays["axis"].f64[3 * i + k] = c.axis[i][k];
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col) {
        d.arrays["t_left"].f64[16 * i + 4 * r + col] = c.t_left[i](r, col);
        d.arrays["t_right"].f64[16 * i + 4 * r + col] = c.t_right[i](r, col);
      }
    d.arrays["lo"].f64[i] = lim.lo[i];
    d.arrays["hi"].f64[i] = lim.hi[i];
  }
  return d;
}

// The RR arrays of one 6R sub-problem, optionally with a leading sample axis
// of 16 (jointlock): fixed-size constants plus a two-monomial COO tensor.
void add_rr(SolverData& d, bool stacked) {
  const int s = stacked ? 16 : 1;
  auto lead = [&](std::vector<std::int64_t> shape) {
    if (stacked) shape.insert(shape.begin(), 16);
    return shape;
  };
  for (const char* k : {"alpha", "a", "d", "theta_offset"}) d.arrays[k] = f64(lead({6}), 0.1);
  for (const char* k : {"t_pre_inv", "t_post_inv"}) {
    d.arrays[k] = f64(lead({4, 4}));
    for (int i = 0; i < s; ++i)
      for (int r = 0; r < 4; ++r) d.arrays[k].f64[16 * i + 5 * r] = 1.0;
  }
  auto rep = [&](std::vector<std::int32_t> v) {
    std::vector<std::int32_t> out;
    for (int i = 0; i < s; ++i) out.insert(out.end(), v.begin(), v.end());
    return out;
  };
  d.arrays["linearity_joint"] = stacked ? i32({16}, rep({2})) : i32_scalar(2);
  d.arrays["drop_joint"] = stacked ? i32({16}, rep({5})) : i32_scalar(5);
  d.arrays["left_bilinear"] = i32(lead({2}), rep({3, 4}));
  d.arrays["right_bilinear"] = i32(lead({2}), rep({0, 1}));
  d.arrays["p_sin"] = f64(lead({14, 9}), 0.5);
  d.arrays["p_cos"] = f64(lead({14, 9}), 0.25);
  // Two monomials per sample: the constant 1 and t[0] t[1] t[11]; two COO
  // entries in each of p_one and q.
  d.arrays["mono_factors"] = i32({2 * s, 3}, rep({-1, -1, -1, 0, 1, 11}));
  d.arrays["po_rc"] = i32({2 * s, 2}, rep({0, 0, 13, 8}));
  d.arrays["po_mono"] = i32({2 * s}, rep({0, 1}));
  d.arrays["po_coeff"] = f64({2 * s}, 1.0);
  d.arrays["q_rc"] = i32({2 * s, 2}, rep({0, 0, 13, 7}));
  d.arrays["q_mono"] = i32({2 * s}, rep({1, 0}));
  d.arrays["q_coeff"] = f64({2 * s}, -1.0);
  if (stacked) {
    std::vector<std::int32_t> off(17);
    for (int i = 0; i <= 16; ++i) off[i] = 2 * i;
    for (const char* k : {"mono_offsets", "po_offsets", "q_offsets"}) d.arrays[k] = i32({17}, off);
  }
}

SolverData general_6r() {
  SolverData d;
  d.family = "ikgeo.general_6r";
  add_joints(d, 6);
  add_rr(d, false);
  return d;
}

SolverData srs(const char* family = "seven_r.srs") {
  SolverData d;
  d.family = family;
  add_joints(d, 7);
  d.arrays["l_se"] = f64_scalar(0.4);
  d.arrays["l_ew"] = f64_scalar(0.4);
  for (const char* k : {"ee_offset_local", "shoulder_pivot", "upper_home", "forearm_home"})
    d.arrays[k] = f64({3}, 0.1);
  d.arrays["r_post_wrist"] = f64({3, 3});
  for (int r = 0; r < 3; ++r) d.arrays["r_post_wrist"].f64[4 * r] = 1.0;
  d.arrays["elbow_index"] = i32_scalar(3);
  d.arrays["general_path"] = i32_scalar(0);
  return d;
}

SolverData spherical_shoulder(const char* family = "seven_r.spherical_shoulder") {
  SolverData d;
  d.family = family;
  add_joints(d, 7);
  d.arrays["coef"] = f64({3, 48}, 0.01);
  return d;
}

void add_jointlock_head(SolverData& d, int kernel) {
  d.family = "jointlock.seven_r";
  add_joints(d, 7);
  d.arrays["inner_kernel"] = i32_scalar(kernel);
  d.arrays["lock_idx"] = i32_scalar(2);
  d.arrays["q_lock"] = f64({16}, 0.0);
  for (int i = 0; i < 16; ++i) d.arrays["q_lock"].f64[i] = -3.0 + 0.375 * i;
}

SolverData jointlock_rr() {
  SolverData d;
  add_jointlock_head(d, 0);
  add_rr(d, true);
  return d;
}

SolverData jointlock_hp() {
  SolverData d;
  add_jointlock_head(d, 1);
  d.arrays["t_u"] = f64({16, 4, 8, 2}, 0.1);
  d.arrays["t_w_pre"] = f64({16, 4, 8, 2}, 0.2);
  d.arrays["dh_a"] = f64({16, 5}, 0.3);
  d.arrays["dh_l"] = f64({16, 5}, 0.4);
  d.arrays["dh_d"] = f64({16, 4}, 0.5);
  d.arrays["theta_offset"] = f64({16, 6}, 0.0);
  for (const char* k : {"t_pre_inv", "t_post_inv", "t_z_neg_d1", "t_joint6_offset_inv"}) {
    d.arrays[k] = f64({16, 4, 4});
    for (int i = 0; i < 16; ++i)
      for (int r = 0; r < 4; ++r) d.arrays[k].f64[16 * i + 5 * r] = 1.0;
  }
  d.arrays["right_pv"] = i32({16}, std::vector<std::int32_t>(16, 0));
  d.arrays["drop_idx"] = i32({16}, std::vector<std::int32_t>(16, 7));
  d.arrays["sub_axis"] = f64({16, 6, 3}, 0.0);
  for (int i = 0; i < 16 * 6; ++i) d.arrays["sub_axis"].f64[3 * i + 2] = 1.0;
  for (const char* k : {"sub_t_left", "sub_t_right"}) {
    d.arrays[k] = f64({16, 6, 4, 4});
    for (int i = 0; i < 16 * 6; ++i)
      for (int r = 0; r < 4; ++r) d.arrays[k].f64[16 * i + 5 * r] = 1.0;
  }
  d.arrays["sub_joint_type"] = i32({16, 6}, std::vector<std::int32_t>(96, 0));
  return d;
}

void check(bool ok, const std::string& what) {
  ++g_cases;
  if (!ok) {
    ++g_failures;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

// The constructor must throw E (and nothing else) for `d`.
template <class E>
void expect_reject(const std::string& name, const SolverData& d) {
  bool thrown = false;
  std::string msg;
  try {
    ssik::Solver s(d);
  } catch (const E& e) {
    thrown = true;
    msg = e.what();
  } catch (const std::exception& e) {
    check(false, name + ": wrong exception type: " + e.what());
    return;
  }
  check(thrown, name + ": accepted");
  if (thrown && msg.empty()) check(false, name + ": empty message");
}

void bad(const std::string& name, SolverData d, const std::function<void(SolverData&)>& mutate) {
  mutate(d);
  expect_reject<ssik::SolverDataError>(name, d);
}

template <class F>
void expect_invalid(const std::string& name, F&& f) {
  bool thrown = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    thrown = true;
  } catch (const std::exception& e) {
    check(false, name + ": wrong exception type: " + e.what());
    return;
  }
  check(thrown, name + ": accepted");
}

}  // namespace

int main() {
  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Every family's well-formed record is accepted.
  for (const auto& [name, d] : std::vector<std::pair<std::string, SolverData>>{
           {"three_parallel", ur5()},
           {"general_6r", general_6r()},
           {"srs", srs()},
           {"srs_polished", srs("seven_r.srs_polished")},
           {"spherical_shoulder", spherical_shoulder()},
           {"spherical_shoulder_polished", spherical_shoulder("seven_r.spherical_shoulder_polished")},
           {"jointlock_rr", jointlock_rr()},
           {"jointlock_hp", jointlock_hp()}}) {
    try {
      ssik::Solver s(d);
      check(s.family() == d.family, name + ": family()");
      check(s.dof() == (d.arrays.at("axis").shape[0]), name + ": dof()");
    } catch (const std::exception& e) {
      check(false, name + ": well-formed record rejected: " + e.what());
    }
  }
  {
    SolverData d = ur5();
    d.family = "ikgeo.spherical_two_parallel";
    try {
      ssik::Solver s(d);
    } catch (const std::exception& e) {
      check(false, std::string("spherical_two_parallel rejected: ") + e.what());
    }
  }
  check(ssik::native_families().size() == 8, "native_families() has the 8 families");

  // A real solve and fk round trip through the dispatch.
  {
    const ssik::Solver s(ur5());
    const std::array<double, 6> q{0.3, -0.7, 0.9, 1.1, -0.5, 0.2};
    const ssik::Pose T = s.fk<6>(q);
    check((T - ssik::fk<6>(ssik::ur5_ik::consts(), q)).norm() == 0.0, "fk<6> matches ssik::fk");
    const auto sols = s.solve<6>(T);
    const auto ref = ssik::ur5_ik::solve(T);
    check(!sols.empty() && sols.size() == ref.size(), "solve<6> matches the artifact's count");
    bool same = sols.size() == ref.size();
    for (std::size_t i = 0; same && i < sols.size(); ++i) same = sols[i].q == ref[i].q;
    check(same, "solve<6> matches the artifact bit for bit");
    ssik::ArtifactParams<6> p;
    p.max_solutions = 0;
    check(s.solve<6>(T, p).empty(), "max_solutions 0 returns nothing");

    // solve() input checks.
    ssik::Pose bad_T = T;
    bad_T(0, 3) = nan;
    expect_invalid("NaN pose", [&] { s.solve<6>(bad_T); });
    bad_T = T;
    bad_T(0, 0) *= 1.01;
    expect_invalid("non-rigid pose", [&] { s.solve<6>(bad_T); });
    bad_T = T;
    bad_T.block<3, 1>(0, 0) *= -1.0;
    expect_invalid("reflection", [&] { s.solve<6>(bad_T); });
    bad_T = T;
    bad_T(3, 0) = 0.5;
    expect_invalid("bottom row", [&] { s.solve<6>(bad_T); });
    p = {};
    p.has_seed = true;
    p.q_seed[2] = std::numeric_limits<double>::infinity();
    expect_invalid("inf seed", [&] { s.solve<6>(T, p); });
    p = {};
    p.max_solutions = -2;
    expect_invalid("max_solutions -2", [&] { s.solve<6>(T, p); });
    p = {};
    p.refinement_max_iters = -1;
    expect_invalid("refinement_max_iters -1", [&] { s.solve<6>(T, p); });
    expect_invalid("solve<7> on a 6-DOF solver", [&] { s.solve<7>(T); });
    expect_invalid("fk<7> on a 6-DOF solver", [&] { s.fk<7>(std::array<double, 7>{}); });
  }

  // Schema and family.
  {
    SolverData d = ur5();
    d.schema = ssik::kSolverDataSchema + 1;
    expect_reject<ssik::UnsupportedSolverData>("schema mismatch", d);
    d = ur5();
    d.family = "ikgeo.spherical";  // a dispatcher family with no native solver
    expect_reject<ssik::UnsupportedSolverData>("non-native family", d);
    d.family = "no.such_family";
    expect_reject<ssik::UnsupportedSolverData>("unknown family", d);
  }

  // Names.
  bad("unknown name", ur5(), [](SolverData& d) { d.arrays["axes"] = f64({6, 3}); });
  bad("missing name", ur5(), [](SolverData& d) { d.arrays.erase("present"); });
  bad("RR name on a 6R geometric family", ur5(), [](SolverData& d) { d.arrays["p_sin"] = f64({14, 9}); });
  bad("HP name on a jointlock RR record", jointlock_rr(),
      [](SolverData& d) { d.arrays["t_u"] = f64({16, 4, 8, 2}); });

  // Shapes and dtypes.
  bad("axis rows != dof", ur5(), [](SolverData& d) { d.arrays["axis"] = f64({5, 3}); });
  bad("7 joints for a 6R family", ur5(), [](SolverData& d) { add_joints(d, 7); });
  bad("6 joints for a 7R family", srs(), [](SolverData& d) { add_joints(d, 6); });
  bad("t_left rank", ur5(), [](SolverData& d) { d.arrays["t_left"] = f64({6, 16}); });
  bad("values shorter than the shape", ur5(), [](SolverData& d) { d.arrays["t_right"].f64.resize(10); });
  bad("values longer than the shape", ur5(), [](SolverData& d) { d.arrays["lo"].f64.push_back(0.0); });
  bad("negative extent", ur5(), [](SolverData& d) { d.arrays["axis"].shape = {-6, -3}; });
  bad("extent product overflow", general_6r(), [](SolverData& d) {
    d.arrays["mono_factors"].shape = {std::int64_t{1} << 62, 3};
  });
  bad("float64 axis given as int32", ur5(),
      [](SolverData& d) { d.arrays["axis"] = i32({6, 3}, std::vector<std::int32_t>(18, 0)); });
  bad("int32 joint_type given as float64", ur5(), [](SolverData& d) { d.arrays["joint_type"] = f64({6}); });
  bad("values in the other dtype's vector", ur5(), [](SolverData& d) { d.arrays["present"].f64 = {1.0}; });
  bad("scalar given as (1,)", srs(), [](SolverData& d) { d.arrays["l_se"] = f64({1}, 0.4); });
  bad("coef (3, 47)", spherical_shoulder(), [](SolverData& d) { d.arrays["coef"] = f64({3, 47}); });
  bad("q_lock (15,)", jointlock_hp(), [](SolverData& d) { d.arrays["q_lock"] = f64({15}); });
  bad("p_sin (14, 8)", general_6r(), [](SolverData& d) { d.arrays["p_sin"] = f64({14, 8}); });
  bad("po_mono shorter than po_rc", general_6r(), [](SolverData& d) { d.arrays["po_mono"] = i32({1}, {0}); });
  bad("q_coeff longer than q_rc", general_6r(), [](SolverData& d) { d.arrays["q_coeff"] = f64({3}); });
  bad("mono_factors (M, 2)", general_6r(),
      [](SolverData& d) { d.arrays["mono_factors"] = i32({2, 2}, {-1, -1, 0, 1}); });
  bad("sub_t_left (16, 6, 4, 3)", jointlock_hp(),
      [](SolverData& d) { d.arrays["sub_t_left"] = f64({16, 6, 4, 3}); });

  // Non-finite values.
  bad("NaN in coef", spherical_shoulder(), [&](SolverData& d) { d.arrays["coef"].f64[100] = nan; });
  bad("inf in t_left", ur5(),
      [](SolverData& d) { d.arrays["t_left"].f64[3] = std::numeric_limits<double>::infinity(); });
  bad("NaN in a COO coefficient", general_6r(), [&](SolverData& d) { d.arrays["q_coeff"].f64[1] = nan; });
  bad("NaN in an HP tensor", jointlock_hp(), [&](SolverData& d) { d.arrays["t_u"].f64[999] = nan; });

  // Out-of-range indices and flags.
  bad("joint_type 2", ur5(), [](SolverData& d) { d.arrays["joint_type"].i32[4] = 2; });
  bad("present -1", ur5(), [](SolverData& d) { d.arrays["present"].i32[0] = -1; });
  bad("elbow_index 7", srs(), [](SolverData& d) { d.arrays["elbow_index"].i32[0] = 7; });
  bad("elbow_index -1", srs(), [](SolverData& d) { d.arrays["elbow_index"].i32[0] = -1; });
  bad("general_path 2", srs(), [](SolverData& d) { d.arrays["general_path"].i32[0] = 2; });
  bad("lock_idx 7", jointlock_rr(), [](SolverData& d) { d.arrays["lock_idx"].i32[0] = 7; });
  bad("inner_kernel 2", jointlock_rr(), [](SolverData& d) { d.arrays["inner_kernel"].i32[0] = 2; });
  bad("drop_idx 8", jointlock_hp(), [](SolverData& d) { d.arrays["drop_idx"].i32[9] = 8; });
  bad("right_pv 2", jointlock_hp(), [](SolverData& d) { d.arrays["right_pv"].i32[0] = 2; });
  bad("sub_joint_type 5", jointlock_hp(), [](SolverData& d) { d.arrays["sub_joint_type"].i32[17] = 5; });

  // The RR tensor: every index is checked against the matrix it addresses.
  bad("COO row 14", general_6r(), [](SolverData& d) { d.arrays["po_rc"].i32[2] = 14; });
  bad("COO row -1", general_6r(), [](SolverData& d) { d.arrays["q_rc"].i32[0] = -1; });
  bad("p_one COO col 9", general_6r(), [](SolverData& d) { d.arrays["po_rc"].i32[3] = 9; });
  bad("q COO col 8", general_6r(), [](SolverData& d) { d.arrays["q_rc"].i32[3] = 8; });
  bad("COO monomial == M", general_6r(), [](SolverData& d) { d.arrays["po_mono"].i32[1] = 2; });
  bad("COO monomial -1", general_6r(), [](SolverData& d) { d.arrays["q_mono"].i32[0] = -1; });
  bad("mono factor 12", general_6r(), [](SolverData& d) { d.arrays["mono_factors"].i32[5] = 12; });
  bad("mono factor -2", general_6r(), [](SolverData& d) { d.arrays["mono_factors"].i32[0] = -2; });
  bad("joint role 6", general_6r(), [](SolverData& d) { d.arrays["drop_joint"].i32[0] = 6; });
  bad("joint roles not a permutation", general_6r(),
      [](SolverData& d) { d.arrays["left_bilinear"].i32[1] = 0; });
  bad("stacked joint roles not a permutation", jointlock_rr(),
      [](SolverData& d) { d.arrays["right_bilinear"].i32[2 * 9] = 5; });

  // Jointlock RR CSR offsets: monotone, from 0 to the row count, and monomial
  // indices local to their sample.
  bad("offsets not starting at 0", jointlock_rr(), [](SolverData& d) { d.arrays["po_offsets"].i32[0] = 1; });
  bad("offsets not ending at the row count", jointlock_rr(),
      [](SolverData& d) { d.arrays["q_offsets"].i32[16] = 31; });
  bad("offsets past the row count", jointlock_rr(),
      [](SolverData& d) { d.arrays["mono_offsets"].i32[16] = 33; });
  bad("offsets decreasing", jointlock_rr(), [](SolverData& d) {
    d.arrays["po_offsets"].i32[5] = 12;
    d.arrays["po_offsets"].i32[6] = 11;
  });
  bad("offsets (16,)", jointlock_rr(), [](SolverData& d) { d.arrays["q_offsets"] = i32({16}, std::vector<std::int32_t>(16, 0)); });
  bad("monomial index of the next sample", jointlock_rr(),
      [](SolverData& d) { d.arrays["po_mono"].i32[2 * 4 + 1] = 2; });
  bad("sample with no monomials but a COO entry", jointlock_rr(), [](SolverData& d) {
    // Sample 3 owns no monomials, yet its COO entries name monomial 0.
    for (int i = 4; i <= 16; ++i) d.arrays["mono_offsets"].i32[i] -= 2;
    d.arrays["mono_factors"].shape = {30, 3};
    d.arrays["mono_factors"].i32.resize(90);
  });

  std::printf("SolverData trust boundary: %d checks, %d failed\n", g_cases, g_failures);
  if (g_failures == 0) {
    std::printf("PASS\n");
    return 0;
  }
  return 1;
}
