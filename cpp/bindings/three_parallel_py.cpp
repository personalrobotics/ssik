// Test-only pybind11 binding (#499): exposes the native three_parallel solver
// as a Python callable so the *existing* Python test suite validates the C++
// backend directly -- no assertion logic re-written in C++. This is NOT a
// shipped artifact and NOT the deployment path (ADR-0001: the native artifact
// needs no Python; Python stays the reference oracle). It exists purely to run
// tests/test_three_parallel.py against both backends.
#include <array>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ssik_cpp/fk.hpp"
#include "ssik_cpp/seven_r/feasible_arcs.hpp"
#include "ssik_cpp/generalized_euler.hpp"
#include "ssik_cpp/seven_r/srs_swivel_limits.hpp"
#include "ssik_cpp/solvers/husty_pfurner.hpp"
#include "ssik_cpp/solvers/spherical_two_parallel.hpp"
#include "ssik_cpp/solvers/srs.hpp"
#include "ssik_cpp/solvers/srs_canonical.hpp"
#include "ssik_cpp/solvers/three_parallel.hpp"

namespace py = pybind11;

namespace {

// Build a native JointConsts<N> from the KinBody arrays marshalled by the Python
// adapter. Shapes: axes (N,3), t_left/t_right (N,4,4), types (N,).
template <int N>
ssik::JointConsts<N> make_consts_n(const py::array_t<double>& axes,
                                   const py::array_t<double>& t_left,
                                   const py::array_t<double>& t_right,
                                   const py::array_t<int>& types) {
  auto a = axes.unchecked<2>();
  auto tl = t_left.unchecked<3>();
  auto tr = t_right.unchecked<3>();
  auto ty = types.unchecked<1>();
  ssik::JointConsts<N> c;
  for (int i = 0; i < N; ++i) {
    c.axis[i] = Eigen::Vector3d(a(i, 0), a(i, 1), a(i, 2));
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col) {
        c.t_left[i](r, col) = tl(i, r, col);
        c.t_right[i](r, col) = tr(i, r, col);
      }
    c.type[i] = ty(i) == 0 ? ssik::JointType::Revolute : ssik::JointType::Prismatic;
  }
  return c;
}

ssik::JointConsts<6> make_consts(const py::array_t<double>& axes,
                                 const py::array_t<double>& t_left,
                                 const py::array_t<double>& t_right,
                                 const py::array_t<int>& types) {
  return make_consts_n<6>(axes, t_left, t_right, types);
}

// Core canonical-SRS solve (#512) for the beachhead validation. SRS geometric
// constants are precomputed in Python and passed in.
py::tuple srs_canonical_solve_py(py::array_t<double> axes, py::array_t<double> t_left,
                                 py::array_t<double> t_right, py::array_t<int> types, double l_se,
                                 double l_ew, py::array_t<double> ee_offset,
                                 py::array_t<double> shoulder_pivot,
                                 py::array_t<double> r_post_wrist, py::array_t<double> target) {
  const ssik::JointConsts<7> c = make_consts_n<7>(axes, t_left, t_right, types);
  ssik::SrsConsts s;
  s.l_se = l_se;
  s.l_ew = l_ew;
  auto eo = ee_offset.unchecked<1>();
  auto sp = shoulder_pivot.unchecked<1>();
  auto rp = r_post_wrist.unchecked<2>();
  for (int i = 0; i < 3; ++i) {
    s.ee_offset_local[i] = eo(i);
    s.shoulder_pivot[i] = sp(i);
    for (int j = 0; j < 3; ++j) s.r_post_wrist(i, j) = rp(i, j);
  }
  auto tm = target.unchecked<2>();
  ssik::Pose T;
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) T(r, col) = tm(r, col);

  const std::vector<ssik::Solution<7>> sols = ssik::srs_canonical_solve(c, s, T);
  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 7});
  py::array_t<double> resids(n);
  auto qm = qs.mutable_unchecked<2>();
  auto rm = resids.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 7; ++j) qm(i, j) = sols[i].q[j];
    rm(i) = sols[i].fk_residual;
  }
  return py::make_tuple(qs, resids);
}

// Core general-path SRS analytical solve (#354): the Davenport sweep for
// concurrent-axis arms the canonical path can't (offset/tilted wrist, non-ZYZ).
// Same pre-finalize candidate contract as srs_canonical_solve; needs the
// branch-enumeration extras (elbow_index, upper_home, forearm_home).
py::tuple srs_general_solve_py(py::array_t<double> axes, py::array_t<double> t_left,
                               py::array_t<double> t_right, py::array_t<int> types, double l_se,
                               double l_ew, py::array_t<double> ee_offset,
                               py::array_t<double> shoulder_pivot, py::array_t<double> r_post,
                               int elbow_index, py::array_t<double> upper_home,
                               py::array_t<double> forearm_home, py::array_t<double> target) {
  const ssik::JointConsts<7> c = make_consts_n<7>(axes, t_left, t_right, types);
  ssik::SrsConsts s;
  s.l_se = l_se;
  s.l_ew = l_ew;
  s.elbow_index = elbow_index;
  auto eo = ee_offset.unchecked<1>();
  auto sp = shoulder_pivot.unchecked<1>();
  auto uh = upper_home.unchecked<1>();
  auto fh = forearm_home.unchecked<1>();
  auto rp = r_post.unchecked<2>();
  for (int i = 0; i < 3; ++i) {
    s.ee_offset_local[i] = eo(i);
    s.shoulder_pivot[i] = sp(i);
    s.upper_home[i] = uh(i);
    s.forearm_home[i] = fh(i);
    for (int j = 0; j < 3; ++j) s.r_post_wrist(i, j) = rp(i, j);
  }
  auto tm = target.unchecked<2>();
  ssik::Pose T;
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) T(r, col) = tm(r, col);

  const std::vector<ssik::Solution<7>> sols = ssik::srs_general_solve(c, s, T);
  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 7});
  py::array_t<double> resids(n);
  auto qm = qs.mutable_unchecked<2>();
  auto rm = resids.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 7; ++j) qm(i, j) = sols[i].q[j];
    rm(i) = sols[i].fk_residual;
  }
  return py::make_tuple(qs, resids);
}

py::tuple three_parallel_solve_py(py::array_t<double> axes, py::array_t<double> t_left,
                                  py::array_t<double> t_right, py::array_t<int> types,
                                  py::array_t<double> target, bool allow_refinement,
                                  int refinement_max_iters) {
  const ssik::JointConsts<6> c = make_consts(axes, t_left, t_right, types);
  auto tm = target.unchecked<2>();
  ssik::Pose T;
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) T(r, col) = tm(r, col);

  const std::vector<ssik::Solution<6>> sols =
      ssik::three_parallel_solve(c, T, {}, allow_refinement, refinement_max_iters);

  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 6});
  py::array_t<double> resids(n);
  auto qm = qs.mutable_unchecked<2>();
  auto rm = resids.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 6; ++j) qm(i, j) = sols[i].q[j];
    rm(i) = sols[i].fk_residual;
  }
  const bool is_ls = n == 0;
  return py::make_tuple(qs, resids, is_ls);
}

// Core spherical_two_parallel solve (#510), for the beachhead validation against
// the Python solver. Caller passes CANONICAL constants.
py::tuple spherical_two_parallel_solve_py(py::array_t<double> axes, py::array_t<double> t_left,
                                          py::array_t<double> t_right, py::array_t<int> types,
                                          py::array_t<double> target, bool allow_refinement,
                                          int refinement_max_iters) {
  const ssik::JointConsts<6> c = make_consts(axes, t_left, t_right, types);
  auto tm = target.unchecked<2>();
  ssik::Pose T;
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) T(r, col) = tm(r, col);

  const std::vector<ssik::Solution<6>> sols =
      ssik::spherical_two_parallel_solve(c, T, {}, allow_refinement, refinement_max_iters);
  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 6});
  py::array_t<double> resids(n);
  auto qm = qs.mutable_unchecked<2>();
  auto rm = resids.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 6; ++j) qm(i, j) = sols[i].q[j];
    rm(i) = sols[i].fk_residual;
  }
  return py::make_tuple(qs, resids, n == 0);
}

// Full artifact-contract solve, family-dispatched (#503/#507/#510): exposes the
// <arm>_ik.solve() signature (limits, seed ranking, seed tolerance,
// max_solutions) so the reused Python tests + the shipped native=True path drive
// the native artifact layer. `family` selects the core solver; the force-refine
// + finalize (limits -> seed -> truncate) tail is shared. New geometric families
// are one more dispatch case here.
py::tuple native_artifact_solve_py(
    const std::string& family, py::array_t<double> axes, py::array_t<double> t_left,
    py::array_t<double> t_right, py::array_t<int> types, py::array_t<double> lo,
    py::array_t<double> hi, py::array_t<int> has_limits, py::array_t<double> target,
    bool respect_limits, bool has_seed, py::array_t<double> q_seed, const std::string& seed_metric,
    bool has_seed_tolerance, double seed_tolerance, int max_solutions, bool allow_rescue,
    int refinement_max_iters) {
  const ssik::JointConsts<6> c = make_consts(axes, t_left, t_right, types);

  ssik::JointLimits<6> lim;
  auto lo_u = lo.unchecked<1>();
  auto hi_u = hi.unchecked<1>();
  auto hl_u = has_limits.unchecked<1>();
  for (int i = 0; i < 6; ++i) {
    lim.lo[i] = lo_u(i);
    lim.hi[i] = hi_u(i);
    lim.present[i] = hl_u(i) != 0;
  }

  ssik::ArtifactParams<6> p;
  p.respect_limits = respect_limits;
  p.has_seed = has_seed;
  if (has_seed) {
    auto qs_u = q_seed.unchecked<1>();
    for (int i = 0; i < 6; ++i) p.q_seed[i] = qs_u(i);
  }
  p.seed_metric = seed_metric == "wrap_l2" ? ssik::SeedMetric::WrapL2 : ssik::SeedMetric::WrapLinf;
  p.has_seed_tolerance = has_seed_tolerance;
  p.seed_tolerance = seed_tolerance;
  p.max_solutions = max_solutions;
  p.allow_rescue = allow_rescue;
  p.refinement_max_iters = refinement_max_iters;

  auto tm = target.unchecked<2>();
  ssik::Pose T;
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) T(r, col) = tm(r, col);

  // Core solve (force-refined, as the artifact always polishes), family-selected;
  // rescue is dormant for these geometric families (guarded on the Python side).
  std::vector<ssik::Solution<6>> core;
  if (family == "ikgeo.spherical_two_parallel") {
    core = ssik::spherical_two_parallel_solve(c, T, {}, /*allow_refinement=*/true,
                                              refinement_max_iters);
  } else {
    core = ssik::three_parallel_solve(c, T, {}, /*allow_refinement=*/true, refinement_max_iters);
  }
  const std::vector<ssik::Solution<6>> sols = ssik::finalize_solutions<6>(core, c, lim, p);

  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 6});
  py::array_t<double> resids(n);
  py::array_t<int> refine(n);  // 0 = none, 1 = lm (Solution.refinement_used)
  auto qm = qs.mutable_unchecked<2>();
  auto rm = resids.mutable_unchecked<1>();
  auto fm = refine.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 6; ++j) qm(i, j) = sols[i].q[j];
    rm(i) = sols[i].fk_residual;
    fm(i) = sols[i].refinement == ssik::Refinement::None ? 0 : 1;
  }
  return py::make_tuple(qs, resids, refine);
}

}  // namespace

// Module name `_ssik_native`: the same source is compiled both as the test-only
// extension (built into cpp/build by scripts/build_cpp_ext.py) and as the
// shipped `ssik._ssik_native` extension in the wheel (hatch_build.py, #506).
// The pybind init symbol is keyed on the leaf name, so both import paths work.
// Test-only: exercise the C++ feasible_arcs geometry (#515) against the Python
// _feasible_param oracle. A synthetic two-harmonic joint family q_i(t) =
// off + a1 cos t + b1 sin t + a2 cos 2t + b2 sin 2t (coeffs shape (K,5)) is
// evaluated identically here and in Python; the arcs must match.
py::list feasible_arcs_test_py(py::array_t<double> coeffs, py::array_t<int> swept,
                               py::array_t<double> lo, py::array_t<double> hi,
                               py::array_t<double> grid_arr, bool bounded) {
  auto co = coeffs.unchecked<2>();
  const int K = static_cast<int>(co.shape(0));
  auto lo_u = lo.unchecked<1>();
  auto hi_u = hi.unchecked<1>();
  std::vector<ssik::feasible::Arc> limits(K);
  for (int i = 0; i < K; ++i) limits[i] = {lo_u(i), hi_u(i)};
  std::vector<int> sw;
  auto sw_u = swept.unchecked<1>();
  for (int i = 0; i < static_cast<int>(sw_u.shape(0)); ++i) sw.push_back(sw_u(i));
  std::vector<double> grid;
  auto g_u = grid_arr.unchecked<1>();
  for (int k = 0; k < static_cast<int>(g_u.shape(0)); ++k) grid.push_back(g_u(k));

  auto q_scalar = [&](double t) {
    std::vector<double> q(K);
    for (int i = 0; i < K; ++i)
      q[i] = co(i, 0) + co(i, 1) * std::cos(t) + co(i, 2) * std::sin(t) +
             co(i, 3) * std::cos(2.0 * t) + co(i, 4) * std::sin(2.0 * t);
    return q;
  };
  const auto arcs = bounded ? ssik::feasible::feasible_arcs_bounded(q_scalar, sw, limits, grid)
                            : ssik::feasible::feasible_arcs(q_scalar, sw, limits, grid);
  py::list out;
  for (const auto& [a, b] : arcs) out.append(py::make_tuple(a, b));
  return out;
}

// Test-only: exercise the C++ SRS swivel-limits resolver (#515) against Python
// resolve_in_limits. Takes the baked SRS geometry (base + branch-enumeration
// extras) + limits; returns the in-limits joint vectors.
py::list srs_resolve_in_limits_py(py::array_t<double> axes, py::array_t<double> t_left,
                                  py::array_t<double> t_right, py::array_t<int> types, double l_se,
                                  double l_ew, py::array_t<double> ee_offset,
                                  py::array_t<double> shoulder_pivot, py::array_t<double> r_post,
                                  int elbow_index, py::array_t<double> upper_home,
                                  py::array_t<double> forearm_home, py::array_t<double> lo,
                                  py::array_t<double> hi, py::array_t<double> target,
                                  double fk_atol) {
  const ssik::JointConsts<7> c = make_consts_n<7>(axes, t_left, t_right, types);
  ssik::SrsConsts s;
  s.l_se = l_se;
  s.l_ew = l_ew;
  s.elbow_index = elbow_index;
  auto eo = ee_offset.unchecked<1>();
  auto sp = shoulder_pivot.unchecked<1>();
  auto uh = upper_home.unchecked<1>();
  auto fh = forearm_home.unchecked<1>();
  auto rp = r_post.unchecked<2>();
  for (int i = 0; i < 3; ++i) {
    s.ee_offset_local[i] = eo(i);
    s.shoulder_pivot[i] = sp(i);
    s.upper_home[i] = uh(i);
    s.forearm_home[i] = fh(i);
    for (int j = 0; j < 3; ++j) s.r_post_wrist(i, j) = rp(i, j);
  }
  std::array<std::array<double, 2>, 7> limits;
  auto lo_u = lo.unchecked<1>();
  auto hi_u = hi.unchecked<1>();
  for (int i = 0; i < 7; ++i) limits[i] = {lo_u(i), hi_u(i)};
  auto tm = target.unchecked<2>();
  ssik::Pose T;
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) T(r, col) = tm(r, col);

  const auto sols = ssik::srs_swivel::resolve_in_limits(c, s, T, limits, fk_atol);
  py::list out;
  for (const auto& sol : sols) {
    py::array_t<double> q(7);
    auto qm = q.mutable_unchecked<1>();
    for (int i = 0; i < 7; ++i) qm(i) = sol.q[i];
    out.append(q);
  }
  return out;
}

// Test-only: exercise the full self-contained SRS artifact solve (#515) --
// seeded_track + srs_canonical_solve + finalize(in_limits_fallback) -- against
// the shipped Python prebuilt solve(). Takes the baked SrsConsts (base +
// branch-enumeration extras), JointLimits, and the ArtifactParams surface.
py::tuple srs_artifact_solve_py(py::array_t<double> axes, py::array_t<double> t_left,
                                py::array_t<double> t_right, py::array_t<int> types, double l_se,
                                double l_ew, py::array_t<double> ee_offset,
                                py::array_t<double> shoulder_pivot, py::array_t<double> r_post,
                                int elbow_index, py::array_t<double> upper_home,
                                py::array_t<double> forearm_home, py::array_t<double> lo,
                                py::array_t<double> hi, py::array_t<int> has_limits,
                                py::array_t<double> target, bool general_path, bool respect_limits,
                                bool has_seed, py::array_t<double> q_seed,
                                const std::string& seed_metric, bool has_seed_tolerance,
                                double seed_tolerance, int max_solutions, bool allow_rescue,
                                int refinement_max_iters) {
  const ssik::JointConsts<7> c = make_consts_n<7>(axes, t_left, t_right, types);
  ssik::SrsConsts s;
  s.l_se = l_se;
  s.l_ew = l_ew;
  s.elbow_index = elbow_index;
  s.general_path = general_path;
  auto eo = ee_offset.unchecked<1>();
  auto sp = shoulder_pivot.unchecked<1>();
  auto uh = upper_home.unchecked<1>();
  auto fh = forearm_home.unchecked<1>();
  auto rp = r_post.unchecked<2>();
  for (int i = 0; i < 3; ++i) {
    s.ee_offset_local[i] = eo(i);
    s.shoulder_pivot[i] = sp(i);
    s.upper_home[i] = uh(i);
    s.forearm_home[i] = fh(i);
    for (int j = 0; j < 3; ++j) s.r_post_wrist(i, j) = rp(i, j);
  }

  ssik::JointLimits<7> lim;
  auto lo_u = lo.unchecked<1>();
  auto hi_u = hi.unchecked<1>();
  auto hl_u = has_limits.unchecked<1>();
  for (int i = 0; i < 7; ++i) {
    lim.lo[i] = lo_u(i);
    lim.hi[i] = hi_u(i);
    lim.present[i] = hl_u(i) != 0;
  }

  ssik::ArtifactParams<7> p;
  p.respect_limits = respect_limits;
  p.has_seed = has_seed;
  if (has_seed) {
    auto qs_u = q_seed.unchecked<1>();
    for (int i = 0; i < 7; ++i) p.q_seed[i] = qs_u(i);
  }
  p.seed_metric = seed_metric == "wrap_l2" ? ssik::SeedMetric::WrapL2 : ssik::SeedMetric::WrapLinf;
  p.has_seed_tolerance = has_seed_tolerance;
  p.seed_tolerance = seed_tolerance;
  p.max_solutions = max_solutions;
  p.allow_rescue = allow_rescue;
  p.refinement_max_iters = refinement_max_iters;

  auto tm = target.unchecked<2>();
  ssik::Pose T;
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) T(r, col) = tm(r, col);

  const std::vector<ssik::Solution<7>> sols = ssik::srs_artifact_solve(c, s, lim, T, p);
  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 7});
  py::array_t<double> resids(n);
  py::array_t<int> refine(n);
  auto qm = qs.mutable_unchecked<2>();
  auto rm = resids.mutable_unchecked<1>();
  auto fm = refine.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 7; ++j) qm(i, j) = sols[i].q[j];
    rm(i) = sols[i].fk_residual;
    fm(i) = sols[i].refinement == ssik::Refinement::None ? 0 : 1;
  }
  return py::make_tuple(qs, resids, refine);
}

// Test-only: exercise the C++ generalized-Euler decomposition (#354) against the
// Python decompose_3axis oracle. Returns the up-to-2 (a,b,c) branches.
py::list decompose_3axis_test_py(py::array_t<double> R_arr, py::array_t<double> n1,
                                 py::array_t<double> n2, py::array_t<double> n3) {
  auto rm = R_arr.unchecked<2>();
  Eigen::Matrix3d R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R(i, j) = rm(i, j);
  auto v = [](py::array_t<double> a) {
    auto u = a.unchecked<1>();
    return Eigen::Vector3d(u(0), u(1), u(2));
  };
  const auto branches = ssik::geuler::decompose_3axis(R, v(n1), v(n2), v(n3));
  py::list out;
  for (const auto& b : branches) {
    py::array_t<double> t(3);
    auto tm = t.mutable_unchecked<1>();
    tm(0) = b.a;
    tm(1) = b.b;
    tm(2) = b.c;
    out.append(t);
  }
  return out;
}

// HP f/g kernel parity (#537): given the baked (4,8,2) tensors T_u / T_w_pre,
// the target's Study DQ sigma_E, and drop_idx, return (f (9,7), g (6,5)) exactly
// as _eliminate.compute_fg_numeric. Validates the sigma_E injection + Cramer
// interpolation + convolution stage bit-for-bit against Python.
py::tuple hp_compute_fg_test_py(py::array_t<double> t_u_arr, py::array_t<double> t_w_pre_arr,
                                py::array_t<double> sigma_e_arr, int drop_idx) {
  auto load_tensor = [](py::array_t<double> a, std::array<Eigen::Matrix<double, 4, 8>, 2>& dst) {
    auto u = a.unchecked<3>();  // (4, 8, 2)
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 8; ++j)
        for (int k = 0; k < 2; ++k) dst[k](i, j) = u(i, j, k);
  };
  ssik::HpConsts hp;
  load_tensor(t_u_arr, hp.t_u);
  load_tensor(t_w_pre_arr, hp.t_w_pre);
  auto se = sigma_e_arr.unchecked<1>();
  ssik::Vec8 sigma_E;
  for (int i = 0; i < 8; ++i) sigma_E[i] = se(i);

  ssik::hp_detail::Mat9x7 f;
  ssik::hp_detail::Mat6x5 g;
  ssik::hp_detail::compute_fg(hp, sigma_E, drop_idx, f, g);

  py::array_t<double> f_out({9, 7});
  auto fm = f_out.mutable_unchecked<2>();
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 7; ++j) fm(i, j) = f(i, j);
  py::array_t<double> g_out({6, 5});
  auto gm = g_out.mutable_unchecked<2>();
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 5; ++j) gm(i, j) = g(i, j);
  return py::make_tuple(f_out, g_out);
}

PYBIND11_MODULE(_ssik_native, m) {
  m.doc() = "Native three_parallel solver binding (test conformance + shipped native backend)";
  m.def("decompose_3axis_test", &decompose_3axis_test_py, py::arg("R"), py::arg("n1"),
        py::arg("n2"), py::arg("n3"));
  m.def("hp_compute_fg_test", &hp_compute_fg_test_py, py::arg("t_u"), py::arg("t_w_pre"),
        py::arg("sigma_e"), py::arg("drop_idx") = 7);
  m.def("three_parallel_solve", &three_parallel_solve_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("target"), py::arg("allow_refinement") = false,
        py::arg("refinement_max_iters") = 15);
  m.def("spherical_two_parallel_solve", &spherical_two_parallel_solve_py, py::arg("axes"),
        py::arg("t_left"), py::arg("t_right"), py::arg("types"), py::arg("target"),
        py::arg("allow_refinement") = false, py::arg("refinement_max_iters") = 15);
  m.def("srs_canonical_solve", &srs_canonical_solve_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("l_se"), py::arg("l_ew"), py::arg("ee_offset"),
        py::arg("shoulder_pivot"), py::arg("r_post_wrist"), py::arg("target"));
  m.def("srs_general_solve", &srs_general_solve_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("l_se"), py::arg("l_ew"), py::arg("ee_offset"),
        py::arg("shoulder_pivot"), py::arg("r_post_wrist"), py::arg("elbow_index"),
        py::arg("upper_home"), py::arg("forearm_home"), py::arg("target"));
  m.def("feasible_arcs_test", &feasible_arcs_test_py, py::arg("coeffs"), py::arg("swept"),
        py::arg("lo"), py::arg("hi"), py::arg("grid"), py::arg("bounded") = false);
  m.def("srs_resolve_in_limits", &srs_resolve_in_limits_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("l_se"), py::arg("l_ew"), py::arg("ee_offset"),
        py::arg("shoulder_pivot"), py::arg("r_post_wrist"), py::arg("elbow_index"),
        py::arg("upper_home"), py::arg("forearm_home"), py::arg("lo"), py::arg("hi"),
        py::arg("target"), py::arg("fk_atol"));
  m.def("srs_artifact_solve", &srs_artifact_solve_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("l_se"), py::arg("l_ew"), py::arg("ee_offset"),
        py::arg("shoulder_pivot"), py::arg("r_post_wrist"), py::arg("elbow_index"),
        py::arg("upper_home"), py::arg("forearm_home"), py::arg("lo"), py::arg("hi"),
        py::arg("has_limits"), py::arg("target"), py::arg("general_path") = false,
        py::arg("respect_limits") = true, py::arg("has_seed") = false, py::arg("q_seed"),
        py::arg("seed_metric") = "wrap_linf", py::arg("has_seed_tolerance") = false,
        py::arg("seed_tolerance") = 0.0, py::arg("max_solutions") = -1,
        py::arg("allow_rescue") = true, py::arg("refinement_max_iters") = 15);
  m.def("native_artifact_solve", &native_artifact_solve_py, py::arg("family"), py::arg("axes"),
        py::arg("t_left"), py::arg("t_right"), py::arg("types"), py::arg("lo"), py::arg("hi"),
        py::arg("has_limits"), py::arg("target"), py::arg("respect_limits") = true,
        py::arg("has_seed") = false, py::arg("q_seed"), py::arg("seed_metric") = "wrap_linf",
        py::arg("has_seed_tolerance") = false, py::arg("seed_tolerance") = 0.0,
        py::arg("max_solutions") = -1, py::arg("allow_rescue") = true,
        py::arg("refinement_max_iters") = 15);
}
