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
#include "ssik_cpp/solvers/general_6r.hpp"
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
// Build an RrCoeffTensor from the baked arrays (ssik._native.rr_native_geometry).
ssik::rr_detail::RrCoeffTensor make_rr_tensor(const py::array_t<double>& p_sin,
                                              const py::array_t<double>& p_cos,
                                              const py::array_t<int>& mono_factors,
                                              const py::array_t<int>& po_rc,
                                              const py::array_t<int>& po_mono,
                                              const py::array_t<double>& po_coeff,
                                              const py::array_t<int>& q_rc,
                                              const py::array_t<int>& q_mono,
                                              const py::array_t<double>& q_coeff) {
  ssik::rr_detail::RrCoeffTensor t;
  auto ps = p_sin.unchecked<2>();
  auto pc = p_cos.unchecked<2>();
  for (int r = 0; r < 14; ++r)
    for (int c = 0; c < 9; ++c) {
      t.p_sin(r, c) = ps(r, c);
      t.p_cos(r, c) = pc(r, c);
    }
  auto mf = mono_factors.unchecked<2>();  // (n_mono, 3)
  for (py::ssize_t m = 0; m < mf.shape(0); ++m)
    t.mono_factors.push_back({mf(m, 0), mf(m, 1), mf(m, 2)});
  auto load_coo = [](const py::array_t<int>& rc, const py::array_t<int>& mono,
                     const py::array_t<double>& coeff, std::vector<int>& rows, std::vector<int>& cols,
                     std::vector<int>& mo, std::vector<double>& co) {
    auto rcm = rc.unchecked<2>();  // rc (n,2)
    auto mm = mono.unchecked<1>();
    auto cc = coeff.unchecked<1>();
    for (py::ssize_t i = 0; i < rcm.shape(0); ++i) {
      rows.push_back(rcm(i, 0));
      cols.push_back(rcm(i, 1));
      mo.push_back(mm(i));
      co.push_back(cc(i));
    }
  };
  load_coo(po_rc, po_mono, po_coeff, t.po_row, t.po_col, t.po_mono, t.po_coeff);
  load_coo(q_rc, q_mono, q_coeff, t.q_row, t.q_col, t.q_mono, t.q_coeff);
  return t;
}

// Marshal the baked RR DH/gauge constants (ssik._native.rr_native_geometry) into
// an RrConsts. Shared by the raw parity binding and the production solve binding.
ssik::RrConsts make_rr_consts(const py::array_t<double>& alpha, const py::array_t<double>& a,
                              const py::array_t<double>& d, const py::array_t<double>& theta_offset,
                              const py::array_t<double>& t_pre_inv,
                              const py::array_t<double>& t_post_inv, int linearity_joint,
                              const py::array_t<int>& left_bilinear,
                              const py::array_t<int>& right_bilinear, int drop_joint) {
  ssik::RrConsts rr;
  auto al = alpha.unchecked<1>(), av = a.unchecked<1>(), dv = d.unchecked<1>(),
       to = theta_offset.unchecked<1>();
  for (int i = 0; i < 6; ++i) {
    rr.alpha[i] = al(i);
    rr.a[i] = av(i);
    rr.d[i] = dv(i);
    rr.theta_offset[i] = to(i);
  }
  auto tpi = t_pre_inv.unchecked<2>(), tpo = t_post_inv.unchecked<2>();
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) {
      rr.t_pre_inv(r, col) = tpi(r, col);
      rr.t_post_inv(r, col) = tpo(r, col);
    }
  rr.linearity_joint = linearity_joint;
  rr.drop_joint = drop_joint;
  auto lb = left_bilinear.unchecked<1>(), rb = right_bilinear.unchecked<1>();
  rr.left_bilinear = {lb(0), lb(1)};
  rr.right_bilinear = {rb(0), rb(1)};
  return rr;
}

// RR numeric-tensor evaluator parity (#555): build an RrCoeffTensor from the
// baked arrays (ssik._native.rr_native_geometry) + a target t12, run the generic
// rr_eval_coeffs, return (p_sin, p_cos, p_one, q) to compare against the Python
// lambdified reference / the emitted rr_coeffs fn.
py::tuple rr_eval_coeffs_test_py(py::array_t<double> p_sin, py::array_t<double> p_cos,
                                 py::array_t<int> mono_factors, py::array_t<int> po_rc,
                                 py::array_t<int> po_mono, py::array_t<double> po_coeff,
                                 py::array_t<int> q_rc, py::array_t<int> q_mono,
                                 py::array_t<double> q_coeff, py::array_t<double> t12_arr) {
  const ssik::rr_detail::RrCoeffTensor t =
      make_rr_tensor(p_sin, p_cos, mono_factors, po_rc, po_mono, po_coeff, q_rc, q_mono, q_coeff);
  double t12[12];
  auto tv = t12_arr.unchecked<1>();
  for (int i = 0; i < 12; ++i) t12[i] = tv(i);
  ssik::rr_detail::PqCoeffs pq;
  ssik::rr_detail::rr_eval_coeffs(t, t12, pq);
  py::array_t<double> ps_o({14, 9}), pc_o({14, 9}), po_o({14, 9}), q_o({14, 8});
  auto pso = ps_o.mutable_unchecked<2>(), pco = pc_o.mutable_unchecked<2>(),
       poo = po_o.mutable_unchecked<2>(), qo = q_o.mutable_unchecked<2>();
  for (int r = 0; r < 14; ++r) {
    for (int c = 0; c < 9; ++c) {
      pso(r, c) = pq.p_sin(r, c);
      pco(r, c) = pq.p_cos(r, c);
      poo(r, c) = pq.p_one(r, c);
    }
    for (int c = 0; c < 8; ++c) qo(r, c) = pq.q(r, c);
  }
  return py::make_tuple(ps_o, pc_o, po_o, q_o);
}

// Full native general_6r artifact solve via the baked RR tensor (#555): the RR
// runtime path a single shipped ext would take. Marshals JointConsts<6> +
// RrConsts + RrCoeffTensor, then general_6r_artifact_solve with a tensor lambda.
// Validated against Python general_6r.solve in tests/test_rr_tensor_cpp.
py::tuple general_6r_tensor_solve_py(
    py::array_t<double> axes, py::array_t<double> t_left, py::array_t<double> t_right,
    py::array_t<int> types, py::array_t<double> alpha, py::array_t<double> a, py::array_t<double> d,
    py::array_t<double> theta_offset, py::array_t<double> t_pre_inv, py::array_t<double> t_post_inv,
    int linearity_joint, py::array_t<int> left_bilinear, py::array_t<int> right_bilinear,
    int drop_joint, py::array_t<double> p_sin, py::array_t<double> p_cos,
    py::array_t<int> mono_factors, py::array_t<int> po_rc, py::array_t<int> po_mono,
    py::array_t<double> po_coeff, py::array_t<int> q_rc, py::array_t<int> q_mono,
    py::array_t<double> q_coeff, py::array_t<double> target) {
  const ssik::JointConsts<6> c = make_consts_n<6>(axes, t_left, t_right, types);
  const ssik::RrConsts rr =
      make_rr_consts(alpha, a, d, theta_offset, t_pre_inv, t_post_inv, linearity_joint,
                     left_bilinear, right_bilinear, drop_joint);
  const ssik::rr_detail::RrCoeffTensor tensor =
      make_rr_tensor(p_sin, p_cos, mono_factors, po_rc, po_mono, po_coeff, q_rc, q_mono, q_coeff);
  auto coeff_fn = [&tensor](const double t12[12], ssik::rr_detail::Mat14x9& ps,
                            ssik::rr_detail::Mat14x9& pc, ssik::rr_detail::Mat14x9& po,
                            ssik::rr_detail::Mat14x8& q) {
    ssik::rr_detail::PqCoeffs pq;
    ssik::rr_detail::rr_eval_coeffs(tensor, t12, pq);
    ps = pq.p_sin;
    pc = pq.p_cos;
    po = pq.p_one;
    q = pq.q;
  };
  auto tm = target.unchecked<2>();
  ssik::Pose T;
  for (int r = 0; r < 4; ++r)
    for (int col = 0; col < 4; ++col) T(r, col) = tm(r, col);
  ssik::JointLimits<6> lim;  // no limits (respect_limits=false path)
  ssik::ArtifactParams<6> p;
  p.respect_limits = false;
  const std::vector<ssik::Solution<6>> sols =
      ssik::general_6r_artifact_solve(c, rr, coeff_fn, lim, T, p);
  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 6});
  auto qm = qs.mutable_unchecked<2>();
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 6; ++j) qm(i, j) = sols[i].q[j];
  return py::make_tuple(qs);
}

// Production RR runtime path (#555): full general_6r artifact solve via the baked
// tensor, with the same ArtifactParams surface + (qs, resids, refine) return as
// native_artifact_solve_py. This is what ssik._native.try_native_solve calls for
// the ikgeo.general_6r family once the tensor is baked (sidecar .npz).
py::tuple general_6r_tensor_artifact_solve_py(
    py::array_t<double> axes, py::array_t<double> t_left, py::array_t<double> t_right,
    py::array_t<int> types, py::array_t<double> lo, py::array_t<double> hi,
    py::array_t<int> has_limits, py::array_t<double> alpha, py::array_t<double> a,
    py::array_t<double> d, py::array_t<double> theta_offset, py::array_t<double> t_pre_inv,
    py::array_t<double> t_post_inv, int linearity_joint, py::array_t<int> left_bilinear,
    py::array_t<int> right_bilinear, int drop_joint, py::array_t<double> p_sin,
    py::array_t<double> p_cos, py::array_t<int> mono_factors, py::array_t<int> po_rc,
    py::array_t<int> po_mono, py::array_t<double> po_coeff, py::array_t<int> q_rc,
    py::array_t<int> q_mono, py::array_t<double> q_coeff, py::array_t<double> target,
    bool respect_limits, bool has_seed, py::array_t<double> q_seed, const std::string& seed_metric,
    bool has_seed_tolerance, double seed_tolerance, int max_solutions, bool allow_rescue,
    int refinement_max_iters) {
  const ssik::JointConsts<6> c = make_consts_n<6>(axes, t_left, t_right, types);
  const ssik::RrConsts rr =
      make_rr_consts(alpha, a, d, theta_offset, t_pre_inv, t_post_inv, linearity_joint,
                     left_bilinear, right_bilinear, drop_joint);
  const ssik::rr_detail::RrCoeffTensor tensor =
      make_rr_tensor(p_sin, p_cos, mono_factors, po_rc, po_mono, po_coeff, q_rc, q_mono, q_coeff);
  auto coeff_fn = [&tensor](const double t12[12], ssik::rr_detail::Mat14x9& ps,
                            ssik::rr_detail::Mat14x9& pc, ssik::rr_detail::Mat14x9& po,
                            ssik::rr_detail::Mat14x8& q) {
    ssik::rr_detail::PqCoeffs pq;
    ssik::rr_detail::rr_eval_coeffs(tensor, t12, pq);
    ps = pq.p_sin;
    pc = pq.p_cos;
    po = pq.p_one;
    q = pq.q;
  };

  ssik::JointLimits<6> lim;
  auto lo_u = lo.unchecked<1>(), hi_u = hi.unchecked<1>();
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

  const std::vector<ssik::Solution<6>> sols =
      ssik::general_6r_artifact_solve(c, rr, coeff_fn, lim, T, p);
  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 6});
  py::array_t<double> resids(n);
  py::array_t<int> refine(n);
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

// HP pencil eigensolve parity (#538): given f (9x7) and g (6x5), return the
// sorted finite real candidate u roots of det S(u)=0, as
// solve_pencil_eigenvalues / _pencil.solve_polynomial_matrix_eigenvalues.
py::array_t<double> hp_pencil_roots_test_py(py::array_t<double> f_arr, py::array_t<double> g_arr,
                                            double real_tol, double max_magnitude) {
  auto fm = f_arr.unchecked<2>();  // (9, 7)
  auto gm = g_arr.unchecked<2>();  // (6, 5)
  ssik::hp_detail::Mat9x7 f;
  ssik::hp_detail::Mat6x5 g;
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 7; ++j) f(i, j) = fm(i, j);
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 5; ++j) g(i, j) = gm(i, j);
  const std::vector<double> roots =
      ssik::hp_detail::solve_pencil_eigenvalues(f, g, real_tol, max_magnitude);
  py::array_t<double> out(static_cast<py::ssize_t>(roots.size()));
  auto om = out.mutable_unchecked<1>();
  for (std::size_t i = 0; i < roots.size(); ++i) om(static_cast<py::ssize_t>(i)) = roots[i];
  return out;
}

// HP (u,w) refinement parity (#539): given baked T_u/T_w_pre + sigma_E, return
// the refined (u,w) pairs, as _eliminate.eliminate_uw_pairs.
py::array_t<double> hp_eliminate_uw_pairs_test_py(py::array_t<double> t_u_arr,
                                                  py::array_t<double> t_w_pre_arr,
                                                  py::array_t<double> sigma_e_arr,
                                                  double accept_residue_tol) {
  auto load = [](py::array_t<double> a, std::array<Eigen::Matrix<double, 4, 8>, 2>& dst) {
    auto u = a.unchecked<3>();
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 8; ++j)
        for (int k = 0; k < 2; ++k) dst[k](i, j) = u(i, j, k);
  };
  ssik::HpConsts hp;
  load(t_u_arr, hp.t_u);
  load(t_w_pre_arr, hp.t_w_pre);
  auto se = sigma_e_arr.unchecked<1>();
  ssik::Vec8 sigma_E;
  for (int i = 0; i < 8; ++i) sigma_E[i] = se(i);
  const auto pairs = ssik::hp_detail::eliminate_uw_pairs(hp, sigma_E, {7, 4, 0}, accept_residue_tol);
  py::array_t<double> out({static_cast<py::ssize_t>(pairs.size()), py::ssize_t{2}});
  auto om = out.mutable_unchecked<2>();
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    om(static_cast<py::ssize_t>(i), 0) = pairs[i][0];
    om(static_cast<py::ssize_t>(i), 1) = pairs[i][1];
  }
  return out;
}

// HP back-substitution parity (#539): given baked T_u/T_w_pre + sigma_E + a
// refined (u,w) + the DH + dispatch flags, return the (v_1..v_6) tuple, as
// _back_substitute.back_substitute_one (Tv1 left).
py::array_t<double> hp_back_substitute_test_py(py::array_t<double> t_u_arr,
                                               py::array_t<double> t_w_pre_arr,
                                               py::array_t<double> sigma_e_arr, double u, double w,
                                               py::array_t<double> dh_a, py::array_t<double> dh_l,
                                               py::array_t<double> dh_d, int right_parametric_var,
                                               int drop_idx) {
  auto load = [](py::array_t<double> a, std::array<Eigen::Matrix<double, 4, 8>, 2>& dst) {
    auto u3 = a.unchecked<3>();
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 8; ++j)
        for (int k = 0; k < 2; ++k) dst[k](i, j) = u3(i, j, k);
  };
  ssik::HpConsts hp;
  load(t_u_arr, hp.t_u);
  load(t_w_pre_arr, hp.t_w_pre);
  hp.drop_idx = drop_idx;
  hp.right_parametric_var = right_parametric_var;
  auto av = dh_a.unchecked<1>(), lv = dh_l.unchecked<1>(), dv = dh_d.unchecked<1>();
  for (int i = 0; i < 5; ++i) {  // dh_a/dh_l = [a_1..a_5] -> hp.a[1..5]
    hp.a[i + 1] = av(i);
    hp.l[i + 1] = lv(i);
  }
  for (int i = 0; i < 4; ++i) hp.d[i + 2] = dv(i);  // dh_d = [d_2..d_5] -> hp.d[2..5]
  auto se = sigma_e_arr.unchecked<1>();
  ssik::Vec8 sigma_E;
  for (int i = 0; i < 8; ++i) sigma_E[i] = se(i);

  const ssik::hp_detail::JointTuple v =
      ssik::hp_detail::back_substitute_one(hp, sigma_E, u, w);
  py::array_t<double> out(6);
  auto om = out.mutable_unchecked<1>();
  for (int i = 0; i < 6; ++i) om(i) = v[i];
  return out;
}

// HP solve_ik parity (#539): baked T_u/T_w_pre + sigma_E + DH + flags -> the
// (n,6) v-tuple candidates, as _back_substitute.solve_ik.
py::array_t<double> hp_solve_ik_test_py(py::array_t<double> t_u_arr, py::array_t<double> t_w_pre_arr,
                                        py::array_t<double> sigma_e_arr, py::array_t<double> dh_a,
                                        py::array_t<double> dh_l, py::array_t<double> dh_d,
                                        int right_parametric_var) {
  auto load = [](py::array_t<double> a, std::array<Eigen::Matrix<double, 4, 8>, 2>& dst) {
    auto u3 = a.unchecked<3>();
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 8; ++j)
        for (int k = 0; k < 2; ++k) dst[k](i, j) = u3(i, j, k);
  };
  ssik::HpConsts hp;
  load(t_u_arr, hp.t_u);
  load(t_w_pre_arr, hp.t_w_pre);
  hp.right_parametric_var = right_parametric_var;
  auto av = dh_a.unchecked<1>(), lv = dh_l.unchecked<1>(), dv = dh_d.unchecked<1>();
  for (int i = 0; i < 5; ++i) {
    hp.a[i + 1] = av(i);
    hp.l[i + 1] = lv(i);
  }
  for (int i = 0; i < 4; ++i) hp.d[i + 2] = dv(i);
  auto se = sigma_e_arr.unchecked<1>();
  ssik::Vec8 sigma_E;
  for (int i = 0; i < 8; ++i) sigma_E[i] = se(i);

  const auto sols = ssik::hp_detail::solve_ik(hp, sigma_E);
  py::array_t<double> out({static_cast<py::ssize_t>(sols.size()), py::ssize_t{6}});
  auto om = out.mutable_unchecked<2>();
  for (std::size_t i = 0; i < sols.size(); ++i)
    for (int j = 0; j < 6; ++j) om(static_cast<py::ssize_t>(i), j) = sols[i][j];
  return out;
}

// HP full artifact solve (#539): JointConsts (POE FK) + baked HpConsts + target
// -> (qs, resids), exactly what ssik::hp_artifact_solve ships.
py::tuple hp_artifact_solve_test_py(py::array_t<double> axes, py::array_t<double> t_left,
                                    py::array_t<double> t_right, py::array_t<int> types,
                                    py::array_t<double> t_u_arr, py::array_t<double> t_w_pre_arr,
                                    py::array_t<double> dh_a, py::array_t<double> dh_l,
                                    py::array_t<double> dh_d, py::array_t<double> theta_offset,
                                    py::array_t<double> t_pre_inv, py::array_t<double> t_post_inv,
                                    py::array_t<double> t_z_neg_d1,
                                    py::array_t<double> t_joint6_offset_inv, int right_parametric_var,
                                    int drop_idx, py::array_t<double> target, bool allow_refinement) {
  const ssik::JointConsts<6> c = make_consts_n<6>(axes, t_left, t_right, types);
  ssik::HpConsts hp;
  auto load = [](py::array_t<double> a, std::array<Eigen::Matrix<double, 4, 8>, 2>& dst) {
    auto u3 = a.unchecked<3>();
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 8; ++j)
        for (int k = 0; k < 2; ++k) dst[k](i, j) = u3(i, j, k);
  };
  load(t_u_arr, hp.t_u);
  load(t_w_pre_arr, hp.t_w_pre);
  auto mat4 = [](py::array_t<double> a) {
    auto u = a.unchecked<2>();
    Eigen::Matrix4d m;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) m(i, j) = u(i, j);
    return m;
  };
  hp.t_pre_inv = mat4(t_pre_inv);
  hp.t_post_inv = mat4(t_post_inv);
  hp.t_z_neg_d1 = mat4(t_z_neg_d1);
  hp.t_joint6_offset_inv = mat4(t_joint6_offset_inv);
  hp.right_parametric_var = right_parametric_var;
  hp.drop_idx = drop_idx;
  auto av = dh_a.unchecked<1>(), lv = dh_l.unchecked<1>(), dv = dh_d.unchecked<1>();
  auto to = theta_offset.unchecked<1>();
  for (int i = 0; i < 5; ++i) {
    hp.a[i + 1] = av(i);
    hp.l[i + 1] = lv(i);
  }
  for (int i = 0; i < 4; ++i) hp.d[i + 2] = dv(i);
  for (int i = 0; i < 6; ++i) hp.theta_offset[i] = to(i);

  ssik::Pose T;
  auto tm = target.unchecked<2>();
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) T(i, j) = tm(i, j);

  ssik::JointLimits<6> lim;  // no limits for the parity test
  lim.present = {false, false, false, false, false, false};
  (void)allow_refinement;  // HP always lm_refines seeds that miss (like general_6r force_refine)
  ssik::ArtifactParams<6> p;
  p.respect_limits = false;
  p.allow_rescue = false;
  const std::vector<ssik::Solution<6>> sols = ssik::hp_artifact_solve(c, hp, lim, T, p);
  const int n = static_cast<int>(sols.size());
  py::array_t<double> qs({n, 6});
  py::array_t<double> resids(n);
  auto qm = qs.mutable_unchecked<2>();
  auto rm = resids.mutable_unchecked<1>();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 6; ++j) qm(i, j) = sols[i].q[j];
    rm(i) = sols[i].fk_residual;
  }
  return py::make_tuple(qs, resids);
}

PYBIND11_MODULE(_ssik_native, m) {
  m.doc() = "Native three_parallel solver binding (test conformance + shipped native backend)";
  m.def("decompose_3axis_test", &decompose_3axis_test_py, py::arg("R"), py::arg("n1"),
        py::arg("n2"), py::arg("n3"));
  m.def("hp_compute_fg_test", &hp_compute_fg_test_py, py::arg("t_u"), py::arg("t_w_pre"),
        py::arg("sigma_e"), py::arg("drop_idx") = 7);
  m.def("rr_eval_coeffs_test", &rr_eval_coeffs_test_py, py::arg("p_sin"), py::arg("p_cos"),
        py::arg("mono_factors"), py::arg("po_rc"), py::arg("po_mono"), py::arg("po_coeff"),
        py::arg("q_rc"), py::arg("q_mono"), py::arg("q_coeff"), py::arg("t12"));
  m.def("general_6r_tensor_solve", &general_6r_tensor_solve_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("alpha"), py::arg("a"), py::arg("d"),
        py::arg("theta_offset"), py::arg("t_pre_inv"), py::arg("t_post_inv"),
        py::arg("linearity_joint"), py::arg("left_bilinear"), py::arg("right_bilinear"),
        py::arg("drop_joint"), py::arg("p_sin"), py::arg("p_cos"), py::arg("mono_factors"),
        py::arg("po_rc"), py::arg("po_mono"), py::arg("po_coeff"), py::arg("q_rc"), py::arg("q_mono"),
        py::arg("q_coeff"), py::arg("target"));
  m.def("general_6r_tensor_artifact_solve", &general_6r_tensor_artifact_solve_py, py::arg("axes"),
        py::arg("t_left"), py::arg("t_right"), py::arg("types"), py::arg("lo"), py::arg("hi"),
        py::arg("has_limits"), py::arg("alpha"), py::arg("a"), py::arg("d"), py::arg("theta_offset"),
        py::arg("t_pre_inv"), py::arg("t_post_inv"), py::arg("linearity_joint"),
        py::arg("left_bilinear"), py::arg("right_bilinear"), py::arg("drop_joint"), py::arg("p_sin"),
        py::arg("p_cos"), py::arg("mono_factors"), py::arg("po_rc"), py::arg("po_mono"),
        py::arg("po_coeff"), py::arg("q_rc"), py::arg("q_mono"), py::arg("q_coeff"),
        py::arg("target"), py::arg("respect_limits"), py::arg("has_seed"), py::arg("q_seed"),
        py::arg("seed_metric"), py::arg("has_seed_tolerance"), py::arg("seed_tolerance"),
        py::arg("max_solutions"), py::arg("allow_rescue"), py::arg("refinement_max_iters"));
  m.def("hp_pencil_roots_test", &hp_pencil_roots_test_py, py::arg("f"), py::arg("g"),
        py::arg("real_tol") = 1e-3, py::arg("max_magnitude") = 1e10);
  m.def("hp_eliminate_uw_pairs_test", &hp_eliminate_uw_pairs_test_py, py::arg("t_u"),
        py::arg("t_w_pre"), py::arg("sigma_e"), py::arg("accept_residue_tol") = 1e-3);
  m.def("hp_back_substitute_test", &hp_back_substitute_test_py, py::arg("t_u"), py::arg("t_w_pre"),
        py::arg("sigma_e"), py::arg("u"), py::arg("w"), py::arg("dh_a"), py::arg("dh_l"),
        py::arg("dh_d"), py::arg("right_parametric_var"), py::arg("drop_idx") = 7);
  m.def("hp_solve_ik_test", &hp_solve_ik_test_py, py::arg("t_u"), py::arg("t_w_pre"),
        py::arg("sigma_e"), py::arg("dh_a"), py::arg("dh_l"), py::arg("dh_d"),
        py::arg("right_parametric_var"));
  m.def("hp_artifact_solve_test", &hp_artifact_solve_test_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("t_u"), py::arg("t_w_pre"), py::arg("dh_a"),
        py::arg("dh_l"), py::arg("dh_d"), py::arg("theta_offset"), py::arg("t_pre_inv"),
        py::arg("t_post_inv"), py::arg("t_z_neg_d1"), py::arg("t_joint6_offset_inv"),
        py::arg("right_parametric_var"), py::arg("drop_idx"), py::arg("target"),
        py::arg("allow_refinement") = true);
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
