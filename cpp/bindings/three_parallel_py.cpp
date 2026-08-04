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
#include "ssik_cpp/solvers/spherical_two_parallel.hpp"
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
PYBIND11_MODULE(_ssik_native, m) {
  m.doc() = "Native three_parallel solver binding (test conformance + shipped native backend)";
  m.def("three_parallel_solve", &three_parallel_solve_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("target"), py::arg("allow_refinement") = false,
        py::arg("refinement_max_iters") = 15);
  m.def("spherical_two_parallel_solve", &spherical_two_parallel_solve_py, py::arg("axes"),
        py::arg("t_left"), py::arg("t_right"), py::arg("types"), py::arg("target"),
        py::arg("allow_refinement") = false, py::arg("refinement_max_iters") = 15);
  m.def("srs_canonical_solve", &srs_canonical_solve_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("l_se"), py::arg("l_ew"), py::arg("ee_offset"),
        py::arg("shoulder_pivot"), py::arg("r_post_wrist"), py::arg("target"));
  m.def("native_artifact_solve", &native_artifact_solve_py, py::arg("family"), py::arg("axes"),
        py::arg("t_left"), py::arg("t_right"), py::arg("types"), py::arg("lo"), py::arg("hi"),
        py::arg("has_limits"), py::arg("target"), py::arg("respect_limits") = true,
        py::arg("has_seed") = false, py::arg("q_seed"), py::arg("seed_metric") = "wrap_linf",
        py::arg("has_seed_tolerance") = false, py::arg("seed_tolerance") = 0.0,
        py::arg("max_solutions") = -1, py::arg("allow_rescue") = true,
        py::arg("refinement_max_iters") = 15);
}
