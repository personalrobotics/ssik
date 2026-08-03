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
#include "ssik_cpp/solvers/three_parallel.hpp"

namespace py = pybind11;

namespace {

// Build the native JointConsts<6> from the KinBody arrays marshalled by the
// Python adapter. Shapes: axes (6,3), t_left/t_right (6,4,4), types (6,).
ssik::JointConsts<6> make_consts(const py::array_t<double>& axes,
                                 const py::array_t<double>& t_left,
                                 const py::array_t<double>& t_right,
                                 const py::array_t<int>& types) {
  auto a = axes.unchecked<2>();
  auto tl = t_left.unchecked<3>();
  auto tr = t_right.unchecked<3>();
  auto ty = types.unchecked<1>();
  ssik::JointConsts<6> c;
  for (int i = 0; i < 6; ++i) {
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

}  // namespace

PYBIND11_MODULE(ssik_cpp_ext, m) {
  m.doc() = "Test-only native-solver binding for C++<->Python conformance (#499)";
  m.def("three_parallel_solve", &three_parallel_solve_py, py::arg("axes"), py::arg("t_left"),
        py::arg("t_right"), py::arg("types"), py::arg("target"), py::arg("allow_refinement") = false,
        py::arg("refinement_max_iters") = 15);
}
