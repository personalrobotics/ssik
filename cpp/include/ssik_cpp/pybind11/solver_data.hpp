// Convert a Python solver-data record into ssik::SolverData (#707), for a
// pybind11 extension that hands one to ssik::Solver. Eigen-free. It uses the
// including translation unit's own pybind11; the ssik::ssik_cpp target does not
// depend on pybind11.
//
// The record is any object with attributes `schema` (int), `family` (str) and
// `arrays` (a mapping from str to numpy array). Every array must be float64 or
// int32 in native byte order and C-contiguous: anything else raises TypeError,
// so no value is converted silently. The values are copied.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ssik_cpp/solver_data.hpp"

namespace ssik::pybind11_interop {

inline ssik::SolverData solver_data_from_python(pybind11::handle obj) {
  namespace py = pybind11;
  ssik::SolverData d;
  d.schema = obj.attr("schema").cast<int>();
  d.family = obj.attr("family").cast<std::string>();
  const py::object items = obj.attr("arrays").attr("items")();
  for (const py::handle item : items) {
    const auto pair = item.cast<py::tuple>();
    const auto name = pair[0].cast<std::string>();
    const py::handle value = pair[1];
    // array_t's isinstance compares the dtype by numpy's equivalence, which
    // also requires native byte order.
    const bool is_f64 = py::isinstance<py::array_t<double, py::array::c_style>>(value);
    const bool is_i32 = !is_f64 && py::isinstance<py::array_t<std::int32_t, py::array::c_style>>(value);
    if (!is_f64 && !is_i32) {
      if (!py::isinstance<py::array>(value))
        throw py::type_error("solver data array '" + name + "' must be a numpy array");
      if (py::isinstance<py::array_t<double>>(value) || py::isinstance<py::array_t<std::int32_t>>(value))
        throw py::type_error("solver data array '" + name + "' must be C-contiguous");
      throw py::type_error("solver data array '" + name + "' must be float64 or int32, got " +
                           py::str(value.attr("dtype")).cast<std::string>());
    }
    const auto arr = py::reinterpret_borrow<py::array>(value);
    ssik::DataArray a;
    a.dtype = is_f64 ? ssik::DType::Float64 : ssik::DType::Int32;
    a.shape.assign(arr.shape(), arr.shape() + arr.ndim());
    const auto n = static_cast<std::size_t>(arr.size());
    if (is_f64) {
      a.f64.resize(n);
      if (n) std::memcpy(a.f64.data(), arr.data(), n * sizeof(double));
    } else {
      a.i32.resize(n);
      if (n) std::memcpy(a.i32.data(), arr.data(), n * sizeof(std::int32_t));
    }
    d.arrays.emplace(name, std::move(a));
  }
  return d;
}

}  // namespace ssik::pybind11_interop
