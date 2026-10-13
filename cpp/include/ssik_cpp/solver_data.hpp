// The complete C++ input of one arm's native solver, as plain named arrays
// (#707). Eigen-free and solver-free, so a consumer's public header or pybind11
// translation unit can hold one without pulling in the solvers.
//
// A SolverData is a family name (the solver name, for example
// "ikgeo.three_parallel"), a schema number and a flat map from array name to a
// float64 or int32 array in C order. ssik::Solver (ssik_cpp/solve.hpp) checks
// it against the schema and prepares the family's solver from it.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ssik {

// The schema this release reads. ssik::Solver accepts exactly this number.
inline constexpr int kSolverDataSchema = 1;

enum class DType : std::uint8_t { Float64, Int32 };

// One array: its dtype, its shape in C order ({} is a scalar) and its values.
// `f64` holds the values when dtype is Float64, `i32` when it is Int32; the
// other vector is empty.
struct DataArray {
  DType dtype = DType::Float64;
  std::vector<std::int64_t> shape;
  std::vector<double> f64;
  std::vector<std::int32_t> i32;
};

struct SolverData {
  int schema = kSolverDataSchema;
  std::string family;
  std::map<std::string, DataArray> arrays;
};

}  // namespace ssik
