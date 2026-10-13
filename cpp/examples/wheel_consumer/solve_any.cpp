// Solve IK for any arm with a native solver through ssik::Solver, from the
// solver data ssik.cpp.solver_data() produced -- one program for every family,
// no generated per-arm header, no Python. check.py writes the inputs and
// compares the output with ssik's in-process native solve.
//
//   solve_any <data> <cases> <output>
//
// data (whitespace-separated tokens, an ssik::SolverData): the family, the
// schema number and the array count; then per array its name, dtype (f64 or
// i32), rank, extents, and the values in C order.
// cases: the case count K, then per case a row-major 4x4 target (16 numbers),
// has_seed (0/1), the dof seed values when has_seed, and max_solutions.
// output: per case the solution count n, then n lines of the dof joint values
// followed by max |Solver::fk(q) - T| over the 16 entries.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "ssik_cpp/solve.hpp"

namespace {

bool read_solver_data(std::istream& in, ssik::SolverData& d) {
  int n_arrays = 0;
  if (!(in >> d.family >> d.schema >> n_arrays) || n_arrays < 0) return false;
  for (int k = 0; k < n_arrays; ++k) {
    std::string name, dtype;
    int rank = 0;
    if (!(in >> name >> dtype >> rank) || rank < 0) return false;
    ssik::DataArray a;
    std::int64_t n = 1;
    a.shape.resize(rank);
    for (auto& e : a.shape) {
      if (!(in >> e) || e < 0) return false;
      n *= e;
    }
    if (dtype == "f64") {
      a.dtype = ssik::DType::Float64;
      a.f64.resize(static_cast<std::size_t>(n));
      for (auto& v : a.f64)
        if (!(in >> v)) return false;
    } else if (dtype == "i32") {
      a.dtype = ssik::DType::Int32;
      a.i32.resize(static_cast<std::size_t>(n));
      for (auto& v : a.i32)
        if (!(in >> v)) return false;
    } else {
      return false;
    }
    if (!d.arrays.emplace(name, std::move(a)).second) return false;
  }
  return true;
}

template <int N>
bool run(const ssik::Solver& solver, std::istream& cases, FILE* out) {
  int k = 0;
  if (!(cases >> k) || k < 0) return false;
  for (int c = 0; c < k; ++c) {
    ssik::Pose T;
    for (int r = 0; r < 4; ++r)
      for (int col = 0; col < 4; ++col)
        if (!(cases >> T(r, col))) return false;
    ssik::ArtifactParams<N> p;  // the Python solve() defaults
    int has_seed = 0;
    if (!(cases >> has_seed)) return false;
    p.has_seed = has_seed != 0;
    if (p.has_seed)
      for (auto& v : p.q_seed)
        if (!(cases >> v)) return false;
    if (!(cases >> p.max_solutions)) return false;

    const auto sols = solver.solve<N>(T, p);
    std::fprintf(out, "%zu\n", sols.size());
    for (const auto& s : sols) {
      const ssik::Pose F = solver.fk<N>(s.q);
      const double fk_err = (F - T).cwiseAbs().maxCoeff();
      for (int i = 0; i < N; ++i) std::fprintf(out, "%.17g ", s.q[i]);
      std::fprintf(out, "%.17g\n", fk_err);
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0] << " <data> <cases> <output>\n";
    return 2;
  }
  std::ifstream data_in(argv[1]);
  ssik::SolverData data;
  if (!data_in || !read_solver_data(data_in, data)) {
    std::cerr << "malformed solver data: " << argv[1] << "\n";
    return 2;
  }
  try {
    const ssik::Solver solver(data);  // checks the data against the schema
    std::ifstream cases(argv[2]);
    FILE* out = std::fopen(argv[3], "w");
    if (out == nullptr) {
      std::cerr << "cannot write " << argv[3] << "\n";
      return 2;
    }
    const bool ok = solver.dof() == 6 ? run<6>(solver, cases, out) : run<7>(solver, cases, out);
    if (std::fclose(out) != 0 || !ok) {
      std::cerr << "malformed cases or failed write: " << argv[2] << "\n";
      return 2;
    }
  } catch (const ssik::SolverDataError& e) {
    std::cerr << "rejected solver data: " << e.what() << "\n";
    return 3;
  }
  return 0;
}
