// ssik native artifact ABI (#485, sub-issue of #482).
//
// The stable C++ surface every native arm artifact exposes. Pure types +
// Eigen; no solver logic here. Header-only.
#pragma once

#include <array>
#include <cstdint>

#include <Eigen/Dense>

namespace ssik {

// Target end-effector pose in the arm's base frame: a 4x4 homogeneous SE(3)
// transform, laid out identically to the Python KinBody FK output.
using Pose = Eigen::Matrix4d;

// How a returned solution was produced -- mirrors Python's
// Solution.refinement_used so behavioral-parity conformance can compare tags.
enum class Refinement : std::uint8_t { None, Lm, Rescue };

// One IK solution for an N-DOF arm: the joint vector plus the FK-closure
// residual (Frobenius norm of fk(q) - target) and the provenance tag.
template <int N>
struct Solution {
  std::array<double, N> q{};
  double fk_residual = 0.0;
  Refinement refinement = Refinement::None;
};

// Solve knobs, mirroring the Python solve() signature so behavioral parity is
// checkable. A default-constructed SolveOptions is the "give me every in-limits
// analytical solution" case.
template <int N>
struct SolveOptions {
  bool respect_limits = true;
  bool has_seed = false;              // when true, rank solutions nearest q_seed
  std::array<double, N> q_seed{};
  int max_solutions = -1;             // -1 = no cap
  bool allow_refinement = false;      // LM polish (polished solver families)
  bool allow_rescue = false;          // T-perturbation rescue at ridge poses
};

// Status of a solve. Non-throwing on the hot path.
enum class Status : std::uint8_t {
  Ok,          // >=1 solution written
  Unreachable, // target outside the workspace (no analytical solution)
  Truncated,   // more solutions existed than the output span could hold
};

// Result of the allocation-free solveInto(): how many solutions were written to
// the caller-owned span, and whether the set was truncated to fit.
struct SolveResult {
  int count = 0;
  Status status = Status::Unreachable;
};

}  // namespace ssik
