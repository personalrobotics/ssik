// Solve IK for a UR-class arm at runtime with the header-only three_parallel
// solver, from joint data that ssik.cpp.joint_data() produced -- no generated
// per-arm header, no Python. check.py writes the input and compares the output
// with ssik's Python solve().
//
//   three_parallel_solve <input> <output>
//
// input (whitespace-separated numbers, the fields of ssik.cpp.JointData):
//   dof (must be 6), then per joint: axis (3), t_left (16, row-major),
//   t_right (16, row-major), type (0 revolute / 1 prismatic);
//   then lo (6), hi (6), present (6, 0/1);
//   then the pose count K and K row-major 4x4 target poses (16 each).
// output: per pose, the solution count n, then n lines of 6 joint values.
#include <cstdio>
#include <fstream>
#include <iostream>

#include "ssik_cpp/solvers/three_parallel.hpp"

namespace {

bool read_mat4(std::istream& in, Eigen::Matrix4d& m) {
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      if (!(in >> m(r, c))) return false;
  return true;
}

bool read_input(std::istream& in, ssik::JointConsts<6>& c, ssik::JointLimits<6>& lim,
                std::vector<ssik::Pose>& poses) {
  int dof = 0;
  if (!(in >> dof) || dof != 6) return false;
  for (int i = 0; i < 6; ++i) {
    for (int k = 0; k < 3; ++k)
      if (!(in >> c.axis[i][k])) return false;
    if (!read_mat4(in, c.t_left[i]) || !read_mat4(in, c.t_right[i])) return false;
    int type = 0;
    if (!(in >> type) || (type != 0 && type != 1)) return false;
    c.type[i] = type == 0 ? ssik::JointType::Revolute : ssik::JointType::Prismatic;
  }
  for (double& v : lim.lo)
    if (!(in >> v)) return false;
  for (double& v : lim.hi)
    if (!(in >> v)) return false;
  for (int i = 0; i < 6; ++i) {
    int present = 0;
    if (!(in >> present)) return false;
    lim.present[i] = present != 0;
  }
  int n_poses = 0;
  if (!(in >> n_poses) || n_poses < 0) return false;
  poses.resize(n_poses);
  for (auto& T : poses)
    if (!read_mat4(in, T)) return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <input> <output>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  ssik::JointConsts<6> consts;
  ssik::JointLimits<6> limits;
  std::vector<ssik::Pose> poses;
  if (!in || !read_input(in, consts, limits, poses)) {
    std::cerr << "malformed input: " << argv[1] << "\n";
    return 2;
  }

  FILE* out = std::fopen(argv[2], "w");
  if (out == nullptr) {
    std::cerr << "cannot write " << argv[2] << "\n";
    return 2;
  }
  const ssik::ArtifactParams<6> params;  // the Python solve() defaults
  for (const auto& T : poses) {
    const auto sols = ssik::three_parallel_artifact_solve(consts, limits, T, params);
    std::fprintf(out, "%zu\n", sols.size());
    for (const auto& s : sols) {
      for (int i = 0; i < 6; ++i) std::fprintf(out, i ? " %.17g" : "%.17g", s.q[i]);
      std::fprintf(out, "\n");
    }
  }
  return std::fclose(out) == 0 ? 0 : 1;
}
