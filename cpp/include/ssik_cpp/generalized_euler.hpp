// Generalized Euler / Davenport decomposition (#354), ported from
// ssik.kinematics._generalized_euler.decompose_3axis. Finds up to two (a, b, c)
// with R == Rot(n1,a) @ Rot(n2,b) @ Rot(n3,c) for ARBITRARY unit axes (classical
// ZYZ Euler is the n1==n3 special case). This is the shoulder/wrist extraction
// the general SRS solve uses to cover non-ZYZ concurrent-axis 7R arms (R1 Pro,
// OpenArm, iiwa7-class offset wrists) that the canonical ZYZ path can't.
//
// n1 . R n3 == n1 . Rot(n2,b) n3 expands to A cos b + B sin b == C (one sinusoid
// in b -> up to two elbow-style branches); per branch, a rotates Rot(n2,b) n3
// onto R n3 about n1, and c rotates R^T n1 onto Rot(n2,b)^T n1 about n3.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "ssik_cpp/rotation.hpp"  // rotation_matrix

namespace ssik {
namespace geuler {

// Amplitude sqrt(A^2+B^2) below this: axes (near-)collinear, no decomposition.
inline constexpr double kDegenerateAmplitude = 1e-12;
// Perpendicular component below this is a gimbal: the outer angle is pinned to 0.
inline constexpr double kGimbalEps = 1e-9;

struct Abc {
  double a, b, c;
};

// Some unit vector perpendicular to axis (_unit_perp).
inline Eigen::Vector3d unit_perp(const Eigen::Vector3d& axis) {
  const Eigen::Vector3d ref =
      std::abs(axis[0]) < 0.9 ? Eigen::Vector3d(1, 0, 0) : Eigen::Vector3d(0, 1, 0);
  return (ref - axis * axis.dot(ref)).normalized();
}

// Rotation about axis carrying v_from's perp component onto v_to's (_angle_about);
// 0 when either perp component vanishes (gimbal).
inline double angle_about(const Eigen::Vector3d& axis, const Eigen::Vector3d& v_from,
                          const Eigen::Vector3d& v_to) {
  const Eigen::Vector3d vf = v_from - axis * axis.dot(v_from);
  const Eigen::Vector3d vt = v_to - axis * axis.dot(v_to);
  if (vf.norm() < kGimbalEps || vt.norm() < kGimbalEps) return 0.0;
  return std::atan2(axis.dot(vf.cross(vt)), vf.dot(vt));
}

// Decompose R into Rot(u1,a) Rot(u2,b) Rot(u3,c). Up to two (a,b,c) triples (one
// at the b-branch boundary); empty if the axes are (near-)collinear.
inline std::vector<Abc> decompose_3axis(const Eigen::Matrix3d& R, Eigen::Vector3d u1,
                                        Eigen::Vector3d u2, Eigen::Vector3d u3) {
  u1.normalize();
  u2.normalize();
  u3.normalize();

  const double a_coef = u1.dot(u3) - u1.dot(u2) * u2.dot(u3);
  const double b_coef = u1.dot(u2.cross(u3));
  const double c_const = u1.dot(R * u3) - u1.dot(u2) * u2.dot(u3);
  const double amplitude = std::hypot(a_coef, b_coef);
  if (amplitude < kDegenerateAmplitude) return {};

  const double phi = std::atan2(b_coef, a_coef);
  const double delta = std::acos(std::clamp(c_const / amplitude, -1.0, 1.0));
  std::vector<double> b_branches;
  if (delta < 1e-12)
    b_branches = {phi + delta};
  else
    b_branches = {phi + delta, phi - delta};

  const Eigen::Vector3d r_n3 = R * u3;
  const Eigen::Vector3d rt_n1 = R.transpose() * u1;
  std::vector<Abc> out;
  for (double b : b_branches) {
    const Eigen::Matrix3d rb = rotation_matrix(u2, b);
    const Eigen::Vector3d vf = rb * u3;
    double a, c;
    if ((vf - u1 * u1.dot(vf)).norm() < kGimbalEps) {
      a = 0.0;
      const Eigen::Matrix3d residual = rotation_matrix(u2, -b) * R;
      const Eigen::Vector3d ref = unit_perp(u3);
      c = angle_about(u3, ref, residual * ref);
    } else {
      a = angle_about(u1, vf, r_n3);
      c = angle_about(u3, rt_n1, rb.transpose() * u1);
    }
    out.push_back({a, b, c});
  }
  return out;
}

}  // namespace geuler
}  // namespace ssik
