// Rodrigues rotation primitive (#486), ported from ssik.subproblems._rotation.
// Element order matches the Python scalarized form exactly so the native
// subproblem hot path is bit-identical to the oracle.
#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace ssik {

// rotate(k, theta, v) = v cos + (k x v) sin + k (k . v)(1 - cos), assuming |k| = 1.
inline Eigen::Vector3d rotate(const Eigen::Vector3d& k, double theta, const Eigen::Vector3d& v) {
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  const double kx = k[0], ky = k[1], kz = k[2];
  const double vx = v[0], vy = v[1], vz = v[2];
  const double m = (1.0 - c) * (kx * vx + ky * vy + kz * vz);
  return {
      vx * c + (ky * vz - kz * vy) * s + kx * m,
      vy * c + (kz * vx - kx * vz) * s + ky * m,
      vz * c + (kx * vy - ky * vx) * s + kz * m,
  };
}

// 3x3 rotation about unit `axis` by `angle` (Rodrigues), element order matching
// ssik.subproblems._rotation.rotation_matrix exactly.
inline Eigen::Matrix3d rotation_matrix(const Eigen::Vector3d& axis, double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double x = axis[0], y = axis[1], z = axis[2];
  const double oc = 1.0 - c;
  Eigen::Matrix3d r;
  r << c + x * x * oc, x * y * oc - z * s, x * z * oc + y * s,
      y * x * oc + z * s, c + y * y * oc, y * z * oc - x * s,
      z * x * oc - y * s, z * y * oc + x * s, c + z * z * oc;
  return r;
}

}  // namespace ssik
