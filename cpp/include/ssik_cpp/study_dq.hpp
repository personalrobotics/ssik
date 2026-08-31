// Study / dual-quaternion primitives for the Husty-Pfurner solver (#537).
//
// Ported scalar-for-scalar from ssik.solvers.husty_pfurner._study (+ the two
// left-mult matrices from _constraints), so the HP native path is bit-for-bit
// with the Python reference on this shared tier. A quaternion is a Vec4
// [q0,q1,q2,q3]; a dual quaternion is a Vec8 [p0..p3, q0..q3] = p + eps*q.
#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace ssik {

using Vec4 = Eigen::Matrix<double, 4, 1>;
using Vec8 = Eigen::Matrix<double, 8, 1>;
using Mat4 = Eigen::Matrix<double, 4, 4>;
using Mat8 = Eigen::Matrix<double, 8, 8>;

namespace study {

// Hamilton product p * q (_study._quat_mul).
inline Vec4 quat_mul(const Vec4& p, const Vec4& q) {
  const double p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
  const double q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];
  Vec4 r;
  r[0] = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3;
  r[1] = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2;
  r[2] = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1;
  r[3] = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0;
  return r;
}

// Quaternion conjugate (_study._quat_conj).
inline Vec4 quat_conj(const Vec4& p) { return Vec4(p[0], -p[1], -p[2], -p[3]); }

// Shepperd (1978) unit quaternion from a 3x3 rotation (_study._quat_from_rot),
// picking the numerically dominant branch.
inline Vec4 quat_from_rot(const Eigen::Matrix3d& R) {
  const double m00 = R(0, 0), m01 = R(0, 1), m02 = R(0, 2);
  const double m10 = R(1, 0), m11 = R(1, 1), m12 = R(1, 2);
  const double m20 = R(2, 0), m21 = R(2, 1), m22 = R(2, 2);
  const double tr = m00 + m11 + m22;
  double q0, q1, q2, q3;
  if (tr > 0.0) {
    const double s = std::sqrt(tr + 1.0) * 2.0;  // s = 4*q0
    q0 = 0.25 * s;
    q1 = (m21 - m12) / s;
    q2 = (m02 - m20) / s;
    q3 = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;  // s = 4*q1
    q0 = (m21 - m12) / s;
    q1 = 0.25 * s;
    q2 = (m01 + m10) / s;
    q3 = (m02 + m20) / s;
  } else if (m11 > m22) {
    const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;  // s = 4*q2
    q0 = (m02 - m20) / s;
    q1 = (m01 + m10) / s;
    q2 = 0.25 * s;
    q3 = (m12 + m21) / s;
  } else {
    const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;  // s = 4*q3
    q0 = (m10 - m01) / s;
    q1 = (m02 + m20) / s;
    q2 = (m12 + m21) / s;
    q3 = 0.25 * s;
  }
  return Vec4(q0, q1, q2, q3);
}

// SE(3) 4x4 -> 8-vec dual quaternion (_study.dq_from_se3).
// sigma = (p, (1/2) t_quat * p), p = unit quat of R, t_quat = (0, t).
inline Vec8 dq_from_se3(const Eigen::Matrix4d& T) {
  const Vec4 p = quat_from_rot(T.block<3, 3>(0, 0));
  const Vec4 t_quat(0.0, T(0, 3), T(1, 3), T(2, 3));
  const Vec4 q = 0.5 * quat_mul(t_quat, p);
  Vec8 sigma;
  sigma.head<4>() = p;
  sigma.tail<4>() = q;
  return sigma;
}

// 4x4 left-multiply matrix L(p): L(p) @ q == p * q (_constraints._quat_left_mult_matrix).
inline Mat4 quat_left_mult_matrix(const Vec4& p) {
  const double p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
  Mat4 L;
  L << p0, -p1, -p2, -p3,  //
      p1, p0, -p3, p2,     //
      p2, p3, p0, -p1,     //
      p3, -p2, p1, p0;
  return L;
}

// 8x8 left-multiply matrix M(eta): M @ sigma == eta * sigma (dq_mul), block
// structure [[Lp, 0], [Lq, Lp]] (_constraints._dq_left_mult_matrix).
inline Mat8 dq_left_mult_matrix(const Vec8& eta) {
  const Mat4 Lp = quat_left_mult_matrix(eta.head<4>());
  const Mat4 Lq = quat_left_mult_matrix(eta.tail<4>());
  Mat8 M = Mat8::Zero();
  M.block<4, 4>(0, 0) = Lp;
  M.block<4, 4>(4, 0) = Lq;
  M.block<4, 4>(4, 4) = Lp;
  return M;
}

}  // namespace study
}  // namespace ssik
