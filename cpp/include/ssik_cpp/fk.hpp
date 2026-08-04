// POE forward kinematics + the per-arm baked kinematic constants (#486).
//
// Mirrors ssik.kinematics.poe_fk.poe_forward_kinematics exactly:
//   T = product_i  T_left[i] * Joint(axis[i], q[i]) * T_right[i]
// where Joint is a Rodrigues rotation (revolute) or a translation along the
// axis (prismatic). The constants are emitted per arm from the one KinBody
// (Band A); this walk is the shared primitive (Band C).
#pragma once

#include <array>

#include <Eigen/Dense>

#include "ssik_cpp/ik_types.hpp"

namespace ssik {

enum class JointType : std::uint8_t { Revolute, Prismatic };

// Baked kinematics for an N-DOF arm: per-joint axis, the fixed T_left/T_right
// frames straddling the joint, and the joint type. Rendered from the KinBody.
template <int N>
struct JointConsts {
  std::array<Eigen::Vector3d, N> axis;
  std::array<Eigen::Matrix4d, N> t_left;
  std::array<Eigen::Matrix4d, N> t_right;
  std::array<JointType, N> type;
};

// base->end pose at config q. Byte-for-byte the same accumulation order as the
// Python reference so FK-closure conformance compares like with like.
template <int N>
Pose fk(const JointConsts<N>& c, const std::array<double, N>& q) {
  Pose t = Pose::Identity();
  for (int i = 0; i < N; ++i) {
    Eigen::Matrix4d joint = Eigen::Matrix4d::Identity();
    const Eigen::Vector3d u = c.axis[i].normalized();  // POE axes are unit; guard anyway
    if (c.type[i] == JointType::Revolute) {
      joint.template block<3, 3>(0, 0) = Eigen::AngleAxisd(q[i], u).toRotationMatrix();
    } else {
      joint.template block<3, 1>(0, 3) = q[i] * u;
    }
    t = t * c.t_left[i] * joint * c.t_right[i];
  }
  return t;
}

// Partial FK: the world frame (rotation R, origin p) at joint k BEFORE its own
// rotation, given config q. Mirrors ssik.solvers.seven_r.srs._frame_at_joint
// exactly (T_left contributes translation only -- rotation block is I in a
// POE-normalized chain -- and T_right's rotation is applied after the joint).
template <int N>
std::pair<Eigen::Matrix3d, Eigen::Vector3d> frame_at_joint(const JointConsts<N>& c,
                                                           const std::array<double, N>& q, int k) {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  for (int i = 0; i < N; ++i) {
    p = p + R * c.t_left[i].template block<3, 1>(0, 3);
    if (i == k) return {R, p};
    R = R * Eigen::AngleAxisd(q[i], c.axis[i].normalized()).toRotationMatrix();
    R = R * c.t_right[i].template block<3, 3>(0, 0);
    p = p + R * c.t_right[i].template block<3, 1>(0, 3);
  }
  return {R, p};
}

}  // namespace ssik
