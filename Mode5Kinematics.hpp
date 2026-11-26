#pragma once

#include "G1Body.hpp"
#include <hrp4_locomotion/RobotState.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

// The pelvis IMU provides attitude. Stance-foot odometry supplies translation
// under the no-slip assumption, starting at the operator-confirmed stance.
class Mode5Kinematics {
 public:
  Mode5Kinematics();
  bool levelAndQuiet(const labrob::RobotState& state);
  void registerGround(const labrob::RobotState& state);
  void locate(labrob::RobotState& state, int stance);
  std::array<double, 2> heights() const;
  double lower(int i) const { return model_.lowerPositionLimit[q_index_[i]]; }
  double upper(int i) const { return model_.upperPositionLimit[q_index_[i]]; }
  double effort(int i) const { return model_.effortLimit[q_index_[i]-1]; }
 private:
  void forward(const labrob::RobotState& state);
  pinocchio::Model model_;
  pinocchio::Data data_;
  std::array<int, 29> q_index_;
  std::array<pinocchio::FrameIndex, 2> feet_;
  std::array<Eigen::Vector3d, 2> anchors_;
  Eigen::Quaterniond heading_ = Eigen::Quaterniond::Identity();
  Eigen::Vector3d position_ = Eigen::Vector3d::Zero();
  int previous_stance_ = 3;
};
