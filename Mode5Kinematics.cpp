#include "Mode5Kinematics.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <hrp4_locomotion/utils.hpp>

Mode5Kinematics::Mode5Kinematics() {
  pinocchio::urdf::buildModel("../g1_description/g1_29dof_with_hand_rev_1_0.urdf",
                              pinocchio::JointModelFreeFlyer(), model_);
  data_ = pinocchio::Data(model_);
  for (int i = 0; i < 29; ++i)
    q_index_[i] = model_.idx_qs.at(model_.getJointId(body_joint_names[i]));
  feet_ = {model_.getFrameId("left_foot_link"), model_.getFrameId("right_foot_link")};
}

void Mode5Kinematics::forward(const labrob::RobotState& state) {
  auto q = labrob::robot_state_to_pinocchio_joint_configuration(model_, state);
  auto v = labrob::robot_state_to_pinocchio_joint_velocity(model_, state);
  q.head<3>().setZero();
  v.head<3>().setZero();
  pinocchio::forwardKinematics(model_, data_, q, v);
  pinocchio::updateFramePlacements(model_, data_);
}

bool Mode5Kinematics::levelAndQuiet(const labrob::RobotState& state) {
  forward(state);
  if (std::abs(data_.oMf[feet_[0]].translation().z() - data_.oMf[feet_[1]].translation().z()) > 0.010 ||
      state.angular_velocity.norm() > 0.10) return false;
  for (const auto id : feet_)
    if (data_.oMf[id].rotation()(2, 2) < std::cos(5.0 * M_PI / 180.0)) return false;
  for (const auto& name : body_joint_names)
    if (std::abs(state.joint_state[name].vel) > 0.12) return false;
  return true;
}

void Mode5Kinematics::registerGround(const labrob::RobotState& state) {
  forward(state);
  Eigen::Vector3d x = data_.oMf[feet_[0]].rotation().col(0) + data_.oMf[feet_[1]].rotation().col(0);
  heading_ = Eigen::AngleAxisd(-std::atan2(x.y(), x.x()), Eigen::Vector3d::UnitZ());
  for (int f = 0; f < 2; ++f) anchors_[f] = heading_ * data_.oMf[feet_[f]].translation();
  position_ = -0.5 * (anchors_[0] + anchors_[1]);
  for (auto& anchor : anchors_) anchor += position_;
  previous_stance_ = 3;
}

void Mode5Kinematics::locate(labrob::RobotState& state, int stance) {
  state.orientation = heading_ * state.orientation;
  forward(state);
  Eigen::Vector3d p = Eigen::Vector3d::Zero(), velocity = Eigen::Vector3d::Zero();
  int count = 0;
  // Use established anchors first, then record the measured touchdown position.
  for (int f = 0; f < 2; ++f) {
    if (!(stance & previous_stance_ & (1 << f))) continue;
    p += anchors_[f] - data_.oMf[feet_[f]].translation();
    velocity -= pinocchio::getFrameVelocity(model_, data_, feet_[f], pinocchio::LOCAL_WORLD_ALIGNED).linear();
    ++count;
  }
  if (!count) throw std::runtime_error("Missing common stance foot during support change");
  position_ = p / count;
  state.position = position_;
  state.linear_velocity = state.orientation.conjugate() * (velocity / count);
  for (int f = 0; f < 2; ++f)
    if ((stance & ~previous_stance_) & (1 << f))
      anchors_[f] = position_ + data_.oMf[feet_[f]].translation();
  previous_stance_ = stance;
}

std::array<double, 2> Mode5Kinematics::heights() const {
  return {position_.z() + data_.oMf[feet_[0]].translation().z(),
          position_.z() + data_.oMf[feet_[1]].translation().z()};
}
