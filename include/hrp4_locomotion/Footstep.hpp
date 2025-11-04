#ifndef LABROB_FOOTSTEP_HPP_
#define LABROB_FOOTSTEP_HPP_

#include <Eigen/Core>
#include <hrp4_locomotion/Foot.hpp>
#include <hrp4_locomotion/WalkingState.hpp>

namespace labrob {

struct Footstep {
  // Left foot pose
  Eigen::Matrix3d left_foot_rotation;
  Eigen::Vector3d left_foot_position;

  // Right foot pose
  Eigen::Matrix3d right_foot_rotation;
  Eigen::Vector3d right_foot_position;

  // Support foot identifier
  Foot support_foot;

  // Walking state
  WalkingState walking_state;

  // Duration in milliseconds
  int64_t duration_ms;

  // Swing foot trajectory height
  double swing_height;

  // Default constructor
  Footstep()
    : left_foot_rotation(Eigen::Matrix3d::Identity()),
      left_foot_position(Eigen::Vector3d::Zero()),
      right_foot_rotation(Eigen::Matrix3d::Identity()),
      right_foot_position(Eigen::Vector3d::Zero()),
      support_foot(Foot::LEFT),
      walking_state(WalkingState::Standing),
      duration_ms(0),
      swing_height(0.0) {}

  // Full constructor
  Footstep(const Eigen::Matrix3d& left_R, const Eigen::Vector3d& left_p,
           const Eigen::Matrix3d& right_R, const Eigen::Vector3d& right_p,
           Foot support, WalkingState state, int64_t duration, double height)
    : left_foot_rotation(left_R),
      left_foot_position(left_p),
      right_foot_rotation(right_R),
      right_foot_position(right_p),
      support_foot(support),
      walking_state(state),
      duration_ms(duration),
      swing_height(height) {}

  // Helper methods to get support/swing foot poses
  const Eigen::Matrix3d& get_support_foot_rotation() const {
    return (support_foot == Foot::LEFT) ? left_foot_rotation : right_foot_rotation;
  }

  const Eigen::Vector3d& get_support_foot_position() const {
    return (support_foot == Foot::LEFT) ? left_foot_position : right_foot_position;
  }

  const Eigen::Matrix3d& get_swing_foot_rotation() const {
    return (support_foot == Foot::LEFT) ? right_foot_rotation : left_foot_rotation;
  }

  const Eigen::Vector3d& get_swing_foot_position() const {
    return (support_foot == Foot::LEFT) ? right_foot_position : left_foot_position;
  }

  Eigen::Matrix3d& get_support_foot_rotation() {
    return (support_foot == Foot::LEFT) ? left_foot_rotation : right_foot_rotation;
  }

  Eigen::Vector3d& get_support_foot_position() {
    return (support_foot == Foot::LEFT) ? left_foot_position : right_foot_position;
  }

  Eigen::Matrix3d& get_swing_foot_rotation() {
    return (support_foot == Foot::LEFT) ? right_foot_rotation : left_foot_rotation;
  }

  Eigen::Vector3d& get_swing_foot_position() {
    return (support_foot == Foot::LEFT) ? right_foot_position : left_foot_position;
  }
};

} // end namespace labrob

#endif // LABROB_FOOTSTEP_HPP_
