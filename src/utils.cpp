#include <hrp4_locomotion/utils.hpp>

// Eigen
#include <Eigen/Geometry>

// hrp4_locomotion
#include <hrp4_locomotion/SE3.hpp>


namespace labrob {

Eigen::Matrix<double, 6, 1>
err_frameplacement(const pinocchio::SE3& Ta, const pinocchio::SE3& Tb) {
  // TODO: how do you use pinocchio::log6?
  Eigen::Matrix<double, 6, 1> err;
  err << err_translation(Ta.translation(), Tb.translation()),
      err_rotation(Ta.rotation(), Tb.rotation());
  return err;
}

Eigen::Vector3d
err_translation(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb) {
  return pa - pb;
}

Eigen::Vector3d
err_rotation(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb) {
  // TODO: how do you use pinocchio::log3?
  Eigen::Matrix3d Rdiff = Rb.transpose() * Ra;
  auto aa = Eigen::AngleAxisd(Rdiff);
  return aa.angle() * Ra * aa.axis();
}

// write function quaternionFromRotVec
Eigen::Quaterniond quaternionFromRotVec(const Eigen::Vector3d& rot_vec) {
  double angle = rot_vec.norm();
  if (angle > M_PI) {
      angle -= 2 * M_PI;
  } else if (angle < -M_PI) {
      angle += 2 * M_PI;
  }
  if (angle < 1e-6) {
      return Eigen::Quaterniond(1, 0, 0, 0); // No rotation
  } else {
      Eigen::Vector3d axis = rot_vec.normalized();
      return Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
  }
}

// write function rotVecFromQuaternion
Eigen::Vector3d rotVecFromQuaternion(const Eigen::Quaterniond& q) {
  Eigen::AngleAxisd axis_angle(q);
  if (axis_angle.angle() > M_PI) {
      axis_angle.angle() -= 2 * M_PI;
  } else if (axis_angle.angle() < -M_PI) {
      axis_angle.angle() += 2 * M_PI;
  }

  return axis_angle.axis() * axis_angle.angle();
}

Eigen::VectorXd
robot_state_to_pinocchio_joint_configuration(
    const pinocchio::Model& robot_model,
    const labrob::RobotState& robot_state
) {
  // labrob::RobotState representation to Pinocchio representation:
  // TODO: RobotState also has information about the velocity of the floating base.
  // TODO: is there a less error-prone way to convert representation?
  Eigen::VectorXd q(robot_model.nq);
  q.head<3>() = robot_state.position;
  q.segment<4>(3) = robot_state.orientation.coeffs();
  // NOTE: start from joint id (2) to skip frames "universe" and "root_joint".
  for(pinocchio::JointIndex joint_id = 2;
      joint_id < (pinocchio::JointIndex) robot_model.njoints;
      ++joint_id) {
    const auto& joint_name = robot_model.names[joint_id];
    q[joint_id + 5] = robot_state.joint_state[joint_name].pos;
  }

  return q;
}

Eigen::VectorXd
robot_state_to_pinocchio_joint_velocity(
    const pinocchio::Model& robot_model,
    const labrob::RobotState& robot_state
) {
  Eigen::VectorXd qdot(robot_model.nv);
  qdot.head<3>() = robot_state.linear_velocity;
  qdot.segment<3>(3) = robot_state.angular_velocity;
  // NOTE: start from joint id (2) to skip frames "universe" and "root_joint".
  for(pinocchio::JointIndex joint_id = 2;
      joint_id < (pinocchio::JointIndex) robot_model.njoints;
      ++joint_id) {
    const auto& joint_name = robot_model.names[joint_id];
    qdot[joint_id + 4] = robot_state.joint_state[joint_name].vel;
  }
  
  return qdot;
}

Eigen::Vector3d
updateState(
    const LIPState& lip_state,
    double zmpDot,
    int dim,
    double com_target_height,
    int64_t control_timestep_msec
) {

  double control_timestep = 0.001 * static_cast<double>(control_timestep_msec);

  double omega = std::sqrt(9.81 / com_target_height);

  // Update the state along the dim-th direction (0,1,2) = (x,y,z)

  double ch = cosh(omega * control_timestep);
  double sh = sinh(omega * control_timestep);

  Eigen::Matrix3d A_upd = Eigen::Matrix3d::Zero();
  Eigen::Vector3d B_upd = Eigen::Vector3d::Zero();
  A_upd<<ch,sh/omega,1-ch,omega*sh,ch,-omega*sh,0,0,1;
  B_upd<<control_timestep-sh/omega,1-ch,control_timestep;

  Eigen::Vector3d currentState = Eigen::Vector3d(
      lip_state.com_pos_(dim),
      lip_state.com_vel_(dim),
      lip_state.zmp_pos_(dim)
  );

  if (dim == 2) return A_upd*(currentState + Eigen::Vector3d(0.0,0.0,com_target_height)) + B_upd*zmpDot - Eigen::Vector3d(0.0,0.0,com_target_height);

  return A_upd*currentState + B_upd*zmpDot;
}

labrob::WalkingState
walkingStateFromString(
    const std::string& walking_state_str
) {
  labrob::WalkingState walking_state;
  if (walking_state_str == "Init") {
    walking_state = labrob::WalkingState::Init;
  } else if (walking_state_str == "PostureRegulation") {
    walking_state = labrob::WalkingState::PostureRegulation;
  } else if (walking_state_str == "Standing") {
    walking_state = labrob::WalkingState::Standing;
  } else if (walking_state_str == "Starting") {
    walking_state = labrob::WalkingState::Starting;
  } else if (walking_state_str == "SingleSupport") {
    walking_state = labrob::WalkingState::SingleSupport;
  } else if (walking_state_str == "DoubleSupport") {
    walking_state = labrob::WalkingState::DoubleSupport;
  } else if (walking_state_str == "Stopping") {
    walking_state = labrob::WalkingState::Stopping;
  } else if (walking_state_str == "AbortStarting") {
    walking_state = labrob::WalkingState::AbortStarting;
  } else if (walking_state_str == "AbortWalking") {
    walking_state = labrob::WalkingState::AbortWalking;
  }
  return walking_state;
}

} // end namespace labrob