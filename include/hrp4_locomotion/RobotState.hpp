#ifndef LABROB_ROBOT_STATE_H_
#define LABROB_ROBOT_STATE_H_

// #include <hrp4_locomotion/JointState.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace labrob {

struct JointData {
  double pos = 0.0;
  double vel = 0.0;
  double acc = 0.0;
  double eff = 0.0;
}; // end struct JointData

class RobotState {
 public:
  RobotState()
  : position(Eigen::Vector3d::Zero()),
    orientation(1.0, 0.0, 0.0, 0.0),
    linear_velocity(Eigen::Vector3d::Zero()),
    angular_velocity(Eigen::Vector3d::Zero()) {
  }

  Eigen::VectorXd
  get_pinocchio_joint_configuration(
      const pinocchio::Model& robot_model
  ) const {
    Eigen::VectorXd q(robot_model.nq);
    q.head<3>() = position;
    q.segment<4>(3) = orientation.coeffs();

    // NOTE: start from joint id (2) to skip frames "universe" and "root_joint".
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) robot_model.njoints; ++joint_id)
      q[joint_id + 5] = joint_state[robot_model.names[joint_id]].pos;

    return q;
  };

  Eigen::VectorXd
  get_pinocchio_joint_velocity(
      const pinocchio::Model& robot_model
  ) const {
    Eigen::VectorXd qdot(robot_model.nv);
    qdot.head<3>() = linear_velocity;
    qdot.segment<3>(3) = angular_velocity;

    // NOTE: start from joint id (2) to skip frames "universe" and "root_joint".
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) robot_model.njoints; ++joint_id)
      qdot[joint_id + 4] = joint_state[robot_model.names[joint_id]].vel;
    
    return qdot;
  };

  Eigen::Vector3d position;
  Eigen::Quaterniond orientation;

  // Velocities of the base link expressed in the reference frame of the base.
  Eigen::Vector3d linear_velocity;
  Eigen::Vector3d angular_velocity;

  mutable std::map<std::string, JointData> joint_state;

  Eigen::Vector3d total_force;

  std::vector<Eigen::Vector3d> contact_points;
  std::vector<Eigen::Vector3d> contact_forces;
}; // end class RobotState

} // end namespace labrob

#endif // LABROB_ROBOT_STATE_H_