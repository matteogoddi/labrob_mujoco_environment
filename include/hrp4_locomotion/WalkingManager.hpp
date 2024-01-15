#pragma once

#include <cmath>
#include <string>
#include <vector>

// Pinocchio
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ros/ros.h>

#include <hrp4_locomotion/ISMPC.hpp>
#include <hrp4_locomotion/JointState.hpp>
#include <hrp4_locomotion/RobotState.hpp>
#include <hrp4_locomotion/WalkingData.hpp>
#include <hrp4_locomotion/utils.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>


namespace labrob {

class WalkingManager {
 public:
  WalkingManager() = default;

  bool init(ros::NodeHandle& node_handle);

  void update(
      const labrob::RobotState& robot_state,
      labrob::RobotState& desired_robot_state
  );

 protected:
  pinocchio::Model robot_model_;
  pinocchio::Data robot_data_;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;
  pinocchio::FrameIndex torso_idx_;

  Eigen::VectorXd q_jnt_des_;

  int64_t controller_timestep_msec_;

  labrob::WalkingData walking_data_;
  std::unique_ptr<labrob::ISMPC> ismpc_ptr_;

  std::unique_ptr<labrob::qpsolvers::QPSolverEigenWrapper<double>> qp_solver_ptr_;

private:
  Eigen::MatrixXd pseudoinverse(const Eigen::MatrixXd& J, double damp=1e-6) const;

  Eigen::Matrix<double, 6, 1> err_frameplacement(const pinocchio::SE3& Ta, const pinocchio::SE3& Tb);
  Eigen::Vector3d err_translation(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb);
  Eigen::Vector3d err_rotation(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb);

  void swingFootTrajectory(
      pinocchio::SE3& swing_foot_pose,
      pinocchio::Motion& swing_foot_velocity
  ) const;

  void swingFootTrajectoryBezier(
      pinocchio::SE3& swing_foot_pose,
      pinocchio::Motion& swing_foot_velocity
  ) const;

  int64_t t_msec_ = 0;

  // Counter relative to reading of footstep planner, to be remove after
  // refactoring of Humanoids 2023:
  int footstep_reference_iter_ = 0;
  //int impulse_application_instant_msec_ = 4400 + 4500;
  double impulse_magnitude_ = 25.0;//16.0;
  std::vector<Eigen::Vector3d> impulses_direction_ = {
      Eigen::Vector3d(-2.0, -1.0, 0.0),
      Eigen::Vector3d(-1.0, 0.0, 0.0),
      Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, -1.0, 0.0),
      Eigen::Vector3d(0.0, 1.0, 0.0)
  };
  std::vector<int64_t> impulses_application_instant_msec_ = {
      4400 + 4500,
      4400 + 9500,
      4400 + 15500,
      4400 + 24500,
      4400 + 30500
  };

  // Log files:
  std::ofstream mpc_timings_log_file_;
  std::ofstream com_log_file_;
  std::ofstream zmp_log_file_;
  std::ofstream configuration_log_file_;
  std::ofstream lsole_log_file_;
  std::ofstream rsole_log_file_;
  std::ofstream lsole_des_log_file_;
  std::ofstream rsole_des_log_file_;

}; // end class WalkingManager

} // end namespace labrob