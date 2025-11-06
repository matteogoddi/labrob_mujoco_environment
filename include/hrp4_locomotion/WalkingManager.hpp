#ifndef LABROB_WALKING_MANAGER_HPP_
#define LABROB_WALKING_MANAGER_HPP_

#include <cmath>
#include <string>
#include <vector>

// Pinocchio
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <hrp4_locomotion/DiscreteLIPDynamics.hpp>
#include <hrp4_locomotion/ISMPC.hpp>
#include <hrp4_locomotion/JointCommand.hpp>
#include <hrp4_locomotion/JointState.hpp>
#include <hrp4_locomotion/RobotState.hpp>
#include <hrp4_locomotion/WalkingData.hpp>
#include <hrp4_locomotion/utils.hpp>
#include <hrp4_locomotion/WholeBodyController.hpp>
#include <hrp4_locomotion/ResidualEstimator.hpp>


namespace labrob {

class WalkingManager {
 public:
  WalkingManager();

  bool init(const labrob::RobotState& initial_robot_state, std::map<std::string, double> &armatures);

  LIPState updateKF(LIPState filtered, LIPState current, const Eigen::Vector3d &input);

  void update(
      const labrob::RobotState& sim_robot_state,
      labrob::JointCommand& joint_command,
      Eigen::VectorXd actual_output
  );

  int64_t get_controller_frequency() const;

 protected:
  pinocchio::Model robot_model;
  pinocchio::Data sim_robot_data;
  pinocchio::Data fb_robot_data;

  Eigen::VectorXd actual_output;

  Eigen::VectorXd integrated_state_pos;
  Eigen::VectorXd integrated_state_vel;

  int njnt;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;
  pinocchio::FrameIndex torso_idx_;
  pinocchio::FrameIndex imu_idx_;

  bool walking_data_initialized_ = false;

  Eigen::VectorXd q_jnt_des_;

  double controller_timestep_msec_;

  labrob::WalkingData walking_data_;
  std::unique_ptr<labrob::ISMPC> ismpc_ptr_;
  std::unique_ptr<labrob::ResidualEstimator> residual_estimator_ptr_;

  Eigen::VectorXd M_armature_;

  LIPState LipState;
  LIPState kf_LipState;

  RobotState fb_robot_state;

  Eigen::VectorXd estimated_force = Eigen::VectorXd::Zero(6);

  std::shared_ptr<WholeBodyController> whole_body_controller_ptr_;

private:

  void swingFootTrajectory(
      pinocchio::SE3& swing_foot_pose,
      pinocchio::Motion& swing_foot_velocity,
      pinocchio::Motion& swing_foot_acceleration
  ) const;

  int64_t controller_frequency_;
  int64_t t_msec_ = 0;

  std::unique_ptr<labrob::DiscreteLIPDynamics> discrete_lip_dynamics_ptr_;

  Eigen::Vector3d prev_angular_momentum_ = Eigen::Vector3d::Zero();

}; // end class WalkingManager

} // end namespace labrob

#endif // LABROB_WALKING_MANAGER_HPP_