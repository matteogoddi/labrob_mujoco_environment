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
// #include <hrp4_locomotion/DdpSolver.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>


namespace labrob {

class WalkingManager {
 public:
  WalkingManager();

  bool init(const labrob::RobotState& initial_robot_state, std::map<std::string, double> &armatures, bool useRobot);

  LIPState updateKF(LIPState filtered, LIPState current, const Eigen::Vector3d &input);

  LIPState updateKF2(LIPState filtered, LIPState current, const Eigen::Vector3d &input);

  RobotState updateEKF(RobotState current_state, bool useRobot, Eigen::VectorXd actual_output);

  void saveLogs();

  void update(
      const labrob::RobotState& sim_robot_state,
      labrob::JointCommand& joint_command, 
      labrob::RobotState& fb_robot_state,
      bool useRobot,
      Eigen::VectorXd actual_output
  );

  int64_t get_controller_frequency() const;

 protected:
  pinocchio::Model sim_robot_model;
  pinocchio::Data sim_robot_data;

  pinocchio::Model fb_robot_model;
  pinocchio::Data fb_robot_data;

  pinocchio::Model predicted_robot_model;
  pinocchio::Data predicted_robot_data;

  pinocchio::Model estimated_robot_model;
  pinocchio::Data estimated_robot_data;

  pinocchio::Model prec_estimated_robot_model;
  pinocchio::Data prec_estimated_robot_data;

  Eigen::MatrixXd P_;
  Eigen::MatrixXd Q;
  Eigen::MatrixXd R;
  Eigen::VectorXd x_estimate;
  Eigen::VectorXd y_pred;
  Eigen::VectorXd y_actual;
  Eigen::VectorXd y_estimate;
  Eigen::VectorXd actual_output;
  Eigen::VectorXd input;
  bool input_initialized = false;
  int n_ekf_output;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;
  pinocchio::FrameIndex torso_idx_;

  Eigen::VectorXd q_jnt_des_;

  labrob::GaitConfiguration initial_gait_configuration;

  double controller_timestep_msec_;

  labrob::WalkingData walking_data_;
  std::unique_ptr<labrob::ISMPC> ismpc_ptr_;

  Eigen::VectorXd M_armature_;

  LIPState sim_filt_LIPstate;
  LIPState sim_filt_LIPstate2;

  LIPState fb_filt_LIPstate;
  LIPState fb_LIPstate;
  
  Eigen::Matrix3d cov_x, cov_y, cov_z;
  double cov_meas_pos, cov_meas_vel, cov_meas_zmp;
  double cov_mod_pos, cov_mod_vel, cov_mod_zmp;

  std::shared_ptr<WholeBodyController> whole_body_controller_ptr_;

  Eigen::Vector3d u0 = Eigen::Vector3d::Zero();

private:

  Eigen::MatrixXd pseudoinverse(const Eigen::MatrixXd& J, double damp=1e-6) const;

  void swingFootTrajectory(
      pinocchio::SE3& swing_foot_pose,
      pinocchio::Motion& swing_foot_velocity,
      pinocchio::Motion& swing_foot_acceleration
  ) const;

  void swingFootTrajectoryBezier(
      pinocchio::SE3& swing_foot_pose,
      pinocchio::Motion& swing_foot_velocity
  ) const;

  int64_t controller_frequency_;
  int64_t t_msec_ = 0;

  std::unique_ptr<labrob::DiscreteLIPDynamics> discrete_lip_dynamics_ptr_;
  std::unique_ptr<labrob::DiscreteLIPDynamics> discrete_lip_dynamics_ptr_mpc_;

  Eigen::Vector3d prev_angular_momentum_ = Eigen::Vector3d::Zero();

  Eigen::MatrixXd Kalman_Gain; 

  std::vector<long long> mpc_timings_log_;
  std::vector<Eigen::Vector3d> mpc_com_log_;
  std::vector<Eigen::Vector3d> mpc_zmp_log_;
  std::vector<Eigen::Vector3d> com_log_;
  std::vector<Eigen::Vector3d> p_lsole_log_;
  std::vector<Eigen::Vector3d> p_rsole_log_;
  std::vector<Eigen::Vector3d> v_lsole_log_;
  std::vector<Eigen::Vector3d> v_rsole_log_;
  std::vector<Eigen::Vector3d> p_lsole_des_log_;
  std::vector<Eigen::Vector3d> p_rsole_des_log_;
  std::vector<Eigen::Vector3d> v_lsole_des_log_;
  std::vector<Eigen::Vector3d> v_rsole_des_log_;
  std::vector<Eigen::Vector3d> angular_momentum_log_;
  // std::vector<Eigen::VectorXd> fl_log_;
  // std::vector<Eigen::VectorXd> fr_log_;
  std::vector<Eigen::VectorXd> cop_computed_log_;
  std::vector<Eigen::VectorXd> mpc_predictions_log_;
  std::vector<Eigen::VectorXd> ekf_base_position_log_;
  std::vector<Eigen::VectorXd> ekf_base_velocity_log_;
  std::vector<Eigen::VectorXd> ekf_base_orientation_log_;
  std::vector<Eigen::VectorXd> ekf_base_angular_velocity_log_;
  std::vector<Eigen::VectorXd> ekf_joint_position_log_;
  std::vector<Eigen::VectorXd> ekf_joint_velocity_log_;
  std::vector<Eigen::VectorXd> sim_base_position_log_;
  std::vector<Eigen::VectorXd> sim_base_velocity_log_;
  std::vector<Eigen::VectorXd> sim_base_orientation_log_;
  std::vector<Eigen::VectorXd> sim_base_angular_velocity_log_;
  std::vector<Eigen::VectorXd> sim_joint_position_log_;
  std::vector<Eigen::VectorXd> sim_joint_velocity_log_;
  std::vector<Eigen::VectorXd> fb_base_position_log_;
  std::vector<Eigen::VectorXd> fb_base_velocity_log_;
  std::vector<Eigen::VectorXd> fb_base_orientation_log_;
  std::vector<Eigen::VectorXd> fb_base_angular_velocity_log_;
  std::vector<Eigen::VectorXd> fb_joint_position_log_;
  std::vector<Eigen::VectorXd> fb_joint_velocity_log_;
  std::vector<Eigen::VectorXd> real_com_log_;
  std::vector<Eigen::VectorXd> predicted_imu_accelerometer_log_;
  std::vector<Eigen::VectorXd> predicted_imu_angular_velocity_log_;
  std::vector<Eigen::VectorXd> predicted_imu_orientation_log_;
  std::vector<Eigen::VectorXd> fb_imu_accelerometer_log_;
  std::vector<Eigen::VectorXd> fb_imu_angular_velocity_log_;
  std::vector<Eigen::VectorXd> fb_imu_orientation_log_;
  std::vector<Eigen::VectorXd> input_torque_log_;

  std::vector<Eigen::MatrixXd> kalman_gain_log_;

  std::vector<long long> execution_time_wbc_log_;
  std::vector<long long> execution_time_mpc_log_;
  std::vector<long long> execution_time_ekf_log_;
  std::vector<long long> execution_time_kf_log_;
  std::vector<long long> execution_time_update_log_;

}; // end class WalkingManager

} // end namespace labrob

#endif // LABROB_WALKING_MANAGER_HPP_