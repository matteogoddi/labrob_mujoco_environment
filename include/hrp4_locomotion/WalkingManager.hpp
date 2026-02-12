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
#include <hrp4_locomotion/RobotState.hpp>
#include <hrp4_locomotion/WalkingData.hpp>
#include <hrp4_locomotion/utils.hpp>
#include <hrp4_locomotion/WholeBodyController.hpp>
#include <hrp4_locomotion/ResidualEstimator.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>


namespace labrob {

class WalkingManager {
 public:
  WalkingManager();

  bool init(const labrob::RobotState& initial_robot_state, std::map<std::string, double> &armatures);

  LIPState updateKF(LIPState filtered, LIPState current, const Eigen::Vector3d &input);

  LIPState updateKF2(LIPState filtered, LIPState current, const Eigen::Vector3d &input);

  Eigen::VectorXd propagateState(
      const Eigen::VectorXd& x,
      const Eigen::VectorXd& q_ddot,
      double dt);

  Eigen::MatrixXd computeNumericalA(
      const Eigen::VectorXd& x,
      const Eigen::VectorXd& q_ddot,
      double dt
  );

  RobotState updateEKF(Eigen::VectorXd actual_output);

  RobotState getNewRobotState();
  RobotState getActualRobotState();

  void saveLogs();

  void update(
      const labrob::RobotState& sim_robot_state,
      labrob::JointCommand& joint_command
  );

  int64_t get_controller_frequency() const;

 protected:
  pinocchio::Model robot_model;
  pinocchio::Data sim_robot_data;
  pinocchio::Data fb_robot_data;
  pinocchio::Data predicted_robot_data;
  pinocchio::Data estimated_robot_data;
  double mass;

  RobotState fb_robot_state;

  Eigen::MatrixXd P_;
  Eigen::MatrixXd Q;
  Eigen::MatrixXd R;
  Eigen::VectorXd x_estimate;
  Eigen::VectorXd y_pred;
  Eigen::VectorXd y_actual;
  Eigen::VectorXd y_estimate;
  Eigen::VectorXd actual_output;
  int n_ekf_output;

  Eigen::VectorXd integrated_state_pos;
  Eigen::VectorXd integrated_state_vel;

  int njnt;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;
  pinocchio::FrameIndex torso_idx_;
  pinocchio::FrameIndex imu_idx_;


  int64_t mpc_prediction_horizon_msec = 1000;
  int64_t mpc_timestep_msec = 50;
  double foot_constraint_square_length = 0.22;
  double foot_constraint_square_width = 0.08;

  Eigen::VectorXd q_jnt_des_;

  labrob::GaitConfiguration initial_gait_configuration;

  double controller_timestep_msec_;

  labrob::WalkingData walking_data_;
  std::unique_ptr<labrob::ISMPC> ismpc_ptr_;
  std::unique_ptr<labrob::ResidualEstimator> residual_estimator_ptr_;

  Eigen::VectorXd M_armature_;

  LIPState LipState;
  LIPState kf_LipState;
  LIPState des_LipState;

  Eigen::Vector3d fixed_com_pos;
  Eigen::Vector3d fixed_com_vel;
  Eigen::Vector3d fixed_zmp_pos;

  int BASE_IDX;
  int IMU_ROTVEC_IDX;
  int JOINTS_IDX;
  int BASE_VEL_IDX;
  int JOINTS_VEL_IDX;
  int LEFT_FOOT_VEL_IDX;
  int RIGHT_FOOT_VEL_IDX;

  double eta2;

  Eigen::VectorXd estimated_force = Eigen::VectorXd::Zero(6);
  
  Eigen::Matrix3d cov_x, cov_y, cov_z;
  double cov_meas_pos, cov_meas_vel, cov_meas_zmp;
  double cov_mod_pos, cov_mod_vel, cov_mod_zmp;

  std::shared_ptr<WholeBodyController> whole_body_controller_ptr_;

  Eigen::MatrixXd J_imu_est, J_imu_dot_est, J_left_foot_est, J_right_foot_est;

private:

  Eigen::MatrixXd pseudoinverse(const Eigen::MatrixXd& J, double damp=1e-6) const;

  void swingFootTrajectory(
      pinocchio::SE3& swing_foot_pose,
      pinocchio::Motion& swing_foot_velocity,
      pinocchio::Motion& swing_foot_acceleration
  ) const;

  int64_t controller_frequency_;
  int64_t t_msec_ = 0;

  std::unique_ptr<labrob::DiscreteLIPDynamics> discrete_lip_dynamics_ptr_;
  std::unique_ptr<labrob::DiscreteLIPDynamics> discrete_lip_dynamics_ptr_mpc_;

  Eigen::Vector3d prev_angular_momentum_ = Eigen::Vector3d::Zero();

  Eigen::MatrixXd Kalman_Gain; 

  // Logs

  std::vector<Eigen::Vector3d> fb_com_position_log_;
  std::vector<Eigen::Vector3d> fb_com_velocity_log_;
  std::vector<Eigen::Vector3d> fb_zmp_position_log_;
  std::vector<Eigen::Vector3d> fb_com_acceleration_log_;

  std::vector<Eigen::Vector3d> kf_com_position_log_;
  std::vector<Eigen::Vector3d> kf_com_velocity_log_;
  std::vector<Eigen::Vector3d> kf_zmp_position_log_;
  std::vector<Eigen::Vector3d> kf_com_acceleration_log_;

  std::vector<Eigen::Vector3d> des_com_position_log_;
  std::vector<Eigen::Vector3d> des_com_velocity_log_;
  std::vector<Eigen::Vector3d> des_zmp_position_log_;
  std::vector<Eigen::Vector3d> des_com_acceleration_log_;

  std::vector<Eigen::Vector3d> ef_zmp_position_log_;

  std::vector<Eigen::Vector3d> torso_orientation_log_;
  std::vector<Eigen::Vector3d> torso_angular_velocity_log_;
  std::vector<Eigen::Vector3d> des_torso_orientation_log_;
  std::vector<Eigen::Vector3d> des_torso_angular_velocity_log_;

  std::vector<Eigen::Vector3d> p_lsole_fb_log_;
  std::vector<Eigen::Vector3d> p_rsole_fb_log_;
  std::vector<Eigen::Vector3d> v_lsole_fb_log_;
  std::vector<Eigen::Vector3d> v_rsole_fb_log_;
  std::vector<Eigen::Vector3d> p_lsole_des_log_;
  std::vector<Eigen::Vector3d> p_rsole_des_log_;
  std::vector<Eigen::Vector3d> v_lsole_des_log_;
  std::vector<Eigen::Vector3d> v_rsole_des_log_;

  
  std::vector<Eigen::Vector3d> fb_lsole_orientation_log_;
  std::vector<Eigen::Vector3d> fb_rsole_orientation_log_;
  std::vector<Eigen::Vector3d> des_lsole_orientation_log_;
  std::vector<Eigen::Vector3d> des_rsole_orientation_log_;

  std::vector<Eigen::Vector3d> estimated_force_lsole_log_;
  std::vector<Eigen::Vector3d> estimated_force_rsole_log_;
  std::vector<Eigen::VectorXd> wbc_accelerations_log_;
  
  std::vector<Eigen::Vector3d> angular_momentum_log_;
  // std::vector<Eigen::VectorXd> fl_log_;
  // std::vector<Eigen::VectorXd> fr_log_;
  std::vector<Eigen::VectorXd> mpc_predictions_log_;
  // ekf state vectors

  std::vector<Eigen::VectorXd> odometry_base_position_log_;
  std::vector<Eigen::VectorXd> odometry_base_velocity_log_;
  std::vector<Eigen::VectorXd> odometry_imu_orientation_log_;
  std::vector<Eigen::VectorXd> odometry_imu_orientation_rpy_log_;
  std::vector<Eigen::VectorXd> measured_imu_orientation_log_;
  std::vector<Eigen::VectorXd> measured_imu_angular_velocity_log_;
  std::vector<Eigen::VectorXd> measured_imu_accelerometer_log_;
  std::vector<Eigen::VectorXd> measured_joint_position_log_;
  std::vector<Eigen::VectorXd> measured_joint_velocity_log_;

  std::vector<Eigen::VectorXd> ekf_base_position_log_;
  std::vector<Eigen::VectorXd> ekf_base_velocity_log_;
  std::vector<Eigen::VectorXd> ekf_base_orientation_log_;
  std::vector<Eigen::VectorXd> ekf_base_orientation_rpy_log_;
  std::vector<Eigen::VectorXd> ekf_base_angular_velocity_log_;
  std::vector<Eigen::VectorXd> ekf_imu_orientation_log_;
  std::vector<Eigen::VectorXd> ekf_imu_orientation_rpy_log_;
  std::vector<Eigen::VectorXd> ekf_imu_angular_velocity_log_;
  std::vector<Eigen::VectorXd> ekf_joint_position_log_;
  std::vector<Eigen::VectorXd> ekf_joint_velocity_log_;

  std::vector<Eigen::VectorXd> mpc_pred_com_pos_log_;
  std::vector<Eigen::VectorXd> mpc_pred_com_vel_log_;
  std::vector<Eigen::VectorXd> mpc_pred_zmp_pos_log_;

  std::vector<Eigen::VectorXd> mpc_zmp_velocity_log_;
  std::vector<Eigen::VectorXd> con_zmp_velocity_log_;

  std::vector<Eigen::VectorXd> input_torque_log_;
  
  std::vector<double> parameters_log_;

  //execution time logs
  std::vector<long long> execution_time_wbc_log_;
  std::vector<long long> execution_time_mpc_log_;
  std::vector<long long> execution_time_ekf_log_;
  std::vector<long long> execution_time_kf_log_;
  std::vector<long long> execution_time_update_log_;

}; // end class WalkingManager

} // end namespace labrob

#endif // LABROB_WALKING_MANAGER_HPP_