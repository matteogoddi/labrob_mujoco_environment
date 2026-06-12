#ifndef LABROB_WALKING_MANAGER_HPP_
#define LABROB_WALKING_MANAGER_HPP_

#include <cmath>
#include <string>
#include <vector>

// Pinocchio
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <DiscreteLIPDynamics.hpp>
#include <ISMPC.hpp>
#include <JISMPC.hpp>
#include <JISMPCWholeBodyController.hpp>
#include <JointCommand.hpp>
#include <ResidualEstimator.hpp>
#include <RobotState.hpp>
#include <StateFiltering.hpp>
#include <WalkingData.hpp>
#include <Logger.hpp>
#include <utils.hpp>
#include <WholeBodyController.hpp>

#include <QpSolver.hpp>


namespace labrob {

class WalkingManager {
 public:
  WalkingManager();

  bool init(const labrob::RobotState& initial_robot_state, std::map<std::string, double> &armatures);

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

  // Switch to JIS-MPC mode (must be called before init()).
  void enableJISMPC(bool enable = true) { use_jismpc_ = enable; }

  // Directory where per-solve MPC snapshots are written.
  // Default: /tmp/mpc_data  (simulation).
  // For real-robot experiments set this to the experiment folder before init().
  void setMpcDataDir(const std::string& dir) { mpc_data_dir_ = dir; }

 protected:
  pinocchio::Model robot_model;
  pinocchio::Data sim_robot_data;
  pinocchio::Data fb_robot_data;
  pinocchio::Data predicted_robot_data;
  pinocchio::Data estimated_robot_data;
  double mass;

  RobotState fb_robot_state;

  Eigen::Matrix3d imu_calibration_matrix;
  std::vector<Eigen::Vector3d> acc_samples;
  std::vector<Eigen::Vector3d> imu_samples;
  
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
  pinocchio::FrameIndex pelvis_idx_;
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

  // JIS-MPC (alternative to IS-MPC)
  bool use_jismpc_ = false;

  // Output directory for per-solve MPC snapshots.
  std::string mpc_data_dir_ = "/tmp/mpc_data";
  int64_t jismpc_timestep_msec_ = 100; // 10 Hz MPC
  std::unique_ptr<labrob::JISMPC> jismpc_ptr_;
  std::shared_ptr<labrob::JISMPCWholeBodyController> jismpc_wbc_ptr_;
  std::unique_ptr<labrob::ResidualEstimator> residual_estimator_ptr_;
  std::unique_ptr<labrob::JointKF> joint_kf_ptr_;
  std::unique_ptr<labrob::BaseEKF> base_ekf_ptr_;
  std::unique_ptr<labrob::CoMKF> com_kf_ptr_;

  Eigen::VectorXd M_armature_;

  LIPState LipState;
  LIPState kf_LipState;
  LIPState des_LipState;

  Eigen::Vector3d p_CoM_init;

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

  std::shared_ptr<WholeBodyController> whole_body_controller_ptr_;
  std::unique_ptr<labrob::RightInvariantEKF> ri_ekf_ptr_;
  std::unique_ptr<labrob::DiligentKio> diligent_kio_ptr_;

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

  Eigen::Vector3d input_gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d input_acc = Eigen::Vector3d::Zero();

  Eigen::MatrixXd Kalman_Gain; 

  // Logs
  Logger logger_;

  // Per-solve ISMPC snapshots (nested structure, managed separately from Logger).
  std::vector<int64_t>                        mpc_snapshot_t_log_;
  std::vector<std::vector<Eigen::VectorXd>>   mpc_snapshot_x_log_;
  std::vector<std::vector<Eigen::VectorXd>>   mpc_snapshot_u_log_;

  std::vector<double> parameters_log_;

}; // end class WalkingManager

} // end namespace labrob

#endif // LABROB_WALKING_MANAGER_HPP_