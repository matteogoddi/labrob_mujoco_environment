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
#include <JointCommand.hpp>
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
  double mass;

  RobotState fb_robot_state;

  Eigen::Matrix3d imu_calibration_matrix;
  std::vector<Eigen::Vector3d> acc_samples;
  std::vector<Eigen::Vector3d> imu_samples;

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

  double eta2;

  Eigen::VectorXd estimated_force = Eigen::VectorXd::Zero(6);

  std::shared_ptr<WholeBodyController> whole_body_controller_ptr_;
  std::unique_ptr<labrob::RightInvariantEKF> ri_ekf_ptr_;
  std::unique_ptr<labrob::DiligentKio> diligent_kio_ptr_;

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

  std::unique_ptr<labrob::SimpleEKF> simple_ekf_ptr_;

  // Filter selection — enable any combination for comparison.
  // The primary filter for WBC control follows this priority:
  //   simple_ekf > ri_ekf > diligent_kio > base_ekf
  // All enabled filters run every step; only the primary feeds fb_robot_state.
  bool use_simple_ekf_   = true;   // discrete-time EKF (SimpleEKF)
  bool use_base_ekf_     = false;  // contact-aided base EKF (BaseEKF)
  bool use_ri_ekf_       = false;  // right-invariant EKF (RightInvariantEKF)
  bool use_diligent_kio_ = false;  // left-invariant EKF (DiligentKio)

  // Logs
  Logger logger_;

  // Per-solve IS-MPC snapshots (nested structure, managed separately from Logger).
  std::vector<int64_t>                        mpc_snapshot_t_log_;
  std::vector<std::vector<Eigen::VectorXd>>   mpc_snapshot_x_log_;
  std::vector<std::vector<Eigen::VectorXd>>   mpc_snapshot_u_log_;

  std::vector<double> parameters_log_;

}; // end class WalkingManager

} // end namespace labrob

#endif // LABROB_WALKING_MANAGER_HPP_