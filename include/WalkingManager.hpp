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
#include <WalkingData.hpp>
#include <Logger.hpp>
#include <utils.hpp>
#include <WholeBodyController.hpp>

#include <QpSolver.hpp>


namespace labrob {

class WalkingManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  WalkingManager();

  bool init(const labrob::RobotState& initial_robot_state, std::map<std::string, double> &armatures);

  RobotState getNewRobotState();
  RobotState getActualRobotState();

  void saveLogs();

  void update(
      labrob::RobotState& robot_state,
      labrob::JointCommand& joint_command
  );

  int64_t get_controller_frequency() const;

  // Exposes WBC joint+base acceleration for external filters (e.g. JointKF)
  const Eigen::VectorXd& get_wbc_q_ddot() const {
    return whole_body_controller_ptr_->get_q_ddot();
  }

  // Returns {left_in_contact, right_in_contact} based on current walking state
  std::array<bool,2> get_contact() const;

  const pinocchio::Model& get_robot_model() const { return robot_model; }

 protected:
  pinocchio::Model robot_model;
  pinocchio::Data robot_data;
  double mass;

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

private:

  // ── CoMKF (inlined) ─────────────────────────────────────────────────────
  LIPState com_kf_step(LIPState filtered, LIPState current,
                       const Eigen::Vector3d& input);

  Eigen::Matrix3d com_kf_cov_x_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d com_kf_cov_y_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d com_kf_cov_z_ = Eigen::Matrix3d::Identity();
  // noise parameters (same defaults as CoMKF class)
  static constexpr double kComKfMeasPos = 1.0e1;
  static constexpr double kComKfMeasVel = 1.0e2;
  static constexpr double kComKfMeasZmp = 1.0e8;
  static constexpr double kComKfModPos  = 1.0;
  static constexpr double kComKfModVel  = 1.0;
  static constexpr double kComKfModZmp  = 1.0;
  // ────────────────────────────────────────────────────────────────────────

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