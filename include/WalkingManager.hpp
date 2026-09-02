#ifndef LABROB_WALKING_MANAGER_HPP_
#define LABROB_WALKING_MANAGER_HPP_

#include <cmath>
#include <string>
#include <vector>

// Pinocchio
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <DiscreteLIPDynamics.hpp>
#include <DiscretePLIPDynamics.hpp>
#include <ISMPC.hpp>
#include <JointCommand.hpp>
#include <RobotState.hpp>
#include <WalkingData.hpp>
#include <Logger.hpp>
#include <utils.hpp>
#include <WholeBodyController.hpp>
#include <WristForceEstimator.hpp>
#include <FootstepPlannerCoop.hpp>
#include <HandAdmittanceController.hpp>

#include <QpSolver.hpp>
#include <globals.h>


namespace labrob {

class WalkingManager {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  WalkingManager();

  bool init(const labrob::RobotState& initial_robot_state, std::map<std::string, double> &armatures);


  void setReactiveStanding(bool r) { reactive_standing_ = r;};
  void setVerboseCoop(bool v) { verbose_coop_ = v;};

  void setHandForces(
    const Eigen::Vector3d& f_l_W,
    const Eigen::Vector3d& f_r_W
  ) {
    hac_f_l_W = f_l_W;
    hac_f_r_W = f_r_W;
  }

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

  // Orientation of the HAC frame F w.r.t. world, cached from the last update()
  const Eigen::Matrix3d& get_R_F_hac() const { return R_F_hac_; }

 protected:
  pinocchio::Model robot_model;
  pinocchio::Data robot_data;
  double mass;

  int njnt;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;
  pinocchio::FrameIndex torso_idx_;
  pinocchio::FrameIndex lwrist_idx_;
  pinocchio::FrameIndex rwrist_idx_;
  pinocchio::FrameIndex pelvis_idx_;
  pinocchio::FrameIndex imu_idx_;

  int64_t mpc_prediction_horizon_msec = 1000;
  int64_t mpc_timestep_msec = 50;
  // Aligned to WholeBodyController::Params::foot_length/foot_width
  // (src/WholeBodyController.cpp), itself aligned to the physical sole
  // footprint. Zero margin against the WBC foot rectangle by design here --
  // tighten deliberately if a conservative ZMP margin is needed.
  double foot_constraint_square_length = 0.14; //0.17;
  double foot_constraint_square_width = 0.04; //0.05;

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

  Eigen::VectorXd estimated_force_sole = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd estimated_force_wrist = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd estimated_moment_sole = Eigen::VectorXd::Zero(6);


  std::shared_ptr<WholeBodyController> whole_body_controller_ptr_;
  std::unique_ptr<labrob::WristForceEstimator> wrist_force_estimator_ptr_;

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
  static constexpr double kComKfMeasZmp = 1.0e3; //1.0e03;
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

  Eigen::Vector3d p_F_hac_;
  Eigen::Matrix3d R_F_hac_;
  Eigen::Vector3d v_F_hac_;
  Eigen::Vector3d v_lsole_;
  Eigen::Vector3d v_rsole_;
  pinocchio::SE3 T_lsole_;
  pinocchio::SE3 T_rsole_;
  void updateHACLocalFrame();

  // HELPERS //
  void showPlan(const labrob::FootstepPlannerCoop::QP2DResult res);
  void showDeque(const labrob::WalkingData wd);
  void logOfpSnapshot(
    int64_t t_msec,
    const labrob::FootstepPlannerCoop::QP2DResult& result,
    const Eigen::Vector2d& p0,
    double yaw0
  );
  void frictionConeRatios(
    const Eigen::VectorXd& flr,
    double mu,
    int nc
  );

  int64_t controller_frequency_;
  int64_t t_msec_ = 0;

  std::unique_ptr<labrob::DiscreteLIPDynamics> discrete_lip_dynamics_ptr_;
  std::unique_ptr<labrob::DiscretePLIPDynamics> discrete_plip_dynamics_ptr_;
  std::unique_ptr<labrob::DiscreteLIPDynamics> discrete_lip_dynamics_ptr_mpc_;
  std::unique_ptr<labrob::HandAdmittanceController> hac_ptr_;

    // Private members online planner
  std::unique_ptr<labrob::FootstepPlannerCoop> coop_planner_ptr_;
  bool reactive_standing_ = true;
  bool verbose_coop_ = false;
  bool coop_walking_triggered_ = false;
  bool coop_walking_active_ = false;
  labrob::Foot prev_support_foot_ = labrob::Foot::LEFT;
  Eigen::Vector2d integral_eh_ = Eigen::Vector2d::Zero();
  int64_t coop_T_ds_ms_ = 800;   // ms DS (T_step=1400ms, ratio=0.4)
  int64_t coop_T_ss_ms_ = 4000;   // ms SS
  double  coop_step_height_ = 0.06;
  bool waiting_for_rolling_ = false;
  bool stopping_requested_ = false;

  // Per-solve online footstep planner (coop) snapshots, written to disk in saveLogs().
  std::vector<int64_t>         ofp_snapshot_t_log_;
  std::vector<Eigen::VectorXd> ofp_snapshot_solution_log_;   ///< qp_solution, stacked [x1,y1,...,xF,yF]
  std::vector<double>          ofp_snapshot_delta_theta_log_;
  std::vector<Eigen::Vector2d> ofp_snapshot_delta_p_log_;
  std::vector<int>             ofp_snapshot_support_foot_log_;  ///< 0 = LEFT, 1 = RIGHT (support_foot_start)
  std::vector<Eigen::Vector2d> ofp_snapshot_p0_log_;            ///< posizione piede di supporto usata come anchor
  std::vector<double>          ofp_snapshot_yaw0_log_;          ///< yaw del piede di supporto usato come anchor [rad]
  double ofp_da_x_ = 0.0;  ///< dimensione x della regione ammissibile cinematica del planner [m]
  double ofp_da_y_ = 0.0;  ///< dimensione y della regione ammissibile cinematica del planner [m]
  double ofp_ell_  = 0.0;  ///< semidistanza laterale nominale tra i piedi usata dal planner [m]

    // Private members HAC
  Eigen::Vector3d hac_f_l_W = Eigen::Vector3d::Zero();
  Eigen::Vector3d hac_f_r_W = Eigen::Vector3d::Zero();
  labrob::Foot hac_last_support_foot_ = labrob::Foot::LEFT; // to detect switch support in HAC

  bool hac_wrist_task_was_active_ = false;  ///< previous-tick state of wrist_task_active, to detect activation edge


  // Private memebers RW-BO
  Eigen::VectorXd torques_filt_;

  Eigen::VectorXd joint_vel_filt_;   // EMA velocità di giunto per l'observer
  bool joint_vel_filt_init_ = false;

  Eigen::Vector3d prev_angular_momentum_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d L_dot_;

  // Private members WBC
  std::vector<double> friction_cone_ratios_left_x_;
  std::vector<double> friction_cone_ratios_left_y_;
  std::vector<double> friction_cone_ratios_right_x_;
  std::vector<double> friction_cone_ratios_right_y_;

  // Logs
  Logger logger_;

  // Per-solve IS-MPC snapshots (nested structure, managed separately from Logger).
  std::vector<int64_t>                        mpc_snapshot_t_log_;
  std::vector<std::vector<Eigen::VectorXd>>   mpc_snapshot_x_log_;
  std::vector<std::vector<Eigen::VectorXd>>   mpc_snapshot_u_log_;

}; // end class WalkingManager

} // end namespace labrob

#endif // LABROB_WALKING_MANAGER_HPP_