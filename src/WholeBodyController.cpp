//
// Created by mmaximo on 20/02/24.
//

#include <WholeBodyController.hpp>

#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/spatial/skew.hpp>

#include <JointCommand.hpp>
#include <utils.hpp>
#include <globals.h>

namespace labrob {

WholeBodyControllerParams WholeBodyControllerParams::getDefaultParams() {
  static WholeBodyControllerParams params;

  params.Kp_motion = 120.0;
  params.Kd_motion = 40.0;
  params.Kp_regulation = 30.0;
  params.Kd_regulation = 10.0;
  params.Kp_orientation = 400.0;
  params.Kd_orientation = 80.0;
  params.Kp_foot = 70.0;
  params.Kd_foot = 35.0;
  params.Kp_wrist = 30.0;
  params.Kd_wrist = 10.0;

  params.Kp_joint_matrix = Eigen::MatrixXd::Identity(6 + 29, 6 + 29) * 90;
  params.Kp_joint_matrix.block(6, 6, 12, 12).setZero();
  params.Kd_joint_matrix = Eigen::MatrixXd::Identity(6 + 29, 6 + 29) * 70;
  params.Kd_joint_matrix.block(6, 6, 12, 12).setZero();
  params.Kp_joint_matrix.block(12, 12, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 120;
  params.Kd_joint_matrix.block(12, 12, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 90;

  params.weight_q_ddot           = 1e-4;
  params.weight_com              = 1;
  params.weight_lsole            = 1;
  params.weight_rsole            = 1;
  params.weight_lwrist            = 1e-3;
  params.weight_rwrist            = 1e-3;
  params.weight_torso            = 1e-4;
  params.weight_pelvis           = 1e-1;
  params.weight_angular_momentum = 1e-4;
  params.weight_regulation       = 1e-4;

  params.cmm_selection_matrix_x = 1e-1;
  params.cmm_selection_matrix_y = 1e-1;
  params.cmm_selection_matrix_z = 1;

  params.beta = 50;
  params.gamma = 30;
  params.mu = 0.8;

  params.foot_length = 0.20;
  params.foot_width  = 0.07;

  return params;
}

WholeBodyController::WholeBodyController(
    const WholeBodyControllerParams& params,
    const pinocchio::Model& robot_model,
    const Eigen::VectorXd& q_jnt_reg,
    double sample_time,
    std::map<std::string, double>& armatures)
    : robot_model_(robot_model),
      q_jnt_reg_(q_jnt_reg),
      sample_time_(sample_time),
      params_(params)
{
  robot_data_ = pinocchio::Data(robot_model_);

  lsole_idx_  = robot_model_.getFrameId("left_foot_link");
  rsole_idx_  = robot_model_.getFrameId("right_foot_link");
  lwrist_idx_  = robot_model_.getFrameId("left_wrist_yaw_link");
  rwrist_idx_  = robot_model_.getFrameId("right_wrist_yaw_link");
  torso_idx_  = robot_model_.getFrameId("torso_link");
  pelvis_idx_ = robot_model_.getFrameId("pelvis");

  J_torso_     = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_pelvis_    = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_lsole_     = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_rsole_     = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_lwrist_     = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_rwrist_     = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_torso_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_pelvis_dot_= Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_lsole_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_rsole_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_lwrist_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_rwrist_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);

  n_joints_         = robot_model.nv - 6;
  n_contacts_       = 4;
  n_wbc_variables_  = 6 + n_joints_ + 2 * 3 * n_contacts_;
  n_wbc_equalities_ = 6 + 2 * 6 + 3 * n_contacts_;
  n_wbc_inequalities_= 2 * n_joints_ + 2 * 4 * n_contacts_;

  const int nj = n_joints_;
  const int nc = n_contacts_;
  const int nv = 6 + nj;

  q_ddot_ = Eigen::VectorXd::Zero(robot_model.nv);
  flr_    = Eigen::VectorXd::Zero(2 * 3 * nc);

  q_dot_des_ = Eigen::VectorXd::Zero(robot_model.nv - 6);
  q_des_ = Eigen::VectorXd::Zero(robot_model.nv - 6);

  M_armature_ = Eigen::VectorXd::Zero(nj);
  for (pinocchio::JointIndex jid = 0; jid < (pinocchio::JointIndex) nj; ++jid)
    M_armature_(jid) = armatures[robot_model_.names[jid + 2]];

  wbc_solver_ptr_ = std::make_unique<labrob::QpSolver>(
      n_wbc_variables_, n_wbc_equalities_, n_wbc_inequalities_, SPEED_ABS, 50, 1e6);

  left_foot_wrench_  = Eigen::VectorXd::Zero(6);
  right_foot_wrench_ = Eigen::VectorXd::Zero(6);

  // ── constants ─────────────────────────────────────────────────────────────

  zero_nv_ = Eigen::VectorXd::Zero(nv);

  err_posture_sel_ = Eigen::MatrixXd::Zero(nv, nv);
  err_posture_sel_.block(6, 6, nj, nj).setIdentity();

  cmm_sel_ = Eigen::MatrixXd::Zero(3, 6);
  cmm_sel_(0, 3) = params_.cmm_selection_matrix_x;
  cmm_sel_(1, 4) = params_.cmm_selection_matrix_y;
  cmm_sel_(2, 5) = params_.cmm_selection_matrix_z;

  f_force_ = Eigen::VectorXd::Zero(3 * nc);

  C_force_block_.resize(4, 3);
  C_force_block_ <<  1.0,  0.0, -params_.mu,
                      0.0,  1.0, -params_.mu,
                     -1.0,  0.0, -params_.mu,
                      0.0, -1.0, -params_.mu;

  d_min_force_ = -10000.0 * Eigen::VectorXd::Ones(4 * nc);
  d_max_force_ = Eigen::VectorXd::Zero(4 * nc);
  b_no_contact_= Eigen::VectorXd::Zero(3 * nc);

  const double fl = params_.foot_length / 2.0;
  const double fw = params_.foot_width  / 2.0;
  pcis_[0] <<  fl,  fw, 0.0;
  pcis_[1] <<  fl, -fw, 0.0;
  pcis_[2] << -fl,  fw, 0.0;
  pcis_[3] << -fl, -fw, 0.0;

  const double dt     = sample_time_;
  const double dt2h   = dt * dt / 2.0;
  C_acc_ = Eigen::MatrixXd::Zero(2 * nj, nv);
  C_acc_.rightCols(nj).topRows(nj).diagonal().setConstant(dt);
  C_acc_.rightCols(nj).bottomRows(nj).diagonal().setConstant(dt2h);

  // ── per-call buffers ──────────────────────────────────────────────────────

  err_posture_     = Eigen::VectorXd::Zero(nv);
  err_posture_vel_ = Eigen::VectorXd::Zero(nv);
  desired_qddot_   = Eigen::VectorXd::Zero(nv);
  H_acc_           = Eigen::MatrixXd::Zero(nv, nv);
  f_acc_           = Eigen::VectorXd::Zero(nv);
  d_min_acc_       = Eigen::VectorXd::Zero(2 * nj);
  d_max_acc_       = Eigen::VectorXd::Zero(2 * nj);
  T_l_             = Eigen::MatrixXd::Zero(6, 3 * nc);
  T_r_             = Eigen::MatrixXd::Zero(6, 3 * nc);
  M_inertia_       = Eigen::MatrixXd::Zero(nv, nv);
  Mu_              = Eigen::MatrixXd::Zero(6, nv);
  Ma_              = Eigen::MatrixXd::Zero(nj, nv);
  cu_              = Eigen::VectorXd::Zero(6);
  ca_              = Eigen::VectorXd::Zero(nj);
  Jlu_             = Eigen::MatrixXd::Zero(6, 6);
  Jla_             = Eigen::MatrixXd::Zero(6, nj);
  Jru_             = Eigen::MatrixXd::Zero(6, 6);
  Jra_             = Eigen::MatrixXd::Zero(6, nj);
  A_acc_wbc_       = Eigen::MatrixXd::Zero(12, nv);
  b_acc_wbc_       = Eigen::VectorXd::Zero(12);
  A_no_contact_    = Eigen::MatrixXd::Zero(3 * nc, 2 * 3 * nc);
  A_dyn_           = Eigen::MatrixXd::Zero(6, n_wbc_variables_);
  A_wbc_           = Eigen::MatrixXd::Zero(n_wbc_equalities_, n_wbc_variables_);
  b_wbc_           = Eigen::VectorXd::Zero(n_wbc_equalities_);
  C_force_left_    = Eigen::MatrixXd::Zero(4 * nc, 3 * nc);
  C_force_right_   = Eigen::MatrixXd::Zero(4 * nc, 3 * nc);
  d_min_wbc_       = Eigen::VectorXd::Zero(n_wbc_inequalities_);
  d_max_wbc_       = Eigen::VectorXd::Zero(n_wbc_inequalities_);

  // H_wbc: force-regularization blocks are constant
  H_wbc_ = Eigen::MatrixXd::Zero(n_wbc_variables_, n_wbc_variables_);
  H_wbc_.block(nv,       nv,       3*nc, 3*nc).diagonal().setConstant(1e-9);
  H_wbc_.block(nv+3*nc, nv+3*nc,   3*nc, 3*nc).diagonal().setConstant(1e-9);

  // f_wbc: force part is always zero
  f_wbc_ = Eigen::VectorXd::Zero(n_wbc_variables_);

  // C_wbc: C_acc block is constant; zero-pad the rest
  C_wbc_ = Eigen::MatrixXd::Zero(n_wbc_inequalities_, n_wbc_variables_);
  C_wbc_.block(0, 0, 2*nj, nv) = C_acc_;

  // d_min/max_wbc: force bounds are constant
  d_min_wbc_.segment(2*nj,       4*nc) = d_min_force_;
  d_min_wbc_.segment(2*nj+4*nc, 4*nc) = d_min_force_;
  d_max_wbc_.segment(2*nj,       4*nc) = d_max_force_;
  d_max_wbc_.segment(2*nj+4*nc, 4*nc) = d_max_force_;
}

labrob::JointCommand
WholeBodyController::compute_inverse_dynamics(
    const pinocchio::Model& robot_model,
    const labrob::RobotState& robot_state,
    pinocchio::Data& robot_data,
    const labrob::GaitConfiguration& current,
    const labrob::GaitConfiguration& desired)
{
  const int nj = n_joints_;
  const int nc = n_contacts_;
  const int nv = 6 + nj;

  auto q    = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
  auto qdot = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);

  pinocchio::jacobianCenterOfMass(robot_model, robot_data, q);
  pinocchio::computeJointJacobiansTimeVariation(robot_model, robot_data, q, qdot);
  pinocchio::framesForwardKinematics(robot_model, robot_data, q);

  pinocchio::getFrameJacobian(robot_model, robot_data, torso_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_torso_);
  pinocchio::getFrameJacobian(robot_model, robot_data, pelvis_idx_, pinocchio::LOCAL_WORLD_ALIGNED, J_pelvis_);
  pinocchio::getFrameJacobian(robot_model, robot_data, lsole_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_lsole_);
  pinocchio::getFrameJacobian(robot_model, robot_data, rsole_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_rsole_);
  pinocchio::getFrameJacobian(robot_model, robot_data, lwrist_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_lwrist_);
  pinocchio::getFrameJacobian(robot_model, robot_data, rwrist_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_rwrist_);

  pinocchio::centerOfMass(robot_model, robot_data, q, qdot, zero_nv_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, torso_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_torso_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, pelvis_idx_, pinocchio::LOCAL_WORLD_ALIGNED, J_pelvis_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, lsole_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_lsole_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, rsole_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_rsole_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, lwrist_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_lwrist_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, rwrist_idx_,  pinocchio::LOCAL_WORLD_ALIGNED, J_rwrist_dot_);

  const auto& J_com                    = robot_data.Jcom;
  const auto& centroidal_momentum_matrix = pinocchio::ccrba(robot_model, robot_data, q, qdot);
  const auto& a_com_drift              = robot_data.acom[0];
  const auto  a_lsole_drift            = J_lsole_dot_ * qdot;
  const auto  a_rsole_drift            = J_rsole_dot_ * qdot;
  const auto  a_lwrist_drift            = J_lwrist_dot_.topRows<3>() * qdot;
  const auto  a_rwrist_drift            = J_rwrist_dot_.topRows<3>() * qdot;
  const auto  a_torso_drift            = J_torso_dot_.bottomRows<3>() * qdot;
  const auto  a_pelvis_drift           = J_pelvis_dot_.bottomRows<3>() * qdot;

  // Desired accelerations
  const auto err_com      = desired.com.pos - current.com.pos;
  const auto err_com_vel  = desired.com.vel - current.com.vel;

  const auto err_lsole     = err_frameplacement(
      pinocchio::SE3(desired.lsole.pos.R, desired.lsole.pos.p),
      pinocchio::SE3(current.lsole.pos.R, current.lsole.pos.p));
  const auto err_lsole_vel = desired.lsole.vel - current.lsole.vel;

  const auto err_rsole     = err_frameplacement(
      pinocchio::SE3(desired.rsole.pos.R, desired.rsole.pos.p),
      pinocchio::SE3(current.rsole.pos.R, current.rsole.pos.p));
  const auto err_rsole_vel = desired.rsole.vel - current.rsole.vel;

  const auto err_lwrist     = desired.lwrist.pos - current.lwrist.pos;
  const auto err_lwrist_vel = desired.lwrist.vel - current.lwrist.vel;

  const auto err_rwrist     = desired.rwrist.pos - current.rwrist.pos;
  const auto err_rwrist_vel = desired.rwrist.vel - current.rwrist.vel;

  const auto err_torso         = err_rotation(desired.torso.pos, current.torso.pos);
  const auto err_torso_vel     = desired.torso.vel - current.torso.vel;
  const auto err_pelvis        = err_rotation(desired.pelvis.pos, current.pelvis.pos);
  const auto err_pelvis_vel    = desired.pelvis.vel - current.pelvis.vel;

  // err_posture / err_posture_vel (pre-allocated member vectors)
  err_posture_.head(6).setZero();
  err_posture_.tail(nj)     = desired.qjnt    - current.qjnt;
  err_posture_vel_.head(6).setZero();
  err_posture_vel_.tail(nj) = desired.qjntdot - current.qjntdot;

  desired_qddot_.head(6).setZero();
  desired_qddot_.tail(nj) = desired.qjntddot;

  const Eigen::VectorXd a_jnt_total =
      desired_qddot_ + params_.Kp_joint_matrix * err_posture_ + params_.Kd_joint_matrix * err_posture_vel_;
  const Eigen::VectorXd a_com_total =
      desired.com.acc + params_.Kp_motion * err_com + params_.Kd_motion * err_com_vel;
  const Eigen::VectorXd a_lsole_total =
      desired.lsole.acc + params_.Kp_foot * err_lsole + params_.Kd_foot * err_lsole_vel;
  const Eigen::VectorXd a_rsole_total =
      desired.rsole.acc + params_.Kp_foot * err_rsole + params_.Kd_foot * err_rsole_vel;
  const Eigen::VectorXd a_lwrist_total =
      desired.lwrist.acc + params_.Kp_wrist * err_lwrist + params_.Kd_wrist * err_lwrist_vel;
  const Eigen::VectorXd a_rwrist_total =
      desired.rwrist.acc + params_.Kp_wrist * err_rwrist + params_.Kd_wrist * err_rwrist_vel;
  const Eigen::VectorXd a_torso_total =
      desired.torso.acc + params_.Kp_orientation * err_torso + params_.Kd_orientation * err_torso_vel;
  const Eigen::VectorXd a_pelvis_total =
      desired.pelvis.acc + params_.Kp_orientation * err_pelvis + params_.Kd_orientation * err_pelvis_vel;

  // print each accelerations
  // std::cout << "a_jnt_total " << a_jnt_total.transpose() << "\n" << std::endl;
  // std::cout << "a_com_total " << a_com_total.transpose() << "\n" << std::endl;
  // std::cout << "a_lsole_total " << a_lsole_total.transpose() << "\n" << std::endl;
  // std::cout << "a_rsole_total " << a_rsole_total.transpose() << "\n" << std::endl;
  // std::cout << "a_torso_total " << a_torso_total.transpose() << "\n" << std::endl;
  // std::cout << "a_pelvis_total " << a_pelvis_total.transpose() << "\n" << std::endl;


  // std::cout << "error posture pos " << err_posture_.transpose() << " vel " << err_posture_vel_.transpose() << "\n" << std::endl;
  // std::cout << "error rsole pos " << err_rsole.transpose() << " vel " << err_rsole_vel.transpose() << "\n" << std::endl;

  // ── H_acc / f_acc (no temporaries, noalias products) ─────────────────────
  H_acc_.setZero();
  H_acc_.diagonal().setConstant(params_.weight_q_ddot);
  H_acc_.noalias() += params_.weight_com     * (J_com.transpose()                 * J_com);
  H_acc_.noalias() += params_.weight_lsole   * (J_lsole_.transpose()              * J_lsole_);
  H_acc_.noalias() += params_.weight_rsole   * (J_rsole_.transpose()              * J_rsole_);
  H_acc_.noalias() += params_.weight_lwrist   * (J_lwrist_.topRows<3>().transpose() * J_lwrist_.topRows<3>());
  H_acc_.noalias() += params_.weight_rwrist   * (J_rwrist_.topRows<3>().transpose() * J_rwrist_.topRows<3>());
  H_acc_.noalias() += params_.weight_torso   * (J_torso_.bottomRows<3>().transpose()  * J_torso_.bottomRows<3>());
  H_acc_.noalias() += params_.weight_pelvis  * (J_pelvis_.bottomRows<3>().transpose() * J_pelvis_.bottomRows<3>());
  H_acc_.noalias() += params_.weight_regulation * err_posture_sel_;
  {
    const double w_amom = params_.weight_angular_momentum * sample_time_ * sample_time_;
    const Eigen::MatrixXd cmm_A = cmm_sel_ * centroidal_momentum_matrix;
    H_acc_.noalias() += w_amom * cmm_A.transpose() * cmm_A;
  }

  f_acc_.setZero();
  f_acc_.noalias() += params_.weight_com    * J_com.transpose()                 * (a_com_drift - a_com_total);
  f_acc_.noalias() += params_.weight_lsole  * J_lsole_.transpose()              * (a_lsole_drift - a_lsole_total);
  f_acc_.noalias() += params_.weight_rsole  * J_rsole_.transpose()              * (a_rsole_drift - a_rsole_total);
  f_acc_.noalias() += params_.weight_lwrist  * J_lwrist_.topRows<3>().transpose() * (a_lwrist_drift - a_lwrist_total);
  f_acc_.noalias() += params_.weight_rwrist  * J_rwrist_.topRows<3>().transpose() * (a_rwrist_drift - a_rwrist_total);
  f_acc_.noalias() += params_.weight_torso  * J_torso_.bottomRows<3>().transpose()  * (a_torso_drift  - a_torso_total);
  f_acc_.noalias() += params_.weight_pelvis * J_pelvis_.bottomRows<3>().transpose() * (a_pelvis_drift - a_pelvis_total);
  f_acc_.noalias() -= params_.weight_regulation * err_posture_sel_ * a_jnt_total;
  {
    const double w_amom_dt = params_.weight_angular_momentum * sample_time_;
    const Eigen::MatrixXd cmm_A = cmm_sel_ * centroidal_momentum_matrix;
    f_acc_.noalias() += w_amom_dt * cmm_A.transpose() * (cmm_A * qdot);
  }

  // Velocity / position limits
  const auto q_jnt_dot_min = -robot_model.velocityLimit.tail(nj);
  const auto q_jnt_dot_max =  robot_model.velocityLimit.tail(nj);
  const auto q_jnt_min     =  robot_model.lowerPositionLimit.tail(nj);
  const auto q_jnt_max     =  robot_model.upperPositionLimit.tail(nj);

  d_min_acc_ << q_jnt_dot_min - current.qjntdot,
                q_jnt_min - current.qjnt - sample_time_ * current.qjntdot;
  d_max_acc_ << q_jnt_dot_max - current.qjntdot,
                q_jnt_max - current.qjnt - sample_time_ * current.qjntdot;

  // Inertia matrix (pinocchio::crba returns const& to data.M — copy once)
  M_inertia_ = pinocchio::crba(robot_model, robot_data, q);
  M_inertia_.triangularView<Eigen::StrictlyLower>() =
      M_inertia_.transpose().triangularView<Eigen::StrictlyLower>();
  M_inertia_.diagonal().tail(nj) += M_armature_;

  const auto& c = pinocchio::rnea(robot_model, robot_data, q, qdot, zero_nv_);

  Mu_ = M_inertia_.topRows(6);
  Ma_ = M_inertia_.bottomRows(nj);
  cu_ = c.head(6);
  ca_ = c.tail(nj);
  Jlu_ = J_lsole_.leftCols(6);
  Jla_ = J_lsole_.rightCols(nj);
  Jru_ = J_rsole_.leftCols(6);
  Jra_ = J_rsole_.rightCols(nj);

  // Rotated contact points
  for (int i = 0; i < nc; ++i) {
    pcis_l_[i] = desired.lsole.pos.R * pcis_[i];
    pcis_r_[i] = desired.rsole.pos.R * pcis_[i];
  }

  // Grasp matrices T_l, T_r
  T_l_.setZero(); T_r_.setZero();
  for (int i = 0; i < nc; ++i) {
    T_l_.block(0, 3*i, 3, 3).setIdentity();
    T_r_.block(0, 3*i, 3, 3).setIdentity();
    T_l_.block(3, 3*i, 3, 3) = pinocchio::skew(pcis_l_[i]);
    T_r_.block(3, 3*i, 3, 3) = pinocchio::skew(pcis_r_[i]);
  }

  // ── combined QP matrices (reuse pre-allocated buffers) ───────────────────

  // H_wbc: update acceleration block (force blocks already set in constructor)
  H_wbc_.topLeftCorner(nv, nv) = H_acc_;
  // f_wbc: update acceleration part (force part stays zero)
  f_wbc_.head(nv) = f_acc_;

  // A_wbc
  A_acc_wbc_.setZero(); b_acc_wbc_.setZero();
  if (current.is_left_foot_support) {
    A_acc_wbc_.topRows(6) = J_lsole_;
    b_acc_wbc_.head(6)    = -J_lsole_dot_*qdot
                            - params_.gamma * J_lsole_*qdot
                            + params_.beta  * err_lsole;
  }
  if (current.is_right_foot_support) {
    A_acc_wbc_.bottomRows(6) = J_rsole_;
    b_acc_wbc_.tail(6)       = -J_rsole_dot_*qdot
                               - params_.gamma * J_rsole_*qdot
                               + params_.beta  * err_rsole;
  }

  A_no_contact_.setZero();
  if (!current.is_left_foot_support)
    A_no_contact_.block(0,     0,     3*nc, 3*nc).setIdentity();
  if (!current.is_right_foot_support)
    A_no_contact_.block(0, 3*nc, 3*nc, 3*nc).setIdentity();

  A_dyn_.setZero();
  A_dyn_.leftCols(nv)               = Mu_;
  A_dyn_.block(0, nv,      6, 3*nc).noalias() = -(Jlu_.transpose() * T_l_);
  A_dyn_.block(0, nv+3*nc, 6, 3*nc).noalias() = -(Jru_.transpose() * T_r_);

  A_wbc_.setZero();
  A_wbc_.block(0,        0,        12,    nv)    = A_acc_wbc_;
  A_wbc_.block(12,       nv,       3*nc,  2*3*nc) = A_no_contact_;
  A_wbc_.bottomRows(6)                           = A_dyn_;

  b_wbc_.head(12)             = b_acc_wbc_;
  b_wbc_.segment(12, 3*nc)    = b_no_contact_;
  b_wbc_.tail(6)              = -cu_;

  // C_wbc: update only friction blocks (C_acc block set in constructor)
  for (int i = 0; i < nc; ++i) {
    C_force_left_.block(4*i, 3*i, 4, 3).noalias()  = C_force_block_ * current.lsole.pos.R.transpose();
    C_force_right_.block(4*i, 3*i, 4, 3).noalias() = C_force_block_ * current.rsole.pos.R.transpose();
  }
  C_wbc_.block(2*nj,       nv,       4*nc, 3*nc) = C_force_left_;
  C_wbc_.block(2*nj+4*nc, nv+3*nc, 4*nc, 3*nc) = C_force_right_;

  // d_min/max_wbc: only update acceleration part (force part set in constructor)
  d_min_wbc_.head(2*nj) = d_min_acc_;
  d_max_wbc_.head(2*nj) = d_max_acc_;

  // Solve
  const Eigen::VectorXd prev_q_ddot = q_ddot_;
  const Eigen::VectorXd prev_flr    = flr_;

  wbc_solver_ptr_->solve(H_wbc_, f_wbc_, A_wbc_, b_wbc_, C_wbc_, d_min_wbc_, d_max_wbc_);

  if (!wbc_solver_ptr_->has_valid_solution()) {
    q_ddot_ = prev_q_ddot;
    flr_    = prev_flr;
  } else {
    const auto& solution = wbc_solver_ptr_->get_solution();
    q_ddot_ = solution.head(nv);
    flr_    = solution.segment(nv, 2 * 3 * nc);
  }

  const auto& fl = flr_.head(3 * nc);
  const auto& fr = flr_.tail(3 * nc);
  left_foot_wrench_  = T_l_ * fl;
  right_foot_wrench_ = T_r_ * fr;

  // Online reference generation with semi-implicit Euler integration
  q_dot_des_ = qdot.tail(nj) + sample_time_ * q_ddot_.tail(nj);
  q_des_ = q.tail(nj) + sample_time_ * q_dot_des_ + 0.5 * (sample_time_ * sample_time_) * q_ddot_.tail(nj);


  Eigen::VectorXd Kd_vec = Eigen::VectorXd::Zero(nj);
  Kd_vec << 2, 2, 2, 3, 2, 2,
            2, 2, 2, 3, 2, 2,
            2, 2, 2,
            2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2;
  Eigen::MatrixXd Kd = Kd_vec.asDiagonal();

  Eigen::VectorXd Kp_vec = Eigen::VectorXd::Zero(nj);
  Kp_vec << 40, 40, 40, 60, 40, 30,    // left leg
            40, 40, 40, 60, 40, 30,    // right leg
            25, 25, 15,                // waist
            12, 12, 12, 7,  4, 4, 4,   // left arm
            12, 12, 12, 7,  4, 4, 4;   // right arm
  

  Eigen::MatrixXd Kp = Kp_vec.asDiagonal();

  
  const Eigen::VectorXd tau = Ma_ * q_ddot_ + ca_
      - Jla_.transpose() * left_foot_wrench_
      - Jra_.transpose() * right_foot_wrench_
      + Kd * (q_dot_des_ - qdot.tail(nj))
      + Kp * (q_des_ - q.tail(nj));

      
  // Check for limit exceeding
  /*
  double upper_limit, lower_limit;
  for (pinocchio::JointIndex jid = 0; jid < (pinocchio::JointIndex) nj; ++jid) {
    
    lower_limit = joint_limits.at(robot_model.names[jid + 2]).lower;
    upper_limit = joint_limits.at(robot_model.names[jid + 2]).upper;

    if (q_des_[jid] < lower_limit)
      std::cout << "Joint " << robot_model.names[jid + 2] << " exceeding lower bound by " << lower_limit - q_des_[jid] << std::endl;

    if (q_des_[jid] > upper_limit)
      std::cout << "Joint " << robot_model.names[jid + 2] << " exceeding upper bound by " << q_des_[jid] - upper_limit << std::endl;
  }
  */

  JointCommand joint_command;
  for (pinocchio::JointIndex jid = 0; jid < (pinocchio::JointIndex) nj; ++jid)
    joint_command[robot_model.names[jid + 2]] = tau[jid];

  return joint_command;
}

} // namespace labrob