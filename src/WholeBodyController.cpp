//
// Created by mmaximo on 20/02/24.
//

#include <hrp4_locomotion/WholeBodyController.hpp>

#include <initializer_list>

// Pinocchio
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>

#include <hrp4_locomotion/JointCommand.hpp>
#include <hrp4_locomotion/utils.hpp>

namespace labrob {

WholeBodyControllerParams WholeBodyControllerParams::getDefaultParams() {
  static WholeBodyControllerParams params;

  params.Kp_motion = 30.0;
  params.Kd_motion = 10.0;
  params.Kp_torso_motion = Eigen::Vector3d::Constant(50.0);
  params.Kd_torso_motion = Eigen::Vector3d::Constant(10.0);
  params.Kp_hand_compliance =
      Eigen::Matrix<double, 6, 1>::Constant(30.0);
  params.Kd_hand_compliance =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  params.S_hand_compliance.setZero();
  params.S_hand_compliance.topLeftCorner<3, 3>().setIdentity();
  params.Kp_regulation = 30.0;
  params.Kd_regulation = 10.0;

  params.weight_q_ddot = 1e-4;
  params.weight_com = 1e-1;
  params.weight_lsole = 1;
  params.weight_rsole = 1;
  params.weight_torso = 1e-1;
  params.weight_lhand_compliance = 2e-2;
  params.weight_rhand_compliance = 2e-2;
  params.weight_angular_momentum = 1e-4;
  params.weight_regulation = 1e-4;
  params.weight_regulation_matrix = Eigen::MatrixXd();
  // params.weight_regulation_matrix.block(10, 10, 1, 1) = Eigen::MatrixXd::Identity(1, 1) * 1e-3;
  // params.weight_regulation_matrix.block(16, 16, 1, 1) = Eigen::MatrixXd::Identity(1, 1) * 1e-3;
  //fare matrice identità e asssegnare valori più alti per l'ankle roll

  params.cmm_selection_matrix_x = 1e-6;
  params.cmm_selection_matrix_y = 1e-6;
  params.cmm_selection_matrix_z = 1e-4;

  params.beta = 10;
  params.gamma = 10;
  params.mu = 0.5;

  params.foot_length = 0.20;
  params.foot_width = 0.07; 

  return params;
}

WholeBodyController::WholeBodyController(
    const WholeBodyControllerParams& params, const pinocchio::Model& robot_model,
    const Eigen::VectorXd& q_jnt_reg,
    double sample_time,
    std::map<std::string, double>& armatures)
    : robot_model_(robot_model),
      q_jnt_reg_(q_jnt_reg),
      sample_time_(sample_time),
      params_(params)
{

  robot_data_ = pinocchio::Data(robot_model_);

  lsole_idx_ = robot_model_.getFrameId("left_foot_link");
  rsole_idx_ = robot_model_.getFrameId("right_foot_link");
  torso_idx_ = robot_model_.getFrameId("torso_link");
  lhand_idx_ = robot_model_.getFrameId("left_wrist_yaw_joint");
  rhand_idx_ = robot_model_.getFrameId("right_wrist_yaw_joint");

  J_torso_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_lsole_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_rsole_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_lhand_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_rhand_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);

  J_torso_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_lsole_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_rsole_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_lhand_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  J_rhand_dot_ = Eigen::MatrixXd::Zero(6, robot_model_.nv);

  const auto collect_velocity_indices = [this](
      std::initializer_list<const char*> joint_names) {
    std::vector<int> indices;
    indices.reserve(joint_names.size());
    for (const char* joint_name : joint_names) {
      if (!robot_model_.existJointName(joint_name)) {
        continue;
      }
      const pinocchio::JointIndex joint_id =
          robot_model_.getJointId(joint_name);
      const auto& joint = robot_model_.joints[joint_id];
      for (int k = 0; k < joint.nv(); ++k) {
        indices.push_back(joint.idx_v() + k);
      }
    }
    return indices;
  };

  left_arm_velocity_indices_ = collect_velocity_indices({
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint"});
  right_arm_velocity_indices_ = collect_velocity_indices({
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint"});

  q_ddot_ = Eigen::VectorXd::Zero(robot_model.nv);
  flr = Eigen::VectorXd::Zero(2 * 3 * 4); // 2 feet, 3 forces per foot, 4 contacts

  n_joints_ = robot_model.nv - 6;
  n_contacts_ = 4;
  n_wbc_variables_ = 6 + n_joints_ + 2 * 3 * n_contacts_;
  n_wbc_equalities_ = 6 + 2 * 6 + 3 * n_contacts_;
  n_wbc_inequalities_ = 2 * n_joints_ + 2 * 4 * n_contacts_;

  const int state_acc_size = 6 + n_joints_;
  if (params_.weight_regulation_matrix.rows() != state_acc_size ||
      params_.weight_regulation_matrix.cols() != state_acc_size) {
    params_.weight_regulation_matrix =
        Eigen::MatrixXd::Identity(state_acc_size, state_acc_size) *
        params_.weight_regulation;
  }

  M_armature_ = Eigen::VectorXd::Zero(n_joints_);
  for (pinocchio::JointIndex joint_id = 0;
       joint_id < (pinocchio::JointIndex) n_joints_;
       ++joint_id) {
    std::string joint_name = robot_model_.names[joint_id + 2];
    M_armature_(joint_id) = armatures[joint_name];
  }

  wbc_solver_ptr_ = std::make_unique<qpsolvers::QPSolverEigenWrapper<double>>(
      std::make_shared<qpsolvers::HPIPMQPSolver>(
          n_wbc_variables_, n_wbc_equalities_ , n_wbc_inequalities_
      )
  );
}

labrob::JointCommand
WholeBodyController::compute_inverse_dynamics(
    const pinocchio::Model& robot_model,
    const labrob::RobotState& robot_state,
    const labrob::RobotState& fb_filt_robot_state,
    pinocchio::Data& robot_data,
    pinocchio::Data& fb_robot_data,
    const labrob::GaitConfiguration& current,
    const labrob::GaitConfiguration& desired,
    const Eigen::Matrix<double, 6, 1>& left_interaction_wrench,
    const Eigen::Matrix<double, 6, 1>& right_interaction_wrench
) {

  auto q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
  auto qdot = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);

  auto q_fb_filt = robot_state_to_pinocchio_joint_configuration(robot_model_, fb_filt_robot_state);
  auto qdot_fb_filt = robot_state_to_pinocchio_joint_velocity(robot_model_, fb_filt_robot_state);

  // Compute pinocchio terms
  pinocchio::jacobianCenterOfMass(robot_model, robot_data, q);
  pinocchio::computeJointJacobiansTimeVariation(robot_model, robot_data, q, qdot);
  pinocchio::framesForwardKinematics(robot_model, robot_data, q);

  pinocchio::jacobianCenterOfMass(robot_model, fb_robot_data, q_fb_filt);
  pinocchio::framesForwardKinematics(robot_model, fb_robot_data, q_fb_filt);

  pinocchio::getFrameJacobian(robot_model, robot_data, torso_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_torso_);
  pinocchio::getFrameJacobian(robot_model, robot_data, lsole_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lsole_);
  pinocchio::getFrameJacobian(robot_model, robot_data, rsole_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rsole_);
  pinocchio::getFrameJacobian(robot_model, robot_data, lhand_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lhand_);
  pinocchio::getFrameJacobian(robot_model, robot_data, rhand_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rhand_);

  pinocchio::centerOfMass(robot_model, robot_data, q, qdot, 0.0 * qdot); // This is to compute the drift term
  pinocchio::centerOfMass(robot_model, fb_robot_data, q_fb_filt, qdot_fb_filt, 0.0 * qdot); // This is to compute the drift term
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, torso_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_torso_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, lsole_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lsole_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, rsole_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rsole_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, lhand_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lhand_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model, robot_data, rhand_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rhand_dot_);

  // Eq. (21) is a residual *arm* task. Masking the torso/base/opposite-arm
  // columns prevents this task from cancelling the torso motion allocated by
  // the CRG. The Jdot drift must use the same mask for the same reason.
  Eigen::MatrixXd J_lhand_arm =
      Eigen::MatrixXd::Zero(6, robot_model_.nv);
  Eigen::MatrixXd J_rhand_arm =
      Eigen::MatrixXd::Zero(6, robot_model_.nv);
  Eigen::MatrixXd J_lhand_arm_dot =
      Eigen::MatrixXd::Zero(6, robot_model_.nv);
  Eigen::MatrixXd J_rhand_arm_dot =
      Eigen::MatrixXd::Zero(6, robot_model_.nv);
  for (const int index : left_arm_velocity_indices_) {
    if (index >= 0 && index < robot_model_.nv) {
      J_lhand_arm.col(index) = J_lhand_.col(index);
      J_lhand_arm_dot.col(index) = J_lhand_dot_.col(index);
    }
  }
  for (const int index : right_arm_velocity_indices_) {
    if (index >= 0 && index < robot_model_.nv) {
      J_rhand_arm.col(index) = J_rhand_.col(index);
      J_rhand_arm_dot.col(index) = J_rhand_dot_.col(index);
    }
  }

  // Apply the Cartesian task selector to the complete hand-task model, not
  // only to its reference. This removes unselected angular costs from both H
  // and f and keeps the achieved-acceleration log in the same selected space.
  J_lhand_arm = params_.S_hand_compliance * J_lhand_arm;
  J_lhand_arm_dot = params_.S_hand_compliance * J_lhand_arm_dot;
  J_rhand_arm = params_.S_hand_compliance * J_rhand_arm;
  J_rhand_arm_dot = params_.S_hand_compliance * J_rhand_arm_dot;

  const auto& J_com = fb_robot_data.Jcom;
  const auto& centroidal_momentum_matrix = pinocchio::ccrba(robot_model, robot_data, q, qdot);
  const auto& a_com_drift = fb_robot_data.acom[0];
  const auto a_lsole_drift = J_lsole_dot_ * qdot;
  const auto a_rsole_drift = J_rsole_dot_ * qdot;
  const auto a_torso_orientation_drift = J_torso_dot_.bottomRows<3>() * qdot;
  const Eigen::Matrix<double, 6, 1> a_lhand_arm_drift =
      J_lhand_arm_dot * qdot;
  const Eigen::Matrix<double, 6, 1> a_rhand_arm_drift =
      J_rhand_arm_dot * qdot;

  // Compute desired accelerations
  auto err_com = desired.com.pos - current.com.pos;
  auto err_com_vel = desired.com.vel - current.com.vel;

  auto err_lsole = err_frameplacement(
      pinocchio::SE3(desired.lsole.pos.R, desired.lsole.pos.p),
      pinocchio::SE3(current.lsole.pos.R, current.lsole.pos.p)
  );
  auto err_lsole_vel = desired.lsole.vel - current.lsole.vel;

  auto err_rsole = err_frameplacement(
      pinocchio::SE3(desired.rsole.pos.R, desired.rsole.pos.p),
      pinocchio::SE3(current.rsole.pos.R, current.rsole.pos.p)
  );
  auto err_rsole_vel = desired.rsole.vel - current.rsole.vel;

  auto err_torso_orientation = err_rotation(desired.torso.pos, current.torso.pos);
  auto err_torso_orientation_vel = desired.torso.vel - current.torso.vel;

  Eigen::VectorXd err_posture(6 + n_joints_);
  err_posture << Eigen::VectorXd::Zero(6), desired.qjnt - current.qjnt;
  Eigen::VectorXd err_posture_vel(6 + n_joints_); 
  err_posture_vel << Eigen::VectorXd::Zero(6), desired.qjntdot - current.qjntdot;
  Eigen::MatrixXd err_posture_selection_matrix = Eigen::MatrixXd::Zero(6 + n_joints_, 6 + n_joints_);
  err_posture_selection_matrix.block(6, 6, n_joints_, n_joints_) = Eigen::MatrixXd::Identity(n_joints_, n_joints_);

  Eigen::MatrixXd cmm_selection_matrix = Eigen::MatrixXd::Zero(3, 6);
  cmm_selection_matrix(0, 3) = params_.cmm_selection_matrix_x;
  cmm_selection_matrix(1, 4) = params_.cmm_selection_matrix_y;
  cmm_selection_matrix(2, 5) = params_.cmm_selection_matrix_z;

  Eigen::VectorXd desired_qddot(6 + n_joints_);
  desired_qddot << Eigen::VectorXd::Zero(6), desired.qjntddot;
  Eigen::VectorXd a_jnt_total = desired_qddot + params_.Kp_regulation * err_posture + params_.Kd_regulation * err_posture_vel;
  Eigen::VectorXd a_com_total = desired.com.acc + params_.Kp_motion * err_com + params_.Kd_motion * err_com_vel;
  Eigen::VectorXd a_lsole_total = desired.lsole.acc + params_.Kp_motion * err_lsole + params_.Kd_motion * err_lsole_vel;
  Eigen::VectorXd a_rsole_total = desired.rsole.acc + params_.Kp_motion * err_rsole + params_.Kd_motion * err_rsole_vel;
  Eigen::VectorXd a_torso_orientation_total =
      desired.torso.acc +
      params_.Kp_torso_motion.cwiseProduct(err_torso_orientation) +
      params_.Kd_torso_motion.cwiseProduct(err_torso_orientation_vel);
  const Eigen::Matrix<double, 6, 1> a_lhand_compliance_correction =
      params_.S_hand_compliance *
      (params_.Kp_hand_compliance.cwiseProduct(
           desired.lhand_compliance_offset) +
       params_.Kd_hand_compliance.cwiseProduct(
           desired.lhand_compliance_velocity));
  const Eigen::Matrix<double, 6, 1> a_rhand_compliance_correction =
      params_.S_hand_compliance *
      (params_.Kp_hand_compliance.cwiseProduct(
           desired.rhand_compliance_offset) +
       params_.Kd_hand_compliance.cwiseProduct(
           desired.rhand_compliance_velocity));
  // Eq. (21) adds the compliance correction to the nominal hand
  // acceleration. This controller has no independent nominal Cartesian hand
  // trajectory, so use the acceleration induced by its nominal arm posture
  // task. Omitting this term would turn a zero CRG residual into a command for
  // zero wrist acceleration and fight the existing whole-body motion.
  const Eigen::Matrix<double, 6, 1> a_lhand_nominal =
      J_lhand_arm * a_jnt_total + a_lhand_arm_drift;
  const Eigen::Matrix<double, 6, 1> a_rhand_nominal =
      J_rhand_arm * a_jnt_total + a_rhand_arm_drift;
  const Eigen::Matrix<double, 6, 1> a_lhand_compliance_total =
      a_lhand_nominal + a_lhand_compliance_correction;
  const Eigen::Matrix<double, 6, 1> a_rhand_compliance_total =
      a_rhand_nominal + a_rhand_compliance_correction;
  if (desired.use_hand_compliance) {
    left_hand_compliance_acceleration_reference_ =
        a_lhand_compliance_total;
    right_hand_compliance_acceleration_reference_ =
        a_rhand_compliance_total;
  } else {
    left_hand_compliance_acceleration_reference_.setZero();
    right_hand_compliance_acceleration_reference_.setZero();
  }

  // Build cost function
  Eigen::MatrixXd H_acc = Eigen::MatrixXd::Zero(6 + n_joints_, 6 + n_joints_);
  Eigen::VectorXd f_acc = Eigen::VectorXd::Zero(6 + n_joints_);

  H_acc += params_.weight_q_ddot * Eigen::MatrixXd::Identity(6 + n_joints_, 6 + n_joints_);
  H_acc += params_.weight_com * (J_com.transpose() * J_com);
  H_acc += params_.weight_lsole * (J_lsole_.transpose() * J_lsole_);
  H_acc += params_.weight_rsole * (J_rsole_.transpose() * J_rsole_);
  H_acc += params_.weight_torso * (J_torso_.bottomRows<3>().transpose() * J_torso_.bottomRows<3>());
  if (desired.use_hand_compliance) {
    H_acc += params_.weight_lhand_compliance *
        (J_lhand_arm.transpose() * J_lhand_arm);
    H_acc += params_.weight_rhand_compliance *
        (J_rhand_arm.transpose() * J_rhand_arm);
  }
  H_acc += params_.weight_regulation_matrix * err_posture_selection_matrix;
  H_acc += params_.weight_angular_momentum * centroidal_momentum_matrix.transpose() * cmm_selection_matrix.transpose() *
      std::pow(sample_time_, 2.0) * cmm_selection_matrix * centroidal_momentum_matrix;

  f_acc += params_.weight_com * J_com.transpose() * (a_com_drift - a_com_total);
  f_acc += params_.weight_lsole * J_lsole_.transpose() * (a_lsole_drift - a_lsole_total);
  f_acc += params_.weight_rsole * J_rsole_.transpose() * (a_rsole_drift - a_rsole_total);
  f_acc += params_.weight_torso * J_torso_.bottomRows<3>().transpose() * (a_torso_orientation_drift - a_torso_orientation_total);
  if (desired.use_hand_compliance) {
    f_acc += params_.weight_lhand_compliance *
        J_lhand_arm.transpose() *
        (a_lhand_arm_drift - a_lhand_compliance_total);
    f_acc += params_.weight_rhand_compliance *
        J_rhand_arm.transpose() *
        (a_rhand_arm_drift - a_rhand_compliance_total);
  }
  f_acc += -params_.weight_regulation_matrix * err_posture_selection_matrix * a_jnt_total;
  f_acc += params_.weight_angular_momentum * centroidal_momentum_matrix.transpose() * cmm_selection_matrix.transpose() *
      sample_time_ * cmm_selection_matrix * centroidal_momentum_matrix * qdot;

  auto q_jnt_dot_min = -robot_model.velocityLimit.tail(n_joints_);
  auto q_jnt_dot_max = robot_model.velocityLimit.tail(n_joints_);
  auto q_jnt_min = robot_model.lowerPositionLimit.tail(n_joints_);
  auto q_jnt_max = robot_model.upperPositionLimit.tail(n_joints_);

  Eigen::MatrixXd C_acc = Eigen::MatrixXd::Zero(2 * n_joints_, 6 + n_joints_);
  Eigen::VectorXd d_min_acc(2 * n_joints_);
  Eigen::VectorXd d_max_acc(2 * n_joints_);
  C_acc.rightCols(n_joints_).topRows(n_joints_).diagonal().setConstant(sample_time_);
  C_acc.rightCols(n_joints_).bottomRows(n_joints_).diagonal().setConstant(std::pow(sample_time_, 2.0) / 2.0);
  d_min_acc << q_jnt_dot_min - current.qjntdot, q_jnt_min - current.qjnt - sample_time_ * current.qjntdot;
  d_max_acc << q_jnt_dot_max - current.qjntdot, q_jnt_max - current.qjnt - sample_time_ * current.qjntdot;

  Eigen::MatrixXd M = pinocchio::crba(robot_model, robot_data, q);
  // We need to do this since the inertia matrix in Pinocchio is only upper triangular
  M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
  M.diagonal().tail(n_joints_) += M_armature_;

  // Computing Coriolis, centrifugal and gravitational effects
  const auto& c = pinocchio::rnea(robot_model, robot_data, q, qdot, Eigen::VectorXd::Zero(6 + n_joints_));

  Eigen::MatrixXd Jlu = J_lsole_.block(0,0,6,6);
  Eigen::MatrixXd Jla = J_lsole_.block(0,6,6,n_joints_);
  Eigen::MatrixXd Jru = J_rsole_.block(0,0,6,6);
  Eigen::MatrixXd Jra = J_rsole_.block(0,6,6,n_joints_);

  Eigen::MatrixXd Mu = M.block(0,0,6,6+n_joints_);
  Eigen::MatrixXd Ma = M.block(6,0,n_joints_,6+n_joints_);

  Eigen::VectorXd cu = c.block(0,0,6,1);
  Eigen::VectorXd ca = c.block(6,0,n_joints_,1);

  std::vector<Eigen::Vector3d> pcis(4);
  pcis[0] <<  params_.foot_length / 2.0,  params_.foot_width / 2.0, 0.0;
  pcis[1] <<  params_.foot_length / 2.0, -params_.foot_width / 2.0, 0.0;
  pcis[2] << -params_.foot_length / 2.0,  params_.foot_width / 2.0, 0.0;
  pcis[3] << -params_.foot_length / 2.0, -params_.foot_width / 2.0, 0.0;

  std::vector<Eigen::Vector3d> pcis_l(4);
  std::vector<Eigen::Vector3d> pcis_r(4);

  for (int i = 0; i < n_contacts_; ++i) {
    pcis_l[i] = desired.lsole.pos.R * pcis[i];
    pcis_r[i] = desired.rsole.pos.R * pcis[i];
  }

  Eigen::MatrixXd T_l(6, 3 * n_contacts_);
  Eigen::MatrixXd T_r(6, 3 * n_contacts_);
  Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  T_l << I3, I3, I3, I3,
         pinocchio::skew(pcis_l[0]), pinocchio::skew(pcis_l[1]), pinocchio::skew(pcis_l[2]), pinocchio::skew(pcis_l[3]);
  T_r << I3, I3, I3, I3,
         pinocchio::skew(pcis_r[0]), pinocchio::skew(pcis_r[1]), pinocchio::skew(pcis_r[2]), pinocchio::skew(pcis_r[3]);

  Eigen::MatrixXd H_force_one = 1e-9 * Eigen::MatrixXd::Identity(3 * n_contacts_, 3 * n_contacts_);
  Eigen::VectorXd f_force_one = Eigen::VectorXd::Zero(3 * n_contacts_);

  // Eq. (3): include the supplied wrist interaction wrenches in the
  // floating-base dynamics. All quantities use LOCAL_WORLD_ALIGNED ordering
  // [force; torque].
  Eigen::VectorXd b_dyn =
      -cu +
      J_lhand_.leftCols(6).transpose() * left_interaction_wrench +
      J_rhand_.leftCols(6).transpose() * right_interaction_wrench;

  Eigen::MatrixXd C_force_block(4, 3);
  C_force_block <<  1.0,  0.0, -params_.mu,
                    0.0,  1.0, -params_.mu,
                   -1.0,  0.0, -params_.mu,
                    0.0, -1.0, -params_.mu;

  Eigen::VectorXd d_min_force_one = -10000.0 * Eigen::VectorXd::Ones(4 * n_contacts_);
  Eigen::VectorXd d_max_force_one = Eigen::VectorXd::Zero(4 * n_contacts_);

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(H_acc.rows() + 2 * H_force_one.rows(), H_acc.cols() + 2 * H_force_one.cols());
  H.block(0, 0, H_acc.rows(), H_acc.cols()) = H_acc;
  H.block(H_acc.rows(), H_acc.cols(), H_force_one.rows(), H_force_one.cols()) = H_force_one;
  H.block(H_acc.rows() + H_force_one.rows(),
          H_acc.cols() + H_force_one.cols(),
          H_force_one.rows(),
          H_force_one.cols()) = H_force_one;
  Eigen::VectorXd f(f_acc.size() + 2 * f_force_one.size());
  f << f_acc, f_force_one, f_force_one;

  Eigen::MatrixXd A_acc = Eigen::MatrixXd::Zero(12, 6 + n_joints_);
  Eigen::VectorXd b_acc = Eigen::VectorXd::Zero(12);
  Eigen::MatrixXd A_no_contact = Eigen::MatrixXd::Zero(3 * n_contacts_, 2 * 3 * n_contacts_);
  Eigen::VectorXd b_no_contact = Eigen::VectorXd::Zero(3 * n_contacts_);

  if (current.is_left_foot_support) {
    A_acc.topRows(6) = J_lsole_;
    b_acc.topRows(6) = -J_lsole_dot_ * qdot - params_.gamma * J_lsole_ * qdot + params_.beta * err_lsole;
  }
  if (current.is_right_foot_support) {
    A_acc.bottomRows(6) = J_rsole_;
    b_acc.bottomRows(6) = -J_rsole_dot_ * qdot - params_.gamma * J_rsole_ * qdot + params_.beta * err_rsole;
  }
  if (!current.is_left_foot_support) {
    A_no_contact.block(0, 0, 3 * n_contacts_, 3 * n_contacts_) = Eigen::MatrixXd::Identity(3 * n_contacts_, 3 * n_contacts_);
  }
  if (!current.is_right_foot_support) {
    A_no_contact.block(0, 3 * n_contacts_, 3 * n_contacts_, 3 * n_contacts_) = Eigen::MatrixXd::Identity(3 * n_contacts_, 3 * n_contacts_);
  }

  Eigen::MatrixXd A_dyn(6, 6 + n_joints_ + 2 * 3 * n_contacts_);
  A_dyn << Mu, -Jlu.transpose() * T_l, -Jru.transpose() * T_r;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(A_acc.rows() + A_no_contact.rows() + A_dyn.rows(), n_wbc_variables_);
  A.block(0, 0, A_acc.rows(), A_acc.cols()) = A_acc;
  A.block(A_acc.rows(), A_acc.cols(), A_no_contact.rows(), A_no_contact.cols()) = A_no_contact;
  A.bottomRows(A_dyn.rows()) = A_dyn;
  Eigen::VectorXd b(b_acc.rows() + b_no_contact.rows() + b_dyn.rows());
  b << b_acc, b_no_contact, b_dyn;

  Eigen::MatrixXd C_force_left = Eigen::MatrixXd::Zero(4 * n_contacts_, 3 * n_contacts_);
  for (int i = 0; i < n_contacts_; ++i) {
    C_force_left.block(4 * i, 3 * i, 4, 3) = C_force_block * current.lsole.pos.R.transpose();
  }
  Eigen::MatrixXd C_force_right = Eigen::MatrixXd::Zero(4 * n_contacts_, 3 * n_contacts_);
  for (int i = 0; i < n_contacts_; ++i) {
    C_force_right.block(4 * i, 3 * i, 4, 3) = C_force_block * current.rsole.pos.R.transpose();
  }
  Eigen::MatrixXd C(C_acc.rows() + 2 * C_force_left.rows(), n_wbc_variables_);
  C << C_acc, Eigen::MatrixXd::Zero(C_acc.rows(), 2 * 3 * n_contacts_),
      Eigen::MatrixXd::Zero(C_force_left.rows(), 6 + n_joints_), C_force_left, Eigen::MatrixXd::Zero(C_force_left.rows(), 3 * n_contacts_),
      Eigen::MatrixXd::Zero(C_force_right.rows(), 6 + n_joints_), Eigen::MatrixXd::Zero(C_force_right.rows(), 3 * n_contacts_), C_force_right;
  Eigen::VectorXd d_min(d_min_acc.rows() + 2 * d_min_force_one.rows());
  Eigen::VectorXd d_max(d_max_acc.rows() + 2 * d_max_force_one.rows());
  d_min << d_min_acc, d_min_force_one, d_min_force_one;
  d_max << d_max_acc, d_max_force_one, d_max_force_one;

  wbc_solver_ptr_->solve(H, f, A, b, C, d_min, d_max);
  Eigen::VectorXd solution = wbc_solver_ptr_->get_solution();
  q_ddot_ = solution.head(6 + n_joints_);
  if (desired.use_hand_compliance) {
    left_hand_compliance_acceleration_achieved_ =
        J_lhand_arm * q_ddot_ + a_lhand_arm_drift;
    right_hand_compliance_acceleration_achieved_ =
        J_rhand_arm * q_ddot_ + a_rhand_arm_drift;
  } else {
    left_hand_compliance_acceleration_achieved_.setZero();
    right_hand_compliance_acceleration_achieved_.setZero();
  }
  flr = solution.segment(6 + n_joints_, 2 * 3 * n_contacts_);
  Eigen::VectorXd fl = flr.head(3 * n_contacts_);
  Eigen::VectorXd fr = flr.tail(3 * n_contacts_);
  Eigen::VectorXd tau =
      Ma * q_ddot_ + ca -
      Jla.transpose() * T_l * fl -
      Jra.transpose() * T_r * fr -
      J_lhand_.rightCols(n_joints_).transpose() * left_interaction_wrench -
      J_rhand_.rightCols(n_joints_).transpose() * right_interaction_wrench;

  left_foot_wrench_ = T_l * fl;
  right_foot_wrench_ = T_r * fr;

  JointCommand joint_command;
  for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) n_joints_; ++joint_id) {
    const auto& joint_name = robot_model.names[joint_id + 2];
    joint_command[joint_name] = tau[joint_id];
  }
  
  return joint_command;
}

Eigen::VectorXd WholeBodyController::get_q_ddot() const {
  return q_ddot_;
}


const Eigen::MatrixXd& WholeBodyController::getLeftFootUnderactuatedJacobian() const { return Jlu_; }
const Eigen::MatrixXd& WholeBodyController::getRightFootUnderactuatedJacobian() const { return Jru_; }

Eigen::VectorXd WholeBodyController::get_flr() const {
  return flr;
}
}
