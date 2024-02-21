//
// Created by mmaximo on 20/02/24.
//

#include <hrp4_locomotion/TorqueWholeBodyController.hpp>

// Pinocchio
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <hrp4_locomotion/JointCommand.hpp>
#include <hrp4_locomotion/utils.hpp>

namespace labrob {

TorqueWholeBodyControllerParams TorqueWholeBodyControllerParams::getDefaultParams() {
  static TorqueWholeBodyControllerParams params;

  params.Kp_motion = 30.0;
  params.Kd_motion = 10.0;
  params.Kp_regulation = 30.0;
  params.Kd_regulation = 10.0;

  params.weight_q_ddot = 1e-4;
  params.weight_com = 0.1;
  params.weight_lsole = 1.0;
  params.weight_rsole = 1.0;
  params.weight_torso = 1e-3;
  params.weight_regulation = 1e-4;

  params.gamma = params.Kd_motion;
  params.mu = 0.5;

  params.foot_length = 0.2;
  params.foot_width = 0.08;

  return params;
}

TorqueWholeBodyController::TorqueWholeBodyController(
    const TorqueWholeBodyControllerParams &params, const pinocchio::Model &robot_model,
    const Eigen::VectorXd &q_jnt_reg,
    double sample_time,
    std::map<std::string, double> &armatures)
    : WholeBodyController(robot_model, q_jnt_reg, sample_time), params_(params) {
  num_joints_ = robot_model.nv - 6;

  M_armature_ = Eigen::VectorXd::Zero(num_joints_);
  for (pinocchio::JointIndex joint_id = 2;
       joint_id < (pinocchio::JointIndex) robot_model_.njoints;
       ++joint_id) {
    std::string joint_name = robot_model_.names[joint_id];
    M_armature_(joint_id - 2) = armatures[joint_name];
  }

  const double Kp_motion = params.Kp_motion;
  const double Kd_motion = params.Kd_motion;

  Kp_com_ = Kp_motion * Eigen::Matrix3d::Identity();
  Kd_com_ = Kd_motion * Eigen::Matrix3d::Identity();
  Kp_torso_orientation_ = Kp_motion * Eigen::Matrix3d::Identity();
  Kd_torso_orientation_ = Kd_motion * Eigen::Matrix3d::Identity();
  Kp_lsole_ = Kp_motion * Eigen::MatrixXd::Identity(6, 6);
  Kd_lsole_ = Kd_motion * Eigen::MatrixXd::Identity(6, 6);
  Kp_lsole_ = Kp_motion * Eigen::MatrixXd::Identity(6, 6);
  Kd_lsole_ = Kd_motion * Eigen::MatrixXd::Identity(6, 6);

  Kp_jnt_ = params.Kp_regulation;
  Kd_jnt_ = params.Kd_regulation;

  num_contacts_ = 4;

  int num_variables_base = 6 + num_joints_;
  int num_variables_foot = 12;
  int num_equalities_base = 6;
  int num_equalities_foot = 6;
  int num_inequalities_base = 2 * num_joints_;
  int num_inequalities_foot = 4 * num_contacts_;

  num_variables_single_ = num_variables_base + num_variables_foot;
  num_equalities_single_ = num_equalities_base + num_equalities_foot;
  num_inequalities_single_ = num_inequalities_base + num_inequalities_foot;

  num_variables_double_ = num_variables_base + 2 * num_variables_foot;
  num_equalities_double_ = num_equalities_base + 2 * num_equalities_foot;
  num_inequalities_double_ = num_inequalities_base + 2 * num_inequalities_foot;

  single_support_solver_ptr_ = std::make_unique<labrob::qpsolvers::QPSolverEigenWrapper<double>>(
      std::make_shared<labrob::qpsolvers::HPIPMQPSolver>(
          num_variables_single_, num_equalities_single_, num_inequalities_single_
      )
  );

  double_support_solver_ptr_ = std::make_unique<labrob::qpsolvers::QPSolverEigenWrapper<double>>(
      std::make_shared<labrob::qpsolvers::HPIPMQPSolver>(
          num_variables_double_, num_equalities_double_, num_inequalities_double_
      )
  );
}

Eigen::VectorXd TorqueWholeBodyController::control(const CoMMotion &desired_com_motion,
                                                   const FootMotion &desired_left_foot_motion,
                                                   const FootMotion &desired_right_foot_motion,
                                                   const Eigen::VectorXd &q,
                                                   const Eigen::VectorXd &q_dot,
                                                   bool is_left_support,
                                                   bool is_right_support) {
  static Eigen::MatrixXd J_torso(6, robot_model_.nv);
  static Eigen::MatrixXd J_lsole(6, robot_model_.nv);
  static Eigen::MatrixXd J_rsole(6, robot_model_.nv);
  static Eigen::MatrixXd J_torso_dot(6, robot_model_.nv);
  static Eigen::MatrixXd J_lsole_dot(6, robot_model_.nv);
  static Eigen::MatrixXd J_rsole_dot(6, robot_model_.nv);

  J_torso.fill(0.0);
  J_lsole.fill(0.0);
  J_rsole.fill(0.0);
  J_torso_dot.fill(0.0);
  J_lsole_dot.fill(0.0);
  J_rsole_dot.fill(0.0);

  bool is_single_support = (is_left_support and is_right_support);

  computePinocchio(q, q_dot);
  getJacobians(q, q_dot, J_torso, J_lsole, J_rsole);
  getJacobiansTimeVariation(q, q_dot, J_torso_dot, J_lsole_dot, J_rsole_dot);

  const auto &J_com = robot_data_.Jcom;
  const auto J_torso_orientation = J_torso.bottomRows<3>();
  const auto J_torso_orientation_dot = J_torso_dot.bottomRows<3>();

  const auto &centroidal_momentum_matrix = pinocchio::ccrba(
      robot_model_,
      robot_data_,
      q,
      q_dot
  );
  auto angular_momentum = (centroidal_momentum_matrix * q_dot).tail<3>();

  const auto &p_com = robot_data_.com[0];
  const auto &torso_orientation = robot_data_.oMf[torso_idx_].rotation();
  const auto &T_lsole = robot_data_.oMf[lsole_idx_];
  const auto &T_rsole = robot_data_.oMf[rsole_idx_];

  const auto v_com = J_com * q_dot;
  const auto v_lsole = J_lsole * q_dot;
  const auto v_rsole = J_rsole * q_dot;

  const auto &a_com_drift = robot_data_.acom[0];
  const auto a_lsole_drift = J_lsole_dot * q_dot;
  const auto a_rsole_drift = J_rsole_dot * q_dot;
  const auto a_torso_orientation_drift = J_torso_orientation_dot * q_dot;

  double left_foot_yaw = std::atan2(
      desired_left_foot_motion.pose.rotation()(1, 0),
      desired_left_foot_motion.pose.rotation()(0, 0)
  );
  double right_foot_yaw = std::atan2(
      desired_right_foot_motion.pose.rotation()(1, 0),
      desired_right_foot_motion.pose.rotation()(0, 0)
  );
  double desired_torso_yaw = (left_foot_yaw + right_foot_yaw) / 2.0;
  Eigen::Matrix3d torso_orientation_des = labrob::Rz(desired_torso_yaw);
  // TODO: include feedforward for torso orientation
  Eigen::Vector3d v_torso_orientation_des = Eigen::Vector3d::Zero();
  Eigen::Vector3d a_torso_orientation_des = Eigen::Vector3d::Zero();

  auto err_com = desired_com_motion.position - p_com;
  auto err_com_vel = desired_com_motion.velocity - v_com;

  auto err_lsole = err_frameplacement(desired_left_foot_motion.pose, T_lsole);
  auto err_lsole_vel = desired_left_foot_motion.velocity - v_lsole;

  auto err_rsole = err_frameplacement(desired_right_foot_motion.pose, T_rsole);
  auto err_rsole_vel = desired_right_foot_motion.velocity - v_rsole;

  auto err_torso_orientation = err_rotation(torso_orientation_des, torso_orientation);
  auto err_torso_orientation_vel = v_torso_orientation_des - J_torso_orientation * q_dot;

  Eigen::VectorXd err_posture(6 + num_joints_);
  err_posture << Eigen::VectorXd::Zero(6), q_jnt_reg_ - q.tail(num_joints_);

  Eigen::MatrixXd err_posture_selection_matrix = Eigen::MatrixXd::Zero(6 + num_joints_, 6 + num_joints_);
  err_posture_selection_matrix.block(6, 6, num_joints_, num_joints_) =
      Eigen::MatrixXd::Identity(num_joints_, num_joints_);
  Eigen::MatrixXd cmm_selection_matrix(3, 6);
  cmm_selection_matrix.topRows(3).setZero();
  cmm_selection_matrix.bottomRows(3).setIdentity();

  Eigen::VectorXd a_jnt_total = Kp_jnt_ * err_posture_selection_matrix - Kd_jnt_ * q_dot;
  Eigen::VectorXd a_com_total = desired_com_motion.acceleration + Kp_com_ * err_com + Kd_com_ * err_com_vel;
  Eigen::VectorXd
      a_lsole_total = desired_left_foot_motion.acceleration + Kp_lsole_ * err_lsole + Kd_lsole_ * err_lsole_vel;
  Eigen::VectorXd
      a_rsole_total = desired_right_foot_motion.acceleration + Kp_rsole_ * err_rsole + Kd_rsole_ * err_rsole_vel;
  Eigen::VectorXd a_torso_orientation_total = a_torso_orientation_des + Kp_torso_orientation_ * err_torso_orientation
      + Kd_torso_orientation_ * err_torso_orientation_vel;

  Eigen::MatrixXd H_acc = Eigen::MatrixXd::Zero(6 + num_joints_, 6 + num_joints_);
  Eigen::VectorXd f_acc = Eigen::VectorXd::Zero(6 + num_joints_);

  H_acc += params_.weight_q_ddot * Eigen::MatrixXd::Identity(6 + num_joints_, 6 + num_joints_);
  H_acc += params_.weight_com * (J_com.transpose() * J_com);
  H_acc += params_.weight_lsole * (J_lsole.transpose() * J_lsole);
  H_acc += params_.weight_rsole * (J_rsole.transpose() * J_rsole);
  H_acc += params_.weight_torso * (J_torso_orientation.transpose() * J_torso);
  H_acc += params_.weight_regulation * (err_posture_selection_matrix.transpose() * err_posture_selection_matrix);
  H_acc += params_.weight_angular_momentum * centroidal_momentum_matrix.transpose() * cmm_selection_matrix.transpose() *
      std::pow(sample_time_, 2.0) * cmm_selection_matrix * centroidal_momentum_matrix;

  f_acc += params_.weight_com * J_com.transpose() * (a_com_drift - a_com_total);
  f_acc += params_.weight_lsole * J_lsole.transpose() * (a_lsole_drift - a_lsole_total);
  f_acc += params_.weight_rsole * J_rsole.transpose() * (a_rsole_drift - a_rsole_total);
  f_acc +=
      params_.weight_torso * J_torso_orientation.transpose() * (a_torso_orientation_drift - a_torso_orientation_total);
  f_acc -=
      params_.weight_regulation * err_posture_selection_matrix.transpose() * err_posture_selection_matrix * a_jnt_total;
  f_acc += params_.weight_angular_momentum * centroidal_momentum_matrix.transpose() * cmm_selection_matrix.transpose() *
      sample_time_ * cmm_selection_matrix * centroidal_momentum_matrix * q_dot;

  auto q_jnt_dot_min = -robot_model_.velocityLimit.tail(num_joints_);
  auto q_jnt_dot_max = robot_model_.velocityLimit.tail(num_joints_);
  auto q_jnt_min = robot_model_.lowerPositionLimit.tail(num_joints_);
  auto q_jnt_max = robot_model_.upperPositionLimit.tail(num_joints_);

  Eigen::MatrixXd C_acc(2 * num_joints_, 6 + num_joints_);
  Eigen::VectorXd d_min_acc(2 * num_joints_);
  Eigen::VectorXd d_max_acc(2 * num_joints_);
  C_acc.rightCols(num_joints_).topRows(num_joints_).diagonal().setConstant(sample_time_);
  C_acc.rightCols(num_joints_).bottomRows(num_joints_).diagonal().setConstant(std::pow(sample_time_, 2.0) / 2.0);
  d_min_acc << q_jnt_dot_min - q_dot.tail(num_joints_), q_jnt_min - q.tail(num_joints_)
      - sample_time_ * q_dot.tail(num_joints_);
  d_max_acc << q_jnt_dot_max - q_dot.tail(num_joints_), q_jnt_max - q.tail(num_joints_)
      - sample_time_ * q_dot.tail(num_joints_);

  Eigen::MatrixXd Jlu = J_lsole.block(0, 0, 6, 6);
  Eigen::MatrixXd Jla = J_lsole.block(0, 6, 6, num_joints_);
  Eigen::MatrixXd Jru = J_rsole.block(0, 0, 6, 6);
  Eigen::MatrixXd Jra = J_rsole.block(0, 6, 6, num_joints_);

  Eigen::MatrixXd M = pinocchio::crba(robot_model_, robot_data_, q);
  // We need to do this since the inertia matrix in Pinocchio is only upper triangular
  M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
  M.diagonal().tail(num_joints_) += M_armature_;

  // Computing Coriolis, centrifugal and gravitational effects
  const auto &c = pinocchio::rnea(
      robot_model_,
      robot_data_,
      q,
      q_dot,
      Eigen::VectorXd::Zero(6 + num_joints_)
  );

  Eigen::MatrixXd Mu = M.block(0, 0, 6, 6 + num_joints_);
  Eigen::MatrixXd Ma = M.block(6, 0, num_joints_, 6 + num_joints_);

  Eigen::VectorXd cu = c.block(0, 0, 6, 1);
  Eigen::VectorXd ca = c.block(6, 0, num_joints_, 1);

  std::vector<Eigen::Vector3d> pcis(4);
  pcis[0] << params_.foot_length / 2.0, params_.foot_width / 2.0, 0.0;
  pcis[1] << params_.foot_length / 2.0, -params_.foot_width / 2.0, 0.0;
  pcis[2] << -params_.foot_length / 2.0, params_.foot_width / 2.0, 0.0;
  pcis[3] << -params_.foot_length / 2.0, -params_.foot_width / 2.0, 0.0;

  Eigen::MatrixXd T(6, 3 * num_contacts_);
  Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  // TODO: consider foot rotation
  T << I3, I3, I3, I3,
      pinocchio::skew(pcis[0]), pinocchio::skew(pcis[1]), pinocchio::skew(pcis[2]), pinocchio::skew(pcis[3]);

  Eigen::MatrixXd C_force_block(4, 3);
  C_force_block << 1.0, 1.0, -params_.mu,
      0.0, 1.0, -params_.mu,
      -1.0, 0.0, -params_.mu,
      0.0, -1.0, -params_.mu;

  Eigen::MatrixXd H_force_one = Eigen::MatrixXd::Identity(3 * num_contacts_, 3 * num_contacts_);
  Eigen::VectorXd f_force_one = Eigen::VectorXd::Zero(3 * num_contacts_);

  Eigen::VectorXd b_dyn(6);
  b_dyn << -cu;

  Eigen::MatrixXd C_force_one = Eigen::MatrixXd(4 * num_contacts_, 3 * num_contacts_);
  for (int i = 0; i < num_contacts_; ++i) {
    C_force_one.block(4 * i, 3 * i, 4, 3) = C_force_block;
  }
  Eigen::VectorXd d_min_force_one = -1000.0 * Eigen::VectorXd::Ones(4 * num_contacts_);
  Eigen::VectorXd d_max_force_one = Eigen::VectorXd::Zero(4 * num_contacts_);

  Eigen::MatrixXd tau;
  if (is_single_support) {
    Eigen::MatrixXd Js, Js_dot, Jsu, Jsa;
    if (is_left_support) {
      Js = J_lsole;
      Js_dot = J_lsole_dot;
      Jsu = Jlu;
      Jsa = Jla;
    } else {
      Js = J_rsole;
      Js_dot = J_rsole_dot;
      Jsu = Jru;
      Jsa = Jra;
    }
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(H_acc.rows() + H_force_one.rows(), H_acc.cols() + H_force_one.cols());
    H.block(0, 0, H_acc.rows(), H_acc.cols()) = H_acc;
    H.block(H_acc.rows(), H_acc.cols(), H_force_one.rows(), H_force_one.cols()) = H_force_one;
    Eigen::VectorXd f(f_acc.size() + f_force_one.size());
    f << f_acc, f_force_one;

    Eigen::MatrixXd A_acc(6, 6 + num_joints_);
    Eigen::VectorXd b_acc(6);
    A_acc = Js;
    b_acc = -Js_dot * q_dot - params_.gamma * Js * q_dot;
    Eigen::MatrixXd A_dyn(6, 6 + num_joints_ + 3 * num_contacts_);
    A_dyn << Mu, -Jsu.transpose() * T;

    Eigen::MatrixXd A(A_acc.rows() + A_dyn.rows(), num_variables_single_);
    A << A_acc, Eigen::MatrixXd::Zero(A_acc.rows(), 3 * num_contacts_),
        A_dyn;
    Eigen::VectorXd b(b_acc.rows() + b_dyn.rows());
    b << b_acc,
        b_dyn;

    Eigen::MatrixXd C(C_acc.rows() + C_force_one.rows(), num_variables_single_);
    C << C_acc, Eigen::MatrixXd::Zero(C_acc.rows(), 3 * num_contacts_);
    Eigen::MatrixXd::Zero(C_force_one.rows(), 6 + num_joints_), C_force_one;
    Eigen::VectorXd d_min(d_min_acc.rows() + d_min_force_one.rows());
    Eigen::VectorXd d_max(d_max_acc.rows() + d_max_force_one.rows());
    d_min << d_min_acc,
        d_min_force_one;
    d_max << d_max_acc,
        d_max_force_one;

    single_support_solver_ptr_->solve(H, f, A, b, C, d_min, d_max);

    Eigen::VectorXd solution = single_support_solver_ptr_->get_solution();

    Eigen::VectorXd q_ddot = solution.head(6 + num_joints_);
    Eigen::VectorXd fs = solution.tail(3 * num_contacts_);
    tau = Ma * q_ddot + ca - Jsa.transpose() * T * fs;
  } else {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(H_acc.rows() + 2 * H_force_one.rows(), H_acc.cols() + 2 * H_force_one.cols());
    H.block(0, 0, H_acc.rows(), H_acc.cols()) = H_acc;
    H.block(H_acc.rows(), H_acc.cols(), H_force_one.rows(), H_force_one.cols()) = H_force_one;
    H.block(H_acc.rows() + H_force_one.rows(), H_acc.cols() + H_force_one.cols(), H_force_one.rows(), H_force_one.cols()) = H_force_one;
    Eigen::VectorXd f(f_acc.size() + 2 * f_force_one.size());
    f << f_acc, f_force_one, f_force_one;

    Eigen::MatrixXd A_acc(12, 6 + num_joints_);
    Eigen::VectorXd b_acc(12);
    A_acc.topRows(6) = J_lsole;
    b_acc.topRows(6) = -J_lsole_dot * q_dot - params_.gamma * J_lsole * q_dot;
    A_acc.bottomRows(6) = J_rsole;
    b_acc.bottomRows(6) = -J_rsole_dot * q_dot - params_.gamma * J_rsole * q_dot;
    Eigen::MatrixXd A_dyn(6, 6 + num_joints_ + 3 * 2 * num_contacts_);
    A_dyn << Mu, -Jlu.transpose() * T, -Jru.transpose() * T;

    Eigen::MatrixXd A(A_acc.rows() + A_dyn.rows(), num_variables_double_);
    A << A_acc, Eigen::MatrixXd::Zero(A_acc.rows(), 3 * 2 * num_contacts_),
        A_dyn;
    Eigen::VectorXd b(b_acc.rows() + b_dyn.rows());
    b << b_acc,
        b_dyn;

    Eigen::MatrixXd C(C_acc.rows() + C_force_one.rows(), num_variables_single_);
    C << C_acc, Eigen::MatrixXd::Zero(C_acc.rows(), 3 * num_contacts_);
    Eigen::MatrixXd::Zero(C_force_one.rows(), 6 + num_joints_), C_force_one;
    Eigen::VectorXd d_min(d_min_acc.rows() + d_min_force_one.rows());
    Eigen::VectorXd d_max(d_max_acc.rows() + d_max_force_one.rows());
    d_min << d_min_acc,
        d_min_force_one;
    d_max << d_max_acc,
        d_max_force_one;

    double_support_solver_ptr_->solve(H, f, A, b, C, d_min, d_max);

    Eigen::VectorXd solution = double_support_solver_ptr_->get_solution();

    Eigen::VectorXd q_ddot = solution.head(6 + num_joints_);
    Eigen::VectorXd f = solution.tail(3 * 2 * num_contacts_);
    Eigen::VectorXd fl = f.head(3 * num_contacts_);
    Eigen::VectorXd fr = f.tail(3 * num_contacts_);
    tau = Ma * q_ddot + ca - Jla.transpose() * T * fl - Jra.transpose() * T * fr;
  }

  return tau;
}

}