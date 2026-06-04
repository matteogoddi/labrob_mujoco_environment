#include <hrp4_locomotion/JISMPCWholeBodyController.hpp>

#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>

namespace labrob {

static constexpr const char* kLeftSoleFrame  = "left_foot_link";
static constexpr const char* kRightSoleFrame = "right_foot_link";

JISMPCWholeBodyControllerParams
JISMPCWholeBodyControllerParams::getDefaultParams() {
  JISMPCWholeBodyControllerParams p;
  return p;
}

JISMPCWholeBodyController::JISMPCWholeBodyController(
    const JISMPCWholeBodyControllerParams& params,
    const pinocchio::Model& robot_model,
    const Eigen::VectorXd& q_jnt_reg,
    double sample_time,
    std::map<std::string, double>& armatures)
    : robot_model_(robot_model),
      robot_data_(robot_model),
      q_jnt_reg_(q_jnt_reg),
      sample_time_(sample_time),
      params_(params)
{
  lsole_idx_ = robot_model_.getFrameId(kLeftSoleFrame);
  rsole_idx_ = robot_model_.getFrameId(kRightSoleFrame);

  n_joints_   = robot_model_.nv - 6;
  n_contacts_ = kFootContactsPerFoot;

  n_wbc_variables_   = 6 + n_joints_ + 2 * 3 * n_contacts_;
  n_wbc_equalities_  = 6 + 2 * 6 + 3 * n_contacts_;
  n_wbc_inequalities_= 2 * n_joints_ + 2 * 4 * n_contacts_;

  J_lsole_.resize(6, robot_model_.nv);
  J_rsole_.resize(6, robot_model_.nv);
  J_lsole_dot_.resize(6, robot_model_.nv);
  J_rsole_dot_.resize(6, robot_model_.nv);

  q_.resize(robot_model_.nq);
  qdot_.resize(robot_model_.nv);
  zero_nv_ = Eigen::VectorXd::Zero(robot_model_.nv);

  M_.resize(robot_model_.nv, robot_model_.nv);
  c_.resize(robot_model_.nv);

  H_acc_.resize(6 + n_joints_, 6 + n_joints_);
  f_acc_.resize(6 + n_joints_);
  C_acc_ = Eigen::MatrixXd::Zero(2 * n_joints_, 6 + n_joints_);
  d_min_acc_.resize(2 * n_joints_);
  d_max_acc_.resize(2 * n_joints_);

  C_acc_.rightCols(n_joints_).topRows(n_joints_).diagonal().setConstant(sample_time_);
  C_acc_.rightCols(n_joints_).bottomRows(n_joints_).diagonal()
      .setConstant(0.5 * sample_time_ * sample_time_);

  pcis_[0] <<  params_.foot_length / 2.0,  params_.foot_width / 2.0, 0.0;
  pcis_[1] <<  params_.foot_length / 2.0, -params_.foot_width / 2.0, 0.0;
  pcis_[2] << -params_.foot_length / 2.0,  params_.foot_width / 2.0, 0.0;
  pcis_[3] << -params_.foot_length / 2.0, -params_.foot_width / 2.0, 0.0;

  T_l_.resize(6, 3 * n_contacts_);
  T_r_.resize(6, 3 * n_contacts_);

  H_force_one_ = 1e-9 * Eigen::MatrixXd::Identity(3 * n_contacts_, 3 * n_contacts_);
  f_force_one_ = Eigen::VectorXd::Zero(3 * n_contacts_);
  b_dyn_.resize(6);

  C_force_block_ <<  1.0,  0.0, -params_.mu,
                     0.0,  1.0, -params_.mu,
                    -1.0,  0.0, -params_.mu,
                     0.0, -1.0, -params_.mu;

  d_min_force_one_ = -10000.0 * Eigen::VectorXd::Ones(4 * n_contacts_);
  d_max_force_one_ = Eigen::VectorXd::Zero(4 * n_contacts_);

  H_ = Eigen::MatrixXd::Zero(n_wbc_variables_, n_wbc_variables_);
  f_.resize(n_wbc_variables_);

  A_acc_        = Eigen::MatrixXd::Zero(12, 6 + n_joints_);
  b_acc_        = Eigen::VectorXd::Zero(12);
  A_no_contact_ = Eigen::MatrixXd::Zero(3 * n_contacts_, 2 * 3 * n_contacts_);
  b_no_contact_ = Eigen::VectorXd::Zero(3 * n_contacts_);
  A_dyn_.resize(6, n_wbc_variables_);
  A_ = Eigen::MatrixXd::Zero(A_acc_.rows() + A_no_contact_.rows() + A_dyn_.rows(), n_wbc_variables_);
  b_.resize(b_acc_.rows() + b_no_contact_.rows() + b_dyn_.rows());

  C_force_left_  = Eigen::MatrixXd::Zero(4 * n_contacts_, 3 * n_contacts_);
  C_force_right_ = Eigen::MatrixXd::Zero(4 * n_contacts_, 3 * n_contacts_);
  C_.resize(C_acc_.rows() + 2 * C_force_left_.rows(), n_wbc_variables_);
  d_min_.resize(d_min_acc_.rows() + 2 * d_min_force_one_.rows());
  d_max_.resize(d_max_acc_.rows() + 2 * d_max_force_one_.rows());

  tau_ = Eigen::VectorXd::Zero(n_joints_);

  M_armature_ = Eigen::VectorXd::Zero(n_joints_);
  for (pinocchio::JointIndex jid = 2;
       jid < (pinocchio::JointIndex) robot_model_.njoints; ++jid) {
    M_armature_(jid - 2) = armatures[robot_model_.names[jid]];
  }

  wbc_solver_ptr_ = std::make_unique<labrob::qpsolvers::QPSolverEigenWrapper<double>>(
      std::make_shared<labrob::qpsolvers::HPIPMQPSolver>(
          n_wbc_variables_, n_wbc_equalities_, n_wbc_inequalities_));
}

labrob::JointCommand
JISMPCWholeBodyController::compute_inverse_dynamics(
    const pinocchio::Model& robot_model,
    const labrob::RobotState& robot_state,
    pinocchio::Data& robot_data,
    const labrob::GaitConfiguration& current,
    const labrob::GaitConfiguration& desired,
    const Eigen::VectorXd& nu_dot_des)
{
  // --- Build q, qdot from RobotState ---
  q_.head<3>() = robot_state.position;
  q_.segment<4>(3) = robot_state.orientation.coeffs();
  qdot_.head<3>()  = robot_state.linear_velocity;
  qdot_.segment<3>(3) = robot_state.angular_velocity;
  for (pinocchio::JointIndex jid = 2;
       jid < (pinocchio::JointIndex) robot_model_.njoints; ++jid) {
    const auto& jname = robot_model_.names[jid];
    q_[jid + 5]    = robot_state.joint_state.at(jname).pos;
    qdot_[jid + 4] = robot_state.joint_state.at(jname).vel;
  }

  // --- Pinocchio kinematics ---
  pinocchio::computeJointJacobiansTimeVariation(robot_model_, robot_data_, q_, qdot_);
  pinocchio::framesForwardKinematics(robot_model_, robot_data_, q_);

  J_lsole_.setZero(); J_rsole_.setZero();
  J_lsole_dot_.setZero(); J_rsole_dot_.setZero();
  pinocchio::getFrameJacobian(robot_model_, robot_data_, lsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lsole_);
  pinocchio::getFrameJacobian(robot_model_, robot_data_, rsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rsole_);
  pinocchio::getFrameJacobianTimeVariation(robot_model_, robot_data_, lsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lsole_dot_);
  pinocchio::getFrameJacobianTimeVariation(robot_model_, robot_data_, rsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rsole_dot_);

  // --- Inertia and Coriolis ---
  M_ = pinocchio::crba(robot_model_, robot_data_, q_);
  M_.triangularView<Eigen::StrictlyLower>() =
      M_.transpose().triangularView<Eigen::StrictlyLower>();
  M_.diagonal().tail(n_joints_) += M_armature_;
  c_ = pinocchio::rnea(robot_model_, robot_data_, q_, qdot_, zero_nv_);

  auto Jlu = J_lsole_.leftCols(6);
  auto Jla = J_lsole_.rightCols(n_joints_);
  auto Jru = J_rsole_.leftCols(6);
  auto Jra = J_rsole_.rightCols(n_joints_);
  auto Mu  = M_.topRows(6);
  auto Ma  = M_.bottomRows(n_joints_);
  auto cu  = c_.head(6);
  auto ca  = c_.tail(n_joints_);

  // --- Cost: minimise ||nu_dot - nu_dot_des||^2 ---
  H_acc_.setZero();
  f_acc_.setZero();
  H_acc_.diagonal().array() += params_.weight_q_ddot + 1.0;
  f_acc_.noalias() -= nu_dot_des;

  // --- Joint velocity / position limits ---
  auto q_jnt_dot_min = -robot_model.velocityLimit.tail(n_joints_);
  auto q_jnt_dot_max =  robot_model.velocityLimit.tail(n_joints_);
  auto q_jnt_min     =  robot_model.lowerPositionLimit.tail(n_joints_);
  auto q_jnt_max     =  robot_model.upperPositionLimit.tail(n_joints_);

  d_min_acc_ << q_jnt_dot_min - current.qjntdot,
                q_jnt_min - current.qjnt - sample_time_ * current.qjntdot;
  d_max_acc_ << q_jnt_dot_max - current.qjntdot,
                q_jnt_max - current.qjnt - sample_time_ * current.qjntdot;

  // --- Friction cone corners ---
  for (int i = 0; i < n_contacts_; ++i) {
    pcis_l_[i].noalias() = desired.lsole.pos.R * pcis_[i];
    pcis_r_[i].noalias() = desired.rsole.pos.R * pcis_[i];
  }
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  T_l_ << I3, I3, I3, I3,
          pinocchio::skew(pcis_l_[0]), pinocchio::skew(pcis_l_[1]),
          pinocchio::skew(pcis_l_[2]), pinocchio::skew(pcis_l_[3]);
  T_r_ << I3, I3, I3, I3,
          pinocchio::skew(pcis_r_[0]), pinocchio::skew(pcis_r_[1]),
          pinocchio::skew(pcis_r_[2]), pinocchio::skew(pcis_r_[3]);

  b_dyn_ = -cu;

  // --- Assemble H, f ---
  H_.setZero();
  H_.block(0, 0, H_acc_.rows(), H_acc_.cols()) = H_acc_;
  H_.block(H_acc_.rows(), H_acc_.cols(),
           H_force_one_.rows(), H_force_one_.cols()) = H_force_one_;
  H_.block(H_acc_.rows() + H_force_one_.rows(),
           H_acc_.cols() + H_force_one_.cols(),
           H_force_one_.rows(), H_force_one_.cols()) = H_force_one_;
  f_ << f_acc_, f_force_one_, f_force_one_;

  // --- Equality constraints ---
  A_acc_.setZero();
  b_acc_.setZero();
  A_no_contact_.setZero();
  b_no_contact_.setZero();

  if (desired.is_left_foot_support) {
    A_acc_.topRows(6) = J_lsole_;
    b_acc_.topRows(6).noalias() =
        -J_lsole_dot_ * qdot_ - params_.gamma * J_lsole_ * qdot_;
  }
  if (desired.is_right_foot_support) {
    A_acc_.bottomRows(6) = J_rsole_;
    b_acc_.bottomRows(6).noalias() =
        -J_rsole_dot_ * qdot_ - params_.gamma * J_rsole_ * qdot_;
  }
  if (!desired.is_left_foot_support) {
    A_no_contact_.block(0, 0, 3 * n_contacts_, 3 * n_contacts_).setIdentity();
  }
  if (!desired.is_right_foot_support) {
    A_no_contact_.block(0, 3 * n_contacts_, 3 * n_contacts_, 3 * n_contacts_).setIdentity();
  }

  A_dyn_.leftCols(6 + n_joints_) = Mu;
  A_dyn_.middleCols(6 + n_joints_, 3 * n_contacts_).noalias() = -Jlu.transpose() * T_l_;
  A_dyn_.rightCols(3 * n_contacts_).noalias() = -Jru.transpose() * T_r_;

  A_.setZero();
  A_.block(0, 0, A_acc_.rows(), A_acc_.cols()) = A_acc_;
  A_.block(A_acc_.rows(), A_acc_.cols(), A_no_contact_.rows(), A_no_contact_.cols()) = A_no_contact_;
  A_.bottomRows(A_dyn_.rows()) = A_dyn_;
  b_ << b_acc_, b_no_contact_, b_dyn_;

  // --- Inequality constraints (joint limits + friction cones) ---
  C_force_left_.setZero();
  for (int i = 0; i < n_contacts_; ++i) {
    C_force_left_.block(4 * i, 3 * i, 4, 3).noalias() =
        C_force_block_ * current.lsole.pos.R.transpose();
  }
  C_force_right_.setZero();
  for (int i = 0; i < n_contacts_; ++i) {
    C_force_right_.block(4 * i, 3 * i, 4, 3).noalias() =
        C_force_block_ * current.rsole.pos.R.transpose();
  }

  C_.setZero();
  C_.block(0, 0, C_acc_.rows(), C_acc_.cols()) = C_acc_;
  C_.block(C_acc_.rows(), 6 + n_joints_,
           C_force_left_.rows(), C_force_left_.cols()) = C_force_left_;
  C_.block(C_acc_.rows() + C_force_left_.rows(),
           6 + n_joints_ + 3 * n_contacts_,
           C_force_right_.rows(), C_force_right_.cols()) = C_force_right_;

  d_min_ << d_min_acc_, d_min_force_one_, d_min_force_one_;
  d_max_ << d_max_acc_, d_max_force_one_, d_max_force_one_;

  // --- Solve ---
  wbc_solver_ptr_->solve(H_, f_, A_, b_, C_, d_min_, d_max_);

  if (wbc_solver_ptr_->get_status() != 0) {
    std::cerr << "[JISMPC WBC] HPIPM status " << wbc_solver_ptr_->get_status()
              << ", using gravity compensation\n";
    tau_ = ca;
  } else {
    const auto& sol = wbc_solver_ptr_->get_solution();
    auto q_ddot = sol.head(6 + n_joints_);
    auto fl     = sol.segment(6 + n_joints_, 3 * n_contacts_);
    auto fr     = sol.tail(3 * n_contacts_);

    tau_.noalias()  = Ma * q_ddot;
    tau_           += ca;
    tau_.noalias() -= Jla.transpose() * T_l_ * fl;
    tau_.noalias() -= Jra.transpose() * T_r_ * fr;
  }

  labrob::JointCommand joint_command;
  for (pinocchio::JointIndex jid = 2;
       jid < (pinocchio::JointIndex) robot_model_.njoints; ++jid) {
    joint_command[robot_model_.names[jid]] = tau_[jid - 2];
  }
  return joint_command;
}

} // namespace labrob