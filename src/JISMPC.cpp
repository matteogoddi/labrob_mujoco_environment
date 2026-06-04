#include <hrp4_locomotion/JISMPC.hpp>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <hrp4_locomotion/WalkingState.hpp>

#include <hpipm_d_dense_qp_ipm.h>

#include <iostream>

namespace labrob {

static constexpr const char* kLeftSoleFrame  = "left_foot_link";
static constexpr const char* kRightSoleFrame = "right_foot_link";

// -----------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------
JISMPC::JISMPC(
    const pinocchio::Model& model,
    int horizon,
    double dt,
    double eta,
    double robot_mass,
    double foot_constraint_x,
    double foot_constraint_y,
    int64_t jismpc_timestep_msec
) : model_(model),
    data_(model),
    C_(horizon),
    dt_(dt),
    eta_(eta),
    robot_mass_(robot_mass),
    foot_x_(foot_constraint_x),
    foot_y_(foot_constraint_y),
    jismpc_timestep_msec_(jismpc_timestep_msec)
{
  lsole_idx_ = model_.getFrameId(kLeftSoleFrame);
  rsole_idx_ = model_.getFrameId(kRightSoleFrame);

  nv_   = model_.nv;
  njnt_ = model_.nv - 6;
  n_var_  = C_ * nv_;
  // Only 2 equality constraints: DCM terminal stability (x, y).
  // Contact is handled by the WBC with hard constraints — the MPC uses soft costs.
  n_eq_   = 2;
  // 2*C inequality constraints: ZMP box per horizon step (x and y, lb≤Cw≤ub per row)
  n_ineq_ = 2 * C_;

  // Auxiliary trajectory storage
  q_aux_.resize(C_ + 1, Eigen::VectorXd::Zero(model_.nq));
  qdot_aux_.resize(C_ + 1, Eigen::VectorXd::Zero(nv_));
  nu_dot_.resize(C_, Eigen::VectorXd::Zero(nv_));

  steps_.resize(C_);
  for (auto& s : steps_) {
    s.J_com.resize(3, nv_);       s.J_com.setZero();
    s.Ag.resize(6, nv_);          s.Ag.setZero();
    s.J_lsole.resize(6, nv_);     s.J_lsole.setZero();
    s.J_rsole.resize(6, nv_);     s.J_rsole.setZero();
    s.Jdot_qdot_lsole.resize(6);  s.Jdot_qdot_lsole.setZero();
    s.Jdot_qdot_rsole.resize(6);  s.Jdot_qdot_rsole.setZero();
    s.Jdot_qdot_com.setZero();
    s.l_contact = true;
    s.r_contact = true;
  }
  l_contact_.resize(C_, true);
  r_contact_.resize(C_, true);

  // QP matrices
  H_.resize(n_var_, n_var_);        H_.setZero();
  f_.resize(n_var_);                f_.setZero();
  A_eq_.resize(n_eq_, n_var_);      A_eq_.setZero();
  b_eq_.resize(n_eq_);              b_eq_.setZero();
  A_ineq_.resize(n_ineq_, n_var_);  A_ineq_.setZero();
  lg_.resize(n_ineq_);
  ug_.resize(n_ineq_);

  nu_dot_0_.resize(nv_);
  nu_dot_0_.setZero();

  qp_solver_ptr_ = std::make_shared<labrob::qpsolvers::QPSolverEigenWrapper<double>>(
      std::make_shared<labrob::qpsolvers::HPIPMQPSolver>(n_var_, n_eq_, n_ineq_, BALANCE, 200)
  );
}

// -----------------------------------------------------------------------
// ZMP reference at a given time from WalkingData
// (same interpolation logic as ISMPC::solve)
// -----------------------------------------------------------------------
Eigen::Vector2d JISMPC::zmpReference(
    const labrob::WalkingData& wd, int64_t t_ms) const
{
  const int64_t dt_ms = jismpc_timestep_msec_;
  const int n_plan = static_cast<int>(wd.footstep_plan.size());

  // Find footstep element k and relative index within it
  int64_t t_rel = t_ms - wd.t0;
  if (t_rel < 0) t_rel = 0;

  int k = 0;
  while (k < n_plan - 1) {
    int64_t dur = wd.footstep_plan[k].getDuration();
    if (t_rel < dur) break;
    t_rel -= dur;
    ++k;
  }
  // Clamp to last element
  k = std::min(k, n_plan - 1);

  const auto& elem = wd.footstep_plan[k];
  const auto& ws   = elem.getWalkingState();
  int64_t dur_k    = elem.getDuration();

  double s = (dur_k > 0) ? static_cast<double>(t_rel) / static_cast<double>(dur_k) : 0.0;
  s = std::max(0.0, std::min(1.0, s));

  double wx = 0.5, wy = 0.5;
  if (ws == WalkingState::PostureRegulation || ws == WalkingState::Standing) {
    wx = 0.5; wy = 0.5;
  } else if (ws == WalkingState::Starting) {
    wx = 0.5 + 0.5 * s; wy = 0.5 - 0.5 * s;
  } else if (ws == WalkingState::SingleSupport) {
    wx = 1.0; wy = 0.0;
  } else if (ws == WalkingState::DoubleSupport) {
    wx = 1.0 - s; wy = s;
  } else if (ws == WalkingState::Stopping) {
    wx = 1.0 - 0.5 * s; wy = 0.5 * s;
  }

  const Eigen::Vector3d& ps = elem.getFeetPlacement().getSupportFootConfiguration().p;
  const Eigen::Vector3d& pw = elem.getFeetPlacement().getSwingFootConfiguration().p;
  return Eigen::Vector2d(wx * ps.x() + wy * pw.x(),
                         wx * ps.y() + wy * pw.y());
}

// -----------------------------------------------------------------------
// Contact flags from WalkingData
// -----------------------------------------------------------------------
static bool leftInContact(const labrob::WalkingData& wd) {
  const auto& ws = wd.footstep_plan.front().getWalkingState();
  if (ws == WalkingState::SingleSupport)
    return wd.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT;
  return true;
}
static bool rightInContact(const labrob::WalkingData& wd) {
  const auto& ws = wd.footstep_plan.front().getWalkingState();
  if (ws == WalkingState::SingleSupport)
    return wd.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT;
  return true;
}

// -----------------------------------------------------------------------
// solve()
// -----------------------------------------------------------------------
void JISMPC::solve(
    const labrob::RobotState& robot_state,
    const labrob::GaitConfiguration& current,
    const labrob::WalkingData& walking_data,
    int64_t t_msec)
{
  const double inv_eta2 = 1.0 / (eta_ * eta_);
  const double inv_mg   = 1.0 / (robot_mass_ * 9.81);

  // --- Update contact flags from walking plan ---
  bool lc = leftInContact(walking_data);
  bool rc = rightInContact(walking_data);
  for (int i = 0; i < C_; ++i) {
    l_contact_[i] = lc;
    r_contact_[i] = rc;
  }

  // --- 1. Propagate auxiliary trajectory ---
  auto q0    = robot_state.get_pinocchio_joint_configuration(model_);
  auto qdot0 = robot_state.get_pinocchio_joint_velocity(model_);

  q_aux_[0]    = q0;
  qdot_aux_[0] = qdot0;
  for (int i = 0; i < C_; ++i) {
    qdot_aux_[i + 1] = qdot_aux_[i] + dt_ * nu_dot_[i];
    pinocchio::integrate(model_, q_aux_[i], qdot_aux_[i] * dt_, q_aux_[i + 1]);
  }

  // --- 2. Compute Jacobians along auxiliary trajectory ---
  for (int i = 0; i < C_; ++i) {
    const auto& qi    = q_aux_[i];
    const auto& qdoti = qdot_aux_[i];
    auto& s = steps_[i];

    pinocchio::forwardKinematics(model_, data_, qi, qdoti);
    pinocchio::computeJointJacobians(model_, data_, qi);
    pinocchio::computeJointJacobiansTimeVariation(model_, data_, qi, qdoti);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::jacobianCenterOfMass(model_, data_, qi, false);
    pinocchio::computeCentroidalMap(model_, data_, qi);
    // centerOfMass with zero qddot gives acom = Jdot_com * qdot (drift)
    pinocchio::centerOfMass(model_, data_, qi, qdoti, Eigen::VectorXd::Zero(nv_));

    s.J_com  = data_.Jcom;
    s.Ag     = data_.Ag;
    s.com_pos = data_.com[0];
    s.com_vel = data_.vcom[0];
    s.Jdot_qdot_com = data_.acom[0];

    s.J_lsole.setZero();
    s.J_rsole.setZero();
    pinocchio::getFrameJacobian(model_, data_, lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, s.J_lsole);
    pinocchio::getFrameJacobian(model_, data_, rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, s.J_rsole);

    Eigen::MatrixXd Jdot_l = Eigen::MatrixXd::Zero(6, nv_);
    Eigen::MatrixXd Jdot_r = Eigen::MatrixXd::Zero(6, nv_);
    pinocchio::getFrameJacobianTimeVariation(model_, data_, lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jdot_l);
    pinocchio::getFrameJacobianTimeVariation(model_, data_, rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jdot_r);

    s.Jdot_qdot_lsole = Jdot_l * qdoti;
    s.Jdot_qdot_rsole = Jdot_r * qdoti;
    s.l_contact = l_contact_[i];
    s.r_contact = r_contact_[i];
  }

  // --- 3. Build QP ---
  // Task weights (mirrors jl2 DDP weights):
  //   Contact foot velocity damping  → wpos=100 when in contact (soft zero-vel)
  //   Swing foot tracking            → wpos=100 when swinging
  //   CoM height                     → wpos=1
  //   Joint regularization           → wpos=1e-9
  const double W_contact = 100.0;  // penalise non-zero velocity at contact foot
  const double W_swing   = 100.0;  // track desired swing foot acceleration
  const double W_com_h   = 1.0;
  const double W_reg     = 1.0;    // regularisation: prevents null-space blowup

  H_.setZero();
  f_.setZero();
  A_eq_.setZero();
  b_eq_.setZero();
  A_ineq_.setZero();

  // --- 3a. Cost (block-diagonal: each nu_dot^i decoupled in cost) ---
  for (int i = 0; i < C_; ++i) {
    const auto& s = steps_[i];
    const int col = i * nv_;

    // Regularisation
    H_.block(col, col, nv_, nv_).diagonal().array() += W_reg;

    // Left foot
    if (s.l_contact) {
      // Contact: penalise foot velocity (J_l * nu_dot ≈ 0, soft zero-velocity)
      H_.block(col, col, nv_, nv_).noalias() +=
          W_contact * s.J_lsole.transpose() * s.J_lsole;
      f_.segment(col, nv_).noalias() +=
          W_contact * s.J_lsole.transpose() * s.Jdot_qdot_lsole;
    } else {
      // Swing: track desired foot acceleration from footstep planner
      // Use PD towards target foot position (current aux trajectory state)
      const Eigen::Vector3d p_des =
          walking_data.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().p;
      const Eigen::Vector3d p_cur = data_.oMf[lsole_idx_].translation();
      const Eigen::Vector3d v_cur = s.J_lsole.topRows<3>() * qdot_aux_[i];
      const double Kp = 200.0, Kd = 20.0;
      Eigen::Matrix<double, 6, 1> a_des;
      a_des.head<3>() = Kp * (p_des - p_cur) - Kd * v_cur;
      a_des.tail<3>().setZero();
      H_.block(col, col, nv_, nv_).noalias() +=
          W_swing * s.J_lsole.transpose() * s.J_lsole;
      f_.segment(col, nv_).noalias() -=
          W_swing * s.J_lsole.transpose() * (a_des - s.Jdot_qdot_lsole);
    }

    // Right foot
    if (s.r_contact) {
      H_.block(col, col, nv_, nv_).noalias() +=
          W_contact * s.J_rsole.transpose() * s.J_rsole;
      f_.segment(col, nv_).noalias() +=
          W_contact * s.J_rsole.transpose() * s.Jdot_qdot_rsole;
    } else {
      const Eigen::Vector3d p_des =
          walking_data.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().p;
      const Eigen::Vector3d p_cur = data_.oMf[rsole_idx_].translation();
      const Eigen::Vector3d v_cur = s.J_rsole.topRows<3>() * qdot_aux_[i];
      const double Kp = 200.0, Kd = 20.0;
      Eigen::Matrix<double, 6, 1> a_des;
      a_des.head<3>() = Kp * (p_des - p_cur) - Kd * v_cur;
      a_des.tail<3>().setZero();
      H_.block(col, col, nv_, nv_).noalias() +=
          W_swing * s.J_rsole.transpose() * s.J_rsole;
      f_.segment(col, nv_).noalias() -=
          W_swing * s.J_rsole.transpose() * (a_des - s.Jdot_qdot_rsole);
    }

    // CoM height: penalise deviation from initial height
    {
      const double h_des = steps_[0].com_pos.z();
      const double h_cur = s.com_pos.z();
      const double v_z   = s.com_vel.z();
      const double a_des_z = 200.0 * (h_des - h_cur) - 20.0 * v_z;
      const Eigen::RowVectorXd Jz = s.J_com.row(2);
      H_.block(col, col, nv_, nv_).noalias() += W_com_h * Jz.transpose() * Jz;
      f_.segment(col, nv_).noalias() -=
          W_com_h * (a_des_z - s.Jdot_qdot_com.z()) * Jz.transpose();
    }
  }

  // --- 3b. DCM terminal stability equality (2 rows: x, y) ---
  // DCM_C = x_c_C + xdot_c_C / eta
  // Free evolution: com_pos[0] + (C*dt + 1/eta)*com_vel[0] + drift
  // Response coeff for nu_dot^k: c(k) = (C-1-k)*dt + dt²/2 + dt/eta
  {
    const auto& s0 = steps_[0];
    double dcm_free_x = s0.com_pos.x() + (C_ * dt_ + 1.0 / eta_) * s0.com_vel.x();
    double dcm_free_y = s0.com_pos.y() + (C_ * dt_ + 1.0 / eta_) * s0.com_vel.y();

    for (int k = 0; k < C_; ++k) {
      double c_drift = (C_ - 1 - k) * dt_ + 0.5 * dt_ * dt_ + dt_ / eta_;
      dcm_free_x += c_drift * steps_[k].Jdot_qdot_com.x();
      dcm_free_y += c_drift * steps_[k].Jdot_qdot_com.y();
    }

    // Target DCM: ZMP reference at end of horizon
    Eigen::Vector2d dcm_target = zmpReference(walking_data,
        t_msec + static_cast<int64_t>(C_) * jismpc_timestep_msec_);

    for (int k = 0; k < C_; ++k) {
      double c_k = (C_ - 1 - k) * dt_ + 0.5 * dt_ * dt_ + dt_ / eta_;
      A_eq_.block(0, k * nv_, 1, nv_) = c_k * steps_[k].J_com.row(0);
      A_eq_.block(1, k * nv_, 1, nv_) = c_k * steps_[k].J_com.row(1);
    }
    b_eq_(0) = dcm_target.x() - dcm_free_x;
    b_eq_(1) = dcm_target.y() - dcm_free_y;
  }

  // --- 3c. ZMP inequality: ref ± foot_box per step ---
  // ZMP_j^x = bias_j + [(dt²/2 - 1/eta²)*J_com[0,:] + inv_mg*Ag[4,:]] * nu_dot^j
  //         + sum_{k<j} c_pos(j,k) * J_com_k[0,:] * nu_dot^k
  for (int j = 0; j < C_; ++j) {
    const auto& sj = steps_[j];

    // Free CoM at step j+1 (contribution of drift, nu_dot=0)
    double com_free_x = steps_[0].com_pos.x() + (j + 1) * dt_ * steps_[0].com_vel.x();
    double com_free_y = steps_[0].com_pos.y() + (j + 1) * dt_ * steps_[0].com_vel.y();
    for (int k = 0; k <= j; ++k) {
      double c_pos = (j - k) * dt_ + 0.5 * dt_ * dt_;
      com_free_x += c_pos * steps_[k].Jdot_qdot_com.x();
      com_free_y += c_pos * steps_[k].Jdot_qdot_com.y();
    }
    double zmp_bias_x = com_free_x - inv_eta2 * sj.Jdot_qdot_com.x();
    double zmp_bias_y = com_free_y - inv_eta2 * sj.Jdot_qdot_com.y();

    A_ineq_.row(j).setZero();
    A_ineq_.row(j + C_).setZero();

    for (int k = 0; k < j; ++k) {
      double c_pos = (j - k) * dt_ + 0.5 * dt_ * dt_;
      A_ineq_.block(j,      k * nv_, 1, nv_) += c_pos * steps_[k].J_com.row(0);
      A_ineq_.block(j + C_, k * nv_, 1, nv_) += c_pos * steps_[k].J_com.row(1);
    }
    // Self contribution: position + acceleration + angular momentum
    double c_self = 0.5 * dt_ * dt_ - inv_eta2;
    A_ineq_.block(j,      j * nv_, 1, nv_) =
        c_self * sj.J_com.row(0) + inv_mg * sj.Ag.row(4);
    A_ineq_.block(j + C_, j * nv_, 1, nv_) =
        c_self * sj.J_com.row(1) - inv_mg * sj.Ag.row(3);

    Eigen::Vector2d zmp_ref = zmpReference(walking_data,
        t_msec + static_cast<int64_t>(j + 1) * jismpc_timestep_msec_);
    lg_(j)      = zmp_ref.x() - foot_x_ - zmp_bias_x;
    ug_(j)      = zmp_ref.x() + foot_x_ - zmp_bias_x;
    lg_(j + C_) = zmp_ref.y() - foot_y_ - zmp_bias_y;
    ug_(j + C_) = zmp_ref.y() + foot_y_ - zmp_bias_y;
  }

  // --- 4. Solve ---
  qp_solver_ptr_->solve(H_, f_, A_eq_, b_eq_, A_ineq_, lg_, ug_);

  if (qp_solver_ptr_->get_status() != 0) {
    std::cerr << "[JISMPC] QP failed with status "
              << qp_solver_ptr_->get_status()
              << ", using previous solution\n";
    // keep nu_dot_[0] from previous iteration
  } else {
    Eigen::VectorXd w = qp_solver_ptr_->get_solution();
    for (int i = 0; i < C_; ++i)
      nu_dot_[i] = w.segment(i * nv_, nv_);
  }

  nu_dot_0_ = nu_dot_[0];

  // Shift warm-start
  for (int i = 0; i < C_ - 1; ++i)
    nu_dot_[i] = nu_dot_[i + 1];
  nu_dot_[C_ - 1].setZero();

  std::cerr << "[JISMPC] nu_dot_0 norm = " << nu_dot_0_.norm()
            << "  max = " << nu_dot_0_.cwiseAbs().maxCoeff() << "\n";
}

} // namespace labrob