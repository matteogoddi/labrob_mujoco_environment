#include <ISMPC.hpp>
#include <chrono> 

namespace labrob {


ISMPC::ISMPC(
    int64_t prediction_horizon_msec,
    int64_t mpc_timestep_msec,
    double eta,
    const Eigen::Vector3d w,
    double foot_constraint_square_length,
    double foot_constraint_square_width
) : mpc_timestep_msec_(mpc_timestep_msec),
    eta_(eta),
    foot_constraint_square_length_(foot_constraint_square_length),
    foot_constraint_square_width_(foot_constraint_square_width),
    input_(Eigen::Vector3d::Zero()) {

  double mpc_timestep = 0.001 * static_cast<double>(mpc_timestep_msec_);

  num_variables_ = 3 * prediction_horizon_msec / mpc_timestep_msec_;
  num_equality_constraints_ = 3;
  num_inequality_constraints_ = num_variables_;

  N_ = prediction_horizon_msec / mpc_timestep_msec;
  mpc_timestep_ = mpc_timestep;

  qp_solver_ptr_ = std::make_unique<labrob::QpSolver>(
      num_variables_, num_equality_constraints_, num_inequality_constraints_);

  // Setup matrices size:
  cost_function_H_ = Eigen::MatrixXd(num_variables_, num_variables_);
  cost_function_f_ = Eigen::VectorXd(num_variables_);

  A_eq_ = Eigen::MatrixXd(num_equality_constraints_, num_variables_);
  b_eq_ = Eigen::VectorXd(num_equality_constraints_);

  A_zmp_ = Eigen::MatrixXd(num_inequality_constraints_, num_variables_);
  b_zmp_max_ = Eigen::VectorXd(num_inequality_constraints_);
  b_zmp_min_ = Eigen::VectorXd(num_inequality_constraints_);

  p_ = Eigen::VectorXd::Ones(N_, 1);
  P_ = Eigen::MatrixXd::Constant(N_, N_, mpc_timestep);

  for (int i = 0; i < N_; ++i) {
    for (int j = 0; j < N_; ++j){
      if (j > i) P_(i, j) = 0;
    }
  }

  A_zmp_.setZero();
  A_zmp_.block(     0,      0, N_, N_) = P_;
  A_zmp_.block(    N_,     N_, N_, N_) = P_;
  A_zmp_.block(2 * N_, 2 * N_, N_, N_) = P_;

  zDotOptimalX = Eigen::VectorXd::Zero(N_);
  zDotOptimalY = Eigen::VectorXd::Zero(N_);
  zDotOptimalZ = Eigen::VectorXd::Zero(N_);

  mc_x_     = Eigen::VectorXd::Zero(N_);
  mc_y_     = Eigen::VectorXd::Zero(N_);
  mc_z_     = Eigen::VectorXd::Zero(N_);
  mc_theta_ = Eigen::VectorXd::Zero(N_);
  b_decay_  = Eigen::VectorXd::Zero(N_);
}

void
ISMPC::solve(
    int64_t time,
    const labrob::WalkingData& walking_data,
    const labrob::LIPState& state,
    const Eigen::Vector3d w
) {

  // express footstep planner inside walking data with support foot as reference frame

  const Eigen::Vector3d com_pos = state.com_pos_;
  const Eigen::Vector3d zmp_pos = state.zmp_pos_;

  // Footstep plan → ZMP centroid reference mc_x/y/z (no allocations)
  std::vector<int> n_k;
  n_k.reserve(walking_data.footstep_plan.size());
  for (const auto& elem : walking_data.footstep_plan)
    n_k.push_back(elem.getDuration() / mpc_timestep_msec_);

  const int n_ini = (time - walking_data.t0) / mpc_timestep_msec_;

  int n = 0, k = 0;
  while (n < N_) {
    const auto& elem         = walking_data.footstep_plan[k];
    const auto& walking_state = elem.getWalkingState();
    int n_bar = n_k[k];
    if (k == 0) n_bar -= n_ini;
    if (n + n_bar >= N_) n_bar = N_ - n;

    const Eigen::Vector3d p_sup  = elem.getFeetPlacement().getSupportFootConfiguration().p;
    const Eigen::Vector3d p_swg  = elem.getFeetPlacement().getSwingFootConfiguration().p;
    const Eigen::Matrix3d& R_sup = elem.getFeetPlacement().getSupportFootConfiguration().R;
    const Eigen::Matrix3d& R_swg = elem.getFeetPlacement().getSwingFootConfiguration().R;
    const double theta_sup = std::atan2(R_sup(1, 0), R_sup(0, 0));
    const double theta_swg = std::atan2(R_swg(1, 0), R_swg(0, 0));

    for (int i = 0; i < n_bar; ++i) {
      double s0, s1;
      if (walking_state == labrob::WalkingState::PostureRegulation ||
          walking_state == labrob::WalkingState::Standing) {
        s0 = 0.5; s1 = 0.5;
      } else if (walking_state == labrob::WalkingState::SingleSupport) {
        s0 = 1.0; s1 = 0.0;
      } else if (walking_state == labrob::WalkingState::Starting) {
        const double s = (k == 0)
            ? static_cast<double>(n_ini + i) / n_k[0]
            : static_cast<double>(i) / n_bar;
        s0 = 0.5 + 0.5 * s; s1 = 0.5 - 0.5 * s;
      } else if (walking_state == labrob::WalkingState::DoubleSupport) {
        const double s = (k == 0)
            ? static_cast<double>(n_ini + i) / n_k[0]
            : static_cast<double>(i) / n_bar;
        s0 = s; s1 = 1.0 - s;
      } else { // Stopping
        const double s = (k == 0)
            ? static_cast<double>(n_ini + i) / n_k[0]
            : static_cast<double>(i) / n_bar;
        s0 = 1.0 - 0.5 * s; s1 = 0.5 * s;
      }
      mc_x_(n + i) = s0 * p_sup.x() + s1 * p_swg.x();
      mc_y_(n + i) = s0 * p_sup.y() + s1 * p_swg.y();
      mc_z_(n + i) = s0 * p_sup.z() + s1 * p_swg.z();
      mc_theta_(n + i) = s0 * theta_sup + s1 * theta_swg;
    }
    n += n_bar;
    ++k;
  }

  const double half_len = foot_constraint_square_length_ / 2.0;
  const double half_wid = foot_constraint_square_width_  / 2.0;
  const double half_height = 0.05; // [cm] above and below the foot

  // Sagittal/lateral (foot-frame) ZMP box constraint: rotate the world-frame
  // ZMP-to-foot offset by the (interpolated) foot yaw before applying the
  // axis-aligned half_len/half_wid bounds. theta is a parameter here (taken
  // from the footstep plan), not a decision variable, so the constraint
  // stays affine in the QP unknowns.
  const Eigen::ArrayXd mc_cos = mc_theta_.array().cos();
  const Eigen::ArrayXd mc_sin = mc_theta_.array().sin();
  const Eigen::ArrayXd dx = mc_x_.array() - zmp_pos(0);
  const Eigen::ArrayXd dy = mc_y_.array() - zmp_pos(1);

  A_zmp_.block(     0,      0, N_, N_) = mc_cos.matrix().asDiagonal() * P_;
  A_zmp_.block(     0,     N_, N_, N_) = mc_sin.matrix().asDiagonal() * P_;
  A_zmp_.block(    N_,      0, N_, N_) = (-mc_sin).matrix().asDiagonal() * P_;
  A_zmp_.block(    N_,     N_, N_, N_) = mc_cos.matrix().asDiagonal() * P_;

  b_zmp_min_.head(N_).array() = -half_len + mc_cos * dx + mc_sin * dy;
  b_zmp_max_.head(N_).array() =  half_len + mc_cos * dx + mc_sin * dy;
  b_zmp_min_.segment(N_, N_).array() = -half_wid - mc_sin * dx + mc_cos * dy;
  b_zmp_max_.segment(N_, N_).array() =  half_wid - mc_sin * dx + mc_cos * dy;

  
  //b_zmp_min_.tail(N_) = mc_z_.array() - (0.0 + zmp_pos(2));
  b_zmp_min_.tail(N_) = -0.0 + (mc_z_.array() - zmp_pos(2));
  //b_zmp_max_.tail(N_) = mc_z_.array() + (half_height - zmp_pos(2));
  b_zmp_max_.tail(N_) = half_height + (mc_z_.array() - zmp_pos(2));

  // Geometric decay b_decay_(i) = exp(-eta*dt)^i (no pow/exp per iteration)
  A_eq_.setZero();
  {
    double acc = 1.0;
    const double base = std::exp(-eta_ * mpc_timestep_);
    for (int i = 0; i < N_; ++i, acc *= base)
      b_decay_(i) = acc;
  }

  const double aeq_scale = (1.0 / eta_) * (1.0 - std::exp(-eta_ * mpc_timestep_));
  A_eq_.block(0,      0, 1, N_) = aeq_scale * b_decay_.transpose();
  A_eq_.block(1,     N_, 1, N_) = aeq_scale * b_decay_.transpose();
  A_eq_.block(2, 2 * N_, 1, N_) = aeq_scale * b_decay_.transpose();

  // Use this for LIP constraint
  /*
  b_eq_ << com_pos(0) + state.com_vel_(0) / eta_ - zmp_pos(0) + w.x()/std::pow(eta_, 2.0),
      com_pos(1) + state.com_vel_(1) / eta_ - zmp_pos(1) + w.y()/std::pow(eta_, 2.0),
      com_pos(2) + state.com_vel_(2) / eta_ - (zmp_pos(2) + 9.81 / std::pow(eta_, 2.0));
    */
  
  // PLIP constraint
  
  b_eq_ << com_pos(0) + state.com_vel_(0) / eta_ - zmp_pos(0) + w.x()/std::pow(eta_, 2.0),
      com_pos(1) + state.com_vel_(1) / eta_ - zmp_pos(1) + w.y()/std::pow(eta_, 2.0),
      com_pos(2) + state.com_vel_(2) / eta_ - zmp_pos(2) + w.z() / std::pow(eta_, 2.0);
    

  const Eigen::MatrixXd H_block = Eigen::MatrixXd::Identity(N_, N_) + beta_ * P_.transpose() * P_;
  cost_function_H_.setZero();
  cost_function_H_.block(     0,      0, N_, N_) = H_block;
  cost_function_H_.block(    N_,     N_, N_, N_) = H_block;
  cost_function_H_.block(2 * N_, 2 * N_, N_, N_) = H_block;

  cost_function_f_.segment(     0, N_).noalias() = beta_ * P_.transpose() * (p_ * zmp_pos.x() - mc_x_);
  cost_function_f_.segment(    N_, N_).noalias() = beta_ * P_.transpose() * (p_ * zmp_pos.y() - mc_y_);
  cost_function_f_.segment(2 * N_, N_).noalias() = beta_ * P_.transpose() * (p_ * zmp_pos.z() - mc_z_);

  // Solve QP
  qp_solver_ptr_->solve(
      cost_function_H_,
      cost_function_f_,
      A_eq_,
      b_eq_,
      A_zmp_,
      b_zmp_min_,
      b_zmp_max_
  );
  auto decisionVariables = qp_solver_ptr_->get_solution();

  // Split the QP solution in ZMP dot and footsteps
  zDotOptimalX = (decisionVariables.segment(     0, N_));
  zDotOptimalY = (decisionVariables.segment(    N_, N_));
  zDotOptimalZ = (decisionVariables.segment(2 * N_, N_));

  input_.x() = zDotOptimalX(0);
  input_.y() = zDotOptimalY(0);
  input_.z() = zDotOptimalZ(0);
}

const Eigen::Vector3d& ISMPC::getInput() const {
  return input_;
}

const Eigen::VectorXd& ISMPC::getInputSequenceX() const {
  return zDotOptimalX;
}
const Eigen::VectorXd& ISMPC::getInputSequenceY() const {
  return zDotOptimalY;
}
const Eigen::VectorXd& ISMPC::getInputSequenceZ() const {
  return zDotOptimalZ;
}

//return A_eq^-1*b_eq
Eigen::Vector3d
ISMPC::getStabConstraintOffset() const {
  //use ldlt
  Eigen::VectorXd offset = A_eq_.ldlt().solve(b_eq_);
  //return only first 3 elements (x,y,z)
  return Eigen::Vector3d(offset(0), offset(1), offset(2));
}

Eigen::Vector3d
ISMPC::getZmpConstraintBoxCenter() const {
  return Eigen::Vector3d(mc_x_(0), mc_y_(0), mc_z_(0));
}

double
ISMPC::getZmpConstraintBoxYaw() const {
  return mc_theta_(0);
}

double
ISMPC::getEta() const {
  return eta_;
}

void
ISMPC::setEta(double eta) {
  eta_ = eta;
}

double
ISMPC::clamp(double n, double n_min, double n_max) {
  return std::max(n_min, std::min(n_max, n));
}

} // end namespace labrob