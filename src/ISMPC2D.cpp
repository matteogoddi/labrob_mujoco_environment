#include <ISMPC2D.hpp>

namespace labrob {

ISMPC2D::ISMPC2D(
    int64_t prediction_horizon_msec,
    int64_t mpc_timestep_msec,
    double eta,
    double foot_constraint_square_length,
    double foot_constraint_square_width
) : mpc_timestep_msec_(mpc_timestep_msec),
    eta_(eta),
    foot_constraint_square_length_(foot_constraint_square_length),
    foot_constraint_square_width_(foot_constraint_square_width),
    input_(Eigen::Vector2d::Zero()) {

  double mpc_timestep = 0.001 * static_cast<double>(mpc_timestep_msec_);

  N_ = prediction_horizon_msec / mpc_timestep_msec;
  mpc_timestep_ = mpc_timestep;

  num_variables_             = 2 * N_;
  num_equality_constraints_  = 2;
  num_inequality_constraints_ = 2 * N_;

  qp_solver_ptr_ = std::make_unique<labrob::QpSolver>(
      num_variables_, num_equality_constraints_, num_inequality_constraints_
    //  , num_inequality_constraints_, 1e-6
    );

  cost_function_H_ = Eigen::MatrixXd(num_variables_, num_variables_);
  cost_function_f_ = Eigen::VectorXd(num_variables_);

  A_eq_ = Eigen::MatrixXd(num_equality_constraints_, num_variables_);
  b_eq_ = Eigen::VectorXd(num_equality_constraints_);

  A_zmp_    = Eigen::MatrixXd(num_inequality_constraints_, num_variables_);
  b_zmp_max_ = Eigen::VectorXd(num_inequality_constraints_);
  b_zmp_min_ = Eigen::VectorXd(num_inequality_constraints_);

  p_ = Eigen::VectorXd::Ones(N_);
  P_ = Eigen::MatrixXd::Constant(N_, N_, mpc_timestep);
  for (int i = 0; i < N_; ++i)
    for (int j = i + 1; j < N_; ++j)
      P_(i, j) = 0.0;

  A_zmp_.setZero();
  A_zmp_.block(0,  0,  N_, N_) = P_;
  A_zmp_.block(N_, N_, N_, N_) = P_;

  zDotOptimalX = Eigen::VectorXd::Zero(N_);
  zDotOptimalY = Eigen::VectorXd::Zero(N_);

  mc_x_    = Eigen::VectorXd::Zero(N_);
  mc_y_    = Eigen::VectorXd::Zero(N_);
  b_decay_ = Eigen::VectorXd::Zero(N_);
}

void
ISMPC2D::solve(
    int64_t time,
    const labrob::WalkingData& walking_data,
    const labrob::LIPState& state
) {
  const Eigen::Vector3d com_pos = state.com_pos_;
  const Eigen::Vector3d zmp_pos = state.zmp_pos_;

  std::vector<int> n_k;
  n_k.reserve(walking_data.footstep_plan.size());
  for (const auto& elem : walking_data.footstep_plan)
    n_k.push_back(elem.getDuration() / mpc_timestep_msec_);

  const int n_ini = (time - walking_data.t0) / mpc_timestep_msec_;

  const int n_elems = static_cast<int>(walking_data.footstep_plan.size());
  int n = 0, k = 0;
  while (n < N_) {
    // Once the plan is exhausted (e.g. collapsed to the single perpetual
    // "Standing" tail element), keep reusing the last element for the rest
    // of the horizon instead of indexing past the end of the deque.
    const int kk = std::min(k, n_elems - 1);
    const auto& elem          = walking_data.footstep_plan[kk];
    const auto& walking_state = elem.getWalkingState();
    int n_bar = n_k[kk];
    if (k == 0) n_bar -= n_ini;
    if (n + n_bar >= N_ || kk == n_elems - 1) n_bar = N_ - n;

    const Eigen::Vector3d p_sup = elem.getFeetPlacement().getSupportFootConfiguration().p;
    const Eigen::Vector3d p_swg = elem.getFeetPlacement().getSwingFootConfiguration().p;

    for (int i = 0; i < n_bar; ++i) {
      double s0, s1;
      if (walking_state == labrob::WalkingState::PostureRegulation ||
          walking_state == labrob::WalkingState::Standing) {
        s0 = 0.5; s1 = 0.5;
      } else if (walking_state == labrob::WalkingState::SingleSupport) {
        s0 = single_support_zmp_blend_; s1 = 1.0 - single_support_zmp_blend_;
      } else if (walking_state == labrob::WalkingState::Starting) {
        // Ramp from Standing (0.5) toward the same blend value SingleSupport
        // locks onto, so there is no discontinuity at the Starting->SingleSupport
        // boundary.
        const double s = (k == 0)
            ? static_cast<double>(n_ini + i) / n_k[0]
            : static_cast<double>(i) / n_bar;
        s0 = 0.5 + (single_support_zmp_blend_ - 0.5) * s; s1 = 1.0 - s0;
      } else if (walking_state == labrob::WalkingState::DoubleSupport) {
        // Ramp from the previous step's committed blend (1 - blend, since
        // p_sup/p_swg swap identity) to this step's committed blend, so
        // there is no discontinuity at either boundary with SingleSupport.
        const double s = (k == 0)
            ? static_cast<double>(n_ini + i) / n_k[0]
            : static_cast<double>(i) / n_bar;
        s0 = (1.0 - single_support_zmp_blend_)
             + (2.0 * single_support_zmp_blend_ - 1.0) * s;
        s1 = 1.0 - s0;
      } else { // Stopping
        // Ramp from the committed blend (end of the last SingleSupport) back
        // down to Standing (0.5), no discontinuity at the SingleSupport->
        // Stopping boundary.
        const double s = (k == 0)
            ? static_cast<double>(n_ini + i) / n_k[0]
            : static_cast<double>(i) / n_bar;
        s0 = single_support_zmp_blend_ - (single_support_zmp_blend_ - 0.5) * s;
        s1 = 1.0 - s0;
      }
      mc_x_(n + i) = s0 * p_sup.x() + s1 * p_swg.x();
      mc_y_(n + i) = s0 * p_sup.y() + s1 * p_swg.y();
    }
    n += n_bar;
    ++k;
  }

  const double half_len = foot_constraint_square_length_ / 2.0;
  const double half_wid = foot_constraint_square_width_  / 2.0;
  b_zmp_min_.head(N_) = mc_x_.array() - (half_len + zmp_pos(0));
  b_zmp_min_.tail(N_) = mc_y_.array() - (half_wid + zmp_pos(1));
  b_zmp_max_.head(N_) = mc_x_.array() + (half_len - zmp_pos(0));
  b_zmp_max_.tail(N_) = mc_y_.array() + (half_wid - zmp_pos(1));

  A_eq_.setZero();
  {
    double acc = 1.0;
    const double base = std::exp(-eta_ * mpc_timestep_);
    for (int i = 0; i < N_; ++i, acc *= base)
      b_decay_(i) = acc;
  }

  const double aeq_scale = (1.0 / eta_) * (1.0 - std::exp(-eta_ * mpc_timestep_));
  A_eq_.block(0, 0,  1, N_) = aeq_scale * b_decay_.transpose();
  A_eq_.block(1, N_, 1, N_) = aeq_scale * b_decay_.transpose();

  b_eq_ << com_pos(0) + state.com_vel_(0) / eta_ - zmp_pos(0),
           com_pos(1) + state.com_vel_(1) / eta_ - zmp_pos(1);

  const Eigen::MatrixXd PtP = P_.transpose() * P_;
  cost_function_H_.setZero();
  cost_function_H_.block(0,  0,  N_, N_) = Eigen::MatrixXd::Identity(N_, N_) + beta_x_ * PtP;
  cost_function_H_.block(N_, N_, N_, N_) = Eigen::MatrixXd::Identity(N_, N_) + beta_y_ * PtP;

  cost_function_f_.head(N_).noalias() = beta_x_ * P_.transpose() * (p_ * zmp_pos.x() - mc_x_);
  cost_function_f_.tail(N_).noalias() = beta_y_ * P_.transpose() * (p_ * zmp_pos.y() - mc_y_);

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

  zDotOptimalX = decisionVariables.head(N_);
  zDotOptimalY = decisionVariables.tail(N_);

  input_.x() = zDotOptimalX(0);
  input_.y() = zDotOptimalY(0);
}

const Eigen::Vector2d& ISMPC2D::getInput() const {
  return input_;
}

const Eigen::VectorXd& ISMPC2D::getInputSequenceX() const {
  return zDotOptimalX;
}

const Eigen::VectorXd& ISMPC2D::getInputSequenceY() const {
  return zDotOptimalY;
}

// Eigen::Vector2d ISMPC2D::getStabConstraintOffset() const {
//   Eigen::VectorXd offset = A_eq_.ldlt().solve(b_eq_);
//   return Eigen::Vector2d(offset(0), offset(1));
// }

double ISMPC2D::getEta() const {
  return eta_;
}

void ISMPC2D::setEta(double eta) {
  eta_ = eta;
}

double ISMPC2D::clamp(double n, double n_min, double n_max) {
  return std::max(n_min, std::min(n_max, n));
}

} // end namespace labrob