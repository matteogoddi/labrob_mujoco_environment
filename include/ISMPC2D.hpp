#ifndef LABROB_ISMPC2D_HPP_
#define LABROB_ISMPC2D_HPP_

#include <memory>
#include <vector>

#include <LIPState.hpp>
#include <WalkingData.hpp>
#include <QpSolver.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace labrob {

class ISMPC2D {
 public:
  ISMPC2D(
      int64_t prediction_horizon_msec,
      int64_t mpc_timestep_msec,
      double eta,
      double foot_constraint_square_length,
      double foot_constraint_square_width
  );

  void solve(
      int64_t time,
      const labrob::WalkingData& walking_data,
      const labrob::LIPState& state
  );

  const Eigen::Vector2d& getInput() const;

  const Eigen::VectorXd& getInputSequenceX() const;
  const Eigen::VectorXd& getInputSequenceY() const;

  // Eigen::Vector2d getStabConstraintOffset() const;

  double getEta() const;
  void setEta(double eta);

  // Current-step ZMP box constraint (for logging/diagnostics): the box is
  // [center - half_size, center + half_size] on each axis, where center is
  // the support/swing-foot-interpolated centroid used at n=0 of the horizon.
  double getZmpBoxCenterX() const { return mc_x_(0); }
  double getZmpBoxCenterY() const { return mc_y_(0); }
  double getFootConstraintLength() const { return foot_constraint_square_length_; }
  double getFootConstraintWidth()  const { return foot_constraint_square_width_; }

  void resetInput() {
    input_ = Eigen::Vector2d::Zero();
  }

 private:
  double clamp(double n, double n_min, double n_max);

  int num_variables_;
  int num_equality_constraints_;
  int num_inequality_constraints_;
  int N_;

  int64_t mpc_timestep_msec_;
  double eta_;
  double foot_constraint_square_length_;
  double foot_constraint_square_width_;

  Eigen::VectorXd p_;
  Eigen::MatrixXd P_;

  Eigen::MatrixXd cost_function_H_;
  Eigen::VectorXd cost_function_f_;
  double beta_x_ = 10000.0;
  double beta_y_ = 10000.0;

  double single_support_zmp_blend_ = 1.0;

  Eigen::MatrixXd A_eq_;
  Eigen::VectorXd b_eq_;

  Eigen::MatrixXd A_zmp_;
  Eigen::VectorXd b_zmp_max_;
  Eigen::VectorXd b_zmp_min_;

  Eigen::Vector2d input_;
  Eigen::VectorXd zDotOptimalX;
  Eigen::VectorXd zDotOptimalY;

  std::unique_ptr<labrob::QpSolver> qp_solver_ptr_;

  Eigen::VectorXd mc_x_, mc_y_;
  Eigen::VectorXd b_decay_;
  double mpc_timestep_;

}; // end class ISMPC2D

} // end namespace labrob

#endif // LABROB_ISMPC2D_HPP_