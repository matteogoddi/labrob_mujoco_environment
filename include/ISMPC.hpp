#ifndef LABROB_ISMPC_HPP_
#define LABROB_ISMPC_HPP_

// STL
#include <fstream>
#include <memory>
#include <vector>

#include <LIPState.hpp>
#include <WalkingData.hpp>

#include <QpSolverMPC.hpp>
#include <QpSolver.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace labrob {

class ISMPC{
 public:
  ISMPC(
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

  const Eigen::Vector3d& getInput() const;

  const Eigen::VectorXd& getInputSequenceX() const;
  const Eigen::VectorXd& getInputSequenceY() const;
  const Eigen::VectorXd& getInputSequenceZ() const;

  Eigen::Vector3d getStabConstraintOffset() const;
  
  double getEta() const;

  void setEta(double eta);

  void resetInput(){
      input_ = Eigen::Vector3d::Zero();
  }

 private:
  // NOTE: std::clamp available from C++17
  double clamp(double n, double n_min, double n_max);

  // Constant parameters:
  int num_variables_;
  int num_equality_constraints_;
  int num_inequality_constraints_;
  int N_;

  int64_t mpc_timestep_msec_;
  int64_t control_timestep_msec_;
  double eta_;
  double foot_constraint_square_length_;
  double foot_constraint_square_width_;

  // Matrices for prediction:
  Eigen::VectorXd p_;
  Eigen::MatrixXd P_;

  // Matrices for cost function:
  Eigen::MatrixXd cost_function_H_;
  Eigen::VectorXd cost_function_f_;
  double beta_x_ = 10000.0;
  double beta_y_ = 10000.0;
  double beta_z_ = 10000.0;
  double alpha_z_ = 0.0;  // CoM height regularization weight


  // Matrices for stability constraint:
  Eigen::MatrixXd A_eq_;
  Eigen::VectorXd b_eq_;

  //Matrices for balance constraint:
  Eigen::MatrixXd A_zmp_;
  Eigen::VectorXd b_zmp_max_;
  Eigen::VectorXd b_zmp_min_;

  Eigen::Vector3d input_;
  Eigen::VectorXd zDotOptimalX;
  Eigen::VectorXd zDotOptimalY;
  Eigen::VectorXd zDotOptimalZ;

  // QP solver:
  std::unique_ptr<labrob::QpSolver> qp_solver_ptr_;

  // Pre-allocated per-solve buffers (no malloc in hot path):
  Eigen::VectorXd mc_x_, mc_y_, mc_z_;  // reference ZMP centroid per step
  Eigen::VectorXd b_decay_;             // geometric decay sequence for A_eq
  double mpc_timestep_;                 // 0.001 * mpc_timestep_msec_

}; // end class ISMPC

} // end namespace labrob

#endif // LABROB_ISMPC_HPP_