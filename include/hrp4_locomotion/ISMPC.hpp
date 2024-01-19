#ifndef LABROB_ISMPC_HPP_
#define LABROB_ISMPC_HPP_

// STL
#include <fstream>
#include <memory>
#include <vector>

#include <hrp4_locomotion/ISMPCState.hpp>
#include <hrp4_locomotion/WalkingData.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace labrob {

class ISMPC{
 public:
  ISMPC(
      int64_t prediction_horizon_msec,
      int64_t mpc_timestep_msec,
      int64_t control_timestep_msec,
      double com_target_height,
      double foot_constraint_square_width,
      const ISMPCState& state
  );

  void solve(int64_t time, const labrob::WalkingData& walking_data);

  const ISMPCState& getState() const;

  void setCOMTargetHeight(double com_target_height);

  void setState(const ISMPCState& state);

  Eigen::Vector3d updateState(double zmpDot, int dim);

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
  double com_target_height_;
  double foot_constraint_square_width_;

  // Matrices for prediction:
  Eigen::VectorXd p_;
  Eigen::MatrixXd P_;

  // Matrices for cost function:
  Eigen::MatrixXd cost_function_H_;
  Eigen::VectorXd cost_function_f_;
  double beta_ = 1000.0;

  // Matrices for stability constraint:
  Eigen::MatrixXd A_eq_;
  Eigen::VectorXd b_eq_;

  //Matrices for balance constraint:
  Eigen::MatrixXd A_zmp_;
  Eigen::VectorXd b_zmp_max_;
  Eigen::VectorXd b_zmp_min_;

  // State:
  ISMPCState state_;

  // QP solver:
  std::shared_ptr<labrob::qpsolvers::QPSolverEigenWrapper<double>> qp_solver_ptr_;

  std::ofstream pred_log_file_;

}; // end class ISMPC

} // end namespace labrob

#endif // LABROB_ISMPC_HPP_