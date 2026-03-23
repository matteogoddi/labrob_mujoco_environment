#ifndef LABROB_ESTIMATE_FORCE_HPP_
#define LABROB_ESTIMATE_FORCE_HPP_

#include <vector>

#include <Eigen/Core>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <hrp4_locomotion/RobotState.hpp>

namespace labrob {

class EstimateForce {
 public:
  EstimateForce();
  explicit EstimateForce(const pinocchio::Model& robot_model);

  void initialize(const pinocchio::Model& robot_model);
  void update(const labrob::RobotState& robot_state);

  const Eigen::VectorXd& getLeftWristWrench() const;
  const Eigen::VectorXd& getRightWristWrench() const;
  const Eigen::VectorXd& getLeftWristWrenchFiltered() const;
  const Eigen::VectorXd& getRightWristWrenchFiltered() const;

 private:
  void computeModelTorque(
      const Eigen::VectorXd& q,
      const Eigen::VectorXd& dq,
      const Eigen::VectorXd& ddq
  );
  void computeResidualTorque(const Eigen::VectorXd& tau_measured);
  Eigen::VectorXd computeEstimatedWrench(
      const Eigen::MatrixXd& jacobian,
      const Eigen::VectorXd& tau_res
  ) const;
  Eigen::VectorXd selectArmResiduals(const std::vector<int>& arm_velocity_indices) const;

 private:
  bool initialized_;

  pinocchio::Model robot_model_;
  pinocchio::Data robot_data_;

  std::vector<int> left_arm_velocity_indices_;
  std::vector<int> right_arm_velocity_indices_;

  pinocchio::FrameIndex left_wrist_frame_id_;
  pinocchio::FrameIndex right_wrist_frame_id_;

  Eigen::VectorXd tau_model_;
  Eigen::VectorXd tau_residual_;

  Eigen::VectorXd left_tau_res_;
  Eigen::VectorXd right_tau_res_;

  Eigen::VectorXd left_wrench_;
  Eigen::VectorXd right_wrench_;
  Eigen::VectorXd left_wrench_filtered_;
  Eigen::VectorXd right_wrench_filtered_;
  Eigen::VectorXd left_wrench_bias_;
  Eigen::VectorXd right_wrench_bias_;

  int bias_sample_count_;
  int bias_sample_target_;

  double alpha_;
  double damping_;
};

} // end namespace labrob

#endif // LABROB_ESTIMATE_FORCE_HPP_