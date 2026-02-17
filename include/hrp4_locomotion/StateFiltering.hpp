#pragma once

#include <Eigen/Dense>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/skew.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace labrob
{

// =======================
// Joint velocity LPF
// =======================
class JointVelocityFilter
{
public:
  JointVelocityFilter(double cutoff_freq, double dt, int nq);

  Eigen::VectorXd filter(const Eigen::VectorXd& qd_meas);

private:
  double alpha_;
  Eigen::VectorXd qd_filtered_;
};

// =======================
// EKF State
// =======================
struct EKFState
{
  Eigen::Vector3d p = Eigen::Vector3d::Zero();
  Eigen::Vector3d v = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  Eigen::Vector3d ba = Eigen::Vector3d::Zero();

  Eigen::Matrix<double,15,15> P =
      Eigen::Matrix<double,15,15>::Identity() * 1e-3;
};

// =======================
// IMU calibration
// =======================
Eigen::Matrix3d calibrateImuRotation(
    const std::vector<Eigen::Vector3d>& acc_samples,
    const Eigen::Matrix3d& R_world_base);

// =======================
// EKF class
// =======================
class BaseEKF
{
public:
  BaseEKF(double dt);

  void setImuExtrinsics(const Eigen::Matrix3d& R_base_imu);

  void predict(const Eigen::Vector3d& omega_m,
               const Eigen::Vector3d& acc_m);

  void updateFootContact(const pinocchio::Model& model,
                         pinocchio::Data& data,
                         int foot_frame_id,
                         const Eigen::VectorXd& q,
                         const Eigen::VectorXd& v);

  const EKFState& getState() const { return state_; }

private:
  double dt_;
  EKFState state_;
  Eigen::Matrix3d R_base_imu_ = Eigen::Matrix3d::Identity();
};

} // namespace state_filtering
