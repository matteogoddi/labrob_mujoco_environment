#pragma once

#include <Eigen/Dense>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/skew.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <iostream>

#include <hrp4_locomotion/utils.hpp>

namespace labrob
{

// =======================
// Joint velocity KF
// =======================
class JointKF
{
public:
  JointKF(double dt, int nq);

  Eigen::VectorXd filter(const Eigen::VectorXd& q_meas, const Eigen::VectorXd& qdd);

private:
  Eigen::VectorXd q_filtered_;
  Eigen::MatrixXd K;
  Eigen::MatrixXd F;
  Eigen::MatrixXd G;
  Eigen::MatrixXd H;
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
// BaseEKF class
// =======================

class BaseEKF
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Constructor
    BaseEKF(const pinocchio::Model& model, Eigen::VectorXd& q_init, double dt):
      model_(model), q_init_(q_init), dt_(dt)
    {
      P_.setIdentity();
      P_.block<3,3>(0,0) *= 1e-2;    // position
      P_.block<3,3>(3,3) *= 1e-2;    // velocity
      P_.block<3,3>(6,6) *= 1e-3;    // orientation
      P_.block<3,3>(9,9) *= 1e-1;    // feet
      P_.block<3,3>(12,12) *= 1e-1;
      P_.block<3,3>(15,15) *= 1e-4;  // biases
      P_.block<3,3>(18,18) *= 1e-4;
      Qc_.setIdentity();
      Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
      Qc_.block<3,3>(3,3) = 0.1 * I;     // accel noise
      Qc_.block<3,3>(6,6) = 0.05 * I;     // gyro noise
      Qc_.block<3,3>(15,15) = 1e-6 * I;   // accel bias RW
      Qc_.block<3,3>(18,18) = 1e-6 * I;   // gyro bias RW
      Qc_.block<3,3>(9,9)  = 1e-5 * I;    // foot noise
      Qc_.block<3,3>(12,12)= 1e-5 * I;
      R_.setIdentity() * 5e-4;
      g_ << 0, 0, -9.81;
      r_ << q_init[0], q_init[1], q_init[2];
      q_ = Eigen::Quaterniond(q_init[6], q_init[3], q_init[4], q_init[5]);
      pinocchio::Data data_(model_);
      pinocchio::forwardKinematics(model_, data_, q_init_);
      pinocchio::framesForwardKinematics(model_, data_, q_init_);
      if (!left_initialized_)
      {
        const auto& bMf = data_.oMf[model_.getFrameId("left_foot_link")];
        pL_ = q_.toRotationMatrix().transpose() * bMf.translation();
        zL_ = q_ * Eigen::Quaterniond(bMf.rotation());
        left_initialized_ = true;
      }
      if (!right_initialized_)
      {
        const auto& bMf = data_.oMf[model_.getFrameId("right_foot_link")];
        pR_ = q_.toRotationMatrix().transpose() * bMf.translation();
        zR_ = q_ * Eigen::Quaterniond(bMf.rotation());
        right_initialized_ = true;
      }
    }

    // Complete filter step (prediction + update)
    void filter(const Eigen::Vector3d& acc_meas,
              const Eigen::Vector3d& gyro_meas,
              const Eigen::VectorXd& joint_pos_meas,
              bool isLeftFootinContact,
              bool isRightFootinContact);
    
    Eigen::Vector3d getBasePosition() const { return r_; }
    Eigen::Vector3d getBaseVelocity() const { return v_; }
    Eigen::Quaterniond getBaseOrientation() const { return q_; }

private:

    pinocchio::Model model_;
    pinocchio::Data data_;
    Eigen::VectorXd q_init_;
    bool left_initialized_ = false;
    bool right_initialized_ = false;

    // Helper
    Eigen::Quaterniond expMap(const Eigen::Vector3d& w)
    {
      double th = w.norm();
      if (th > M_PI) th -= 2*M_PI;
      else if (th < -M_PI) th += 2*M_PI;

      if(th < 1e-8)
        return Eigen::Quaterniond::Identity();
      
      Eigen::Vector3d axis = w/th;
      return Eigen::Quaterniond(Eigen::AngleAxisd(th, axis));
    }

    Eigen::Vector3d logMap(const Eigen::Quaterniond& q)
    {
      Eigen::AngleAxisd aa(q);
      if (aa.angle() > M_PI) aa.angle() -= 2*M_PI;
      else if (aa.angle() < -M_PI) aa.angle() += 2*M_PI;

      return aa.axis() * aa.angle();
    }

    double dt_;
    int NX = 27;

    // Nominal state
    Eigen::Vector3d r_;
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_;

    Eigen::Vector3d pL_;
    Eigen::Vector3d pR_;
    Eigen::Quaterniond zL_;
    Eigen::Quaterniond zR_;

    Eigen::Vector3d bf_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d bw_ = Eigen::Vector3d::Zero();

    // Covariance               
    Eigen::Matrix<double,27,27> P_;
    Eigen::Matrix<double,27,27> Qc_;
    Eigen::Matrix<double,12,12> R_;

    Eigen::Vector3d g_;
};

class CoMKF
{
public:
  CoMKF(double dt, double eta): dt_(dt), eta_(eta) {};

  LIPState filter(LIPState filtered, LIPState current, const Eigen::Vector3d& input);

  double getEta() const { return eta_; };
  void setEta(double eta) { eta_ = eta; };

private:
  double dt_;
  double eta_;

  Eigen::Matrix3d cov_x = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d cov_y = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d cov_z = Eigen::Matrix3d::Identity();

  double cov_meas_pos = 1.0e1;
  double cov_meas_vel = 1.0e2;
  double cov_meas_zmp = 1.0e8;

  double cov_mod_pos = 1.0;
  double cov_mod_vel = 1.0;
  double cov_mod_zmp = 1.0;
};

} // namespace state_filtering
