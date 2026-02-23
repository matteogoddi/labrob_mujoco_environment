#include <hrp4_locomotion/StateFiltering.hpp>
#include <pinocchio/spatial/explog.hpp>

namespace labrob
{

//////////////////////////
// Joint velocity LPF
//////////////////////////

JointVelocityFilter::JointVelocityFilter(double cutoff_freq,
                                         double dt,
                                         int nq)
{
  double rc = 1.0 / (2.0 * M_PI * cutoff_freq);
  alpha_ = dt / (dt + rc);
  qd_filtered_ = Eigen::VectorXd::Zero(nq);
}

Eigen::VectorXd
JointVelocityFilter::filter(const Eigen::VectorXd& qd_meas)
{
  qd_filtered_ = alpha_ * qd_meas +
                 (1.0 - alpha_) * qd_filtered_;
  return qd_filtered_;
}

//////////////////////////
// IMU calibration
//////////////////////////

Eigen::Matrix3d calibrateImuRotation(
    const std::vector<Eigen::Vector3d>& acc_samples,
    const Eigen::Matrix3d& R_world_base)
{
  Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero();
  for (const auto& a : acc_samples) acc_mean += a;
  acc_mean /= acc_samples.size();

  Eigen::Vector3d g_imu = acc_mean.normalized();

  Eigen::Vector3d g_world(0.0, 0.0, 1.0);
  Eigen::Vector3d g_base = R_world_base.transpose() * g_world;

  Eigen::Vector3d v = g_imu.cross(g_base);
  double c = g_imu.dot(g_base);

  Eigen::Matrix3d vx;
  vx <<     0, -v.z(),  v.y(),
         v.z(),     0, -v.x(),
        -v.y(),  v.x(),     0;

  Eigen::Matrix3d R_delta =
      Eigen::Matrix3d::Identity() +
      vx +
      vx * vx * (1.0 / (1.0 + c));

  return R_delta;
}

//////////////////////////
// EUCLIDEAN EKF FOR BASE ESTIMATION 
//////////////////////////

BaseEKF::BaseEKF(double dt) : dt_(dt) {}

void BaseEKF::baseEstimation(
    const Eigen::Vector3d& omega_m,
    const Eigen::Vector3d& acc_m,
    const pinocchio::Model& model,
    pinocchio::Data& data,
    const Eigen::VectorXd& q,
    const Eigen::VectorXd& v,
    bool left_contact,
    bool right_contact,
    int left_foot_frame,
    int right_foot_frame)
{

  // =====================================================
  // 1) PREDICT STEP (IMU propagation)
  // =====================================================

  Eigen::Vector3d omega =
      R_base_imu_ * omega_m - state_.bg;
  Eigen::Vector3d acc =
      R_base_imu_ * acc_m - state_.ba;

  Eigen::Vector3d g(0, 0, -9.81);

  // Nominal propagation
  state_.R = state_.R *
             pinocchio::exp3(omega * dt_);

  Eigen::Vector3d a_world =
      state_.R * acc + g;

  state_.p += state_.v * dt_
              + 0.5 * a_world * dt_ * dt_;

  state_.v += a_world * dt_;

  // ================= Jacobian F =================
  Eigen::Matrix<double,15,15> F =
      Eigen::Matrix<double,15,15>::Identity();

  F.block<3,3>(0,3) =
      Eigen::Matrix3d::Identity() * dt_;

  F.block<3,3>(3,6) =
      -state_.R * pinocchio::skew(acc) * dt_;

  F.block<3,3>(3,12) =
      -state_.R * dt_;

  F.block<3,3>(6,6) =
      Eigen::Matrix3d::Identity()
      - pinocchio::skew(omega) * dt_;

  F.block<3,3>(6,9) =
      -Eigen::Matrix3d::Identity() * dt_;

  // ================= Process noise =================
  Eigen::Matrix<double,15,15> Q =
      Eigen::Matrix<double,15,15>::Zero();

  double sg = 1e-4;
  double sa = 1e-3;
  double sbg = 1e-6;
  double sba = 1e-6;

  Q.block<3,3>(3,3) =
      Eigen::Matrix3d::Identity() * sa * dt_;
  Q.block<3,3>(6,6) =
      Eigen::Matrix3d::Identity() * sg * dt_;
  Q.block<3,3>(9,9) =
      Eigen::Matrix3d::Identity() * sbg * dt_;
  Q.block<3,3>(12,12) =
      Eigen::Matrix3d::Identity() * sba * dt_;

  // ================= Prediction Covariance Update =================
  state_.P = F * state_.P * F.transpose() + Q;

  // =====================================================
  // 2) UPDATE STEP (FOOT CONTACTS)
  // =====================================================

  auto updateSingleFoot = [&](int foot_frame_id)
  {
    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::updateFramePlacements(model, data);

    pinocchio::Motion vf =
        pinocchio::getFrameVelocity(
            model, data,
            foot_frame_id,
            pinocchio::LOCAL_WORLD_ALIGNED);

    Eigen::Vector3d residual = -vf.linear();  // zero velocity assumption

    Eigen::Vector3d rf = model.frames[foot_frame_id].placement.translation();

    Eigen::Matrix<double,3,15> H = Eigen::Matrix<double,3,15>::Zero();

    H.block<3,3>(0,3) = Eigen::Matrix3d::Identity();

    H.block<3,3>(0,6) = -pinocchio::skew(state_.R * rf);

    Eigen::Matrix3d Rm = Eigen::Matrix3d::Identity() * 1e-3;

    Eigen::Matrix3d S = H * state_.P * H.transpose() + Rm;

    Eigen::Matrix<double,15,3> K = state_.P * H.transpose() * S.inverse();

    Eigen::Matrix<double,15,1> dx = K * residual;

    // Inject correction
    state_.p += dx.segment<3>(0);
    state_.v += dx.segment<3>(3);

    Eigen::Vector3d dtheta = dx.segment<3>(6);

    state_.R = state_.R * pinocchio::exp3(dtheta);

    state_.bg += dx.segment<3>(9);
    state_.ba += dx.segment<3>(12);

    Eigen::Matrix<double,15,15> I = Eigen::Matrix<double,15,15>::Identity();

    state_.P = (I - K * H) * state_.P;
  };

  if (left_contact)
    updateSingleFoot(left_foot_frame);

  if (right_contact)
    updateSingleFoot(right_foot_frame);
}

// TO DO: DILIGENT EKF (LIE GROUP PAPER)

} // namespace state_filtering
