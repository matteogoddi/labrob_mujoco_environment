#include <hrp4_locomotion/StateFiltering.hpp>
#include <pinocchio/spatial/explog.hpp>
#include <fstream>
#include <iostream>

namespace labrob
{

//////////////////////////
// Joint velocity KF
//////////////////////////

JointKF::JointKF(double dt, int nq)
{
    // STATE, INPUT, OUTPUT MATRICES
    F = Eigen::MatrixXd::Identity(2 * nq, 2 * nq);
    F.block(0, nq, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * dt;
    G = Eigen::MatrixXd::Zero(2 * nq, nq);
    G.block(nq, 0, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * dt;
    H = Eigen::MatrixXd::Zero(nq, 2 * nq);
    H.block(0, 0, nq, nq) = Eigen::MatrixXd::Identity(nq, nq);

    q_filtered_ = Eigen::VectorXd::Zero(2 * nq);
    
    std::cout << "dimensioni di q " << q_filtered_.size() << std::endl;

    // COVARIANCE MATRICES

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(2 * nq, 2 * nq) * 1e-2;
    Q.block(0, 0, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * 1e-4;
    // Q = Eigen::MatrixXd::Identity(2 * nq, 2 * nq);
    // Q.block(0, 0, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * 0.25 * dt^4;
    // Q.block(0, nq, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * 0.5 * dt^3;
    // Q.block(nq, 0, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * 0.5 * dt^3;
    // Q.block(nq, nq, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * dt^2;
    // Q = Q * 1e-6;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nq, nq) * 1e-5;
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(2 * nq, 2 * nq) * 1e-3;

    // KALMAN GAIN 

    K = Eigen::MatrixXd::Zero(2 * nq, nq);
    for (int i = 0; i < 1000; ++i) {
        Eigen::MatrixXd P_pred = F * P * F.transpose() + Q;
        Eigen::MatrixXd S = H * P_pred * H.transpose() + R;
        K = P_pred * H.transpose() * S.inverse();
        P = (Eigen::MatrixXd::Identity(2 * nq, 2 * nq) - K * H) * P_pred;
    }
}

Eigen::VectorXd
JointKF::filter(const Eigen::VectorXd& q_meas, const Eigen::VectorXd& qdd)
{
  q_filtered_ = F * q_filtered_ + G * qdd;
  q_filtered_ = q_filtered_ + K * (q_meas - H * q_filtered_);
  return q_filtered_;
}

//////////////////////////
// IMU calibration (NOT ACTUALLY USING THIS, USE CALIBRATION VIA APP)
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

void BaseEKF::filter(const Eigen::Vector3d& acc_meas,
                          const Eigen::Vector3d& gyro_meas,
                          const Eigen::VectorXd& qj,
                          const Eigen::VectorXd& vj,
                          bool left_contact,
                          bool right_contact)
{
  // =====================
  // 1) NOMINAL PREDICTION
  // =====================
  Eigen::Vector3d acc = acc_meas - bf_;
  Eigen::Vector3d omega = gyro_meas - bw_;

  Eigen::Matrix3d C = q_.toRotationMatrix();

  Eigen::Vector3d a_world = C.transpose() * acc + g_;

  r_ += dt_ * v_ + 0.5 * dt_ * dt_ * a_world;
  v_ += dt_ * a_world;
  q_ = (expMap(dt_ * omega) * q_).normalized();

  // =====================
  // 2) COVARIANCE PREDICTION
  // =====================
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(NX, NX);

  int ir=0, iv=3, iphi=6, ipL=9, ipR=12, ibf=15, ibw=18, ithetaL=21, ithetaR=24;

  F.block<3,3>(ir, iv) = Eigen::Matrix3d::Identity() * dt_;
  F.block<3,3>(iv, iphi) = -C.transpose() * skew(acc) * dt_;
  F.block<3,3>(iv, ibf)  = -C.transpose() * dt_;
  F.block<3,3>(iphi, iphi) = Eigen::Matrix3d::Identity() - skew(omega) * dt_;
  F.block<3,3>(iphi, ibw) = -Eigen::Matrix3d::Identity() * dt_;


  //   Eigen::Matrix<double, NX, NX> Q = Qc_ * dt_;
  P_ = F * P_ * F.transpose() + Q_;

  // =====================
  // 3) KINEMATICS
  // =====================

  Eigen::VectorXd pos = Eigen::VectorXd::Zero(model_.nq);
  pos.head(3) = r_;
  pos.segment(3,4) << q_.w(), q_.x(), q_.y(), q_.z();
  pos.segment(7, qj.size()) = qj;
  Eigen::VectorXd vel = Eigen::VectorXd::Zero(model_.nv);
  vel.head(3) = v_;
  vel.segment(3,3) = Eigen::Vector3d::Zero();
  vel.segment(6, vj.size()) = vj;

  data_ = pinocchio::Data(model_);
  pinocchio::forwardKinematics(model_, data_, pos);
  pinocchio::updateFramePlacements(model_, data_);

  pinocchio::SE3 T_wb = pinocchio::SE3::Identity();
  T_wb.rotation() = q_.toRotationMatrix().transpose();
  T_wb.translation() = r_;

  auto processFoot = [&](int frameId,
                         Eigen::Vector3d& p,
                         Eigen::Quaterniond& z,
                         int ip,
                         int itheta,
                         bool contact,
                         Eigen::MatrixXd& H_accum,
                         Eigen::VectorXd& e_accum)
  {
    if (!contact)
      return;

    const auto& oMf = data_.oMf[frameId];
    pinocchio::SE3 T_bf = T_wb.inverse() * oMf;

    Eigen::Vector3d s_p = T_bf.translation();
    Eigen::Quaterniond s_z(T_bf.rotation());

    Eigen::Vector3d s_p_hat = C * (p - r_);
    Eigen::Quaterniond s_z_hat = q_ * z.inverse();

    Eigen::Vector3d e_p = s_p - s_p_hat;
    Eigen::Vector3d e_z = logMap(s_z * s_z_hat.inverse());

    int old_rows = e_accum.rows();
    e_accum.conservativeResize(old_rows + 6);
    e_accum.segment(old_rows,3) = e_p;
    e_accum.segment(old_rows+3,3) = e_z;

    H_accum.conservativeResize(old_rows + 6, NX);
    H_accum.block(old_rows, 0, 6, NX).setZero();

    H_accum.block<3,3>(old_rows, ir)   = -C;
    H_accum.block<3,3>(old_rows, iphi) = skew(C * (p - r_));
    H_accum.block<3,3>(old_rows, ip)   = C;

    H_accum.block<3,3>(old_rows+3, iphi) = Eigen::Matrix3d::Identity();
    H_accum.block<3,3>(old_rows+3, itheta) =
        - (q_ * z.inverse()).toRotationMatrix();

    std::cout << "ciao" << std::endl;
  };

  Eigen::MatrixXd H(0, NX);
  Eigen::VectorXd e(0);

  processFoot(model_.getFrameId("left_foot_link"),  pL_, zL_, ipL, ithetaL, left_contact,  H, e);
  processFoot(model_.getFrameId("right_foot_link"), pR_, zR_, ipR, ithetaR, right_contact, H, e);

  std::cout << "ciao1" << std::endl;

  if (e.size() == 0)
    return;

  // =====================
  // 4) EKF UPDATE
  // =====================
//   int m = e.size();
//   Eigen::MatrixXd R = Eigen::MatrixXd::Zero(m, m);
//   for (int i = 0; i < m/6; ++i)
//     R.block<6,6>(6*i,6*i) = Rc_6_;

  std::cout << "ciao" << std::endl;

  Eigen::MatrixXd K =
      P_ * H.transpose() * (H * P_ * H.transpose() + R_).inverse();

  Eigen::VectorXd dx = K * e;

  std::cout << "ciao" << std::endl;

  r_  += dx.segment<3>(ir);
  v_  += dx.segment<3>(iv);
  pL_ += dx.segment<3>(ipL);
  pR_ += dx.segment<3>(ipR);
  bf_ += dx.segment<3>(ibf);
  bw_ += dx.segment<3>(ibw);

  q_  = (expMap(dx.segment<3>(iphi)) * q_).normalized();
  zL_ = (expMap(dx.segment<3>(ithetaL)) * zL_).normalized();
  zR_ = (expMap(dx.segment<3>(ithetaR)) * zR_).normalized();

  std::cout << "ciao" << std::endl;

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(NX, NX);
  P_ = (I - K * H) * P_;
}

// TO DO: DILIGENT EKF (LIE GROUP PAPER)

} // namespace state_filtering
