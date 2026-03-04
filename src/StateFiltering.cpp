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
                          bool left_contact,
                          bool right_contact)
{
  // =====================
  // 1) NOMINAL PREDICTION
  // =====================
  Eigen::Vector3d acc = acc_meas - bf_;
  Eigen::Vector3d omega = gyro_meas - bw_;

  Eigen::Matrix3d C = q_.toRotationMatrix().transpose();

  Eigen::Vector3d a_world = C.transpose() * acc;
  // a_world.setZero();
  std::cout << "acc " << a_world.transpose() << std::endl;

  r_ += dt_ * v_ + 0.5 * dt_ * dt_ * a_world;
  v_ += dt_ * a_world;
  q_ = (expMap(dt_ * omega) * q_).normalized();
  std::cout << "Predicted position: " << r_.transpose() << std::endl;
  std::cout << "Predicted velocity: " << v_.transpose() << std::endl;
  std::cout << "Predicted orientation: " << q_.coeffs().transpose() << std::endl;


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

  Eigen::MatrixXd Lc = Eigen::MatrixXd::Identity(NX, NX);
  Lc.block<3,3>(ir, ir) = -C.transpose();
  Lc.block<3,3>(iphi, iphi) = C.transpose();


  Qc_.block<3,3>(ipL, ipL) = 1e-6 * Eigen::Matrix3d::Identity();
  Qc_.block<3,3>(ipL, ipL) = 1e-6 * Eigen::Matrix3d::Identity();
  if (!left_contact){
    Qc_.block<3,3>(ipL, ipL) = 1 * Eigen::Matrix3d::Identity();
  } else if (!right_contact){
    Qc_.block<3,3>(ipR,ipR) = 1 * Eigen::Matrix3d::Identity();
  }

  Eigen::MatrixXd Q_ = F * Lc * Qc_ * Lc.transpose() * F.transpose() * dt_;
  // Q_ = Qc_;
  
  P_ = F * P_ * F.transpose() + Q_;

  // =====================
  // 3) KINEMATICS
  // =====================

  Eigen::VectorXd pos = Eigen::VectorXd::Zero(model_.nq);
  // pos.head(3) << r_;
  // pos.segment(3, 4) << q_.coeffs();
  pos[6] = 1;
  pos.tail(qj.size()) = qj;

  data_ = pinocchio::Data(model_);
  pinocchio::forwardKinematics(model_, data_, pos);
  pinocchio::framesForwardKinematics(model_, data_, pos);
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
    pinocchio::SE3 T_bf = oMf;

    Eigen::Vector3d s_p = T_bf.translation();
    Eigen::Quaterniond s_z(T_bf.rotation());

    Eigen::Vector3d s_p_hat = C * (p - r_);
    Eigen::Quaterniond s_z_hat = q_ * z.inverse();

    Eigen::VectorXd e_p = Eigen::VectorXd::Zero(4);
    e_p.head<3>() = s_p - s_p_hat;
    e_p(3) = -p.z();
    Eigen::Vector3d e_z = Eigen::VectorXd::Zero(3);
    e_z.head<3>() = logMap(s_z * s_z_hat.inverse());
    // e_z.tail<2>() = -logMap(z).head<2>();

    int old_rows = e_accum.rows();
    e_accum.conservativeResize(old_rows + e_p.size() + e_z.size());
    e_accum.segment(old_rows, e_p.size()) = e_p;
    e_accum.segment(old_rows + e_p.size(), e_z.size()) = e_z;

    H_accum.conservativeResize(old_rows + e_p.size() + e_z.size(), NX);
    H_accum.block(old_rows, 0, + e_p.size() + e_z.size(), NX).setZero();

    H_accum.block<3,3>(old_rows, ir)   = -C;
    H_accum.block<3,3>(old_rows, iphi) = skew(C * (p - r_));
    H_accum.block<3,3>(old_rows, ip)   = C;
    H_accum(old_rows + 3, ip + 2) = 1;

    H_accum.block<3,3>(old_rows+4, iphi) = Eigen::Matrix3d::Identity();
    H_accum.block<3,3>(old_rows+4, itheta) =
        - (q_ * z.inverse()).toRotationMatrix();
    // H_accum.block<2,2>(old_rows + 7, itheta) = Eigen::Matrix2d::Identity();
  };

  Eigen::MatrixXd H(0, NX);
  Eigen::VectorXd e(0);

  processFoot(model_.getFrameId("left_foot_link"),  pL_, zL_, ipL, ithetaL, left_contact,  H, e);
  processFoot(model_.getFrameId("right_foot_link"), pR_, zR_, ipR, ithetaR, right_contact, H, e);

  std::cout << "Measurement error: " << e.transpose() << std::endl; 

  if (e.size() == 0)
    return;

  // =====================
  // 4) EKF UPDATE
  // =====================
  int m = e.size();
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m, m) * 1e-5;
//   for (int i = 0; i < m/6; ++i)
//     R.block<6,6>(6*i,6*i) = Rc_6_;

  Eigen::MatrixXd K =
      P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();

  Eigen::VectorXd dx = K * e;

  r_  += dx.segment<3>(ir);
  v_  += dx.segment<3>(iv);
  pL_ += dx.segment<3>(ipL);
  pR_ += dx.segment<3>(ipR);
  bf_ += dx.segment<3>(ibf);
  bw_ += dx.segment<3>(ibw);
  std::cout << "bias f" << bf_ << " bias w" << bw_ << std::endl;

  q_  = (expMap(dx.segment<3>(iphi)) * q_).normalized();
  zL_ = (expMap(dx.segment<3>(ithetaL)) * zL_).normalized();
  zR_ = (expMap(dx.segment<3>(ithetaR)) * zR_).normalized();

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(NX, NX);
  P_ = (I - K * H) * P_;
}

//////////////////////////
// COM KALMAN FILTER
//////////////////////////
LIPState CoMKF::filter(LIPState filtered, LIPState current, const Eigen::Vector3d &input) {

  // DYNAMIC AND INPUT MATRICES

  double ch = cosh(eta_ * dt_);
  double sh = sinh(eta_ * dt_);
  Eigen::MatrixXd A_lip = Eigen::MatrixXd::Zero(3,3);
  Eigen::VectorXd B_lip = Eigen::VectorXd::Zero(3);
  A_lip << ch,sh/eta_,1-ch,eta_*sh,ch,-eta_*sh,0,0,1;
  B_lip << dt_-sh/eta_,1-ch,dt_;

  Eigen::Vector3d x_measure, y_measure, z_measure;
  if (std::isnan(current.zmp_pos_(0))) {
    std::cout << "NaN ZMP measurement detected, using filtered value instead." << std::endl;
    x_measure = Eigen::Vector3d(current.com_pos_(0), filtered.com_vel_(0), filtered.zmp_pos_(0));
    y_measure = Eigen::Vector3d(current.com_pos_(1), filtered.com_vel_(1), filtered.zmp_pos_(1));
    z_measure = Eigen::Vector3d(current.com_pos_(2), filtered.com_vel_(2), filtered.zmp_pos_(2));
  } else {
    x_measure = Eigen::Vector3d(current.com_pos_(0), current.com_vel_(0), current.zmp_pos_(0));
    y_measure = Eigen::Vector3d(current.com_pos_(1), current.com_vel_(1), current.zmp_pos_(1));
    z_measure = Eigen::Vector3d(current.com_pos_(2), current.com_vel_(2), current.zmp_pos_(2));
  }
  Eigen::Vector3d x_est = Eigen::Vector3d(filtered.com_pos_(0), filtered.com_vel_(0), filtered.zmp_pos_(0));
  Eigen::Vector3d y_est = Eigen::Vector3d(filtered.com_pos_(1), filtered.com_vel_(1), filtered.zmp_pos_(1));
  Eigen::Vector3d z_est = Eigen::Vector3d(filtered.com_pos_(2), filtered.com_vel_(2), filtered.zmp_pos_(2));

  Eigen::MatrixXd F_kf = A_lip;
  Eigen::MatrixXd G_kf = B_lip;
  Eigen::MatrixXd H_kf = Eigen::Matrix3d::Identity();

  Eigen::MatrixXd R_kf = Eigen::MatrixXd::Identity(3,3);
  R_kf.diagonal() << cov_meas_pos, cov_meas_vel, cov_meas_zmp;
  Eigen::MatrixXd Q_kf = Eigen::MatrixXd::Identity(3,3);
  Q_kf.diagonal() << cov_mod_pos, cov_mod_vel, cov_mod_zmp;

  double input_x = input.x();
  double input_y = input.y();
  double input_z = input.z();

  // X_PRED BY DYNAMICS AND UPDATE COVARIANCE

  Eigen::VectorXd x_pred = F_kf * x_est + G_kf * input_x;
  Eigen::MatrixXd cov_x_pred = F_kf * cov_x * F_kf.transpose() + Q_kf;

  // KALMAN GAIN

  Eigen::MatrixXd K_kf = cov_x_pred * H_kf.transpose() * (H_kf * cov_x_pred * H_kf.transpose() + R_kf).inverse();

  // STATE ESTIMATE (SPLIT INTO X,Y AND Z)

  x_est = x_pred + K_kf * (x_measure - H_kf * x_pred);
  cov_x = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_x_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();

  Eigen::VectorXd y_pred = F_kf * y_est + G_kf * input_y;
  Eigen::MatrixXd cov_y_pred = F_kf * cov_y * F_kf.transpose() + Q_kf;

  K_kf = cov_y_pred * H_kf.transpose() * (H_kf * cov_y_pred * H_kf.transpose() + R_kf).inverse();

  y_est = y_pred + K_kf * (y_measure - H_kf * y_pred);
  cov_y = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_y_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();

  Eigen::VectorXd z_pred = F_kf * z_est + G_kf * input_z + Eigen::Vector3d(0.0, -9.81 * dt_, 0.0);
  Eigen::MatrixXd cov_z_pred = F_kf * cov_z * F_kf.transpose() + Q_kf;

  K_kf = cov_z_pred * H_kf.transpose() * (H_kf * cov_z_pred * H_kf.transpose() + R_kf).inverse();

  z_est = z_pred + K_kf * (z_measure - H_kf * z_pred);
  cov_z = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_z_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();

  // FILL CURRENT STATE FOR RETURN

  current.com_pos_ = Eigen::Vector3d(x_est(0), y_est(0), z_est(0));
  current.com_vel_ = Eigen::Vector3d(x_est(1), y_est(1), z_est(1));
  current.zmp_pos_ = Eigen::Vector3d(x_est(2), y_est(2), z_est(2));

  return current;
}

// TO DO: DILIGENT EKF (LIE GROUP PAPER)

} // namespace state_filtering
