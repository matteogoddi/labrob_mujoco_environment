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
                          const Eigen::VectorXd& qj_dot,
                          const Eigen::VectorXd& q_ddot,
                          bool left_contact,
                          bool right_contact)
{
  // =====================
  // 1) NOMINAL PREDICTION
  // =====================

  Eigen::Matrix3d C = q_.toRotationMatrix().transpose();

  Eigen::Vector3d acc = R_base_imu * (acc_meas - bf_);
  omega_ = C.transpose() * R_base_imu * (gyro_meas - bw_);

  Eigen::Vector3d a_world = C.transpose() * acc;
  // a_world.setZero();
  // omega_.setZero();
  std::cout << "acc meas " << acc_meas.transpose() << std::endl;
  std::cout << "acc " << acc.transpose() << std::endl;
  std::cout << "bf " << bf_.transpose() << std::endl;
  std::cout << "omega " << omega_.transpose() << std::endl;


  r_ += dt_ * v_ + 0.5 * dt_ * dt_ * a_world;
  v_ += dt_ * a_world;
  q_ = (q_ * expMap(dt_ * omega_)).normalized();
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
  F.block<3,3>(iphi, iphi) = Eigen::Matrix3d::Identity() - skew(C * omega_) * dt_;
  F.block<3,3>(iphi, ibw) = -Eigen::Matrix3d::Identity() * dt_;

  Eigen::MatrixXd Lc = Eigen::MatrixXd::Identity(NX, NX);
  Lc.block<3,3>(ir, ir) = -C.transpose();
  Lc.block<3,3>(iphi, iphi) = C.transpose();


  Qc_.block<3,3>(ipL, ipL) = 1e-5 * Eigen::Matrix3d::Identity();
  Qc_.block<3,3>(ipR, ipR) = 1e-5 * Eigen::Matrix3d::Identity();
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
  pos.head(3) << r_;
  pos.segment(3, 4) << q_.coeffs();
  // pos[6] = 1;
  pos.tail(qj.size()) = qj;

  Eigen::VectorXd vel = Eigen::VectorXd::Zero(model_.nv);
  vel.head(3) << v_;
  vel.segment(3, 3) << omega_;
  vel.tail(qj_dot.size()) = qj_dot;

  data_ = pinocchio::Data(model_);
  pinocchio::forwardKinematics(model_, data_, pos, vel);
  pinocchio::framesForwardKinematics(model_, data_, pos);
  pinocchio::updateFramePlacements(model_, data_);
  // pinocchio::jacobianCenterOfMass(model_, data_, pos);
  pinocchio::computeJointJacobians(model_, data_, pos);

  pinocchio::SE3 T_wb = pinocchio::SE3::Identity();
  T_wb.rotation() = q_.toRotationMatrix();
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

    const auto& T_wf = data_.oMf[frameId];
    pinocchio::SE3 T_bf = pinocchio::SE3::Identity();
    T_bf.rotation() = T_wb.rotation().transpose() * T_wf.rotation();
    T_bf.translation() = T_wb.rotation().transpose() * (T_wf.translation() - T_wb.translation());

    Eigen::Vector3d s_p = T_bf.translation();
    Eigen::Quaterniond s_z(T_bf.rotation());

    Eigen::Vector3d s_p_hat = C * (p - r_);
    Eigen::Quaterniond s_z_hat = q_ * z.inverse();

    Eigen::MatrixXd J_foot = Eigen::MatrixXd::Zero(6, model_.nv);
    pinocchio::getFrameJacobian(
      model_, 
      data_, 
      frameId, 
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
      J_foot
    );

    Eigen::VectorXd e_p = Eigen::VectorXd::Zero(4);
    e_p.head<3>() = s_p - s_p_hat;
    e_p(3) = -p.z();
    Eigen::Vector3d e_z = Eigen::VectorXd::Zero(3);
    e_z.head<3>() = logMap(s_z * s_z_hat.inverse());
    // e_z.tail<2>() = -logMap(z).head<2>();
    Eigen::VectorXd e_foot = Eigen::VectorXd::Zero(6);
    e_foot = - J_foot * vel;


    int old_rows = e_accum.rows();
    e_accum.conservativeResize(old_rows + e_p.size() + e_z.size() + e_foot.size());
    e_accum.segment(old_rows, e_p.size()) = e_p;
    e_accum.segment(old_rows + e_p.size(), e_z.size()) = e_z;
    e_accum.segment(old_rows + e_p.size() + e_z.size(), e_foot.size()) = e_foot;

    H_accum.conservativeResize(old_rows + e_p.size() + e_z.size() + e_foot.size(), NX);
    H_accum.block(old_rows, 0, e_p.size() + e_z.size() + e_foot.size(), NX).setZero();

    H_accum.block<3,3>(old_rows, ir)   = -C;
    H_accum.block<3,3>(old_rows, iphi) = skew(C * (p - r_));
    H_accum.block<3,3>(old_rows, ip)   = C;
    H_accum(old_rows + 3, ip + 2) = 1;

    H_accum.block<3,3>(old_rows+e_p.size(), iphi) = Eigen::Matrix3d::Identity();
    H_accum.block<3,3>(old_rows+e_p.size(), itheta) = - C * (q_ * z.inverse()).toRotationMatrix();
    // H_accum.block<2,2>(old_rows + 7, itheta) = Eigen::Matrix2d::Identity();

    H_accum.block<6,3>(old_rows + e_p.size() + e_z.size(), iv) = J_foot.block<6,3>(0,0);
    // H_accum.block<3,3>(old_rows + e_p.size() + e_z.size(), iphi) =
    // J_foot.block<3,3>(0,3);
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
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m, m) * 1e-3;
  R(3,3) = 1e5; // high noise for foot
  R(3 + e.size()/2, 3 + e.size()/2) = 1e5; //high noise for foot

  Eigen::MatrixXd K =
      P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();

  Eigen::VectorXd dx = K * e;

  r_  += dx.segment<3>(ir);
  v_  += dx.segment<3>(iv);
  pL_ += dx.segment<3>(ipL);
  pR_ += dx.segment<3>(ipR);
  bf_ += dx.segment<3>(ibf);
  bw_ += dx.segment<3>(ibw);

  q_  = (expMap(dx.segment<3>(iphi)) * q_).normalized();
  zL_ = (expMap(dx.segment<3>(ithetaL)) * zL_).normalized();
  zR_ = (expMap(dx.segment<3>(ithetaR)) * zR_).normalized();

  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(NX, NX);
  P_ = (I - K * H) * P_;
}

// void BaseEKF::filter(
//     const Eigen::Vector3d& acc_meas,
//     const Eigen::Vector3d& gyro_meas,
//     const Eigen::VectorXd& qj,
//     const Eigen::VectorXd& qj_dot,
//     bool left_contact,
//     bool right_contact)
// {

// //================================================
// // 1) IMU PREPROCESS
// //================================================


// int ir=0, iv=3, iphi=6, ipL=9, ipR=12, ithetaL=15, ithetaR=18, iomega=21;

// Eigen::Vector3d acc =
//     R_base_imu * acc_meas;

// Eigen::Vector3d omega_meas =
//     R_base_imu * gyro_meas;

// // base -> world
// Eigen::Matrix3d R = q_.toRotationMatrix();

// //================================================
// // 2) NOMINAL PREDICTION
// //================================================

// Eigen::Vector3d a_world =
//     R * acc + g_;

// r_ += dt_ * v_ + 0.5 * dt_ * dt_ * a_world;

// v_ += dt_ * a_world;

// q_ =
// (expMap(dt_ * omega_) * q_).normalized();

// //================================================
// // 3) COVARIANCE PREDICTION
// //================================================

// Eigen::MatrixXd F =
// Eigen::MatrixXd::Identity(NX,NX);

// F.block<3,3>(ir,iv) =
// Eigen::Matrix3d::Identity()*dt_;

// F.block<3,3>(iv,iphi) =
// - R * skew(acc) * dt_;

// F.block<3,3>(iphi,iphi) =
// Eigen::Matrix3d::Identity()
// - skew(omega_)*dt_;

// F.block<3,3>(iphi,iomega) =
// Eigen::Matrix3d::Identity()*dt_;

// // noise injection

// Eigen::MatrixXd L =
// Eigen::MatrixXd::Identity(NX,NX);

// // continuous noise

// Eigen::MatrixXd Qc =
// Eigen::MatrixXd::Zero(NX,NX);

// double sigma_acc = 0.5;
// double sigma_gyro = 0.1;

// Qc.block<3,3>(iv,iv) =
// sigma_acc*Eigen::Matrix3d::Identity();

// Qc.block<3,3>(iomega,iomega) =
// sigma_gyro*Eigen::Matrix3d::Identity();

// // swing feet

// double stance = 1e-6;
// double swing = 1e-1;

// Qc.block<3,3>(ipL,ipL) =
// (left_contact?stance:swing) *
// Eigen::Matrix3d::Identity();

// Qc.block<3,3>(ipR,ipR) =
// (right_contact?stance:swing) *
// Eigen::Matrix3d::Identity();

// // discretization

// Eigen::MatrixXd Q =
// L*Qc*L.transpose()*dt_;

// P_ =
// F*P_*F.transpose() + Q;

// //================================================
// // 4) PINOCCHIO KINEMATICS
// //================================================

// Eigen::VectorXd q =
// Eigen::VectorXd::Zero(model_.nq);

// q.head<3>() = r_;

// q.segment<4>(3) <<
// q_.w(), q_.x(), q_.y(), q_.z();

// q.tail(qj.size()) = qj;

// Eigen::VectorXd v =
// Eigen::VectorXd::Zero(model_.nv);

// v.head<3>() = v_;

// v.segment<3>(3) = omega_;

// v.tail(qj_dot.size()) = qj_dot;

// pinocchio::forwardKinematics(
// model_,data_,q,v);

// pinocchio::updateFramePlacements(
// model_,data_);

// //================================================
// // 5) FOOT UPDATE
// //================================================

// Eigen::MatrixXd H(0,NX);
// Eigen::VectorXd e(0);

// auto processFoot =
// [&](int frameId,
// Eigen::Vector3d& p,
// Eigen::Quaterniond& z,
// int ip,
// int itheta,
// bool contact)
// {

// if(!contact)
// return;

// const auto& T_wf =
// data_.oMf[frameId];

// // predicted foot in base

// Eigen::Vector3d s_hat =
// R.transpose()*(p-r_);

// Eigen::Quaterniond q_hat =
// q_*z.inverse();

// // measured

// Eigen::Vector3d s =
// R.transpose()*
// (T_wf.translation()-r_);

// Eigen::Quaterniond q_meas(
// R.transpose()*T_wf.rotation());

// // residual

// Eigen::Vector3d ep =
// s - s_hat;

// Eigen::Vector3d ez =
// logMap(q_meas*q_hat.inverse());

// // foot velocity

// Eigen::MatrixXd J(6,model_.nv);

// pinocchio::getFrameJacobian(
// model_,
// data_,
// frameId,
// pinocchio::LOCAL_WORLD_ALIGNED,
// J);

// Eigen::Vector3d vfoot =
// J.topRows(3)*v;

// Eigen::Vector3d ev =
// - vfoot;

// // stack residual

// int old = e.rows();

// e.conservativeResize(
// old+9);

// e.segment<3>(old) = ep;

// e.segment<3>(old+3) = ez;

// e.segment<3>(old+6) = ev;

// // Jacobian

// H.conservativeResize(
// old+9,NX);

// H.block(old,0,9,NX).setZero();

// H.block<3,3>(old,ir) =
// - R.transpose();

// H.block<3,3>(old,iphi) =
// skew(s_hat);

// H.block<3,3>(old,ip) =
// R.transpose();

// H.block<3,3>(old+3,iphi) =
// Eigen::Matrix3d::Identity();

// H.block<3,3>(old+3,itheta) =
// -(q_*z.inverse()).toRotationMatrix();

// H.block<3,3>(old+6,iv) =
// J.block<3,3>(0,0);

// H.block<3,3>(old+6,iomega) =
// J.block<3,3>(0,3);

// };

// processFoot(
// model_.getFrameId("left_foot_link"),
// pL_,zL_,ipL,ithetaL,left_contact);

// processFoot(
// model_.getFrameId("right_foot_link"),
// pR_,zR_,ipR,ithetaR,right_contact);

// //================================================
// // 6) GYRO UPDATE
// //================================================

// Eigen::Vector3d eomega =
// omega_meas - omega_;

// int old = e.rows();

// e.conservativeResize(old+3);

// e.segment<3>(old) = eomega;

// H.conservativeResize(old+3,NX);

// H.block(old,0,3,NX).setZero();

// H.block<3,3>(old,iomega) =
// Eigen::Matrix3d::Identity();

// //================================================
// // 7) EKF UPDATE
// //================================================

// if(e.size()==0)
// return;

// Eigen::MatrixXd Rm =
// Eigen::MatrixXd::Identity(
// e.size(),e.size())*1e-4;

// Eigen::MatrixXd K =
// P_*H.transpose()*
// (H*P_*H.transpose()+Rm).inverse();

// Eigen::VectorXd dx =
// K*e;

// // state update

// r_ += dx.segment<3>(ir);

// v_ += dx.segment<3>(iv);

// pL_ += dx.segment<3>(ipL);

// pR_ += dx.segment<3>(ipR);

// omega_ += dx.segment<3>(iomega);

// q_ =
// (expMap(dx.segment<3>(iphi))*q_)
// .normalized();

// zL_ =
// (expMap(dx.segment<3>(ithetaL))*zL_)
// .normalized();

// zR_ =
// (expMap(dx.segment<3>(ithetaR))*zR_)
// .normalized();

// // covariance

// Eigen::MatrixXd I =
// Eigen::MatrixXd::Identity(NX,NX);

// P_ =
// (I-K*H)*P_;

// }

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
