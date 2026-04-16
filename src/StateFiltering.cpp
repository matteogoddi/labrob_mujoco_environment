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
    const int nx = 2 * nq + 3;
    const int nu = nq + 3;
    F = Eigen::MatrixXd::Identity(nx, nx);
    F.block(0, nq, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * dt;
    G = Eigen::MatrixXd::Zero(nx, nu);
    G.block(nq, 0, nq, nq) = Eigen::MatrixXd::Identity(nq, nq) * dt;
    G.block(2*nq, nq, 3, 3) = Eigen::Matrix3d::Identity() * dt;
    H = Eigen::MatrixXd::Zero(nu, nx);
    H.block(0, 0, nq, nq) = Eigen::MatrixXd::Identity(nq, nq);
    H.block(nq, 2 * nq, 3, 3) = Eigen::Matrix3d::Identity();

    q_filtered_ = Eigen::VectorXd::Zero(nx);
    JointPos_ = Eigen::VectorXd::Zero(nq);
    JointVel_ = Eigen::VectorXd::Zero(nq);
    Omega_ = Eigen::VectorXd::Zero(3);

    // COVARIANCE MATRICES

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nx, nx) * 1e-3;
    Q.block(0, 0, nx, nx) = Eigen::MatrixXd::Identity(nx, nx) * 1e-4;
    Q.block(nx-3, nx-3, 3, 3) = Eigen::Matrix3d::Identity() * 1e-4;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nu, nu) * 1e-5;
    R.block(nu-3, nu-3, 3, 3) = Eigen::Matrix3d::Identity() * 1e-1;
    Eigen::MatrixXd P = Eigen::MatrixXd::Identity(nx, nx) * 1e-5;

    // KALMAN GAIN 

    K = Eigen::MatrixXd::Zero(nx, nu);
    for (int i = 0; i < 1000; ++i) {
        Eigen::MatrixXd P_pred = F * P * F.transpose() + Q;
        Eigen::MatrixXd S = H * P_pred * H.transpose() + R;
        K = P_pred * H.transpose() * S.inverse();
        P = (Eigen::MatrixXd::Identity(nx, nx) - K * H) * P_pred;
    }
}

void
JointKF::filter(const Eigen::VectorXd& q_meas, const Eigen::VectorXd& qdd)
{
    q_filtered_ = F * q_filtered_ + G * qdd;
    q_filtered_ = q_filtered_ + K * (q_meas - H * q_filtered_);
    JointPos_ = q_filtered_.head(29);
    JointVel_ = q_filtered_.segment(29, 29);
    Omega_ = q_filtered_.tail(3);
    std::cout << "ciao" << std::endl;
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

  Eigen::Matrix3d C_world_to_base = q_.toRotationMatrix();
  // C_world_to_base = (Eigen::Matrix3d::Identity() + 2*q_.w()*skew(q_.vec()) + 2 * skew(q_.vec()) * skew(q_.vec())).transpose();


  Eigen::VectorXd pos = Eigen::VectorXd::Zero(model_.nq);
  pos.head(3) << r_;
  pos.segment(3, 4) << q_.x(), q_.y(), q_.z(), q_.w();
  pos.tail(qj.size()) = qj;

  pinocchio::forwardKinematics(model_, data_, pos);
  pinocchio::framesForwardKinematics(model_, data_, pos);
  pinocchio::computeJointJacobians(model_, data_, pos);

  Eigen::VectorXd vel = Eigen::VectorXd::Zero(model_.nv);
  vel.head(3) << C_world_to_base * v_;
  vel.segment(3, 3) << omega_;
  vel.tail(qj_dot.size()) = qj_dot;

  Eigen::Vector3d acc = R_base_imu * acc_meas - bf_;

  omega_ = R_base_imu * gyro_meas - bw_;
  omega_world = C_world_to_base.transpose() * omega_;

  Eigen::Vector3d a_world = C_world_to_base.transpose() * acc + g_;

  Eigen::Vector3d r_prec = r_;
  Eigen::Vector3d v_prec = v_;
  Eigen::Quaterniond q_prec = q_;
  r_ += dt_ * v_ + 0.5 * dt_ * dt_ * a_world;
  v_ += dt_ * a_world;
  q_ = (expMap(dt_ * omega_) * q_).normalized();


  // =====================
  // 2) COVARIANCE PREDICTION
  // =====================
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(NX, NX);

  int ir=0, iv=3, iphi=6, ipL=9, ipR=12, ithetaL=15, ithetaR=18, ibf=21, ibw=24;

  F.block<3,3>(ir, iv) = Eigen::Matrix3d::Identity() * dt_;
  F.block<3,3>(iv, iphi) = -C_world_to_base.transpose() * skew(acc) * dt_;
  F.block<3,3>(iv, ibf)  = -C_world_to_base.transpose() * dt_;
  F.block<3,3>(iphi, iphi) = Eigen::Matrix3d::Identity() - skew(omega_) * dt_;
  F.block<3,3>(iphi, ibw) = -Eigen::Matrix3d::Identity() * dt_;

  Eigen::MatrixXd Lc = Eigen::MatrixXd::Zero(NX, NX - 3); // no noise for base position dynamics
  Lc.block<3,3>(iv, 0) = -C_world_to_base.transpose();
  Lc.block<3,3>(iphi, 3) = -Eigen::Matrix3d::Identity();
  Lc.block<3,3>(ipL, 6) = C_world_to_base.transpose();
  Lc.block<3,3>(ipR, 9) = C_world_to_base.transpose();
  Lc.block<3,3>(ithetaL, 12) = Eigen::Matrix3d::Identity();
  Lc.block<3,3>(ithetaR, 15) = Eigen::Matrix3d::Identity();
  Lc.block<3,3>(ibf, 18) = Eigen::Matrix3d::Identity();
  Lc.block<3,3>(ibw, 21) = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double,24,24> Qc_step = Qc_;
  if (!left_contact)  Qc_step.block<3,3>(6,6)  = 100.0 * Eigen::Matrix3d::Identity();
  if (!right_contact) Qc_step.block<3,3>(9,9)  = 100.0 * Eigen::Matrix3d::Identity();
  if (!left_contact)  Qc_step.block<3,3>(12,12)  = 100.0 * Eigen::Matrix3d::Identity();
  if (!right_contact) Qc_step.block<3,3>(15,15)  = 100.0 * Eigen::Matrix3d::Identity();
  Eigen::MatrixXd Q_ = F * Lc * Qc_step * Lc.transpose() * F.transpose() * dt_;
  
  P_ = F * P_ * F.transpose() + Q_;

  // =====================
  // 3) KINEMATICS
  // =====================

  pinocchio::SE3 T_wb = data_.oMi[1];

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
    pinocchio::SE3 T_bf = T_wb.inverse() * T_wf;

    Eigen::Vector3d s_p = T_bf.translation();
    Eigen::Vector3d s_p_hat = C_world_to_base * (p - r_prec);

    Eigen::Quaterniond s_z(T_bf.rotation());
    Eigen::Quaterniond s_z_hat = q_prec * z.inverse();



    Eigen::MatrixXd J_foot = Eigen::MatrixXd::Zero(6, model_.nv);
    pinocchio::getFrameJacobian(
      model_,
      data_,
      frameId,
      pinocchio::ReferenceFrame::LOCAL,
      J_foot
    );

    Eigen::Vector3d e_p = s_p - s_p_hat;
    Eigen::Vector3d e_z = logMap(s_z * s_z_hat.inverse());
    Eigen::Vector3d e_foot = -(v_prec + C_world_to_base.transpose() * skew(omega_) * s_p + C_world_to_base.transpose() * J_foot.block(0, 6, 3, model_.nv - 6) * vel.tail(model_.nv - 6));
    // e_foot = - J_foot.block(0, 0, 3, model_.nv) * vel;

    int old_rows = e_accum.rows();
    e_accum.conservativeResize(old_rows + e_p.size() + e_z.size());
    e_accum.segment(old_rows, e_p.size()) = e_p;
    e_accum.segment(old_rows + e_p.size(), e_z.size()) = e_z;
    // e_accum.segment(old_rows + e_p.size(), e_foot.size()) = e_foot;

    H_accum.conservativeResize(old_rows + e_p.size() + e_z.size(), NX);
    H_accum.block(old_rows, 0, e_p.size() + e_z.size(), NX).setZero();

    H_accum.block<3,3>(old_rows, ir)   = -C_world_to_base;
    H_accum.block<3,3>(old_rows, iphi) = skew(C_world_to_base * (p - r_prec));
    H_accum.block<3,3>(old_rows, ip)   = C_world_to_base;

    H_accum.block<3,3>(old_rows+3, iphi) = Eigen::Matrix3d::Identity();
    H_accum.block<3,3>(old_rows+3, itheta) =
        - (q_ * z.inverse()).toRotationMatrix();

    // H_accum.block<3,3>(old_rows + e_p.size(), iv) = Eigen::Matrix3d::Identity();
    // H_accum.block<3,3>(old_rows + e_p.size(), iphi) = skew(C_world_to_base.transpose() * skew(omega_) * s_p) 
    //     + skew(C_world_to_base.transpose() * J_foot.block(0, 6, 3, model_.nv - 6) * vel.tail(model_.nv - 6));
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
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m, m) * 1e-4;
//   R(3,3) = 1e-8; // high noise for foot
//   R(3 + e.size()/2, 3 + e.size()/2) = 1e-8; //high noise for foot
  // R.block<3,3>(0,0) = Eigen::MatrixXd::Identity(3, 3) * 1;
  // R.block<3,3>(0+e.size()/2,0+e.size()/2) = Eigen::MatrixXd::Identity(3, 3) * 1;

  Eigen::MatrixXd K = P_ * H.transpose() * (H * P_ * H.transpose() + R).inverse();

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
  auto IKH = I - K * H;
  P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
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

// ============================================================================
//  Static math helpers
// ============================================================================

Eigen::Matrix3d RightInvariantEKF::skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<    0, -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;
    return S;
}

Eigen::Matrix3d RightInvariantEKF::expSO3(const Eigen::Vector3d& phi)
{
    const double th = phi.norm();
    if (th < 1e-9)
        return Eigen::Matrix3d::Identity() + skew(phi);
    const Eigen::Matrix3d K = skew(phi / th);
    return Eigen::Matrix3d::Identity()
         + std::sin(th)       * K
         + (1.0-std::cos(th)) * K * K;
}

Eigen::Vector3d RightInvariantEKF::logSO3(const Eigen::Matrix3d& R)
{
    const double cos_th = std::clamp(0.5*(R.trace()-1.0), -1.0, 1.0);
    const double th     = std::acos(cos_th);
    if (th < 1e-9) return Eigen::Vector3d::Zero();
    const double s = th / (2.0*std::sin(th));
    return s * Eigen::Vector3d(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));
}

Eigen::Matrix3d RightInvariantEKF::leftJacobianSO3(const Eigen::Vector3d& phi)
{
    // J_l(φ) = I + ((1−cosθ)/θ²) [φ]× + ((θ−sinθ)/θ³) [φ]×²
    const double th = phi.norm();
    if (th < 1e-7)
        return Eigen::Matrix3d::Identity() + 0.5*skew(phi);
    const Eigen::Matrix3d K = skew(phi);
    return Eigen::Matrix3d::Identity()
         + ((1.0-std::cos(th))/(th*th)) * K
         + ((th-std::sin(th)) /(th*th*th)) * K*K;
}

Eigen::Matrix3d RightInvariantEKF::projectToSO3(const Eigen::Matrix3d& M)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Vector3d sv = svd.singularValues();
    // Force det = +1
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    S(2,2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
    return svd.matrixU() * S * svd.matrixV().transpose();
}

// ============================================================================
//  SE_{N+2}(3) group operations
// ============================================================================

// groupExp:  xi (15-dim Lie algebra vector) → state matrix (7×7)
//
// xi = [ξᴿ(0:3) | ξᵛ(3:6) | ξᵖ(6:9) | ξᵈ⁰(9:12) | ξᵈ¹(12:15)]
//
// The matrix exponential of Lg(xi) acts as:
//   exp(Lg(xi)).block<3,3>(0,0) = expSO3(ξᴿ)          (rotation block)
//   exp(Lg(xi)).block<3,1>(0,k) = J_l(ξᴿ) * ξ_col_k  (each vector column)
//
// This is the standard SE(3) result generalised to all N+2 extra columns.
Eigen::Matrix<double, RightInvariantEKF::DIM_X,
                       RightInvariantEKF::DIM_X>
RightInvariantEKF::groupExp(
    const Eigen::Matrix<double,3*(N_FEET+3),1>& xi) const
{
    const Eigen::Vector3d phi = xi.template head<3>();
    const Eigen::Matrix3d dR  = expSO3(phi);
    const Eigen::Matrix3d Jl  = leftJacobianSO3(phi);

    // Initialise to identity (sets up bottom-right scalar identity block)
    Eigen::Matrix<double,DIM_X,DIM_X> E =
        Eigen::Matrix<double,DIM_X,DIM_X>::Identity();

    // Rotation block (top-left 3×3)
    E.template block<3,3>(0,0) = dR;

    // Vector columns: col k  ←  Jl * xi_segment_for_col_k
    // Column mapping:
    //   col COL_V=3 ← xi[3:6]  (ξᵛ)
    //   col COL_P=4 ← xi[6:9]  (ξᵖ)
    //   col COL_D+i ← xi[9+3i : 12+3i]  (ξᵈⁱ)
    //
    // In general: xi segment for column c (c = 3..DIM_X-1) starts at 3*(c-2).
    //   c=3: xi[3:6]   c=4: xi[6:9]   c=5: xi[9:12]   c=6: xi[12:15]
    for (int c = COL_V; c < DIM_X; ++c) {
        const int xi_off = 3*(c - 2);   // 3*(3-2)=3, 3*(4-2)=6, 3*(5-2)=9, ...
        E.template block<3,1>(0,c) = Jl * xi.template segment<3>(xi_off);
    }

    return E;
}

// groupInverse:  X⁻¹ for X ∈ SE_{N+2}(3)
//
// For X = [R  cols ; 0  I]:
//   X⁻¹ = [R^T  -R^T·col₃  -R^T·col₄  … ; 0  I]
Eigen::Matrix<double, RightInvariantEKF::DIM_X,
                       RightInvariantEKF::DIM_X>
RightInvariantEKF::groupInverse(
    const Eigen::Matrix<double,DIM_X,DIM_X>& X) const
{
    Eigen::Matrix<double,DIM_X,DIM_X> Xinv =
        Eigen::Matrix<double,DIM_X,DIM_X>::Identity();

    const Eigen::Matrix3d RT = X.template block<3,3>(0,0).transpose();
    Xinv.template block<3,3>(0,0) = RT;

    // Vector columns: -R^T * col_c  for c = COL_V..DIM_X-1
    for (int c = COL_V; c < DIM_X; ++c)
        Xinv.template block<3,1>(0,c) = -RT * X.template block<3,1>(0,c);

    return Xinv;
}

// adjoint:  AdX (NR×NR) for X ∈ SE_{N+2}(3) augmented with bias
//
// Lie algebra block (paper Sec. III-A), rows/cols: ξᴿ,ξᵛ,ξᵖ,ξᵈ⁰,ξᵈ¹:
//   AdX[XI_R,  XI_R ] = R
//   AdX[XI_V,  XI_R ] = (v)×R,   AdX[XI_V,  XI_V ] = R
//   AdX[XI_P,  XI_R ] = (p)×R,   AdX[XI_P,  XI_P ] = R
//   AdX[XI_D+3i,XI_R] = (dᵢ)×R, AdX[XI_D+3i,XI_D+3i] = R
//
// Bias block (Euclidean):
//   AdX[XI_BG, XI_BG] = I₃
//   AdX[XI_BA, XI_BA] = I₃
//   all other bias blocks = 0
Eigen::Matrix<double, RightInvariantEKF::NR,
                       RightInvariantEKF::NR>
RightInvariantEKF::adjoint(
    const Eigen::Matrix<double,DIM_X,DIM_X>& X) const
{
    Eigen::Matrix<double,NR,NR> Ad = Eigen::Matrix<double,NR,NR>::Zero();

    const Eigen::Matrix3d& R = X.template block<3,3>(0,0);
    const Eigen::Vector3d  v = X.template block<3,1>(0,COL_V);
    const Eigen::Vector3d  p = X.template block<3,1>(0,COL_P);

    // ── Lie algebra part ──────────────────────────────────────────────────
    // ξᴿ row block
    Ad.template block<3,3>(XI_R, XI_R) = R;

    // ξᵛ row block
    Ad.template block<3,3>(XI_V, XI_R) = skew(v) * R;
    Ad.template block<3,3>(XI_V, XI_V) = R;

    // ξᵖ row block
    Ad.template block<3,3>(XI_P, XI_R) = skew(p) * R;
    Ad.template block<3,3>(XI_P, XI_P) = R;

    // ξᵈⁱ row blocks
    for (int i = 0; i < N_FEET; ++i) {
        const Eigen::Vector3d di = X.template block<3,1>(0, COL_D+i);
        const int row = XI_D + 3*i;
        Ad.template block<3,3>(row, XI_R) = skew(di) * R;
        Ad.template block<3,3>(row, row)  = R;
    }

    // ── Bias part (Euclidean, identity) ───────────────────────────────────
    Ad.template block<3,3>(XI_BG, XI_BG) = Eigen::Matrix3d::Identity();
    Ad.template block<3,3>(XI_BA, XI_BA) = Eigen::Matrix3d::Identity();

    return Ad;
}

// ============================================================================
//  Constructor
// ============================================================================
RightInvariantEKF::RightInvariantEKF(
    const pinocchio::Model&              model,
    const Eigen::VectorXd&               q_init,
    double                               dt,
    const std::array<FootConfig,N_FEET>& feet,
    const NoiseParams&                   noise)
    : model_(model), data_(model), dt_(dt), feet_(feet), noise_(noise)
{
    active_contact_.fill(false);
    bg_.setZero();
    ba_.setZero();
    omega_b_.setZero();

    // ── Initial rotation R_WB from q_init ─────────────────────────────────
    // Pinocchio stores quat as (x,y,z,w) in slots [3:7] of q_init.
    // The quaternion represents body→world, so R_WB = q.toRotationMatrix().
    const Eigen::Quaterniond q0(q_init[6], q_init[3], q_init[4], q_init[5]);
    const Eigen::Matrix3d R_WB = q0.normalized().toRotationMatrix();


    // ── Initial state matrix X_ ───────────────────────────────────────────
    // Identity initialises the bottom-right (N+2)×(N+2) scalar block to I,
    // and zeros all top-row vector slots.  We then fill them in.
    X_ = Eigen::Matrix<double,DIM_X,DIM_X>::Identity();
    X_.template block<3,3>(0,0) = R_WB;        // rotation
    // v column initialised to zero (already via Identity)
    X_.template block<3,1>(0,COL_P) = q_init.head<3>();  // position

    // ── R_imu_to_body ──────────────────────────────────────────────────────
    // R_WB = R_world_body; data_.oMf[imu].rotation() = R_world_imu.
    // We want R such that: v_body = R * v_imu.
    // R_body_imu = R_world_body^T * R_world_imu = R_WB^T * R_world_imu.
    pinocchio::forwardKinematics(model_, data_, q_init);
    pinocchio::updateFramePlacements(model_, data_);

    const int imu_torso_id = model_.getFrameId("imu_in_pelvis");
    R_imu_torso_to_body_ = R_WB.transpose()
                   * data_.oMf[imu_torso_id].rotation();

    const int imu_pelvis_id = model_.getFrameId("imu_in_pelvis");
    R_imu_pelvis_to_body_ = R_WB.transpose()
                   * data_.oMf[imu_pelvis_id].rotation();

    // ── Initial contact positions from FK ──────────────────────────────────
    for (int i = 0; i < N_FEET; ++i) {
        const int fid = model_.getFrameId(feet_[i].frame_name);
        X_.template block<3,1>(0, COL_D + feet_[i].contact_idx)
            = data_.oMf[fid].translation();
    }

    // ── Initial covariance P ──────────────────────────────────────────────
    P_.setZero();
    P_.template block<3,3>(XI_R,  XI_R)  = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_V,  XI_V)  = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_P,  XI_P)  = 0.01 * Eigen::Matrix3d::Identity();
    for (int i = 0; i < N_FEET; ++i)
        P_.template block<3,3>(XI_D+3*i, XI_D+3*i)
            = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_BG, XI_BG) = 1e-4 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_BA, XI_BA) = 1e-4 * Eigen::Matrix3d::Identity();

    // ── Continuous process noise Qc (in ξ-basis, before AdX) ─────────────
    // Channels and their ξ-slots (paper eqs 5,8 and Sec. III-B):
    //   wᵍ  → ξᴿ  (gyro  noise  drives rotation error)
    //   wᵃ  → ξᵛ  (accel noise  drives velocity error)
    //   0   → ξᵖ  (no direct process noise on position)
    //   wᵛᵢ → ξᵈⁱ (slip  noise  drives contact error)
    //   wᵇᵍ → δbᵍ
    //   wᵇᵃ → δbᵃ
    const double sg2  = noise_.gyro_noise    * noise_.gyro_noise;
    const double sa2  = noise_.accel_noise   * noise_.accel_noise;
    const double sv2  = noise_.contact_noise * noise_.contact_noise;
    const double sbg2 = noise_.gyro_bias_rw  * noise_.gyro_bias_rw;
    const double sba2 = noise_.accel_bias_rw * noise_.accel_bias_rw;

    Qc_.setZero();
    Qc_.template block<3,3>(XI_R,  XI_R)  = sg2  * Eigen::Matrix3d::Identity();
    Qc_.template block<3,3>(XI_V,  XI_V)  = sa2  * Eigen::Matrix3d::Identity();
    // XI_P block stays zero
    for (int i = 0; i < N_FEET; ++i)
        Qc_.template block<3,3>(XI_D+3*i, XI_D+3*i) = sv2  * Eigen::Matrix3d::Identity();
    Qc_.template block<3,3>(XI_BG, XI_BG) = sbg2 * Eigen::Matrix3d::Identity();
    Qc_.template block<3,3>(XI_BA, XI_BA) = sba2 * Eigen::Matrix3d::Identity();
}

// ============================================================================
//  addContact  (paper Sec. V: contact switching)
// ============================================================================
void RightInvariantEKF::addContact(int foot_idx,
                                   const Eigen::VectorXd& joint_pos)
{
    // Build current Pinocchio config to evaluate FK
    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head<3>()     = getPosition();
    q_pin.segment<4>(3) = getQuaternion().coeffs();   // (x,y,z,w)
    q_pin.tail(joint_pos.size()) = joint_pos;

    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);

    const int fid = model_.getFrameId(feet_[foot_idx].frame_name);
    const int ci  = feet_[foot_idx].contact_idx;

    // Reset contact position in world frame
    X_.template block<3,1>(0, COL_D + ci) = data_.oMf[fid].translation();

    // Reset covariance block for this contact (cross-terms zeroed)
    const int row = XI_D + 3*foot_idx;
    P_.block(row, 0,   3, NR).setZero();
    P_.block(0,   row, NR, 3).setZero();
    P_.template block<3,3>(row, row) = 0.01 * Eigen::Matrix3d::Identity();

    active_contact_[foot_idx] = true;
}

// ============================================================================
//  removeContact
// ============================================================================
void RightInvariantEKF::removeContact(int foot_idx)
{
    active_contact_[foot_idx] = false;
    // Process noise inflation for non-active contacts is handled in filter()
    // via Qc_step.
}

// ============================================================================
//  filter  –  one full RI-EKF step
// ============================================================================
void RightInvariantEKF::filter(
    const Eigen::Vector3d&         gyro_meas,
    const Eigen::Vector3d&         acc_meas,
    const Eigen::VectorXd&         joint_pos,
    const Eigen::VectorXd&         joint_vel,
    const std::array<bool,N_FEET>& contact)
{
    // ── Contact management ─────────────────────────────────────────────────
    for (int i = 0; i < N_FEET; ++i) {
        if (contact[i] && !active_contact_[i])       addContact(i, joint_pos);
        else if (!contact[i] && active_contact_[i])  removeContact(i);
    }

    // =========================================================================
    // 0)  IMU PRE-PROCESSING
    //     Rotate to body frame, remove bias.
    // =========================================================================

    const Eigen::Vector3d omega_b = R_imu_torso_to_body_ * gyro_meas - bg_;
    const Eigen::Vector3d f_body  = R_imu_pelvis_to_body_ * acc_meas  - ba_;
    omega_b_ = omega_b;

    // Current R_WB (world ← body)
    const Eigen::Matrix3d R_WB = X_.template block<3,3>(0,0);

    // =========================================================================
    // 1)  NOMINAL STATE PROPAGATION  (paper eqs. 4, 7)
    //
    // Ṙ = R (ω)×            →  R_{k+1} = R_k · expSO3(ω_b · Δt)
    // v̇ = R f_body + g      →  v_{k+1} = v_k + (R_k f_body + g) · Δt
    // ṗ = v                 →  p_{k+1} = p_k + v_k·Δt + ½(R_k f_body+g)·Δt²
    // ḋᵢ = 0                →  d_{k+1} = d_k
    // =========================================================================

    const Eigen::Vector3d v_k     = X_.template block<3,1>(0,COL_V);
    const Eigen::Vector3d p_k     = X_.template block<3,1>(0,COL_P);
    const Eigen::Vector3d a_world = R_WB * f_body + g_;

    X_.template block<3,3>(0,0)    = projectToSO3(R_WB * expSO3(omega_b * dt_));
    X_.template block<3,1>(0,COL_V)= v_k + a_world * dt_;
    X_.template block<3,1>(0,COL_P)= p_k + v_k*dt_ + 0.5*a_world*dt_*dt_;
    // Contact columns: unchanged

    // Re-read updated R for subsequent calculations
    const Eigen::Matrix3d& R_new = X_.template block<3,3>(0,0);

    // =========================================================================
    // 2)  COVARIANCE PROPAGATION  (paper eqs. 7-8)
    //
    // Continuous Riccati:  Ṗ = A P + P Aᵀ + Q̂
    //
    // State-transition matrix (discrete, first-order):  Φ ≈ I + A·Δt
    //
    // A (time-invariant Lie part, paper eq. 8):
    //   A(XI_V, XI_R) = g× = skew(g)
    //   A(XI_P, XI_V) = I
    //   A(XI_P, XI_R) = 0  (second-order term only in Φ²)
    //
    // A (bias coupling, paper Sec. IV):
    //   A(XI_R, XI_BG) = -I        (gyro  bias → rotation error)
    //   A(XI_V, XI_BA) = -R_WB     (accel bias → velocity error)
    //
    // Φ = I + A·Δt + ½A²·Δt²  (exact for nilpotent A without bias;
    //     with bias we use first-order: Φ ≈ I + A·Δt)
    //
    // Q̂ = AdX̂ · Qc_step · AdX̂ᵀ · Δt
    // =========================================================================

    Eigen::Matrix<double,NR,NR> Phi = Eigen::Matrix<double,NR,NR>::Identity();

    // Lie algebra part (A·Δt)
    Phi.template block<3,3>(XI_V, XI_R)  = skew(g_)                     * dt_;
    Phi.template block<3,3>(XI_P, XI_V)  = Eigen::Matrix3d::Identity()   * dt_;
    // Second-order term (from A²·Δt²/2): A²(XI_P,XI_R) = I·g×
    Phi.template block<3,3>(XI_P, XI_R)  = 0.5 * skew(g_) * dt_ * dt_;

    // Bias coupling terms (paper Sec. IV)
    // These make At time-varying (through R_WB); we use the *pre-update* R_WB.
    Phi.template block<3,3>(XI_R, XI_BG) = -Eigen::Matrix3d::Identity() * dt_;
    Phi.template block<3,3>(XI_V, XI_BA) = -R_WB                        * dt_;

    // Inflate process noise for non-active contacts
    Eigen::Matrix<double,NR,NR> Qc_step = Qc_;
    for (int i = 0; i < N_FEET; ++i) {
        if (!active_contact_[i])
            Qc_step.template block<3,3>(XI_D+3*i, XI_D+3*i)
                = 1.0 * Eigen::Matrix3d::Identity();
    }

    // Q̂ = AdX̂ · Qc_step · AdX̂ᵀ · Δt
    const Eigen::Matrix<double,NR,NR> AdX  = adjoint(X_);
    const Eigen::Matrix<double,NR,NR> Qhat = AdX * Qc_step * AdX.transpose() * dt_;

    P_ = Phi * P_ * Phi.transpose() + Qhat;

    // =========================================================================
    // 3)  FORWARD KINEMATICS
    //     Evaluated at current state estimate + measured joint angles.
    //     Pinocchio free-flyer convention:
    //       q_pin[0:3]  = base position
    //       q_pin[3:7]  = quaternion (x,y,z,w)  body→world
    //       q_pin[7:nq] = joint angles
    //       v_pin[0:3]  = base linear velocity in BODY frame
    //       v_pin[3:6]  = base angular velocity in body frame
    //       v_pin[6:nv] = joint velocities
    // =========================================================================

    const int n_joints = static_cast<int>(joint_pos.size());

    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head<3>()      = getPosition();
    q_pin.segment<4>(3)  = getQuaternion().coeffs();   // (x,y,z,w)
    q_pin.tail(n_joints) = joint_pos;

    // Pinocchio expects base linear velocity in BODY frame
    Eigen::VectorXd v_pin = Eigen::VectorXd::Zero(model_.nv);
    v_pin.head<3>()      = R_new.transpose() * getVelocity();  // body frame
    v_pin.segment<3>(3)  = omega_b;
    v_pin.tail(n_joints) = joint_vel;

    data_ = pinocchio::Data(model_);
    pinocchio::forwardKinematics(model_, data_, q_pin, v_pin);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_, q_pin);

    // =========================================================================
    // 4)  MEASUREMENT MODEL  (paper Sec. III-C, eqs. 11-14)
    //
    // Right-invariant FK measurement:  Yₜ = Xₜ⁻¹ b + Vₜ
    //
    //   b = [0; 0; 1; -1]   (selects p and -dᵢ columns in homogeneous coords)
    //   hp(α̃) = R^T(dᵢ - p)  (foot pos in body frame from STATE)
    //   Yₜ = [hp_meas; 0; 1; -1]  (hp from FK encoders)
    //
    // Innovation (paper after eq. 13):
    //   z = (X̂ Yₜ)_{top 3}  =  R̂ hp_meas + p̂ - dᵢ  =  p_foot_FK - dᵢ
    //
    // This equals the world-frame error between FK foot position and estimated
    // contact position.  It is independent of base position/orientation errors
    // when the state is correct (trajectory-independent — key RI property).
    //
    // Measurement Jacobian (paper eq. 13, CONSTANT regardless of state!):
    //   H = [0  0  -I  I  0  0]   for contact i
    //        ξᴿ ξᵛ  ξᵖ ξᵈⁱ δbᵍ δbᵃ
    //
    // Measurement noise (paper eq. 14):
    //   N̂ = R̂ Jv_body Σα Jv_bodyᵀ R̂ᵀ  =  J_world Σα J_worldᵀ
    //   (the two expressions are equivalent; we use the simpler world-frame one)
    // =========================================================================

    Eigen::MatrixXd H_all(0, NR);
    Eigen::VectorXd z_all(0);
    Eigen::MatrixXd N_all(0, 0);

    for (int i = 0; i < N_FEET; ++i) {
        if (!active_contact_[i]) continue;

        const int ci  = feet_[i].contact_idx;
        const int fid = model_.getFrameId(feet_[i].frame_name);

        // FK foot position in world frame (from Pinocchio)
        const Eigen::Vector3d p_foot_fk = data_.oMf[fid].translation();

        // Estimated contact position from state
        const Eigen::Vector3d d_i = X_.template block<3,1>(0, COL_D + ci);

        // Innovation:  z = p_foot_FK - d_i  (world frame)
        //   = R̂ hp_meas + p̂ - d_i  which simplifies exactly to this
        const Eigen::Vector3d z_i = p_foot_fk - d_i;

        // Measurement Jacobian H_i (3 × NR)
        // H = [0  0  -I  ...  I  ...  0  0]
        //      XI_R XI_V XI_P   XI_D+3i  XI_BG XI_BA
        Eigen::Matrix<double,3,NR> Hi;
        Hi.setZero();
        Hi.template block<3,3>(0, XI_P)       = -Eigen::Matrix3d::Identity();
        Hi.template block<3,3>(0, XI_D + 3*i) =  Eigen::Matrix3d::Identity();

        // Measurement noise: N̂ = J_world · Σα · J_worldᵀ
        // J_world = top 3 rows (linear) of LOCAL_WORLD_ALIGNED Jacobian,
        //           joint columns only (rightCols(n_joints))
        Eigen::MatrixXd J_full = Eigen::MatrixXd::Zero(6, model_.nv);
        pinocchio::getFrameJacobian(model_, data_, fid,
                                    pinocchio::LOCAL_WORLD_ALIGNED, J_full);
        const Eigen::MatrixXd Jv_world =
            J_full.topRows<3>().rightCols(n_joints);   // 3 × n_joints

        const double se2 = noise_.encoder_noise * noise_.encoder_noise;
        const Eigen::Matrix3d Ni = Jv_world * se2 * Jv_world.transpose();

        // Accumulate
        const int old = static_cast<int>(z_all.rows());
        z_all.conservativeResize(old + 3);
        z_all.segment(old, 3) = z_i;

        H_all.conservativeResize(old + 3, NR);
        H_all.block(old, 0, 3, NR) = Hi;

        N_all.conservativeResize(old + 3, old + 3);
        N_all.block(old,  0,     3, old).setZero();
        N_all.block(0,    old, old,   3).setZero();
        N_all.template block<3,3>(old, old) = Ni;
    }

    if (z_all.size() == 0)
        return;   // no active contacts: pure propagation

    // =========================================================================
    // 5)  KALMAN UPDATE  (paper eq. 14)
    //
    //   S   = H P Hᵀ + N̂
    //   K   = P Hᵀ S⁻¹                    (NR × m gain)
    //   ξ⁺  = K z                          (correction in ξ-basis)
    //
    // State update (right-invariant, paper eq. 14):
    //   X̂⁺ = exp(Lg(ξ_lie⁺)) · X̂          (left multiplication)
    //   b̂⁺ = b̂ + ξ_bias⁺                   (Euclidean bias update)
    //
    // Covariance update (Joseph form for numerical stability):
    //   P⁺ = (I − KH) P (I − KH)ᵀ + K N̂ Kᵀ
    // =========================================================================

    const Eigen::MatrixXd S = H_all * P_ * H_all.transpose() + N_all;
    const Eigen::MatrixXd K = P_ * H_all.transpose() * S.inverse();

    const Eigen::VectorXd xi_corr = K * z_all;   // dim NR = 21

    // Extract Lie algebra correction (first 3*(N_FEET+3) = 15 components)
    Eigen::Matrix<double,3*(N_FEET+3),1> xi_lie;
    xi_lie.template head<3>() = xi_corr.template segment<3>(XI_R);
    xi_lie.template segment<3>(3) = xi_corr.template segment<3>(XI_V);
    xi_lie.template segment<3>(6) = xi_corr.template segment<3>(XI_P);
    for (int i = 0; i < N_FEET; ++i)
        xi_lie.template segment<3>(9 + 3*i) = xi_corr.template segment<3>(XI_D + 3*i);

    // State update: X̂⁺ = exp(Lg(ξ_lie)) · X̂
    X_ = groupExp(xi_lie) * X_;

    // Re-project R to SO(3) after numerical accumulation
    X_.template block<3,3>(0,0) = projectToSO3(X_.template block<3,3>(0,0));

    // Bias update (Euclidean)
    bg_ += xi_corr.template segment<3>(XI_BG);
    ba_ += xi_corr.template segment<3>(XI_BA);

    // Covariance update: Joseph form
    const Eigen::Matrix<double,NR,NR> I_mat =
        Eigen::Matrix<double,NR,NR>::Identity();
    const Eigen::MatrixXd IKH = I_mat - K * H_all;
    P_ = IKH * P_ * IKH.transpose() + K * N_all * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());   // enforce symmetry
}

// ============================================================================
//  Static SO(3) helpers
// ============================================================================

Eigen::Matrix3d DiligentKio::skew(const Eigen::Vector3d& v)
{
    Eigen::Matrix3d S;
    S <<    0, -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;
    return S;
}

Eigen::Matrix3d DiligentKio::expSO3(const Eigen::Vector3d& phi)
{
    const double th = phi.norm();
    if (th < 1e-9)
        return Eigen::Matrix3d::Identity() + skew(phi);
    const Eigen::Matrix3d K = skew(phi / th);
    return Eigen::Matrix3d::Identity()
         + std::sin(th) * K
         + (1.0 - std::cos(th)) * K * K;
}

Eigen::Vector3d DiligentKio::logSO3(const Eigen::Matrix3d& R)
{
    const double cos_th = std::clamp(0.5*(R.trace()-1.0), -1.0, 1.0);
    const double th     = std::acos(cos_th);
    if (th < 1e-9) return Eigen::Vector3d::Zero();
    const double s = th / (2.0*std::sin(th));
    return s * Eigen::Vector3d(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));
}

Eigen::Matrix3d DiligentKio::leftJacobianSO3(const Eigen::Vector3d& phi)
{
    // J_l(φ) = I + ((1−cosθ)/θ²) [φ]× + ((θ−sinθ)/θ³) [φ]×²
    const double th = phi.norm();
    if (th < 1e-7)
        return Eigen::Matrix3d::Identity() + 0.5*skew(phi);
    const Eigen::Matrix3d K = skew(phi);
    return Eigen::Matrix3d::Identity()
         + ((1.0-std::cos(th))/(th*th)) * K
         + ((th-std::sin(th)) /(th*th*th)) * K*K;
}

Eigen::Matrix3d DiligentKio::leftJacobianInvSO3(const Eigen::Vector3d& phi)
{
    // J_l^{-1}(φ) = I − ½[φ]× + (1/θ² − (1+cosθ)/(2θ sinθ)) [φ]×²
    const double th = phi.norm();
    if (th < 1e-7)
        return Eigen::Matrix3d::Identity() - 0.5*skew(phi);
    const Eigen::Matrix3d K = skew(phi);
    const double c = (1.0/( th*th )) - (1.0+std::cos(th))/(2.0*th*std::sin(th));
    return Eigen::Matrix3d::Identity() - 0.5*K + c*K*K;
}

// Q matrix  (Barfoot & Furgale 2014, Appendix A)
// Used in SE(3) left Jacobian:  ΦSE(3) = [J(ω)   Q(ρ,ω)]
//                                         [0₃     J(ω)  ]
Eigen::Matrix3d DiligentKio::QmatrixSE3(const Eigen::Vector3d& rho,
                                        const Eigen::Vector3d& phi)
{
    const double th = phi.norm();
    const Eigen::Matrix3d Kp = skew(rho);
    const Eigen::Matrix3d Kw = skew(phi);

    if (th < 1e-7) {
        // small-angle series: Q ≈ ½ S(ρ) + (1/6) S(φ)S(ρ) + (1/6) S(ρ)S(φ)
        return 0.5*Kp + (1.0/6.0)*(Kw*Kp + Kp*Kw);
    }

    const double th2 = th*th, th3 = th2*th, th4 = th2*th2, th5 = th4*th;
    const double s = std::sin(th), c = std::cos(th);

    const double c1 = (th - s)   / th3;
    const double c2 = (th2 + 2.0*c - 2.0) / (2.0*th4);
    const double c3 = (2.0*th - 3.0*s + th*c) / (2.0*th5);

    return 0.5*Kp
         + c1*(Kw*Kp + Kp*Kw + Kw*Kp*Kw)
         - c2*(Kw*Kw*Kp + Kp*Kw*Kw - 3.0*Kw*Kp*Kw)
         - 0.5*(c2 - 3.0*c3)*(Kw*Kp*Kw*Kw + Kw*Kw*Kp*Kw);
}

Eigen::Matrix3d DiligentKio::projectToSO3(const Eigen::Matrix3d& M)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    S(2,2) = (svd.matrixU()*svd.matrixV().transpose()).determinant();
    return svd.matrixU() * S * svd.matrixV().transpose();
}

// ============================================================================
//  SE(3) helpers
// ============================================================================

// log^∨_SE(3): returns [v(0:3); ω(3:6)]  (paper Table II convention)
Eigen::Matrix<double,6,1>
DiligentKio::logSE3(const Eigen::Matrix3d& R_h, const Eigen::Vector3d& t_h)
{
    Eigen::Matrix<double,6,1> xi;
    const Eigen::Vector3d phi = logSO3(R_h);             // ω part
    xi.tail<3>() = phi;
    xi.head<3>() = leftJacobianInvSO3(phi) * t_h;        // v part
    return xi;
}

// Left Jacobian of SE(3), 6×6 (paper appendix / Barfoot & Furgale 2014)
// ΦSE(3)([v;ω]) = [J_l(ω)   Q(v,ω)]
//                 [0₃        J_l(ω)]
Eigen::Matrix<double,6,6>
DiligentKio::leftJacobianSE3(const Eigen::Vector3d& eps_v,
                              const Eigen::Vector3d& eps_omega)
{
    Eigen::Matrix<double,6,6> Phi = Eigen::Matrix<double,6,6>::Zero();
    const Eigen::Matrix3d Jl = leftJacobianSO3(eps_omega);
    const Eigen::Matrix3d Q  = QmatrixSE3(eps_v, eps_omega);
    Phi.block<3,3>(0,0) = Jl;
    Phi.block<3,3>(0,3) = Q;
    Phi.block<3,3>(3,3) = Jl;
    return Phi;
}

// ============================================================================
//  Constructor
// ============================================================================

DiligentKio::DiligentKio(const pinocchio::Model&              model,
                         const Eigen::VectorXd&               q_init,
                         double                               dt,
                         const std::array<FootConfig,N_FEET>& feet,
                         const NoiseParams&                   noise)
    : model_(model), data_(model), dt_(dt), feet_(feet), noise_(noise)
{
    active_contact_.fill(false);
    omega_b_.setZero();
    ba_.setZero();
    bg_.setZero();

    // ── Initial rotation R_WB from q_init ─────────────────────────────────
    // q_init[3:7] = (x,y,z,w) quaternion, body→world
    const Eigen::Quaterniond q0(q_init[6], q_init[3], q_init[4], q_init[5]);
    R_ = q0.normalized().toRotationMatrix();   // body→world
    p_ = q_init.head<3>();
    v_.setZero();

    // ── R_imu_to_body ──────────────────────────────────────────────────────
    // Paper assumes S ≡ B (IMU = body). If they differ:
    // R_body_imu = R_WB^T * R_world_imu  (rotate world→body, then world→imu^T)
    pinocchio::forwardKinematics(model_, data_, q_init);
    pinocchio::updateFramePlacements(model_, data_);

    const int imu_id = model_.getFrameId("imu_in_pelvis");
    // data_.oMf[imu_id].rotation() = R_world_imu (maps imu vectors to world)
    // We want: v_body = R_imu_to_body * v_imu
    // R_imu_to_body = R_world_body^T * R_world_imu = R_WB^T * R_world_imu
    R_imu_to_body_ = R_.transpose() * data_.oMf[imu_id].rotation();

    // ── Initial foot poses from FK ─────────────────────────────────────────
    for (int i = 0; i < N_FEET; ++i) {
        const int fid = model_.getFrameId(feet_[i].frame_name);
        if (i == 0) {
            dl_ = data_.oMf[fid].translation();
            Zl_ = data_.oMf[fid].rotation();   // foot→world
        } else {
            dr_ = data_.oMf[fid].translation();
            Zr_ = data_.oMf[fid].rotation();
        }
    }

    // ── Initial covariance P (paper Table III initial std devs) ───────────
    P_.setZero();
    const double sp  = 0.01*0.01;   // position: 0.01m std
    const double sR  = (10.0*M_PI/180.0)*(10.0*M_PI/180.0);  // 10° std
    const double sv  = 0.5*0.5;     // velocity: 0.5 m/s std
    const double sdf = 0.01*0.01;   // foot pos: 0.01m
    const double sZf = sR;          // foot rot: 10°
    const double sba = 0.01*0.01;
    const double sbg = 0.002*0.002;

    P_.block<3,3>(EI_P, EI_P)   = sp  * Eigen::Matrix3d::Identity();
    P_.block<3,3>(EI_R, EI_R)   = sR  * Eigen::Matrix3d::Identity();
    P_.block<3,3>(EI_V, EI_V)   = sv  * Eigen::Matrix3d::Identity();
    P_.block<3,3>(EI_DL,EI_DL)  = sdf * Eigen::Matrix3d::Identity();
    P_.block<3,3>(EI_ZL,EI_ZL)  = sZf * Eigen::Matrix3d::Identity();
    P_.block<3,3>(EI_DR,EI_DR)  = sdf * Eigen::Matrix3d::Identity();
    P_.block<3,3>(EI_ZR,EI_ZR)  = sZf * Eigen::Matrix3d::Identity();
    P_.block<3,3>(EI_BA,EI_BA)  = sba * Eigen::Matrix3d::Identity();
    P_.block<3,3>(EI_BG,EI_BG)  = sbg * Eigen::Matrix3d::Identity();

    // ── Discrete process noise Q  (paper eq. 12, scaled by ΔT²) ──────────
    // Noise vector w same ordering as ε (eq. 12):
    //   w = (-0.5 wa ΔT, -wω, -wa, w^LF_v, w^LF_ω, w^RF_v, w^RF_ω, wb) ΔT
    // Cov(w): each component multiplied by ΔT at end, so ΔT² factor overall.
    const double dt2 = dt_*dt_;

    const double qa  = noise_.accel_noise    * noise_.accel_noise;
    const double qo  = noise_.gyro_noise     * noise_.gyro_noise;
    const double qba = noise_.accel_bias_rw  * noise_.accel_bias_rw;
    const double qbg = noise_.gyro_bias_rw   * noise_.gyro_bias_rw;
    const double qfv = noise_.foot_lin_noise * noise_.foot_lin_noise;
    const double qfw = noise_.foot_ang_noise * noise_.foot_ang_noise;

    // Noise on p: from (-0.5 wa ΔT) component → variance = 0.25 qa ΔT² per step
    // but the noise is wa * ΔT² (already multiplied by ΔT in w), then squared:
    // Actually Cov(w_p) = 0.25 * qa * ΔT² * ΔT² = 0.25 * qa * dt^4... 
    // For practical purposes: Q_p = 0.25 * qa * dt^2 * dt^2 is tiny.
    // Use Q_p = 0.25 * qa * dt^2 (first order approx at typical dt=0.01s)
    Q_.setZero();
    Q_.block<3,3>(EI_P, EI_P)  = 0.25*qa * dt2 * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(EI_R, EI_R)  = qo  * dt2 * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(EI_V, EI_V)  = qa  * dt2 * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(EI_DL,EI_DL) = qfv * dt2 * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(EI_ZL,EI_ZL) = qfw * dt2 * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(EI_DR,EI_DR) = qfv * dt2 * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(EI_ZR,EI_ZR) = qfw * dt2 * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(EI_BA,EI_BA) = qba * dt2 * Eigen::Matrix3d::Identity();
    Q_.block<3,3>(EI_BG,EI_BG) = qbg * dt2 * Eigen::Matrix3d::Identity();
}

// ============================================================================
//  addContact / removeContact  (paper Sec. III-B)
// ============================================================================

void DiligentKio::addContact(int foot_idx, const Eigen::VectorXd& joint_pos)
{
    // Re-initialise foot pose from FK at current base estimate
    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head<3>()     = p_;
    q_pin.segment<4>(3) = getQuaternion().coeffs();
    q_pin.tail(joint_pos.size()) = joint_pos;

    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);

    const int fid = model_.getFrameId(feet_[foot_idx].frame_name);
    if (foot_idx == 0) {
        dl_ = data_.oMf[fid].translation();
        Zl_ = data_.oMf[fid].rotation();
    } else {
        dr_ = data_.oMf[fid].translation();
        Zr_ = data_.oMf[fid].rotation();
    }

    // Reset foot covariance block (cross-terms zeroed, diagonal re-initialised)
    const int row_d = (foot_idx==0) ? EI_DL : EI_DR;
    const int row_z = (foot_idx==0) ? EI_ZL : EI_ZR;

    P_.block(row_d, 0,   3, NR).setZero();
    P_.block(0,   row_d, NR, 3).setZero();
    P_.block(row_z, 0,   3, NR).setZero();
    P_.block(0,   row_z, NR, 3).setZero();

    const double sp = 0.01*0.01;
    const double sR = (10.0*M_PI/180.0)*(10.0*M_PI/180.0);
    P_.block<3,3>(row_d, row_d) = sp * Eigen::Matrix3d::Identity();
    P_.block<3,3>(row_z, row_z) = sR * Eigen::Matrix3d::Identity();

    active_contact_[foot_idx] = true;
}

void DiligentKio::removeContact(int foot_idx)
{
    active_contact_[foot_idx] = false;
    // Foot noise inflation handled per-step in filter()
}

// ============================================================================
//  filter  –  one full DILIGENT-KIO step  (Algorithm 1)
// ============================================================================
void DiligentKio::filter(const Eigen::Vector3d&         gyro_meas,
                         const Eigen::Vector3d&         accel_meas,
                         const Eigen::VectorXd&         joint_pos,
                         const Eigen::VectorXd&         joint_vel,
                         const std::array<bool,N_FEET>& contact)
{
    // ── Contact management ─────────────────────────────────────────────────
    for (int i = 0; i < N_FEET; ++i) {
        if (contact[i] && !active_contact_[i])      addContact(i, joint_pos);
        else if (!contact[i] && active_contact_[i]) removeContact(i);
    }

    // =========================================================================
    // 0)  IMU PRE-PROCESSING  (paper eq. 9)
    //
    //   BαA,B = (ã − ba − wa) + R^T g   [linear accel in body frame]
    //   BωA,B = ω̃ − bg − wω             [angular velocity in body frame]
    //
    // We drop the noise terms (they go into the process noise covariance).
    // =========================================================================

    omega_b_ = R_imu_to_body_ * gyro_meas  - bg_;
    const Eigen::Vector3d f_b     = R_imu_to_body_ * accel_meas - ba_;
    // BαA,B = f_b + R^T g_world (specific force + gravity rotated to body)
    const Eigen::Vector3d alpha_b = f_b + R_.transpose() * g_world_;
    // Base acceleration in world frame
    const Eigen::Vector3d a_world = R_ * alpha_b;   // = R f_b + g_world

    // =========================================================================
    // 1)  NOMINAL STATE PROPAGATION  (paper eq. 10)
    //
    //   pk+1 = pk + vk ΔT + ½ Rk Bα ΔT²
    //   Rk+1 = Rk expSO3(Bω ΔT)
    //   vk+1 = vk + Rk Bα ΔT
    //   df/Zf: unchanged (constant contact model)
    //   b: unchanged
    //
    // This is X_{k+1|k} = X_{k|k} exp^∧_M(Ω̂)  with Ω̂ from eq. (11).
    // =========================================================================

    const Eigen::Vector3d v_prev = v_;
    p_ += v_prev * dt_ + 0.5 * R_ * alpha_b * dt_ * dt_;
    v_ += R_ * alpha_b * dt_;
    R_  = projectToSO3(R_ * expSO3(omega_b_ * dt_));
    // feet and biases unchanged

    // =========================================================================
    // 2)  COVARIANCE PROPAGATION  (paper Algorithm 1)
    //
    //   Pk+1|k = Fk Pk|k Fk^T + ΦM(-Ω̂) Qk ΦM(-Ω̂)^T
    //
    // We use the first-order approximations:
    //   Fk  ≈ I₂₇ + F̄k   (paper eq. 14 + identity from AdM ≈ I)
    //   ΦM(-Ω̂) ≈ I₂₇     (ΦM → I for small ΔT)
    //
    // So:  Pk+1|k ≈ (I + F̄k) Pk|k (I + F̄k)^T + Qk
    //
    // F̄k (paper eq. 14) with column/row ordering [εp,εR,εv,εdl,εZl,εdr,εZr,εba,εbg]:
    //
    //   F̄k[EI_P, EI_R]  = S(Ξ)       where Ξ = R^T v ΔT + ½ R^T g ΔT²
    //   F̄k[EI_P, EI_V]  = I₃ ΔT
    //   F̄k[EI_P, EI_BA] = -½ I₃ ΔT²
    //   F̄k[EI_R, EI_BG] = -I₃ ΔT
    //   F̄k[EI_V, EI_R]  = S(g̃)      where g̃ = R^T g ΔT
    //   F̄k[EI_V, EI_BA] = -I₃ ΔT
    //
    // Note: quantities are evaluated at the PRE-PROPAGATION state X_{k|k},
    // but since we already updated R_ above, we use the pre-update values.
    // For simplicity (and following standard practice) we use the post-
    // update R_ here — the error is O(ΔT²).
    // =========================================================================

    // Ξ = R^T v_prev ΔT + ½ R^T g_world ΔT²
    const Eigen::Vector3d Xi  = R_.transpose() * v_prev * dt_
                               + 0.5 * R_.transpose() * g_world_ * dt_*dt_;
    // g̃ = R^T g_world ΔT
    const Eigen::Vector3d g_tilde = R_.transpose() * g_world_ * dt_;

    Eigen::Matrix<double,NR,NR> Fk = Eigen::Matrix<double,NR,NR>::Identity();

    // F̄k off-diagonal blocks  (added to the I₂₇)
    Fk.block<3,3>(EI_P, EI_R)  += skew(Xi);
    Fk.block<3,3>(EI_P, EI_V)  += Eigen::Matrix3d::Identity() * dt_;
    Fk.block<3,3>(EI_P, EI_BA) -= 0.5 * Eigen::Matrix3d::Identity() * dt_*dt_;
    Fk.block<3,3>(EI_R, EI_BG) -= Eigen::Matrix3d::Identity() * dt_;
    Fk.block<3,3>(EI_V, EI_R)  += skew(g_tilde);
    Fk.block<3,3>(EI_V, EI_BA) -= Eigen::Matrix3d::Identity() * dt_;

    // Per-step process noise: inflate foot noise when not in contact
    Eigen::Matrix<double,NR,NR> Q_step = Q_;
    for (int i = 0; i < N_FEET; ++i) {
        if (!active_contact_[i]) {
            const int rd = (i==0) ? EI_DL : EI_DR;
            const int rz = (i==0) ? EI_ZL : EI_ZR;
            Q_step.block<3,3>(rd,rd) = 100.0 * Eigen::Matrix3d::Identity();
            Q_step.block<3,3>(rz,rz) = 100.0 * Eigen::Matrix3d::Identity();
        }
    }

    P_ = Fk * P_ * Fk.transpose() + Q_step;

    // =========================================================================
    // 3)  FORWARD KINEMATICS
    //     Build Pinocchio state from current estimate + measured joint angles.
    //
    //     Pinocchio free-flyer:
    //       q_pin[0:3]  = base position
    //       q_pin[3:7]  = quaternion (x,y,z,w) body→world
    //       q_pin[7:nq] = joint positions
    //       v_pin[0:3]  = base LINEAR velocity in BODY frame
    //       v_pin[3:6]  = base angular velocity in body frame
    // =========================================================================

    const int n_j = static_cast<int>(joint_pos.size());

    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head<3>()     = p_;
    q_pin.segment<4>(3) = getQuaternion().coeffs();
    q_pin.tail(n_j)     = joint_pos;

    Eigen::VectorXd v_pin = Eigen::VectorXd::Zero(model_.nv);
    v_pin.head<3>()     = R_.transpose() * v_;   // body frame
    v_pin.segment<3>(3) = omega_b_;
    v_pin.tail(n_j)     = joint_vel;

    data_ = pinocchio::Data(model_);
    pinocchio::forwardKinematics(model_, data_, q_pin, v_pin);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_, q_pin);

    // =========================================================================
    // 4)  MEASUREMENT MODEL  (paper Sec. III-C, eqs. 15-17)
    //
    // For each foot f in contact, the FK measurement is:
    //   z^f = B H_F = FK(s̃) ∈ SE(3)   (pose of foot in base frame)
    //
    // Predicted measurement from state (paper eq. 16):
    //   h(X̂) = [R̂^T Ẑ_f   R̂^T (d̂_f − p̂)]  ∈ SE(3)
    //            [0₁,₃       1              ]
    //
    // Innovation (Algorithm 1):
    //   z̃ = log^∨_SE(3)(h(X̂)^{-1} · z_meas)  ∈ ℝ⁶
    //
    // Measurement Jacobian (paper eq. 17):
    //   H^LF = [-Ẑ_l^T R̂   -Ẑ_l^T S(p̂−d̂_l) R̂  0₃  I₃  0₃  0₃,₁₂]  ← translation rows
    //           [0₃          -Ẑ_l^T R̂            0₃  0₃  I₃  0₃,₁₂]  ← rotation rows
    //
    // Measurement noise (paper Sec. III-C):
    //   N_f = F J_{B,F} Σ_α J_{B,F}^T F^T  ∈ ℝ⁶ˣ⁶
    //   where F J_{B,F} is the 6×n_j Jacobian in the LOCAL (foot) frame.
    // =========================================================================

    Eigen::MatrixXd H_all(0, NR);
    Eigen::VectorXd z_all(0);
    Eigen::MatrixXd N_all(0, 0);

    for (int i = 0; i < N_FEET; ++i) {
        if (!active_contact_[i]) continue;

        const int fid          = model_.getFrameId(feet_[i].frame_name);
        const Eigen::Matrix3d& Zf = (i==0) ? Zl_ : Zr_;
        const Eigen::Vector3d& df = (i==0) ? dl_ : dr_;
        const int EI_DI        = (i==0) ? EI_DL : EI_DR;
        const int EI_ZI        = (i==0) ? EI_ZL : EI_ZR;

        // FK measurement: pose of foot in BASE frame
        // B H_F = T_world_base^{-1} * T_world_foot
        const Eigen::Matrix3d R_wf  = data_.oMf[fid].rotation();    // foot→world
        const Eigen::Vector3d p_wf  = data_.oMf[fid].translation(); // foot pos world
        // Foot in base frame:
        const Eigen::Matrix3d FK_R  = R_.transpose() * R_wf;         // foot→base
        const Eigen::Vector3d FK_p  = R_.transpose() * (p_wf - p_);  // foot pos base

        // Predicted measurement from state  h(X̂):
        // h_R = R̂^T Ẑ_f (foot→base from state)
        // h_t = R̂^T (d̂_f − p̂) (foot pos in base from state)
        const Eigen::Matrix3d h_R  = R_.transpose() * Zf;
        const Eigen::Vector3d h_t  = R_.transpose() * (df - p_);

        // h(X̂)^{-1} = [h_R^T, -h_R^T h_t; 0, 1]
        //            = [Ẑ_f^T R̂, -Ẑ_f^T(d̂_f - p̂); 0, 1]
        //            = [Zf^T R, -(Zf^T) h_t; 0, 1] ... same as [h_R^T, -h_R^T h_t]
        const Eigen::Matrix3d hinv_R = h_R.transpose();   // = Zf^T R_
        const Eigen::Vector3d hinv_t = -hinv_R * h_t;     // = Zf^T R_ * -(h_t)

        // Product:  hinv * FK_meas  =  [hinv_R FK_R, hinv_R FK_p + hinv_t]
        const Eigen::Matrix3d prod_R = hinv_R * FK_R;
        const Eigen::Vector3d prod_t = hinv_R * FK_p + hinv_t;

        // Innovation z̃ = log^∨_SE(3)(prod)  ∈ ℝ⁶, [v(0:3); ω(3:6)]
        const Eigen::Matrix<double,6,1> z_i = logSE3(prod_R, prod_t);

        // Measurement Jacobian H_i ∈ ℝ⁶ˣ²⁷ (paper eq. 17)
        // ZfR = Ẑ_f^T R̂  (3×3)
        const Eigen::Matrix3d ZfR  = Zf.transpose() * R_;
        // S(p-df) R = S(p̂ − d̂_f) R̂
        const Eigen::Matrix3d SpR  = skew(p_ - df) * R_;

        Eigen::Matrix<double,6,NR> Hi;
        Hi.setZero();
        // Translation rows (top 3):
        Hi.block<3,3>(0, EI_P)  = -ZfR;
        Hi.block<3,3>(0, EI_R)  = -Zf.transpose() * SpR;  // = -Zf^T S(p-df) R
        Hi.block<3,3>(0, EI_DI) =  Eigen::Matrix3d::Identity();
        // Rotation rows (bottom 3):
        Hi.block<3,3>(3, EI_R)  = -ZfR;
        Hi.block<3,3>(3, EI_ZI) =  Eigen::Matrix3d::Identity();

        // Measurement noise  N_f = F J Σα J^T F^T  (foot-frame Jacobian)
        Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(6, model_.nv);
        pinocchio::getFrameJacobian(model_, data_, fid,
                                    pinocchio::LOCAL, J_local);
        const Eigen::MatrixXd Jf = J_local.rightCols(n_j);  // 6×n_j, LOCAL frame

        const double se2 = (noise_.encoder_noise_deg * M_PI/180.0)
                         * (noise_.encoder_noise_deg * M_PI/180.0);
        const Eigen::Matrix<double,6,6> Ni = Jf * (se2 * Eigen::MatrixXd::Identity(n_j,n_j))
                                                * Jf.transpose();

        // Accumulate
        const int old = static_cast<int>(z_all.rows());
        z_all.conservativeResize(old + 6);
        z_all.segment(old, 6) = z_i;

        H_all.conservativeResize(old + 6, NR);
        H_all.block(old, 0, 6, NR) = Hi;

        N_all.conservativeResize(old + 6, old + 6);
        N_all.block(old,  0,    6, old).setZero();
        N_all.block(0,    old, old, 6 ).setZero();
        N_all.block<6,6>(old, old) = Ni;
    }

    if (z_all.size() == 0)
        return;   // no contacts: pure propagation

    // =========================================================================
    // 5)  MEASUREMENT UPDATE  (Algorithm 1)
    //
    //   S     = H P H^T + N
    //   K     = P H^T S^{-1}
    //   m^-   = K z̃
    //   X̂⁺ = X̂ exp^∧_M(m^-)     ← LEFT multiplication (left-invariant error)
    //   P⁺    = ΦM(−m^−) (I − KH) P ΦM(−m^−)^T
    //         ≈ (I − KH) P (I − KH)^T + K N K^T   [Joseph form, ΦM ≈ I]
    // =========================================================================

    const Eigen::MatrixXd S = H_all * P_ * H_all.transpose() + N_all;
    const Eigen::MatrixXd K = P_ * H_all.transpose() * S.inverse();

    const Eigen::VectorXd m = K * z_all;   // dim 27

    // ── Apply correction via exp^∧_M(m)  (left multiply each Lie-group block)
    //
    // SE₂(3) block: (p, R, v) → X_base * exp_SE₂(3)(m_base)
    //   = (R · J_l(φ) ε_p + p,  R · exp(φ),  R · J_l(φ) ε_v + v)
    //   where φ = m[EI_R], ε_p = m[EI_P], ε_v = m[EI_V]
    //
    const Eigen::Vector3d phi_base = m.segment<3>(EI_R);
    const Eigen::Vector3d eps_p    = m.segment<3>(EI_P);
    const Eigen::Vector3d eps_v    = m.segment<3>(EI_V);
    const Eigen::Matrix3d Jl_base  = leftJacobianSO3(phi_base);

    p_ += R_ * Jl_base * eps_p;
    v_ += R_ * Jl_base * eps_v;
    R_  = projectToSO3(R_ * expSO3(phi_base));

    // SE(3) block for each foot: (df, Zf) → X_f * exp_SE(3)(m_foot)
    //   = (Zf · J_l(φ_f) ε_df + df,  Zf · exp(φ_f))
    for (int i = 0; i < N_FEET; ++i) {
        const int  eid = (i==0) ? EI_DL : EI_DR;
        const int  eiz = (i==0) ? EI_ZL : EI_ZR;
        Eigen::Matrix3d& Zf = (i==0) ? Zl_ : Zr_;
        Eigen::Vector3d& df = (i==0) ? dl_ : dr_;

        const Eigen::Vector3d phi_f = m.segment<3>(eiz);
        const Eigen::Vector3d eps_d = m.segment<3>(eid);
        const Eigen::Matrix3d Jl_f  = leftJacobianSO3(phi_f);

        df += Zf * Jl_f * eps_d;
        Zf  = projectToSO3(Zf * expSO3(phi_f));
    }

    // T(6) block for biases (Euclidean):
    ba_ += m.segment<3>(EI_BA);
    bg_ += m.segment<3>(EI_BG);

    // ── Covariance update: Joseph form ─────────────────────────────────────
    const Eigen::Matrix<double,NR,NR> I_mat =
        Eigen::Matrix<double,NR,NR>::Identity();
    const Eigen::MatrixXd IKH = I_mat - K * H_all;
    P_ = IKH * P_ * IKH.transpose() + K * N_all * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
}

} // namespace state_filtering
