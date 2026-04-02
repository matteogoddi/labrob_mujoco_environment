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

  Eigen::Matrix3d C_world_to_base = q_.toRotationMatrix().transpose();
  // C_world_to_base = (Eigen::Matrix3d::Identity() + 2*q_.w()*skew(q_.vec()) + 2 * skew(q_.vec()) * skew(q_.vec())).transpose();


  Eigen::VectorXd pos = Eigen::VectorXd::Zero(model_.nq);
  pos.head(3) << r_;
  pos.segment(3, 4) << q_.x(), q_.y(), q_.z(), q_.w();
  pos.tail(qj.size()) = qj;

  pinocchio::forwardKinematics(model_, data_, pos);
  pinocchio::framesForwardKinematics(model_, data_, pos);
  pinocchio::computeJointJacobians(model_, data_, pos);

  Eigen::VectorXd vel = Eigen::VectorXd::Zero(model_.nv);
  vel.head(3) << v_;
  vel.segment(3, 3) << omega_world;
  vel.tail(qj_dot.size()) = qj_dot;

  Eigen::Vector3d acc = R_base_imu * acc_meas - bf_;

  omega_ = R_base_imu * gyro_meas - bw_;
  omega_world = C_world_to_base.transpose() * omega_;

  Eigen::Vector3d a_world = C_world_to_base.transpose() * acc + g_;

  Eigen::Vector3d r_prec = r_;
  Eigen::Vector3d v_prec = v_;
  r_ += dt_ * v_ + 0.5 * dt_ * dt_ * a_world;
  v_ += dt_ * a_world;
  q_ = (q_ * expMap(dt_ * omega_)).normalized();


  // =====================
  // 2) COVARIANCE PREDICTION
  // =====================
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(NX, NX);

  int ir=0, iv=3, iphi=6, ipL=9, ipR=12, ithetaL=15, ithetaR=18, ibf=21, ibw=24;

  F.block<3,3>(ir, iv) = Eigen::Matrix3d::Identity() * dt_;
  F.block<3,3>(iv, iphi) = -C_world_to_base.transpose() * skew(acc) * dt_;
  // F.block<3,3>(iv, ibf)  = -C_world_to_base.transpose() * dt_;
  F.block<3,3>(iphi, iphi) = Eigen::Matrix3d::Identity() - skew(omega_) * dt_;
  // F.block<3,3>(iphi, ibw) = -Eigen::Matrix3d::Identity() * dt_;

  Eigen::MatrixXd Lc = Eigen::MatrixXd::Zero(NX, NX - 3); // no noise for base position dynamics
  Lc.block<3,3>(iv, 0) = -C_world_to_base.transpose();
  Lc.block<3,3>(iphi, 3) = -Eigen::Matrix3d::Identity();
  Lc.block<3,3>(ipL, 6) = C_world_to_base.transpose();
  Lc.block<3,3>(ipR, 9) = C_world_to_base.transpose();
  // Lc.block<3,3>(ibf, 12) = Eigen::Matrix3d::Identity();
  // Lc.block<3,3>(ibw, 15) = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double,18,18> Qc_step = Qc_;
  if (!left_contact)  Qc_step.block<3,3>(6,6)  = 1.0 * Eigen::Matrix3d::Identity();
  if (!right_contact) Qc_step.block<3,3>(9,9)  = 1.0 * Eigen::Matrix3d::Identity();
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
    Eigen::Quaterniond s_z_hat = q_.inverse() * z.inverse();



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

    H_accum.block<3,3>(old_rows+3, iphi) = -Eigen::Matrix3d::Identity();
    H_accum.block<3,3>(old_rows+3, itheta) =
        - (q_.inverse() * z.inverse()).toRotationMatrix();

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
  // bf_ += dx.segment<3>(ibf);
  // bw_ += dx.segment<3>(ibw);

  q_  = (q_ * expMap(dx.segment<3>(iphi))).normalized();
  zL_ = (zL_ * expMap(dx.segment<3>(ithetaL))).normalized();
  zR_ = (zR_ * expMap(dx.segment<3>(ithetaR))).normalized();

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

    const int imu_torso_id = model_.getFrameId("imu_in_torso");
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
                = 100.0 * Eigen::Matrix3d::Identity();
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

} // namespace state_filtering
