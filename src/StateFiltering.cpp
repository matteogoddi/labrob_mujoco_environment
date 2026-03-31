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
  omega_world = C_world_to_base * omega_;

  Eigen::Vector3d a_world = C_world_to_base * acc;

  Eigen::Vector3d r_prec = r_;
  Eigen::Vector3d v_prec = v_;
  r_ += dt_ * v_ + 0.5 * dt_ * dt_ * a_world;
  v_ += dt_ * a_world;
  q_ = (q_ * expMap(dt_ * omega_)).normalized();


  // =====================
  // 2) COVARIANCE PREDICTION
  // =====================
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(NX, NX);

  int ir=0, iv=3, iphi=6, ipL=9, ipR=12, ibf=15, ibw=18;

  F.block<3,3>(ir, iv) = Eigen::Matrix3d::Identity() * dt_;
  F.block<3,3>(iv, iphi) = skew(C_world_to_base.transpose() * acc) * dt_;
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

  Eigen::Matrix<double,12,12> Qc_step = Qc_;
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
                         int ip,
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

    

    Eigen::MatrixXd J_foot = Eigen::MatrixXd::Zero(6, model_.nv);
    pinocchio::getFrameJacobian(
      model_,
      data_,
      frameId,
      pinocchio::ReferenceFrame::LOCAL,
      J_foot
    );

    Eigen::VectorXd e_p = Eigen::VectorXd::Zero(3);
    e_p.head<3>() = s_p - s_p_hat;
    Eigen::VectorXd e_foot = Eigen::VectorXd::Zero(3);
    e_foot = -(v_prec + C_world_to_base.transpose() * skew(omega_) * s_p + C_world_to_base.transpose() * J_foot.block(0, 6, 3, model_.nv - 6) * vel.tail(model_.nv - 6));
    // e_foot = - J_foot.block(0, 0, 3, model_.nv) * vel;

    int old_rows = e_accum.rows();
    e_accum.conservativeResize(old_rows + e_p.size());
    e_accum.segment(old_rows, e_p.size()) = e_p;
    // e_accum.segment(old_rows + e_p.size(), e_foot.size()) = e_foot;

    H_accum.conservativeResize(old_rows + e_p.size(), NX);
    H_accum.block(old_rows, 0, e_p.size(), NX).setZero();

    H_accum.block<3,3>(old_rows, ir)   = -C_world_to_base;
    H_accum.block<3,3>(old_rows, iphi) = skew(C_world_to_base * (p - r_prec));
    H_accum.block<3,3>(old_rows, ip)   = C_world_to_base;

    // H_accum.block<3,3>(old_rows + e_p.size(), iv) = Eigen::Matrix3d::Identity();
    // H_accum.block<3,3>(old_rows + e_p.size(), iphi) = skew(C_world_to_base.transpose() * skew(omega_) * s_p) 
    //     + skew(C_world_to_base.transpose() * J_foot.block(0, 6, 3, model_.nv - 6) * vel.tail(model_.nv - 6));
  };

  Eigen::MatrixXd H(0, NX);
  Eigen::VectorXd e(0);
  processFoot(model_.getFrameId("left_foot_link"),  pL_, ipL, left_contact,  H, e);
  processFoot(model_.getFrameId("right_foot_link"), pR_, ipR, right_contact, H, e);

  std::cout << "Measurement error: " << e.transpose() << std::endl; 

  if (e.size() == 0)
    return;

  // =====================
  // 4) EKF UPDATE
  // =====================
  int m = e.size();
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m, m) * 1e-100;
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
    // Rodrigues formula: I + sin(θ) K + (1-cos(θ)) K²
    return Eigen::Matrix3d::Identity()
         + std::sin(th) * K
         + (1.0 - std::cos(th)) * K * K;
}

Eigen::Vector3d RightInvariantEKF::logSO3(const Eigen::Matrix3d& R)
{
    // trace → angle
    const double cos_th = std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
    const double th     = std::acos(cos_th);
    if (std::abs(th) < 1e-9)
        return Eigen::Vector3d::Zero();

    // vee of (R - Rᵀ) / (2 sin θ)
    const double s = 0.5 / std::sin(th);
    return th * s * Eigen::Vector3d(R(2,1)-R(1,2),
                                    R(0,2)-R(2,0),
                                    R(1,0)-R(0,1));
}

Eigen::Matrix3d RightInvariantEKF::leftJacobianSO3(const Eigen::Vector3d& phi)
{
    // J_l(φ) = I + ((1-cos θ)/θ²) φ× + ((θ-sin θ)/θ³) φ×²
    const double th = phi.norm();
    if (th < 1e-7)
        return Eigen::Matrix3d::Identity() + 0.5 * skew(phi);

    const Eigen::Matrix3d K = skew(phi);
    return Eigen::Matrix3d::Identity()
         + ((1.0 - std::cos(th)) / (th * th)) * K
         + ((th - std::sin(th)) / (th * th * th)) * K * K;
}

// ============================================================================
//  SE_{N+2}(3) group operations
// ============================================================================

Eigen::Matrix<double, RightInvariantEKF::DIM_X,
                       RightInvariantEKF::DIM_X>
RightInvariantEKF::groupExp(const Eigen::VectorXd& xi) const
{
    // xi layout: [ξᴿ(0:3) | ξᵛ(3:6) | ξᵖ(6:9) | ξᵈ⁰(9:12) | ξᵈ¹(12:15)]
    // (bias part is NOT part of the group, handled separately)

    const Eigen::Vector3d phi = xi.head<3>();                // rotation error
    const Eigen::Matrix3d dR  = expSO3(phi);
    const Eigen::Matrix3d Jl  = leftJacobianSO3(phi);

    // The SE_{N+2}(3) exponential maps each column vector via Jl:
    //   exp(Lg(ξ)).col(k) = Jl · ξ_col_k   for k = v, p, d0, d1, ...
    // (This is the standard SE(3) result generalised to multiple columns.)

    Eigen::Matrix<double, DIM_X, DIM_X> E = Eigen::Matrix<double, DIM_X, DIM_X>::Identity();
    E.block<3,3>(0,0) = dR;

    // number of additional column vectors = DIM_X - 1 - 3 = N_FEET + 1
    // columns: v(1), p(2), d0(3), d1(4), ...
    for (int col = 1; col < DIM_X - 1; ++col) {
        // corresponding xi segment starts at offset 3*(col-1)+3 ... wait:
        // xi[0:3]=φ, xi[3:6]=ξv, xi[6:9]=ξp, xi[9:12]=ξd0, xi[12:15]=ξd1
        // col 1 → ξv = xi[3:6]
        // col 2 → ξp = xi[6:9]
        // col 3 → ξd0 = xi[9:12]
        // col 4 → ξd1 = xi[12:15]
        const int seg_start = 3 * col;  // xi[3], xi[6], xi[9], xi[12]
        E.block<3,1>(0, col) = Jl * xi.segment<3>(seg_start);
    }

    return E;
}

Eigen::VectorXd
RightInvariantEKF::groupLog(
    const Eigen::Matrix<double,DIM_X,DIM_X>& X) const
{
    const Eigen::Matrix3d R   = X.block<3,3>(0,0);
    const Eigen::Vector3d phi = logSO3(R);
    const Eigen::Matrix3d Jl_inv =
        (Eigen::Matrix3d::Identity() - 0.5 * skew(phi)
         + (1.0/(phi.squaredNorm() + 1e-20))
           * (1.0 - 0.5*phi.norm()*std::cos(0.5*phi.norm())/std::sin(0.5*phi.norm()+1e-20))
           * skew(phi) * skew(phi));
    // Simpler: use the approximate inverse Jl⁻¹ ≈ I - φ×/2 for small angles
    // For robustness we use the full expression:
    const double th = phi.norm();
    Eigen::Matrix3d Jlinv;
    if (th < 1e-7) {
        Jlinv = Eigen::Matrix3d::Identity() - 0.5 * skew(phi);
    } else {
        Jlinv = Eigen::Matrix3d::Identity()
              - 0.5 * skew(phi)
              + (1.0/(th*th)) * (1.0 - th*std::cos(0.5*th)/(2.0*std::sin(0.5*th)))
                * skew(phi) * skew(phi);
    }

    // xi layout: [φ | ξv | ξp | ξd0 | ξd1]
    Eigen::VectorXd xi(3 * DIM_X - 3);  // = 3*(N_FEET+3)  wrong — let's be explicit
    // Actually xi has 3*(DIM_X-1) = 3*(N_FEET+3) components
    xi.resize(3 * (DIM_X - 1));
    xi.head<3>() = phi;
    for (int col = 1; col < DIM_X; ++col) {
        xi.segment<3>(3 * col) = Jlinv * X.block<3,1>(0, col);
    }
    return xi;
}

Eigen::Matrix<double, RightInvariantEKF::DIM_X,
                       RightInvariantEKF::DIM_X>
RightInvariantEKF::groupInverse(
    const Eigen::Matrix<double,DIM_X,DIM_X>& X) const
{
    // For X ∈ SE_{N+2}(3):
    //   X⁻¹ = block matrix where:
    //     top-left 3×3  = R^T
    //     col k (k=1..N+1): = -R^T · col_k(X)
    //     bottom-right identity block unchanged
    Eigen::Matrix<double, DIM_X, DIM_X> Xinv =
        Eigen::Matrix<double, DIM_X, DIM_X>::Identity();

    const Eigen::Matrix3d RT = X.block<3,3>(0,0).transpose();
    Xinv.block<3,3>(0,0) = RT;
    for (int col = 1; col < DIM_X; ++col)
        Xinv.block<3,1>(0, col) = -RT * X.block<3,1>(0, col);

    return Xinv;
}

Eigen::Matrix<double, RightInvariantEKF::NR,
                       RightInvariantEKF::NR>
RightInvariantEKF::adjoint(
    const Eigen::Matrix<double,DIM_X,DIM_X>& X) const
{
    // Paper Sec. III-A:
    //   AdX = block matrix of size NR×NR (without bias rows/cols)
    //   but we extend it to include the 6 bias rows as identity.
    //
    // For the SE_{N+2}(3) part the adjoint is (paper Sec. III-A):
    //
    //   AdX = ┌ R    0    0    0    0  ┐   rows: ξᴿ
    //         │ v×R  R    0    0    0  │   rows: ξᵛ
    //         │ p×R  0    R    0    0  │   rows: ξᵖ
    //         │ d₀×R 0    0    R    0  │   rows: ξᵈ⁰
    //         │ d₁×R 0    0    0    R  │   rows: ξᵈ¹
    //         └  0   0    0    0    0  ┘   rows: bias (identity)
    //
    // (Only the 3*(N+3) × 3*(N+3) upper-left block is non-trivial.)

    const Eigen::Matrix3d& R = X.block<3,3>(0,0);
    const Eigen::Vector3d  v = X.block<3,1>(0,1);
    const Eigen::Vector3d  p = X.block<3,1>(0,2);

    // number of contact columns = N_FEET
    // total Lie algebra dimension without bias = 3*(2+N_FEET+1) = 3*(N_FEET+3)
    const int n_lie = 3 * (N_FEET + 3);

    Eigen::Matrix<double, NR, NR> Ad = Eigen::Matrix<double, NR, NR>::Zero();

    // ξᴿ → ξᴿ  :  R
    Ad.block<3,3>(XI_R, XI_R) = R;

    // ξᴿ → ξᵛ  :  v× R
    Ad.block<3,3>(XI_V, XI_R) = skew(v) * R;
    // ξᵛ → ξᵛ  :  R
    Ad.block<3,3>(XI_V, XI_V) = R;

    // ξᴿ → ξᵖ  :  p× R
    Ad.block<3,3>(XI_P, XI_R) = skew(p) * R;
    // ξᵖ → ξᵖ  :  R
    Ad.block<3,3>(XI_P, XI_P) = R;

    // Contact columns
    for (int i = 0; i < N_FEET; ++i) {
        const Eigen::Vector3d di = X.block<3,1>(0, 3 + i);
        const int row = XI_D + 3*i;

        // ξᴿ → ξᵈⁱ  :  dᵢ× R
        Ad.block<3,3>(row, XI_R) = skew(di) * R;
        // ξᵈⁱ → ξᵈⁱ : R
        Ad.block<3,3>(row, row)  = R;
    }

    // Bias rows: identity (biases are Euclidean, not part of the Lie group)
    Ad.block<3,3>(XI_BG, XI_BG) = Eigen::Matrix3d::Identity();
    Ad.block<3,3>(XI_BA, XI_BA) = Eigen::Matrix3d::Identity();

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

    // ── Initial state matrix X ─────────────────────────────────────────────
    // Extract R_WB (world←body) from q_init (quaternion x,y,z,w  body→world)
    const Eigen::Quaterniond q_init_q(q_init[6], q_init[3],
                                      q_init[4], q_init[5]);
    const Eigen::Matrix3d R_WB = q_init_q.normalized().toRotationMatrix();
    const Eigen::Vector3d p_init = q_init.head<3>();

    X_ = Eigen::Matrix<double, DIM_X, DIM_X>::Identity();
    X_.block<3,3>(0,0) = R_WB;
    X_.block<3,1>(0,3) = p_init;
    // velocity initialised to zero — already in X_ via Identity

    // ── R_imu_to_body ──────────────────────────────────────────────────────
    pinocchio::forwardKinematics(model_, data_, q_init);
    pinocchio::updateFramePlacements(model_, data_);

    const int imu_id = model_.getFrameId("imu_in_torso");
    // R_world_imu = data_.oMf[imu_id].rotation()
    // R_body_imu = R_world_body^T * R_world_imu
    //            = R_WB^T * R_world_imu
    R_imu_to_body_ = R_WB.transpose() * data_.oMf[imu_id].rotation();

    // ── Initial foot positions ─────────────────────────────────────────────
    for (int i = 0; i < N_FEET; ++i) {
        const int fid = model_.getFrameId(feet_[i].frame_name);
        const Eigen::Vector3d p_foot = data_.oMf[fid].translation();
        X_.block<3,1>(0, 4 + feet_[i].contact_idx) = p_foot;
    }


    // ── Initial covariance P ──────────────────────────────────────────────
    P_.setZero();
    P_.block<3,3>(XI_R,  XI_R)  = 0.01  * Eigen::Matrix3d::Identity();
    P_.block<3,3>(XI_V,  XI_V)  = 0.01  * Eigen::Matrix3d::Identity();
    P_.block<3,3>(XI_P,  XI_P)  = 0.01  * Eigen::Matrix3d::Identity();
    for (int i = 0; i < N_FEET; ++i)
        P_.block<3,3>(XI_D+3*i, XI_D+3*i) = 0.01 * Eigen::Matrix3d::Identity();
    P_.block<3,3>(XI_BG, XI_BG) = 0.0001 * Eigen::Matrix3d::Identity();
    P_.block<3,3>(XI_BA, XI_BA) = 0.0001 * Eigen::Matrix3d::Identity();

    // ── Continuous process noise Qc ────────────────────────────────────────
    // In the basis of ξ (error state).  Before AdX multiplication.
    // The mapping follows paper eq. 8 and Sec. III-B:
    //   Q̂ = AdX * Cov(w) * AdXᵀ
    // where Cov(w) = blkdiag(Σᵍ, Σᵃ, 0, Σᵛ_rot_for_each_contact)
    // We store Qc = Cov(w) here (in the noise-input basis).
    // The actual per-step Q̂ is computed in filter().
    //
    // Noise vector w = [wᵍ(3) | wᵃ(3) | 0(3) | hR·wᵛ(3)×N_FEET | wᵇᵍ(3) | wᵇᵃ(3)]
    // In ξ basis this maps as (paper eqs 5,8):
    //   noise on ξᴿ  ← wᵍ   (gyro)
    //   noise on ξᵛ  ← wᵃ   (accel)
    //   noise on ξᵖ  ← 0    (position has no direct noise)
    //   noise on ξᵈⁱ ← hR·wᵛ (contact slip)
    //   noise on δbᵍ ← wᵇᵍ  (gyro bias RW)
    //   noise on δbᵃ ← wᵇᵃ  (accel bias RW)
    //
    // The continuous Qc (in ξ basis, before AdX) is therefore:
    const double sg2 = noise_.gyro_noise    * noise_.gyro_noise;
    const double sa2 = noise_.accel_noise   * noise_.accel_noise;
    const double sv2 = noise_.contact_noise * noise_.contact_noise;
    const double sbg2= noise_.gyro_bias_rw  * noise_.gyro_bias_rw;
    const double sba2= noise_.accel_bias_rw * noise_.accel_bias_rw;

    Qc_.setZero();
    Qc_.block<3,3>(XI_R,  XI_R)  = sg2 * Eigen::Matrix3d::Identity();
    Qc_.block<3,3>(XI_V,  XI_V)  = sa2 * Eigen::Matrix3d::Identity();
    // position noise = 0 (no direct process noise on p)
    for (int i = 0; i < N_FEET; ++i)
        Qc_.block<3,3>(XI_D+3*i, XI_D+3*i) = sv2 * Eigen::Matrix3d::Identity();
    Qc_.block<3,3>(XI_BG, XI_BG) = sbg2 * Eigen::Matrix3d::Identity();
    Qc_.block<3,3>(XI_BA, XI_BA) = sba2 * Eigen::Matrix3d::Identity();
}

// ============================================================================
//  addContact / removeContact  (paper Sec. V)
// ============================================================================

void RightInvariantEKF::addContact(int foot_idx,
                                   const Eigen::VectorXd& joint_pos)
{
    // Reset contact position from FK using current base estimate
    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head<3>()     = X_.block<3,1>(0,2);  // current p
    // Build quaternion from current R_WB
    const Eigen::Quaterniond q_cur(X_.block<3,3>(0,0));
    q_pin.segment<4>(3) = q_cur.coeffs();
    q_pin.tail(joint_pos.size()) = joint_pos;

    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);

    const int fid = model_.getFrameId(feet_[foot_idx].frame_name);
    const int ci  = feet_[foot_idx].contact_idx;

    // Set contact position in world frame
    X_.block<3,1>(0, 3 + ci) = data_.oMf[fid].translation();

    // Reset covariance block for this contact point
    const int row = XI_D + 3 * foot_idx;
    P_.block<3, NR>(row, 0).setZero();
    P_.block<NR, 3>(0, row).setZero();
    P_.block<3,3>(row, row) = 0.01 * Eigen::Matrix3d::Identity();

    active_contact_[foot_idx] = true;
}

void RightInvariantEKF::removeContact(int foot_idx)
{
    // Inflate the contact point process noise so P grows fast
    // (the contact position becomes unobservable when not in contact)
    active_contact_[foot_idx] = false;
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
    // ── Contact management ────────────────────────────────────────────────
    for (int i = 0; i < N_FEET; ++i) {
        if (contact[i] && !active_contact_[i])
            addContact(i, joint_pos);
        else if (!contact[i] && active_contact_[i])
            removeContact(i);
    }

    // =========================================================================
    // 0)  IMU pre-processing
    //     Rotate raw measurements to body frame and remove biases.
    // =========================================================================

    // Bias-compensated IMU in body frame
    const Eigen::Vector3d omega_b = R_imu_to_body_ * gyro_meas  - bg_;
    const Eigen::Vector3d f_b     = R_imu_to_body_ * acc_meas   - ba_;

    // Current rotation R = R_WB (world←body)
    const Eigen::Matrix3d& R = X_.block<3,3>(0,0);

    // =========================================================================
    // 1)  NOMINAL STATE PROPAGATION  (paper eq. 4 and 7)
    //
    // Continuous dynamics (paper eq. 4):
    //   Ṙ = R (ω̃)×
    //   v̇ = R ã + g
    //   ṗ = v
    //   ḋᵢ = 0  (contact position constant)
    //
    // Discrete integration with zero-order hold on inputs:
    //   R_{k+1} = R_k · expSO3(ω_b · Δt)
    //   v_{k+1} = v_k + (R_k · f_b + g) · Δt
    //   p_{k+1} = p_k + v_k · Δt + ½ (R_k · f_b + g) · Δt²
    //   d_{k+1} = d_k
    // =========================================================================

    const Eigen::Vector3d a_world = R * f_b;
    const Eigen::Vector3d v_k     = X_.block<3,1>(0,1);
    const Eigen::Vector3d p_k     = X_.block<3,1>(0,2);

    // Update state matrix
    X_.block<3,3>(0,0) = R * expSO3(omega_b * dt_);
    X_.block<3,1>(0,1) = v_k + a_world * dt_;
    X_.block<3,1>(0,2) = p_k + v_k * dt_ + 0.5 * a_world * dt_ * dt_;
    // contact columns unchanged
    X_.block<3,3>(0,0) = X_.block<3,3>(0,0); // already updated
    // Normalise rotation
    Eigen::Matrix3d Rnew = X_.block<3,3>(0,0);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rnew,
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    X_.block<3,3>(0,0) = svd.matrixU()
                        * Eigen::Matrix3d(svd.singularValues().cwiseSign().asDiagonal())
                        * svd.matrixV().transpose();

    // =========================================================================
    // 2)  COVARIANCE PROPAGATION  (paper eq. 7-8)
    //
    // Continuous Riccati: dP/dt = A P + P Aᵀ + Q̂
    // where:
    //   A = ┌ 0      0    0    0  ┐  (paper eq. 8, time-invariant!)
    //       │ g×     0    0    0  │
    //       │ 0      I    0    0  │
    //       │ 0      0    0    0  │
    //       └ 0      0    0    0  ┘
    //   (rows/cols: ξᴿ, ξᵛ, ξᵖ, ξᵈ, δb)
    //   (A has only two non-zero blocks: A(ξᵛ,ξᴿ)=g×  and  A(ξᵖ,ξᵛ)=I)
    //
    //   Q̂ = AdX̂ · Cov(w) · AdX̂ᵀ
    //
    // Discrete approximation:
    //   Φ = expm(A·Δt)  — exact for nilpotent A (paper's observability remark)
    //   P_{k+1} = Φ P Φᵀ + Q̂·Δt
    //
    // Because A is nilpotent of degree 3, expm(A·Δt) is a polynomial:
    //   Φ = I + A·Δt + ½ A²·Δt²
    // (paper observability section: the state transition matrix is exact polynomial)
    // =========================================================================

    // Build Φ (paper's Φ = expm(At·Δt), exact polynomial)
    Eigen::Matrix<double, NR, NR> Phi =
        Eigen::Matrix<double, NR, NR>::Identity();

    // A·Δt  blocks:
    //   (ξᵛ, ξᴿ):  g× · Δt
    //   (ξᵖ, ξᵛ):  I  · Δt
    const Eigen::Matrix3d g_cross_dt = skew(g_) * dt_;
    Phi.block<3,3>(XI_V, XI_R) += g_cross_dt;
    Phi.block<3,3>(XI_P, XI_V) += Eigen::Matrix3d::Identity() * dt_;

    // A²·Δt²/2  blocks:
    //   A² = A·A.  Only non-zero: (ξᵖ, ξᴿ) = I * g× * Δt² (from ξᵖ←ξᵛ←ξᴿ)
    Phi.block<3,3>(XI_P, XI_R) += 0.5 * skew(g_) * dt_ * dt_;

    // Q̂ = AdX̂ · Qc · AdX̂ᵀ · Δt
    // (inflate contact noise for non-active feet)
    Eigen::Matrix<double, NR, NR> Qc_step = Qc_;
    for (int i = 0; i < N_FEET; ++i) {
        if (!active_contact_[i]) {
            // large noise when foot is not in contact
            Qc_step.block<3,3>(XI_D+3*i, XI_D+3*i) =
                100.0 * Eigen::Matrix3d::Identity();
        }
    }


    const Eigen::Matrix<double, NR, NR> AdX  = adjoint(X_);
    const Eigen::Matrix<double, NR, NR> Qhat = AdX * Qc_step * AdX.transpose();

    P_ = Phi * P_ * Phi.transpose() + Qhat * dt_;

    // =========================================================================
    // 3)  FORWARD KINEMATICS
    //     Evaluate FK at current state estimate + measured joint angles.
    // =========================================================================

    const Eigen::Matrix3d R_new = X_.block<3,3>(0,0);
    const Eigen::Quaterniond q_cur(R_new);

    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head<3>()      = X_.block<3,1>(0,2);   // position
    q_pin.segment<4>(3)  = q_cur.coeffs();        // (x,y,z,w)
    q_pin.tail(joint_pos.size()) = joint_pos;

    data_ = pinocchio::Data(model_);
    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_, q_pin);

    // =========================================================================
    // 4)  MEASUREMENT UPDATE  (paper Sec. III-C, eqs. 11-14)
    //
    // Right-invariant FK measurement model:
    //   Yₜ = Xₜ⁻¹ b + Vₜ
    //
    // where for each contact i:
    //   b = [0; 0; 1; -1]  (4-vector in homogeneous coords)
    //   Yₜ = [hp(α̃ₜ); 0; 1; -1]
    //   hp(α̃ₜ) = Rᵀ(dᵢ - p)  ← foot position in body frame from state
    //
    // The innovation is (paper eq. 13 linearised):
    //   z = X̂ₜ Yₜ  (first 3 components, after projection Π)
    //     = R̂ hp(α̃) + p̂ - dᵢ_hat
    //
    // Measurement Jacobian (paper eq. 13):
    //   H = [0  0  -I  I]  (3 × NR per contact)
    //
    // Measurement noise (paper eq. 14):
    //   N̂ = R̂ Jv Σα Jvᵀ R̂ᵀ
    // =========================================================================

    // Accumulate H, z, N over all active contacts
    Eigen::MatrixXd H_all(0, NR);
    Eigen::VectorXd z_all(0);
    Eigen::MatrixXd N_all(0, 0);

    for (int i = 0; i < N_FEET; ++i) {
        if (!active_contact_[i]) continue;

        const int ci  = feet_[i].contact_idx;
        const int fid = model_.getFrameId(feet_[i].frame_name);

        // FK: foot position relative to body, in body frame
        // hp(α̃) = R^T (dᵢ - p)   from state  (paper eq. 11)
        // measured hp from FK: B_p_BC = R^T * (p_foot_world - p_base_world)
        const Eigen::Vector3d p_foot_world = data_.oMf[fid].translation();
        const Eigen::Vector3d p_base_world = X_.block<3,1>(0,2);
        const Eigen::Vector3d d_i          = X_.block<3,1>(0, 3 + ci);
        const Eigen::Matrix3d Rhat         = X_.block<3,3>(0,0);

        // Measured hp (from FK encoder reading)
        const Eigen::Vector3d hp_meas = Rhat.transpose() * (p_foot_world - p_base_world);

        // Innovation z = X̂ Y - b  (paper eq. 14, first 3 components)
        // X̂ Y = X̂ (X̂⁻¹ b + V) = b + X̂ V
        // We compute X̂ Yₜ directly:
        //   Yₜ = [hp_meas; 0; 1; -1]  (extended homogeneous vector)
        //   (X̂ Yₜ)_top3 = R̂ hp_meas + v̂·0 + p̂·1 + dᵢ·(-1)
        //                = R̂ hp_meas + p̂ - dᵢ
        const Eigen::Vector3d z_i =
            Rhat * hp_meas + p_base_world - d_i;
        // (this equals R̂ R̂ᵀ (p_foot - p) + p - dᵢ
        //            = p_foot - p + p - dᵢ
        //            = p_foot - dᵢ
        //  which is exactly the world-frame foot position error vs state estimate)

        // Measurement Jacobian (paper eq. 13):
        //   H_i = [0₃  0₃  -I₃  ...  I₃  ...  0₃  0₃]
        //          ξᴿ  ξᵛ   ξᵖ       ξᵈⁱ      δbᵍ δbᵃ
        Eigen::MatrixXd Hi = Eigen::MatrixXd::Zero(3, NR);
        Hi.block<3,3>(0, XI_P)      = -Eigen::Matrix3d::Identity();
        Hi.block<3,3>(0, XI_D+3*i)  =  Eigen::Matrix3d::Identity();
        // (paper has H = [0  0  -I  I] for the contact block)

        // Measurement noise: N̂ = R̂ Jv Σα Jvᵀ R̂ᵀ  (paper eq. 14)
        const int n_joints = static_cast<int>(joint_pos.size());
        Eigen::MatrixXd J_full = Eigen::MatrixXd::Zero(6, model_.nv);

        pinocchio::getFrameJacobian(model_, data_, fid,
                                    pinocchio::LOCAL_WORLD_ALIGNED, J_full);
        // Take linear (top 3) rows, joint columns only (rightmost n_joints cols)
        const Eigen::MatrixXd Jv =
            J_full.topRows<3>().rightCols(n_joints);  // 3 × n_j, world frame

        const double se2 = noise_.encoder_noise * noise_.encoder_noise;
        // const Eigen::Matrix3d Ni =
        //     Rhat * (Jv * Rhat.transpose()) * se2 * (Jv * Rhat.transpose()).transpose() * Rhat.transpose();
        // Simplified: N̂ = R̂ Jv_body Σα Jv_bodyᵀ R̂ᵀ
        // where Jv_body = R̂ᵀ Jv (world→body rotation of Jacobian rows)
        // = (R̂ᵀ Jv) Σα (R̂ᵀ Jv)ᵀ rotated back to world frame
        // which equals: Jv Σα Jvᵀ  (rotation cancels)
        // Paper actually uses: N̂ = R̂ Jv Σα Jvᵀ R̂ᵀ where Jv is in body frame
        // Let's be precise:
        // Jv_body = top 3 rows of LOCAL Jacobian (body frame linear velocity)
        Eigen::MatrixXd J_local = Eigen::MatrixXd::Zero(6, model_.nv);
        pinocchio::getFrameJacobian(model_, data_, fid,
                                    pinocchio::LOCAL, J_local);

        const Eigen::MatrixXd Jv_body =
            J_local.topRows<3>().rightCols(n_joints);  // 3 × n_j, body frame

        const Eigen::Matrix3d Ni_correct =
            Rhat * (Jv_body * se2 * Jv_body.transpose()) * Rhat.transpose();

        // Accumulate
        const int old = static_cast<int>(z_all.rows());
        z_all.conservativeResize(old + 3);
        z_all.segment(old, 3) = z_i;

        H_all.conservativeResize(old + 3, NR);
        H_all.block(old, 0, 3, NR) = Hi;

        N_all.conservativeResize(old + 3, old + 3);
        N_all.block(old, 0,    3, old).setZero();
        N_all.block(0,   old, old, 3  ).setZero();
        N_all.block<3,3>(old, old) = Ni_correct;
    }

    if (z_all.size() == 0)
        return;  // no active contacts, pure propagation

    // =========================================================================
    // 5)  KALMAN UPDATE  (paper eq. 14)
    //
    //   S  = H P Hᵀ + N̂
    //   K  = P Hᵀ S⁻¹
    //   ξ⁺ = K z                    (innovation in the ξ basis)
    //   X̂⁺ = exp(Kg · ξ⁺) · X̂      (right-invariant update, paper eq. 14)
    //   P⁺ = (I - K H) P             (Joseph form below for stability)
    // =========================================================================

    const int m = static_cast<int>(z_all.rows());

    const Eigen::MatrixXd S = H_all * P_ * H_all.transpose() + N_all;
    const Eigen::MatrixXd K = P_ * H_all.transpose() * S.inverse();

    // Correction in ξ space
    const Eigen::VectorXd xi_corr = K * z_all;   // dim NR

    // Extract Lie-algebra part of xi_corr (first 3*(N_FEET+3) components)
    const int n_lie = 3 * (N_FEET + 3);

    // Build the correction Lie algebra element and apply group exponential
    // paper eq. 14:  X̂⁺ = exp(Lg(ξ_lie)) · X̂
    Eigen::VectorXd xi_lie(n_lie);
    xi_lie.head<3>()                   = xi_corr.segment<3>(XI_R);
    xi_lie.segment<3>(3)               = xi_corr.segment<3>(XI_V);
    xi_lie.segment<3>(6)               = xi_corr.segment<3>(XI_P);
    for (int i = 0; i < N_FEET; ++i){
        xi_lie.segment<3>(9 + 3*i)     = xi_corr.segment<3>(XI_D + 3*i);
    }

    const Eigen::Matrix<double,DIM_X,DIM_X> dX = groupExp(xi_lie);
    X_ = dX * X_;

    // Re-normalise rotation after update
    Eigen::JacobiSVD<Eigen::Matrix3d> svd2(X_.block<3,3>(0,0),
        Eigen::ComputeFullU | Eigen::ComputeFullV);
    X_.block<3,3>(0,0) = svd2.matrixU()
                        * Eigen::Matrix3d(svd2.singularValues().cwiseSign().asDiagonal())
                        * svd2.matrixV().transpose();

    // Bias update (Euclidean part, paper Sec. IV)
    bg_ += xi_corr.segment<3>(XI_BG);
    ba_ += xi_corr.segment<3>(XI_BA);

    // Covariance update: Joseph stabilised form
    const Eigen::Matrix<double, NR, NR> I_mat =
        Eigen::Matrix<double, NR, NR>::Identity();
    const Eigen::MatrixXd IKH = I_mat - K * H_all;
    P_ = IKH * P_ * IKH.transpose() + K * N_all * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
}

} // namespace state_filtering
