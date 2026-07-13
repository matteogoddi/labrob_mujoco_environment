#include <StateEstimator.hpp>

#include <pinocchio/spatial/explog.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <utils.hpp>

namespace labrob {

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
    Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
    S(2,2) = (svd.matrixU() * svd.matrixV().transpose()).determinant();
    return svd.matrixU() * S * svd.matrixV().transpose();
}

// ============================================================================
//  SE_{N+2}(3) group operations
// ============================================================================

Eigen::Matrix<double, RightInvariantEKF::DIM_X,
                       RightInvariantEKF::DIM_X>
RightInvariantEKF::groupExp(
    const Eigen::Matrix<double,3*(N_FEET+3),1>& xi) const
{
    const Eigen::Vector3d phi = xi.template head<3>();
    const Eigen::Matrix3d dR  = expSO3(phi);
    const Eigen::Matrix3d Jl  = leftJacobianSO3(phi);

    Eigen::Matrix<double,DIM_X,DIM_X> E =
        Eigen::Matrix<double,DIM_X,DIM_X>::Identity();

    E.template block<3,3>(0,0) = dR;

    for (int c = COL_V; c < DIM_X; ++c) {
        const int xi_off = 3*(c - 2);
        E.template block<3,1>(0,c) = Jl * xi.template segment<3>(xi_off);
    }

    return E;
}

Eigen::Matrix<double, RightInvariantEKF::DIM_X,
                       RightInvariantEKF::DIM_X>
RightInvariantEKF::groupInverse(
    const Eigen::Matrix<double,DIM_X,DIM_X>& X) const
{
    Eigen::Matrix<double,DIM_X,DIM_X> Xinv =
        Eigen::Matrix<double,DIM_X,DIM_X>::Identity();

    const Eigen::Matrix3d RT = X.template block<3,3>(0,0).transpose();
    Xinv.template block<3,3>(0,0) = RT;

    for (int c = COL_V; c < DIM_X; ++c)
        Xinv.template block<3,1>(0,c) = -RT * X.template block<3,1>(0,c);

    return Xinv;
}

Eigen::Matrix<double, RightInvariantEKF::NR,
                       RightInvariantEKF::NR>
RightInvariantEKF::adjoint(
    const Eigen::Matrix<double,DIM_X,DIM_X>& X) const
{
    Eigen::Matrix<double,NR,NR> Ad = Eigen::Matrix<double,NR,NR>::Zero();

    const Eigen::Matrix3d& R = X.template block<3,3>(0,0);
    const Eigen::Vector3d  v = X.template block<3,1>(0,COL_V);
    const Eigen::Vector3d  p = X.template block<3,1>(0,COL_P);

    Ad.template block<3,3>(XI_R, XI_R) = R;

    Ad.template block<3,3>(XI_V, XI_R) = skew(v) * R;
    Ad.template block<3,3>(XI_V, XI_V) = R;

    Ad.template block<3,3>(XI_P, XI_R) = skew(p) * R;
    Ad.template block<3,3>(XI_P, XI_P) = R;

    for (int i = 0; i < N_FEET; ++i) {
        const Eigen::Vector3d di = X.template block<3,1>(0, COL_D+i);
        const int row = XI_D + 3*i;
        Ad.template block<3,3>(row, XI_R) = skew(di) * R;
        Ad.template block<3,3>(row, row)  = R;
    }

    Ad.template block<3,3>(XI_BG, XI_BG) = Eigen::Matrix3d::Identity();
    Ad.template block<3,3>(XI_BA, XI_BA) = Eigen::Matrix3d::Identity();

    return Ad;
}

// ============================================================================
//  RightInvariantEKF constructor
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

    const Eigen::Quaterniond q0(q_init[6], q_init[3], q_init[4], q_init[5]);
    const Eigen::Matrix3d R_WB = q0.normalized().toRotationMatrix();

    X_ = Eigen::Matrix<double,DIM_X,DIM_X>::Identity();
    X_.template block<3,3>(0,0)    = R_WB;
    X_.template block<3,1>(0,COL_P)= q_init.head<3>();

    pinocchio::forwardKinematics(model_, data_, q_init);
    pinocchio::updateFramePlacements(model_, data_);
    for (int i = 0; i < N_FEET; ++i) {
        const int fid = model_.getFrameId(feet_[i].frame_name);
        X_.template block<3,1>(0, COL_D + feet_[i].contact_idx)
            = data_.oMf[fid].translation();
    }

    P_.setZero();
    P_.template block<3,3>(XI_R,  XI_R)  = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_V,  XI_V)  = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_P,  XI_P)  = 0.01 * Eigen::Matrix3d::Identity();
    for (int i = 0; i < N_FEET; ++i)
        P_.template block<3,3>(XI_D+3*i, XI_D+3*i) = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_BG, XI_BG) = 1e-4 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_BA, XI_BA) = 1e-4 * Eigen::Matrix3d::Identity();

    const double sg2  = noise_.gyro_noise    * noise_.gyro_noise;
    const double sa2  = noise_.accel_noise   * noise_.accel_noise;
    const double sv2  = noise_.contact_noise * noise_.contact_noise;
    const double sbg2 = noise_.gyro_bias_rw  * noise_.gyro_bias_rw;
    const double sba2 = noise_.accel_bias_rw * noise_.accel_bias_rw;

    Qc_.setZero();
    Qc_.template block<3,3>(XI_R,  XI_R)  = sg2  * Eigen::Matrix3d::Identity();
    Qc_.template block<3,3>(XI_V,  XI_V)  = sa2  * Eigen::Matrix3d::Identity();
    for (int i = 0; i < N_FEET; ++i)
        Qc_.template block<3,3>(XI_D+3*i, XI_D+3*i) = sv2  * Eigen::Matrix3d::Identity();
    Qc_.template block<3,3>(XI_BG, XI_BG) = sbg2 * Eigen::Matrix3d::Identity();
    Qc_.template block<3,3>(XI_BA, XI_BA) = sba2 * Eigen::Matrix3d::Identity();
}

// ============================================================================
//  initialize
// ============================================================================
void RightInvariantEKF::initialize(const Eigen::VectorXd& q_init,
                                   const Eigen::VectorXd& joint_pos)
{
    const Eigen::Quaterniond q0(q_init[6], q_init[3], q_init[4], q_init[5]);
    X_.template block<3,3>(0,0)    = q0.normalized().toRotationMatrix();
    X_.template block<3,1>(0,COL_P)= q_init.head<3>();
    X_.template block<3,1>(0,COL_V).setZero();

    bg_.setZero();
    ba_.setZero();
    omega_b_.setZero();

    P_.setZero();
    P_.template block<3,3>(XI_R,  XI_R)  = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_V,  XI_V)  = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_P,  XI_P)  = 0.01 * Eigen::Matrix3d::Identity();
    for (int i = 0; i < N_FEET; ++i)
        P_.template block<3,3>(XI_D+3*i, XI_D+3*i) = 0.01 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_BG, XI_BG) = 1e-4 * Eigen::Matrix3d::Identity();
    P_.template block<3,3>(XI_BA, XI_BA) = 1e-4 * Eigen::Matrix3d::Identity();

    active_contact_.fill(false);
    for (int i = 0; i < N_FEET; ++i)
        addContact(i, joint_pos);
}

// ============================================================================
//  addContact / removeContact
// ============================================================================
void RightInvariantEKF::addContact(int foot_idx,
                                   const Eigen::VectorXd& joint_pos)
{
    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head<3>()     = getPosition();
    q_pin.segment<4>(3) = getQuaternion().coeffs();
    q_pin.tail(joint_pos.size()) = joint_pos;

    pinocchio::forwardKinematics(model_, data_, q_pin);
    pinocchio::updateFramePlacements(model_, data_);

    const int fid = model_.getFrameId(feet_[foot_idx].frame_name);
    const int ci  = feet_[foot_idx].contact_idx;

    X_.template block<3,1>(0, COL_D + ci) = data_.oMf[fid].translation();

    const int row = XI_D + 3*ci;
    P_.block(row, 0,   3, NR).setZero();
    P_.block(0,   row, NR, 3).setZero();
    P_.template block<3,3>(row, row) = 0.01 * Eigen::Matrix3d::Identity();

    active_contact_[foot_idx] = true;
}

void RightInvariantEKF::removeContact(int foot_idx)
{
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
    for (int i = 0; i < N_FEET; ++i) {
        if (contact[i] && !active_contact_[i])       addContact(i, joint_pos);
        else if (!contact[i] && active_contact_[i])  removeContact(i);
    }

    // 0)  IMU pre-processing
    const Eigen::Vector3d omega_b = gyro_meas - bg_;
    const Eigen::Vector3d f_body  = acc_meas  - ba_;
    omega_b_ = omega_b;

    const Eigen::Matrix3d R_WB = X_.template block<3,3>(0,0);

    // 1)  Nominal state propagation
    const Eigen::Vector3d v_k     = X_.template block<3,1>(0,COL_V);
    const Eigen::Vector3d p_k     = X_.template block<3,1>(0,COL_P);
    const Eigen::Vector3d a_world = R_WB * f_body + g_;

    X_.template block<3,3>(0,0)    = projectToSO3(R_WB * expSO3(omega_b * dt_));
    X_.template block<3,1>(0,COL_V)= v_k + a_world * dt_;
    X_.template block<3,1>(0,COL_P)= p_k + v_k*dt_ + 0.5*a_world*dt_*dt_;

    const Eigen::Matrix3d& R_new = X_.template block<3,3>(0,0);

    // 2)  Covariance propagation
    Eigen::Matrix<double,NR,NR> Phi = Eigen::Matrix<double,NR,NR>::Identity();

    Phi.template block<3,3>(XI_V, XI_R)  = skew(g_)                   * dt_;
    Phi.template block<3,3>(XI_P, XI_V)  = Eigen::Matrix3d::Identity() * dt_;
    Phi.template block<3,3>(XI_P, XI_R)  = 0.5 * skew(g_) * dt_ * dt_;

    Phi.template block<3,3>(XI_R, XI_BG) = -R_WB                      * dt_;
    Phi.template block<3,3>(XI_V, XI_BG) = -skew(v_k) * R_WB          * dt_;
    Phi.template block<3,3>(XI_V, XI_BA) = -R_WB                      * dt_;
    Phi.template block<3,3>(XI_P, XI_BG) = -skew(p_k) * R_WB          * dt_;
    Phi.template block<3,3>(XI_P, XI_BA) = -0.5 * R_WB                * dt_ * dt_;
    for (int i = 0; i < N_FEET; ++i) {
        const Eigen::Vector3d d_i = X_.template block<3,1>(0, COL_D + i);
        Phi.template block<3,3>(XI_D + 3*i, XI_BG) = -skew(d_i) * R_WB * dt_;
    }

    Eigen::Matrix<double,NR,NR> Qc_step = Qc_;
    for (int i = 0; i < N_FEET; ++i) {
        if (!active_contact_[i])
            Qc_step.template block<3,3>(XI_D+3*i, XI_D+3*i)
                = 1.0 * Eigen::Matrix3d::Identity();
    }

    const Eigen::Matrix<double,NR,NR> AdX  = adjoint(X_);
    const Eigen::Matrix<double,NR,NR> Qhat = AdX * Qc_step * AdX.transpose() * dt_;

    P_ = Phi * P_ * Phi.transpose() + Qhat;

    // 3)  Forward kinematics
    const int n_joints = static_cast<int>(joint_pos.size());

    Eigen::VectorXd q_pin = Eigen::VectorXd::Zero(model_.nq);
    q_pin.head<3>()      = getPosition();
    q_pin.segment<4>(3)  = getQuaternion().coeffs();
    q_pin.tail(n_joints) = joint_pos;

    Eigen::VectorXd v_pin = Eigen::VectorXd::Zero(model_.nv);
    v_pin.head<3>()      = R_new.transpose() * getVelocity();
    v_pin.segment<3>(3)  = omega_b;
    v_pin.tail(n_joints) = joint_vel;

    data_ = pinocchio::Data(model_);
    pinocchio::forwardKinematics(model_, data_, q_pin, v_pin);
    pinocchio::updateFramePlacements(model_, data_);
    pinocchio::computeJointJacobians(model_, data_, q_pin);

    // 4)  Measurement model (right-invariant FK constraints)
    Eigen::MatrixXd H_all(0, NR);
    Eigen::VectorXd z_all(0);
    Eigen::MatrixXd N_all(0, 0);

    for (int i = 0; i < N_FEET; ++i) {
        if (!active_contact_[i]) continue;

        const int ci  = feet_[i].contact_idx;
        const int fid = model_.getFrameId(feet_[i].frame_name);

        const Eigen::Vector3d p_foot_fk = data_.oMf[fid].translation();
        const Eigen::Vector3d d_i = X_.template block<3,1>(0, COL_D + ci);
        const Eigen::Vector3d z_i = p_foot_fk - d_i;

        Eigen::Matrix<double,3,NR> Hi;
        Hi.setZero();
        Hi.template block<3,3>(0, XI_P)        = -Eigen::Matrix3d::Identity();
        Hi.template block<3,3>(0, XI_D + 3*ci) =  Eigen::Matrix3d::Identity();

        Eigen::MatrixXd J_full = Eigen::MatrixXd::Zero(6, model_.nv);
        pinocchio::getFrameJacobian(model_, data_, fid,
                                    pinocchio::LOCAL_WORLD_ALIGNED, J_full);
        const Eigen::MatrixXd Jv_world =
            J_full.topRows<3>().rightCols(n_joints);

        const double se2 = noise_.encoder_noise * noise_.encoder_noise;
        const Eigen::Matrix3d Ni = Jv_world * se2 * Jv_world.transpose();

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
        return;

    // 5)  Kalman update
    const Eigen::MatrixXd S = H_all * P_ * H_all.transpose() + N_all;
    const Eigen::MatrixXd K = P_ * H_all.transpose() * S.inverse();

    const Eigen::VectorXd xi_corr = K * z_all;

    Eigen::Matrix<double,3*(N_FEET+3),1> xi_lie;
    xi_lie.template head<3>()      = xi_corr.template segment<3>(XI_R);
    xi_lie.template segment<3>(3)  = xi_corr.template segment<3>(XI_V);
    xi_lie.template segment<3>(6)  = xi_corr.template segment<3>(XI_P);
    for (int i = 0; i < N_FEET; ++i)
        xi_lie.template segment<3>(9 + 3*i) = xi_corr.template segment<3>(XI_D + 3*i);

    X_ = groupExp(xi_lie) * X_;
    X_.template block<3,3>(0,0) = projectToSO3(X_.template block<3,3>(0,0));

    bg_ += xi_corr.template segment<3>(XI_BG);
    ba_ += xi_corr.template segment<3>(XI_BA);

    const Eigen::Matrix<double,NR,NR> I_mat =
        Eigen::Matrix<double,NR,NR>::Identity();
    const Eigen::MatrixXd IKH = I_mat - K * H_all;
    P_ = IKH * P_ * IKH.transpose() + K * N_all * K.transpose();
    P_ = 0.5 * (P_ + P_.transpose());
}

// ============================================================================
//  StateEstimator
// ============================================================================

static pinocchio::Model buildModelFromUrdf(const std::string& urdf_path)
{
    pinocchio::Model full_model;
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), full_model);
    return pinocchio::buildReducedModel(full_model, {}, pinocchio::neutral(full_model));
}

StateEstimator::StateEstimator(const std::string& urdf_path,
                               double dt,
                               const NoiseParams& noise)
    : StateEstimator(buildModelFromUrdf(urdf_path), dt, noise)
{}

StateEstimator::StateEstimator(const pinocchio::Model& model,
                               double dt,
                               const NoiseParams& noise)
    : model_(model), dt_(dt)
{
    njnt_ = model_.nv - 6;

    Eigen::VectorXd q0 = pinocchio::neutral(model_);

    std::array<RightInvariantEKF::FootConfig, 2> feet = {{
        {"left_foot_link",  0},
        {"right_foot_link", 1}
    }};
    ri_ekf_ = std::make_unique<RightInvariantEKF>(model_, q0, dt_, feet, noise);
}

void StateEstimator::activate(const RobotState& robot_state,
                              const Eigen::VectorXd& /*q_joints_unused*/)
{
    if (active_) return;
    active_ = true;

    Eigen::VectorXd q_joints_pin(njnt_);
    for (int i = 0; i < njnt_; ++i)
        q_joints_pin(i) = robot_state.joint_state.at(model_.names[i + 2]).pos;

    {
        pinocchio::Data data_init(model_);
        Eigen::VectorXd q_fk = pinocchio::neutral(model_);
        q_fk.tail(njnt_) = q_joints_pin;
        pinocchio::forwardKinematics(model_, data_init, q_fk);
        pinocchio::updateFramePlacements(model_, data_init);

        auto fk_foot = [&](const std::string& frame) -> std::pair<Eigen::Matrix3d, Eigen::Vector3d> {
            const auto& T = data_init.oMf[model_.getFrameId(frame)];
            Eigen::Matrix3d R_wb = T.rotation().transpose();
            Eigen::Vector3d p_wb = -R_wb * T.translation();
            return {R_wb, p_wb};
        };

        auto [R_l, p_l] = fk_foot("left_foot_link");
        auto [R_r, p_r] = fk_foot("right_foot_link");

        Eigen::Vector3d p_wb(0.0, 0.0, 0.5 * (p_l.z() + p_r.z()));
        Eigen::Matrix3d R_avg = 0.5 * (R_l + R_r);
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R_avg, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_wb = svd.matrixU()
                             * Eigen::DiagonalMatrix<double,3>(1, 1, (svd.matrixU()*svd.matrixV().transpose()).determinant())
                             * svd.matrixV().transpose();

        Eigen::Quaterniond q_wb(R_wb);
        Eigen::VectorXd q_init = pinocchio::neutral(model_);
        q_init.head<3>()       = p_wb;
        q_init.segment<4>(3)   = q_wb.coeffs();
        q_init.tail(njnt_)     = q_joints_pin;

        ri_ekf_->initialize(q_init, q_joints_pin);
    }
}

void StateEstimator::update(RobotState& robot_state,
                            const Eigen::Vector3d& gyro,
                            const Eigen::Vector3d& acc,
                            const std::array<bool,2>& contact)
{
    if (!active_) return;

    Eigen::VectorXd jnt_pos(njnt_);
    Eigen::VectorXd jnt_vel(njnt_);
    for (int i = 0; i < njnt_; ++i) {
        const std::string& name = model_.names[i + 2];
        jnt_pos(i) = robot_state.joint_state.at(name).pos;
        jnt_vel(i) = robot_state.joint_state.at(name).vel;
    }

    ri_ekf_->filter(gyro, acc, jnt_pos, jnt_vel, contact);
    robot_state.position         = ri_ekf_->getPosition();
    robot_state.orientation      = ri_ekf_->getQuaternion();
    robot_state.linear_velocity  = ri_ekf_->getQuaternion().toRotationMatrix().transpose()
                                   * ri_ekf_->getVelocity();
    robot_state.angular_velocity = ri_ekf_->getOmegaBody();
}

} // namespace labrob