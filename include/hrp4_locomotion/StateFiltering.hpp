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
      data_ = pinocchio::Data(model_);
      P_.setIdentity();
      P_.block<3,3>(0,0) *= 1e-1;    // position
      P_.block<3,3>(3,3) *= 1e-1;    // velocity
      P_.block<3,3>(6,6) *= 1e-2;    // orientation
      P_.block<3,3>(9,9) *= 1e-2;    // feet
      P_.block<3,3>(12,12) *= 1e-2;
      // P_.block<3,3>(15,15) *= 1e-5;  // biases
      // P_.block<3,3>(18,18) *= 1e-5;
      
      Qc_.setIdentity();
      Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
      Qc_.block<3,3>(0,0) = 0.05 * I;     // accel noise
      Qc_.block<3,3>(3,3) = 0.01 * I;     // gyro noise
      Qc_.block<3,3>(6,6)  = 1e-5 * I;    // foot noise
      Qc_.block<3,3>(9,9) = 1e-5 * I;     // foot noise
      // Qc_.block<3,3>(12,12) = 1e-6 * I;   // accel bias
      // Qc_.block<3,3>(15,15) = 1e-6 * I;   // gyro bias

      R_.setIdentity() * 5e-4;
      g_ << 0, 0, -9.81;
    }

    // Complete filter step (prediction + update)
    void filter(const Eigen::Vector3d& acc_meas,
              const Eigen::Vector3d& gyro_meas,
              const Eigen::VectorXd& joint_pos_meas,
              const Eigen::VectorXd& joint_vel_meas,
              const Eigen::VectorXd& q_ddot,
              bool isLeftFootinContact,
              bool isRightFootinContact);
    
    Eigen::Vector3d getBasePosition() const { return r_; }
    Eigen::Vector3d getBaseVelocity() const { return v_; }
    Eigen::Quaterniond getBaseOrientation() const { return q_; }
    Eigen::Vector3d getBaseOmega() const { return omega_world; }
    void initialize(const Eigen::VectorXd& q_init, 
                    const Eigen::Vector3d& pL_init, 
                    const Eigen::Vector3d& pR_init) {
      r_ << q_init[0], q_init[1], q_init[2];
      q_ = Eigen::Quaterniond(q_init[6], q_init[3], q_init[4], q_init[5]);
      
      // pinocchio::forwardKinematics(model_, data_, q_init_);
      // pinocchio::framesForwardKinematics(model_, data_, q_init_);
      // pinocchio::computeJointJacobians(model_, data_, q_init_);

      // const auto& bMf_l = data_.oMf[model_.getFrameId("left_foot_link")];
      pL_ = pL_init;
      // zL_ = Eigen::Quaterniond(bMf_l.rotation());
      
      // const auto& bMf_r = data_.oMf[model_.getFrameId("right_foot_link")];
      pR_ = pR_init;
      // zR_ = Eigen::Quaterniond(bMf_r.rotation());

      // define R_base_imu to rotate measurements from IMU frame to base frame
      R_base_imu = Eigen::Matrix3d::Identity();
      // R_base_imu = q_.toRotationMatrix().transpose() * data_.oMf[model_.getFrameId("imu_in_torso")].rotation();


    }

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

      if(th < 1e-5)
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
    int NX = 15;

    // Nominal state
    Eigen::Vector3d r_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d omega_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega_world = Eigen::Vector3d::Zero();

    Eigen::Vector3d pL_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d pR_ = Eigen::Vector3d::Zero();
    // Eigen::Quaterniond zL_;
    // Eigen::Quaterniond zR_;

    Eigen::Vector3d bf_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d bw_ = Eigen::Vector3d::Zero();

    Eigen::MatrixXd R_base_imu;

    // Covariance               
    Eigen::Matrix<double,15,15> P_;
    Eigen::Matrix<double,12,12> Qc_;
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
struct NoiseParams {
  double gyro_noise = 0.0;
  double accel_noise = 0.0;
  double contact_noise = 0.0;
  double gyro_bias_rw = 0.0;
  double accel_bias_rw = 0.0;
  double encoder_noise = 0.0;
};
/**
 * Contact-Aided Right-Invariant Extended Kalman Filter (RI-EKF)
 *
 * Reference (primary):
 *   Hartley, Ghaffari, Grizzle, Eustice —
 *   "Contact-Aided Invariant Extended Kalman Filtering for Robot State
 *   Estimation", IJRR 2020 (arXiv 1904.09251).
 *   RSS 2018 version: arXiv 1805.10410.
 *
 * ════════════════════════════════════════════════════════════════════
 *  MATHEMATICAL STRUCTURE
 * ════════════════════════════════════════════════════════════════════
 *
 * State matrix  Xₜ ∈ SE_{N+2}(3)  (paper Sec. III-A):
 *
 *         ┌ R   v   p   d₁  ⋯  dₙ ┐
 *   Xₜ =  │ 0   1   0   0   ⋯   0 │
 *         │ 0   0   1   0   ⋯   0 │
 *         │ 0   0   0   1   ⋯   0 │
 *         │ ⋮               ⋱   ⋮ │
 *         └ 0   0   0   0   ⋯   1 ┘
 *
 * where:
 *   R  ∈ SO(3)  – rotation world←body  (R_WB)
 *   v  ∈ ℝ³    – body velocity in world frame  (ᵂvᵂᴮ)
 *   p  ∈ ℝ³    – body position in world frame  (ᵂpᵂᴮ)
 *   dᵢ ∈ ℝ³   – i-th contact position in world frame  (ᵂpᵂCᵢ)
 *
 * The matrix dimension is (N+4) × (N+4).
 *
 * IMU biases (paper Sec. IV) are augmented as separate Euclidean
 * states.  The full error state is:
 *
 *   ξ ∈ ℝ^{3(N+3)+6}  =  [ξᴿ(0:3) | ξᵛ(3:6) | ξᵖ(6:9) |
 *                           ξᵈ¹(9:12) | … | ξᵈᴺ(9+3N:12+3N) |
 *                           δbᵍ(…) | δbᵃ(…)]
 *
 * RIGHT-INVARIANT ERROR (paper eq. 1):
 *   ηᵣ = X̂ₜ Xₜ⁻¹
 *
 * This choice makes the error dynamics trajectory-independent
 * (log-linear) — the key property of the RI-EKF.
 *
 * ════════════════════════════════════════════════════════════════════
 *  CONVENTIONS
 * ════════════════════════════════════════════════════════════════════
 *
 *  • R = X.block<3,3>(0,0)  is world←body (R_WB), i.e. maps body
 *    frame vectors to world frame. This matches the paper's R_WB.
 *    Note: this is the OPPOSITE convention from many EKF papers
 *    (Rotella/Bloesch) which use world→body.
 *
 *  • The IMU measures:
 *      ω̃  = ω + bᵍ + nᵍ    angular velocity, body frame
 *      ã  = R^T(a - g) + bᵃ + nᵃ  specific force, body frame
 *    For MuJoCo: ã_sensor = ã  (specific force in sensor frame)
 *    → after rotating to body:  f_body = R_imu_body * ã_sensor
 *      True accel: a_world = R * (f_body - bᵃ) + g
 *
 *  • Contact noise:  the contact point velocity is zero plus noise
 *      ᶜṽᵂᶜ = 0 = ᶜvᵂᶜ + wᵛ,  wᵛ ~ N(0, Σᵛ)
 *    The contact point dynamics are  ḋᵢ = R·hR(α̃)·(-wᵛ)
 *    where hR(α̃) is the rotation from contact frame to body frame.
 *
 * ════════════════════════════════════════════════════════════════════
 *  NOISE PARAMETERS
 * ════════════════════════════════════════════════════════════════════
 *  All noise is specified as continuous-time spectral densities.
 */
class RightInvariantEKF
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr int N_FEET = 2;   ///< number of contact points

    // Error-state dimension:
    //   3*R + 3*v + 3*p + 3*N_FEET*d + 3*bg + 3*ba
    static constexpr int NR = 3 * (3 + N_FEET) + 6;  // = 9 + 3*N + 6 = 21 for N=2

    // Error-state block offsets  (in the ξ vector)
    static constexpr int XI_R  = 0;          ///< rotation error ξᴿ
    static constexpr int XI_V  = 3;          ///< velocity error ξᵛ
    static constexpr int XI_P  = 6;          ///< position error ξᵖ
    // contact point errors: XI_D + 3*i  for i=0..N_FEET-1
    static constexpr int XI_D  = 9;          ///< first contact ξᵈ⁰
    static constexpr int XI_BG = 9 + 3*N_FEET;   ///< gyro  bias  δbᵍ
    static constexpr int XI_BA = 9 + 3*N_FEET + 3; ///< accel bias  δbᵃ

    // State matrix dimension  (N+4) × (N+4)
    static constexpr int DIM_X = N_FEET + 4;

    struct FootConfig {
        std::string frame_name;  ///< Pinocchio frame name
        int         contact_idx; ///< index i into dᵢ column of X
    };

    // struct NoiseParams {
    //     // Continuous-time spectral densities (standard deviations per √Hz)
    //     double gyro_noise    = 0.01;    ///< σᵍ  [rad/s/√Hz]
    //     double accel_noise   = 0.1;     ///< σᵃ  [m/s²/√Hz]
    //     double contact_noise = 0.01;    ///< σᵛ  [m/s/√Hz]  (slip noise)
    //     double gyro_bias_rw  = 0.0001;  ///< σᵇᵍ [rad/s²/√Hz]
    //     double accel_bias_rw = 0.001;   ///< σᵇᵃ [m/s³/√Hz]
    //     // Encoder noise for N_hat computation
    //     double encoder_noise = 0.01;    ///< σᵅ  [rad/√Hz]
    // };

    /**
     * @param model    Pinocchio model (free-flyer, already built)
     * @param q_init   Full Pinocchio config at t=0.
     *                 q_init[3:7] = quaternion (x,y,z,w)  body→world = world←body stored as R_WB.
     * @param dt       Filter timestep [s]
     * @param feet     Per-foot config: frame name + contact index
     * @param noise    Continuous-time noise parameters
     */
    RightInvariantEKF(const pinocchio::Model&               model,
                      const Eigen::VectorXd&                q_init,
                      double                                dt,
                      const std::array<FootConfig,N_FEET>&  feet,
                      const NoiseParams&                    noise = NoiseParams{});

    /**
     * One full RI-EKF step: propagation + correction.
     *
     * @param gyro_meas     Raw gyroscope reading, sensor frame [rad/s]
     * @param acc_meas      Raw accelerometer reading, sensor frame [m/s²]
     * @param joint_pos     Joint positions α̃  (Pinocchio ordering)
     * @param joint_vel     Joint velocities α̃̇  (unused in RI-EKF update;
     *                      kept for API consistency / future extension)
     * @param contact       contact[i] = true when foot i is in contact
     */
    void filter(const Eigen::Vector3d&          gyro_meas,
                const Eigen::Vector3d&          acc_meas,
                const Eigen::VectorXd&          joint_pos,
                const Eigen::VectorXd&          joint_vel,
                const std::array<bool,N_FEET>&  contact);

    // ── Accessors ─────────────────────────────────────────────────────────
    /// Rotation matrix world←body (R_WB = X.block<3,3>(0,0))
    Eigen::Matrix3d    getRotation()    const { return X_.block<3,3>(0,0); }
    /// Base position in world frame
    Eigen::Vector3d    getPosition()    const { return X_.block<3,1>(0,2); }
    /// Base velocity in world frame
    Eigen::Vector3d    getVelocity()    const { return X_.block<3,1>(0,1); }
    /// Contact point i position in world frame
    Eigen::Vector3d    getContact(int i)const { return X_.block<3,1>(0, 3+i); }
    /// Gyroscope bias in body frame
    Eigen::Vector3d    getBiasGyro()    const { return bg_; }
    /// Accelerometer bias in body frame
    Eigen::Vector3d    getBiasAccel()   const { return ba_; }
    /// Quaternion body→world (same as R_WB expressed as quaternion)
    Eigen::Quaterniond getQuaternion()  const {
        return Eigen::Quaterniond(X_.block<3,3>(0,0)).normalized();
    }

    // ── Contact management ────────────────────────────────────────────────

    /**
     * Called when foot i makes contact.  Resets the contact position
     * using current FK estimate and inflates covariance for that block.
     * (Paper Sec. V: contact switching.)
     */
    void addContact(int foot_idx,
                    const Eigen::VectorXd& joint_pos);

    /**
     * Called when foot i loses contact.  Inflates the process noise for
     * that contact point so P grows quickly.
     */
    void removeContact(int foot_idx);

private:
    // ── SE_{N+2}(3) operations ─────────────────────────────────────────────

    /**
     * Exponential map for SE_{N+2}(3).
     * Maps a Lie algebra element ξ ∈ ℝ^{3(N+2)} to the group element.
     *
     * For the rotation block we use the exact Rodrigues formula.
     * For the remaining columns we use the first-order approximation
     * (I + φ×)·col  which is exact to first order and used in the filter.
     *
     * Full matrix exponential of the Lie algebra element Lg(ξ):
     *
     *   exp(Lg(ξ)) = I + Γ₀(φ)·Lg(ξ) + Γ₁(φ)·Lg(ξ)²
     *
     * where Γ₀, Γ₁ are the standard SE(3) coefficients.
     * We implement the exact version.
     *
     * @param xi_R   rotation  part of ξ  (3-vector)
     * @param xi_cols remaining column vectors  (3 × (N+1) matrix)
     */
    Eigen::Matrix<double,DIM_X,DIM_X>
    groupExp(const Eigen::VectorXd& xi) const;

    /**
     * Logarithm map for SE_{N+2}(3): group element → ξ vector.
     * Used for residual computation.
     */
    Eigen::VectorXd groupLog(
        const Eigen::Matrix<double,DIM_X,DIM_X>& X) const;

    /** Inverse of a state matrix X ∈ SE_{N+2}(3). */
    Eigen::Matrix<double,DIM_X,DIM_X>
    groupInverse(const Eigen::Matrix<double,DIM_X,DIM_X>& X) const;

    /** Adjoint matrix AdX for X ∈ SE_{N+2}(3) (paper eq. in Sec. III-A). */
    Eigen::Matrix<double,NR,NR>
    adjoint(const Eigen::Matrix<double,DIM_X,DIM_X>& X) const;

    // ── Math helpers ───────────────────────────────────────────────────────
    static Eigen::Matrix3d  skew(const Eigen::Vector3d& v);
    static Eigen::Matrix3d  expSO3(const Eigen::Vector3d& phi);
    static Eigen::Vector3d  logSO3(const Eigen::Matrix3d& R);

    /** SO(3) left Jacobian  J_l(φ).  Used in exp map of SE(3) cols. */
    static Eigen::Matrix3d  leftJacobianSO3(const Eigen::Vector3d& phi);

    // ── Model / data ───────────────────────────────────────────────────────
    pinocchio::Model model_;
    pinocchio::Data  data_;
    double           dt_;

    std::array<FootConfig, N_FEET> feet_;
    NoiseParams noise_;

    /// Rotation: IMU sensor frame → body frame (fixed at init)
    Eigen::Matrix3d R_imu_to_body_;

    /// Gravity in world frame
    const Eigen::Vector3d g_ {0.0, 0.0, -9.81};

    // ── State ──────────────────────────────────────────────────────────────
    /// State matrix X ∈ SE_{N+2}(3)
    Eigen::Matrix<double, DIM_X, DIM_X> X_;

    /// IMU biases (Euclidean, body frame)
    Eigen::Vector3d bg_;   ///< gyroscope bias
    Eigen::Vector3d ba_;   ///< accelerometer bias

    /// Error-state covariance  P ∈ ℝ^{NR×NR}
    Eigen::Matrix<double, NR, NR> P_;

    /// Continuous noise covariance  Q_c  in the ξ basis
    /// (before AdX transformation — paper eq. 8)
    Eigen::Matrix<double, NR, NR> Qc_;

    // track which contacts are currently active
    std::array<bool, N_FEET> active_contact_;
};

} // namespace state_filtering
