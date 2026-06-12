#pragma once

#include <Eigen/Dense>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/skew.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <iostream>

#include <utils.hpp>

namespace labrob
{

// =======================
// Joint velocity KF
// =======================
class JointKF
{
public:
  JointKF(double dt, int nq);

  void filter(const Eigen::VectorXd& q_meas, const Eigen::VectorXd& qdd);

  Eigen::VectorXd getFilteredJointPositions() const {return JointPos_;}
  Eigen::VectorXd getFilteredJointVelocities() const {return JointVel_;}
  Eigen::VectorXd getFilteredOmega() const {return Omega_;}
private:
  Eigen::VectorXd q_filtered_;
  Eigen::MatrixXd K;
  Eigen::MatrixXd F;
  Eigen::MatrixXd G;
  Eigen::MatrixXd H;
  Eigen::VectorXd JointPos_;
  Eigen::VectorXd JointVel_;
  Eigen::VectorXd Omega_;
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
      P_.block<3,3>(0,0) *= 1e-4;    // position
      P_.block<3,3>(3,3) *= 1e-4;    // velocity
      P_.block<3,3>(6,6) *= 1e-4;    // orientation
      P_.block<3,3>(9,9) *= 1e-4;    // feet
      P_.block<3,3>(12,12) *= 1e-4;
      P_.block<3,3>(15,15) *= 1e-5;  // feet orientation
      P_.block<3,3>(18,18) *= 1e-5;
      P_.block<3,3>(21,21) *= 1e-3;  // biases
      P_.block<3,3>(24,24) *= 1e-3;
      
      Qc_.setIdentity();
      Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
      Qc_.block<3,3>(0,0) = 0.005 * I;     // accel noise
      Qc_.block<3,3>(3,3) = 0.001 * I;     // gyro noise
      Qc_.block<3,3>(6,6)  = 1e-4 * I;    // foot noise
      Qc_.block<3,3>(9,9) = 1e-4 * I;     
      Qc_.block<3,3>(12,12) = 1e-5 * I;   // feet orientation noise
      Qc_.block<3,3>(15,15) = 1e-5 * I; 
      Qc_.block<3,3>(18,18) = 1e-3 * I;   // bias noise
      Qc_.block<3,3>(21,21) = 1e-3 * I; 

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
    Eigen::Quaterniond getBaseOrientation() const { return q_.inverse(); }
    Eigen::Vector3d getBaseOmega() const { return omega_; }
    void initialize(const Eigen::VectorXd& q_init, 
                    const Eigen::Vector3d& pL_init, 
                    const Eigen::Vector3d& pR_init) {
      r_ << q_init[0], q_init[1], q_init[2];
      q_ = Eigen::Quaterniond(q_init[6], q_init[3], q_init[4], q_init[5]).inverse();
      prev_left_contact_  = true;
      prev_right_contact_ = true;

      pinocchio::Data data(model_);
      pinocchio::forwardKinematics(model_, data, q_init);
      pinocchio::framesForwardKinematics(model_, data, q_init);

      pL_ = pL_init;
      zL_ = Eigen::Quaterniond(data.oMf[model_.getFrameId("left_foot_link")].rotation()).normalized();
      pR_ = pR_init;
      zR_ = Eigen::Quaterniond(data.oMf[model_.getFrameId("right_foot_link")].rotation()).normalized();

      omega_prev_    = Eigen::Vector3d::Zero();
      acc_body_prev_ = Eigen::Vector3d::Zero();

      // R_base_imu: ruota misure dal frame IMU al frame body.
      // R_world_imu da FK (usa la variabile locale data, non data_ che non è aggiornata).
      // q_.toRotationMatrix() = R_BW; R_BW * R_world_imu = R_body_imu.
      Eigen::Matrix3d R_world_imu = data.oMf[model_.getFrameId("imu_in_pelvis")].rotation();
      R_base_imu = q_.toRotationMatrix() * R_world_imu;

    // const int imu_torso_id = model_.getFrameId("imu_in_torso");
    // R_imu_torso_to_body_ = R_WB.transpose()
    //                * data_.oMf[imu_torso_id].rotation();


    }

private:

    pinocchio::Model model_;
    pinocchio::Data data_;
    Eigen::VectorXd q_init_;
    bool left_initialized_ = false;
    bool right_initialized_ = false;
    bool prev_left_contact_  = true;
    bool prev_right_contact_ = true;

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
    const int NX = 27;

    // Nominal state
    Eigen::Vector3d r_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d omega_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega_world = Eigen::Vector3d::Zero();

    Eigen::Vector3d pL_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d pR_ = Eigen::Vector3d::Zero();
    Eigen::Quaterniond zL_;
    Eigen::Quaterniond zR_;

    Eigen::Vector3d bf_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d bw_ = Eigen::Vector3d::Zero();

    Eigen::MatrixXd R_base_imu;

    // Inputs at k-1 (used for prediction x[k-1]→x[k] at the next call)
    Eigen::Vector3d omega_prev_     = Eigen::Vector3d::Zero();  // body-frame ω at k-1
    Eigen::Vector3d acc_body_prev_  = Eigen::Vector3d::Zero();  // body-frame specific force at k-1

    // Covariance
    Eigen::Matrix<double,27,27> P_;
    Eigen::Matrix<double,24,24> Qc_;

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
  double gyro_noise = 0.01;
  double accel_noise = 0.1;
  double contact_noise = 0.01;
  double gyro_bias_rw = 0.0001;
  double accel_bias_rw = 0.001;
  double encoder_noise = 0.01;
  double foot_lin_noise    = 0.009;   ///< σv_feet [m/s/√Hz] (slip)
  double foot_ang_noise    = 0.004;   ///< σω_feet [rad/s/√Hz]
  double encoder_noise_deg = 0.1;     ///< σα [deg]
};
/**
 * Contact-Aided Right-Invariant Extended Kalman Filter (RI-EKF)
 *
 * Reference:
 *   Hartley, Ghaffari, Grizzle, Eustice —
 *   "Contact-Aided Invariant Extended Kalman Filtering for Robot State
 *    Estimation", IJRR 2020 (arXiv 1904.09251).
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  STATE MATRIX  X ∈ SE_{N+2}(3)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  For N_FEET = 2 contact points the state matrix is (N+5)×(N+5) = 7×7:
 *
 *        col:  0  1  2 | 3  | 4  | 5   | 6
 *              ─────── | ── | ── | ─── | ───
 *  row 0─2:  [  R      | v  | p  | d₀  | d₁ ]   ← 3 top rows
 *  row 3  :  [  0  0  0| 1  | 0  | 0   | 0  ]
 *  row 4  :  [  0  0  0| 0  | 1  | 0   | 0  ]
 *  row 5  :  [  0  0  0| 0  | 0  | 1   | 0  ]
 *  row 6  :  [  0  0  0| 0  | 0  | 0   | 1  ]
 *
 *  Where:
 *    R  ∈ SO(3)  – rotation  world ← body  (R_WB)
 *    v  ∈ ℝ³    – base velocity in world frame
 *    p  ∈ ℝ³    – base position in world frame
 *    dᵢ ∈ ℝ³   – i-th contact position in world frame
 *
 *  Column offsets (named constants):
 *    COL_R = 0  (R occupies cols 0,1,2)
 *    COL_V = 3
 *    COL_P = 4
 *    COL_D = 5  (d₀ at col 5,  d₁ at col 6)
 *
 *  General formula:  DIM_X = N_FEET + 5
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  ERROR STATE  ξ ∈ ℝ^NR
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  Lie algebra part  (3*(N+3) = 15 components for N=2):
 *    ξᴿ   [0 :3 ]  – rotation  error
 *    ξᵛ   [3 :6 ]  – velocity  error
 *    ξᵖ   [6 :9 ]  – position  error
 *    ξᵈ⁰  [9 :12]  – left  contact error
 *    ξᵈ¹  [12:15]  – right contact error
 *
 *  Euclidean part (6 components):
 *    δbᵍ  [15:18]  – gyroscope  bias error
 *    δbᵃ  [18:21]  – accelerometer bias error
 *
 *  Total:  NR = 3*(N_FEET+3) + 6  =  21  for N_FEET=2
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  RIGHT-INVARIANT ERROR  (paper eq. 1)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *   ηᵣ = X̂ · X⁻¹
 *
 *  This choice makes the error dynamics trajectory-independent (log-linear),
 *  which is the central property of the RI-EKF.
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  IMU AND SENSOR CONVENTIONS
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  IMU model (MuJoCo accelerometer = specific force in sensor frame):
 *    ω̃ = ω + bᵍ + nᵍ           angular velocity, body frame
 *    ã = R_WB^T(a−g) + bᵃ + nᵃ  specific force,  sensor frame
 *
 *  After rotating to body and removing bias:
 *    ω_corr = R_imu_body * ω̃ − b̂ᵍ
 *    f_body  = R_imu_body * ã  − b̂ᵃ   (specific force, body frame)
 *    a_world = R_WB * f_body + g
 */
class RightInvariantEKF
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 
    // ── Dimensions ────────────────────────────────────────────────────────
    static constexpr int N_FEET = 2;
 
    // State matrix dimension: (N+5)×(N+5)  for N=N_FEET
    static constexpr int DIM_X  = N_FEET + 5;   // = 7
 
    // Column offsets inside X_
    static constexpr int COL_V  = 3;            // velocity
    static constexpr int COL_P  = 4;            // position
    static constexpr int COL_D  = 5;            // first contact  (COL_D + i for contact i)
 
    // Error-state dimension
    static constexpr int NR     = 3*(N_FEET+3) + 6;  // = 21
 
    // Error-state block offsets
    static constexpr int XI_R   = 0;
    static constexpr int XI_V   = 3;
    static constexpr int XI_P   = 6;
    static constexpr int XI_D   = 9;            // XI_D + 3*i for contact i
    static constexpr int XI_BG  = 9  + 3*N_FEET;    // = 15
    static constexpr int XI_BA  = 12 + 3*N_FEET;    // = 18
 
    // ── User-facing types ─────────────────────────────────────────────────
    struct FootConfig {
        std::string frame_name;   ///< Pinocchio frame name
        int         contact_idx;  ///< 0 or 1  (index i into dᵢ column of X)
    };
 
    // struct NoiseParams {
    //     // Continuous-time standard deviations (per √Hz)
    //     double gyro_noise    = 0.01;    ///< σᵍ  gyro white noise [rad/s/√Hz]
    //     double accel_noise   = 0.1;     ///< σᵃ  accel white noise [m/s²/√Hz]
    //     double contact_noise = 0.01;    ///< σᵛ  slip noise [m/s/√Hz]
    //     double gyro_bias_rw  = 0.0001;  ///< σᵇᵍ gyro  bias RW [rad/s²/√Hz]
    //     double accel_bias_rw = 0.001;   ///< σᵇᵃ accel bias RW [m/s³/√Hz]
    //     double encoder_noise = 0.01;    ///< σᵅ  encoder noise [rad/√Hz]
    // };
 
    /**
     * @param model   Pinocchio model (free-flyer base, already built)
     * @param q_init  Full Pinocchio config at t=0.
     *                q_init[3:7] = quaternion (x,y,z,w) representing
     *                the rotation body→world (= R_WB as a quaternion).
     * @param dt      Filter timestep [s]
     * @param feet    Per-foot config (must be exactly N_FEET entries)
     * @param noise   Noise parameters
     */
    RightInvariantEKF(const pinocchio::Model&               model,
                      const Eigen::VectorXd&                q_init,
                      double                                dt,
                      const std::array<FootConfig,N_FEET>&  feet,
                      const NoiseParams&                    noise = NoiseParams{});
 
    /**
     * One full RI-EKF step: propagation + correction.
     *
     * @param gyro_meas     Raw gyroscope reading,     sensor frame [rad/s]
     * @param acc_meas      Raw accelerometer reading, sensor frame [m/s²]
     * @param joint_pos     Joint positions α̃ (Pinocchio ordering)
     * @param joint_vel     Joint velocities α̃̇ (Pinocchio ordering)
     * @param contact       contact[i] = true when foot i is in stance
     */
    void filter(const Eigen::Vector3d&         gyro_meas,
                const Eigen::Vector3d&         acc_meas,
                const Eigen::VectorXd&         joint_pos,
                const Eigen::VectorXd&         joint_vel,
                const std::array<bool,N_FEET>& contact);
 
    // ── Accessors ─────────────────────────────────────────────────────────
    /// R_WB: rotation world←body
    Eigen::Matrix3d    getRotation()       const { return X_.block<3,3>(0,0); }
    /// Base velocity in world frame
    Eigen::Vector3d    getVelocity()       const { return X_.block<3,1>(0,COL_V); }
    /// Base position in world frame
    Eigen::Vector3d    getPosition()       const { return X_.block<3,1>(0,COL_P); }
    /// Base omega in world frame
    Eigen::Vector3d    getOmegaBody()      const { return omega_b_; }
    /// Contact position i in world frame
    Eigen::Vector3d    getContact(int i)   const { return X_.block<3,1>(0,COL_D+i); }
    /// Gyroscope  bias in body frame
    Eigen::Vector3d    getBiasGyro()       const { return bg_; }
    /// Accelerometer bias in body frame
    Eigen::Vector3d    getBiasAccel()      const { return ba_; }
    /// Quaternion representing R_WB (body→world)
    Eigen::Quaterniond getQuaternion()     const {
        return Eigen::Quaterniond(X_.block<3,3>(0,0)).normalized();
    }
 
    /// Reset contact i: re-init position from FK and inflate covariance.
    void addContact   (int foot_idx, const Eigen::VectorXd& joint_pos);
    /// Mark contact i as lost: inflate process noise so P grows freely.
    void removeContact(int foot_idx);
 
private:
    // ── SE_{N+2}(3) operations ────────────────────────────────────────────
 
    /**
     * Group exponential: xi (Lie algebra vector, dim = 3*(N_FEET+3)) → X
     *
     * xi layout: [ξᴿ(0:3) | ξᵛ(3:6) | ξᵖ(6:9) | ξᵈ⁰(9:12) | ξᵈ¹(12:15)]
     *
     * Uses exact Rodrigues for R block; left-Jacobian weighted terms for
     * the vector columns (v, p, d₀, d₁).
     */
    Eigen::Matrix<double,DIM_X,DIM_X>
    groupExp(const Eigen::Matrix<double,3*(N_FEET+3),1>& xi) const;
 
    /**
     * Group inverse:  X⁻¹ for X ∈ SE_{N+2}(3).
     *
     * X⁻¹ = [  R^T   | -R^T·v  -R^T·p  -R^T·d₀  -R^T·d₁ ]
     *        [  0     |   identity block (N+2)×(N+2)       ]
     */
    Eigen::Matrix<double,DIM_X,DIM_X>
    groupInverse(const Eigen::Matrix<double,DIM_X,DIM_X>& X) const;
 
    /**
     * Adjoint representation AdX (NR×NR, including bias rows).
     *
     * Lie algebra block (paper Sec. III-A):
     *   AdX = [ R         0   0   0   0  ]  rows: ξᴿ
     *         [ (v)×R     R   0   0   0  ]  rows: ξᵛ
     *         [ (p)×R     0   R   0   0  ]  rows: ξᵖ
     *         [ (d₀)×R    0   0   R   0  ]  rows: ξᵈ⁰
     *         [ (d₁)×R    0   0   0   R  ]  rows: ξᵈ¹
     *
     * Bias block (Euclidean, not part of the Lie group):
     *   AdX[bias, bias] = I₆    (identity)
     *   AdX[Lie, bias]  = 0     (no cross-coupling in adjoint)
     */
    Eigen::Matrix<double,NR,NR>
    adjoint(const Eigen::Matrix<double,DIM_X,DIM_X>& X) const;
 
    // ── SO(3) and math helpers ────────────────────────────────────────────
    static Eigen::Matrix3d  skew(const Eigen::Vector3d& v);
    static Eigen::Matrix3d  expSO3(const Eigen::Vector3d& phi);
    static Eigen::Vector3d  logSO3(const Eigen::Matrix3d& R);
    static Eigen::Matrix3d  leftJacobianSO3(const Eigen::Vector3d& phi);
    static Eigen::Matrix3d  projectToSO3(const Eigen::Matrix3d& M);
 
    // ── Model / data ──────────────────────────────────────────────────────
    pinocchio::Model model_;
    pinocchio::Data  data_;
    double           dt_;
 
    std::array<FootConfig,N_FEET> feet_;
    NoiseParams noise_;
 
    /// Fixed rotation: IMU sensor frame → body frame
    Eigen::Matrix3d R_imu_pelvis_to_body_;
    Eigen::Matrix3d R_imu_torso_to_body_;
 
    /// Gravity vector in world frame
    const Eigen::Vector3d g_ {0.0, 0.0, -9.81};
 
    // ── State ─────────────────────────────────────────────────────────────
    Eigen::Matrix<double,DIM_X,DIM_X> X_;   ///< State matrix ∈ SE_{N+2}(3)
    Eigen::Vector3d omega_b_;               ///< Base angular velocity, body frame
    Eigen::Vector3d bg_;                     ///< Gyroscope  bias, body frame
    Eigen::Vector3d ba_;                     ///< Accelerometer bias, body frame
    Eigen::Matrix<double,NR,NR> P_;          ///< Error-state covariance
    Eigen::Matrix<double,NR,NR> Qc_;         ///< Continuous process noise (ξ basis)
 
    std::array<bool,N_FEET> active_contact_;
};

/**
 * DILIGENT-KIO: Discrete LIe Group ExteNded Kalman filTer for
 * contact-aided Kinematic-Inertial Odometry.
 *
 * Reference:
 *   Ramadoss, Romualdi, Dafarra, Andrade Chavez, Traversaro, Pucci —
 *   "DILIGENT-KIO: A Proprioceptive Base Estimator for Humanoid Robots
 *   using Extended Kalman Filtering on Matrix Lie Groups", ICRA 2021.
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  KEY ADVANCE OVER HARTLEY 2020
 * ════════════════════════════════════════════════════════════════════════════
 *  Hartley uses SE_{N+2}(3) but cannot include foot ORIENTATIONS in the state
 *  (flat-foot rotational constraints violate the semi-direct product rule).
 *  DILIGENT-KIO uses the composite group M = SE₂(3) × SE(3)² × T(6) which
 *  naturally accommodates full foot poses.
 *
 *  The measurement model uses the full 6-DOF relative pose from FK (position
 *  AND orientation of each foot in the base frame), giving 6 measurement rows
 *  per contact foot vs 3 in the point-foot filters.
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  STATE SPACE  M = SE₂(3) × SE(3)² × T(6)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  The state X ∈ M is stored as separate components for clarity:
 *    Base (SE₂(3)):  R  ∈ SO(3)  – rotation    world ← body  (R_WB)
 *                    p  ∈ ℝ³    – position     in world frame
 *                    v  ∈ ℝ³    – velocity     in world frame
 *    Left foot (SE(3)):  Zl ∈ SO(3)  – rotation  world ← left-foot
 *                        dl ∈ ℝ³    – position  in world frame
 *    Right foot (SE(3)): Zr, dr  (same)
 *    Biases (T(6)):      ba ∈ ℝ³  – accelerometer bias  in body frame
 *                        bg ∈ ℝ³  – gyroscope bias       in body frame
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  ERROR STATE  ε ∈ ℝ²⁷   (paper eq. 5)
 * ════════════════════════════════════════════════════════════════════════════
 *
 *  Using the LEFT-INVARIANT error  X̂⁻¹X  (Algorithm 1 of paper):
 *
 *   ε = [εp(0:3) | εR(3:6) | εv(6:9) | εdl(9:12) | εZl(12:15) |
 *         εdr(15:18) | εZr(18:21) | εba(21:24) | εbg(24:27)]
 *
 *  (NB: This is left-invariant unlike Hartley's right-invariant formulation.)
 *
 * ════════════════════════════════════════════════════════════════════════════
 *  IMU CONVENTION
 * ════════════════════════════════════════════════════════════════════════════
 *  IMU frames S and body frame B are assumed to coincide (paper Sec. III-B).
 *  If not coincident, rotate the measurements before passing to the filter:
 *    gyro_body  = R_S_to_B * gyro_meas
 *    accel_body = R_S_to_B * accel_meas
 *
 *  Accelerometer model (paper eq. 9):
 *    ã = R^T (Äo_B − g_world) + ba + na    [specific force in body frame]
 *  so: R ã − g_world = Äo_B − R R^T g... = (accel_world − g_world)... 
 *  In practice (MuJoCo): ã_sensor = R_WB^T (a_true_world − g) + ba + na
 *  → after rotating to body:  f_body = R_imu_body * accel_meas − ba
 *  → world accel:             a_world = R_WB * f_body + g
 */
class DiligentKio
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr int N_FEET = 2;
    static constexpr int NR     = 27;   // error-state dimension

    // ── Error-state block offsets (paper eq. 5) ───────────────────────────
    static constexpr int EI_P  = 0;
    static constexpr int EI_R  = 3;
    static constexpr int EI_V  = 6;
    static constexpr int EI_DL = 9;
    static constexpr int EI_ZL = 12;
    static constexpr int EI_DR = 15;
    static constexpr int EI_ZR = 18;
    static constexpr int EI_BA = 21;
    static constexpr int EI_BG = 24;

    struct FootConfig {
        std::string frame_name;   ///< Pinocchio frame name
        int         state_idx;    ///< 0 = left, 1 = right
    };

    // struct NoiseParams {
    //     // Continuous-time standard deviations per √Hz (paper Table III)
    //     double accel_noise       = 0.09;    ///< σa  [m/s²/√Hz]
    //     double gyro_noise        = 0.01;    ///< σω  [rad/s/√Hz]
    //     double accel_bias_rw     = 0.01;    ///< σba [m/s²/√Hz]   (bias random walk)
    //     double gyro_bias_rw      = 0.001;   ///< σbg [rad/s/√Hz]
    //     double foot_lin_noise    = 0.009;   ///< σv_feet [m/s/√Hz] (slip)
    //     double foot_ang_noise    = 0.004;   ///< σω_feet [rad/s/√Hz]
    //     double encoder_noise_deg = 0.1;     ///< σα [deg]
    // };

    /**
     * @param model   Pinocchio model (free-flyer base)
     * @param q_init  Full Pinocchio config at t=0.
     *                q_init[3:7] = quaternion (x,y,z,w) body→world (= R_WB quaternion)
     * @param dt      Filter timestep [s]
     * @param feet    {left, right} foot configurations
     * @param noise   Noise parameters
     */
    DiligentKio(const pinocchio::Model&               model,
                const Eigen::VectorXd&                q_init,
                double                                dt,
                const std::array<FootConfig,N_FEET>&  feet,
                const NoiseParams&                    noise = NoiseParams{});

    /**
     * One full DILIGENT-KIO step: propagation + update (Algorithm 1).
     *
     * @param gyro_meas     Raw gyroscope,     body/sensor frame [rad/s]
     * @param accel_meas    Raw accelerometer, body/sensor frame [m/s²]
     * @param joint_pos     Joint positions  (Pinocchio ordering)
     * @param joint_vel     Joint velocities (Pinocchio ordering)
     * @param contact       contact[0]=left, contact[1]=right
     */
    void filter(const Eigen::Vector3d&         gyro_meas,
                const Eigen::Vector3d&         accel_meas,
                const Eigen::VectorXd&         joint_pos,
                const Eigen::VectorXd&         joint_vel,
                const std::array<bool,N_FEET>& contact);

    // ── Accessors ─────────────────────────────────────────────────────────
    Eigen::Matrix3d    getRotation()      const { return R_; }    // R_WB: body→world
    Eigen::Vector3d    getPosition()      const { return p_; }
    Eigen::Vector3d    getVelocity()      const { return v_; }
    Eigen::Quaterniond getQuaternion()    const { return Eigen::Quaterniond(R_).normalized(); }
    Eigen::Vector3d    getBiasAccel()     const { return ba_; }
    Eigen::Vector3d    getBiasGyro()      const { return bg_; }
    Eigen::Vector3d    getOmegaBody()     const { return omega_b_; }
    Eigen::Vector3d    getFootPos(int i)  const { return (i==0) ? dl_ : dr_; }
    Eigen::Matrix3d    getFootRot(int i)  const { return (i==0) ? Zl_ : Zr_; }

    /// Reset a foot's pose from FK when contact is made.
    void addContact   (int foot_idx, const Eigen::VectorXd& joint_pos);
    void removeContact(int foot_idx);

private:
    // ── SO(3) helpers ─────────────────────────────────────────────────────
    static Eigen::Matrix3d skew  (const Eigen::Vector3d& v);
    static Eigen::Matrix3d expSO3(const Eigen::Vector3d& phi);
    static Eigen::Vector3d logSO3(const Eigen::Matrix3d& R);
    static Eigen::Matrix3d leftJacobianSO3    (const Eigen::Vector3d& phi);
    static Eigen::Matrix3d leftJacobianInvSO3 (const Eigen::Vector3d& phi);
    static Eigen::Matrix3d QmatrixSE3         (const Eigen::Vector3d& rho,
                                               const Eigen::Vector3d& phi);
    static Eigen::Matrix3d projectToSO3(const Eigen::Matrix3d& M);

    // ── SE(3) helpers ─────────────────────────────────────────────────────

    /// SE(3) log^∨: SE(3) → ℝ⁶,  output = [v(0:3); ω(3:6)]
    static Eigen::Matrix<double,6,1> logSE3(const Eigen::Matrix3d& R_h,
                                            const Eigen::Vector3d& t_h);

    /// Left Jacobian of SE(3): ℝ⁶ → ℝ⁶×⁶
    /// ΦSE(3)(ε) = [J(εω)   Q(εv, εω)]
    ///              [0₃      J(εω)    ]
    static Eigen::Matrix<double,6,6> leftJacobianSE3(
        const Eigen::Vector3d& eps_v, const Eigen::Vector3d& eps_omega);

    // ── Model / data ──────────────────────────────────────────────────────
    pinocchio::Model model_;
    pinocchio::Data  data_;
    double           dt_;

    std::array<FootConfig,N_FEET> feet_;
    NoiseParams noise_;

    /// R_imu_to_body (rotation: IMU sensor frame → body frame, fixed at init)
    Eigen::Matrix3d R_imu_to_body_;

    const Eigen::Vector3d g_world_ {0.0, 0.0, -9.81};

    // ── Nominal state ─────────────────────────────────────────────────────
    Eigen::Matrix3d R_;          ///< R_WB: body→world  (paper's R in eq.10)
    Eigen::Vector3d p_;          ///< base position, world frame
    Eigen::Vector3d v_;          ///< base velocity, world frame
    Eigen::Matrix3d Zl_, Zr_;   ///< foot orientations (foot→world)
    Eigen::Vector3d dl_, dr_;   ///< foot positions, world frame
    Eigen::Vector3d ba_, bg_;   ///< IMU biases, body frame
    Eigen::Vector3d omega_b_;   ///< last bias-corrected angular velocity, body frame

    // ── Filter matrices ───────────────────────────────────────────────────
    Eigen::Matrix<double,NR,NR> P_;   ///< error-state covariance
    Eigen::Matrix<double,NR,NR> Q_;   ///< discrete process noise (pre-computed)

    std::array<bool,N_FEET> active_contact_;
};

}