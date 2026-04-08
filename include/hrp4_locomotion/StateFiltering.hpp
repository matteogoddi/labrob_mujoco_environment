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
      P_.block<3,3>(0,0) *= 1e-1;    // position
      P_.block<3,3>(3,3) *= 1e-1;    // velocity
      P_.block<3,3>(6,6) *= 1e-2;    // orientation
      P_.block<3,3>(9,9) *= 1e-2;    // feet
      P_.block<3,3>(12,12) *= 1e-2;
      P_.block<3,3>(15,15) *= 1e-5;  // feet orientation
      P_.block<3,3>(18,18) *= 1e-5;
      P_.block<3,3>(21,21) *= 1e-5;  // biases
      P_.block<3,3>(24,24) *= 1e-5;
      
      Qc_.setIdentity();
      Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
      Qc_.block<3,3>(0,0) = 0.01 * I;     // accel noise
      Qc_.block<3,3>(3,3) = 0.001 * I;     // gyro noise
      Qc_.block<3,3>(6,6)  = 1e-6 * I;    // foot noise
      Qc_.block<3,3>(9,9) = 1e-6 * I;     
      Qc_.block<3,3>(12,12) = 1e-5 * I;   // feet orientation noise
      Qc_.block<3,3>(15,15) = 1e-5 * I; 
      Qc_.block<3,3>(18,18) = 1e-5 * I;   // bias noise
      Qc_.block<3,3>(21,21) = 1e-5 * I; 

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

      pinocchio::Model model = model_;
      pinocchio::Data data(model);
      pinocchio::forwardKinematics(model_, data, q_init_);
      pinocchio::framesForwardKinematics(model_, data, q_init_);

      // const auto& bMf_l = data_.oMf[model_.getFrameId("left_foot_link")];
      pL_ = pL_init;
      zL_ = Eigen::Quaterniond::Identity();
      
      // const auto& bMf_r = data_.oMf[model_.getFrameId("right_foot_link")];
      pR_ = pR_init;
      zR_ = Eigen::Quaterniond::Identity();

      // define R_base_imu to rotate measurements from IMU frame to base frame
      // R_base_imu = Eigen::Matrix3d::Identity();
      R_base_imu = data_.oMf[model_.getFrameId("imu_in_pelvis")].rotation();

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
}