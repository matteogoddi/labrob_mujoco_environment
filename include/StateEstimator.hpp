#pragma once

#include <array>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <pinocchio/spatial/skew.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <RobotState.hpp>
#include <utils.hpp>

namespace labrob {

// ── Noise parameters (continuous-time std-devs per √Hz) ──────────────────────
struct NoiseParams {
    double gyro_noise        = 0.01;    ///< σᵍ  gyro white noise [rad/s/√Hz]
    double accel_noise       = 0.1;     ///< σᵃ  accel white noise [m/s²/√Hz]
    double contact_noise     = 0.01;    ///< σᵛ  slip noise [m/s/√Hz]
    double gyro_bias_rw      = 0.0001;  ///< σᵇᵍ gyro  bias RW [rad/s²/√Hz]
    double accel_bias_rw     = 0.001;   ///< σᵇᵃ accel bias RW [m/s³/√Hz]
    double encoder_noise     = 0.01;    ///< σᵅ  encoder noise [rad/√Hz]
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
 *  IMU sensor frame and body frame are assumed aligned.
 *
 *  IMU model (MuJoCo accelerometer = specific force in sensor frame):
 *    ω̃ = ω + bᵍ + nᵍ           angular velocity, body frame
 *    ã = R_WB^T(a−g) + bᵃ + nᵃ  specific force,  sensor frame
 *
 *  After removing bias:
 *    ω_corr = ω̃ − b̂ᵍ
 *    f_body  = ã − b̂ᵃ                 (specific force, body frame)
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
     * @param gyro_meas     Raw gyroscope reading,     body frame [rad/s]
     * @param acc_meas      Raw accelerometer reading, body frame [m/s²]
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
    /// Base omega in body frame
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

    /// Re-initialise the whole filter (state, biases, covariance) from a full
    /// Pinocchio configuration, then re-acquire both contacts from FK.
    void initialize   (const Eigen::VectorXd& q_init, const Eigen::VectorXd& joint_pos);
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

// ── StateEstimator: robot-level wrapper around RightInvariantEKF ─────────────
class StateEstimator {
public:
    StateEstimator(const std::string& urdf_path,
                   double dt,
                   const NoiseParams& noise = NoiseParams{});

    StateEstimator(const pinocchio::Model& model,
                   double dt,
                   const NoiseParams& noise = NoiseParams{});

    // Call once when ready to activate (e.g. on gamepad A press).
    // Joint positions are taken from robot_state in Pinocchio ordering.
    void activate(const RobotState& robot_state, const Eigen::VectorXd& q_joints);

    // One filter step. Updates robot_state base pose/velocity in-place.
    // contact[0]=left, contact[1]=right.
    void update(RobotState& robot_state,
                const Eigen::Vector3d& gyro,
                const Eigen::Vector3d& acc,
                const std::array<bool,2>& contact);

    bool is_active() const { return active_; }

private:
    pinocchio::Model model_;
    double           dt_;
    int              njnt_;

    std::unique_ptr<RightInvariantEKF> ri_ekf_;

    bool active_ = false;
};

} // namespace labrob
