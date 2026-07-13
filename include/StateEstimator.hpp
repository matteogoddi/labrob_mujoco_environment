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
    double gyro_noise      = 0.01;
    double accel_noise     = 0.1;
    double contact_noise   = 0.01;
    double gyro_bias_rw    = 0.0001;
    double accel_bias_rw   = 0.001;
    double encoder_noise   = 0.01;
    double foot_lin_noise  = 0.009;
    double foot_ang_noise  = 0.004;
    double encoder_noise_deg = 0.1;
};

/**
 * Contact-Aided Right-Invariant Extended Kalman Filter (RI-EKF)
 *
 * Reference:
 *   Hartley, Ghaffari, Grizzle, Eustice —
 *   "Contact-Aided Invariant Extended Kalman Filtering for Robot State
 *    Estimation", IJRR 2020 (arXiv 1904.09251).
 *
 * State matrix X ∈ SE_{N+2}(3), N_FEET=2, DIM_X=7:
 *
 *   [ R  | v  p  d₀  d₁ ]   (R ∈ SO(3), v/p/dᵢ ∈ ℝ³)
 *   [ 0  | identity 4×4 ]
 *
 * Error state ξ ∈ ℝ²¹: [ξᴿ ξᵛ ξᵖ ξᵈ⁰ ξᵈ¹ δbᵍ δbᵃ]
 */
class RightInvariantEKF
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static constexpr int N_FEET = 2;
    static constexpr int DIM_X  = N_FEET + 5;       // = 7
    static constexpr int COL_V  = 3;
    static constexpr int COL_P  = 4;
    static constexpr int COL_D  = 5;
    static constexpr int NR     = 3*(N_FEET+3) + 6; // = 21
    static constexpr int XI_R   = 0;
    static constexpr int XI_V   = 3;
    static constexpr int XI_P   = 6;
    static constexpr int XI_D   = 9;
    static constexpr int XI_BG  = 9  + 3*N_FEET;    // = 15
    static constexpr int XI_BA  = 12 + 3*N_FEET;    // = 18

    struct FootConfig {
        std::string frame_name;
        int         contact_idx;
    };

    RightInvariantEKF(const pinocchio::Model&               model,
                      const Eigen::VectorXd&                q_init,
                      double                                dt,
                      const std::array<FootConfig,N_FEET>&  feet,
                      const NoiseParams&                    noise = NoiseParams{});

    void filter(const Eigen::Vector3d&         gyro_meas,
                const Eigen::Vector3d&         acc_meas,
                const Eigen::VectorXd&         joint_pos,
                const Eigen::VectorXd&         joint_vel,
                const std::array<bool,N_FEET>& contact);

    Eigen::Matrix3d    getRotation()   const { return X_.block<3,3>(0,0); }
    Eigen::Vector3d    getVelocity()   const { return X_.block<3,1>(0,COL_V); }
    Eigen::Vector3d    getPosition()   const { return X_.block<3,1>(0,COL_P); }
    Eigen::Vector3d    getOmegaBody()  const { return omega_b_; }
    Eigen::Vector3d    getContact(int i) const { return X_.block<3,1>(0,COL_D+i); }
    Eigen::Vector3d    getBiasGyro()   const { return bg_; }
    Eigen::Vector3d    getBiasAccel()  const { return ba_; }
    Eigen::Quaterniond getQuaternion() const {
        return Eigen::Quaterniond(X_.block<3,3>(0,0)).normalized();
    }

    void initialize  (const Eigen::VectorXd& q_init, const Eigen::VectorXd& joint_pos);
    void addContact  (int foot_idx, const Eigen::VectorXd& joint_pos);
    void removeContact(int foot_idx);

private:
    Eigen::Matrix<double,DIM_X,DIM_X>
    groupExp(const Eigen::Matrix<double,3*(N_FEET+3),1>& xi) const;

    Eigen::Matrix<double,DIM_X,DIM_X>
    groupInverse(const Eigen::Matrix<double,DIM_X,DIM_X>& X) const;

    Eigen::Matrix<double,NR,NR>
    adjoint(const Eigen::Matrix<double,DIM_X,DIM_X>& X) const;

    static Eigen::Matrix3d skew(const Eigen::Vector3d& v);
    static Eigen::Matrix3d expSO3(const Eigen::Vector3d& phi);
    static Eigen::Vector3d logSO3(const Eigen::Matrix3d& R);
    static Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& phi);
    static Eigen::Matrix3d projectToSO3(const Eigen::Matrix3d& M);

    pinocchio::Model model_;
    pinocchio::Data  data_;
    double           dt_;

    std::array<FootConfig,N_FEET> feet_;
    NoiseParams noise_;

    const Eigen::Vector3d g_ {0.0, 0.0, -9.81};

    Eigen::Matrix<double,DIM_X,DIM_X> X_;
    Eigen::Vector3d omega_b_;
    Eigen::Vector3d bg_;
    Eigen::Vector3d ba_;
    Eigen::Matrix<double,NR,NR> P_;
    Eigen::Matrix<double,NR,NR> Qc_;

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

    void activate(const RobotState& robot_state, const Eigen::VectorXd& q_joints);

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