#pragma once

#include <array>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <pinocchio/multibody/model.hpp>

#include <RobotState.hpp>
#include <StateFiltering.hpp>

namespace labrob {

class StateEstimator {
public:
    enum class Filter { RightInvariantEKF, DiligentKio };

    // Build pinocchio model from URDF internally.
    StateEstimator(const std::string& urdf_path,
                   double dt,
                   Filter filter = Filter::RightInvariantEKF,
                   const NoiseParams& noise = NoiseParams{});

    StateEstimator(const pinocchio::Model& model,
                   double dt,
                   Filter filter = Filter::RightInvariantEKF,
                   const NoiseParams& noise = NoiseParams{});

    // Call once when ready to activate (e.g. on gamepad A press).
    // q_joints: current joint positions in Pinocchio ordering (size njnt).
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
    Filter           filter_;

    std::unique_ptr<RightInvariantEKF> ri_ekf_;
    std::unique_ptr<DiligentKio>       diligent_kio_;

    bool active_ = false;
};

} // namespace labrob