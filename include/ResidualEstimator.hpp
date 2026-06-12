#ifndef LABROB_RESIDUAL_ESTIMATOR_HPP_
#define LABROB_RESIDUAL_ESTIMATOR_HPP_

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <mujoco/mujoco.h>

#include <RobotState.hpp>
#include <WholeBodyController.hpp> 

namespace labrob {


class ResidualEstimator {
public:
    // Constructor for full robot residual estimation
    ResidualEstimator(
        const pinocchio::Model& robot_model,
        double Ki,
        const std::map<std::string, double>& armatures,
        const std::vector<std::string>& right_arm_joint_names = {});

    // Constructor for fixed-base right arm residual estimation
    ResidualEstimator(
        const pinocchio::Model& robot_model,
        const pinocchio::Model& full_robot_model,
        double Ki,
        const std::map<std::string, double>& armatures,
        const std::vector<std::string>& right_arm_joint_names);

    // Full robot residual computation
    Eigen::VectorXd computeResidual(
        const RobotState& robot_state,
        pinocchio::Data& robot_data,
        const Eigen::VectorXd& measured_torques,
        const Eigen::MatrixXd& J_l,
        const Eigen::MatrixXd& J_r,
        const Eigen::VectorXd& W_l,
        const Eigen::VectorXd& W_r,
        double dt,
        mjModel* mj_model,        // mujoco model
        mjData* mj_data           // mujoco data
        );         // MuJoCo data
    // Add this to ResidualEstimator.hpp in the public section
    Eigen::VectorXd computeResidualWithWBCWrenches(
        const RobotState& robot_state,
        pinocchio::Data& robot_data,
        const Eigen::VectorXd& measured_torques,
        //const Eigen::VectorXd& wbc_left_foot_wrench,
        //const Eigen::VectorXd& wbc_right_foot_wrench,
        const labrob::WholeBodyController& wbc_controller,
        double dt);

    // Eigen::VectorXd computeResidualWithWBC(
    //     const RobotState& robot_state,
    //     pinocchio::Data& robot_data,
    //     const Eigen::VectorXd& measured_torques,
    //     const labrob::WholeBodyController& wbc_controller, // Pass WBC controller
    //     double dt,
    //     mjModel* mj_model,
    //     mjData* mj_data);

    // Right arm residual with full robot dynamics (selection matrix approach)
    Eigen::VectorXd computeRightArmResidual(
        const RobotState& robot_state,
        pinocchio::Data& robot_data,
        const Eigen::VectorXd& measured_torques,
        const Eigen::MatrixXd& J_l,
        const Eigen::MatrixXd& J_r,
        const Eigen::VectorXd& W_l,
        const Eigen::VectorXd& W_r,
        double dt);
    
    // Right arm residual with fixed base (isolated arm) - NEW IMPROVED VERSION
    Eigen::VectorXd computeRightArmResidualFixedBase(
        const RobotState& robot_state,
        pinocchio::Data& right_arm_data,
        const Eigen::VectorXd& measured_torques_right_arm,
        double dt);

    // NEW: Compute external force at wrist from arm residual
    Eigen::VectorXd estimateWristForce(const Eigen::VectorXd& arm_residual,
                                      pinocchio::Data& right_arm_data) const;

    // NEW: Get right arm configuration from full robot state
    void getRightArmConfiguration(const RobotState& robot_state,
                                 Eigen::VectorXd& q_arm,
                                 Eigen::VectorXd& v_arm) const;


    // Getters
    const Eigen::VectorXd& getResidual() const { return r_; }
    const Eigen::VectorXd& getRightArmResidual() const { return r_arm_; }
    Eigen::VectorXd getGeneralizedMomentum() const { return h_prev_; }
    Eigen::VectorXd getDynamicsTerms() const { return c_prev_; }
    Eigen::VectorXd getFeetEstimatedForce() const { return f_estimated; }

    // Get the right arm model for external use
    const pinocchio::Model& getRightArmModel() const { 
        return is_fixed_base_mode_ ? robot_model_ : robot_model_; 
    }

private:
    // Helper method to extract right arm states from full robot state
    void extractRightArmStates(const RobotState& robot_state,
                              Eigen::VectorXd& q_arm,
                              Eigen::VectorXd& v_arm) const;

    // Helper to build right arm mapping
    void buildRightArmMapping();

    // Member variables
    const pinocchio::Model& robot_model_;           // Current model (full or reduced)
    const pinocchio::Model* full_robot_model_ptr_;  // Pointer to full robot model
    double Ki_;                                     // Observer gain
    Eigen::VectorXd r_;                            // Full body residual vector
    Eigen::VectorXd r_w; 
    Eigen::VectorXd r_arm_;                        // Right arm residual vector
    Eigen::VectorXd M_armature_;                   // Motor armature inertias
    Eigen::VectorXd h_prev_;                       // Previous momentum
    Eigen::VectorXd c_prev_;                       // Previous dynamics terms
    Eigen::VectorXd f_estimated;                        // Estimated external forces
    
    // Right arm specific members
    std::vector<pinocchio::JointIndex> right_arm_joint_indices_;  // Joint indices in full model
    std::vector<std::string> right_arm_joint_names_;              // Joint names
    std::vector<int> right_arm_q_indices_;                        // Configuration indices
    std::vector<int> right_arm_v_indices_;                        // Velocity indices
    int right_arm_nv_;                                           // Number of DOF in right arm
    int right_arm_nq_;                                           // Number of configuration variables
    Eigen::MatrixXd S_right_arm_;                               // Selection matrix for velocities
    
    // Fixed base specific members
    Eigen::VectorXd integral_term_arm_;            // Integral term for arm residual
    Eigen::VectorXd h_arm_prev_;                   // Previous arm momentum
    
    // Flags
    bool is_fixed_base_mode_;                      // True for fixed base operation
    bool has_right_arm_joints_;                    // True if right arm joints are specified
    
    // Frame indices
    pinocchio::FrameIndex wrist_frame_idx_;        // Wrist frame index for force estimation

    std::pair<std::pair<Eigen::Vector3d, Eigen::Vector3d>, std::pair<Eigen::Vector3d, Eigen::Vector3d>> 
    getLeftRightFootForcesAndTorques(mjModel* m, mjData* d);

     // NEW: Store WBC-related data
    Eigen::VectorXd left_foot_wrench_wbc_;
    Eigen::VectorXd right_foot_wrench_wbc_;
    Eigen::MatrixXd J_lu_wbc_;
    Eigen::MatrixXd J_ru_wbc_;
    Eigen::VectorXd tau_wbc_;

};

} // namespace labrob

#endif // LABROB_RESIDUAL_ESTIMATOR_HPP_