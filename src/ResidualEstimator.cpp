#include <hrp4_locomotion/ResidualEstimator.hpp>
#include <hrp4_locomotion/utils.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <hrp4_locomotion/WholeBodyController.hpp> 

namespace labrob {

// Constructor for full robot residual estimation
ResidualEstimator::ResidualEstimator(
    const pinocchio::Model& robot_model,
    double Ki,
    const std::map<std::string, double>& armatures,
    const std::vector<std::string>& right_arm_joint_names)
    : robot_model_(robot_model),
      full_robot_model_ptr_(nullptr),
      Ki_(Ki),
      r_(Eigen::VectorXd::Zero(robot_model.nv)),
      r_arm_(Eigen::VectorXd::Zero(right_arm_joint_names.size())),
      M_armature_(Eigen::VectorXd::Zero(robot_model.nv - 6)),
      h_prev_(Eigen::VectorXd::Zero(robot_model.nv)),
      c_prev_(Eigen::VectorXd::Zero(robot_model.nv)),
      right_arm_joint_names_(right_arm_joint_names),
      right_arm_nv_(0),
      right_arm_nq_(0),
      is_fixed_base_mode_(false),
      has_right_arm_joints_(!right_arm_joint_names.empty())
{
    // Initialize armatures for actuated joints
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) robot_model_.njoints; ++joint_id) {
        std::string joint_name = robot_model_.names[joint_id];
        if(armatures.find(joint_name) != armatures.end()) {
            M_armature_(joint_id - 2) = armatures.at(joint_name);
        }
    }

    if (has_right_arm_joints_) {
        buildRightArmMapping();
        integral_term_arm_ = Eigen::VectorXd::Zero(right_arm_nv_);
        h_arm_prev_ = Eigen::VectorXd::Zero(right_arm_nv_);
        
        // Try to find wrist frame
        std::vector<std::string> possible_wrist_frames = {"R_SHOULDER_P", "R_SHOULDER_R",  "R_SHOULDER_Y", "R_ELBOW_P", "R_WRIST_Y"};
        wrist_frame_idx_ = 0;
        for (const auto& frame_name : possible_wrist_frames) {
            if (robot_model_.existFrame(frame_name)) {
                wrist_frame_idx_ = robot_model_.getFrameId(frame_name);
                break;
            }
        }
    }
}

// Updated  Constructor for fixed-base right arm residual estimation
ResidualEstimator::ResidualEstimator(
    const pinocchio::Model& robot_model,
    const pinocchio::Model& full_robot_model,
    double Ki,
    const std::map<std::string, double>& armatures,
    const std::vector<std::string>& right_arm_joint_names)
    : robot_model_(robot_model),
      full_robot_model_ptr_(&full_robot_model),
      Ki_(Ki),
      r_(Eigen::VectorXd::Zero(right_arm_joint_names.size())), // FIXED: Use actual arm joints size
      r_arm_(Eigen::VectorXd::Zero(right_arm_joint_names.size())), // FIXED
      M_armature_(Eigen::VectorXd::Zero(right_arm_joint_names.size())), // FIXED
      h_prev_(Eigen::VectorXd::Zero(right_arm_joint_names.size())), // FIXED
      c_prev_(Eigen::VectorXd::Zero(right_arm_joint_names.size())), // FIXED
      right_arm_joint_names_(right_arm_joint_names),
      right_arm_nv_(right_arm_joint_names.size()), // FIXED: Use actual arm DOF
      right_arm_nq_(right_arm_joint_names.size()), // FIXED
      is_fixed_base_mode_(true),
      has_right_arm_joints_(true)
{
    // Initialize armatures only for the actual arm joints (skip virtual base)
    for(size_t i = 0; i < right_arm_joint_names_.size(); ++i) {
        const std::string& joint_name = right_arm_joint_names_[i];
        if(armatures.find(joint_name) != armatures.end()) {
            M_armature_(i) = armatures.at(joint_name);
            std::cout << "Set armature for " << joint_name << ": " << armatures.at(joint_name) << std::endl;
        } else {
            std::cout << "Warning: No armature found for joint " << joint_name << std::endl;
        }
    }

    // Build mapping for full robot model
    for (const auto& name : right_arm_joint_names_) {
        if (full_robot_model.existJointName(name)) {
            pinocchio::JointIndex joint_id = full_robot_model.getJointId(name);
            right_arm_joint_indices_.push_back(joint_id);
        }
    }

    integral_term_arm_ = Eigen::VectorXd::Zero(right_arm_nv_);
    h_arm_prev_ = Eigen::VectorXd::Zero(right_arm_nv_);

    // Find wrist frame in reduced model
    std::vector<std::string> possible_wrist_frames = {"R_SHOULDER_P",
        "R_SHOULDER_R", 
        "R_SHOULDER_Y",
        "R_ELBOW_P",
        "R_WRIST_Y"};
    wrist_frame_idx_ = 0;
    for (const auto& frame_name : possible_wrist_frames) {
        if (robot_model_.existFrame(frame_name)) {
            wrist_frame_idx_ = robot_model_.getFrameId(frame_name);
            std::cout << "Found wrist frame: " << frame_name << " with ID: " << wrist_frame_idx_ << std::endl;
            break;
        }
    }
}

void ResidualEstimator::buildRightArmMapping() {
    right_arm_nv_ = 0;
    right_arm_nq_ = 0;
    
    // Build joint indices and compute dimensions
    for (const auto& name : right_arm_joint_names_) {
        if (robot_model_.existJointName(name)) {
            pinocchio::JointIndex joint_id = robot_model_.getJointId(name);
            right_arm_joint_indices_.push_back(joint_id);
            
            const auto& joint = robot_model_.joints[joint_id];
            right_arm_nv_ += joint.nv();
            right_arm_nq_ += joint.nq();
            
            // Store configuration and velocity indices
            for (int i = 0; i < joint.nq(); ++i) {
                right_arm_q_indices_.push_back(joint.idx_q() + i);
            }
            for (int i = 0; i < joint.nv(); ++i) {
                right_arm_v_indices_.push_back(joint.idx_v() + i);
            }
        }
    }

    // Build selection matrix
    S_right_arm_ = Eigen::MatrixXd::Zero(right_arm_nv_, robot_model_.nv);
    for (int i = 0; i < right_arm_nv_; ++i) {
        S_right_arm_(i, right_arm_v_indices_[i]) = 1.0;
    }
    
    // Resize arm residual
    r_arm_ = Eigen::VectorXd::Zero(right_arm_nv_);
}

void ResidualEstimator::getRightArmConfiguration(const RobotState& robot_state,
                                                Eigen::VectorXd& q_arm,
                                                Eigen::VectorXd& v_arm) const {
    if (is_fixed_base_mode_) {
        // For fixed base mode, extract from full robot state
        extractRightArmStates(robot_state, q_arm, v_arm);
    } else {
        // For full robot mode, use selection approach
        auto q_full = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
        auto v_full = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);
        
        q_arm = Eigen::VectorXd::Zero(right_arm_nq_);
        v_arm = Eigen::VectorXd::Zero(right_arm_nv_);
        
        for (int i = 0; i < right_arm_nq_; ++i) {
            q_arm(i) = q_full(right_arm_q_indices_[i]);
        }
        for (int i = 0; i < right_arm_nv_; ++i) {
            v_arm(i) = v_full(right_arm_v_indices_[i]);
        }
    }
}
void ResidualEstimator::extractRightArmStates(const RobotState& robot_state,
                                             Eigen::VectorXd& q_arm,
                                             Eigen::VectorXd& v_arm) const {
    if (!is_fixed_base_mode_ || !full_robot_model_ptr_) {
        throw std::runtime_error("extractRightArmStates can only be called in fixed base mode");
    }

    // Get full robot configuration
    Eigen::VectorXd q_full = robot_state_to_pinocchio_joint_configuration(*full_robot_model_ptr_, robot_state);
    Eigen::VectorXd v_full = robot_state_to_pinocchio_joint_velocity(*full_robot_model_ptr_, robot_state);
    
    // CRITICAL FIX: Print model dimensions for debugging
    // static bool debug_printed = false;
    // if (!debug_printed) {
    //     std::cout << "=== MODEL DEBUG INFO ===" << std::endl;
    //     std::cout << "Full robot model nq: " << full_robot_model_ptr_->nq << ", nv: " << full_robot_model_ptr_->nv << std::endl;
    //     std::cout << "Reduced arm model nq: " << robot_model_.nq << ", nv: " << robot_model_.nv << std::endl;
    //     std::cout << "Expected arm DOF: " << right_arm_joint_names_.size() << std::endl;
    //     std::cout << "Right arm joint indices in full model: ";
    //     for (auto idx : right_arm_joint_indices_) {
    //         std::cout << idx << " ";
    //     }
    //     std::cout << std::endl;
        
    //     // Print joint structure of reduced model
    //     std::cout << "Reduced model joint structure:" << std::endl;
    //     for (size_t i = 0; i < robot_model_.njoints; ++i) {
    //         std::cout << "Joint " << i << ": " << robot_model_.names[i] 
    //                   << " nq=" << robot_model_.joints[i].nq() 
    //                   << " nv=" << robot_model_.joints[i].nv()
    //                   << " idx_q=" << robot_model_.joints[i].idx_q()
    //                   << " idx_v=" << robot_model_.joints[i].idx_v() << std::endl;
    //     }
    //     debug_printed = true;
    // }
    
    // FIXED: Initialize with correct reduced model dimensions
    q_arm = Eigen::VectorXd::Zero(robot_model_.nq);
    v_arm = Eigen::VectorXd::Zero(robot_model_.nv);
    
    // CRITICAL FIX: Handle different model structures correctly
    if (robot_model_.nv == right_arm_joint_names_.size()) {
        // Case 1: Pure arm model (no virtual base) - direct mapping
        std::cout << "Using direct mapping (no virtual base)" << std::endl;
        for (size_t i = 0; i < right_arm_joint_indices_.size(); ++i) {
            pinocchio::JointIndex full_joint_id = right_arm_joint_indices_[i];
            
            // Get indices in full model
            int full_joint_idx_q = full_robot_model_ptr_->joints[full_joint_id].idx_q();
            int full_joint_idx_v = full_robot_model_ptr_->joints[full_joint_id].idx_v();
            int full_joint_nq = full_robot_model_ptr_->joints[full_joint_id].nq();
            int full_joint_nv = full_robot_model_ptr_->joints[full_joint_id].nv();
            
            // Direct mapping to reduced model
            for (int j = 0; j < full_joint_nq; ++j) {
                q_arm(i * full_joint_nq + j) = q_full(full_joint_idx_q + j);
            }
            for (int j = 0; j < full_joint_nv; ++j) {
                v_arm(i * full_joint_nv + j) = v_full(full_joint_idx_v + j);
            }
        }
    } else {
        // Case 2: Model with virtual base - use joint mapping
        //std::cout << "Using joint mapping (with virtual base)" << std::endl;
        for (size_t i = 0; i < right_arm_joint_indices_.size(); ++i) {
            pinocchio::JointIndex full_joint_id = right_arm_joint_indices_[i];
            // Get indices in full model
            int full_joint_idx_q = full_robot_model_ptr_->joints[full_joint_id].idx_q();
            int full_joint_idx_v = full_robot_model_ptr_->joints[full_joint_id].idx_v();
            int full_joint_nq = full_robot_model_ptr_->joints[full_joint_id].nq();
            int full_joint_nv = full_robot_model_ptr_->joints[full_joint_id].nv();
            
            // Find corresponding joint in reduced model by name
            std::string joint_name = full_robot_model_ptr_->names[full_joint_id];
            pinocchio::JointIndex reduced_joint_id = 0;
            
            // Find joint in reduced model
            for (size_t j = 0; j < robot_model_.njoints; ++j) {
                if (robot_model_.names[j] == joint_name) {
                    reduced_joint_id = j;
                    break;
                }
            }
            
            if (reduced_joint_id > 0) {
                int reduced_joint_idx_q = robot_model_.joints[reduced_joint_id].idx_q();
                int reduced_joint_idx_v = robot_model_.joints[reduced_joint_id].idx_v();
                
                // Copy states
                for (int j = 0; j < full_joint_nq; ++j) {
                    q_arm(reduced_joint_idx_q + j) = q_full(full_joint_idx_q + j);
                }
                for (int j = 0; j < full_joint_nv; ++j) {
                    v_arm(reduced_joint_idx_v + j) = v_full(full_joint_idx_v + j);
                }
            }
        }
    }
    
    // Debug: Print extracted values
    static int extract_counter = 0;
    if (extract_counter % 100 == 0) {
        std::cout << "Extracted q_arm: " << q_arm.transpose() << std::endl;
        std::cout << "Extracted v_arm: " << v_arm.transpose() << std::endl;
    }
    extract_counter++;
}


// // Full robot residual computation 
// Eigen::VectorXd ResidualEstimator::computeResidual(
//     const RobotState& robot_state,
//     pinocchio::Data& robot_data,
//     const Eigen::VectorXd& measured_torques,
//     const Eigen::MatrixXd& J_l,
//     const Eigen::MatrixXd& J_r,
//     const Eigen::VectorXd& W_l,
//     const Eigen::VectorXd& W_r,
//     double dt,
//     mjModel* mj_model,        // model
//     mjData* mj_data)          //mjdata  
// {
//     static Eigen::VectorXd integral_term_ = Eigen::VectorXd::Zero(robot_model_.nv);

//     auto q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
//     auto v = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);

//     //------contact forces from mujoco
//     // Extract foot forces and torques from MuJoCo
//     auto [left_data, right_data] = getLeftRightFootForcesAndTorques(mj_model, mj_data);
//     auto [left_foot_force, left_foot_torque] = left_data;
//     auto [right_foot_force, right_foot_torque] = right_data;
    
//     // Create combined force vectors for your residual computation
//     Eigen::VectorXd mujoco_left_wrench(6);
//     mujoco_left_wrench << left_foot_force, left_foot_torque;
    
//     Eigen::VectorXd mujoco_right_wrench(6);
//     mujoco_right_wrench << right_foot_force, right_foot_torque;
// //------

    
//     pinocchio::crba(robot_model_, robot_data, q);
//     Eigen::MatrixXd H = robot_data.M;
//     H.diagonal().tail(robot_model_.nv - 6) += M_armature_;
    
//     Eigen::MatrixXd C = pinocchio::computeCoriolisMatrix(robot_model_, robot_data, q, v);
//     pinocchio::computeGeneralizedGravity(robot_model_, robot_data, q);
//     Eigen::VectorXd g = robot_data.g;

//     Eigen::VectorXd tau_m = Eigen::VectorXd::Zero(robot_model_.nv);
//     tau_m.tail(robot_model_.nv - 6) = measured_torques;
//     //r_.setZero();
//      // 2. Left wrist Jacobian
//     Eigen::MatrixXd J_fl(6, robot_model_.nv); 
//     J_fl.setZero();
//     pinocchio::getFrameJacobian(robot_model_, robot_data,
//     robot_model_.getFrameId("l_foot"),
//     pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_fl);

//     // 2. Left wrist Jacobian
//     Eigen::MatrixXd J_fr(6, robot_model_.nv); 
//     J_fr.setZero();
//     pinocchio::getFrameJacobian(robot_model_, robot_data,
//     robot_model_.getFrameId("r_foot"),
//     pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_fr);

//     Eigen::VectorXd m_contact_forces = J_fl.transpose() * mujoco_left_wrench + J_fr.transpose() * mujoco_right_wrench;
//     Eigen::VectorXd contact_forces = J_l.transpose() * W_l + J_r.transpose() * W_r;
//     Eigen::VectorXd C_T_v = C.transpose() * v;
    
//     //Eigen::VectorXd integrand = tau_m + contact_forces + C_T_v - g + r_;
//     Eigen::VectorXd integrand = tau_m + m_contact_forces + C_T_v - g + r_;
//     integral_term_ += dt * integrand;
//     Ki_ = 5;
//     Eigen::VectorXd h_q_dot = H * v;
//     r_ = Ki_ * (h_q_dot - integral_term_);

//     // Force estimation for torso (your existing code)
//     int N = robot_model_.nv; // 

//     // Extract the last N elements (joint part of residual)
//     Eigen::VectorXd r_joints = r_.tail(N);  // size N x 1

//     // Get torso Jacobian
//     Eigen::MatrixXd J_torso(6, robot_model_.nv); 
//     J_torso.setZero(); // Initialize to zero
//     const std::string torso_frame_name = "R_WRIST_Y_S";  
//     pinocchio::getFrameJacobian(robot_model_, robot_data, robot_model_.getFrameId(torso_frame_name), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_torso);
    
//     Eigen::MatrixXd J_torso_F = J_torso.topRows(3); // 3 x nv

//     Eigen::MatrixXd J_torso_T = J_torso_F.transpose();

//     // 1. Right wrist Jacobian (assume task frame) -------------------------------------------------------
//     Eigen::MatrixXd J_r_wrist(6, robot_model_.nv); 
//     J_r_wrist.setZero();
//     pinocchio::getFrameJacobian(robot_model_, robot_data,
//     robot_model_.getFrameId("R_WRIST_Y_S"),
//     pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_r_wrist);

//     // 2. Left wrist Jacobian
//     Eigen::MatrixXd J_l_wrist(6, robot_model_.nv); 
//     J_l_wrist.setZero();
//     pinocchio::getFrameJacobian(robot_model_, robot_data,
//     robot_model_.getFrameId("L_WRIST_Y_S"),
//     pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_l_wrist);

//     Eigen::MatrixXd J_r_F = J_r_wrist.topRows(3); // 3 x nv
//     Eigen::MatrixXd J_l_F = J_l_wrist.topRows(3); // 3 x nv

//     // J_l_F and J_r_F are both 3 x nv
//     Eigen::MatrixXd J_stacked(6, robot_model_.nv); // 6 x nv
//     J_stacked.topRows(3) = J_l_F; // First 3 rows: left wrist
//     J_stacked.bottomRows(3) = J_r_F; // Last 3 rows: right wrist

//     //Eigen::MatrixXd J_stacked(12, robot_model_.nv); // 6 x nv
//     ///J_stacked.setZero();
//     // J_stacked.topRows(6) = J_l_wrist; // First 3 rows: left wrist
//     // J_stacked.bottomRows(6) = J_r_wrist; // Last 3 rows: right wrist

//     Eigen::MatrixXd J_stacked_T = J_stacked.transpose();



//     //Eigen::MatrixXd J_torso_pinv = pseudoInverse(J_torso_transpose);

//     //Eigen::MatrixXd J_torso_pinv = J_torso.completeOrthogonalDecomposition().pseudoInverse();
//    // Eigen::MatrixXd J_torso_transpose = J_torso.transpose();


//     //Eigen::MatrixXd J_torso_pinv = J_torso_T.completeOrthogonalDecomposition().pseudoInverse();
//     Eigen::MatrixXd J_torso_pinv = J_stacked_T.completeOrthogonalDecomposition().pseudoInverse();

//     //Eigen::VectorXd f_torso = computeTorsoForce_SVD(J_torso, r_, 1e-6);


//     // std::ofstream q_file;
//     // q_file.open("/tmp/joint_positions.txt", std::ios_base::app);

//     // if (q_file.is_open()) {
//     //     for (int row = 0; row < J_torso_pinv.rows(); ++row) {
//     //         for (int col = 0; col < J_torso_pinv.cols(); ++col) {
//     //             q_file << J_torso_pinv(row, col) << " ";
//     //         }
//     //     }
//     //     q_file << std::endl;
//     //     q_file.close();
//     // }


    


//     Eigen::VectorXd f_torso = J_torso_pinv * r_;  


//     //------------------------real contact force from mujoco



//     // Log results
//     std::ofstream force_file("/tmp/computed_force.txt", std::ios_base::app);
//     if (force_file.is_open()) {
//         force_file << f_torso.head(6).transpose() << std::endl;
//         force_file.close();
//     }

//     std::ofstream residual_file("/tmp/residual.txt", std::ios_base::app);
//     if (residual_file.is_open()) {
//         residual_file << r_.head(6).transpose() << std::endl;
//         residual_file.close();
//     }

//     return r_;
// }


Eigen::VectorXd ResidualEstimator::computeResidualWithWBCWrenches(
    const RobotState& robot_state,
    pinocchio::Data& robot_data,
    const Eigen::VectorXd& measured_torques,
    //const Eigen::VectorXd& wbc_left_foot_wrench,
    //const Eigen::VectorXd& wbc_right_foot_wrench,
    const labrob::WholeBodyController& wbc_controller,
    double dt)
{
    static Eigen::VectorXd integral_term_ = Eigen::VectorXd::Zero(robot_model_.nv);

    auto q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
    auto v = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);

    // Compute dynamics terms
    pinocchio::crba(robot_model_, robot_data, q);
    Eigen::MatrixXd H = robot_data.M;
    H.diagonal().tail(robot_model_.nv - 6) += M_armature_;
    
    Eigen::MatrixXd C = pinocchio::computeCoriolisMatrix(robot_model_, robot_data, q, v);
    pinocchio::computeGeneralizedGravity(robot_model_, robot_data, q);
    Eigen::VectorXd g = robot_data.g;

    // Get foot Jacobians for transforming wrenches
    Eigen::MatrixXd J_left_foot(6, robot_model_.nv); 
    J_left_foot.setZero();
    pinocchio::getFrameJacobian(robot_model_, robot_data,
        robot_model_.getFrameId("left_ankle_roll_link"),
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_left_foot);

    Eigen::MatrixXd J_right_foot(6, robot_model_.nv); 
    J_right_foot.setZero();
    pinocchio::getFrameJacobian(robot_model_, robot_data,
        robot_model_.getFrameId("right_ankle_roll_link"),
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_right_foot);
    // Use underactuated Jacobians from WBC
    const Eigen::MatrixXd& Jlu = wbc_controller.getLeftFootUnderactuatedJacobian();
    const Eigen::MatrixXd& Jru = wbc_controller.getRightFootUnderactuatedJacobian();
    
    // Get full wrenches from WBC
    const Eigen::VectorXd& wbc_left_foot_wrench = wbc_controller.getLeftFootWrench();
    const Eigen::VectorXd& wbc_right_foot_wrench = wbc_controller.getRightFootWrench();

    // For underactuated Jacobians, you might need to extract only the relevant components
    // This depends on how your WBC formulates the problem
    //Eigen::VectorXd contact_forces = Jlu.transpose() * wbc_left_foot_wrench + Jru.transpose() * wbc_right_foot_wrench;
    
    


    // Use WBC-computed wrenches
    Eigen::VectorXd contact_forces = J_left_foot.transpose() * wbc_left_foot_wrench + J_right_foot.transpose() * wbc_right_foot_wrench;
    //std::cout <<"Wrenches Right:" <<wbc_right_foot_wrench.transpose() << std::endl;
    // Measured torques
    // std::ofstream q_file;
    // q_file.open("/tmp/joint_positions.txt", std::ios_base::app);

    // if (q_file.is_open()) {
    //     for (int row = 0; row < contact_forces.rows(); ++row) {
    //         for (int col = 0; col < contact_forces.cols(); ++col) {
    //             q_file << contact_forces(row, col) << " ";
    //         }
    //     }
    //     q_file << std::endl;
    //     q_file.close();
    // }

    Eigen::VectorXd tau_m = Eigen::VectorXd::Zero(robot_model_.nv);
    tau_m.tail(robot_model_.nv - 6) = measured_torques;
    
    Eigen::VectorXd r_w;
    // The terms to compute the residual
    Eigen::VectorXd C_T_v = C.transpose() * v;
    Eigen::VectorXd integrand = tau_m + C_T_v - g + r_;
    integral_term_ += dt * integrand;
    
    Ki_ = 20;
    Eigen::VectorXd h_q_dot = H * v;
    r_ = Ki_ * (h_q_dot - integral_term_);

    
    // Force re-contruction from the residual.
    Eigen::MatrixXd J_left(6, robot_model_.nv); 
    J_left.setZero();
    pinocchio::getFrameJacobian(robot_model_, robot_data,
        robot_model_.getFrameId("left_ankle_roll_link"),
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_left);

    Eigen::MatrixXd J_right(6, robot_model_.nv); 
    J_right.setZero();
    pinocchio::getFrameJacobian(robot_model_, robot_data,
        robot_model_.getFrameId("right_ankle_roll_link"),
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_right);

    // I am doing this for segregation of forces and torques elements of the jacobian, as the first 3 contains the linear elemets
    Eigen::MatrixXd J_left_pos = J_left.topRows(3);
    Eigen::MatrixXd J_right_pos = J_right.topRows(3);

    // Stack Jacobians
    Eigen::MatrixXd J_stacked(6, robot_model_.nv);
    J_stacked.topRows(3) = J_left_pos;
    J_stacked.bottomRows(3) = J_right_pos;

    Eigen::MatrixXd J_stacked_T = J_stacked.transpose();
    Eigen::MatrixXd J_pinv = J_stacked_T.completeOrthogonalDecomposition().pseudoInverse();
    
    // Estimate forces
    f_estimated = J_pinv * r_;
    // std::cout << "debug: " << f_estimated.head(3).transpose() << std::endl;

    // Log results - ensure we write 6 values per line for proper parsing
    // std::ofstream force_file("/tmp/computed_force_wbc.txt", std::ios_base::app);
    // if (force_file.is_open()) {
    //     // Write all 6 components on one line
    //     for (int i = 0; i < 6; ++i) {
    //         if (i < f_estimated.size()) {
    //             force_file << f_estimated(i) << " ";
    //         } else {
    //             force_file << "0.0 ";  // Pad if needed
    //         }
    //     }
    //     force_file << std::endl;
    //     force_file.close();
    // }
    
    // std::ofstream comp_file("/tmp/computed_force_wbc.txt", std::ios_base::app);
    // if (comp_file.is_open()) {
    //     comp_file << f_estimated.head(6).transpose() << std::endl;
    //     comp_file.close();
    // }

    // std::ofstream residual_file_w("/tmp/residual_wbc.txt", std::ios_base::app);
    // if (residual_file_w.is_open()) {
    //     // Write first 6 residual components
    //     for (int i = 0; i < 6 && i < r_.size(); ++i) {
    //         residual_file_w << r_(i) << " ";
    //     }
    //     residual_file_w << std::endl;
    //     residual_file_w.close();
    // }

    return r_;
}

// Eigen::VectorXd ResidualEstimator::getEstimatedForce() const {
//     return f_estimated;
// }

// Right arm residual with full robot dynamics (selection matrix approach)
Eigen::VectorXd ResidualEstimator::computeRightArmResidual(
    const RobotState& robot_state,
    pinocchio::Data& robot_data,
    const Eigen::VectorXd& measured_torques,
    const Eigen::MatrixXd& J_l,
    const Eigen::MatrixXd& J_r, 
    const Eigen::VectorXd& W_l,
    const Eigen::VectorXd& W_r,
    double dt)
{
    if (!has_right_arm_joints_ || is_fixed_base_mode_) {
        throw std::runtime_error("computeRightArmResidual requires full robot mode with right arm joints");
    }

    static Eigen::VectorXd integral_term_arm = Eigen::VectorXd::Zero(right_arm_nv_);

    auto q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
    auto v = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);

    // Compute dynamics terms
    pinocchio::crba(robot_model_, robot_data, q);
    Eigen::MatrixXd H = robot_data.M;
    H.diagonal().tail(robot_model_.nv - 6) += M_armature_;
    
    Eigen::MatrixXd C = pinocchio::computeCoriolisMatrix(robot_model_, robot_data, q, v);
    pinocchio::computeGeneralizedGravity(robot_model_, robot_data, q);
    Eigen::VectorXd g = robot_data.g;

    // Project to right arm subspace
    Eigen::MatrixXd H_arm = S_right_arm_ * H * S_right_arm_.transpose();
    Eigen::MatrixXd C_arm = S_right_arm_ * C * S_right_arm_.transpose();
    Eigen::VectorXd g_arm = S_right_arm_ * g;
    Eigen::VectorXd v_arm = S_right_arm_ * v;

    // Get right arm torques
    Eigen::VectorXd tau_m_arm = Eigen::VectorXd::Zero(right_arm_nv_);
    for (int i = 0; i < right_arm_nv_; ++i) {
        int joint_idx = right_arm_v_indices_[i] - 6; // Subtract 6 for floating base
        if (joint_idx >= 0 && joint_idx < measured_torques.size()) {
            tau_m_arm(i) = measured_torques(joint_idx);
        }
    }

    // Compute residual
    Eigen::VectorXd contact_forces_arm = S_right_arm_ * (J_l.transpose() * W_l + J_r.transpose() * W_r);
    Eigen::VectorXd C_T_v_arm = C_arm.transpose() * v_arm;
    Eigen::VectorXd integrand = tau_m_arm + C_T_v_arm - g_arm + r_arm_;
    integral_term_arm += dt * integrand;

    Eigen::VectorXd h_arm = H_arm * v_arm;
    r_arm_ = Ki_ * (h_arm - integral_term_arm);

    // Estimate wrist force
    Eigen::VectorXd f_wrist = estimateWristForce(r_arm_, robot_data);

    // Log results
    std::ofstream arm_file("/tmp/residual_right_arm.txt", std::ios_base::app);
    if (arm_file.is_open()) {
        arm_file << r_arm_.transpose() << std::endl;
        arm_file.close();
    }

    std::ofstream force_file("/tmp/arm_force.txt", std::ios_base::app);
    if (force_file.is_open()) {
        force_file << f_wrist.transpose() << std::endl;
        force_file.close();
    }

    return r_arm_;
}

// Improved fixed base residual computation
Eigen::VectorXd ResidualEstimator::computeRightArmResidualFixedBase(
    const RobotState& robot_state,
    pinocchio::Data& right_arm_data,
    const Eigen::VectorXd& measured_torques_right_arm,
    double dt)
{
    if (!is_fixed_base_mode_) {
        throw std::runtime_error("computeRightArmResidualFixedBase can only be called in fixed base mode");
    }

    // Extract right arm joint states from full robot state
    Eigen::VectorXd q_arm, v_arm;
    extractRightArmStates(robot_state, q_arm, v_arm);
    
    // DEBUG: Check dimensions and values
    // static int debug_counter = 0;
    // if (debug_counter % 100 == 0) {
    //     std::cout << "=== RESIDUAL DEBUG " << debug_counter << " ===" << std::endl;
    //     std::cout << "q_arm size: " << q_arm.size() << ", content: " << q_arm.transpose() << std::endl;
    //     std::cout << "v_arm size: " << v_arm.size() << ", content: " << v_arm.transpose() << std::endl;
    //     std::cout << "measured_torques size: " << measured_torques_right_arm.size() 
    //               << ", content: " << measured_torques_right_arm.transpose() << std::endl;
    //     std::cout << "Expected arm DOF: " << right_arm_joint_names_.size() << std::endl;
    //     std::cout << "Model nv: " << robot_model_.nv << ", nq: " << robot_model_.nq << std::endl;
    // }

    // Compute dynamics terms using the reduced right arm model
    pinocchio::crba(robot_model_, right_arm_data, q_arm);
    Eigen::MatrixXd H_full = right_arm_data.M;
    
    // CRITICAL FIX: Determine the actual structure of the reduced model
    int expected_arm_dof = right_arm_joint_names_.size();  // 5 DOF
    int model_total_dof = robot_model_.nv;
    
    Eigen::MatrixXd H;
    Eigen::VectorXd v_arm_actuated;
    Eigen::VectorXd g_arm_actuated;
    Eigen::MatrixXd C;
    
    if (model_total_dof == expected_arm_dof) {
        // Case 1: Pure arm model (5x5) - no virtual base
        std::cout << "Using pure arm model (no virtual base)" << std::endl;
        H = H_full;
        H.diagonal() += M_armature_;
        v_arm_actuated = v_arm;
        
        // Compute Coriolis and gravity
        C = pinocchio::computeCoriolisMatrix(robot_model_, right_arm_data, q_arm, v_arm);
        pinocchio::computeGeneralizedGravity(robot_model_, right_arm_data, q_arm);
        g_arm_actuated = right_arm_data.g;
        
    } else if (model_total_dof > expected_arm_dof) {
        // Case 2: Model with virtual base - extract actuated part
        int virtual_base_dof = model_total_dof - expected_arm_dof;
        // std::cout << "Using model with virtual base, extracting " << expected_arm_dof 
        //           << " DOF from " << model_total_dof << " total DOF" << std::endl;
        
        if (H_full.rows() < virtual_base_dof + expected_arm_dof) {
            std::cerr << "Error: Model size mismatch. Expected at least " 
                      << virtual_base_dof + expected_arm_dof << " but got " << H_full.rows() << std::endl;
            return Eigen::VectorXd::Zero(expected_arm_dof);
        }
        
        // Extract only the actuated joints part
        H = H_full.block(virtual_base_dof, virtual_base_dof, expected_arm_dof, expected_arm_dof);
        H.diagonal() += M_armature_;
        v_arm_actuated = v_arm.tail(expected_arm_dof);
        
        // Compute Coriolis and gravity for full model, then extract
        C = pinocchio::computeCoriolisMatrix(robot_model_, right_arm_data, q_arm, v_arm);
        pinocchio::computeGeneralizedGravity(robot_model_, right_arm_data, q_arm);
        
        // Extract actuated parts
        Eigen::MatrixXd C_actuated = C.block(virtual_base_dof, virtual_base_dof, expected_arm_dof, expected_arm_dof);
        g_arm_actuated = right_arm_data.g.tail(expected_arm_dof);
        C = C_actuated;
        
    } else {
        std::cerr << "Error: Model has fewer DOF than expected arm joints!" << std::endl;
        return Eigen::VectorXd::Zero(expected_arm_dof);
    }
    
    // Verify dimensions match
    if (measured_torques_right_arm.size() != expected_arm_dof) {
        // std::cerr << "Error: Torque vector size " << measured_torques_right_arm.size() 
        //           << " doesn't match expected arm DOF " << expected_arm_dof << std::endl;
        return Eigen::VectorXd::Zero(expected_arm_dof);
    }
    
    if (r_.size() != expected_arm_dof) {
        r_ = Eigen::VectorXd::Zero(expected_arm_dof);
        integral_term_arm_ = Eigen::VectorXd::Zero(expected_arm_dof);
    }
    
    // Current generalized momentum (arm joints only)
    Eigen::VectorXd h_current = H * v_arm_actuated;
    
    // Momentum observer implementation
    Eigen::VectorXd coriolis_forces = C.transpose() * v_arm_actuated;
    
    // Update integral term
    Eigen::VectorXd integrand = measured_torques_right_arm + coriolis_forces - g_arm_actuated + r_;
    integral_term_arm_ += dt * integrand;
    Ki_ = 5;
    // Compute residual
    r_ = Ki_ * (h_current - integral_term_arm_);

    // Eigen::MatrixXd JWrist(6, robot_model_.nv); 
    // JWrist.setZero();
    // pinocchio::getFrameJacobian(robot_model_, robot_data, 
    // robot_model_.getFrameId("R_WRIST_Y_S"),
    // pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, JWrist);

    // Eigen::MatrixXd JWrist_T = JWrist.transpose();

    // Debug output
    // if (debug_counter % 100 == 0) {
    //     std::cout << "h_current: " << h_current.transpose() << std::endl;
    //     std::cout << "integral_term: " << integral_term_arm_.transpose() << std::endl;
    //     std::cout << "residual: " << r_.transpose() << std::endl;
    //     std::cout << "coriolis_forces: " << coriolis_forces.transpose() << std::endl;
    //     std::cout << "gravity: " << g_arm_actuated.transpose() << std::endl;
    // }
    // debug_counter++;
    std::vector<std::string> right_arm_joint_names_ = {
        "R_SHOULDER_P", 
        "R_SHOULDER_R", 
        "R_SHOULDER_Y",
        "R_ELBOW_P", 
        "R_WRIST_Y"
    };

// Get the full Jacobian (6 x nv)
    Eigen::MatrixXd J_full(6, robot_model_.nv);
    J_full.setZero();
    pinocchio::getFrameJacobian(
        robot_model_, 
        right_arm_data, 
        robot_model_.getFrameId("R_WRIST_Y_S"), 
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
        J_full);

    // Prepare reduced Jacobian (6 x 5)
    int n_joints = right_arm_joint_names_.size();
    Eigen::MatrixXd J_reduced(6, n_joints);
    J_reduced.setZero();

    // Map joint names to velocity indices and slice Jacobian
    for (int i = 0; i < n_joints; ++i) {
        const std::string& joint_name = right_arm_joint_names_[i];
        pinocchio::JointIndex joint_id = robot_model_.getJointId(joint_name);

        if (joint_id == 0) {
            std::cerr << "Joint name " << joint_name << " not found in model!" << std::endl;
            continue;
        }

        int idx_v = robot_model_.joints[joint_id].idx_v();  // Velocity index
        J_reduced.col(i) = J_full.col(idx_v);
    }
    


    
    // Use only position part (first 3 rows) for force estimation
    //std::cout<<"Size of the J Matrix"<<J_Iarm.size()<<std::endl;
    
    Eigen::MatrixXd J_reduced_T = J_reduced.transpose();  
    Eigen::MatrixXd J_reduced_T_pinv = J_reduced_T.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::VectorXd f_Iarm = J_reduced_T_pinv * r_;  

    // Estimate Cartesian force at wrist
    Eigen::VectorXd f_wrist = estimateWristForce(r_, right_arm_data);

    // Log results
    std::ofstream log_file("/tmp/arm_residual_fixed_base.txt", std::ios_base::app);
    if (log_file.is_open()) {
        log_file << r_.transpose() << std::endl;
        log_file.close();
    }

    std::ofstream force_file("/tmp/arm_force_fixed_base.txt", std::ios_base::app);
    if (force_file.is_open()) {
        force_file << f_Iarm.transpose() << std::endl;
        force_file.close();
    }

    return r_;
}

// force torque extraction from mujoco 
std::pair<std::pair<Eigen::Vector3d, Eigen::Vector3d>, std::pair<Eigen::Vector3d, Eigen::Vector3d>> 
ResidualEstimator::getLeftRightFootForcesAndTorques(mjModel* m, mjData* d) {
    Eigen::Vector3d left_foot_force = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_force = Eigen::Vector3d::Zero();
    Eigen::Vector3d left_foot_torque = Eigen::Vector3d::Zero();
    Eigen::Vector3d right_foot_torque = Eigen::Vector3d::Zero();
    
    static double force[6];  // [fx, fy, fz, mx, my, mz]
    static double result_force[3];
    
    // Get ankle body IDs for torque calculation
    int left_ankle_id = mj_name2id(m, mjOBJ_BODY, "L_ANKLE_P_S");
    int right_ankle_id = mj_name2id(m, mjOBJ_BODY, "R_ANKLE_P_S");
    
    // If the exact names don't match, try these alternatives:
    if (left_ankle_id < 0) left_ankle_id = mj_name2id(m, mjOBJ_BODY, "L_ANKLE");
    if (right_ankle_id < 0) right_ankle_id = mj_name2id(m, mjOBJ_BODY, "LEFT_ANKLE");
    if (right_ankle_id < 0) right_ankle_id = mj_name2id(m, mjOBJ_BODY, "R_ANKLE");
    if (right_ankle_id < 0) right_ankle_id = mj_name2id(m, mjOBJ_BODY, "RIGHT_ANKLE");
    
    for (int i = 0; i < d->ncon; ++i) {
        mj_contactForce(m, d, i, force);
        
        // Transform force to world frame
        mju_mulMatTVec(result_force, d->contact[i].frame, force, 3, 3);
        
        // Check which foot this contact belongs to
        int geom1 = d->contact[i].geom1;
        int geom2 = d->contact[i].geom2;
        int body1 = m->geom_bodyid[geom1];
        int body2 = m->geom_bodyid[geom2];
        
        const char* body1_name = mj_id2name(m, mjOBJ_BODY, body1);
        const char* body2_name = mj_id2name(m, mjOBJ_BODY, body2);
        
        bool is_left_foot = false;
        bool is_right_foot = false;
        int ankle_body_id = -1;
        
        if (body1_name) {
            std::string name1(body1_name);
            if (name1.find("L_ANKLE_P_S") != std::string::npos || name1.find("LEFT") != std::string::npos) {
                is_left_foot = true;
                ankle_body_id = left_ankle_id;
            }
            else if (name1.find("R_ANKLE_P_S") != std::string::npos || name1.find("RIGHT") != std::string::npos) {
                is_right_foot = true;
                ankle_body_id = right_ankle_id;
            }
        }
        if (body2_name && !is_left_foot && !is_right_foot) {
            std::string name2(body2_name);
            if (name2.find("L_ANKLE_P_S") != std::string::npos || name2.find("LEFT") != std::string::npos) {
                is_left_foot = true;
                ankle_body_id = left_ankle_id;
            }
            else if (name2.find("R_ANKLE_P_S") != std::string::npos || name2.find("RIGHT") != std::string::npos) {
                is_right_foot = true;
                ankle_body_id = right_ankle_id;
            }
        }
        
        if (is_left_foot || is_right_foot) {
            Eigen::Vector3d contact_force(result_force[0], result_force[1], result_force[2]);
            
            // Calculate torque about ankle joint
            Eigen::Vector3d contact_torque = Eigen::Vector3d::Zero();
            
            if (ankle_body_id >= 0) {
                // Get contact point position
                Eigen::Vector3d contact_pos(d->contact[i].pos[0], d->contact[i].pos[1], d->contact[i].pos[2]);
                
                // Get ankle position
                Eigen::Vector3d ankle_pos(d->xipos[ankle_body_id * 3 + 0],
                                        d->xipos[ankle_body_id * 3 + 1],
                                        d->xipos[ankle_body_id * 3 + 2]);
                
                // Calculate moment arm (vector from ankle to contact point)
                Eigen::Vector3d moment_arm = contact_pos - ankle_pos;
                
                // Calculate torque = moment_arm × force
                contact_torque = moment_arm.cross(contact_force);
            }
            
            if (is_left_foot) {
                left_foot_force += contact_force;
                left_foot_torque += contact_torque;
            } else if (is_right_foot) {
                right_foot_force += contact_force;
                right_foot_torque += contact_torque;
            }
        }
    }
    
    return std::make_pair(
        std::make_pair(left_foot_force, left_foot_torque),
        std::make_pair(right_foot_force, right_foot_torque)
    );
}
//-------------------------------------Force Extraction from Mujoco ^

Eigen::VectorXd ResidualEstimator::estimateWristForce(const Eigen::VectorXd& arm_residual,
                                                     pinocchio::Data& arm_data) const {
    try {
        // Get wrist Jacobian
        Eigen::MatrixXd J_wrist(6, robot_model_.nv);
        J_wrist.setZero();

        pinocchio::getFrameJacobian(robot_model_, arm_data, robot_model_.getFrameId("R_WRIST_Y_S"),
                                   pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_wrist);

        // Use only position part (first 3 rows) for force estimation
        Eigen::MatrixXd J_pos1 = J_wrist.rightCols(5);
        Eigen::MatrixXd J_pos = J_pos1.topRows(3);

        Eigen::MatrixXd J_pos_T = J_pos.transpose();    
        
        // Compute pseudoinverse
        Eigen::MatrixXd J_pinv = J_pos_T.completeOrthogonalDecomposition().pseudoInverse();
        
        // Estimate force
        Eigen::VectorXd f_wrist = J_pinv * arm_residual;
        //std::cout<<"Size of the J" << f_wrist << std::endl;
        
        return f_wrist;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in wrist force estimation: " << e.what() << std::endl;
        return Eigen::VectorXd::Zero(3);
    }
}

} // namespace labrob