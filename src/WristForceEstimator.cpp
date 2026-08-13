#include <WristForceEstimator.hpp>
#include <utils.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include <WholeBodyController.hpp>
#include <iomanip>
#include <utility> 

namespace labrob {

    // Constructor for full robot residual estimation
    WristForceEstimator::WristForceEstimator(
        const pinocchio::Model& robot_model,
        const std::map<std::string, double>& armatures,
        double Ki,
        const std::string& right_wrist_frame,
        const std::string& left_wrist_frame,
        const std::string& right_foot_frame,
        const std::string& left_foot_frame)
        : robot_model_(robot_model),
        M_armature_(Eigen::VectorXd::Zero(robot_model.nv - 6)),
        M_(Eigen::MatrixXd::Zero(robot_model.nv, robot_model.nv)),
        C_(Eigen::MatrixXd::Zero(robot_model.nv, robot_model.nv)),
        g_(Eigen::VectorXd::Zero(robot_model.nv)),
        tau_m_(Eigen::VectorXd::Zero(robot_model.nv)),
        p_(Eigen::VectorXd::Zero(robot_model.nv)),
        p0_(Eigen::VectorXd::Zero(robot_model.nv)),
        J_rw_(Eigen::MatrixXd::Zero(6, robot_model.nv)),
        J_lw_(Eigen::MatrixXd::Zero(6, robot_model.nv)),
        J_rf_(Eigen::MatrixXd::Zero(6, robot_model.nv)),
        J_lf_(Eigen::MatrixXd::Zero(6, robot_model.nv)),
        J_ext_(Eigen::MatrixXd::Zero(24, robot_model.nv)),
        integral_term_(Eigen::VectorXd::Zero(robot_model.nv)),
        r_(Eigen::VectorXd::Zero(robot_model.nv)),
        Ki_(Ki),
        W_ext_(Eigen::VectorXd::Zero(24)),
        W_ext_filt_(Eigen::VectorXd::Zero(24)),
        filter_alpha_x_(0.00628), // 0.00628 designed basing on T_step
        filter_alpha_y_(0.006),
        filter_alpha_z_(0.006),
        alpha_matrix_(Eigen::MatrixXd::Zero(24, 24)),
        right_wrist_frame_(right_wrist_frame),
        left_wrist_frame_(left_wrist_frame),
        right_foot_frame_(right_foot_frame),
        left_foot_frame_(left_foot_frame)
        
        
    {
        // Initialize armatures for actuated joints
        for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) robot_model_.njoints; ++joint_id) {
            std::string joint_name = robot_model_.names[joint_id];
            if(armatures.find(joint_name) != armatures.end()) {
                M_armature_(joint_id - 2) = armatures.at(joint_name);
            }
        }

        // Initialize parameters for QP
        n_wrench_qp_variables_ = 24;            // 6 for each contact point (left foot, right foot, left wrist, right wrist)
        n_wrench_qp_equalities_ = 0;
        n_wrench_qp_inequalities_ = 2;         // non-negativity of vertical forces for each foot
        // Large slack weight makes the Fz >= 0 inequality effectively rigid, unlike the
        // default (1e-6) soft-constraint weight
        wrench_solver_ptr_ = std::make_unique<labrob::QpSolver>(
            n_wrench_qp_variables_, n_wrench_qp_equalities_, n_wrench_qp_inequalities_,
            SPEED_ABS, 50, 1e6);
        A_qp_.resize(0, n_wrench_qp_variables_);
        b_qp_.resize(0);
        C_qp_ = Eigen::MatrixXd::Zero(n_wrench_qp_inequalities_, n_wrench_qp_variables_);
        C_qp_(0, 14)  = -1.0;                 // -Fz right foot
        C_qp_(1, 20)  = -1.0;                  // -Fz left foot
        ug_qp_ = Eigen::VectorXd::Zero(n_wrench_qp_inequalities_);
        lg_qp_ = Eigen::VectorXd::Constant(n_wrench_qp_inequalities_, -1e10);


        // Initialize alpha matrix for low-pass filter
        alpha_matrix_.diagonal() << filter_alpha_x_, filter_alpha_y_, filter_alpha_z_, filter_alpha_x_, filter_alpha_y_, filter_alpha_z_, 
                                    filter_alpha_x_, filter_alpha_y_, filter_alpha_z_, filter_alpha_x_, filter_alpha_y_, filter_alpha_z_,
                                    0.001, 0.001, 0.2, 0.001, 0.001, 0.001,
                                    0.001, 0.001, 0.2, 0.001, 0.001, 0.001;


        
        // Build residual-space weight matrix W_nv_, weighted toward the left arm DOFs.
        // Base weight 1.0 everywhere; high weight on the 7 left/right arm joints so the WLS
        // privileges reconstruction of the residual on the arm chains that carry the
        // wrist-force information.
        W_nv_ = Eigen::MatrixXd::Identity(robot_model.nv, robot_model.nv);
        
        
        const std::vector<std::string> arm_joints = {
            // left arm
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            // right arm
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"
        };

        const double arm_weight = 1000.0;       // 1000 is needed for accurate estimation on wrist forces
        for (const auto& jname : arm_joints) {
            if (robot_model_.existJointName(jname)) {
                pinocchio::JointIndex jid = robot_model_.getJointId(jname);
                int vidx = robot_model_.idx_vs[jid];
                W_nv_(vidx, vidx) = arm_weight;
            }
        }

        W_ext_arm_      = Eigen::VectorXd::Zero(24);
        W_ext_arm_filt_ = Eigen::VectorXd::Zero(24);

        
    }
    

    // ########## PUBLIC METHODS ############ //
    void WristForceEstimator::update(
        const RobotState& robot_state,
        pinocchio::Data& robot_data,
        const Eigen::VectorXd& measured_torques,
        double dt)
    {
        updateDynamicTerms(robot_state, robot_data, measured_torques);
        updateForceJacobians(robot_data);
        computeFullResidual(robot_state, dt);
        //estimateWrenches();                         // unweighted, for feet/ZMP
        estimateWrenchesQP();                         // QP, for feet/ZMP
        //estimateWrenchesWeighted();                   // arm-weighted, for HAC wrists
        lowPassFilter();
    }

    
    
    
    
    // ########## PRIVATE METHODS ########## //

    // Update dynamic quantities
    void WristForceEstimator::updateDynamicTerms(
        const RobotState& robot_state,
        pinocchio::Data& robot_data,
        const Eigen::VectorXd& measured_torques)

    {
        // Current state 
        auto q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
        auto v = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);
        
        // Update
        pinocchio::computeAllTerms(robot_model_, robot_data, q, v);

        // Frame Jaobians
        pinocchio::computeJointJacobians(robot_model_, robot_data, q);
        pinocchio::updateFramePlacements(robot_model_, robot_data);
        
        // Inertia matrix
        robot_data.M.triangularView<Eigen::StrictlyLower>() = robot_data.M.transpose().triangularView<Eigen::StrictlyLower>();  // symmetrize M
        M_ = robot_data.M;
        M_.diagonal().tail(robot_model_.nv - 6) += M_armature_;

        // Generalized momentum
        p_ = M_ * v;

        // Coriolis and centrigual matrix
        C_ = pinocchio::computeCoriolisMatrix(robot_model_, robot_data, q, v);

        // Gravity vector
        pinocchio::computeGeneralizedGravity(robot_model_, robot_data, q);
        g_ = robot_data.g;

        // Measured torques from WBC
        tau_m_.tail(robot_model_.nv - 6) = measured_torques;

    }



    // Update wrists and feet Jacobians
    void WristForceEstimator::updateForceJacobians(pinocchio::Data& robot_data)
    {

        // Right wrist Jacobian
        pinocchio::getFrameJacobian(robot_model_, robot_data, 
            robot_model_.getFrameId(right_wrist_frame_), 
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rw_);

        // Left wrist Jacobian
        pinocchio::getFrameJacobian(robot_model_, robot_data, 
            robot_model_.getFrameId(left_wrist_frame_), 
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lw_);

        // Right foot Jacobian
        pinocchio::getFrameJacobian(robot_model_, robot_data, 
            robot_model_.getFrameId(right_foot_frame_), 
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rf_);

        // Left foot Jacobian
        pinocchio::getFrameJacobian(robot_model_, robot_data, 
            robot_model_.getFrameId(left_foot_frame_), 
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lf_);

        // Stacked Jacobian
        J_ext_ << J_rw_, J_lw_, J_rf_, J_lf_;

    }



    // Compute full residual recursively
    void WristForceEstimator::computeFullResidual(
        const RobotState& robot_state, 
        double dt)

    {
        // Current state
        auto v = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);
        
        // Transpose of Coriolis and centrifugal matrix
        Eigen::MatrixXd C_T = C_.transpose();


        // Capture p0_ only once the robot is actually still, so the reference
        // momentum is not frozen during a transient. Until then, hold the
        // integral and keep the residual at zero.
        if (!initialized_) {
            double v_norm = v.tail(robot_model_.nv - 6).norm();   // actuated joints only
            static int dbg = 0;
            if (dbg++ % 100 == 0)
                std::cout << "[RW-BO] waiting for still: ||v|| = " << v_norm
                        << " (threshold " << v_still_threshold_ << ")" << std::endl;
            if (v_norm < v_still_threshold_) {
                p0_ = p_;
                integral_term_.setZero();
                initialized_ = true;
                std::cout << "[RW-BO]: Generalized momentum initialized p(t0) = " << p0_ << std::endl;

            } else {
                r_.setZero();   // not yet initialized: emit no force estimate
                return;         // skip integration this step
            }
        }
        
        

        // Integrand
        Eigen::VectorXd integrand = tau_m_ + (C_T * v) - g_ + r_;

        // Integral term
        integral_term_ += dt * integrand;

        // Update residual
        r_ = Ki_ * (p_ - integral_term_ - p0_);
    }


    // Estimate wrenches
    void WristForceEstimator::estimateWrenches()
    {
        Eigen::MatrixXd J_ext_T_pinv = J_ext_.transpose().completeOrthogonalDecomposition().pseudoInverse();
        W_ext_ = J_ext_T_pinv * r_;

    }

    void WristForceEstimator::estimateWrenchesQP() {
        
        // Solve the QP problem: min_W 0.5 * W^T H W + f^T W
        // subject to A W = b (equality constraints)
        // and C W <= d (inequality constraints)

        // H = J_ext^T J_ext
        //H_qp_ = J_ext_ * J_ext_.transpose();
        H_qp_ = J_ext_ * W_nv_ * J_ext_.transpose();

        // f = -J_ext^T r
        f_qp_ = - J_ext_ * W_nv_ * r_;


        // Solve the QP problem using the QpSolver
        wrench_solver_ptr_->solve(H_qp_, f_qp_, A_qp_, b_qp_, C_qp_, lg_qp_, ug_qp_);

        // Get the solution
        W_ext_ = wrench_solver_ptr_->get_solution();

    }


    // Estimate wrenches with WLS weighted toward the left arm (for HAC wrist use).
    // Solves  min_W (J_ext^T W - r)^T W_nv (J_ext^T W - r)
    // => W = (J_ext W_nv J_ext^T)^{-1} J_ext W_nv r
    void WristForceEstimator::estimateWrenchesWeighted()
    {
        Eigen::MatrixXd JW   = J_ext_ * W_nv_;          // 24 x nv
        Eigen::MatrixXd A    = JW * J_ext_.transpose(); // 24 x 24, symmetric PSD
        Eigen::VectorXd b    = JW * r_;                 // 24 x 1
        // Robust solve (handles rank deficiency better than explicit inverse)
        W_ext_arm_ = A.completeOrthogonalDecomposition().solve(b);
    }
    
    // Low-pass filter estimated wrenches
    void WristForceEstimator::lowPassFilter()
    {
        W_ext_filt_ = alpha_matrix_ * W_ext_ + (Eigen::MatrixXd::Identity(24, 24) - alpha_matrix_) * W_ext_filt_;

        W_ext_arm_filt_ = alpha_matrix_ * W_ext_arm_ + (Eigen::MatrixXd::Identity(24, 24) - alpha_matrix_) * W_ext_arm_filt_;
    }


}