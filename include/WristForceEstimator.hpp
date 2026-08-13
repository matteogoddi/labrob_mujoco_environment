#ifndef LABROB_WRIST_FORCE_ESTIMATOR_HPP_
#define LABROB_WRIST_FORCE_ESTIMATOR_HPP_

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

    class WristForceEstimator {

        public:

            // Constructor for full robot residual estimation
            WristForceEstimator(
                const pinocchio::Model& robot_model,
                const std::map<std::string, double>& armatures,
                double Ki,
                const std::string& right_wrist_frame,
                const std::string& left_wrist_frame,
                const std::string& right_foot_frame,
                const std::string& left_foot_frame
            );

            // Online update
            void update(
                const RobotState& robot_state,
                pinocchio::Data& robot_data,
                const Eigen::VectorXd& measured_torques,
                double dt
            );


            // Getters with filter
            const Eigen::VectorXd& getResidual()          const { return r_; }
            const Eigen::VectorXd& getEstimatedWrenches() const { return W_ext_filt_; }

            //Eigen::Vector3d getWeightedRightWristForce() const { return W_ext_arm_filt_.segment<3>(0); }
            //Eigen::Vector3d getWeightedLeftWristForce()  const { return W_ext_arm_filt_.segment<3>(6); }
            Eigen::Vector3d getWeightedRightWristForce() const { return W_ext_filt_.segment<3>(0); }
            Eigen::Vector3d getWeightedLeftWristForce()  const { return W_ext_filt_.segment<3>(6); }

            Eigen::Matrix<double,6,1> getRightFootWrench() const { return W_ext_filt_.segment<6>(12); }
            Eigen::Matrix<double,6,1> getLeftFootWrench()  const { return W_ext_filt_.segment<6>(18); }
            const Eigen::VectorXd& getGeneralizedMomentum()        const { return p_; }
            const Eigen::VectorXd& getInitialGeneralizedMomentum() const { return p0_; }
            Eigen::VectorXd getTauMinusG() const { return tau_m_ - g_; }


        private:
            
            // Model variables
            const pinocchio::Model& robot_model_;
            Eigen::VectorXd M_armature_;                            // Motor armature inertias
            Eigen::MatrixXd M_;                                     // Generalized inertia matrix
            Eigen::VectorXd p_;                                     // Generalized momentum
            Eigen::MatrixXd C_;                                     // Coriolis and centrifugal matrix
            Eigen::VectorXd g_;                                     // Gravity vector
            Eigen::VectorXd tau_m_;                                 // Measured torques from WBC (control action)
            Eigen::MatrixXd J_ext_;                                 // Stacked Jacobians of all external wrenches
            Eigen::VectorXd integral_term_;                         // Integral term for arm residual
            
            // Observer variables
            double Ki_;                                             // Observer gain
            Eigen::VectorXd p0_;                                    // Initial generalized momentum
            bool initialized_ = false;
            Eigen::VectorXd r_;                                     // Full body residual vector
            Eigen::VectorXd W_ext_;                                 // Feet + wrists stacked wrenches
            Eigen::VectorXd W_ext_filt_;                            // Feet + wrists stacked wrenches after filtering noise out
            double filter_alpha_x_;                                 // EMA parameter: filtering along x axis
            double filter_alpha_y_;                                 // EMA parameter: filtering along y axis
            double filter_alpha_z_;                                 // EMA parameter: filtering along z axis
            Eigen::MatrixXd alpha_matrix_;                          // Diagonal matrix of EMA alpha parameters
            double v_still_threshold_ = 0.2;                        // rad/s, below this the robot is considered still: 0.02 for simulation, 0.2 for real experiments


            // WLS weighted toward left arm (for HAC wrist estimates)
            Eigen::MatrixXd W_nv_;                                  // nv x nv diagonal weight matrix (residual space)
            Eigen::VectorXd W_ext_arm_;                             // wrenches from arm-weighted WLS
            Eigen::VectorXd W_ext_arm_filt_;                        // filtered version

            // Right wrist variables
            const std::string right_wrist_frame_;
            Eigen::MatrixXd J_rw_;                                   // Right wrist Jacobian
            
            // Left wrist variables
            const std::string left_wrist_frame_;
            Eigen::MatrixXd J_lw_;                                   // Left wrist Jacobian

            // Right foot variables
            const std::string right_foot_frame_;
            Eigen::MatrixXd J_rf_;                                   // Right foot Jacobian

            // Left foot variables
            const std::string left_foot_frame_;
            Eigen::MatrixXd J_lf_;                                   // Left foot Jacobian


            // Foot wrench estimation with QP solver
            std::unique_ptr<labrob::QpSolver> wrench_solver_ptr_;
            int n_wrench_qp_variables_, n_wrench_qp_equalities_, n_wrench_qp_inequalities_;
            Eigen::MatrixXd H_qp_;
            Eigen::VectorXd f_qp_;
            Eigen::MatrixXd A_qp_;
            Eigen::VectorXd b_qp_;
            Eigen::MatrixXd C_qp_;
            Eigen::VectorXd ug_qp_;
            Eigen::VectorXd lg_qp_;


            // ########## METHODS ########## //

            void updateDynamicTerms(
                const RobotState& robot_state, 
                pinocchio::Data& robot_data, 
                const Eigen::VectorXd& measured_torques
            );

            void updateForceJacobians(pinocchio::Data& robot_data);

            void computeFullResidual(const RobotState& robot_state, double dt);

            void estimateWrenches();

            void estimateWrenchesQP();

            void estimateWrenchesWeighted();

            void lowPassFilter();


    };


} // namespace labrob

#endif // LABROB_WRIST_FORCE_ESTIMATOR_HPP_