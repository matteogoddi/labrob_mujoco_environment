#ifndef VISC1_HUMANOID_CONTACT_FORCES_ESTIMATOR
#define VISC1_HUMANOID_CONTACT_FORCES_ESTIMATOR

#include "MomentumObserver.hpp"

class HumanoidContactForcesEstimator{
    public:
        HumanoidContactForcesEstimator(
            const MomentumObserver& momentum_observer,
            const pinocchio::Model& robot_model, 
            const pinocchio::Data& robot_data,
            const int left_foot_frame_id,
            const int right_foot_frame_id,
            const int left_hand_frame_id,
            const int right_hand_frame_id);

        Eigen::VectorXd update(const Eigen::VectorXd& q,const Eigen::VectorXd& qdot, const Eigen::VectorXd& tau, const double time);
        Eigen::VectorXd getLeftFootWrench();
        Eigen::VectorXd getRightFootWrench();
        Eigen::VectorXd getLeftHandWrench();
        Eigen::VectorXd getRightHandWrench();
        double getJacobianConditionNumber();
        inline bool isLeftArmInCollision() { return left_arm_collision_state; }
        inline bool isRightArmInCollision() { return right_arm_collision_state; }
        
        inline Eigen::VectorXd getLeftArmResidual() { return last_r_larm; }
        inline Eigen::VectorXd getRightArmResidual() { return last_r_rarm; }

    protected:

    private:
        float epsilon = 1.0e-4;
        MomentumObserver momentum_observer;
        pinocchio::Model robot_model;
        pinocchio::Data robot_data;
        int left_foot_frame_id;
        int right_foot_frame_id;
        int left_hand_frame_id;
        int right_hand_frame_id;
        Eigen::VectorXd left_foot_wrench;
        Eigen::VectorXd right_foot_wrench;
        Eigen::VectorXd left_hand_wrench;
        Eigen::VectorXd right_hand_wrench;
        double jacobian_condition_number;
        bool left_arm_collision_state;
        bool right_arm_collision_state;
        Eigen::VectorXd last_r_larm;
        Eigen::VectorXd last_r_rarm;
        Eigen::MatrixXd left_arm_buffer;
        Eigen::MatrixXd right_arm_buffer;
        Eigen::VectorXd left_arm_max_r;
        Eigen::VectorXd right_arm_max_r;
        Eigen::VectorXd left_arm_min_r;
        Eigen::VectorXd right_arm_min_r;
        int left_arm_collision_link;
        int right_arm_collision_link;
        std::vector<int> left_arm_v_indices;
        std::vector<int> right_arm_v_indices;

        void updateBuffers(const Eigen::VectorXd& new_left_arm_r, const Eigen::VectorXd& new_right_arm_r);
};

#endif