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
            const int right_foot_frame_id);

        Eigen::VectorXd update(const Eigen::VectorXd& q,const Eigen::VectorXd& qdot, const Eigen::VectorXd& tau);
        Eigen::VectorXd getLeftFootWrench();
        Eigen::VectorXd getRightFootWrench();
        
        

    protected:

    private:
        MomentumObserver momentum_observer;
        pinocchio::Model robot_model;
        pinocchio::Data robot_data;
        int left_foot_frame_id;
        int right_foot_frame_id;
        Eigen::VectorXd left_foot_wrench;
        Eigen::VectorXd right_foot_wrench;
};

#endif