#ifndef VISC1_MOMENTUM_OBSERVER
#define VISC1_MOMENTUM_OBSERVER

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include "pinocchio/algorithm/crba.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>

class MomentumObserver{
    public:
        MomentumObserver(
            const Eigen::VectorXd& diagKo,
            const pinocchio::Model& robot_model, 
            const pinocchio::Data& robot_data,
            double dt,
            int dim_buffer_r,
            double epsilon);

        Eigen::VectorXd update(const Eigen::VectorXd& q,const Eigen::VectorXd& qdot, const Eigen::VectorXd& tau);
        Eigen::VectorXd reconstructForceWrench(const Eigen::MatrixXd& J);
        Eigen::VectorXd estimateContactPointInLinkReferenceFrame(const Eigen::VectorXd& F);
        int collisionDetect();

    protected:

    private:
        pinocchio::Model robot_model;
        pinocchio::Data robot_data;
        Eigen::MatrixXd Ko;
        Eigen::VectorXd r_integral_term;
        Eigen::VectorXd last_r;
        Eigen::VectorXd P0;
        bool is_zero_initialized;
        double dt;
        double epsilon;
        Eigen::MatrixXd buffer;
        Eigen::VectorXd max_r;
        void updateBuffer(const Eigen::VectorXd& new_r);
};


#endif