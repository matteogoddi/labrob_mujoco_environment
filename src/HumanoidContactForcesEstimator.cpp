#include <hrp4_locomotion/HumanoidContactForcesEstimator.hpp>

HumanoidContactForcesEstimator::HumanoidContactForcesEstimator(
            const MomentumObserver& momentum_observer,
            const pinocchio::Model& robot_model, 
            const pinocchio::Data& robot_data,
            const int left_foot_frame_id,
            const int right_foot_frame_id)
            :
            momentum_observer(momentum_observer),
            robot_model(robot_model),
            robot_data(robot_data),
            left_foot_frame_id(left_foot_frame_id),
            right_foot_frame_id(right_foot_frame_id)
            {
                left_foot_wrench = Eigen::VectorXd::Zero(6);
                right_foot_wrench = Eigen::VectorXd::Zero(6);
            }

Eigen::VectorXd HumanoidContactForcesEstimator::update(const Eigen::VectorXd& q,const Eigen::VectorXd& qdot, const Eigen::VectorXd& tau){

    Eigen::VectorXd r = momentum_observer.update(q, qdot, tau);

    Eigen::MatrixXd Jlsole = Eigen::MatrixXd::Zero(6, robot_model.nv);
    Eigen::MatrixXd Jrsole = Eigen::MatrixXd::Zero(6, robot_model.nv);
    pinocchio::framesForwardKinematics(robot_model, robot_data, q);
    pinocchio::updateFramePlacements(robot_model, robot_data);
    pinocchio::getFrameJacobian(robot_model, robot_data, left_foot_frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jlsole);
    pinocchio::getFrameJacobian(robot_model, robot_data, right_foot_frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jrsole);
    Eigen::MatrixXd J_stack(Jlsole.rows() + Jrsole.rows(), Jlsole.cols());
    J_stack.topRows(Jlsole.rows()) = Jlsole;
    J_stack.bottomRows(Jrsole.rows()) = Jrsole;

    Eigen::VectorXd reconstructed_wrench = momentum_observer.reconstructForceWrench(J_stack);
    left_foot_wrench = reconstructed_wrench.head(6);
    right_foot_wrench = reconstructed_wrench.tail(6);

    return r;
}

Eigen::VectorXd HumanoidContactForcesEstimator::getLeftFootWrench(){
    return left_foot_wrench;
}

Eigen::VectorXd HumanoidContactForcesEstimator::getRightFootWrench(){
    return right_foot_wrench;
}
