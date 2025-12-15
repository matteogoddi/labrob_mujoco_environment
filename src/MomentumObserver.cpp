#include <hrp4_locomotion/MomentumObserver.hpp>

MomentumObserver::MomentumObserver(
            const Eigen::VectorXd& diagKo,
            const pinocchio::Model& robot_model, 
            const pinocchio::Data& robot_data,
            double dt,
            int dim_buffer_r,
            double epsilon)
            :
            robot_model(robot_model),
            robot_data(robot_data),
            dt(dt),
            epsilon(epsilon)
            {
                if(robot_model.nv != diagKo.size()){
                    std::cerr << "ERRORE: dimensione passate all'osservatore errate" << std::endl;
                    return;
                }
                Ko = diagKo.asDiagonal();
                r_integral_term = Eigen::VectorXd::Zero(robot_model.nv);
                last_r = Eigen::VectorXd::Zero(robot_model.nv);
                P0 = Eigen::VectorXd::Zero(robot_model.nv);
                is_zero_initialized = false;
                buffer = Eigen::MatrixXd::Zero(robot_model.nv, dim_buffer_r);
                max_r = Eigen::VectorXd::Zero(robot_model.nv);
            }

Eigen::VectorXd MomentumObserver::update(const Eigen::VectorXd& q, const Eigen::VectorXd& qdot, const Eigen::VectorXd& tau){
    Eigen::MatrixXd M = pinocchio::crba(robot_model, robot_data, q);
    // We need to do this since the inertia matrix in Pinocchio is only upper triangular
    M.triangularView<Eigen::StrictlyLower>() = M.transpose().triangularView<Eigen::StrictlyLower>();
    const auto& c = pinocchio::rnea(robot_model, robot_data, q, qdot, Eigen::VectorXd::Zero(robot_model.nv));
    const auto& coriolis = pinocchio::computeCoriolisMatrix(robot_model, robot_data, q, qdot);
    const auto& coriolisT = coriolis.transpose();
    const auto& gravity = c - coriolis * qdot;
    const auto& beta = gravity - coriolisT * qdot;
    const auto& Pt = M*qdot;
    if(!is_zero_initialized){
      P0 = Pt;
      is_zero_initialized = true;
    }

    

    // Residual method
    r_integral_term += (tau - beta + last_r)* dt;
    Eigen::VectorXd r = Ko * (Pt  - r_integral_term - P0);
    updateBuffer(last_r);
    last_r = r;

    return r;
}

//Call before passing the jacobian
//pinocchio::framesForwardKinematics(robot_model, robot_data, q);
//pinocchio::updateFramePlacements(robot_model, robot_data);
//pinocchio::computeJointJacobians(robot_model, robot_data, q);
//pinocchio::getFrameJacobian(robot_model, robot_data, id_, pinocchio::ReferenceFrame::YOURCHOICE, J);
Eigen::VectorXd MomentumObserver::reconstructForceWrench(const Eigen::MatrixXd& J){
      return J.transpose().completeOrthogonalDecomposition().solve(last_r);
}

//To use this you make the assumption that there are no momentum applied by the force wrench
Eigen::VectorXd MomentumObserver::estimateContactPointInLinkReferenceFrame(const Eigen::VectorXd& F){
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(3,3);
    S << 0, -F[2], F[1], F[2], 0, -F[0], -F[1], F[0], 0;
    return S.transpose().completeOrthogonalDecomposition().solve(F.tail(3));
}

int MomentumObserver::collisionDetect(){
    for(int i = robot_model.nv; i>0; i--)
        if(abs(last_r[i-1]) > max_r(i-1) + epsilon)
            return i;
    return 0;
}

void MomentumObserver::updateBuffer(const Eigen::VectorXd& new_r){

    for(int i = 0; i<robot_model.nv; i++){
        max_r(i) = abs(buffer(i, buffer.cols()-2)); //suppose that the oldest in the buffer is the max (the -1 must be trashed)
        for(int j = buffer.cols()-1; j>=0 ; j--){ //update the buffer
            if(j>0){
                buffer(i,j) = buffer(i,j-1);
            }else{
                buffer(i,j) = new_r(i);
            }
            max_r(i) = max_r(i) >= abs(buffer(i,j)) ? max_r(i) : abs(buffer(i,j)); //recheck for maximum

        }

    }

}