#include <hrp4_locomotion/HumanoidContactForcesEstimator.hpp>

HumanoidContactForcesEstimator::HumanoidContactForcesEstimator(
            const MomentumObserver& momentum_observer,
            const pinocchio::Model& robot_model, 
            const pinocchio::Data& robot_data,
            const int left_foot_frame_id,
            const int right_foot_frame_id,
            const int left_hand_frame_id,
            const int right_hand_frame_id)
            :
            momentum_observer(momentum_observer),
            robot_model(robot_model),
            robot_data(robot_data),
            left_foot_frame_id(left_foot_frame_id),
            right_foot_frame_id(right_foot_frame_id),
            left_hand_frame_id(left_hand_frame_id),
            right_hand_frame_id(right_hand_frame_id)
            {
                left_foot_wrench = Eigen::VectorXd::Zero(6);
                right_foot_wrench = Eigen::VectorXd::Zero(6);
                left_hand_wrench = Eigen::VectorXd::Zero(6);
                right_hand_wrench = Eigen::VectorXd::Zero(6);


                // 1. Definisci i nomi dei giunti delle braccia (nomi esatti dall'URDF del G1)
                std::vector<std::string> larm_names = {
                    "left_shoulder_pitch_joint", 
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint", 
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                    "left_hand_palm_joint"
                };

                std::vector<std::string> rarm_names = {
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                    "right_hand_palm_joint"
                };

                // 2. Trova i loro indici di velocità (idx_v) nel modello Pinocchio
                for(const auto& name : larm_names) {
                    if(robot_model.existJointName(name)) {
                        auto joint_id = robot_model.getJointId(name);
                        // idx_v() restituisce l'indice di partenza nel vettore nv (coppie/velocità)
                        left_arm_v_indices.push_back(robot_model.joints[joint_id].idx_v());
                    }
                }

                for(const auto& name : rarm_names) {
                    if(robot_model.existJointName(name)) {
                        auto joint_id = robot_model.getJointId(name);
                        right_arm_v_indices.push_back(robot_model.joints[joint_id].idx_v());
                    }
                }
                
                int buffer_size = 1/0.001;
                left_arm_buffer = Eigen::MatrixXd::Zero(left_arm_v_indices.size(), buffer_size);
                right_arm_buffer = Eigen::MatrixXd::Zero(right_arm_v_indices.size(), buffer_size);
                last_r_larm = Eigen::VectorXd::Zero(left_arm_v_indices.size());
                last_r_rarm = Eigen::VectorXd::Zero(right_arm_v_indices.size());
                left_arm_max_r = Eigen::VectorXd::Zero(left_arm_v_indices.size());
                right_arm_max_r = Eigen::VectorXd::Zero(right_arm_v_indices.size());
                left_arm_min_r = Eigen::VectorXd::Zero(left_arm_v_indices.size());
                right_arm_min_r = Eigen::VectorXd::Zero(right_arm_v_indices.size());
                left_arm_collision_state = false;
                right_arm_collision_state = false;
                left_arm_collision_link = 0;
                right_arm_collision_link = 0;

                // Print di debug per sicurezza
                //std::cout << "Trovati " << left_arm_v_indices.size() << " giunti per il braccio SX." << std::endl;
                //std::cout << "Trovati " << right_arm_v_indices.size() << " giunti per il braccio DX." << std::endl;
            }

Eigen::VectorXd HumanoidContactForcesEstimator::update(const Eigen::VectorXd& q,const Eigen::VectorXd& qdot, const Eigen::VectorXd& tau, const double time){

    Eigen::VectorXd r = momentum_observer.update(q, qdot, tau);

    Eigen::MatrixXd Jlsole = Eigen::MatrixXd::Zero(6, robot_model.nv);
    Eigen::MatrixXd Jrsole = Eigen::MatrixXd::Zero(6, robot_model.nv);
    Eigen::MatrixXd Jlhand = Eigen::MatrixXd::Zero(6, robot_model.nv);
    Eigen::MatrixXd Jrhand = Eigen::MatrixXd::Zero(6, robot_model.nv);
    pinocchio::framesForwardKinematics(robot_model, robot_data, q);
    pinocchio::updateFramePlacements(robot_model, robot_data);
    pinocchio::getFrameJacobian(robot_model, robot_data, left_foot_frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jlsole);
    pinocchio::getFrameJacobian(robot_model, robot_data, right_foot_frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jrsole);
    pinocchio::getFrameJacobian(robot_model, robot_data, left_hand_frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jlhand);
    pinocchio::getFrameJacobian(robot_model, robot_data, right_hand_frame_id, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, Jrhand);
    
    // SPLITTED
    int n_larm = left_arm_v_indices.size();
    int n_rarm = right_arm_v_indices.size();
    Eigen::VectorXd r_larm = Eigen::VectorXd::Zero(n_larm);
    Eigen::VectorXd r_rarm = Eigen::VectorXd::Zero(n_rarm);
    Eigen::MatrixXd Jlhand_local = Eigen::MatrixXd::Zero(6, n_larm);
    Eigen::MatrixXd Jrhand_local = Eigen::MatrixXd::Zero(6, n_rarm);

    for(int i = 0; i < n_larm; ++i) {
        int idx = left_arm_v_indices[i];
        Jlhand_local.col(i) = Jlhand.col(idx);
        r_larm(i) = r(idx);
    }
    
    for(int i = 0; i < n_rarm; ++i) {
        int idx = right_arm_v_indices[i];
        Jrhand_local.col(i) = Jrhand.col(idx);
        r_rarm(i) = r(idx);
    }
    //#1
    updateBuffers(last_r_larm, last_r_rarm);
    last_r_larm = r_larm;
    last_r_rarm = r_rarm;
    //std::cout << "r left arm: " << r_larm.transpose() << std::endl;
    //std::cout << "r right arm: " << r_rarm.transpose() << std::endl;

    if(time >0.5){
        if(left_arm_collision_state == false){
        for(int i = n_larm; i>0; i--){
            if(abs(last_r_larm[i-1]) > left_arm_max_r(i-1) + epsilon){
                left_arm_collision_link = i;
                left_arm_collision_state = true;
                break;
            }
        }

    }

    if(right_arm_collision_state == false){
        for(int i = n_rarm; i>0; i--){
            if(abs(last_r_rarm[i-1]) > right_arm_max_r(i-1) + epsilon){
                right_arm_collision_link = i;
                right_arm_collision_state = true;
                break;
            }
        }

    }

    if(left_arm_collision_state == true){ //Check if collision ended
        if(abs(last_r_larm[left_arm_collision_link-1]) < left_arm_min_r(left_arm_collision_link-1) - epsilon){
            left_arm_collision_state = false;
            left_arm_collision_link = 0;
        }
    }

    if(right_arm_collision_state == true){ //Check if collision ended
        if(abs(last_r_rarm[right_arm_collision_link-1]) < right_arm_min_r(right_arm_collision_link-1) - epsilon){
            right_arm_collision_state = false;
            right_arm_collision_link = 0;
        }
    }    



    if(left_arm_collision_state){
        left_hand_wrench = Jlhand_local.transpose().completeOrthogonalDecomposition().solve(last_r_larm);
    }else left_hand_wrench = Eigen::VectorXd::Zero(6);

    if(right_arm_collision_state){
        right_hand_wrench = Jrhand_local.transpose().completeOrthogonalDecomposition().solve(last_r_rarm);
    }else right_hand_wrench = Eigen::VectorXd::Zero(6);

    }
    
    
    
    //#2
    //Eigen::VectorXd r_arms = Eigen::VectorXd::Zero(n_larm + n_rarm);
    //r_arms.head(n_larm) = r_larm;
    //r_arms.tail(n_rarm) = r_rarm;
    //Eigen::MatrixXd J_hands_stack(Jlhand_local.rows() + Jrhand_local.rows(), n_larm + n_rarm);
    //J_hands_stack.topRows(Jlhand_local.rows()) = Jlhand_local;
    //J_hands_stack.bottomRows(Jrhand_local.rows()) = Jrhand_local;
    //Eigen::VectorXd hands_wrenches = J_hands_stack.transpose().completeOrthogonalDecomposition().solve(r_arms);
    //left_hand_wrench = hands_wrenches.head(6);
    //right_hand_wrench = hands_wrenches.tail(6);

    Eigen::VectorXd r_hands_effect = Jlhand.transpose() * left_hand_wrench + Jrhand.transpose() * right_hand_wrench;
    Eigen::VectorXd r_remaining = r - r_hands_effect;

    Eigen::MatrixXd J_feet_stack(Jlsole.rows() + Jrsole.rows(), robot_model.nv);
    J_feet_stack.topRows(Jlsole.rows()) = Jlsole;
    J_feet_stack.bottomRows(Jrsole.rows()) = Jrsole;

    Eigen::VectorXd feet_wrenches = J_feet_stack.transpose().completeOrthogonalDecomposition().solve(r_remaining);
    left_foot_wrench = feet_wrenches.head(6);
    right_foot_wrench = feet_wrenches.tail(6);

    ////FULL
    //Eigen::MatrixXd J_stack(Jlsole.rows() + Jrsole.rows() + Jlhand.rows() + Jrhand.rows(), Jlsole.cols());
    //J_stack.topRows(Jlsole.rows()) = Jlsole;
    //J_stack.middleRows(Jlsole.rows(), Jrsole.rows()) = Jrsole;
    //J_stack.middleRows(Jlsole.rows() + Jrsole.rows(), Jlhand.rows()) = Jlhand;
    //J_stack.bottomRows(Jrhand.rows()) = Jrhand;

    //ONLY FEET
    //Eigen::MatrixXd J_stack(Jlsole.rows() + Jrsole.rows(), Jlsole.cols());
    //J_stack.topRows(Jlsole.rows()) = Jlsole;
    //J_stack.bottomRows(Jrsole.rows()) = Jrsole;

    //Eigen::JacobiSVD<Eigen::MatrixXd> svd_lsole(Jlsole);
    //Eigen::JacobiSVD<Eigen::MatrixXd> svd_rsole(Jrsole);
    //Eigen::JacobiSVD<Eigen::MatrixXd> svd_lhand(Jlhand);
    //Eigen::JacobiSVD<Eigen::MatrixXd> svd_rhand(Jrhand);
    //double lsole_condition_number = svd_lsole.singularValues()(0) / svd_lsole.singularValues()(svd_lsole.singularValues().size() - 1);
    //double rsole_condition_number = svd_rsole.singularValues()(0) / svd_rsole.singularValues()(svd_rsole.singularValues().size() - 1);
    //double lhand_condition_number = svd_lhand.singularValues()(0) / svd_lhand.singularValues()(svd_lhand.singularValues().size() - 1);
    //double rhand_condition_number = svd_rhand.singularValues()(0) / svd_rhand.singularValues()(svd_rhand.singularValues().size() - 1);
    //std::cout << "Condition Number of LEFT FOOT Jacobian: " << lsole_condition_number << std::endl;
    //std::cout << "Condition Number of RIGHT FOOT Jacobian: " << rsole_condition_number << std::endl;
    //std::cout << "Condition Number of LEFT HAND Jacobian: " << lhand_condition_number << std::endl;
    //std::cout << "Condition Number of RIGHT HAND Jacobian: " << rhand_condition_number << std::endl;

    //Eigen::JacobiSVD<Eigen::MatrixXd> svd(J_stack);
    //jacobian_condition_number = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

    //Eigen::VectorXd reconstructed_wrench = momentum_observer.reconstructForceWrench(J_stack);
    //FULL
    //left_foot_wrench = reconstructed_wrench.head(6);
    //right_foot_wrench = reconstructed_wrench.segment(6,6);
    //left_hand_wrench = reconstructed_wrench.segment(12,6);
    //right_hand_wrench = reconstructed_wrench.tail(6);

    //ONLY FEET
    //left_foot_wrench = reconstructed_wrench.head(6);
    //right_foot_wrench = reconstructed_wrench.tail(6);

    return r;
}

Eigen::VectorXd HumanoidContactForcesEstimator::getLeftFootWrench(){
    return left_foot_wrench;
}

Eigen::VectorXd HumanoidContactForcesEstimator::getRightFootWrench(){
    return right_foot_wrench;
}

Eigen::VectorXd HumanoidContactForcesEstimator::getLeftHandWrench(){
    return left_hand_wrench;
}

Eigen::VectorXd HumanoidContactForcesEstimator::getRightHandWrench(){
    return right_hand_wrench;
}

double HumanoidContactForcesEstimator::getJacobianConditionNumber(){
    return jacobian_condition_number;
}


void HumanoidContactForcesEstimator::updateBuffers(const Eigen::VectorXd& new_left_arm_r, const Eigen::VectorXd& new_right_arm_r){

    for(int i = 0; i<left_arm_v_indices.size(); i++){
        left_arm_max_r(i) = abs(left_arm_buffer(i, left_arm_buffer.cols()-2)); //suppose that the oldest in the buffer is the max and the min (the -1 must be trashed)
        left_arm_min_r(i) = abs(left_arm_buffer(i, left_arm_buffer.cols()-2));
        for(int j = left_arm_buffer.cols()-1; j>=0 ; j--){ //update the buffer
            if(j>0){
                left_arm_buffer(i,j) = left_arm_buffer(i,j-1);
            }else{
                left_arm_buffer(i,j) = new_left_arm_r(i);
            }
            left_arm_max_r(i) = left_arm_max_r(i) >= abs(left_arm_buffer(i,j)) ? left_arm_max_r(i) : abs(left_arm_buffer(i,j)); //recheck for maximum
            left_arm_min_r(i) = left_arm_min_r(i) <= abs(left_arm_buffer(i,j)) ? left_arm_min_r(i) : abs(left_arm_buffer(i,j)); //recheck for minimum

        }

    }


    for(int i = 0; i<right_arm_v_indices.size(); i++){
        right_arm_max_r(i) = abs(right_arm_buffer(i, right_arm_buffer.cols()-2)); //suppose that the oldest in the buffer is the max and the min (the -1 must be trashed)
        right_arm_min_r(i) = abs(right_arm_buffer(i, right_arm_buffer.cols()-2));
        for(int j = right_arm_buffer.cols()-1; j>=0 ; j--){ //update the buffer
            if(j>0){
                right_arm_buffer(i,j) = right_arm_buffer(i,j-1);
            }else{
                right_arm_buffer(i,j) = new_right_arm_r(i);
            }
            right_arm_max_r(i) = right_arm_max_r(i) >= abs(right_arm_buffer(i,j)) ? right_arm_max_r(i) : abs(right_arm_buffer(i,j)); //recheck for maximum
            right_arm_min_r(i) = right_arm_min_r(i) <= abs(right_arm_buffer(i,j)) ? right_arm_min_r(i) : abs(right_arm_buffer(i,j)); //recheck for minimum

        }

    }

}