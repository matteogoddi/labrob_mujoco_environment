#include <hrp4_locomotion/WalkingManager.hpp>

// STL
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

// Eigen
#include <Eigen/Core>

// Pinocchio
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <hrp4_locomotion/GaitConfiguration.hpp>
#include <hrp4_locomotion/JointCommand.hpp>
#include <hrp4_locomotion/TimingLaw.hpp>
#include <hrp4_locomotion/utils.hpp>
#include <hrp4_locomotion/ResidualEstimator.hpp>

#include <hrp4_locomotion/globals.h>

namespace labrob {

WalkingManager::WalkingManager() :
    kf_LipState(Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero())
{

}

bool
WalkingManager::init(const labrob::RobotState& initial_robot_state,
                     std::map<std::string, double> &armatures) {
    cov_x = Eigen::Matrix3d::Identity();
    cov_y = Eigen::Matrix3d::Identity();
    cov_z = Eigen::Matrix3d::Identity();

    cov_meas_pos = 1.0e1;
    cov_meas_vel = 1.0e2;
    cov_meas_zmp = 1.0e8;

    cov_mod_pos = 1.0;
    cov_mod_vel = 1.0;
    cov_mod_zmp = 1.0;

    // estimated_force = Eigen::VectorXd::Zero(6);

    // Pre-allocation of arrays (maximum size is 50000)

    int64_t max_steps = 50000;

    sim_com_position_log_.reserve(max_steps);
    sim_com_velocity_log_.reserve(max_steps);
    sim_zmp_position_log_.reserve(max_steps);
    
    fb_com_position_log_.reserve(max_steps);
    fb_com_velocity_log_.reserve(max_steps);
    fb_zmp_position_log_.reserve(max_steps);

    kf_com_position_log_.reserve(max_steps);
    kf_com_velocity_log_.reserve(max_steps);
    kf_zmp_position_log_.reserve(max_steps);

    des_com_position_log_.reserve(max_steps);
    des_com_velocity_log_.reserve(max_steps);
    des_zmp_position_log_.reserve(max_steps);

    ef_zmp_position_log_.reserve(max_steps);

    // base_estimate_log_.reserve(max_steps);
    // orientation_estimate_log_.reserve(max_steps);
    // left_foot_position_base_estimation_log_.reserve(max_steps);
    // right_foot_position_base_estimation_log_.reserve(max_steps);
    // left_foot_position_with_zero_base_log_.reserve(max_steps);
    // right_foot_position_with_zero_base_log_.reserve(max_steps);

    p_lsole_sim_log_.reserve(max_steps);
    p_rsole_sim_log_.reserve(max_steps);
    v_lsole_sim_log_.reserve(max_steps);
    v_rsole_sim_log_.reserve(max_steps);
    p_lsole_fb_log_.reserve(max_steps);
    p_rsole_fb_log_.reserve(max_steps);
    v_lsole_fb_log_.reserve(max_steps);
    v_rsole_fb_log_.reserve(max_steps);
    p_lsole_des_log_.reserve(max_steps);
    p_rsole_des_log_.reserve(max_steps);
    v_lsole_des_log_.reserve(max_steps);
    v_rsole_des_log_.reserve(max_steps);

    estimated_force_lsole_log_.reserve(max_steps);
    estimated_force_rsole_log_.reserve(max_steps);

    angular_momentum_log_.reserve(max_steps);
    mpc_predictions_log_.reserve(max_steps);

    measured_imu_orientation_log_.reserve(max_steps);
    measured_imu_angular_velocity_log_.reserve(max_steps);
    measured_imu_accelerometer_log_.reserve(max_steps);
    measured_joint_position_log_.reserve(max_steps);
    measured_joint_velocity_log_.reserve(max_steps);

    ekf_base_position_log_.reserve(max_steps);
    ekf_base_velocity_log_.reserve(max_steps);
    ekf_base_orientation_log_.reserve(max_steps);
    ekf_base_angular_velocity_log_.reserve(max_steps);
    ekf_joint_position_log_.reserve(max_steps);
    ekf_joint_velocity_log_.reserve(max_steps);

    sim_base_position_log_.reserve(max_steps);
    sim_base_velocity_log_.reserve(max_steps);
    sim_base_orientation_log_.reserve(max_steps);
    sim_base_angular_velocity_log_.reserve(max_steps);
    sim_joint_position_log_.reserve(max_steps);
    sim_joint_velocity_log_.reserve(max_steps);

    // estimated_imu_accelerometer_log_.reserve(max_steps);
    // estimated_imu_angular_velocity_log_.reserve(max_steps);
    // estimated_imu_orientation_log_.reserve(max_steps);

    execution_time_wbc_log_.reserve(max_steps);
    execution_time_mpc_log_.reserve(max_steps);
    execution_time_ekf_log_.reserve(max_steps);
    execution_time_kf_log_.reserve(max_steps);

    input_torque_log_.reserve(max_steps);

    kalman_gain_log_.reserve(max_steps);

    // Read URDF from file:
    std::string robot_description_filename = "../g1_description/unitreeg1_2.urdf";

    // Build Pinocchio model and data from URDF:
    pinocchio::Model full_robot_model;

    pinocchio::JointModelFreeFlyer root_joint;
    pinocchio::urdf::buildModel(
        robot_description_filename,
        root_joint,
        full_robot_model
    );
    const std::vector<std::string> joint_to_lock_names{
    };
    std::vector<pinocchio::JointIndex> joint_ids_to_lock;
    for (const auto& joint_name : joint_to_lock_names) {
        if (full_robot_model.existJointName(joint_name)) {
        joint_ids_to_lock.push_back(full_robot_model.getJointId(joint_name));
        }
    }

    robot_model = pinocchio::buildReducedModel(
        full_robot_model,
        joint_ids_to_lock,
        pinocchio::neutral(full_robot_model)
    );
    sim_robot_data = pinocchio::Data(robot_model);

    njnt = robot_model.nv - 6;

    // Init desired lsole and rsole poses:
    auto q_init = robot_state_to_pinocchio_joint_configuration(
        robot_model,
        initial_robot_state
    );
    auto qdot_init = robot_state_to_pinocchio_joint_velocity(
        robot_model,
        initial_robot_state
    );
    pinocchio::forwardKinematics(robot_model, sim_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, sim_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, sim_robot_data, q_init);
    pinocchio::centerOfMass(robot_model, sim_robot_data, q_init, false);

    integrated_state_pos = Eigen::VectorXd::Zero(6 + njnt);
    integrated_state_vel = Eigen::VectorXd::Zero(6 + njnt);

    integrated_state_pos.head<3>() = q_init.head<3>();
    integrated_state_pos.segment<3>(3) = rotVecFromQuaternion(Eigen::Quaterniond(
        q_init[6], q_init[3], q_init[4], q_init[5]
    ));
    integrated_state_pos.tail(njnt) = q_init.tail(njnt);
    integrated_state_vel = qdot_init;

    fb_robot_data = pinocchio::Data(robot_model);
    predicted_robot_data = pinocchio::Data(robot_model);
    estimated_robot_data = pinocchio::Data(robot_model);

    fb_robot_state = initial_robot_state;

    pinocchio::forwardKinematics(robot_model, fb_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, fb_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, fb_robot_data, q_init);

    pinocchio::forwardKinematics(robot_model, predicted_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, predicted_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, predicted_robot_data, q_init);

    pinocchio::forwardKinematics(robot_model, estimated_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, estimated_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, estimated_robot_data, q_init);


    n_ekf_output = njnt + 3 + njnt + 3 + 6 + 6;

    P_ = Eigen::MatrixXd::Identity(2 * (njnt + 6), 2 * (njnt + 6)) * 1e-6;
    P_.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1;
    P_.block(njnt + 6, njnt + 6, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-3;
    P_.block(3,3,3,3) = Eigen::MatrixXd::Identity(3, 3) * 1;
    P_.block(njnt + 6 + 3, njnt + 6 + 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1;

    Q = Eigen::MatrixXd::Zero(2*(njnt+6), 2*(njnt+6));

    // Rumore su posizione base (m^2)
    Q.block<3,3>(0,0) = 1e-6 * Eigen::Matrix3d::Identity();

    // Rumore su orientazione base (rad^2)
    Q.block<3,3>(3,3) = 1e-6 * Eigen::Matrix3d::Identity();

    // Rumore su giunti (rad^2)
    Q.block(6, 6, njnt, njnt) = 1e-6 * Eigen::MatrixXd::Identity(njnt, njnt);

    // Rumore su velocità lineari+angolari base
    Q.block(njnt+6, njnt+6, 6, 6) = 1e-4 * Eigen::MatrixXd::Identity(6,6);

    // Rumore su velocità giunti
    Q.block(2*6+njnt, 2*6+njnt, njnt, njnt) = 1e-2 * Eigen::MatrixXd::Identity(njnt,njnt);


    R = Eigen::MatrixXd::Zero(n_ekf_output, n_ekf_output);

    // 1) Orientazione IMU (rad^2)
    R.block<3,3>(0,0) = 1e-3 * Eigen::Matrix3d::Identity();

    // 2) Posizione giunti (rad^2)
    R.block(3, 3, njnt, njnt) = 1e-6 * Eigen::MatrixXd::Identity(njnt, njnt);

    // 3) Velocità angolare IMU (rad^2/s^2)
    R.block(njnt+3, njnt+3, 3, 3) = 1e-2 * Eigen::Matrix3d::Identity();

    // 4) Velocità giunti (rad^2/s^2)
    R.block(njnt+6, njnt+6, njnt, njnt) = 1e-1 * Eigen::MatrixXd::Identity(njnt, njnt);

    // 5) Accelerometro IMU (m^2/s^4)
    // R.block(2*njnt+6, 2*njnt+6, 3, 3) = 1e-2 * Eigen::Matrix3d::Identity();

    // 6) Velocità piedi (m^2/s^2)
    R.block(2*njnt+6, 2*njnt+6, 3, 3) = 1e-4 * Eigen::Matrix3d::Identity();
    R.block(2*njnt+9, 2*njnt+9, 3, 3) = 1e-4 * Eigen::Matrix3d::Identity();

    // 7) Posizione piedi (m^2)
    R.block(2*njnt+12, 2*njnt+12, 3, 3) = 1e-4 * Eigen::Matrix3d::Identity();
    R.block(2*njnt+15, 2*njnt+15, 3, 3) = 1e-4 * Eigen::Matrix3d::Identity();

    x_estimate = Eigen::VectorXd::Zero(2 * (njnt + 6));
    x_estimate.head(3) = q_init.head(3);
    x_estimate.segment(3, 3) = rotVecFromQuaternion(Eigen::Quaterniond(
        q_init[6], q_init[3], q_init[4], q_init[5]
    ));
    x_estimate.segment(3 + 3, njnt) = q_init.tail(njnt);
    x_estimate.tail(njnt + 6) = qdot_init;
    y_pred = Eigen::VectorXd::Zero(n_ekf_output);
    y_actual = Eigen::VectorXd::Zero(n_ekf_output);
    y_estimate = Eigen::VectorXd::Zero(n_ekf_output);

    lsole_idx_ = robot_model.getFrameId("left_foot_link");
    rsole_idx_ = robot_model.getFrameId("right_foot_link");
    torso_idx_ = robot_model.getFrameId("torso_link");
    imu_idx_ = robot_model.getFrameId("imu_in_torso");
    const auto& T_lsole_init = sim_robot_data.oMf[lsole_idx_];
    const auto& T_rsole_init = sim_robot_data.oMf[rsole_idx_];

    M_armature_ = Eigen::VectorXd::Zero(njnt);
    for (pinocchio::JointIndex joint_id = 0;
        joint_id < (pinocchio::JointIndex) njnt;
        ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        M_armature_(joint_id) = armatures[joint_name];
    }

    q_jnt_des_ = q_init.tail(njnt);

    // TODO: init using node handle.
    controller_frequency_ = 500;
    controller_timestep_msec_ = 1000 / controller_frequency_;

    walking_data_.initializeWalkingData(
        controller_timestep_msec_,
        labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
        labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation())
    );

    if(!useRobot){
        walking_data_.addSteps(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation())
        );
    };
    

    // Save and read again footstep plan to double check it's working:
    //std::string footstep_plan_path = "/tmp/ditch-footstep-plan-argos.txt";
    //labrob::saveFootstepPlan(walking_data_.footstep_plan, footstep_plan_path);
    //labrob::readFootstepPlan(footstep_plan_path, walking_data_.footstep_plan);
    //labrob::readArgosFootstepPlan(footstep_plan_path, walking_data_.footstep_plan);


    // Init MPC:
    Eigen::Vector3d p_CoM_sim = sim_robot_data.com[0];
    int64_t mpc_prediction_horizon_msec = 2000;
    int64_t mpc_timestep_msec = 100;
    double com_target_height = p_CoM_sim.z() - T_lsole_init.translation().z();
    double foot_constraint_square_length = 0.22;
    double foot_constraint_square_width = 0.08;
    Eigen::Vector3d p_ZMP_sim = p_CoM_sim - Eigen::Vector3d(0.0, 0.0, com_target_height);
    kf_LipState = labrob::LIPState(
        p_CoM_sim,
        Eigen::Vector3d::Zero(),
        p_ZMP_sim
    );
    ismpc_ptr_ = std::make_unique<labrob::ISMPC>(
        mpc_prediction_horizon_msec,
        mpc_timestep_msec,
        std::sqrt(9.81 / com_target_height),
        foot_constraint_square_length,
        foot_constraint_square_width
    );

    auto params = WholeBodyControllerParams::getDefaultParams();
    whole_body_controller_ptr_ = std::make_shared<WholeBodyController>(
        params,
        robot_model,
        q_jnt_des_,
        0.001 * controller_timestep_msec_,
        armatures
    );

    // Init discrete LIP dynamics:
    discrete_lip_dynamics_ptr_ = std::make_unique<labrob::DiscreteLIPDynamics>(
        std::sqrt(9.81 / com_target_height),
        0.001 * controller_timestep_msec_
    );

    discrete_lip_dynamics_ptr_mpc_ = std::make_unique<labrob::DiscreteLIPDynamics>(
        std::sqrt(9.81 / com_target_height),
        0.1 * controller_timestep_msec_
    );

    Kalman_Gain = Eigen::MatrixXd::Zero(2 * (njnt + 6), n_ekf_output);
    // std::ifstream kalman_gain_file("../mean_kalman_gain.txt");
    // if (kalman_gain_file.is_open()) {
    //     for (int i = 0; i < Kalman_Gain.rows(); i++) {
    //         for (int j = 0; j < Kalman_Gain.cols(); j++) {
    //             kalman_gain_file >> Kalman_Gain(i, j);
    //         }
    //     }
    //     kalman_gain_file.close();
    // } else {
    //     std::cerr << "Unable to open file mean_kalman_gain.txt";
    // }

    J_imu_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        estimated_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_est
    );
    J_imu_dot_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobianTimeVariation(
        robot_model,
        estimated_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_dot_est
    );
    J_left_foot_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        estimated_robot_data,
        lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_left_foot_est
    );
    J_right_foot_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        estimated_robot_data,
        rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_right_foot_est
    );

    residual_estimator_ptr_ = std::make_unique<ResidualEstimator>(robot_model, 1.0, armatures);

    return true;
}

RobotState WalkingManager::updateEKF(Eigen::VectorXd actual_output) {

    double left_support_check = 1.0;
    double right_support_check = 1.0;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport){
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT){
            right_support_check = 0.0;
            left_support_check = 1.0;
        }
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT){
            left_support_check = 0.0;
            right_support_check = 1.0;
        }
    }


    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(2 * (njnt + 6));
    x_pred.head(njnt + 6) = x_estimate.head(njnt + 6) + x_estimate.tail(njnt + 6) * 0.001 * controller_timestep_msec_ 
        + 0.5 * (0.001 * controller_timestep_msec_) * (0.001 * controller_timestep_msec_) * whole_body_controller_ptr_->get_q_ddot();
    x_pred.tail(njnt + 6) = x_estimate.tail(njnt + 6) + whole_body_controller_ptr_->get_q_ddot() * controller_timestep_msec_ * 0.001;


    Eigen::VectorXd q_pred = Eigen::VectorXd::Zero(njnt + 7);
    q_pred.head(3) = x_pred.head(3);
    q_pred.segment(3, 4) = Eigen::Vector4d(
        quaternionFromRotVec(x_pred.segment(3, 3)).x(),
        quaternionFromRotVec(x_pred.segment(3, 3)).y(),
        quaternionFromRotVec(x_pred.segment(3, 3)).z(),
        quaternionFromRotVec(x_pred.segment(3, 3)).w()
    );
    q_pred.tail(njnt) = x_pred.segment(3 + 3, njnt);

    pinocchio::forwardKinematics(robot_model, predicted_robot_data, q_pred);
    pinocchio::jacobianCenterOfMass(robot_model, predicted_robot_data, q_pred);
    pinocchio::computeJointJacobians(robot_model, predicted_robot_data, q_pred);
    pinocchio::computeCentroidalMomentum(robot_model, predicted_robot_data, q_pred, x_pred.tail(njnt + 6));
    pinocchio::framesForwardKinematics(robot_model, predicted_robot_data, q_pred);

    Eigen::MatrixXd J_imu_pred = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        predicted_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_pred
    );
    Eigen::MatrixXd J_imu_dot_pred = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobianTimeVariation(
        robot_model,
        predicted_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_dot_pred
    );
    Eigen::MatrixXd J_left_foot_pred = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        predicted_robot_data,
        lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_left_foot_pred
    );
    Eigen::MatrixXd J_right_foot_pred = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        predicted_robot_data,
        rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_right_foot_pred
    );

    Eigen::Quaterniond pred_imu_orientation = Eigen::Quaterniond(
        predicted_robot_data.oMf[imu_idx_].rotation()
    );
    y_pred.head(3) = rotVecFromQuaternion(pred_imu_orientation);
    y_pred.segment(3, njnt) = q_pred.tail(njnt);
    y_pred.segment(njnt + 3, 3) = J_imu_pred.bottomRows(3) * x_pred.tail(njnt + 6);
    y_pred.segment(njnt + 3 + 3, njnt) = x_pred.tail(njnt);
    // y_pred.segment(njnt + 3 + njnt + 3, 3) = J_imu_pred.topRows(3) * whole_body_controller_ptr_->get_q_ddot() + J_imu_dot_pred.topRows(3) * x_pred.tail(njnt + 6);
    y_pred.segment(njnt + 3 + njnt + 3, 3) = J_left_foot_pred.topRows(3) * x_pred.tail(njnt + 6) * left_support_check;
    y_pred.segment(njnt + 3 + njnt + 3 + 3, 3) = J_right_foot_pred.topRows(3) * x_pred.tail(njnt + 6) * right_support_check;
    y_pred.segment(njnt + 3 + njnt + 3 + 6, 3) = predicted_robot_data.oMf[lsole_idx_].translation() * left_support_check;
    y_pred.segment(njnt + 3 + njnt + 3 + 6 + 3, 3) = predicted_robot_data.oMf[rsole_idx_].translation() * right_support_check;


    //MATRICE C:

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_ekf_output, 2 * (njnt + 6));
    // C.block(0, 0, 3, njnt + 6) = J_imu_est.bottomRows(3);
    // C.block(3, 6, njnt, njnt) = Eigen::MatrixXd::Identity(njnt, njnt);
    // C.block(njnt + 3, njnt + 6, 3, njnt + 6) = J_imu_est.bottomRows(3);
    // C.block(njnt + 6, njnt + 6 + 6, njnt, njnt) = Eigen::MatrixXd::Identity(njnt, njnt);
    // C.block(2 * (njnt + 3), njnt + 6, 3, njnt + 6) = J_imu_dot_est.topRows(3);
    // C.block(2 * (njnt + 3) + 3, njnt + 6, 3, njnt + 6) = J_left_foot_est.topRows(3);
    // C.block(2 * (njnt + 3) + 6, njnt + 6, 3, njnt + 6) = J_right_foot_est.topRows(3);
    // C.block(2 * (njnt + 3) + 9, 0, 3, njnt + 6) = J_left_foot_est.topRows(3);
    // C.block(2 * (njnt + 3) + 12, 0, 3, njnt + 6) = J_right_foot_est.topRows(3);

    //MATRICE C 

    // Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_ekf_output, 2 * (njnt + 6));
    C.block(0, 0, 3, njnt + 6) = J_imu_est.bottomRows(3);
    C.block(3, 6, njnt, njnt) = Eigen::MatrixXd::Identity(njnt, njnt);
    C.block(njnt + 3, njnt + 6, 3, njnt + 6) = J_imu_est.bottomRows(3);
    C.block(njnt + 6, njnt + 6 + 6, njnt, njnt) = Eigen::MatrixXd::Identity(njnt, njnt);
    // C.block(2 * (njnt + 3), njnt + 6, 3, njnt + 6) = J_imu_dot_est.topRows(3);
    C.block(2 * (njnt + 3), njnt + 6, 3, njnt + 6) = J_left_foot_est.topRows(3);
    C.block(2 * (njnt + 3) + 3, njnt + 6, 3, njnt + 6) = J_right_foot_est.topRows(3);
    C.block(2 * (njnt + 3) + 6, 0, 3, njnt + 6) = J_left_foot_est.topRows(3);
    C.block(2 * (njnt + 3) + 9, 0, 3, njnt + 6) = J_right_foot_est.topRows(3);



    //MATRICE D:

    // Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_ekf_output, njnt + 6);
    // D.block(2 * (njnt + 6) - 6, 0, 3, njnt + 6) = J_imu_est.topRows(3);

    //MATRICE A:

    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2 * (njnt + 6), 2 * (njnt + 6));
    A.block(0, njnt + 6, njnt + 6, njnt + 6) = controller_timestep_msec_ * 0.001 * Eigen::MatrixXd::Identity(njnt + 6, njnt + 6);

    //PREDICTION COVARIANCE E KALMAN GAIN
    Eigen::MatrixXd Lambda_ = A * P_ * A.transpose() + Q;
    Kalman_Gain = Lambda_ * C.transpose() * (C * Lambda_ * C.transpose() + R).inverse();

    // Eigen::LLT<Eigen::MatrixXd> llt(C * Lambda_ * C.transpose() + R);
    // Eigen::MatrixXd MatInv = llt.solve(Eigen::MatrixXd::Identity(n_ekf_output, n_ekf_output));
    // Kalman_Gain = Lambda_ * C.transpose() * MatInv;

    // Eigen::MatrixXd S = C * Lambda_ * C.transpose() + R;   // innovation covariance
    // Kalman_Gain = Lambda_ * C.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));

    // Eigen::MatrixXd S = C * Lambda_ * C.transpose() + R;
    // Kalman_Gain = Lambda_ * C.transpose();
    // Kalman_Gain = S.ldlt().solve(Kalman_Gain.transpose()).transpose();

    // kalman_gain_log_.push_back(Kalman_Gain);

    P_ = (Eigen::MatrixXd::Identity(2 * (njnt + 6), 2 * (njnt + 6)) - Kalman_Gain * C) * Lambda_;

    y_actual = actual_output;
    // is it transpose?
    Eigen::Matrix3d R_world_imu = predicted_robot_data.oMf[imu_idx_].rotation();
    y_actual.segment(njnt + 3, 3) = R_world_imu * y_actual.segment(njnt + 3, 3);
    // y_actual.segment(njnt + 3 + njnt + 3, 3) = R_world_imu * (y_actual.segment(njnt + 3 + njnt + 3, 3)) - Eigen::Vector3d(0, 0, 9.81);
    
    //get feet position from walking data using desired Gait configuration
    Eigen::Vector3d left_foot_position = walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().p.transpose();
    Eigen::Vector3d right_foot_position = walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().p.transpose();

    y_actual.segment(njnt + 3 + njnt + 3 + 6, 3) = left_foot_position * left_support_check;
    y_actual.segment(njnt + 3 + njnt + 6 + 6, 3) = right_foot_position * right_support_check;

    x_estimate = x_pred + Kalman_Gain * (y_actual - y_pred);

    Eigen::VectorXd q_estimate = Eigen::VectorXd::Zero(njnt + 7);
    q_estimate.head(3) = x_estimate.head(3);
    q_estimate.segment(3, 4) = Eigen::Vector4d(
        quaternionFromRotVec(x_estimate.segment(3, 3)).x(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).y(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).z(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).w()
    );
    q_estimate.tail(njnt) = x_estimate.segment(3 + 3, njnt);

    pinocchio::forwardKinematics(robot_model, estimated_robot_data, q_estimate);
    pinocchio::jacobianCenterOfMass(robot_model, estimated_robot_data, q_estimate);
    pinocchio::computeJointJacobians(robot_model, estimated_robot_data, q_estimate);
    pinocchio::computeCentroidalMomentum(robot_model, estimated_robot_data, q_estimate, x_estimate.tail(njnt + 6));
    pinocchio::framesForwardKinematics(robot_model, estimated_robot_data, q_estimate);

    J_imu_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        estimated_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_est
    );
    J_imu_dot_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobianTimeVariation(
        robot_model,
        estimated_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_dot_est
    );
    J_left_foot_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        estimated_robot_data,
        lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_left_foot_est
    );
    J_right_foot_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        estimated_robot_data,
        rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_right_foot_est
    );


    Eigen::Quaterniond estimated_imu_orientation = Eigen::Quaterniond(
        estimated_robot_data.oMf[imu_idx_].rotation()
    );
    y_estimate.head(3) = rotVecFromQuaternion(estimated_imu_orientation);
    y_estimate.segment(3, njnt) = q_estimate.tail(njnt);
    y_estimate.segment(njnt + 3, 3) = J_imu_est.bottomRows(3) * x_estimate.tail(njnt + 6);
    y_estimate.segment(njnt + 3 + 3, njnt) = x_estimate.tail(njnt);
    // y_estimate.segment(njnt + 3 + njnt + 3, 3) = J_imu_est.topRows(3) * whole_body_controller_ptr_->get_q_ddot() + J_imu_dot_est.topRows(3) * x_estimate.tail(njnt + 6);
    y_estimate.segment(njnt + 3 + njnt + 3, 3) = Eigen::Vector3d::Zero(); //zeros for feet velocities
    y_estimate.segment(njnt + 3 + njnt + 3 + 3, 3) = Eigen::Vector3d::Zero(); //zeros for feet velocities
    y_estimate.segment(njnt + 3 + njnt + 3 + 6, 3) = estimated_robot_data.oMf[lsole_idx_].translation() * left_support_check;
    y_estimate.segment(njnt + 3 + njnt + 3 + 6 + 3, 3) = estimated_robot_data.oMf[rsole_idx_].translation() * right_support_check;

    RobotState current_state;

    current_state.position = x_estimate.head(3);
    current_state.orientation = quaternionFromRotVec(x_estimate.segment<3>(3));
    current_state.linear_velocity = x_estimate.segment(njnt + 6, 3);
    current_state.angular_velocity = x_estimate.segment(njnt + 6 + 3, 3);
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        current_state.joint_state[joint_name].pos = x_estimate(joint_id + 6);
        current_state.joint_state[joint_name].vel = x_estimate(njnt + joint_id + 6 + 6);
    }

    return current_state;
}

LIPState WalkingManager::updateKF(LIPState filtered, LIPState current, const Eigen::Vector3d &input) {
  double omega = ismpc_ptr_->getOmega();

  double ch = cosh(omega*controller_timestep_msec_*0.001);
  double sh = sinh(omega*controller_timestep_msec_*0.001);
  Eigen::MatrixXd A_lip = Eigen::MatrixXd::Zero(3,3);
  Eigen::VectorXd B_lip = Eigen::VectorXd::Zero(3);
  A_lip << ch,sh/omega,1-ch,omega*sh,ch,-omega*sh,0,0,1;
  B_lip << controller_timestep_msec_* 0.001-sh/omega,1-ch,controller_timestep_msec_* 0.001;

  Eigen::Vector3d x_measure, y_measure, z_measure;
  if (std::isnan(current.zmp_pos_(0))) {
    std::cout << "NaN ZMP measurement detected, using filtered value instead." << std::endl;
    x_measure = Eigen::Vector3d(current.com_pos_(0), filtered.com_vel_(0), filtered.zmp_pos_(0));
    y_measure = Eigen::Vector3d(current.com_pos_(1), filtered.com_vel_(1), filtered.zmp_pos_(1));
    z_measure = Eigen::Vector3d(current.com_pos_(2), filtered.com_vel_(2), filtered.zmp_pos_(2));
  } else {
    x_measure = Eigen::Vector3d(current.com_pos_(0), current.com_vel_(0), current.zmp_pos_(0));
    y_measure = Eigen::Vector3d(current.com_pos_(1), current.com_vel_(1), current.zmp_pos_(1));
    z_measure = Eigen::Vector3d(current.com_pos_(2), current.com_vel_(2), current.zmp_pos_(2));
  }
  Eigen::Vector3d x_est = Eigen::Vector3d(filtered.com_pos_(0), filtered.com_vel_(0), filtered.zmp_pos_(0));
  Eigen::Vector3d y_est = Eigen::Vector3d(filtered.com_pos_(1), filtered.com_vel_(1), filtered.zmp_pos_(1));
  Eigen::Vector3d z_est = Eigen::Vector3d(filtered.com_pos_(2), filtered.com_vel_(2), filtered.zmp_pos_(2));

  Eigen::MatrixXd F_kf = A_lip;
  Eigen::MatrixXd G_kf = B_lip;
  Eigen::MatrixXd H_kf = Eigen::Matrix3d::Identity();

  Eigen::MatrixXd R_kf = Eigen::MatrixXd::Identity(3,3);
  R_kf.diagonal() << cov_meas_pos, cov_meas_vel, cov_meas_zmp;
  Eigen::MatrixXd Q_kf = Eigen::MatrixXd::Identity(3,3);
  Q_kf.diagonal() << cov_mod_pos, cov_mod_vel, cov_mod_zmp;

  double input_x = input.x();
  double input_y = input.y();
  double input_z = input.z();

  Eigen::VectorXd x_pred = F_kf * x_est + G_kf * input_x;
  Eigen::MatrixXd cov_x_pred = F_kf * cov_x * F_kf.transpose() + Q_kf;

  Eigen::MatrixXd K_kf = cov_x_pred * H_kf.transpose() * (H_kf * cov_x_pred * H_kf.transpose() + R_kf).inverse();

  x_est = x_pred + K_kf * (x_measure - H_kf * x_pred);
  cov_x = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_x_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();

  Eigen::VectorXd y_pred = F_kf * y_est + G_kf * input_y;
  Eigen::MatrixXd cov_y_pred = F_kf * cov_y * F_kf.transpose() + Q_kf;

  K_kf = cov_y_pred * H_kf.transpose() * (H_kf * cov_y_pred * H_kf.transpose() + R_kf).inverse();

  y_est = y_pred + K_kf * (y_measure - H_kf * y_pred);
  cov_y = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_y_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();

  Eigen::VectorXd z_pred = F_kf * z_est + G_kf * input_z + Eigen::Vector3d(0.0, -9.81 * controller_timestep_msec_* 0.001, 0.0);
  Eigen::MatrixXd cov_z_pred = F_kf * cov_z * F_kf.transpose() + Q_kf;

  K_kf = cov_z_pred * H_kf.transpose() * (H_kf * cov_z_pred * H_kf.transpose() + R_kf).inverse();

  z_est = z_pred + K_kf * (z_measure - H_kf * z_pred);
  cov_z = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_z_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();

  current.com_pos_ = Eigen::Vector3d(x_est(0), y_est(0), z_est(0));
  current.com_vel_ = Eigen::Vector3d(x_est(1), y_est(1), z_est(1));
  current.zmp_pos_ = Eigen::Vector3d(x_est(2), y_est(2), z_est(2));

  return current;
}

LIPState WalkingManager::updateKF2(LIPState filtered, LIPState current, const Eigen::Vector3d &input) {

    double omega = ismpc_ptr_->getOmega();

    double ch = cosh(omega*controller_timestep_msec_*0.001);
    double sh = sinh(omega*controller_timestep_msec_*0.001);
    Eigen::MatrixXd A_lip = Eigen::MatrixXd::Zero(3,3);
    Eigen::VectorXd B_lip = Eigen::VectorXd::Zero(3);
    A_lip << ch,sh/omega,1-ch,omega*sh,ch,-omega*sh,0,0,1;
    B_lip << controller_timestep_msec_* 0.001-sh/omega,1-ch,controller_timestep_msec_* 0.001;

    Eigen::Vector2d x_measure, y_measure, z_measure;
    x_measure = Eigen::Vector2d(current.com_pos_(0), current.com_vel_(0));
    y_measure = Eigen::Vector2d(current.com_pos_(1), current.com_vel_(1));
    z_measure = Eigen::Vector2d(current.com_pos_(2), current.com_vel_(2));
    Eigen::Vector3d x_est = Eigen::Vector3d(filtered.com_pos_(0), filtered.com_vel_(0), filtered.zmp_pos_(0));
    Eigen::Vector3d y_est = Eigen::Vector3d(filtered.com_pos_(1), filtered.com_vel_(1), filtered.zmp_pos_(1));
    Eigen::Vector3d z_est = Eigen::Vector3d(filtered.com_pos_(2), filtered.com_vel_(2), filtered.zmp_pos_(2));

    Eigen::MatrixXd F_kf = A_lip;
    Eigen::MatrixXd G_kf = B_lip;

    Eigen::MatrixXd H_kf = Eigen::MatrixXd::Zero(2, 3);
    H_kf.block(0,0,2,2) = Eigen::MatrixXd::Identity(2,2);

    Eigen::MatrixXd R_kf = Eigen::MatrixXd::Identity(2,2);
    R_kf.diagonal() << cov_meas_pos, cov_meas_vel;

    Eigen::MatrixXd Q_kf = Eigen::MatrixXd::Identity(3,3);
    Q_kf.diagonal() << cov_mod_pos, cov_mod_vel, cov_mod_zmp;

    double input_x = input.x();
    double input_y = input.y();
    double input_z = input.z();

    Eigen::VectorXd x_pred = F_kf * x_est + G_kf * input_x;
    Eigen::MatrixXd cov_x_pred = F_kf * cov_x * F_kf.transpose() + Q_kf;

    Eigen::MatrixXd K_kf = cov_x_pred * H_kf.transpose() * (H_kf * cov_x_pred * H_kf.transpose() + R_kf).inverse();

    x_est = x_pred + K_kf * (x_measure - H_kf * x_pred);
    cov_x = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_x_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();
    Eigen::VectorXd y_pred = F_kf * y_est + G_kf * input_y;
    Eigen::MatrixXd cov_y_pred = F_kf * cov_y * F_kf.transpose() + Q_kf;

    K_kf = cov_y_pred * H_kf.transpose() * (H_kf * cov_y_pred * H_kf.transpose() + R_kf).inverse();

    y_est = y_pred + K_kf * (y_measure - H_kf * y_pred);
    cov_y = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_y_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();

    Eigen::VectorXd z_pred = F_kf * z_est + G_kf * input_z + Eigen::Vector3d(0.0, -9.81 * controller_timestep_msec_* 0.001, 0.0);
    Eigen::MatrixXd cov_z_pred = F_kf * cov_z * F_kf.transpose() + Q_kf;

    K_kf = cov_z_pred * H_kf.transpose() * (H_kf * cov_z_pred * H_kf.transpose() + R_kf).inverse();

    z_est = z_pred + K_kf * (z_measure - H_kf * z_pred);
    cov_z = (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf) * cov_z_pred * (Eigen::MatrixXd::Identity(3,3) - K_kf * H_kf).transpose() + K_kf * R_kf * K_kf.transpose();

    current.com_pos_ = Eigen::Vector3d(x_est(0), y_est(0), z_est(0));
    current.com_vel_ = Eigen::Vector3d(x_est(1), y_est(1), z_est(1));
    current.zmp_pos_ = Eigen::Vector3d(x_est(2), y_est(2), z_est(2));

    return current;
}

void
WalkingManager::update(
    const labrob::RobotState& sim_robot_state,
    labrob::JointCommand& joint_command,
    Eigen::VectorXd actual_output
) {

    auto start_update = std::chrono::high_resolution_clock::now();

    // Update walking state:
    walking_data_.updateWalkingState(t_msec_);

    double eta2 = std::pow(ismpc_ptr_->getOmega(), 2.0);
    double mass = pinocchio::computeTotalMass(robot_model);

    Eigen::Vector3d left_foot_force = estimated_force.head(3);
    Eigen::Vector3d right_foot_force = estimated_force.tail(3);
    Eigen::Vector3d total_force = left_foot_force + right_foot_force;

    auto q = robot_state_to_pinocchio_joint_configuration(robot_model, sim_robot_state);
    auto qdot = robot_state_to_pinocchio_joint_velocity(robot_model, sim_robot_state);

    // Perform forward kinematics on the whole tree and update robot data:
    pinocchio::forwardKinematics(robot_model, sim_robot_data, q);

    // // NOTE: jacobianCenterOfMass calls forwardKinematics and
    //       computeJointJacobians.
    pinocchio::jacobianCenterOfMass(robot_model, sim_robot_data, q);
    pinocchio::computeJointJacobiansTimeVariation(robot_model, sim_robot_data, q, qdot);
    pinocchio::framesForwardKinematics(robot_model, sim_robot_data, q);
    pinocchio::centerOfMass(robot_model, sim_robot_data, q, qdot, 0.0 * qdot); // This is used to compute the CoM drift (J_com_dot * qdot)
    const auto& centroidal_momentum_matrix = pinocchio::ccrba(
        robot_model,
        sim_robot_data,
        q,
        qdot
    );

    auto angular_momentum = (centroidal_momentum_matrix * qdot).tail<3>();

    const auto& T_torso_sim = sim_robot_data.oMf[torso_idx_];
    auto torso_orientation_sim = T_torso_sim.rotation();
    Eigen::MatrixXd J_torso_sim = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        sim_robot_data,
        torso_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_torso_sim
    );

    const auto& T_lsole_sim = sim_robot_data.oMf[lsole_idx_];
    Eigen::MatrixXd J_lsole_sim = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        sim_robot_data,
        lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_lsole_sim
    );

    const auto& v_lsole_sim = J_lsole_sim * qdot;

    const auto& T_rsole_sim = sim_robot_data.oMf[rsole_idx_];
    Eigen::MatrixXd J_rsole_sim = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        sim_robot_data,
        rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_rsole_sim
    );
    const auto& v_rsole_sim = J_rsole_sim * qdot;

    const auto& p_CoM_sim = sim_robot_data.com[0];
    const auto& J_CoM_sim = sim_robot_data.Jcom;
    const auto& a_CoM_drift_sim = sim_robot_data.acom[0];
    Eigen::Vector3d v_CoM_sim = J_CoM_sim * qdot;
    Eigen::Vector3d zmp_3d_sim;
    // zmp_3d_sim.z() = sim_robot_state.position(2) - sim_robot_state.total_force.z() / (mass * eta2);
    // zmp_3d_sim.x() = 0.0;
    // zmp_3d_sim.y() = 0.0;
    // for (int i = 0; i < sim_robot_state.contact_points.size(); ++i) {
    //     auto &pi = sim_robot_state.contact_points[i];
    //     auto &fi = sim_robot_state.contact_forces[i];
    //     zmp_3d_sim.x() += (pi.x() * fi.z() / sim_robot_state.total_force.z() + (zmp_3d_sim.z() - pi.z()) * fi.x() / sim_robot_state.total_force.z());
    //     zmp_3d_sim.y() += (pi.y() * fi.z() / sim_robot_state.total_force.z() + (zmp_3d_sim.z() - pi.z()) * fi.y() / sim_robot_state.total_force.z());
    // }
    zmp_3d_sim.z() = p_CoM_sim.z() - (a_CoM_drift_sim.z() + 9.81) / eta2;
    zmp_3d_sim.x() = p_CoM_sim.x() - a_CoM_drift_sim.x() / eta2;
    zmp_3d_sim.y() = p_CoM_sim.y() - a_CoM_drift_sim.y() / eta2;

    // compute zmp 3d using the 6d vector estimated forces, first three are left foot, second three are right foot
    // zmp_3d_sim.z() = sim_robot_state.position(2) - total_force.z() / (mass * eta2);
    // zmp_3d_sim.x() = 0.0;
    // zmp_3d_sim.y() = 0.0;
    // if (total_force.z() > 1e-5) {
    //     if (left_foot_force.z() > 1e-5) {
    //         zmp_3d_sim.x() += (T_lsole_sim.translation().x() * left_foot_force.z() / total_force.z() + (zmp_3d_sim.z() - T_lsole_sim.translation().z()) * left_foot_force.x() / total_force.z());
    //         zmp_3d_sim.y() += (T_lsole_sim.translation().y() * left_foot_force.z() / total_force.z() + (zmp_3d_sim.z() - T_lsole_sim.translation().z()) * left_foot_force.y() / total_force.z());
    //     }
    //     if (right_foot_force.z() > 1e-5) {
    //         zmp_3d_sim.x() += (T_rsole_sim.translation().x() * right_foot_force.z() / total_force.z() + (zmp_3d_sim.z() - T_rsole_sim.translation().z()) * right_foot_force.x() / total_force.z());
    //         zmp_3d_sim.y() += (T_rsole_sim.translation().y() * right_foot_force.z() / total_force.z() + (zmp_3d_sim.z() - T_rsole_sim.translation().z()) * right_foot_force.y() / total_force.z());
    //     }
    // }
    

    // // compute the angle between the acceleration vector and the gravity vector [0, 0, -9.81]
    // Eigen::Vector3d gravity_vector(0.0, 0.0, -9.81);
    // double angle_acc_gravity = std::acos(imu_accelerometer.normalized().dot(gravity_vector.normalized()));

    // //if time is between 3000 and 6000 msec, compute mean angle
    // if (t_msec_ == 3000) {
    //     angle_acc_gravity_sum_ = 0.0;
    //     angle_acc_gravity_count_ = 0;
    // }
    // if (t_msec_ >= 12000 && t_msec_ <= 14000) {
    //     angle_acc_gravity_sum_ += angle_acc_gravity;
    //     angle_acc_gravity_count_++;
    // }

    // compute rotation to get to sim_robot_state.orientation from imu_orientation which is obtained from first 3 components of actual output
    // if (t_msec_ == 10000){
    //     Eigen::Quaterniond sim_imu_orientation = sim_robot_state.orientation;
    //     Eigen::Quaterniond measured_imu_orientation = quaternionFromRotVec(actual_output.head(3));
    //     rotation_correction = sim_imu_orientation * measured_imu_orientation.conjugate();
    // }

    if (isIMUcalibrating){
        if (t_msec_ - startTimeIMUcalibrating <= 2000){
            imu_accelerometer_sum_ += imu_accelerometer;
            imu_accelerometer_count_++;
        } else {
            std::cout << "IMU calibrated successfully!" << std::endl;
            isIMUcalibrating = false;
            isEKFactive = true;
            startTimeEKF = t_msec_;
            // compute new rotation correction to bring imu accelerometer in line with 0 0 1
            Eigen::Vector3d gravity_vector(0.0, 0.0, 9.81);
            Eigen::Vector3d imu_acc_normalized = (imu_accelerometer_sum_/imu_accelerometer_count_).normalized();
            Eigen::Vector3d gravity_normalized = gravity_vector.normalized();
            double cos_theta = imu_acc_normalized.dot(gravity_normalized);
            //clamp cos_theta to be between -1 and 1
            if (cos_theta > 1.0) {
                cos_theta = 1.0;
            }
            if (cos_theta < -1.0) {
                cos_theta = -1.0;
            }
            double angle = std::acos(cos_theta);
            Eigen::Vector3d rotation_axis = imu_acc_normalized.cross(gravity_normalized);
            if (rotation_axis.norm() < 1e-6) {
                rotation_axis = Eigen::Vector3d(1.0, 0.0, 0.0); // arbitrary axis
            } else {
                rotation_axis.normalize();
            }
            Eigen::AngleAxisd angle_axis_rotation(angle, rotation_axis);
            rotation_correction = Eigen::Quaterniond(angle_axis_rotation);

            //rotate to z robot axis from imu feedback orientation
            //SECOND POSSIBLE ROTATION
            // rotation_correction = sim_robot_state.orientation * quaternionFromRotVec(actual_output.head(3)).conjugate();
        }
    }

    if (isEKFactive && t_msec_ >= startTimeEKF) {
        Eigen::Quaterniond feedback_imu_orientation = quaternionFromRotVec(actual_output.head(3));
        Eigen::Quaterniond corrected_imu_orientation = rotation_correction * feedback_imu_orientation;
        actual_output.head(3) = rotVecFromQuaternion(corrected_imu_orientation);

        Eigen::Matrix3d rotation_matrix = rotation_correction.toRotationMatrix();
        actual_output.segment(3 + njnt, 3) = rotation_matrix * actual_output.segment(3 + njnt, 3);

        imu_accelerometer = rotation_matrix * imu_accelerometer;

        RobotState base_estimation_robot_state;
        base_estimation_robot_state.position = Eigen::Vector3d(0,0,0);
        base_estimation_robot_state.orientation = quaternionFromRotVec(actual_output.head(3));
        for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
            std::string joint_name = robot_model.names[joint_id + 2];
            base_estimation_robot_state.joint_state[joint_name].pos = actual_output(3 + joint_id);
            base_estimation_robot_state.joint_state[joint_name].vel = actual_output(njnt + 3 + 3 + joint_id);
        }
        base_estimation_robot_state.linear_velocity = Eigen::Vector3d::Zero();
        base_estimation_robot_state.angular_velocity = Eigen::Vector3d::Zero();

        pinocchio::Data base_estimation_robot_data(robot_model);
        auto q_base_est = robot_state_to_pinocchio_joint_configuration(robot_model, base_estimation_robot_state);
        pinocchio::forwardKinematics(robot_model, base_estimation_robot_data, q_base_est);
        pinocchio::framesForwardKinematics(robot_model, base_estimation_robot_data, q_base_est);

        double foot_line_angle;
        if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
            if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT){

                Eigen::Vector3d left_foot_orientation = base_estimation_robot_data.oMf[lsole_idx_].rotation() * Eigen::Vector3d::UnitX();
                double left_foot_yaw = atan2(left_foot_orientation.y(), left_foot_orientation.x());
                foot_line_angle = left_foot_yaw;
            }
            else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT){
                Eigen::Vector3d right_foot_orientation = base_estimation_robot_data.oMf[rsole_idx_].rotation() * Eigen::Vector3d::UnitX();
                double right_foot_yaw = atan2(right_foot_orientation.y(), right_foot_orientation.x());
                foot_line_angle = right_foot_yaw;
            }
        }else{
            Eigen::Vector3d left_foot_orientation = base_estimation_robot_data.oMf[lsole_idx_].rotation() * Eigen::Vector3d::UnitX();
            double left_foot_yaw = atan2(left_foot_orientation.y(), left_foot_orientation.x());
            Eigen::Vector3d right_foot_orientation = base_estimation_robot_data.oMf[rsole_idx_].rotation() * Eigen::Vector3d::UnitX();
            double right_foot_yaw = atan2(right_foot_orientation.y(), right_foot_orientation.x());
            foot_line_angle = 0.5 * (left_foot_yaw + right_foot_yaw);
        }

        Eigen::AngleAxisd yaw_correction(-foot_line_angle, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond q_yaw(yaw_correction);
        actual_output.head(3) = rotVecFromQuaternion(q_yaw * base_estimation_robot_state.orientation);

        //rotate angular velocity as well
        actual_output.segment(3 + njnt, 3) = q_yaw.toRotationMatrix() * actual_output.segment(3 + njnt, 3);


    }

    if(!useRobot){
        Eigen::Vector3d left_foot_position;
        Eigen::Vector3d right_foot_position;

        left_foot_position = walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().p.transpose();
        right_foot_position = walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().p.transpose();
        //fill actual output with sim robot state values
        actual_output.head(3) = rotVecFromQuaternion(sim_robot_state.orientation);
        for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
            std::string joint_name = robot_model.names[joint_id + 2];
            actual_output(3 + joint_id) = sim_robot_state.joint_state[joint_name].pos;
            actual_output(3 + njnt + 3 + joint_id) = sim_robot_state.joint_state[joint_name].vel;
        }
        actual_output.segment(njnt + 3, 3) = sim_robot_state.angular_velocity;
        actual_output.segment(njnt + 3 + njnt + 3, 3) = Eigen::Vector3d::Zero(); //zeros for feet velocities
        actual_output.segment(njnt + 3 + njnt + 3 + 6, 3) = left_foot_position;
        actual_output.segment(njnt + 3 + njnt + 3 + 6 + 3, 3) = right_foot_position;
    }


    ////////////////////////
    // BASE ESTIMATION
    ////////////////////////
    // WORK IN PROGRESS
    
    //start measuring time
    auto start_ekf = std::chrono::high_resolution_clock::now();
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            if(t_msec_ >= startTimeEKF && isEKFactive) {
                fb_robot_state = updateEKF(actual_output);
            }
            else{
                //TODO: don't know what to do if not using the EKF
                fb_robot_state = sim_robot_state;
            }

        }
        #pragma omp section
        {
        }
    } // end of parallel sections
    auto end_ekf = std::chrono::high_resolution_clock::now();

    ////////////////////
    // END BASE ESTIMATE
    ///////////////////



    auto q_fb_filt = robot_state_to_pinocchio_joint_configuration(robot_model, fb_robot_state);
    auto qdot_fb_filt = robot_state_to_pinocchio_joint_velocity(robot_model, fb_robot_state);

    // Perform forward kinematics on the whole tree and update robot data:
    pinocchio::forwardKinematics(robot_model, fb_robot_data, q_fb_filt);

    // // NOTE: jacobianCenterOfMass calls forwardKinematics and
    //       computeJointJacobians.
    pinocchio::jacobianCenterOfMass(robot_model, fb_robot_data, q_fb_filt);
    pinocchio::computeJointJacobiansTimeVariation(robot_model, fb_robot_data, q_fb_filt, qdot_fb_filt);
    pinocchio::framesForwardKinematics(robot_model, fb_robot_data, q_fb_filt);
    pinocchio::centerOfMass(robot_model, fb_robot_data, q_fb_filt, qdot_fb_filt, 0.0 * qdot_fb_filt); // This is used to compute the CoM drift (J_com_dot * qdot)

    const auto& p_CoM_fb = fb_robot_data.com[0];
    const auto& a_CoM_drift_fb = fb_robot_data.acom[0];
    const auto& J_CoM_fb = fb_robot_data.Jcom;
    Eigen::Vector3d v_CoM_fb = J_CoM_fb * qdot_fb_filt;
    const auto& T_torso_fb = fb_robot_data.oMf[torso_idx_];
    auto torso_orientation_fb = T_torso_fb.rotation();
    Eigen::MatrixXd J_torso_fb = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        fb_robot_data,
        torso_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_torso_fb
    );

    const auto& T_lsole_fb = fb_robot_data.oMf[lsole_idx_];
    Eigen::MatrixXd J_lsole_fb = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        fb_robot_data,
        lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_lsole_fb
    );

    const auto& v_lsole_fb = J_lsole_fb * qdot_fb_filt;

    const auto& T_rsole_fb = fb_robot_data.oMf[rsole_idx_];
    Eigen::MatrixXd J_rsole_fb = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        fb_robot_data,
        rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_rsole_fb
    );
    const auto& v_rsole_fb = J_rsole_fb * qdot_fb_filt;   

    Eigen::Vector3d zmp_3d_fb;
    // zmp_3d_fb.z() = fb_robot_state.position(2) - fb_robot_state.total_force.z() / (mass * eta2);
    // zmp_3d_fb.x() = 0.0;
    // zmp_3d_fb.y() = 0.0;
    // for (int i = 0; i < fb_robot_state.contact_points.size(); ++i) {
    //     auto &pi = fb_robot_state.contact_points[i];
    //     auto &fi = fb_robot_state.contact_forces[i];
    //     zmp_3d_fb.x() += (pi.x() * fi.z() / fb_robot_state.total_force.z() + (zmp_3d_fb.z() - pi.z()) * fi.x() / fb_robot_state.total_force.z());
    //     zmp_3d_fb.y() += (pi.y() * fi.z() / fb_robot_state.total_force.z() + (zmp_3d_fb.z() - pi.z()) * fi.y() / fb_robot_state.total_force.z());
    // }

    zmp_3d_fb.z() = fb_robot_state.position(2) - total_force.z() / (mass * eta2);
    zmp_3d_fb.x() = 0.0;
    zmp_3d_fb.y() = 0.0;
    if (total_force.z() > 1e-5) {
        if (left_foot_force.z() > 1e-5) {
            zmp_3d_fb.x() += (T_lsole_fb.translation().x() * left_foot_force.z() / total_force.z() + (zmp_3d_fb.z() - T_lsole_fb.translation().z()) * left_foot_force.x() / total_force.z());
            zmp_3d_fb.y() += (T_lsole_fb.translation().y() * left_foot_force.z() / total_force.z() + (zmp_3d_fb.z() - T_lsole_fb.translation().z()) * left_foot_force.y() / total_force.z());
        }
        if (right_foot_force.z() > 1e-5) {
            zmp_3d_fb.x() += (T_rsole_fb.translation().x() * right_foot_force.z() / total_force.z() + (zmp_3d_fb.z() - T_rsole_fb.translation().z()) * right_foot_force.x() / total_force.z());
            zmp_3d_fb.y() += (T_rsole_fb.translation().y() * right_foot_force.z() / total_force.z() + (zmp_3d_fb.z() - T_rsole_fb.translation().z()) * right_foot_force.y() / total_force.z());
        }
    }

    ef_zmp_position_log_.push_back(zmp_3d_fb.transpose());

    // zmp_3d_fb.z() = p_CoM_fb.z() - (a_CoM_drift_fb.z() + 9.81) / eta2;
    // zmp_3d_fb.x() = p_CoM_fb.x() - a_CoM_drift_fb.x() / eta2;
    // zmp_3d_fb.y() = p_CoM_fb.y() - a_CoM_drift_fb.y() / eta2;

    integrated_state_pos.head(3) = sim_robot_state.position;
    integrated_state_pos.segment<3>(3) = rotVecFromQuaternion(sim_robot_state.orientation);
    integrated_state_vel.head(3) = sim_robot_state.linear_velocity;
    integrated_state_vel.segment<3>(3) = sim_robot_state.angular_velocity;
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        integrated_state_pos(6 + joint_id) = sim_robot_state.joint_state[joint_name].pos;
        integrated_state_vel(6 + joint_id) = sim_robot_state.joint_state[joint_name].vel;
    }

    if(isTotalBodyLoopClosed && t_msec_ >= startTimeTotalBodyCL){
        integrated_state_pos.head(3) = fb_robot_state.position;
        integrated_state_pos.segment<3>(3) = rotVecFromQuaternion(fb_robot_state.orientation);
        integrated_state_vel.head(3) = fb_robot_state.linear_velocity;
        integrated_state_vel.segment<3>(3) = fb_robot_state.angular_velocity;
        for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
            std::string joint_name = robot_model.names[joint_id + 2];
            integrated_state_pos(6 + joint_id) = fb_robot_state.joint_state[joint_name].pos;
            integrated_state_vel(6 + joint_id) = fb_robot_state.joint_state[joint_name].vel;
        }
    }


    /////////////////////////////////////
    // 
    // START KF
    //
    /////////////////////////////////////

    auto start_kf = std::chrono::high_resolution_clock::now();
    LipState = LIPState(p_CoM_sim, J_CoM_sim * qdot, zmp_3d_sim);
    if (t_msec_ >= startTimeCoMCL && isCoMLoopClosed){
        if (t_msec_ == startTimeCoMCL){
            std::cout << "Using feedback Center of Mass" << std::endl;
        }
        LipState = LIPState(p_CoM_fb, J_CoM_fb * qdot_fb_filt, zmp_3d_fb);
    }
    kf_LipState = updateKF(kf_LipState, LipState, ismpc_ptr_->getInput());
    auto end_kf = std::chrono::high_resolution_clock::now();

    ////////////////////////////////////
    // END KF
    ////////////////////////////////////


    if (switchWalkingState){
        if (walking_data_.getWalkingState() == WalkingState::Standing) {
            walking_data_.addSteps(
                labrob::SE3(T_lsole_sim.rotation(), T_lsole_sim.translation()),
                labrob::SE3(T_rsole_sim.rotation(), T_rsole_sim.translation())
            );
            switchWalkingState = false;
        } else if (walking_data_.getWalkingState() == WalkingState::DoubleSupport) {
            std::cout << "Removing steps" << std::endl;
            walking_data_.removeSteps();
            switchWalkingState = false;
        }
    }

    

    // Fill current gait configuration:
    labrob::GaitConfiguration current_gait_configuration;
    current_gait_configuration.qjnt = q.tail(njnt);
    current_gait_configuration.qjntdot = qdot.tail(njnt);
    if (t_msec_ >= startTimeTotalBodyCL && isTotalBodyLoopClosed){
        current_gait_configuration.qjnt = q_fb_filt.tail(njnt);
        current_gait_configuration.qjntdot = qdot_fb_filt.tail(njnt);
    }

    current_gait_configuration.is_left_foot_support = true;
    current_gait_configuration.is_right_foot_support = true;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
    if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT) current_gait_configuration.is_right_foot_support = false;
    else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT) current_gait_configuration.is_left_foot_support = false;
    }

    // current_gait_configuration.com.pos = p_CoM_sim;
    // current_gait_configuration.com.vel = v_CoM_sim;
    // if (t_msec_ >= startTimeCoMCL && isCoMLoopClosed){
    //     current_gait_configuration.com.pos = p_CoM_fb;
    //     current_gait_configuration.com.vel = v_CoM_fb;
    // }


    current_gait_configuration.com.pos = kf_LipState.com_pos_;
    current_gait_configuration.com.vel = kf_LipState.com_vel_;
    
    current_gait_configuration.torso.pos = sim_robot_data.oMf[torso_idx_].rotation();
    current_gait_configuration.torso.vel = J_torso_sim.bottomRows<3>() * qdot;
    if (t_msec_ >= startTimeTotalBodyCL && isTotalBodyLoopClosed){
        current_gait_configuration.torso.pos = fb_robot_data.oMf[torso_idx_].rotation();
        current_gait_configuration.torso.vel = J_torso_fb.bottomRows<3>() * qdot_fb_filt;
    }

    current_gait_configuration.lsole.pos = labrob::SE3(sim_robot_data.oMf[lsole_idx_].rotation(), sim_robot_data.oMf[lsole_idx_].translation());
    current_gait_configuration.lsole.vel = J_lsole_sim * qdot;
    if (t_msec_ >= startTimeTotalBodyCL && isTotalBodyLoopClosed){
        current_gait_configuration.lsole.pos = labrob::SE3(fb_robot_data.oMf[lsole_idx_].rotation(), fb_robot_data.oMf[lsole_idx_].translation());
        current_gait_configuration.lsole.vel = J_lsole_fb * qdot_fb_filt;
    }

    current_gait_configuration.rsole.pos = labrob::SE3(sim_robot_data.oMf[rsole_idx_].rotation(), sim_robot_data.oMf[rsole_idx_].translation());
    current_gait_configuration.rsole.vel = J_rsole_sim * qdot;
    if (t_msec_ >= startTimeTotalBodyCL && isTotalBodyLoopClosed){
        current_gait_configuration.rsole.pos = labrob::SE3(fb_robot_data.oMf[rsole_idx_].rotation(), fb_robot_data.oMf[rsole_idx_].translation());
        current_gait_configuration.rsole.vel = J_rsole_fb * qdot_fb_filt;
    }

    /////////////////////////////////////
    // 
    // START MPC
    //
    /////////////////////////////////////

    auto start_mpc = std::chrono::system_clock::now();
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            ismpc_ptr_->solve(t_msec_, walking_data_, kf_LipState);
        }
        #pragma omp section
        {
        }
    } 
    auto end_mpc = std::chrono::system_clock::now();

    LIPState lip_state;
    lip_state = discrete_lip_dynamics_ptr_->integrate(kf_LipState, ismpc_ptr_->getInput());
    // lip_state = kf_LipState;

    // Eigen::VectorXd inputSequenceX = ismpc_ptr_->getInputSequenceX();
    // Eigen::VectorXd inputSequenceY = ismpc_ptr_->getInputSequenceY();
    // Eigen::VectorXd inputSequenceZ = ismpc_ptr_->getInputSequenceZ();

    // LIPState LipState_mpc = kf_LipState;

    // for (int i = 0; i < 20; ++i) {
    //     LipState_mpc = discrete_lip_dynamics_ptr_mpc_->integrate(
    //         LipState_mpc,
    //         Eigen::Vector3d(inputSequenceX(i), inputSequenceY(i), inputSequenceZ(i))
    //     );

    //     mpc_predictions_log_.push_back(LipState_mpc.com_pos_);
    //     mpc_predictions_vel_log_.push_back(LipState_mpc.com_vel_);
    //     mpc_predictions_zmp_log_.push_back(LipState_mpc.zmp_pos_);
    // }

    Eigen::Vector3d p_CoM_des = lip_state.com_pos_;
    Eigen::Vector3d v_CoM_des = lip_state.com_vel_;
    Eigen::Vector3d p_ZMP_des = lip_state.zmp_pos_;

    // Fill desired gait configuration:
    labrob::GaitConfiguration desired_gait_configuration;
    desired_gait_configuration.qjnt = q_jnt_des_;
    desired_gait_configuration.qjntdot = Eigen::VectorXd::Zero(njnt);
    desired_gait_configuration.qjntddot = Eigen::VectorXd::Zero(njnt);

    desired_gait_configuration.com.pos = lip_state.com_pos_;
    desired_gait_configuration.com.vel = lip_state.com_vel_;
    desired_gait_configuration.com.acc = eta2 * (lip_state.com_pos_ - lip_state.zmp_pos_) - Eigen::Vector3d(0.0, 0.0, 9.81);

    // Feet tasks
    if (current_gait_configuration.is_left_foot_support && current_gait_configuration.is_right_foot_support) {
        desired_gait_configuration.lsole.pos = walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration();
        desired_gait_configuration.lsole.vel = Eigen::VectorXd::Zero(6);
        desired_gait_configuration.lsole.acc = Eigen::VectorXd::Zero(6);
        desired_gait_configuration.rsole.pos = walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration();
        desired_gait_configuration.rsole.vel = Eigen::VectorXd::Zero(6);
        desired_gait_configuration.rsole.acc = Eigen::VectorXd::Zero(6);
    } else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT) {
        desired_gait_configuration.lsole.pos = walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration();
        desired_gait_configuration.lsole.vel = Eigen::VectorXd::Zero(6);
        desired_gait_configuration.lsole.acc = Eigen::VectorXd::Zero(6);
        pinocchio::SE3 desired_rsole_pose;
        pinocchio::Motion desired_rsole_vel;
        pinocchio::Motion desired_rsole_acc;
        swingFootTrajectory(desired_rsole_pose, desired_rsole_vel, desired_rsole_acc);
        desired_gait_configuration.rsole.pos.R = desired_rsole_pose.rotation();
        desired_gait_configuration.rsole.pos.p = desired_rsole_pose.translation();
        desired_gait_configuration.rsole.vel << desired_rsole_vel.linear(), desired_rsole_vel.angular();
        desired_gait_configuration.rsole.acc << desired_rsole_acc.linear(), desired_rsole_acc.angular();
    } else {
        desired_gait_configuration.rsole.pos = walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration();
        desired_gait_configuration.rsole.vel = Eigen::VectorXd::Zero(6);
        desired_gait_configuration.rsole.acc = Eigen::VectorXd::Zero(6);
        pinocchio::SE3 desired_lsole_pose;
        pinocchio::Motion desired_lsole_vel;
        pinocchio::Motion desired_lsole_acc;
        swingFootTrajectory(desired_lsole_pose, desired_lsole_vel, desired_lsole_acc);
        desired_gait_configuration.lsole.pos.R = desired_lsole_pose.rotation();
        desired_gait_configuration.lsole.pos.p = desired_lsole_pose.translation();
        desired_gait_configuration.lsole.vel << desired_lsole_vel.linear(), desired_lsole_vel.angular();
        desired_gait_configuration.lsole.acc << desired_lsole_acc.linear(), desired_lsole_acc.angular();
    }

    // Torso task
    double left_foot_yaw = std::atan2(desired_gait_configuration.lsole.pos.R(1, 0), desired_gait_configuration.lsole.pos.R(0, 0));
    double right_foot_yaw = std::atan2(desired_gait_configuration.rsole.pos.R(1, 0), desired_gait_configuration.rsole.pos.R(0, 0));
    desired_gait_configuration.torso.pos = Rz((left_foot_yaw + right_foot_yaw) / 2.0);
    desired_gait_configuration.torso.vel = (desired_gait_configuration.lsole.vel.tail(3) + desired_gait_configuration.rsole.vel.tail(3)) / 2.0;
    desired_gait_configuration.torso.acc = (desired_gait_configuration.lsole.acc.tail(3) + desired_gait_configuration.rsole.acc.tail(3)) / 2.0;

    /////////////////////////////////////
    // 
    // START WHOLE BODY CONTROLLER
    //
    /////////////////////////////////////

    auto start_wbc = std::chrono::system_clock::now();
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            if (t_msec_ >= startTimeTotalBodyCL && isTotalBodyLoopClosed){
                joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
                    robot_model,
                    fb_robot_state,
                    fb_robot_state,
                    fb_robot_data,
                    fb_robot_data,
                    current_gait_configuration,
                    desired_gait_configuration
                );
            }else if (t_msec_ >= startTimeCoMCL && isCoMLoopClosed && !isTotalBodyLoopClosed){
                joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
                    robot_model,
                    sim_robot_state,
                    fb_robot_state,
                    sim_robot_data,
                    fb_robot_data,
                    current_gait_configuration,
                    desired_gait_configuration
                );
            } else {
                // Use the MPC to compute the joint command:
                joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
                    robot_model,
                    sim_robot_state,
                    sim_robot_state,
                    sim_robot_data,
                    sim_robot_data,
                    current_gait_configuration,
                    desired_gait_configuration
                );
            }
        }
        #pragma omp section
        {
        }
    } // end of parallel sections
    auto end_wbc = std::chrono::system_clock::now();

    // Get measured joint torques from the joint command
    Eigen::VectorXd measured_torques(robot_model.nv - 6);  // Exclude floating base
    int idx = 0;
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) robot_model.njoints; ++joint_id) {
        const auto& joint_name = robot_model.names[joint_id];
        measured_torques(idx++) = joint_command[joint_name];
    }


    // In your update function, after WBC computation:
    Eigen::VectorXd wbc_left_wrench = whole_body_controller_ptr_->getLeftFootWrench();
    Eigen::VectorXd wbc_right_wrench = whole_body_controller_ptr_->getRightFootWrench();

    residual_estimator_ptr_->computeResidualWithWBCWrenches(
        fb_robot_state,
        fb_robot_data,
        measured_torques,
        //wbc_left_wrench,
        //wbc_right_wrench,
        *whole_body_controller_ptr_,
        controller_timestep_msec_*0.001
    );
    estimated_force = residual_estimator_ptr_->getFeetEstimatedForce();



    // Update timing in milliseconds.
    // NOTE: assuming update() is actually called every controller_timestep_msec_
    //       milliseconds.
    t_msec_ += controller_timestep_msec_;
    prev_angular_momentum_ = angular_momentum;


    sim_com_position_log_.push_back(p_CoM_sim.transpose());
    fb_com_position_log_.push_back(p_CoM_fb.transpose());
    kf_com_position_log_.push_back(kf_LipState.com_pos_.transpose());
    des_com_position_log_.push_back(p_CoM_des.transpose());

    sim_com_velocity_log_.push_back(v_CoM_sim.transpose());
    fb_com_velocity_log_.push_back(v_CoM_fb.transpose());
    kf_com_velocity_log_.push_back((kf_LipState.com_vel_).transpose());
    des_com_velocity_log_.push_back(v_CoM_des.transpose());

    sim_zmp_position_log_.push_back(zmp_3d_sim.transpose());
    fb_zmp_position_log_.push_back(zmp_3d_fb.transpose());
    kf_zmp_position_log_.push_back(kf_LipState.zmp_pos_.transpose());
    des_zmp_position_log_.push_back(p_ZMP_des.transpose());

    p_lsole_sim_log_.push_back(T_lsole_sim.translation().transpose());
    p_rsole_sim_log_.push_back(T_rsole_sim.translation().transpose());
    v_lsole_sim_log_.push_back(v_lsole_sim.head<3>().transpose());
    v_rsole_sim_log_.push_back(v_rsole_sim.head<3>().transpose());
    p_lsole_fb_log_.push_back(T_lsole_fb.translation().transpose());
    p_rsole_fb_log_.push_back(T_rsole_fb.translation().transpose());
    v_lsole_fb_log_.push_back(v_lsole_fb.head<3>().transpose());
    v_rsole_fb_log_.push_back(v_rsole_fb.head<3>().transpose());
    p_lsole_des_log_.push_back(desired_gait_configuration.lsole.pos.p.transpose());
    p_rsole_des_log_.push_back(desired_gait_configuration.rsole.pos.p.transpose());
    v_lsole_des_log_.push_back(desired_gait_configuration.lsole.vel.head<3>().transpose());
    v_rsole_des_log_.push_back(desired_gait_configuration.rsole.vel.head<3>().transpose());

    estimated_force_lsole_log_.push_back(estimated_force.head<3>().transpose());
    estimated_force_rsole_log_.push_back(estimated_force.tail<3>().transpose());

    angular_momentum_log_.push_back(angular_momentum.transpose());
    // log measurements present in actual output
    measured_imu_orientation_log_.push_back(Eigen::Vector4d(
        quaternionFromRotVec(actual_output.head(3)).w(),
        quaternionFromRotVec(actual_output.head(3)).x(),
        quaternionFromRotVec(actual_output.head(3)).y(),
        quaternionFromRotVec(actual_output.head(3)).z()
    ).transpose());
    measured_imu_angular_velocity_log_.push_back(actual_output.segment<3>(3 + njnt).transpose());
    measured_joint_position_log_.push_back(Eigen::VectorXd(njnt).transpose());
    measured_joint_velocity_log_.push_back(Eigen::VectorXd(njnt).transpose());
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        measured_joint_position_log_.back()(joint_id) = actual_output(3 + joint_id);
        measured_joint_velocity_log_.back()(joint_id) = actual_output(3 + njnt + 3 + joint_id);
    }
    measured_imu_accelerometer_log_.push_back(imu_accelerometer.transpose());
    // Log the filtered state:
    ekf_base_position_log_.push_back(fb_robot_state.position.transpose());
    ekf_base_velocity_log_.push_back(fb_robot_state.linear_velocity.transpose());
    ekf_base_orientation_log_.push_back(Eigen::Vector4d(
        fb_robot_state.orientation.w(),
        fb_robot_state.orientation.x(),
        fb_robot_state.orientation.y(),
        fb_robot_state.orientation.z()
    ).transpose());
    ekf_base_angular_velocity_log_.push_back(fb_robot_state.angular_velocity.transpose());
    ekf_joint_position_log_.push_back(Eigen::VectorXd(njnt).transpose());
    ekf_joint_velocity_log_.push_back(Eigen::VectorXd(njnt).transpose());
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        ekf_joint_position_log_.back()(joint_id) = fb_robot_state.joint_state[joint_name].pos;
        ekf_joint_velocity_log_.back()(joint_id) = fb_robot_state.joint_state[joint_name].vel;
    }
    sim_base_position_log_.push_back(sim_robot_state.position.transpose());
    sim_base_velocity_log_.push_back(sim_robot_state.linear_velocity.transpose());
    sim_base_orientation_log_.push_back(Eigen::Vector4d(
        sim_robot_state.orientation.w(),
        sim_robot_state.orientation.x(),
        sim_robot_state.orientation.y(),
        sim_robot_state.orientation.z()
    ).transpose());
    sim_base_angular_velocity_log_.push_back(sim_robot_state.angular_velocity.transpose());
    sim_joint_position_log_.push_back(Eigen::VectorXd(njnt).transpose());
    sim_joint_velocity_log_.push_back(Eigen::VectorXd(njnt).transpose());
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        sim_joint_position_log_.back()(joint_id) = sim_robot_state.joint_state[joint_name].pos;
        sim_joint_velocity_log_.back()(joint_id) = sim_robot_state.joint_state[joint_name].vel;
    }
    input_torque_log_.push_back(Eigen::VectorXd(njnt).transpose());
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        input_torque_log_.back()(joint_id) = joint_command[joint_name];
    }

    auto end_update = std::chrono::high_resolution_clock::now();


    auto update_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_update - start_update).count();
    auto ekf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ekf - start_ekf).count();
    auto kf_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_kf - start_kf).count();
    auto mpc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_mpc - start_mpc).count();
    auto wbc_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_wbc - start_wbc).count();

    execution_time_update_log_.push_back(update_duration);
    execution_time_ekf_log_.push_back(ekf_duration);
    execution_time_kf_log_.push_back(kf_duration);
    execution_time_mpc_log_.push_back(mpc_duration);
    execution_time_wbc_log_.push_back(wbc_duration);
}

RobotState WalkingManager::getNewRobotState(RobotState robot_state){
    Eigen::VectorXd wbc_qddot = whole_body_controller_ptr_->get_q_ddot();
    // integrated_state_pos = integrated_state_pos + integrated_state_vel * controller_timestep_msec_ * 0.001 + 0.5 * wbc_qddot * std::pow(controller_timestep_msec_ * 0.001, 2);
    // integrated_state_vel = integrated_state_vel + wbc_qddot * controller_timestep_msec_ * 0.001;


    integrated_state_vel = integrated_state_vel + wbc_qddot * controller_timestep_msec_ * 0.001;
    integrated_state_pos = integrated_state_pos + integrated_state_vel * controller_timestep_msec_ * 0.001;

    
    robot_state.position = integrated_state_pos.head<3>();
    robot_state.orientation = quaternionFromRotVec(integrated_state_pos.segment<3>(3));
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        robot_state.joint_state[joint_name].pos = integrated_state_pos(6 + joint_id);
        robot_state.joint_state[joint_name].vel = integrated_state_vel(6 + joint_id);
    }
    robot_state.linear_velocity = integrated_state_vel.head<3>();
    robot_state.angular_velocity = integrated_state_vel.segment<3>(3);
    return robot_state;
}

void WalkingManager::saveLogs() {

    //compute mean kalman gain matrix and save it to a file
    Eigen::MatrixXd mean_Kalman_Gain = Eigen::MatrixXd::Zero(kalman_gain_log_[0].rows(), kalman_gain_log_[0].cols());
    for (auto& Kalman_Gain : kalman_gain_log_) {
        mean_Kalman_Gain += Kalman_Gain;
    }
    mean_Kalman_Gain /= kalman_gain_log_.size();
    std::ofstream mean_kalman_gain_file("../mean_kalman_gain.txt");
    for (int i = 0; i < mean_Kalman_Gain.rows(); ++i) {
        for (int j = 0; j < mean_Kalman_Gain.cols(); ++j) {
            mean_kalman_gain_file << mean_Kalman_Gain(i, j);
            if (j < mean_Kalman_Gain.cols() - 1) mean_kalman_gain_file << " ";
        }
        mean_kalman_gain_file << "\n";
    }

    std::ofstream joint_names_file("/tmp/joint_names.txt");
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        joint_names_file << joint_name << "\n";
    }

    std::ofstream sim_com_position_file("/tmp/sim_com_position.txt");
    for (auto& v : sim_com_position_log_) {
        sim_com_position_file << v.transpose() << "\n";
    }

    std::ofstream sim_com_velocity_file("/tmp/sim_com_velocity.txt");
    for (auto& v : sim_com_velocity_log_) {
        sim_com_velocity_file << v.transpose() << "\n";
    }

    std::ofstream sim_zmp_position_file("/tmp/sim_zmp_position.txt");
    for (auto& v : sim_zmp_position_log_) {
        sim_zmp_position_file << v.transpose() << "\n";
    }

    std::ofstream fb_com_position_file("/tmp/fb_com_position.txt");
    for (auto& v : fb_com_position_log_) {
        fb_com_position_file << v.transpose() << "\n";
    }

    std::ofstream fb_com_velocity_file("/tmp/fb_com_velocity.txt");
    for (auto& v : fb_com_velocity_log_) {
        fb_com_velocity_file << v.transpose() << "\n";
    }

    std::ofstream fb_zmp_position_file("/tmp/fb_zmp_position.txt");
    for (auto& v : fb_zmp_position_log_) {
        fb_zmp_position_file << v.transpose() << "\n";
    }

    std::ofstream kf_com_position_file("/tmp/kf_com_position.txt");
    for (auto& v : kf_com_position_log_) {
        kf_com_position_file << v.transpose() << "\n";
    }

    std::ofstream kf_com_velocity_file("/tmp/kf_com_velocity.txt");
    for (auto& v : kf_com_velocity_log_) {
        kf_com_velocity_file << v.transpose() << "\n";
    }   

    std::ofstream kf_zmp_position_file("/tmp/kf_zmp_position.txt");
    for (auto& v : kf_zmp_position_log_) {
        kf_zmp_position_file << v.transpose() << "\n";
    }

    std::ofstream des_com_position_file("/tmp/des_com_position.txt");
    for (auto& v : des_com_position_log_) {
        des_com_position_file << v.transpose() << "\n";
    }

    std::ofstream des_com_velocity_file("/tmp/des_com_velocity.txt");
    for (auto& v : des_com_velocity_log_) {
        des_com_velocity_file << v.transpose() << "\n";
    }

    std::ofstream des_zmp_position_file("/tmp/des_zmp_position.txt");
    for (auto& v : des_zmp_position_log_) {
        des_zmp_position_file << v.transpose() << "\n";
    }

    std::ofstream ef_zmp_position_file("/tmp/ef_zmp_position.txt");
    for (auto& v : ef_zmp_position_log_) {
        ef_zmp_position_file << v.transpose() << "\n";
    }

    std::ofstream p_lsole_sim_file("/tmp/p_lsole_sim.txt");
    for (auto& v : p_lsole_sim_log_) {
        p_lsole_sim_file << v.transpose() << "\n";
    }
    std::ofstream p_rsole_sim_file("/tmp/p_rsole_sim.txt");
    for (auto& v : p_rsole_sim_log_) {
        p_rsole_sim_file << v.transpose() << "\n";
    }

    std::ofstream v_lsole_sim_file("/tmp/v_lsole_sim.txt");
    for (auto& v : v_lsole_sim_log_) {
        v_lsole_sim_file << v.transpose() << "\n";
    }

    std::ofstream v_rsole_sim_file("/tmp/v_rsole_sim.txt");
    for (auto& v : v_rsole_sim_log_) {
        v_rsole_sim_file << v.transpose() << "\n";
    }

    std::ofstream p_lsole_fb_file("/tmp/p_lsole_fb.txt");
    for (auto& v : p_lsole_fb_log_) {
        p_lsole_fb_file << v.transpose() << "\n";
    }

    std::ofstream p_rsole_fb_file("/tmp/p_rsole_fb.txt");
    for (auto& v : p_rsole_fb_log_) {
        p_rsole_fb_file << v.transpose() << "\n";
    }

    std::ofstream v_lsole_fb_file("/tmp/v_lsole_fb.txt");
    for (auto& v : v_lsole_fb_log_) {
        v_lsole_fb_file << v.transpose() << "\n";
    }

    std::ofstream v_rsole_fb_file("/tmp/v_rsole_fb.txt");
    for (auto& v : v_rsole_fb_log_) {
        v_rsole_fb_file << v.transpose() << "\n";
    }

    std::ofstream p_lsole_des_file("/tmp/p_lsole_des.txt");
    for (auto& v : p_lsole_des_log_) {
        p_lsole_des_file << v.transpose() << "\n";
    }

    std::ofstream p_rsole_des_file("/tmp/p_rsole_des.txt");
    for (auto& v : p_rsole_des_log_) {
        p_rsole_des_file << v.transpose() << "\n";
    }

    std::ofstream v_lsole_des_file("/tmp/v_lsole_des.txt");
    for (auto& v : v_lsole_des_log_) {
        v_lsole_des_file << v.transpose() << "\n";
    }

    std::ofstream v_rsole_des_file("/tmp/v_rsole_des.txt");
    for (auto& v : v_rsole_des_log_) {
        v_rsole_des_file << v.transpose() << "\n";
    }

    std::ofstream estimated_force_lsole_file("/tmp/estimated_force_lsole.txt");
    for (auto& v : estimated_force_lsole_log_) {
        estimated_force_lsole_file << v.transpose() << "\n";
    }

    std::ofstream estimated_force_rsole_file("/tmp/estimated_force_rsole.txt");
    for (auto& v : estimated_force_rsole_log_) {
        estimated_force_rsole_file << v.transpose() << "\n";
    }

    std::ofstream angular_momentum_file("/tmp/angular_momentum.txt");
    for (auto& v : angular_momentum_log_) {
        angular_momentum_file << v.transpose() << "\n";
    }
    
    std::ofstream measured_joint_position_file("/tmp/measured_joint_position.txt");
    for (auto& v : measured_joint_position_log_) {
        measured_joint_position_file << v.transpose() << "\n";
    }

    std::ofstream measured_joint_velocity_file("/tmp/measured_joint_velocity.txt");
    for (auto& v : measured_joint_velocity_log_) {
        measured_joint_velocity_file << v.transpose() << "\n";
    }

    std::ofstream measured_imu_accelerometer_log_file("/tmp/measured_imu_accelerometer.txt");
    for (auto& v : measured_imu_accelerometer_log_) {
        measured_imu_accelerometer_log_file << v.transpose() << "\n";
    }

    std::ofstream measured_imu_angular_velocity_log_file("/tmp/measured_imu_angular_velocity.txt");
    for (auto& v : measured_imu_angular_velocity_log_) {
        measured_imu_angular_velocity_log_file << v.transpose() << "\n";
    }

    std::ofstream measured_imu_orientation_log_file("/tmp/measured_imu_orientation.txt");
    for (auto& v : measured_imu_orientation_log_) {
        measured_imu_orientation_log_file << v.transpose() << "\n";
    }

    std::ofstream ekf_base_position_file("/tmp/ekf_base_position.txt");
    for (auto& v : ekf_base_position_log_) {
        ekf_base_position_file << v.transpose() << "\n";
    }

    std::ofstream ekf_base_velocity_file("/tmp/ekf_base_velocity.txt");
    for (auto& v : ekf_base_velocity_log_) {
        ekf_base_velocity_file << v.transpose() << "\n";
    }

    std::ofstream ekf_base_orientation_file("/tmp/ekf_base_orientation.txt");
    for (auto& v : ekf_base_orientation_log_) {
        ekf_base_orientation_file << v.transpose() << "\n";
    }

    std::ofstream ekf_base_angular_velocity_file("/tmp/ekf_base_angular_velocity.txt");
    for (auto& v : ekf_base_angular_velocity_log_) {
        ekf_base_angular_velocity_file << v.transpose() << "\n";
    }

    std::ofstream ekf_joint_position_file("/tmp/ekf_joint_position.txt");
    for (auto& v : ekf_joint_position_log_) {
        ekf_joint_position_file << v.transpose() << "\n";
    }

    std::ofstream ekf_joint_velocity_file("/tmp/ekf_joint_velocity.txt");
    for (auto& v : ekf_joint_velocity_log_) {
        ekf_joint_velocity_file << v.transpose() << "\n";
    }

    std::ofstream sim_base_position_file("/tmp/sim_base_position.txt");
    for (auto& v : sim_base_position_log_) {
        sim_base_position_file << v.transpose() << "\n";
    }

    std::ofstream sim_base_velocity_file("/tmp/sim_base_velocity.txt");
    for (auto& v : sim_base_velocity_log_) {
        sim_base_velocity_file << v.transpose() << "\n";
    }

    std::ofstream sim_base_orientation_file("/tmp/sim_base_orientation.txt");
    for (auto& v : sim_base_orientation_log_) {
        sim_base_orientation_file << v.transpose() << "\n";
    }

    std::ofstream sim_joint_position_file("/tmp/sim_joint_position.txt");
    for (auto& v : sim_joint_position_log_) {
        sim_joint_position_file << v.transpose() << "\n";
    }

    std::ofstream sim_joint_velocity_file("/tmp/sim_joint_velocity.txt");
    for (auto& v : sim_joint_velocity_log_) {
        sim_joint_velocity_file << v.transpose() << "\n";
    }

    std::ofstream sim_base_angular_velocity_file("/tmp/sim_base_angular_velocity.txt");
    for (auto& v : sim_base_angular_velocity_log_) {
        sim_base_angular_velocity_file << v.transpose() << "\n";
    }

    std::ofstream execution_time_ekf_file("/tmp/execution_time_ekf.txt");
    for (auto& t : execution_time_ekf_log_) {
        execution_time_ekf_file << t << "\n";
    }

    std::ofstream execution_time_kf_file("/tmp/execution_time_kf.txt");
    for (auto& t : execution_time_kf_log_) {
        execution_time_kf_file << t << "\n";
    }

    std::ofstream execution_time_mpc_file("/tmp/execution_time_mpc.txt");
    for (auto& t : execution_time_mpc_log_) {
        execution_time_mpc_file << t << "\n";
    }

    std::ofstream execution_time_wbc_file("/tmp/execution_time_wbc.txt");
    for (auto& t : execution_time_wbc_log_) {
        execution_time_wbc_file << t << "\n";
    }

    std::ofstream execution_time_update_file("/tmp/execution_time_update.txt");
    for (auto& t : execution_time_update_log_) {
        execution_time_update_file << t << "\n";
    }

    // std::ofstream estimated_imu_accelerometer_file("/tmp/estimated_imu_accelerometer.txt");
    // for (auto& v : estimated_imu_accelerometer_log_) {
    //     estimated_imu_accelerometer_file << v.transpose() << "\n";
    // }

    // std::ofstream estimated_imu_angular_velocity_file("/tmp/estimated_imu_angular_velocity.txt");
    // for (auto& v : estimated_imu_angular_velocity_log_) {
    //     estimated_imu_angular_velocity_file << v.transpose() << "\n";
    // }

    // std::ofstream estimated_imu_orientation_file("/tmp/estimated_imu_orientation.txt");
    // for (auto& v : estimated_imu_orientation_log_) {
    //     estimated_imu_orientation_file << v.transpose() << "\n";
    // }

    std::ofstream input_torque_file("/tmp/input_torque.txt");
    for (auto& v : input_torque_log_) {
        input_torque_file << v.transpose() << "\n";
    }


}


int64_t
WalkingManager::get_controller_frequency() const {
  return controller_frequency_;
}

const pinocchio::Model&
WalkingManager::getRobotModel() const {
    return robot_model;
}

void
WalkingManager::setDesiredStepLengthX(double desired_step_length_x) {
    walking_data_.setStepLengthX(std::clamp(desired_step_length_x, 0.0, 0.1));
}

void
WalkingManager::setDesiredStepCount(int desired_step_count) {
    walking_data_.setNumberOfSteps(std::max(0, desired_step_count));
}

Eigen::MatrixXd
WalkingManager::pseudoinverse(const Eigen::MatrixXd& J, double damp) const {
  auto J_T = J.transpose();
  auto Id = Eigen::MatrixXd::Identity(J.cols(), J.cols());
  return (J_T * J + damp * Id).inverse() * J_T;
}

void
WalkingManager::swingFootTrajectory(
    pinocchio::SE3& swing_foot_pose,
    pinocchio::Motion& swing_foot_velocity,
    pinocchio::Motion& swing_foot_acceleration
) const {
  // NOTE: assuming there are at least two elements in the footstep plan.
  // NOTE: assuming roll and pitch are always zero for the swing foot.
  double t = 0.001 * static_cast<double>(t_msec_ - walking_data_.t0 + controller_timestep_msec_);
  double swing_duration = 0.001 * static_cast<double>(walking_data_.footstep_plan.front().getDuration());
  labrob::QuinticPolynomialTimingLaw timing_law(swing_duration);
  double s = timing_law.eval(t);
  double s_dot = timing_law.eval_dt(t);
  double s_ddot = timing_law.eval_dt_dt(t);

  const auto& feet_placement = walking_data_.footstep_plan[0].getFeetPlacement();
  const auto& target_feet_placement = walking_data_.footstep_plan[1].getFeetPlacement();
  const auto& support_foot_identity = feet_placement.getSupportFoot();
  const auto& support_foot_configuration = feet_placement.getSupportFootConfiguration();
  const auto& starting_swing_foot_configuration = feet_placement.getSwingFootConfiguration();
  const auto& target_swing_foot_configuration =
      (support_foot_identity == labrob::Foot::LEFT ?
             target_feet_placement.getRightFootConfiguration() :
             target_feet_placement.getLeftFootConfiguration()
      );
  const auto& p0 = starting_swing_foot_configuration.p;
  const auto& R0 = starting_swing_foot_configuration.R;
  const auto& pf = target_swing_foot_configuration.p;
  const auto& Rf = target_swing_foot_configuration.R;
  double yaw0 = std::atan2(R0(1, 0), R0(0, 0));
  double yawf = std::atan2(Rf(1, 0), Rf(0, 0));

  pinocchio::SE3 desired_swing_foot_pose;
  desired_swing_foot_pose.translation().x() = p0.x() + (pf.x() - p0.x()) * s;
  desired_swing_foot_pose.translation().y() = p0.y() + (pf.y() - p0.y()) * s;
  double zs = support_foot_configuration.p.z();
  double z0 = p0.z();
  double zf = pf.z();
  double h_z = walking_data_.footstep_plan[0].getSwingFootTrajectoryHeight();
  double a = 2.0 * z0 - 4.0 * h_z + 2.0 * zf - 4.0 * zs;
  double b = 4.0 * h_z - 3.0 * z0 - zf + 4.0 * zs;
  double c = z0;
  desired_swing_foot_pose.translation().z() = a * s * s + b * s + c;
  double desired_swing_foot_yaw = yaw0 + angle_difference(yawf, yaw0) * s;
  desired_swing_foot_pose.rotation() = Rz(desired_swing_foot_yaw);

  pinocchio::Motion desired_swing_foot_velocity(
      Eigen::Vector3d(pf.x() - p0.x(), pf.y() - p0.y(), 2 * a * s + b) * s_dot,
      Eigen::Vector3d(0.0, 0.0, angle_difference(yawf, yaw0)) * s_dot
  );

  pinocchio::Motion desired_swing_foot_acceleration(
      Eigen::Vector3d((pf.x() - p0.x()) * s_ddot, (pf.y() - p0.y()) * s_ddot, 2 * a * s_dot * s_dot + (2 * a * s + b) * s_ddot),
      Eigen::Vector3d(0.0, 0.0, angle_difference(yawf, yaw0)) * s_ddot
  );

  swing_foot_pose = desired_swing_foot_pose;
  swing_foot_velocity = desired_swing_foot_velocity;
  swing_foot_acceleration = desired_swing_foot_acceleration;
}

} // end namespace labrob
