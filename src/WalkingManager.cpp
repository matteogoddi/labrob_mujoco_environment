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
#include <pinocchio/spatial/explog.hpp>

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

//INIT FUNCTION START

bool
WalkingManager::init(const labrob::RobotState& initial_robot_state,
                     std::map<std::string, double> &armatures) {

    //PRE-ALLOCATIONS FOR LOGS

    int64_t max_steps = 50000;

    fb_com_position_log_.reserve(max_steps);
    fb_com_velocity_log_.reserve(max_steps);
    fb_zmp_position_log_.reserve(max_steps);

    kf_com_position_log_.reserve(max_steps);
    kf_com_velocity_log_.reserve(max_steps);
    kf_zmp_position_log_.reserve(max_steps);

    des_com_position_log_.reserve(max_steps);
    des_com_velocity_log_.reserve(max_steps);
    des_zmp_position_log_.reserve(max_steps);
    des_com_acceleration_log_.reserve(max_steps);

    ef_zmp_position_log_.reserve(max_steps);

    p_lsole_fb_log_.reserve(max_steps);
    p_rsole_fb_log_.reserve(max_steps);
    v_lsole_fb_log_.reserve(max_steps);
    v_rsole_fb_log_.reserve(max_steps);
    p_lsole_des_log_.reserve(max_steps);
    p_rsole_des_log_.reserve(max_steps);
    v_lsole_des_log_.reserve(max_steps);
    v_rsole_des_log_.reserve(max_steps);

    fb_lsole_orientation_log_.reserve(max_steps);
    fb_rsole_orientation_log_.reserve(max_steps);
    des_lsole_orientation_log_.reserve(max_steps);
    des_rsole_orientation_log_.reserve(max_steps);

    estimated_force_lsole_log_.reserve(max_steps);
    estimated_force_rsole_log_.reserve(max_steps);
    input_torque_log_.reserve(max_steps);
    wbc_accelerations_log_.reserve(max_steps);

    angular_momentum_log_.reserve(max_steps);
    mpc_predictions_log_.reserve(max_steps);

    ekf_base_position_log_.reserve(max_steps);
    ekf_base_velocity_log_.reserve(max_steps);
    ekf_base_orientation_log_.reserve(max_steps);
    ekf_base_orientation_rpy_log_.reserve(max_steps);
    ekf_base_angular_velocity_log_.reserve(max_steps);
    ekf_imu_orientation_log_.reserve(max_steps);
    ekf_imu_orientation_rpy_log_.reserve(max_steps);
    ekf_imu_angular_velocity_log_.reserve(max_steps);
    ekf_joint_position_log_.reserve(max_steps);
    ekf_joint_velocity_log_.reserve(max_steps);

    odometry_base_position_log_.reserve(max_steps);
    odometry_base_velocity_log_.reserve(max_steps);
    odometry_imu_orientation_log_.reserve(max_steps);
    odometry_imu_orientation_rpy_log_.reserve(max_steps);
    measured_imu_orientation_log_.reserve(max_steps);
    measured_imu_orientation_rpy_log_.reserve(max_steps);
    measured_imu_angular_velocity_log_.reserve(max_steps);
    measured_imu_accelerometer_log_.reserve(max_steps);
    measured_joint_position_log_.reserve(max_steps);
    measured_joint_velocity_log_.reserve(max_steps);

    execution_time_wbc_log_.reserve(max_steps);
    execution_time_mpc_log_.reserve(max_steps);
    execution_time_ekf_log_.reserve(max_steps);
    execution_time_kf_log_.reserve(max_steps);

    mpc_pred_com_pos_log_.reserve(3*max_steps);
    mpc_pred_com_vel_log_.reserve(3*max_steps);
    mpc_pred_zmp_pos_log_.reserve(3*max_steps);

    torso_orientation_log_.reserve(max_steps);
    torso_angular_velocity_log_.reserve(max_steps);
    des_torso_orientation_log_.reserve(max_steps);
    des_torso_angular_velocity_log_.reserve(max_steps);

    mpc_zmp_velocity_log_.reserve(max_steps);
    con_zmp_velocity_log_.reserve(max_steps);

    // READING ROBOT DESCRIPTION (URDF) AND BUILDING PINOCCHIO MODEL

    std::string robot_description_filename = "../robot/g1/g1_description/g1_29dof_rev_1_0.urdf";

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

    mass = pinocchio::computeTotalMass(robot_model);
    njnt = robot_model.nv - 6;

    // Init desired lsole and rsole poses:
    auto q_init = robot_state_to_pinocchio_joint_configuration(
        robot_model,
        initial_robot_state
    );

    // INIT ROBOT STATE, DATA AND PINOCCHIO QUANTITIES

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
    integrated_state_vel = Eigen::VectorXd::Zero(6 + njnt);

    fb_robot_data = pinocchio::Data(robot_model);
    predicted_robot_data = pinocchio::Data(robot_model);
    estimated_robot_data = pinocchio::Data(robot_model);

    fb_robot_state = initial_robot_state;

    fixed_com_pos = Eigen::Vector3d::Zero();
    fixed_com_vel = Eigen::Vector3d::Zero();
    fixed_zmp_pos = Eigen::Vector3d::Zero();

    // INIT FEEDBACK, PREDICTED AND ESTIMATED ROBOT DATA AND COMPUTING PINOCCHIO QUANTITIES

    pinocchio::forwardKinematics(robot_model, fb_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, fb_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, fb_robot_data, q_init);

    pinocchio::forwardKinematics(robot_model, predicted_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, predicted_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, predicted_robot_data, q_init);

    pinocchio::forwardKinematics(robot_model, estimated_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, estimated_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, estimated_robot_data, q_init);

    BASE_IDX = 0;
    IMU_ROTVEC_IDX = BASE_IDX + 3;
    JOINTS_IDX = IMU_ROTVEC_IDX + 3;

    // ADDITIONAL OUTPUTS FOR EKF

    // BASE_VEL_IDX = JOINTS_IDX + njnt;
    // JOINTS_VEL_IDX = BASE_VEL_IDX + 3;
    // LEFT_FOOT_VEL_IDX = JOINTS_VEL_IDX + njnt;
    // RIGHT_FOOT_VEL_IDX = LEFT_FOOT_VEL_IDX + 6;

    // STATE COVARIANCE MATRIX

    n_ekf_output = JOINTS_IDX + njnt;

    P_ = Eigen::MatrixXd::Identity(2 * (njnt + 6), 2 * (njnt + 6)) * 1e-6;
    P_.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1;
    P_.block(njnt + 6, njnt + 6, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-3;
    P_.block(3,3,3,3) = Eigen::MatrixXd::Identity(3, 3) * 1;
    P_.block(njnt + 6 + 3, njnt + 6 + 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1;

    // PROCESS NOISE COVARIANCE MATRIX

    Q = Eigen::MatrixXd::Zero(2*(njnt+6), 2*(njnt+6));

    //base position
    Q.block<3,3>(0,0) = 1e-6 * Eigen::Matrix3d::Identity();

    //imu orientation
    Q.block<3,3>(3,3) = 1e-6 * Eigen::Matrix3d::Identity();

    //joint position
    Q.block(6, 6, njnt, njnt) = 1e-6 * Eigen::MatrixXd::Identity(njnt, njnt);

    //base linear and angular velocities
    Q.block(njnt+6, njnt+6, 6, 6) = 1e-4 * Eigen::MatrixXd::Identity(6,6);

    //joint velocities
    Q.block(2*6+njnt, 2*6+njnt, njnt, njnt) = 1e-4 * Eigen::MatrixXd::Identity(njnt,njnt);

    // OUTPUT NOISE COVARIANCE MATRIX

    R = Eigen::MatrixXd::Zero(n_ekf_output, n_ekf_output);

    //base position
    R.block(BASE_IDX, BASE_IDX, 3, 3) = 1e-5 * Eigen::Matrix3d::Identity();

    //imu orientation
    R.block(IMU_ROTVEC_IDX, IMU_ROTVEC_IDX, 3, 3) = 1e-5 * Eigen::MatrixXd::Identity(3, 3);

    //joint position
    R.block(JOINTS_IDX, JOINTS_IDX, njnt, njnt) = 1e-5 * Eigen::MatrixXd::Identity(njnt, njnt);

    //ADDITIONAL OUTPUTS FOR EKF

    //base linear velocity
    // R.block(BASE_VEL_IDX, BASE_VEL_IDX, 3, 3) = 1e-4 * Eigen::MatrixXd::Identity(3, 3);

    //imu angular velocity
    // R.block(3 + njnt + 3, 3 + njnt + 3, 3, 3) = 1e-3 * Eigen::MatrixXd::Identity(3, 3);

    //joint velocity
    // R.block(JOINTS_VEL_IDX, JOINTS_VEL_IDX, njnt, njnt) = 25*1e-1 * Eigen::MatrixXd::Identity(njnt, njnt);

    //feet velocity
    // R.block(LEFT_FOOT_VEL_IDX, LEFT_FOOT_VEL_IDX, 12, 12) = 1e-1 * Eigen::MatrixXd::Identity(12, 12);

    // INIT ESTIMATE STATE AND OUTPUT

    x_estimate = Eigen::VectorXd::Zero(2 * (njnt + 6));
    x_estimate.head(3) = q_init.head(3);
    x_estimate.segment(3, 3) = rotVecFromQuaternion(Eigen::Quaterniond(
        q_init[6], q_init[3], q_init[4], q_init[5]
    ));
    x_estimate.segment(3 + 3, njnt) = q_init.tail(njnt);
    y_pred = Eigen::VectorXd::Zero(n_ekf_output);
    y_actual = Eigen::VectorXd::Zero(n_ekf_output);
    y_estimate = Eigen::VectorXd::Zero(n_ekf_output);


    // GET INDICES OF INTEREST AND ARMATURES

    lsole_idx_ = robot_model.getFrameId("left_foot_link");
    rsole_idx_ = robot_model.getFrameId("right_foot_link");
    torso_idx_ = robot_model.getFrameId("torso_link");
    pelvis_idx_ = robot_model.getFrameId("pelvis");
    imu_idx_ = robot_model.getFrameId("imu_in_pelvis");
    const auto& T_lsole_init = sim_robot_data.oMf[lsole_idx_];
    const auto& T_rsole_init = sim_robot_data.oMf[rsole_idx_];

    M_armature_ = Eigen::VectorXd::Zero(njnt);
    for (pinocchio::JointIndex joint_id = 0;
        joint_id < (pinocchio::JointIndex) njnt;
        ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        M_armature_(joint_id) = armatures[joint_name];
    }

    // SET JOINT DES AS INITIAL POSE

    q_jnt_des_ = q_init.tail(njnt);

    // CONTROLLER FREQUENCY

    // TODO: init using node handle.
    controller_frequency_ = 500;
    controller_timestep_msec_ = 1000 / controller_frequency_;

    // WALKING DATA INIT WITH INITIAL FEET POSES

    walking_data_.initializeWalkingData(
        controller_timestep_msec_,
        labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
        labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation())
    );

    if(!useRobot && false){
        walking_data_.addSteps(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
            0.0
        );
    };

    parameters_log_.push_back(startTimeWBCCL);

    // Save and read again footstep plan to double check it's working:
    //std::string footstep_plan_path = "/tmp/ditch-footstep-plan-argos.txt";
    //labrob::saveFootstepPlan(walking_data_.footstep_plan, footstep_plan_path);
    //labrob::readFootstepPlan(footstep_plan_path, walking_data_.footstep_plan);
    //labrob::readArgosFootstepPlan(footstep_plan_path, walking_data_.footstep_plan);

    // INIT LIP MODEL, IS-MPC, WHOLE-BODY CONTROLLER AND DISCRETE LIP DYNAMICS

    Eigen::Vector3d p_CoM_sim = sim_robot_data.com[0];
    double com_target_height = p_CoM_sim.z() - T_lsole_init.translation().z();
    eta2 = 9.81 / com_target_height;
    Eigen::Vector3d p_ZMP_sim = p_CoM_sim - Eigen::Vector3d(0.0, 0.0, com_target_height);
    kf_LipState = labrob::LIPState(
        p_CoM_sim,
        Eigen::Vector3d::Zero(),
        p_ZMP_sim
    );
    des_LipState = kf_LipState;
    ismpc_ptr_ = std::make_unique<labrob::ISMPC>(
        mpc_prediction_horizon_msec,
        mpc_timestep_msec,
        std::sqrt(eta2),
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

    discrete_lip_dynamics_ptr_ = std::make_unique<labrob::DiscreteLIPDynamics>(
        std::sqrt(eta2),
        0.001 * controller_timestep_msec_
    );

    discrete_lip_dynamics_ptr_mpc_ = std::make_unique<labrob::DiscreteLIPDynamics>(
        std::sqrt(eta2),
        0.001 * mpc_timestep_msec
    );

    joint_kf_ptr_ = std::make_unique<labrob::JointKF>(
        0.001 * controller_timestep_msec_,
        njnt
    );

    base_ekf_ptr_ = std::make_unique<labrob::BaseEKF>(
        robot_model,
        q_init,
        0.001 * controller_timestep_msec_
    );

    com_kf_ptr_ = std::make_unique<labrob::CoMKF>(
        0.001 * controller_timestep_msec_,
        std::sqrt(eta2)
    );

    // INIT KALMAN GAIN FOR EKF

    Kalman_Gain = Eigen::MatrixXd::Zero(2 * (njnt + 6), n_ekf_output);

    // GET JACOBIANS FOR EKF OUTPUTS

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

    // std::unique_ptr<RightInvariantEKF> ri_ekf_ptr_;

    // Inizializzazione (in init o costruttore del WalkingManager)
    NoiseParams noise;
    noise.gyro_noise    = 0.005;
    noise.accel_noise   = 0.05;
    noise.contact_noise = 0.01;
    noise.gyro_bias_rw  = 0.0001;
    noise.accel_bias_rw = 0.001;
    noise.encoder_noise = 0.005;

    std::array<RightInvariantEKF::FootConfig, 2> feet = {{
        {"left_foot_link",  0},
        {"right_foot_link", 1}
    }};
    ri_ekf_ptr_ = std::make_unique<RightInvariantEKF>(
        robot_model, q_init, 0.001 * controller_timestep_msec_, feet, noise);

    std::array<DiligentKio::FootConfig, 2> feet_kio = {{
        {"left_foot_link",  0},
        {"right_foot_link", 1}
    }};
    diligent_kio_ptr_ = std::make_unique<DiligentKio>(
        robot_model, q_init, 0.001 * controller_timestep_msec_, feet_kio, noise);

    return true;
}

// PROPAGATE STATE FUNCTION START TO COMPUTE NUMERICAL A MATRIX FOR EKF

Eigen::VectorXd WalkingManager::propagateState(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& q_ddot,
    double dt)
{
    Eigen::VectorXd x_next = Eigen::VectorXd::Zero(x.size());

    // --- Build Pinocchio configuration
    Eigen::VectorXd q(njnt + 7);
    q.head(3) = x.head(3);
    q.segment(3,4) = Eigen::Vector4d(
        quaternionFromRotVec(x.segment<3>(3)).x(),
        quaternionFromRotVec(x.segment<3>(3)).y(),
        quaternionFromRotVec(x.segment<3>(3)).z(),
        quaternionFromRotVec(x.segment<3>(3)).w()
    );
    q.tail(njnt) = x.segment(6, njnt);

    Eigen::VectorXd v = x.segment(njnt + 6, njnt + 6);

    // --- Integrate
    Eigen::VectorXd q_next =
        pinocchio::integrate(robot_model, q, v * dt);

    // --- Fill next state
    x_next.head(3) = q_next.head(3);
    x_next.segment<3>(3) =
        rotVecFromQuaternion(Eigen::Quaterniond(
            q_next[6], q_next[3], q_next[4], q_next[5]
        ));
    x_next.segment(6, njnt) = q_next.tail(njnt);

    x_next.tail(njnt + 6) = v + q_ddot * dt;

    return x_next;
}

// NUMERICAL A COMPUTATION FUNCTION START

Eigen::MatrixXd WalkingManager::computeNumericalA(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& q_ddot,
    double dt)
{
    const int nx = x.size();
    Eigen::MatrixXd A(nx, nx);
    A.setZero();

    Eigen::VectorXd x0 = propagateState(x, q_ddot, dt);

    const double eps = 1e-6;

    for (int i = 0; i < nx; ++i)
    {
        Eigen::VectorXd x_plus = x;
        Eigen::VectorXd x_minus = x;

        // --- configuration perturbation
        if (i < njnt + 6)
        {
            Eigen::VectorXd dq = Eigen::VectorXd::Zero(njnt + 6);
            dq(i) = eps;

            Eigen::VectorXd q(njnt + 7);
            q.head(3) = x.head(3);
            q.segment(3,4) = Eigen::Vector4d(
                quaternionFromRotVec(x.segment<3>(3)).x(),
                quaternionFromRotVec(x.segment<3>(3)).y(),
                quaternionFromRotVec(x.segment<3>(3)).z(),
                quaternionFromRotVec(x.segment<3>(3)).w()
            );
            q.tail(njnt) = x.segment(6, njnt);

            Eigen::VectorXd q_p = pinocchio::integrate(robot_model, q, dq);
            Eigen::VectorXd q_m = pinocchio::integrate(robot_model, q, -dq);

            x_plus.head(3) = q_p.head(3);
            x_plus.segment<3>(3) =
                rotVecFromQuaternion(Eigen::Quaterniond(
                    q_p[6], q_p[3], q_p[4], q_p[5]
                ));
            x_plus.segment(6, njnt) = q_p.tail(njnt);

            x_minus.head(3) = q_m.head(3);
            x_minus.segment<3>(3) =
                rotVecFromQuaternion(Eigen::Quaterniond(
                    q_m[6], q_m[3], q_m[4], q_m[5]
                ));
            x_minus.segment(6, njnt) = q_m.tail(njnt);
        }
        else
        {
            x_plus(i) += eps;
            x_minus(i) -= eps;
        }

        Eigen::VectorXd f_plus  = propagateState(x_plus,  q_ddot, dt);
        Eigen::VectorXd f_minus = propagateState(x_minus, q_ddot, dt);

        A.col(i) = (f_plus - f_minus) / (2.0 * eps);
    }

    return A;
}

// EKF FUNCTION START

RobotState WalkingManager::updateEKF(Eigen::VectorXd y_actual) {

    //USE DIFFERENT COVARIANCES FOR FEET IN AIR AND ON THE GROUND IF USE FEET POS/VEL IN EKF OUTPUTS


    // double left_support_check = 1.0;
    // double right_support_check = 1.0;
    // if (walking_data_.getWalkingState() == WalkingState::SingleSupport){
    //     if (walking_data_.footstep_plan.frontrobot_model().getFeetPlacement().getSupportFoot() == Foot::LEFT){
    //         R.block(RIGHT_FOOT_VEL_IDX, RIGHT_FOOT_VEL_IDX, 6, 6) = 10000 * Eigen::MatrixXd::Identity(6,6);
    //         R.block(LEFT_FOOT_VEL_IDX, LEFT_FOOT_VEL_IDX, 6, 6) = 1e-5 * Eigen::MatrixXd::Identity(6,6);
    //     }
    //     if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT){
    //         R.block(LEFT_FOOT_VEL_IDX, LEFT_FOOT_VEL_IDX, 6, 6) = 10000 * Eigen::MatrixXd::Identity(6,6);
    //         R.block(RIGHT_FOOT_VEL_IDX, RIGHT_FOOT_VEL_IDX, 6, 6) = 1e-5 * Eigen::MatrixXd::Identity(6,6);
    //     }
    // }

    // INIT X PRED

    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(2 * (njnt + 6));
    // x_pred.segment(njnt + 6, njnt + 6) = x_estimate.segment(njnt + 6, njnt + 6) + (whole_body_controller_ptr_->get_q_ddot()) * controller_timestep_msec_ * 0.001;
    // x_pred.head(njnt + 6) = x_estimate.head(njnt + 6) + x_estimate.segment(njnt + 6, njnt + 6) * 0.001 * controller_timestep_msec_;

    // INTEGRATE STATE USING PINOCCHIO::INTEGRATE

    Eigen::VectorXd q_current = Eigen::VectorXd::Zero(njnt + 7 + njnt + 6);
    q_current.head(3) = x_estimate.head(3);
    q_current.segment(3, 4) = Eigen::Vector4d(
        quaternionFromRotVec(x_estimate.segment(3, 3)).x(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).y(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).z(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).w()
    );
    q_current.segment(7, njnt) = x_estimate.segment(3 + 3, njnt);
    q_current.tail(6 + njnt) = x_estimate.tail(njnt + 6);
    Eigen::VectorXd q_next = pinocchio::integrate(
        robot_model,
        q_current.head(njnt + 7),
        q_current.tail(njnt + 6) * 0.001 * controller_timestep_msec_
    );
    x_pred.head(3) = q_next.head(3);
    x_pred.segment(3, 3) = rotVecFromQuaternion(Eigen::Quaterniond(
        q_next[6], q_next[3], q_next[4], q_next[5]
    ));
    x_pred.segment(3 + 3, njnt) = q_next.segment(7, njnt);
    x_pred.tail(njnt + 6) = x_estimate.tail(njnt + 6) + whole_body_controller_ptr_->get_q_ddot() * controller_timestep_msec_ * 0.001;

    // TAKE ONLY POSITIONS TO CONVERT FROM ROTVEC TO QUATERNION

    Eigen::VectorXd q_pred = Eigen::VectorXd::Zero(njnt + 7);
    q_pred.head(3) = x_pred.head(3);
    q_pred.segment(3, 4) = Eigen::Vector4d(
        quaternionFromRotVec(x_pred.segment(3, 3)).x(),
        quaternionFromRotVec(x_pred.segment(3, 3)).y(),
        quaternionFromRotVec(x_pred.segment(3, 3)).z(),
        quaternionFromRotVec(x_pred.segment(3, 3)).w()
    );
    q_pred.tail(njnt) = x_pred.segment(3 + 3, njnt);

    // COMPUTE PINOCCHIO QUANTITIES FOR PREDICTION AND JACOBIANS

    pinocchio::forwardKinematics(robot_model, predicted_robot_data, q_pred);
    pinocchio::framesForwardKinematics(robot_model, predicted_robot_data, q_pred);
    pinocchio::updateFramePlacements(robot_model, predicted_robot_data);

    // INIT Y PRED

    Eigen::Quaterniond pred_imu_orientation = Eigen::Quaterniond(
        predicted_robot_data.oMf[imu_idx_].rotation()
    );
    y_pred.segment(BASE_IDX, 3) = x_pred.head(3);
    y_pred.segment(IMU_ROTVEC_IDX, 3) = rotVecFromQuaternion(pred_imu_orientation);
    y_pred.segment(JOINTS_IDX, njnt) = q_pred.tail(njnt);

    // ADDITIONAL OUTPUTS FOR EKF

    // y_pred.segment(BASE_VEL_IDX, 3) = x_pred.segment(6 + njnt, 3);
    // y_pred.segment(3 + njnt + 3, 3) = J_imu_est.bottomRows(3) * x_pred.tail(njnt + 6);
    // y_pred.segment(LEFT_FOOT_VEL_IDX, 6) = J_left_foot_pred * x_pred.segment(njnt + 6, njnt + 6);
    // y_pred.segment(RIGHT_FOOT_VEL_IDX, 6) = J_right_foot_pred * x_pred.segment(njnt + 6, njnt + 6);
    // y_pred.segment(JOINTS_VEL_IDX, njnt) = x_pred.segment(6 + 6 + njnt, njnt);


    // C MATRIX:

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_ekf_output, 2 * (njnt + 6));
    C.block(BASE_IDX, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    C.block(IMU_ROTVEC_IDX, 0, 3, njnt + 6) = J_imu_est.bottomRows(3);
    C.block(JOINTS_IDX, 6, njnt, njnt) = Eigen::MatrixXd::Identity(njnt, njnt);

    // ADDITIONAL OUTPUTS FOR EKF

    // C.block(BASE_VEL_IDX, njnt + 6, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    // C.block(3 + njnt + 3, njnt + 6, 3, njnt + 6) = J_imu_est.bottomRows(3);
    // C.block(JOINTS_VEL_IDX, njnt + 6 + 6, njnt, njnt) = Eigen::MatrixXd::Identity(njnt, njnt);
    // C.block(LEFT_FOOT_VEL_IDX, njnt + 6, 6, njnt + 6) = J_left_foot_est;
    // C.block(RIGHT_FOOT_VEL_IDX, njnt + 6, 6, njnt + 6) = J_right_foot_est;

    // A MATRIX:

    // Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2 * (njnt + 6), 2 * (njnt + 6));
    // A.block(0, njnt + 6, njnt + 6, njnt + 6) = controller_timestep_msec_ * 0.001 * Eigen::MatrixXd::Identity(njnt + 6, njnt + 6);

    // UNCOMMENT ABOVE TO USE APPROXIMATED A. UNCOMMENT BELOW TO USE NUMERICAL A

    Eigen::MatrixXd A = computeNumericalA(x_estimate,
                      whole_body_controller_ptr_->get_q_ddot(),
                      controller_timestep_msec_ * 0.001);

    //PREDICTION COVARIANCE (lambda) AND KALMAN GAIN

    Eigen::MatrixXd Lambda_ = A * P_ * A.transpose() + Q;
    // Kalman_Gain = Lambda_ * C.transpose() * (C * Lambda_ * C.transpose() + R).inverse();

    // ALTERNATIVE WAYS TO COMPUTE KALMAN GAIN BELOW (USE IF INVERSION TOO SLOW)

    // Eigen::LLT<Eigen::MatrixXd> llt(C * Lambda_ * C.transpose() + R);
    // Eigen::MatrixXd MatInv = llt.solve(Eigen::MatrixXd::Identity(n_ekf_output, n_ekf_output));
    // Kalman_Gain = Lambda_ * C.transpose() * MatInv;

    Eigen::MatrixXd S = C * Lambda_ * C.transpose() + R;   // innovation covariance
    Kalman_Gain = Lambda_ * C.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));

    // Eigen::MatrixXd S = C * Lambda_ * C.transpose() + R;
    // Kalman_Gain = Lambda_ * C.transpose();
    // Kalman_Gain = S.ldlt().solve(Kalman_Gain.transpose()).transpose();

    // PROCESS COVARIANCE MATRIX UPDATE

    P_ = (Eigen::MatrixXd::Identity(2 * (njnt + 6), 2 * (njnt + 6)) - Kalman_Gain * C) * Lambda_;

    // y_actual = actual_output;

    // STATE ESTIMATE

    x_estimate = x_pred + Kalman_Gain * (y_actual - y_pred);

    // TAKE ONLY POSITIONS TO CONVERT FROM ROTVEC TO QUATERNION

    Eigen::VectorXd q_estimate = Eigen::VectorXd::Zero(njnt + 7);
    q_estimate.head(3) = x_estimate.head(3);
    q_estimate.segment(3, 4) = Eigen::Vector4d(
        quaternionFromRotVec(x_estimate.segment(3, 3)).x(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).y(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).z(),
        quaternionFromRotVec(x_estimate.segment(3, 3)).w()
    );
    q_estimate.tail(njnt) = x_estimate.segment(3 + 3, njnt);

    // COMPUTE PINOCCHIO QUANTITIES FOR ESTIMATE AND JACOBIANS

    pinocchio::forwardKinematics(robot_model, estimated_robot_data, q_estimate);
    pinocchio::jacobianCenterOfMass(robot_model, estimated_robot_data, q_estimate);
    pinocchio::computeJointJacobians(robot_model, estimated_robot_data, q_estimate);
    pinocchio::framesForwardKinematics(robot_model, estimated_robot_data, q_estimate);

    J_imu_est = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        estimated_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_est
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

    // FILL CURRENT STATE FOR ACTUAL RETURN

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

// UPDATE FUNCTION START

void
WalkingManager::update(
    const labrob::RobotState& sim_robot_state,
    labrob::JointCommand& joint_command
) {

    auto start_update = std::chrono::high_resolution_clock::now();

    // SET FORCE ESTIMATION

    Eigen::Vector3d left_foot_force = estimated_force.head(3);
    Eigen::Vector3d right_foot_force = estimated_force.tail(3);
    Eigen::Vector3d total_force = left_foot_force + right_foot_force;

    // UPDATE FORWARD KINEMATICS, LIP AND PINOCCHIO QUANTITIES

    auto q = sim_robot_state.get_pinocchio_joint_configuration(robot_model);
    auto qdot = sim_robot_state.get_pinocchio_joint_velocity(robot_model);

    pinocchio::forwardKinematics(robot_model, sim_robot_data, q);

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

    const auto& T_pelvis_sim = sim_robot_data.oMf[pelvis_idx_];
    auto pelvis_orientation_sim = T_pelvis_sim.rotation();
    Eigen::MatrixXd J_pelvis_sim = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        sim_robot_data,
        pelvis_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_pelvis_sim
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

    Eigen::MatrixXd J_imu_sim = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        sim_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_sim
    );

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

    // USE SIM VALUES IF NOT USING ROBOT

    if (!useRobot){
        for (int i = 0; i < njnt; ++i) {
            std::string joint_name = robot_model.names[i + 2];
            measured_joint_position(i) = sim_robot_state.joint_state[joint_name].pos;
            measured_joint_velocity(i) = sim_robot_state.joint_state[joint_name].vel;
        }
        measured_imu_quaternion = Eigen::Vector4d( Eigen::Quaterniond(sim_robot_data.oMf[imu_idx_].rotation()).w(),
                                    Eigen::Quaterniond(sim_robot_data.oMf[imu_idx_].rotation()).x(),
                                    Eigen::Quaterniond(sim_robot_data.oMf[imu_idx_].rotation()).y(),
                                    Eigen::Quaterniond(sim_robot_data.oMf[imu_idx_].rotation()).z()
        );
        measured_imu_rpy = labrob::rpyFromQuaternion(Eigen::Quaterniond(measured_imu_quaternion(0), measured_imu_quaternion(1), measured_imu_quaternion(2), measured_imu_quaternion(3)));
        // measured_imu_angular_velocity = J_imu_sim.bottomRows(3) * qdot;
        // measured_imu_accelerometer = Eigen::Vector3d(0, 0, 0);
        odometry_base_position = Eigen::Vector3d(sim_robot_state.position.x(), sim_robot_state.position.y(), sim_robot_state.position.z());
        odometry_base_velocity = Eigen::Vector3d(sim_robot_state.linear_velocity.x(), sim_robot_state.linear_velocity.y(), sim_robot_state.linear_velocity.z());
        odometry_imu_quaternion = measured_imu_quaternion;
        odometry_imu_rpy = measured_imu_rpy;
    }

    // FILL ACTUAL OUTPUT VECTOR FOR EKF

    Eigen::VectorXd actual_output = Eigen::VectorXd::Zero(n_ekf_output);
    actual_output.segment(BASE_IDX, 3) = odometry_base_position;
    actual_output.segment(IMU_ROTVEC_IDX, 3) = rotVecFromQuaternion(Eigen::Quaterniond(
        odometry_imu_quaternion[0], odometry_imu_quaternion[1], odometry_imu_quaternion[2], odometry_imu_quaternion[3]
    ));
    // actual_output.segment(IMU_ROTVEC_IDX, 3) = rotVecFromQuaternion(Eigen::Quaterniond(
    //     sim_robot_data.oMf[imu_idx_].rotation()));
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        actual_output(JOINTS_IDX + joint_id) = measured_joint_position(joint_id);
        // actual_output(JOINTS_VEL_IDX + joint_id) = measured_joint_velocity(joint_id);
    }

    // ADDITIONAL OUTPUTS FOR EKF

    // actual_output.segment(BASE_VEL_IDX, 3) = odometry_base_velocity;
    // actual_output.segment(LEFT_FOOT_VEL_IDX, 6) = Eigen::VectorXd::Zero(6);
    // actual_output.segment(RIGHT_FOOT_VEL_IDX, 6) = Eigen::VectorXd::Zero(6);

    ////////////////////////
    // EKF FUNCTION CALL (IN DIFFERENT THREADS)
    ////////////////////////

    bool left_support_check = true;
    bool right_support_check = true;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport){
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT){
            right_support_check = false;
        }
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT){
            left_support_check = false;
        }
    }

    auto start_ekf = std::chrono::high_resolution_clock::now();
    Eigen::VectorXd input1 = Eigen::VectorXd::Zero(measured_joint_position.size() + measured_imu_angular_velocity.size());
    input1.head(measured_joint_position.size()) = measured_joint_position;
    input1.tail(measured_imu_angular_velocity.size()) = measured_imu_angular_velocity;
    Eigen::VectorXd input2 = Eigen::VectorXd::Zero(njnt + 3);
    input2.head(njnt) = whole_body_controller_ptr_->get_q_ddot().tail(njnt);
    input2.tail(3) = whole_body_controller_ptr_->get_q_ddot().segment(3,3);
    joint_kf_ptr_->filter(input1, input2);
    if (t_msec_ == 1000){
        //concatenate q.head(7) with q_filtered
        Eigen::VectorXd input(njnt + 7);
        input.head(7) = q.head(7);
        input.tail(njnt) = joint_kf_ptr_->getFilteredJointPositions();
        base_ekf_ptr_->initialize(input, T_lsole_sim.translation(), T_rsole_sim.translation());
        std::cout << "INITIALIZATION" << std::endl;

        Eigen::VectorXd q_joints = joint_kf_ptr_->getFilteredJointPositions();
        ri_ekf_ptr_->addContact(0, q_joints);  // piede sinistro
        ri_ekf_ptr_->addContact(1, q_joints);  // piede destro


        diligent_kio_ptr_->addContact(0, q_joints);  // piede sinistro
        diligent_kio_ptr_->addContact(1, q_joints);  // piede destro
    }
    if (t_msec_ >= 1000 && true){

        input_acc = measured_imu_accelerometer;
        input_gyro = measured_imu_angular_velocity; // sim_robot_state.angular_velocity; //measured_imu_angular_velocity;
        base_ekf_ptr_->filter(input_acc,
            input_gyro,
            joint_kf_ptr_->getFilteredJointPositions(),
            joint_kf_ptr_->getFilteredJointVelocities(),
            whole_body_controller_ptr_->get_q_ddot().head(6),
            left_support_check,
            right_support_check
        );

        std::array<bool,2> contact = {left_support_check, right_support_check};
        ri_ekf_ptr_->filter(input_gyro, input_acc,
                            joint_kf_ptr_->getFilteredJointPositions(),
                            joint_kf_ptr_->getFilteredJointPositions(),
                            contact);
        
        // std::array<bool,2> contact = {left_support_check, right_support_check};
        diligent_kio_ptr_->filter(input_gyro, input_acc,
                            joint_kf_ptr_->getFilteredJointPositions(),
                            joint_kf_ptr_->getFilteredJointPositions(),
                            contact);
        
        // input_acc = measured_imu_accelerometer;
        // input_gyro = sim_robot_state.angular_velocity; //measured_imu_angular_velocity;

        std::cout << "RI EKF orientation " << ri_ekf_ptr_->getQuaternion().coeffs().transpose() << std::endl;
        std::cout << "RI EKF position " << ri_ekf_ptr_->getPosition().transpose() << std::endl;
        std::cout << "RI EKF velocity " << ri_ekf_ptr_->getVelocity().transpose() << std::endl;
        std::cout << "RI Omega Body: " << ri_ekf_ptr_->getOmegaBody().transpose() << std::endl;

        // angular velocity dal giroscopio bias-compensato in body frame
        // fb_robot_state.angular_velocity = ri_ekf_ptr_->getOmegaBody();
    }
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            if(t_msec_ >= startTimeEKF && isEKFactive) {
                fb_robot_state = updateEKF(actual_output);
                if (t_msec_ >= 5000 && true){
                    std::cout << "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA " << std::endl;
                    fb_robot_state.orientation = base_ekf_ptr_->getBaseOrientation();
                    fb_robot_state.position = base_ekf_ptr_->getBasePosition();
                    fb_robot_state.linear_velocity = fb_robot_state.orientation.toRotationMatrix().transpose() * base_ekf_ptr_->getBaseVelocity();
                    fb_robot_state.angular_velocity = fb_robot_state.orientation.toRotationMatrix().transpose() * fb_robot_data.oMf[imu_idx_].rotation() * joint_kf_ptr_->getFilteredOmega(); // per ora metto quello del joint KF, poi vediamo se mettere quello del BEKF o del RI EKF
                    // fb_robot_state.angular_velocity = base_ekf_ptr_->getBaseOmega();
                    fb_robot_state.angular_velocity = sim_robot_state.angular_velocity; // per ora metto quello del sim, poi vediamo se mettere quello del BEKF o del RI EKF


                    // use q_filtered for the joints
                    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
                        std::string joint_name = robot_model.names[joint_id + 2];
                        fb_robot_state.joint_state[joint_name].pos = joint_kf_ptr_->getFilteredJointPositions()(joint_id);
                        fb_robot_state.joint_state[joint_name].vel = joint_kf_ptr_->getFilteredJointVelocities()(joint_id);
                    }

                    std::cout << "Big EKF omega: " << fb_robot_state.angular_velocity.transpose() << std::endl;
                    std::cout << "Joint KF omega: " << (fb_robot_state.orientation.toRotationMatrix() * joint_kf_ptr_->getFilteredOmega()).transpose() << std::endl;
                    std::cout << "BEKF omega: " << (fb_robot_state.orientation.toRotationMatrix() * base_ekf_ptr_->getBaseOmega()).transpose() << std::endl;
                    std::cout << "input gyro: " << (fb_robot_state.orientation.toRotationMatrix() * input_gyro).transpose() << std::endl;
                    std::cout << "RI EKF omega: " << (fb_robot_state.orientation.toRotationMatrix() * ri_ekf_ptr_->getOmegaBody()).transpose() << std::endl;
                    std::cout << "Omega actual: " << input_gyro.transpose() << std::endl;

                    // fb_robot_state.orientation    = ri_ekf_ptr_->getQuaternion();
                    // fb_robot_state.position       = ri_ekf_ptr_->getPosition();
                    // fb_robot_state.linear_velocity= fb_robot_state.orientation.toRotationMatrix().transpose() * ri_ekf_ptr_->getVelocity();
                    // fb_robot_state.angular_velocity = fb_robot_state.orientation.toRotationMatrix().transpose() * measured_imu_angular_velocity; // per ora metto quello misurato, poi vediamo se mettere quello del RI EKF o del BEKF;


                    // fb_robot_state.orientation    = diligent_kio_ptr_->getQuaternion();
                    // fb_robot_state.position       = diligent_kio_ptr_->getPosition();
                    // fb_robot_state.linear_velocity= fb_robot_state.orientation.toRotationMatrix().transpose() * diligent_kio_ptr_->getVelocity();
                    // fb_robot_state.angular_velocity = fb_robot_data.oMf[imu_idx_].rotation() * fb_robot_state.orientation.toRotationMatrix().transpose() * joint_kf_ptr_->getFilteredOmega();
                }
            }
            else{
                fb_robot_state = sim_robot_state;
            }

        }
        #pragma omp section
        {
        }
    }
    auto end_ekf = std::chrono::high_resolution_clock::now();

    std::cout << "GT position " << sim_robot_state.position.transpose() << std::endl;
    std::cout << "GT orientation " << sim_robot_state.orientation.coeffs().transpose() << std::endl;
    std::cout << "GT velocity " << sim_robot_state.linear_velocity.transpose() << std::endl;
    std::cout << "BEKF position " << base_ekf_ptr_->getBasePosition().transpose() << std::endl;
    std::cout << "BEKF orientation " << base_ekf_ptr_->getBaseOrientation().coeffs().transpose() << std::endl;
    std::cout << "BEKF velocity " << base_ekf_ptr_->getBaseVelocity().transpose() << std::endl;
    // std::cout << "BEKF omega " << base_ekf_ptr_->getBaseOmega().transpose() << std::endl;

    ////////////////////
    // END EKF CALL
    ///////////////////

    // UPDATE FORWARD KINEMATICS, LIP AND PINOCCHIO QUANTITIES FOR FEEDBACK FILTERED STATE

    auto q_fb_filt = robot_state_to_pinocchio_joint_configuration(robot_model, fb_robot_state);
    auto qdot_fb_filt = robot_state_to_pinocchio_joint_velocity(robot_model, fb_robot_state);

    pinocchio::forwardKinematics(robot_model, fb_robot_data, q_fb_filt);

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

    const auto& T_pelvis_fb = fb_robot_data.oMf[pelvis_idx_];
    auto pelvis_orientation_fb = T_pelvis_fb.rotation();
    Eigen::MatrixXd J_pelvis_fb = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        fb_robot_data,
        pelvis_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_pelvis_fb
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

    Eigen::MatrixXd J_imu_fb = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        fb_robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu_fb
    );

    if (t_msec_ == 5000){
        walking_data_.swapStanding(
            labrob::SE3(T_lsole_fb.rotation(), Eigen::Vector3d(T_lsole_fb.translation().x(), T_lsole_fb.translation().y(), (T_lsole_fb.translation().z() + T_lsole_fb.translation().z())/2)),
            labrob::SE3(T_rsole_fb.rotation(), Eigen::Vector3d(T_rsole_fb.translation().x(), T_rsole_fb.translation().y(), (T_rsole_fb.translation().z() + T_rsole_fb.translation().z())/2))
        );
    }

    // IF USE ROBOT, USE FEEDBACK FEET POSITIONS FOR STEPS

    if (loopClosed && t_msec_ >= startTimeWBCCL && isWBCLoopClosed && useRobot) {

        walking_data_.swapStanding(
            labrob::SE3(T_lsole_fb.rotation(), Eigen::Vector3d(T_lsole_fb.translation().x(), T_lsole_fb.translation().y(), (T_lsole_fb.translation().z() + T_lsole_fb.translation().z())/2)),
            labrob::SE3(T_rsole_fb.rotation(), Eigen::Vector3d(T_rsole_fb.translation().x(), T_rsole_fb.translation().y(), (T_rsole_fb.translation().z() + T_rsole_fb.translation().z())/2))
        );
        fixed_com_pos = p_CoM_fb;
        fixed_com_vel = v_CoM_fb;
        double com_target_height = p_CoM_fb.z() - (T_lsole_fb.translation().z() + T_lsole_fb.translation().z())/2;
        eta2 = 9.81 / com_target_height;
        fixed_zmp_pos = p_CoM_fb - Eigen::Vector3d(0, 0, com_target_height);

        kf_LipState = labrob::LIPState(
            fixed_com_pos,
            Eigen::Vector3d::Zero(),
            fixed_zmp_pos
        );
        des_LipState = labrob::LIPState(
            fixed_com_pos,
            Eigen::Vector3d::Zero(),
            fixed_zmp_pos
        );
        // ismpc_ptr_->resetInput();
        ismpc_ptr_->setEta(std::sqrt(eta2));
        discrete_lip_dynamics_ptr_->setEta(std::sqrt(eta2));
        discrete_lip_dynamics_ptr_mpc_->setEta(std::sqrt(eta2));
        com_kf_ptr_->setEta(std::sqrt(eta2));

        // remove previous parameters
        parameters_log_.clear();
        parameters_log_.push_back(startTimeWBCCL);

        loopClosed = false;
    }

    // ZMP POSITION UPDATE

    Eigen::Vector3d zmp_3d_fb;

    // FIRST FORMULA FOR ZMP POSITION WITH FORCE ESTIMATION WITH 4 CONTACT POINTS PER FOOT

    // zmp_3d_fb.z() = fb_robot_state.position(2) - fb_robot_state.total_force.z() / (mass * eta2);
    // zmp_3d_fb.x() = 0.0;
    // zmp_3d_fb.y() = 0.0;
    // for (int i = 0; i < fb_robot_state.contact_points.size(); ++i) {
    //     auto &pi = fb_robot_state.contact_points[i];
    //     auto &fi = fb_robot_state.contact_forces[i];
    //     zmp_3d_fb.x() += (pi.x() * fi.z() / fb_robot_state.total_force.z() + (zmp_3d_fb.z() - pi.z()) * fi.x() / fb_robot_state.total_force.z());
    //     zmp_3d_fb.y() += (pi.y() * fi.z() / fb_robot_state.total_force.z() + (zmp_3d_fb.z() - pi.z()) * fi.y() / fb_robot_state.total_force.z());
    // }

    // zmp_3d_fb.z() = fb_robot_state.position(2) - total_force.z() / (mass * eta2);

    zmp_3d_fb.z() = (T_lsole_fb.translation().z() + T_rsole_fb.translation().z()) /2;
    zmp_3d_fb.x() = 0.0;
    zmp_3d_fb.y() = 0.0;
    if (total_force.z() > 1e-5) {

        // FIRST FORMULA FOR ZMP POSITION WITH FORCE ESTIMATION WITH 1 CONTACT POINT PER FOOT

        // if (left_foot_force.z() > 1e-5) {
        //     zmp_3d_fb.x() += (T_lsole_fb.translation().x() * left_foot_force.z() / total_force.z() + (zmp_3d_fb.z() - T_lsole_fb.translation().z()) * left_foot_force.x() / total_force.z());
        //     zmp_3d_fb.y() += (T_lsole_fb.translation().y() * left_foot_force.z() / total_force.z() + (zmp_3d_fb.z() - T_lsole_fb.translation().z()) * left_foot_force.y() / total_force.z());
        // }
        // if (right_foot_force.z() > 1e-5) {
        //     zmp_3d_fb.x() += (T_rsole_fb.translation().x() * right_foot_force.z() / total_force.z() + (zmp_3d_fb.z() - T_rsole_fb.translation().z()) * right_foot_force.x() / total_force.z());
        //     zmp_3d_fb.y() += (T_rsole_fb.translation().y() * right_foot_force.z() / total_force.z() + (zmp_3d_fb.z() - T_rsole_fb.translation().z()) * right_foot_force.y() / total_force.z());
        // }

        // SECOND FORMULA FOR ZMP POSITION WITH FORCE ESTIMATION WITH 1 CONTACT POINT PER FOOT

        zmp_3d_fb.x() =
            ( left_foot_force.z()  * T_lsole_fb.translation().x() +
            right_foot_force.z() * T_rsole_fb.translation().x() ) / total_force.z();

        zmp_3d_fb.y() =
            ( left_foot_force.z()  * T_lsole_fb.translation().y() +
            right_foot_force.z() * T_rsole_fb.translation().y() ) / total_force.z();
    }

    ef_zmp_position_log_.push_back(zmp_3d_fb.transpose());

    // THIRD FORMULA FOR ZMP POSITION WITHOUT FORCE ESTIMATION (ONLY LIP EQUATIONS)

    zmp_3d_fb.z() = p_CoM_fb.z() - (a_CoM_drift_fb.z() + 9.81) / eta2;
    zmp_3d_fb.x() = p_CoM_fb.x() - a_CoM_drift_fb.x() / eta2;
    zmp_3d_fb.y() = p_CoM_fb.y() - a_CoM_drift_fb.y() / eta2;

    // FILL INTEGRATED STATE FOR FEEDBACK CONTROL

    if(isWBCLoopClosed && t_msec_ >= startTimeWBCCL){
        integrated_state_pos.head(3) = fb_robot_state.position;
        integrated_state_pos.segment<3>(3) = rotVecFromQuaternion(fb_robot_state.orientation);
        integrated_state_vel.head(3) = fb_robot_state.linear_velocity;
        integrated_state_vel.segment<3>(3) = fb_robot_state.angular_velocity;
        for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
            std::string joint_name = robot_model.names[joint_id + 2];
            integrated_state_pos(6 + joint_id) = fb_robot_state.joint_state[joint_name].pos;
            integrated_state_vel(6 + joint_id) = fb_robot_state.joint_state[joint_name].vel;
        }
    } else {
        integrated_state_pos.head(3) = sim_robot_state.position;
        integrated_state_pos.segment<3>(3) = rotVecFromQuaternion(sim_robot_state.orientation);
        integrated_state_vel.head(3) = sim_robot_state.linear_velocity;
        integrated_state_vel.segment<3>(3) = sim_robot_state.angular_velocity;
        for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
            std::string joint_name = robot_model.names[joint_id + 2];
            integrated_state_pos(6 + joint_id) = sim_robot_state.joint_state[joint_name].pos;
            integrated_state_vel(6 + joint_id) = sim_robot_state.joint_state[joint_name].vel;
        }
    }

    // ADD STEPS FOR SIMULATION

    if(!useRobot && t_msec_ == 5000 && true){
        double yaw_angle = rpyFromQuaternion(Eigen::Quaterniond(fb_robot_data.oMf[imu_idx_].rotation())).z();
        walking_data_.addSteps(
            labrob::SE3(T_lsole_fb.rotation(), T_lsole_fb.translation()),
            labrob::SE3(T_rsole_fb.rotation(), T_rsole_fb.translation()),
            yaw_angle
        );
    };

    walking_data_.updateWalkingState(t_msec_);

    /////////////////////////////////////
    // KF FUNCTION CALL
    /////////////////////////////////////

    auto start_kf = std::chrono::high_resolution_clock::now();
    if (t_msec_ >= startTimeWBCCL && isWBCLoopClosed){
        if (t_msec_ == startTimeWBCCL){
            std::cout << "Using feedback Center of Mass" << std::endl;
        }
        LipState = LIPState(p_CoM_fb, J_CoM_fb * qdot_fb_filt, zmp_3d_fb);
    } else {
        LipState = LIPState(p_CoM_sim, J_CoM_sim * qdot, zmp_3d_sim);
    }
    kf_LipState = com_kf_ptr_->filter(kf_LipState, LipState, ismpc_ptr_->getInput());
    // kf_LipState = LipState;
    auto end_kf = std::chrono::high_resolution_clock::now();

    ////////////////////////////////////
    // END KF FUNCTION CALL
    ////////////////////////////////////

    // IF STANDING, ADD STEPS TO START WALKING AGAIN OR IF DOUBLE SUPPORT, REMOVE STEPS TO GO BACK TO STANDING

    if (switchWalkingState){
        double yaw_angle = rpyFromQuaternion(Eigen::Quaterniond(fb_robot_data.oMf[imu_idx_].rotation())).z();
        if (walking_data_.getWalkingState() == WalkingState::Standing) {
            walking_data_.addSteps(
                labrob::SE3(T_lsole_fb.rotation(), Eigen::Vector3d(T_lsole_fb.translation().x(), T_lsole_fb.translation().y(), (T_lsole_fb.translation().z() + T_lsole_fb.translation().z())/2)),
                labrob::SE3(T_rsole_fb.rotation(), Eigen::Vector3d(T_rsole_fb.translation().x(), T_rsole_fb.translation().y(), (T_rsole_fb.translation().z() + T_rsole_fb.translation().z())/2)),
                yaw_angle
            );
            switchWalkingState = false;
        } else if (walking_data_.getWalkingState() == WalkingState::DoubleSupport) {
            std::cout << "Removing steps" << std::endl;
            walking_data_.removeSteps();
            switchWalkingState = false;
        }
    }

    // FILL CURRENT GAIT CONFIGURATION

    // joints
    labrob::GaitConfiguration current_gait_configuration;
    if (t_msec_ >= startTimeWBCCL && isWBCLoopClosed){
        current_gait_configuration.qjnt = q_fb_filt.tail(njnt);
        current_gait_configuration.qjntdot = qdot_fb_filt.tail(njnt);
    } else {
        current_gait_configuration.qjnt = q.tail(njnt);
        current_gait_configuration.qjntdot = qdot.tail(njnt);
    }

    current_gait_configuration.is_left_foot_support = true;
    current_gait_configuration.is_right_foot_support = true;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
    if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT) current_gait_configuration.is_right_foot_support = false;
    else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT) current_gait_configuration.is_left_foot_support = false;
    }

    current_gait_configuration.com.pos = kf_LipState.com_pos_;
    current_gait_configuration.com.vel = kf_LipState.com_vel_;

    //torso
    if (t_msec_ >= startTimeWBCCL && isWBCLoopClosed){
        current_gait_configuration.torso.pos = fb_robot_data.oMf[torso_idx_].rotation();
        current_gait_configuration.torso.vel = J_torso_fb.bottomRows<3>() * qdot_fb_filt;
    } else {
        current_gait_configuration.torso.pos = sim_robot_data.oMf[torso_idx_].rotation();
        current_gait_configuration.torso.vel = J_torso_sim.bottomRows<3>() * qdot;
    }

    //pelvis
    if (t_msec_ >= startTimeWBCCL && isWBCLoopClosed){
        current_gait_configuration.pelvis.pos = fb_robot_data.oMf[pelvis_idx_].rotation();
        current_gait_configuration.pelvis.vel = J_pelvis_fb.bottomRows<3>() * qdot_fb_filt;
    } else {
        current_gait_configuration.pelvis.pos = sim_robot_data.oMf[pelvis_idx_].rotation();
        current_gait_configuration.pelvis.vel = J_pelvis_sim.bottomRows<3>() * qdot;
    }

    //feet
    if (t_msec_ >= startTimeWBCCL && isWBCLoopClosed){
        current_gait_configuration.lsole.pos = labrob::SE3(fb_robot_data.oMf[lsole_idx_].rotation(), fb_robot_data.oMf[lsole_idx_].translation());
        current_gait_configuration.lsole.vel = J_lsole_fb * qdot_fb_filt;
    } else {
        current_gait_configuration.lsole.pos = labrob::SE3(sim_robot_data.oMf[lsole_idx_].rotation(), sim_robot_data.oMf[lsole_idx_].translation());
        current_gait_configuration.lsole.vel = J_lsole_sim * qdot;
    }

    if (t_msec_ >= startTimeWBCCL && isWBCLoopClosed){
        current_gait_configuration.rsole.pos = labrob::SE3(fb_robot_data.oMf[rsole_idx_].rotation(), fb_robot_data.oMf[rsole_idx_].translation());
        current_gait_configuration.rsole.vel = J_rsole_fb * qdot_fb_filt;
    } else {
        current_gait_configuration.rsole.pos = labrob::SE3(sim_robot_data.oMf[rsole_idx_].rotation(), sim_robot_data.oMf[rsole_idx_].translation());
        current_gait_configuration.rsole.vel = J_rsole_sim * qdot;
    }

    /////////////////////////////////////
    // MPC FUNCTION CALL
    /////////////////////////////////////

    LIPState mpc_LipState_prec = des_LipState;

    Eigen::Vector3d foot_pose = Eigen::Vector3d::Zero();
    auto start_mpc = std::chrono::system_clock::now();
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT){
                foot_pose = Eigen::Vector3d(current_gait_configuration.lsole.pos.p.x(), current_gait_configuration.lsole.pos.p.y(), current_gait_configuration.lsole.pos.p.z());
                // std::cout << "Left foot support" << std::endl;
            }
            else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT){
                foot_pose = Eigen::Vector3d(current_gait_configuration.rsole.pos.p.x(), current_gait_configuration.rsole.pos.p.y(), current_gait_configuration.rsole.pos.p.z());
                // std::cout << "Right foot support" << std::endl;
            }
            foot_pose.setZero();
            if(isMPCLoopClosed && t_msec_>= startTimeMPCCL){
                ismpc_ptr_->solve(t_msec_, walking_data_, kf_LipState, foot_pose);
                kf_LipState.com_pos_ -= foot_pose;
                kf_LipState.zmp_pos_ -= foot_pose;
                des_LipState = discrete_lip_dynamics_ptr_->integrate(
                    kf_LipState,
                    ismpc_ptr_->getInput()
                );
                kf_LipState.com_pos_ += foot_pose;
                kf_LipState.zmp_pos_ += foot_pose;
            }
            else{
                ismpc_ptr_->solve(t_msec_, walking_data_, des_LipState, foot_pose);
                des_LipState.com_pos_ -= foot_pose;
                des_LipState.zmp_pos_ -= foot_pose;
                des_LipState = discrete_lip_dynamics_ptr_->integrate(
                    des_LipState,
                    ismpc_ptr_->getInput()
                );
            }
            des_LipState.com_pos_ += foot_pose;
            des_LipState.zmp_pos_ += foot_pose;
        }
        #pragma omp section
        {
        }
    }

    mpc_zmp_velocity_log_.push_back(ismpc_ptr_->getInput().transpose());
    // con_zmp_velocity_log_.push_back(ismpc_ptr_->getStabConstraintOffset().transpose());


    auto end_mpc = std::chrono::system_clock::now();

    /////////////////////////////////////
    // MPC FUNCTION CALL END
    /////////////////////////////////////

    // LOG MPC PREDICTIONS FOR GIF FILES

    Eigen::VectorXd inputSequenceX = ismpc_ptr_->getInputSequenceX();
    Eigen::VectorXd inputSequenceY = ismpc_ptr_->getInputSequenceY();
    Eigen::VectorXd inputSequenceZ = ismpc_ptr_->getInputSequenceZ();

    LIPState LipState_mpc = mpc_LipState_prec;

    for (int i = 0; i < 20; ++i) {
        LipState_mpc = discrete_lip_dynamics_ptr_mpc_->integrate(
            LipState_mpc,
            Eigen::Vector3d(inputSequenceX(i), inputSequenceY(i), inputSequenceZ(i))
        );

        mpc_pred_com_pos_log_.push_back(LipState_mpc.com_pos_);
        mpc_pred_com_vel_log_.push_back(LipState_mpc.com_vel_);
        mpc_pred_zmp_pos_log_.push_back(LipState_mpc.zmp_pos_);
    }

    // FILL DESIRED GAIT CONFIGURATION

    // joint
    labrob::GaitConfiguration desired_gait_configuration;
    desired_gait_configuration.qjnt = q_jnt_des_;
    desired_gait_configuration.qjntdot = Eigen::VectorXd::Zero(njnt);
    desired_gait_configuration.qjntddot = Eigen::VectorXd::Zero(njnt);

    // simple arm swing policy based on hip pitch variation and step length
    double left_hip_pitch_current = current_gait_configuration.qjnt(robot_model.getJointId("left_hip_pitch_joint") - 2);
    double left_hip_pitch_desired = desired_gait_configuration.qjnt(robot_model.getJointId("left_hip_pitch_joint") - 2);
    double left_hip_pitch_variation = left_hip_pitch_desired - left_hip_pitch_current;
    double right_hip_pitch_current = current_gait_configuration.qjnt(robot_model.getJointId("right_hip_pitch_joint") - 2);
    double right_hip_pitch_desired = desired_gait_configuration.qjnt(robot_model.getJointId("right_hip_pitch_joint") - 2);
    double right_hip_pitch_variation = right_hip_pitch_desired - right_hip_pitch_current;
    double arm_swing_amplitude = 3; // adjust this value to change the amount of arm swing
    // compute difference bewteen foot poses in x direction to adjust arm swing amplitude based on step length
    double left_foot_x = walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().p.x();
    double right_foot_x = walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().p.x();
    double step_length = left_foot_x - right_foot_x;
    arm_swing_amplitude *= step_length / 0.3; // assuming 0.3m is the nominal step length

    // ARMS SWING, UNCOMMENT TO ENABLE

    // desired_gait_configuration.qjnt(robot_model.getJointId("left_shoulder_pitch_joint") - 2) +=  1 * left_hip_pitch_variation - 1 * right_hip_pitch_variation;
    // desired_gait_configuration.qjnt(robot_model.getJointId("right_shoulder_pitch_joint") - 2) += - 1 * left_hip_pitch_variation + 1 * right_hip_pitch_variation;

    // desired_gait_configuration.qjnt(robot_model.getJointId("left_shoulder_pitch_joint") - 2) += - 2 * step_length;
    // desired_gait_configuration.qjnt(robot_model.getJointId("right_shoulder_pitch_joint") - 2) += 2 * step_length;

    // desired_gait_configuration.qjnt(robot_model.getJointId("left_elbow_joint") - 2) += 1 * left_hip_pitch_variation - 1 * right_hip_pitch_variation;
    // desired_gait_configuration.qjnt(robot_model.getJointId("right_elbow_joint") - 2) += - 1 * left_hip_pitch_variation + 1 * right_hip_pitch_variation;

    // desired_gait_configuration.qjnt(robot_model.getJointId("left_elbow_joint") - 2) += - 1 * step_length;
    // desired_gait_configuration.qjnt(robot_model.getJointId("right_elbow_joint") - 2) += 1 * step_length;


    // CoM (take des values from des lip state, mpc output)
    Eigen::Vector3d p_CoM_des;
    Eigen::Vector3d v_CoM_des;
    Eigen::Vector3d p_ZMP_des;
    p_CoM_des = des_LipState.com_pos_;
    v_CoM_des = des_LipState.com_vel_;
    p_ZMP_des = des_LipState.zmp_pos_;

    desired_gait_configuration.com.pos = p_CoM_des;
    desired_gait_configuration.com.vel = v_CoM_des;
    desired_gait_configuration.com.acc = eta2 * (p_CoM_des - p_ZMP_des) - Eigen::Vector3d(0.0, 0.0, 9.81);

    // feet tasks
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

    // torso task
    double left_foot_yaw = std::atan2(desired_gait_configuration.lsole.pos.R(1, 0), desired_gait_configuration.lsole.pos.R(0, 0));
    double right_foot_yaw = std::atan2(desired_gait_configuration.rsole.pos.R(1, 0), desired_gait_configuration.rsole.pos.R(0, 0));
    desired_gait_configuration.torso.pos = Rz((left_foot_yaw + right_foot_yaw) / 2.0);
    desired_gait_configuration.torso.vel = (desired_gait_configuration.lsole.vel.tail(3) + desired_gait_configuration.rsole.vel.tail(3)) / 2.0;
    desired_gait_configuration.torso.acc = (desired_gait_configuration.lsole.acc.tail(3) + desired_gait_configuration.rsole.acc.tail(3)) / 2.0;

    // pelvis task
    // double left_foot_yaw = std::atan2(desired_gait_configuration.lsole.pos.R(1, 0), desired_gait_configuration.lsole.pos.R(0, 0));
    // double right_foot_yaw = std::atan2(desired_gait_configuration.rsole.pos.R(1, 0), desired_gait_configuration.rsole.pos.R(0, 0));
    desired_gait_configuration.pelvis.pos = Rz((left_foot_yaw + right_foot_yaw) / 2.0);
    desired_gait_configuration.pelvis.vel = (desired_gait_configuration.lsole.vel.tail(3) + desired_gait_configuration.rsole.vel.tail(3)) / 2.0;
    desired_gait_configuration.pelvis.acc = (desired_gait_configuration.lsole.acc.tail(3) + desired_gait_configuration.rsole.acc.tail(3)) / 2.0;

    /////////////////////////////////////
    // START WHOLE BODY CONTROLLER FUNCTION CALL
    /////////////////////////////////////

    auto start_wbc = std::chrono::system_clock::now();
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            if (t_msec_ >= startTimeWBCCL && isWBCLoopClosed){
                joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
                    robot_model,
                    fb_robot_state,
                    fb_robot_data,
                    current_gait_configuration,
                    desired_gait_configuration,
                    foot_pose
                );
            } else {
                // Use the MPC to compute the joint command:
                joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
                    robot_model,
                    sim_robot_state,
                    sim_robot_data,
                    current_gait_configuration,
                    desired_gait_configuration,
                    foot_pose
                );
            }
        }
        #pragma omp section
        {
        }
    } // end of parallel sections
    auto end_wbc = std::chrono::system_clock::now();

    /////////////////////////////////////
    // END WHOLE BODY CONTROLLER FUNCTION CALL
    /////////////////////////////////////

    // get measured joint torques from the joint command
    Eigen::VectorXd measured_torques(robot_model.nv - 6);  // Exclude floating base
    int idx = 0;
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) robot_model.njoints; ++joint_id) {
        const auto& joint_name = robot_model.names[joint_id];
        measured_torques(idx++) = joint_command[joint_name];
    }

    // FORCE ESTIMATION WITH RESIDUAL ESTIMATOR

    residual_estimator_ptr_->computeResidualWithWBCWrenches(
        fb_robot_state,
        fb_robot_data,
        measured_torques,
        *whole_body_controller_ptr_,
        controller_timestep_msec_*0.001
    );
    estimated_force = residual_estimator_ptr_->getFeetEstimatedForce();

    // Update timing in milliseconds.
    // NOTE: assuming update() is actually called every controller_timestep_msec_
    //       milliseconds.
    t_msec_ += controller_timestep_msec_;
    prev_angular_momentum_ = angular_momentum;

    // PUSHBACK FOR LOGS AND SAVE LOGS FUNCTION

    fb_com_position_log_.push_back(p_CoM_fb.transpose());
    kf_com_position_log_.push_back(kf_LipState.com_pos_.transpose());
    des_com_position_log_.push_back(p_CoM_des.transpose());

    fb_com_velocity_log_.push_back(v_CoM_fb.transpose());
    kf_com_velocity_log_.push_back((kf_LipState.com_vel_).transpose());
    des_com_velocity_log_.push_back(v_CoM_des.transpose());

    fb_zmp_position_log_.push_back(zmp_3d_fb.transpose());
    kf_zmp_position_log_.push_back(kf_LipState.zmp_pos_.transpose());
    des_zmp_position_log_.push_back(p_ZMP_des.transpose());

    des_com_acceleration_log_.push_back(desired_gait_configuration.com.acc.transpose());

    p_lsole_fb_log_.push_back(T_lsole_fb.translation().transpose());
    p_rsole_fb_log_.push_back(T_rsole_fb.translation().transpose());
    v_lsole_fb_log_.push_back(v_lsole_fb.head<3>().transpose());
    v_rsole_fb_log_.push_back(v_rsole_fb.head<3>().transpose());
    p_lsole_des_log_.push_back(desired_gait_configuration.lsole.pos.p.transpose());
    p_rsole_des_log_.push_back(desired_gait_configuration.rsole.pos.p.transpose());
    v_lsole_des_log_.push_back(desired_gait_configuration.lsole.vel.head<3>().transpose());
    v_rsole_des_log_.push_back(desired_gait_configuration.rsole.vel.head<3>().transpose());

    fb_lsole_orientation_log_.push_back(T_lsole_fb.rotation().eulerAngles(0,1,2).transpose());
    fb_rsole_orientation_log_.push_back(T_rsole_fb.rotation().eulerAngles(0,1,2).transpose());
    des_lsole_orientation_log_.push_back(desired_gait_configuration.lsole.pos.R.eulerAngles(0,1,2).transpose());
    des_rsole_orientation_log_.push_back(desired_gait_configuration.rsole.pos.R.eulerAngles(0,1,2).transpose());

    estimated_force_lsole_log_.push_back(estimated_force.head<3>().transpose());
    estimated_force_rsole_log_.push_back(estimated_force.tail<3>().transpose());
    wbc_accelerations_log_.push_back(whole_body_controller_ptr_->get_q_ddot().transpose());

    angular_momentum_log_.push_back(angular_momentum.transpose());
    // log measurements present in actual output
    odometry_base_position_log_.push_back(odometry_base_position.transpose());
    odometry_base_velocity_log_.push_back(odometry_base_velocity.transpose());
    odometry_imu_orientation_log_.push_back(odometry_imu_quaternion.transpose());
    odometry_imu_orientation_rpy_log_.push_back(odometry_imu_rpy.transpose());
    measured_imu_orientation_log_.push_back(measured_imu_quaternion.transpose());
    measured_imu_orientation_rpy_log_.push_back(measured_imu_rpy.transpose());
    measured_imu_angular_velocity_log_.push_back(measured_imu_angular_velocity.transpose());
    measured_joint_position_log_.push_back(Eigen::VectorXd(njnt).transpose());
    measured_joint_velocity_log_.push_back(Eigen::VectorXd(njnt).transpose());
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        measured_joint_position_log_.back()(joint_id) = measured_joint_position(joint_id);
        measured_joint_velocity_log_.back()(joint_id) = measured_joint_velocity(joint_id);
    }
    measured_imu_accelerometer_log_.push_back(measured_imu_accelerometer.transpose());
    // Log the filtered state:
    ekf_base_position_log_.push_back(fb_robot_state.position.transpose());
    ekf_base_velocity_log_.push_back(fb_robot_state.linear_velocity.transpose());
    ekf_base_orientation_log_.push_back(Eigen::Vector4d(
        fb_robot_state.orientation.w(),
        fb_robot_state.orientation.x(),
        fb_robot_state.orientation.y(),
        fb_robot_state.orientation.z()
    ).transpose());
    ekf_base_orientation_rpy_log_.push_back(rpyFromQuaternion(fb_robot_state.orientation).transpose());
    ekf_base_angular_velocity_log_.push_back(fb_robot_state.angular_velocity.transpose());
    ekf_imu_orientation_log_.push_back(Eigen::Vector4d(Eigen::Quaterniond(fb_robot_data.oMf[imu_idx_].rotation()).w(),
        Eigen::Quaterniond(fb_robot_data.oMf[imu_idx_].rotation()).x(),
        Eigen::Quaterniond(fb_robot_data.oMf[imu_idx_].rotation()).y(),
        Eigen::Quaterniond(fb_robot_data.oMf[imu_idx_].rotation()).z()).transpose());
    ekf_imu_orientation_rpy_log_.push_back(rpyFromQuaternion(Eigen::Quaterniond(fb_robot_data.oMf[imu_idx_].rotation())).transpose());
    ekf_imu_angular_velocity_log_.push_back(fb_robot_state.angular_velocity.transpose());
    ekf_joint_position_log_.push_back(Eigen::VectorXd(njnt).transpose());
    ekf_joint_velocity_log_.push_back(Eigen::VectorXd(njnt).transpose());
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        ekf_joint_position_log_.back()(joint_id) = fb_robot_state.joint_state[joint_name].pos;
        ekf_joint_velocity_log_.back()(joint_id) = fb_robot_state.joint_state[joint_name].vel;
    }
    input_torque_log_.push_back(Eigen::VectorXd(njnt).transpose());
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        input_torque_log_.back()(joint_id) = joint_command[joint_name];
    }
    torso_orientation_log_.push_back(rpyFromQuaternion(Eigen::Quaterniond(current_gait_configuration.torso.pos)).transpose());
    torso_angular_velocity_log_.push_back(current_gait_configuration.torso.vel.tail<3>().transpose());
    des_torso_orientation_log_.push_back(rpyFromQuaternion(Eigen::Quaterniond(desired_gait_configuration.torso.pos)).transpose());
    des_torso_angular_velocity_log_.push_back(desired_gait_configuration.torso.vel.tail<3>().transpose());

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

RobotState WalkingManager::getNewRobotState(){
    Eigen::VectorXd wbc_qddot = whole_body_controller_ptr_->get_q_ddot();
    // integrated_state_pos = integrated_state_pos + integrated_state_vel * controller_timestep_msec_ * 0.001 + 0.5 * wbc_qddot * std::pow(controller_timestep_msec_ * 0.001, 2);
    // integrated_state_vel = integrated_state_vel + wbc_qddot * controller_timestep_msec_ * 0.001;


    // integrated_state_vel = integrated_state_vel + wbc_qddot * controller_timestep_msec_ * 0.001;
    // integrated_state_pos = integrated_state_pos + integrated_state_vel * controller_timestep_msec_ * 0.001;

    //integrate using pinocchio integrate
    Eigen::VectorXd pinocchio_state = Eigen::VectorXd::Zero(robot_model.nq);
    pinocchio_state.head(3) = integrated_state_pos.head(3);
    pinocchio_state.segment(3,4) = Eigen::Vector4d(
        quaternionFromRotVec(integrated_state_pos.segment(3,3)).x(),
        quaternionFromRotVec(integrated_state_pos.segment(3,3)).y(),
        quaternionFromRotVec(integrated_state_pos.segment(3,3)).z(),
        quaternionFromRotVec(integrated_state_pos.segment(3,3)).w()
    );
    pinocchio_state.tail(njnt) = integrated_state_pos.tail(njnt);
    pinocchio_state = pinocchio::integrate(robot_model, pinocchio_state, integrated_state_vel * controller_timestep_msec_ * 0.001);

    integrated_state_vel = integrated_state_vel + wbc_qddot * controller_timestep_msec_ * 0.001;

    RobotState robot_state;

    robot_state.position = pinocchio_state.head<3>();
    robot_state.orientation = Eigen::Quaterniond(
        pinocchio_state(6),
        pinocchio_state(3),
        pinocchio_state(4),
        pinocchio_state(5)
    );
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        robot_state.joint_state[joint_name].pos = pinocchio_state(7 + joint_id);
        robot_state.joint_state[joint_name].vel = integrated_state_vel(6 + joint_id);
    }
    robot_state.linear_velocity = integrated_state_vel.head<3>();
    robot_state.angular_velocity = integrated_state_vel.segment<3>(3);
    return robot_state;
}

RobotState WalkingManager::getActualRobotState(){
    return fb_robot_state;
}

void WalkingManager::saveLogs() {


    std::ofstream joint_names_file("/tmp/joint_names.txt");
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        joint_names_file << joint_name << "\n";
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

    std::ofstream des_com_acceleration_file("/tmp/des_com_acceleration.txt");
    for (auto& v : des_com_acceleration_log_) {
        des_com_acceleration_file << v.transpose() << "\n";
    }

    std::ofstream ef_zmp_position_file("/tmp/ef_zmp_position.txt");
    for (auto& v : ef_zmp_position_log_) {
        ef_zmp_position_file << v.transpose() << "\n";
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

    std::ofstream fb_lsole_orientation_file("/tmp/fb_lsole_orientation.txt");
    for (auto& v : fb_lsole_orientation_log_) {
        fb_lsole_orientation_file << v.transpose() << "\n";
    }

    std::ofstream fb_rsole_orientation_file("/tmp/fb_rsole_orientation.txt");
    for (auto& v : fb_rsole_orientation_log_) {
        fb_rsole_orientation_file << v.transpose() << "\n";
    }

    std::ofstream des_lsole_orientation_file("/tmp/des_lsole_orientation.txt");
    for (auto& v : des_lsole_orientation_log_) {
        des_lsole_orientation_file << v.transpose() << "\n";
    }

    std::ofstream des_rsole_orientation_file("/tmp/des_rsole_orientation.txt");
    for (auto& v : des_rsole_orientation_log_) {
        des_rsole_orientation_file << v.transpose() << "\n";
    }

    std::ofstream estimated_force_lsole_file("/tmp/estimated_force_lsole.txt");
    for (auto& v : estimated_force_lsole_log_) {
        estimated_force_lsole_file << v.transpose() << "\n";
    }

    std::ofstream estimated_force_rsole_file("/tmp/estimated_force_rsole.txt");
    for (auto& v : estimated_force_rsole_log_) {
        estimated_force_rsole_file << v.transpose() << "\n";
    }

    std::ofstream wbc_accelerations_file("/tmp/wbc_accelerations.txt");
    for (auto& v : wbc_accelerations_log_) {
        wbc_accelerations_file << v.transpose() << "\n";
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

    std::ofstream measured_imu_orientation_rpy_log_file("/tmp/measured_imu_orientation_rpy.txt");
    for (auto& v : measured_imu_orientation_rpy_log_) {
        measured_imu_orientation_rpy_log_file << v.transpose() << "\n";
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

    std::ofstream ekf_base_orientation_rpy_file("/tmp/ekf_base_orientation_rpy.txt");
    for (auto& v : ekf_base_orientation_rpy_log_) {
        ekf_base_orientation_rpy_file << v.transpose() << "\n";
    }

    std::ofstream ekf_base_angular_velocity_file("/tmp/ekf_base_angular_velocity.txt");
    for (auto& v : ekf_base_angular_velocity_log_) {
        ekf_base_angular_velocity_file << v.transpose() << "\n";
    }

    std::ofstream ekf_imu_orientation_file("/tmp/ekf_imu_orientation.txt");
    for (auto& v : ekf_imu_orientation_log_) {
        ekf_imu_orientation_file << v.transpose() << "\n";
    }

    std::ofstream ekf_imu_orientation_rpy_file("/tmp/ekf_imu_orientation_rpy.txt");
    for (auto& v : ekf_imu_orientation_rpy_log_) {
        ekf_imu_orientation_rpy_file << v.transpose() << "\n";
    }

    std::ofstream ekf_imu_angular_velocity_file("/tmp/ekf_imu_angular_velocity.txt");
    for (auto& v : ekf_imu_angular_velocity_log_) {
        ekf_imu_angular_velocity_file << v.transpose() << "\n";
    }

    std::ofstream ekf_joint_position_file("/tmp/ekf_joint_position.txt");
    for (auto& v : ekf_joint_position_log_) {
        ekf_joint_position_file << v.transpose() << "\n";
    }

    std::ofstream ekf_joint_velocity_file("/tmp/ekf_joint_velocity.txt");
    for (auto& v : ekf_joint_velocity_log_) {
        ekf_joint_velocity_file << v.transpose() << "\n";
    }

    std::ofstream odometry_base_position_file("/tmp/odometry_base_position.txt");
    for (auto& v : odometry_base_position_log_) {
        odometry_base_position_file << v.transpose() << "\n";
    }

    std::ofstream odometry_base_velocity_file("/tmp/odometry_base_velocity.txt");
    for (auto& v : odometry_base_velocity_log_) {
        odometry_base_velocity_file << v.transpose() << "\n";
    }

    std::ofstream odometry_imu_orientation_file("/tmp/odometry_imu_orientation.txt");
    for (auto& v : odometry_imu_orientation_log_) {
        odometry_imu_orientation_file << v.transpose() << "\n";
    }

    std::ofstream odometry_imu_orientation_rpy_file("/tmp/odometry_imu_orientation_rpy.txt");
    for (auto& v : odometry_imu_orientation_rpy_log_) {
        odometry_imu_orientation_rpy_file << v.transpose() << "\n";
    }

    std::ofstream mpc_pred_com_pos_file("/tmp/mpc_pred_com_pos.txt");
    for (auto& v : mpc_pred_com_pos_log_) {
        mpc_pred_com_pos_file << v.transpose() << "\n";
    }

    std::ofstream mpc_pred_com_vel_file("/tmp/mpc_pred_com_vel.txt");
    for (auto& v : mpc_pred_com_vel_log_) {
        mpc_pred_com_vel_file << v.transpose() << "\n";
    }

    std::ofstream mpc_pred_zmp_pos_file("/tmp/mpc_pred_zmp_pos.txt");
    for (auto& v : mpc_pred_zmp_pos_log_) {
        mpc_pred_zmp_pos_file << v.transpose() << "\n";
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

    std::ofstream input_torque_file("/tmp/input_torque.txt");
    for (auto& v : input_torque_log_) {
        input_torque_file << v.transpose() << "\n";
    }

    std::ofstream torso_orientation_file("/tmp/torso_orientation.txt");
    for (auto& v : torso_orientation_log_) {
        torso_orientation_file << v.transpose() << "\n";
    }

    std::ofstream torso_angular_velocity_file("/tmp/torso_angular_velocity.txt");
    for (auto& v : torso_angular_velocity_log_) {
        torso_angular_velocity_file << v.transpose() << "\n";
    }

    std::ofstream des_torso_orientation_file("/tmp/des_torso_orientation.txt");
    for (auto& v : des_torso_orientation_log_) {
        des_torso_orientation_file << v.transpose() << "\n";
    }

    std::ofstream des_torso_angular_velocity_file("/tmp/des_torso_angular_velocity.txt");
    for (auto& v : des_torso_angular_velocity_log_) {
        des_torso_angular_velocity_file << v.transpose() << "\n";
    }

    std::ofstream parameters_log_file("/tmp/parameters_log.txt");
    for (auto& param : parameters_log_) {
        parameters_log_file << param << "\n";
    }

    std::ofstream mpc_zmp_velocity_log_file("/tmp/mpc_zmp_velocity_log.txt");
    for (auto& v : mpc_zmp_velocity_log_) {
        mpc_zmp_velocity_log_file << v.transpose() << "\n";
    }

    std::ofstream con_zmp_velocity_log_file("/tmp/con_zmp_velocity_log.txt");
    for (auto& v : con_zmp_velocity_log_) {
        con_zmp_velocity_log_file << v.transpose() << "\n";
    }
}


int64_t
WalkingManager::get_controller_frequency() const {
  return controller_frequency_;
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
