#include <WalkingManager.hpp>

// STL
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
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

#include <GaitConfiguration.hpp>
#include <JointCommand.hpp>
#include <TimingLaw.hpp>
#include <utils.hpp>

#include <globals.h>

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

    for (const char* name : {
        "fb_com_position",  "fb_com_velocity",  "fb_zmp_position",
        "kf_com_position",  "kf_com_velocity",  "kf_zmp_position",
        "des_com_position", "des_com_velocity", "des_zmp_position", "des_com_acceleration",
        "ef_zmp_position",
        "p_lsole_fb", "p_rsole_fb", "v_lsole_fb", "v_rsole_fb",
        "p_lsole_des", "p_rsole_des", "v_lsole_des", "v_rsole_des",
        "fb_lsole_orientation",  "fb_rsole_orientation",
        "des_lsole_orientation", "des_rsole_orientation",
        "estimated_force_lsole", "estimated_force_rsole",
        "wbc_accelerations", "angular_momentum", "input_torque",
        "odometry_base_position",       "odometry_base_velocity",
        "odometry_imu_orientation",     "odometry_imu_orientation_rpy",
        "measured_imu_orientation",     "measured_imu_orientation_rpy",
        "measured_imu_angular_velocity","measured_imu_accelerometer",
        "measured_joint_position",      "measured_joint_velocity",
        "ekf_base_position",      "ekf_base_velocity",
        "ekf_base_orientation",   "ekf_base_orientation_rpy", "ekf_base_angular_velocity",
        "ekf_imu_orientation",    "ekf_imu_orientation_rpy",  "ekf_imu_angular_velocity",
        "ekf_joint_position",     "ekf_joint_velocity",
        "mpc_pred_com_pos", "mpc_pred_com_vel", "mpc_pred_zmp_pos",
        "mpc_zmp_velocity", "con_zmp_velocity",
        "torso_orientation",     "torso_angular_velocity",
        "des_torso_orientation", "des_torso_angular_velocity"
    }) { logger_.reserve(name, max_steps); }

    for (const char* name : {
        "execution_time_wbc", "execution_time_mpc",
        "execution_time_ekf", "execution_time_kf", "execution_time_update"
    }) { logger_.reserveScalar(name, max_steps); }

    // MPC per-solve snapshots saved at fixed 10 Hz (every 100 ms), independent of horizon.
    constexpr int64_t kMpcSnapshotPeriodMs = 100;
    int64_t max_mpc_solves = max_steps * 2 / kMpcSnapshotPeriodMs;
    mpc_snapshot_t_log_.reserve(max_mpc_solves);
    mpc_snapshot_x_log_.reserve(max_mpc_solves);
    mpc_snapshot_u_log_.reserve(max_mpc_solves);

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

    fb_robot_state = initial_robot_state;

    fixed_com_pos = Eigen::Vector3d::Zero();
    fixed_com_vel = Eigen::Vector3d::Zero();
    fixed_zmp_pos = Eigen::Vector3d::Zero();

    pinocchio::forwardKinematics(robot_model, fb_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, fb_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, fb_robot_data, q_init);


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
    p_CoM_init = p_CoM_sim;
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

    simple_ekf_ptr_ = std::make_unique<labrob::SimpleEKF>(
        robot_model, q_init, 0.001 * controller_timestep_msec_);

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
            measured_joint_position(i) = sim_robot_state.joint_state.at(joint_name).pos;
            measured_joint_velocity(i) = sim_robot_state.joint_state.at(joint_name).vel;
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

    // FILL ACTUAL OUTPUT VECTOR FOR SIMPLE EKF

    Eigen::VectorXd actual_output = Eigen::VectorXd::Zero(6 + njnt);
    actual_output.head(3) = odometry_base_position;
    actual_output.segment(3, 3) = rotVecFromQuaternion(Eigen::Quaterniond(
        odometry_imu_quaternion[0], odometry_imu_quaternion[1], odometry_imu_quaternion[2], odometry_imu_quaternion[3]
    ));
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex)njnt; ++joint_id)
        actual_output(6 + joint_id) = measured_joint_position(joint_id);

    ////////////////////////
    // FILTER CALLS
    ////////////////////////

    bool left_support_check  = true;
    bool right_support_check = true;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT)
            right_support_check = false;
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT)
            left_support_check = false;
    }
    const std::array<bool,2> contact = {left_support_check, right_support_check};

    auto start_ekf = std::chrono::high_resolution_clock::now();

    // JointKF — always active, lightweight pre-filter that feeds IMU-based filters
    {
        Eigen::VectorXd input1(measured_joint_position.size() + measured_imu_angular_velocity.size());
        input1 << measured_joint_position, measured_imu_angular_velocity;
        Eigen::VectorXd input2(njnt + 3);
        input2 << whole_body_controller_ptr_->get_q_ddot().tail(njnt),
                  whole_body_controller_ptr_->get_q_ddot().segment(3, 3);
        joint_kf_ptr_->filter(input1, input2);
    }

    // One-time initialization of IMU-based filters at t=1000 ms
    if (t_msec_ == 1000) {
        const Eigen::VectorXd q_joints = joint_kf_ptr_->getFilteredJointPositions();
        Eigen::VectorXd q_full(njnt + 7);
        q_full.head(7)  = q.head(7);
        q_full.tail(njnt) = q_joints;

        if (use_base_ekf_)
            base_ekf_ptr_->initialize(q_full, T_lsole_sim.translation(), T_rsole_sim.translation());
        if (use_ri_ekf_) {
            ri_ekf_ptr_->addContact(0, q_joints);
            ri_ekf_ptr_->addContact(1, q_joints);
        }
        if (use_diligent_kio_) {
            diligent_kio_ptr_->addContact(0, q_joints);
            diligent_kio_ptr_->addContact(1, q_joints);
        }
    }

    // Run active filters and select primary output for control
    if (t_msec_ >= startTimeEKF && isEKFactive) {
        input_acc  = measured_imu_accelerometer;
        input_gyro = measured_imu_angular_velocity;

        if (use_simple_ekf_)
            simple_ekf_ptr_->filter(actual_output, whole_body_controller_ptr_->get_q_ddot());

        if (use_base_ekf_ && t_msec_ >= 1000)
            base_ekf_ptr_->filter(input_acc, input_gyro,
                joint_kf_ptr_->getFilteredJointPositions(),
                joint_kf_ptr_->getFilteredJointVelocities(),
                whole_body_controller_ptr_->get_q_ddot().head(6),
                left_support_check, right_support_check);

        if (use_ri_ekf_ && t_msec_ >= 1000)
            ri_ekf_ptr_->filter(input_gyro, input_acc,
                joint_kf_ptr_->getFilteredJointPositions(),
                joint_kf_ptr_->getFilteredJointVelocities(), contact);

        if (use_diligent_kio_ && t_msec_ >= 1000)
            diligent_kio_ptr_->filter(input_gyro, input_acc,
                joint_kf_ptr_->getFilteredJointPositions(),
                joint_kf_ptr_->getFilteredJointVelocities(), contact);

        // Fill joints from JointKF — used by all IMU-based filters
        auto fill_joints_from_kf = [&]() {
            for (pinocchio::JointIndex id = 0; id < (pinocchio::JointIndex)njnt; ++id) {
                const std::string& name = robot_model.names[id + 2];
                fb_robot_state.joint_state[name].pos = joint_kf_ptr_->getFilteredJointPositions()(id);
                fb_robot_state.joint_state[name].vel = joint_kf_ptr_->getFilteredJointVelocities()(id);
            }
        };

        // Primary filter for WBC control (priority: simple > ri > diligent > base)
        if (use_simple_ekf_) {
            fb_robot_state = simple_ekf_ptr_->getState();
        } else if (use_ri_ekf_ && t_msec_ >= 1000) {
            fb_robot_state.position         = ri_ekf_ptr_->getPosition();
            fb_robot_state.orientation      = ri_ekf_ptr_->getQuaternion();
            fb_robot_state.linear_velocity  = ri_ekf_ptr_->getVelocity();
            fb_robot_state.angular_velocity = ri_ekf_ptr_->getOmegaBody();
            fill_joints_from_kf();
        } else if (use_diligent_kio_ && t_msec_ >= 1000) {
            fb_robot_state.position         = diligent_kio_ptr_->getPosition();
            fb_robot_state.orientation      = diligent_kio_ptr_->getQuaternion();
            fb_robot_state.linear_velocity  = diligent_kio_ptr_->getVelocity();
            fb_robot_state.angular_velocity = diligent_kio_ptr_->getOmegaBody();
            fill_joints_from_kf();
        } else if (use_base_ekf_ && t_msec_ >= 1000) {
            fb_robot_state.orientation      = base_ekf_ptr_->getBaseOrientation();
            fb_robot_state.position         = base_ekf_ptr_->getBasePosition();
            fb_robot_state.linear_velocity  =
                fb_robot_state.orientation.toRotationMatrix().transpose()
                * base_ekf_ptr_->getBaseVelocity();
            fb_robot_state.angular_velocity = base_ekf_ptr_->getBaseOmega();
            fill_joints_from_kf();
        }
    } else {
        fb_robot_state = sim_robot_state;
    }
    auto end_ekf = std::chrono::high_resolution_clock::now();

    ////////////////////
    // END FILTER CALLS
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

    // if (t_msec_ == 5000){
    //     walking_data_.swapStanding(
    //         labrob::SE3(T_lsole_fb.rotation(), Eigen::Vector3d(T_lsole_fb.translation().x(), T_lsole_fb.translation().y(), (T_lsole_fb.translation().z() + T_lsole_fb.translation().z())/2)),
    //         labrob::SE3(T_rsole_fb.rotation(), Eigen::Vector3d(T_rsole_fb.translation().x(), T_rsole_fb.translation().y(), (T_rsole_fb.translation().z() + T_rsole_fb.translation().z())/2))
    //     );
    // }

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

    logger_.log("ef_zmp_position", zmp_3d_fb);

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
            integrated_state_pos(6 + joint_id) = sim_robot_state.joint_state.at(joint_name).pos;
            integrated_state_vel(6 + joint_id) = sim_robot_state.joint_state.at(joint_name).vel;
        }
    }

    // ADD STEPS FOR SIMULATION

    if(!useRobot && t_msec_ == 1000 && false){
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

    auto start_mpc = std::chrono::system_clock::now();

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            if(isMPCLoopClosed && t_msec_>= startTimeMPCCL){
                ismpc_ptr_->solve(t_msec_, walking_data_, kf_LipState);
                des_LipState = discrete_lip_dynamics_ptr_->integrate(
                    kf_LipState,
                    ismpc_ptr_->getInput()
                );
            }
            else{
                ismpc_ptr_->solve(t_msec_, walking_data_, des_LipState);
                des_LipState = discrete_lip_dynamics_ptr_->integrate(
                    des_LipState,
                    ismpc_ptr_->getInput()
                );
            }
        }
        #pragma omp section
        {
        }
    }

    logger_.log("mpc_zmp_velocity", ismpc_ptr_->getInput());

        // LOG MPC PREDICTIONS FOR GIF FILES
        Eigen::VectorXd inputSequenceX = ismpc_ptr_->getInputSequenceX();
        Eigen::VectorXd inputSequenceY = ismpc_ptr_->getInputSequenceY();
        Eigen::VectorXd inputSequenceZ = ismpc_ptr_->getInputSequenceZ();
        LIPState LipState_mpc = mpc_LipState_prec;
        // for (int i = 0; i < 20; ++i) {
        //     LipState_mpc = discrete_lip_dynamics_ptr_mpc_->integrate(
        //         LipState_mpc,
        //         Eigen::Vector3d(inputSequenceX(i), inputSequenceY(i), inputSequenceZ(i))
        //     );
        //     mpc_pred_com_pos_log_.push_back(LipState_mpc.com_pos_);
        //     mpc_pred_com_vel_log_.push_back(LipState_mpc.com_vel_);
        //     mpc_pred_zmp_pos_log_.push_back(LipState_mpc.zmp_pos_);
        // }

        // Buffer per-solve MPC snapshot in memory; written to disk in saveLogs().
        constexpr int64_t kMpcSnapshotPeriodMs = 100;
        if (t_msec_ % kMpcSnapshotPeriodMs == 0) {
            const int N_log = static_cast<int>(inputSequenceX.size());

            std::vector<Eigen::VectorXd> x_rows;
            x_rows.reserve(N_log + 1);
            Eigen::VectorXd row0(9);
            row0 << mpc_LipState_prec.com_pos_, mpc_LipState_prec.com_vel_, mpc_LipState_prec.zmp_pos_;
            x_rows.push_back(row0);
            LIPState pred = mpc_LipState_prec;
            for (int i = 0; i < N_log; ++i) {
                pred = discrete_lip_dynamics_ptr_mpc_->integrate(
                    pred, Eigen::Vector3d(inputSequenceX(i), inputSequenceY(i), inputSequenceZ(i)));
                Eigen::VectorXd row(9);
                row << pred.com_pos_, pred.com_vel_, pred.zmp_pos_;
                x_rows.push_back(row);
            }

            std::vector<Eigen::VectorXd> u_rows;
            u_rows.reserve(N_log);
            for (int i = 0; i < N_log; ++i) {
                Eigen::VectorXd row(3);
                row << inputSequenceX(i), inputSequenceY(i), inputSequenceZ(i);
                u_rows.push_back(row);
            }

            mpc_snapshot_t_log_.push_back(t_msec_);
            mpc_snapshot_x_log_.push_back(std::move(x_rows));
            mpc_snapshot_u_log_.push_back(std::move(u_rows));
        }

    auto end_mpc = std::chrono::system_clock::now();

    /////////////////////////////////////
    // MPC FUNCTION CALL END
    /////////////////////////////////////

    // FILL DESIRED GAIT CONFIGURATION

    // joint
    labrob::GaitConfiguration desired_gait_configuration;
    desired_gait_configuration.qjnt = q_jnt_des_;
    desired_gait_configuration.qjntdot = Eigen::VectorXd::Zero(njnt);
    desired_gait_configuration.qjntddot = Eigen::VectorXd::Zero(njnt);

    // simple arm swing policy based on hip pitch variation and step length
    // double left_hip_pitch_current = current_gait_configuration.qjnt(robot_model.getJointId("left_hip_pitch_joint") - 2);
    // double left_hip_pitch_desired = desired_gait_configuration.qjnt(robot_model.getJointId("left_hip_pitch_joint") - 2);
    // double left_hip_pitch_variation = left_hip_pitch_desired - left_hip_pitch_current;
    // double right_hip_pitch_current = current_gait_configuration.qjnt(robot_model.getJointId("right_hip_pitch_joint") - 2);
    // double right_hip_pitch_desired = desired_gait_configuration.qjnt(robot_model.getJointId("right_hip_pitch_joint") - 2);
    // double right_hip_pitch_variation = right_hip_pitch_desired - right_hip_pitch_current;
    // double arm_swing_amplitude = 3; // adjust this value to change the amount of arm swing
    // // compute difference bewteen foot poses in x direction to adjust arm swing amplitude based on step length
    // double left_foot_x = walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().p.x();
    // double right_foot_x = walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().p.x();
    // double step_length = left_foot_x - right_foot_x;
    // arm_swing_amplitude *= step_length / 0.3; // assuming 0.3m is the nominal step length

    // ARMS SWING, UNCOMMENT TO ENABLE

    // desired_gait_configuration.qjnt(robot_model.getJointId("left_shoulder_pitch_joint") - 2) +=  1 * left_hip_pitch_variation - 1 * right_hip_pitch_variation;
    // desired_gait_configuration.qjnt(robot_model.getJointId("right_shoulder_pitch_joint") - 2) += - 1 * left_hip_pitch_variation + 1 * right_hip_pitch_variation;

    // desired_gait_configuration.qjnt(robot_model.getJointId("left_shoulder_pitch_joint") - 2) += - 2 * step_length;
    // desired_gait_configuration.qjnt(robot_model.getJointId("right_shoulder_pitch_joint") - 2) += 2 * step_length;

    // desired_gait_configuration.qjnt(robot_model.getJointId("left_elbow_joint") - 2) += 1 * left_hip_pitch_variation - 1 * right_hip_pitch_variation;
    // desired_gait_configuration.qjnt(robot_model.getJointId("right_elbow_joint") - 2) += - 1 * left_hip_pitch_variation + 1 * right_hip_pitch_variation;

    // desired_gait_configuration.qjnt(robot_model.getJointId("left_elbow_joint") - 2) += - 1 * step_length;
    // desired_gait_configuration.qjnt(robot_model.getJointId("right_elbow_joint") - 2) += 1 * step_length;


    // CoM desired from IS-MPC LIP integration
    desired_gait_configuration.com.pos = des_LipState.com_pos_;
    desired_gait_configuration.com.vel = des_LipState.com_vel_;
    desired_gait_configuration.com.acc = eta2 * (des_LipState.com_pos_ - des_LipState.zmp_pos_)
                                       - Eigen::Vector3d(0.0, 0.0, 9.81);

    // assign constant value to com
    if(useRobot){
        if(isWBCLoopClosed && t_msec_ >= startTimeWBCCL){
            desired_gait_configuration.com.pos = fixed_com_pos;
        } else {
            desired_gait_configuration.com.pos = p_CoM_init;
        }
    } else {
        desired_gait_configuration.com.pos = Eigen::Vector3d(0.03, 0.0, 0.65);
    }
    desired_gait_configuration.com.vel = Eigen::Vector3d::Zero();
    desired_gait_configuration.com.acc = Eigen::Vector3d::Zero();

    // contact flags
    desired_gait_configuration.is_left_foot_support  = current_gait_configuration.is_left_foot_support;
    desired_gait_configuration.is_right_foot_support = current_gait_configuration.is_right_foot_support;

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
                    desired_gait_configuration
                );
            } else {
                joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
                    robot_model,
                    sim_robot_state,
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

    /////////////////////////////////////
    // END WHOLE BODY CONTROLLER FUNCTION CALL
    /////////////////////////////////////

    estimated_force = Eigen::VectorXd::Zero(6);

    // Update timing in milliseconds.
    // NOTE: assuming update() is actually called every controller_timestep_msec_
    //       milliseconds.
    t_msec_ += controller_timestep_msec_;
    prev_angular_momentum_ = angular_momentum;

    // LOGS

    logger_.log("fb_com_position",  p_CoM_fb);
    logger_.log("kf_com_position",  kf_LipState.com_pos_);
    logger_.log("des_com_position", desired_gait_configuration.com.pos);

    logger_.log("fb_com_velocity",  v_CoM_fb);
    logger_.log("kf_com_velocity",  kf_LipState.com_vel_);
    logger_.log("des_com_velocity", desired_gait_configuration.com.vel);

    logger_.log("fb_zmp_position",  zmp_3d_fb);
    logger_.log("kf_zmp_position",  kf_LipState.zmp_pos_);
    logger_.log("des_zmp_position", des_LipState.zmp_pos_);

    logger_.log("des_com_acceleration", desired_gait_configuration.com.acc);

    logger_.log("p_lsole_fb", T_lsole_fb.translation());
    logger_.log("p_rsole_fb", T_rsole_fb.translation());
    logger_.log("v_lsole_fb", v_lsole_fb.head<3>());
    logger_.log("v_rsole_fb", v_rsole_fb.head<3>());
    logger_.log("p_lsole_des", desired_gait_configuration.lsole.pos.p);
    logger_.log("p_rsole_des", desired_gait_configuration.rsole.pos.p);
    logger_.log("v_lsole_des", desired_gait_configuration.lsole.vel.head<3>());
    logger_.log("v_rsole_des", desired_gait_configuration.rsole.vel.head<3>());

    logger_.log("fb_lsole_orientation",  T_lsole_fb.rotation().eulerAngles(0,1,2));
    logger_.log("fb_rsole_orientation",  T_rsole_fb.rotation().eulerAngles(0,1,2));
    logger_.log("des_lsole_orientation", desired_gait_configuration.lsole.pos.R.eulerAngles(0,1,2));
    logger_.log("des_rsole_orientation", desired_gait_configuration.rsole.pos.R.eulerAngles(0,1,2));

    logger_.log("estimated_force_lsole", estimated_force.head<3>());
    logger_.log("estimated_force_rsole", estimated_force.tail<3>());
    logger_.log("wbc_accelerations",     whole_body_controller_ptr_->get_q_ddot());
    logger_.log("angular_momentum",      angular_momentum);

    logger_.log("odometry_base_position",        odometry_base_position);
    logger_.log("odometry_base_velocity",        odometry_base_velocity);
    logger_.log("odometry_imu_orientation",      odometry_imu_quaternion);
    logger_.log("odometry_imu_orientation_rpy",  odometry_imu_rpy);
    logger_.log("measured_imu_orientation",      measured_imu_quaternion);
    logger_.log("measured_imu_orientation_rpy",  measured_imu_rpy);
    logger_.log("measured_imu_angular_velocity", measured_imu_angular_velocity);
    logger_.log("measured_imu_accelerometer",    measured_imu_accelerometer);

    {
        Eigen::VectorXd jp(njnt), jv(njnt);
        for (pinocchio::JointIndex id = 0; id < (pinocchio::JointIndex)njnt; ++id) {
            jp(id) = measured_joint_position(id);
            jv(id) = measured_joint_velocity(id);
        }
        logger_.log("measured_joint_position", jp);
        logger_.log("measured_joint_velocity", jv);
    }

    logger_.log("ekf_base_position",         fb_robot_state.position);
    logger_.log("ekf_base_velocity",         fb_robot_state.linear_velocity);
    logger_.log("ekf_base_orientation",      Eigen::Vector4d(
        fb_robot_state.orientation.w(), fb_robot_state.orientation.x(),
        fb_robot_state.orientation.y(), fb_robot_state.orientation.z()));
    logger_.log("ekf_base_orientation_rpy",  rpyFromQuaternion(fb_robot_state.orientation));
    logger_.log("ekf_base_angular_velocity", fb_robot_state.angular_velocity);

    {
        const Eigen::Quaterniond q_imu(fb_robot_data.oMf[imu_idx_].rotation());
        logger_.log("ekf_imu_orientation",     Eigen::Vector4d(q_imu.w(), q_imu.x(), q_imu.y(), q_imu.z()));
        logger_.log("ekf_imu_orientation_rpy", rpyFromQuaternion(q_imu));
    }
    logger_.log("ekf_imu_angular_velocity", fb_robot_state.angular_velocity);

    {
        Eigen::VectorXd jp(njnt), jv(njnt);
        for (pinocchio::JointIndex id = 0; id < (pinocchio::JointIndex)njnt; ++id) {
            const std::string& jname = robot_model.names[id + 2];
            jp(id) = fb_robot_state.joint_state[jname].pos;
            jv(id) = fb_robot_state.joint_state[jname].vel;
        }
        logger_.log("ekf_joint_position", jp);
        logger_.log("ekf_joint_velocity", jv);
    }

    {
        Eigen::VectorXd tau(njnt);
        for (pinocchio::JointIndex id = 0; id < (pinocchio::JointIndex)njnt; ++id)
            tau(id) = joint_command[robot_model.names[id + 2]];
        logger_.log("input_torque", tau);
    }

    logger_.log("torso_orientation",          rpyFromQuaternion(Eigen::Quaterniond(current_gait_configuration.torso.pos)));
    logger_.log("torso_angular_velocity",     current_gait_configuration.torso.vel.tail<3>());
    logger_.log("des_torso_orientation",      rpyFromQuaternion(Eigen::Quaterniond(desired_gait_configuration.torso.pos)));
    logger_.log("des_torso_angular_velocity", desired_gait_configuration.torso.vel.tail<3>());

    auto end_update = std::chrono::high_resolution_clock::now();

    auto update_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_update - start_update).count();
    auto ekf_duration    = std::chrono::duration_cast<std::chrono::microseconds>(end_ekf   - start_ekf).count();
    auto kf_duration     = std::chrono::duration_cast<std::chrono::microseconds>(end_kf    - start_kf).count();
    auto mpc_duration    = std::chrono::duration_cast<std::chrono::microseconds>(end_mpc   - start_mpc).count();
    auto wbc_duration    = std::chrono::duration_cast<std::chrono::microseconds>(end_wbc   - start_wbc).count();

    logger_.log("execution_time_update", static_cast<double>(update_duration));
    logger_.log("execution_time_ekf",    static_cast<double>(ekf_duration));
    logger_.log("execution_time_kf",     static_cast<double>(kf_duration));
    logger_.log("execution_time_mpc",    static_cast<double>(mpc_duration));
    logger_.log("execution_time_wbc",    static_cast<double>(wbc_duration));
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
    std::filesystem::remove_all("/tmp/robot_logs");
    std::filesystem::create_directories("/tmp/robot_logs");

    std::ofstream joint_names_file("/tmp/robot_logs/joint_names.txt");
    for (pinocchio::JointIndex id = 0; id < (pinocchio::JointIndex)njnt; ++id)
        joint_names_file << robot_model.names[id + 2] << "\n";

    logger_.save("/tmp/robot_logs");

    // MPC per-solve snapshots: nested structure, written to individual subdirectories.
    constexpr const char* mpc_data_dir = "/tmp/mpc_data";
    std::filesystem::remove_all(mpc_data_dir);
    std::filesystem::create_directories(mpc_data_dir);
    for (std::size_t k = 0; k < mpc_snapshot_t_log_.size(); ++k) {
        const std::string subdir = std::string(mpc_data_dir) + "/" + std::to_string(mpc_snapshot_t_log_[k]);
        std::filesystem::create_directories(subdir);
        {
            std::ofstream fx(subdir + "/x.txt");
            for (const auto& row : mpc_snapshot_x_log_[k])
                fx << row.transpose() << "\n";
        }
        {
            std::ofstream fu(subdir + "/u.txt");
            for (const auto& row : mpc_snapshot_u_log_[k])
                fu << row.transpose() << "\n";
        }
    }

    std::ofstream parameters_log_file("/tmp/robot_logs/parameters_log.txt");
    for (double param : parameters_log_)
        parameters_log_file << param << "\n";
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
