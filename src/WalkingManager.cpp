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

LIPState
WalkingManager::com_kf_step(LIPState filtered, LIPState current,
                             const Eigen::Vector3d& input)
{
    const double eta = std::sqrt(eta2);
    const double dt  = 0.001 * controller_timestep_msec_;
    const double ch  = std::cosh(eta * dt);
    const double sh  = std::sinh(eta * dt);

    Eigen::Matrix3d A;
    A << ch, sh/eta, 1-ch, eta*sh, ch, -eta*sh, 0, 0, 1;
    Eigen::Vector3d B(dt - sh/eta, 1 - ch, dt);

    Eigen::Matrix3d R_kf = Eigen::Matrix3d::Identity();
    R_kf.diagonal() << kComKfMeasPos, kComKfMeasVel, kComKfMeasZmp;
    Eigen::Matrix3d Q_kf = Eigen::Matrix3d::Identity();
    Q_kf.diagonal() << kComKfModPos, kComKfModVel, kComKfModZmp;

    auto kf_axis = [&](Eigen::Matrix3d& cov, const Eigen::Vector3d& x_est,
                       const Eigen::Vector3d& z, double u,
                       const Eigen::Vector3d& z_fallback) -> Eigen::Vector3d {
        const Eigen::Vector3d& meas = std::isnan(z(0)) ? z_fallback : z;
        Eigen::Vector3d x_pred = A * x_est + B * u;
        Eigen::Matrix3d cov_pred = A * cov * A.transpose() + Q_kf;
        Eigen::Matrix3d K = cov_pred * (cov_pred + R_kf).inverse();
        Eigen::Vector3d x_upd = x_pred + K * (meas - x_pred);
        cov = (Eigen::Matrix3d::Identity() - K) * cov_pred * (Eigen::Matrix3d::Identity() - K).transpose()
            + K * R_kf * K.transpose();
        return x_upd;
    };

    bool nan_zmp = std::isnan(current.zmp_pos_(0));
    Eigen::Vector3d x_fb(filtered.com_pos_(0), filtered.com_vel_(0), filtered.zmp_pos_(0));
    Eigen::Vector3d y_fb(filtered.com_pos_(1), filtered.com_vel_(1), filtered.zmp_pos_(1));
    Eigen::Vector3d z_fb(filtered.com_pos_(2), filtered.com_vel_(2), filtered.zmp_pos_(2));

    Eigen::Vector3d xz(current.com_pos_(0), current.com_vel_(0), current.zmp_pos_(0));
    Eigen::Vector3d yz(current.com_pos_(1), current.com_vel_(1), current.zmp_pos_(1));
    Eigen::Vector3d zz(current.com_pos_(2), current.com_vel_(2), current.zmp_pos_(2));

    auto xe = kf_axis(com_kf_cov_x_, x_fb, xz, input.x(), x_fb);
    auto ye = kf_axis(com_kf_cov_y_, y_fb, yz, input.y(), y_fb);
    // z-axis: add gravity term to prediction (z_pred += [0, -9.81*dt, 0])
    {
        Eigen::Vector3d x_pred = A * z_fb + B * input.z();
        x_pred(1) -= 9.81 * dt;
        Eigen::Matrix3d cov_pred = A * com_kf_cov_z_ * A.transpose() + Q_kf;
        Eigen::Matrix3d K = cov_pred * (cov_pred + R_kf).inverse();
        const Eigen::Vector3d& meas = nan_zmp ? z_fb : zz;
        zz = x_pred + K * (meas - x_pred);
        com_kf_cov_z_ = (Eigen::Matrix3d::Identity() - K) * cov_pred * (Eigen::Matrix3d::Identity() - K).transpose()
            + K * R_kf * K.transpose();
    }

    current.com_pos_ = Eigen::Vector3d(xe(0), ye(0), zz(0));
    current.com_vel_ = Eigen::Vector3d(xe(1), ye(1), zz(1));
    current.zmp_pos_ = Eigen::Vector3d(xe(2), ye(2), zz(2));
    return current;
}

std::array<bool,2>
WalkingManager::get_contact() const
{
    bool left  = true;
    bool right = true;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
        auto foot = walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot();
        if (foot == Foot::LEFT)  right = false;
        if (foot == Foot::RIGHT) left  = false;
    }
    return {left, right};
}

//INIT FUNCTION START

bool
WalkingManager::init(const labrob::RobotState& initial_robot_state,
                     std::map<std::string, double> &armatures) {

    //PRE-ALLOCATIONS FOR LOGS

    int64_t max_steps = 50000;

    for (const char* name : {
        "com_position",  "com_velocity",  "zmp_position",
        "kf_com_position",  "kf_com_velocity",  "kf_zmp_position",
        "des_com_position", "des_com_velocity", "des_zmp_position", "des_com_acceleration",
        "ef_zmp_position",
        "p_lsole", "p_rsole", "v_lsole", "v_rsole",
        "p_lsole_des", "p_rsole_des", "v_lsole_des", "v_rsole_des",
        "lsole_orientation",  "rsole_orientation",
        "des_lsole_orientation", "des_rsole_orientation",
        "estimated_force_lsole", "estimated_force_rsole",
        "wbc_accelerations", "angular_momentum", "input_torque",
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
    robot_data = pinocchio::Data(robot_model);

    mass = pinocchio::computeTotalMass(robot_model);
    njnt = robot_model.nv - 6;

    // Init desired lsole and rsole poses:
    auto q_init = robot_state_to_pinocchio_joint_configuration(
        robot_model,
        initial_robot_state
    );

    // INIT ROBOT STATE, DATA AND PINOCCHIO QUANTITIES

    pinocchio::forwardKinematics(robot_model, robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, robot_data, q_init);
    pinocchio::centerOfMass(robot_model, robot_data, q_init, false);


    fixed_com_pos = Eigen::Vector3d::Zero();
    fixed_com_vel = Eigen::Vector3d::Zero();
    fixed_zmp_pos = Eigen::Vector3d::Zero();


    // GET INDICES OF INTEREST AND ARMATURES

    lsole_idx_ = robot_model.getFrameId("left_foot_link");
    rsole_idx_ = robot_model.getFrameId("right_foot_link");
    torso_idx_ = robot_model.getFrameId("torso_link");
    pelvis_idx_ = robot_model.getFrameId("pelvis");
    imu_idx_ = robot_model.getFrameId("imu_in_pelvis");
    const auto& T_lsole_init = robot_data.oMf[lsole_idx_];
    const auto& T_rsole_init = robot_data.oMf[rsole_idx_];

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


    // Save and read again footstep plan to double check it's working:
    //std::string footstep_plan_path = "/tmp/ditch-footstep-plan-argos.txt";
    //labrob::saveFootstepPlan(walking_data_.footstep_plan, footstep_plan_path);
    //labrob::readFootstepPlan(footstep_plan_path, walking_data_.footstep_plan);
    //labrob::readArgosFootstepPlan(footstep_plan_path, walking_data_.footstep_plan);

    // INIT LIP MODEL, IS-MPC, WHOLE-BODY CONTROLLER AND DISCRETE LIP DYNAMICS

    Eigen::Vector3d p_CoM = robot_data.com[0];
    p_CoM_init = p_CoM;
    double com_target_height = p_CoM.z() - T_lsole_init.translation().z();
    eta2 = 9.81 / com_target_height;
    Eigen::Vector3d p_ZMP = p_CoM - Eigen::Vector3d(0.0, 0.0, com_target_height);
    kf_LipState = labrob::LIPState(
        p_CoM,
        Eigen::Vector3d::Zero(),
        p_ZMP
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

    // CoMKF covariance matrices are reset to identity (default) at construction.
    com_kf_cov_x_ = Eigen::Matrix3d::Identity();
    com_kf_cov_y_ = Eigen::Matrix3d::Identity();
    com_kf_cov_z_ = Eigen::Matrix3d::Identity();

    return true;
}

// UPDATE FUNCTION START

void
WalkingManager::update(
    labrob::RobotState& robot_state,
    labrob::JointCommand& joint_command
) {

    auto start_update = std::chrono::high_resolution_clock::now();

    // SET FORCE ESTIMATION

    Eigen::Vector3d left_foot_force = estimated_force.head(3);
    Eigen::Vector3d right_foot_force = estimated_force.tail(3);
    Eigen::Vector3d total_force = left_foot_force + right_foot_force;

    // UPDATE FORWARD KINEMATICS, LIP AND PINOCCHIO QUANTITIES

    auto q = robot_state.get_pinocchio_joint_configuration(robot_model);
    auto qdot = robot_state.get_pinocchio_joint_velocity(robot_model);

    pinocchio::forwardKinematics(robot_model, robot_data, q);

    pinocchio::jacobianCenterOfMass(robot_model, robot_data, q);
    pinocchio::computeJointJacobiansTimeVariation(robot_model, robot_data, q, qdot);
    pinocchio::framesForwardKinematics(robot_model, robot_data, q);
    pinocchio::centerOfMass(robot_model, robot_data, q, qdot, 0.0 * qdot); // This is used to compute the CoM drift (J_com_dot * qdot)
    const auto& centroidal_momentum_matrix = pinocchio::ccrba(
        robot_model,
        robot_data,
        q,
        qdot
    );

    auto angular_momentum = (centroidal_momentum_matrix * qdot).tail<3>();

    const auto& T_torso = robot_data.oMf[torso_idx_];
    auto torso_orientation = T_torso.rotation();
    Eigen::MatrixXd J_torso = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        robot_data,
        torso_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_torso
    );

    const auto& T_pelvis = robot_data.oMf[pelvis_idx_];
    auto pelvis_orientation = T_pelvis.rotation();
    Eigen::MatrixXd J_pelvis = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        robot_data,
        pelvis_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_pelvis
    );

    const auto& T_lsole = robot_data.oMf[lsole_idx_];
    Eigen::MatrixXd J_lsole = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        robot_data,
        lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_lsole
    );

    const auto& v_lsole = J_lsole * qdot;

    const auto& T_rsole = robot_data.oMf[rsole_idx_];
    Eigen::MatrixXd J_rsole = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        robot_data,
        rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_rsole
    );
    const auto& v_rsole = J_rsole * qdot;

    Eigen::MatrixXd J_imu = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        robot_data,
        imu_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_imu
    );

    const auto& p_CoM = robot_data.com[0];
    const auto& J_CoM = robot_data.Jcom;
    const auto& a_CoM_drift = robot_data.acom[0];
    Eigen::Vector3d v_CoM = J_CoM * qdot;
    Eigen::Vector3d zmp_3d;
    // zmp_3d.z() = robot_state.position(2) - robot_state.total_force.z() / (mass * eta2);
    // zmp_3d.x() = 0.0;
    // zmp_3d.y() = 0.0;
    // for (int i = 0; i < robot_state.contact_points.size(); ++i) {
    //     auto &pi = robot_state.contact_points[i];
    //     auto &fi = robot_state.contact_forces[i];
    //     zmp_3d.x() += (pi.x() * fi.z() / robot_state.total_force.z() + (zmp_3d.z() - pi.z()) * fi.x() / robot_state.total_force.z());
    //     zmp_3d.y() += (pi.y() * fi.z() / robot_state.total_force.z() + (zmp_3d.z() - pi.z()) * fi.y() / robot_state.total_force.z());
    // }
    zmp_3d.z() = p_CoM.z() - (a_CoM_drift.z() + 9.81) / eta2;
    zmp_3d.x() = p_CoM.x() - a_CoM_drift.x() / eta2;
    zmp_3d.y() = p_CoM.y() - a_CoM_drift.y() / eta2;

    // compute zmp 3d using the 6d vector estimated forces, first three are left foot, second three are right foot
    // zmp_3d.z() = robot_state.position(2) - total_force.z() / (mass * eta2);
    // zmp_3d.x() = 0.0;
    // zmp_3d.y() = 0.0;
    // if (total_force.z() > 1e-5) {
    //     if (left_foot_force.z() > 1e-5) {
    //         zmp_3d.x() += (T_lsole.translation().x() * left_foot_force.z() / total_force.z() + (zmp_3d.z() - T_lsole.translation().z()) * left_foot_force.x() / total_force.z());
    //         zmp_3d.y() += (T_lsole.translation().y() * left_foot_force.z() / total_force.z() + (zmp_3d.z() - T_lsole.translation().z()) * left_foot_force.y() / total_force.z());
    //     }
    //     if (right_foot_force.z() > 1e-5) {
    //         zmp_3d.x() += (T_rsole.translation().x() * right_foot_force.z() / total_force.z() + (zmp_3d.z() - T_rsole.translation().z()) * right_foot_force.x() / total_force.z());
    //         zmp_3d.y() += (T_rsole.translation().y() * right_foot_force.z() / total_force.z() + (zmp_3d.z() - T_rsole.translation().z()) * right_foot_force.y() / total_force.z());
    //     }
    // }

    walking_data_.updateWalkingState(t_msec_);

    /////////////////////////////////////
    // KF FUNCTION CALL
    /////////////////////////////////////

    auto start_kf = std::chrono::high_resolution_clock::now();
    LipState = LIPState(p_CoM, J_CoM * qdot, zmp_3d);
    kf_LipState = com_kf_step(kf_LipState, LipState, ismpc_ptr_->getInput());
    auto end_kf = std::chrono::high_resolution_clock::now();

    ////////////////////////////////////
    // END KF FUNCTION CALL
    ////////////////////////////////////

    // IF STANDING, ADD STEPS TO START WALKING AGAIN OR IF DOUBLE SUPPORT, REMOVE STEPS TO GO BACK TO STANDING

    if (switchWalkingState){
        double yaw_angle = rpyFromQuaternion(Eigen::Quaterniond(robot_data.oMf[imu_idx_].rotation())).z();
        if (walking_data_.getWalkingState() == WalkingState::Standing) {
            walking_data_.addSteps(
                labrob::SE3(T_lsole.rotation(), Eigen::Vector3d(T_lsole.translation().x(), T_lsole.translation().y(), (T_lsole.translation().z() + T_lsole.translation().z())/2)),
                labrob::SE3(T_rsole.rotation(), Eigen::Vector3d(T_rsole.translation().x(), T_rsole.translation().y(), (T_rsole.translation().z() + T_rsole.translation().z())/2)),
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

    current_gait_configuration.qjnt = q.tail(njnt);
    current_gait_configuration.qjntdot = qdot.tail(njnt);

    current_gait_configuration.is_left_foot_support = true;
    current_gait_configuration.is_right_foot_support = true;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
    if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT) current_gait_configuration.is_right_foot_support = false;
    else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT) current_gait_configuration.is_left_foot_support = false;
    }

    current_gait_configuration.com.pos = kf_LipState.com_pos_;
    current_gait_configuration.com.vel = kf_LipState.com_vel_;
    current_gait_configuration.torso.pos = robot_data.oMf[torso_idx_].rotation();
    current_gait_configuration.torso.vel = J_torso.bottomRows<3>() * qdot;
    current_gait_configuration.pelvis.pos = robot_data.oMf[pelvis_idx_].rotation();
    current_gait_configuration.pelvis.vel = J_pelvis.bottomRows<3>() * qdot;
    //feet
    current_gait_configuration.lsole.pos = labrob::SE3(robot_data.oMf[lsole_idx_].rotation(), robot_data.oMf[lsole_idx_].translation());
    current_gait_configuration.lsole.vel = J_lsole * qdot;
    current_gait_configuration.rsole.pos = labrob::SE3(robot_data.oMf[rsole_idx_].rotation(), robot_data.oMf[rsole_idx_].translation());
    current_gait_configuration.rsole.vel = J_rsole * qdot;

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

    // CoM desired from IS-MPC LIP integration
    desired_gait_configuration.com.pos = des_LipState.com_pos_;
    desired_gait_configuration.com.vel = des_LipState.com_vel_;
    desired_gait_configuration.com.acc = eta2 * (des_LipState.com_pos_ - des_LipState.zmp_pos_)
                                       - Eigen::Vector3d(0.0, 0.0, 9.81);

    // assign constant value to com
    // if(useRobot){
    //     if(isWBCLoopClosed && t_msec_ >= startTimeWBCCL){
    //         desired_gait_configuration.com.pos = fixed_com_pos;
    //     } else {
    //         desired_gait_configuration.com.pos = p_CoM_init;
    //     }
    // } else {
    //     desired_gait_configuration.com.pos = Eigen::Vector3d(0.03, 0.0, 0.65);
    // }
    // desired_gait_configuration.com.vel = Eigen::Vector3d::Zero();
    // desired_gait_configuration.com.acc = Eigen::Vector3d::Zero();

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
            joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
                robot_model,
                robot_state,
                robot_data,
                current_gait_configuration,
                desired_gait_configuration
            );
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

    logger_.log("com_position",  p_CoM);
    logger_.log("kf_com_position",  kf_LipState.com_pos_);
    logger_.log("des_com_position", desired_gait_configuration.com.pos);

    logger_.log("com_velocity",  v_CoM);
    logger_.log("kf_com_velocity",  kf_LipState.com_vel_);
    logger_.log("des_com_velocity", desired_gait_configuration.com.vel);

    logger_.log("zmp_position",  zmp_3d);
    logger_.log("kf_zmp_position",  kf_LipState.zmp_pos_);
    logger_.log("des_zmp_position", des_LipState.zmp_pos_);

    logger_.log("des_com_acceleration", desired_gait_configuration.com.acc);

    logger_.log("p_lsole", T_lsole.translation());
    logger_.log("p_rsole", T_rsole.translation());
    logger_.log("v_lsole", v_lsole.head<3>());
    logger_.log("v_rsole", v_rsole.head<3>());
    logger_.log("p_lsole_des", desired_gait_configuration.lsole.pos.p);
    logger_.log("p_rsole_des", desired_gait_configuration.rsole.pos.p);
    logger_.log("v_lsole_des", desired_gait_configuration.lsole.vel.head<3>());
    logger_.log("v_rsole_des", desired_gait_configuration.rsole.vel.head<3>());

    logger_.log("lsole_orientation",  T_lsole.rotation().eulerAngles(0,1,2));
    logger_.log("rsole_orientation",  T_rsole.rotation().eulerAngles(0,1,2));
    logger_.log("des_lsole_orientation", desired_gait_configuration.lsole.pos.R.eulerAngles(0,1,2));
    logger_.log("des_rsole_orientation", desired_gait_configuration.rsole.pos.R.eulerAngles(0,1,2));

    logger_.log("estimated_force_lsole", estimated_force.head<3>());
    logger_.log("estimated_force_rsole", estimated_force.tail<3>());
    logger_.log("wbc_accelerations",     whole_body_controller_ptr_->get_q_ddot());
    logger_.log("angular_momentum",      angular_momentum);

    logger_.log("ekf_base_position",         robot_state.position);
    logger_.log("ekf_base_velocity",         robot_state.linear_velocity);
    logger_.log("ekf_base_orientation",      Eigen::Vector4d(
        robot_state.orientation.w(), robot_state.orientation.x(),
        robot_state.orientation.y(), robot_state.orientation.z()));
    logger_.log("ekf_base_orientation_rpy",  rpyFromQuaternion(robot_state.orientation));
    logger_.log("ekf_base_angular_velocity", robot_state.angular_velocity);

    {
        const Eigen::Quaterniond q_imu(robot_data.oMf[imu_idx_].rotation());
        logger_.log("ekf_imu_orientation",     Eigen::Vector4d(q_imu.w(), q_imu.x(), q_imu.y(), q_imu.z()));
        logger_.log("ekf_imu_orientation_rpy", rpyFromQuaternion(q_imu));
    }
    logger_.log("ekf_imu_angular_velocity", robot_state.angular_velocity);

    {
        Eigen::VectorXd jp(njnt), jv(njnt);
        for (pinocchio::JointIndex id = 0; id < (pinocchio::JointIndex)njnt; ++id) {
            const std::string& jname = robot_model.names[id + 2];
            jp(id) = robot_state.joint_state.at(jname).pos;
            jv(id) = robot_state.joint_state.at(jname).vel;
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
    auto ekf_duration    = 0;
    auto kf_duration     = std::chrono::duration_cast<std::chrono::microseconds>(end_kf    - start_kf).count();
    auto mpc_duration    = std::chrono::duration_cast<std::chrono::microseconds>(end_mpc   - start_mpc).count();
    auto wbc_duration    = std::chrono::duration_cast<std::chrono::microseconds>(end_wbc   - start_wbc).count();

    logger_.log("execution_time_update", static_cast<double>(update_duration));
    logger_.log("execution_time_ekf",    static_cast<double>(ekf_duration));
    logger_.log("execution_time_kf",     static_cast<double>(kf_duration));
    logger_.log("execution_time_mpc",    static_cast<double>(mpc_duration));
    logger_.log("execution_time_wbc",    static_cast<double>(wbc_duration));
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
