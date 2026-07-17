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
#include <RobotInterface.hpp>

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
        "des_com_position_plip", "des_com_velocity_plip", "des_zmp_position_plip", "des_com_acceleration_plip",
        "ef_zmp_position",
        "p_lsole", "p_rsole", "v_lsole", "v_rsole",
        "p_lsole_des", "p_rsole_des", "v_lsole_des", "v_rsole_des",
        "lsole_orientation",  "rsole_orientation",
        "des_lsole_orientation", "des_rsole_orientation",
        "estimated_force_lsole", "estimated_force_rsole",
        "estimated_force_lwrist", "estimated_force_rwrist",
        "left_arm_residual", "right_arm_residual",
        "left_arm_tau_g", "right_arm_tau_g",
        "initial_generalized_momentum", "generalized_momentum",
        "wbc_force_lsole", "wbc_force_rsole",
        "wbc_accelerations", "angular_momentum", "input_torque", "motor_torque_filt",
        "mpc_pred_com_pos", "mpc_pred_com_vel", "mpc_pred_zmp_pos",
        "mpc_zmp_velocity", "con_zmp_velocity",
        "torso_orientation",     "torso_angular_velocity",
        "des_torso_orientation", "des_torso_angular_velocity",
        "hac_eh", "hac_eh_dot"
    }) { logger_.reserve(name, max_steps); }

    for (const char* name : {
        "execution_time_wbc", "execution_time_mpc",
        "execution_time_ekf", "execution_time_kf", "execution_time_update", 
        "execution_time_res_obs", "execution_time_hac", "execution_time_coop_planner",
        "residual_vector_norm"
    }) { logger_.reserveScalar(name, max_steps); }

    // MPC per-solve snapshots saved at fixed 10 Hz (every 100 ms), independent of horizon.
    constexpr int64_t kMpcSnapshotPeriodMs = 100;
    int64_t max_mpc_solves = max_steps * 2 / kMpcSnapshotPeriodMs;
    mpc_snapshot_t_log_.reserve(max_mpc_solves);
    mpc_snapshot_x_log_.reserve(max_mpc_solves);
    mpc_snapshot_u_log_.reserve(max_mpc_solves);

    // READING ROBOT DESCRIPTION (URDF) AND BUILDING PINOCCHIO MODEL

    // Rubber hand urdf
    // std::string robot_description_filename = "../robot/g1/g1_description/g1_29dof_rev_1_0.urdf";

    // Dex3-1 hand urdf
    // std::string robot_description_filename = "../robot/g1/g1_description/g1_29dof_dex3.urdf";
    std::string robot_description_filename = "../robot/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf";

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
    joint_vel_filt_ = Eigen::VectorXd::Zero(njnt);

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
    lwrist_idx_ = robot_model.getFrameId("left_wrist_yaw_link");
    rwrist_idx_ = robot_model.getFrameId("right_wrist_yaw_link");
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

    if(false){
        walking_data_.addSteps(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
            0.0
        );
    };

    initial_gait_configuration.qjnt = q_jnt_des_;
    initial_gait_configuration.qjntdot = Eigen::VectorXd::Zero(njnt);
    initial_gait_configuration.qjntddot = Eigen::VectorXd::Zero(njnt);

    // CoM initial from IS-MPC LIP integration
    initial_gait_configuration.com.pos = robot_data.com[0];
    initial_gait_configuration.com.vel = Eigen::Vector3d::Zero();
    initial_gait_configuration.com.acc = Eigen::Vector3d::Zero();

    // contact flags

    initial_gait_configuration.lsole.pos = walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration();
    initial_gait_configuration.lsole.vel = Eigen::VectorXd::Zero(6);
    initial_gait_configuration.lsole.acc = Eigen::VectorXd::Zero(6);
    initial_gait_configuration.rsole.pos = walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration();
    initial_gait_configuration.rsole.vel = Eigen::VectorXd::Zero(6);
    initial_gait_configuration.rsole.acc = Eigen::VectorXd::Zero(6);

    // torso task
    double left_foot_yaw_init = std::atan2(initial_gait_configuration.lsole.pos.R(1, 0), initial_gait_configuration.lsole.pos.R(0, 0));
    double right_foot_yaw_init = std::atan2(initial_gait_configuration.rsole.pos.R(1, 0), initial_gait_configuration.rsole.pos.R(0, 0));
    
    initial_gait_configuration.torso.pos = robot_data.oMf[torso_idx_].rotation();
    initial_gait_configuration.torso.vel = Eigen::Vector3d::Zero();
    initial_gait_configuration.torso.acc = Eigen::Vector3d::Zero();
    initial_gait_configuration.pelvis.pos = initial_robot_state.orientation;
    initial_gait_configuration.pelvis.vel = Eigen::Vector3d::Zero();
    initial_gait_configuration.pelvis.acc = Eigen::Vector3d::Zero();


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

    /*
    ismpc_ptr_ = std::make_unique<labrob::ISMPC>(
        mpc_prediction_horizon_msec,
        mpc_timestep_msec,
        std::sqrt(eta2),
        foot_constraint_square_length,
        foot_constraint_square_width
    );
    */

    ismpc_ptr_ = std::make_unique<labrob::ISMPC>(
        mpc_prediction_horizon_msec,
        mpc_timestep_msec,
        std::sqrt(eta2),
        Eigen::Vector3d::Zero(),            // w = 0
        foot_constraint_square_length,
        foot_constraint_square_width
    );


    // INIT HAC

    // F origin = midpoint feet, orientation = support foot (lsole at the beginning)
    const Eigen::Vector3d p_lsole_init = T_lsole_init.translation();
    const Eigen::Vector3d p_rsole_init = T_rsole_init.translation();
    Eigen::Vector3d p_F_init;
    p_F_init.x() = 0.5 * (p_lsole_init.x() + p_rsole_init.x());
    p_F_init.y() = 0.5 * (p_lsole_init.y() + p_rsole_init.y());
    p_F_init.z() = 0.5 * (p_lsole_init.z() + p_rsole_init.z());
    const Eigen::Matrix3d R_F_init = T_lsole_init.rotation(); // starting support = left foot
 
    // --- HAC: initial hand positions as rest positions ---
    const Eigen::Vector3d p_lhand_W = robot_data.oMf[lwrist_idx_].translation();
    const Eigen::Vector3d p_rhand_W = robot_data.oMf[rwrist_idx_].translation();
    // --- HAC: Average Hand Error Threshold ---
    const double eh_threshold = 0.05; // [m]
    const int below_threshold_counter = 0;
    const int below_threshold_limit = 5; // max number of samples of eh below the threshold preventing stopping procedure to start
    // Map to F frame --> obtain rest positions in local frame F
    const Eigen::Vector3d r_l_bar = R_F_init.transpose() * (p_lhand_W - p_F_init);
    const Eigen::Vector3d r_r_bar = R_F_init.transpose() * (p_rhand_W - p_F_init);

 
    // --- HAC: parameters (follower, no object: f_bar = 0) ---
    // M = 5 kg on all 3 axes
    // K_F = diag{20, 20, 100} N/m (stiffer on z to sustain arms weight)
    // C_F = diag{50, 50, 150} N·s/m
    const Eigen::Vector3d mass_diag(5.0, 5.0, 5.0);
    const Eigen::Vector3d damping_diag(50.0, 50.0, 150.0);
    const Eigen::Vector3d stiffness_diag(20.0, 20.0, 100.0);
    const Eigen::Vector3d f_l_bar_W = Eigen::Vector3d::Zero(); // no object
    const Eigen::Vector3d f_r_bar_W = Eigen::Vector3d::Zero();
 
    hac_ptr_ = std::make_unique<labrob::HandAdmittanceController>(
        controller_timestep_msec_ / 1000.0,   // dt in seconds
        mass_diag, damping_diag, stiffness_diag,
        r_l_bar, r_r_bar,
        f_l_bar_W, f_r_bar_W, 
        eh_threshold, below_threshold_limit, below_threshold_counter
    );
    hac_ptr_->reset(p_lhand_W, p_rhand_W, p_F_init, R_F_init);

    // Initialize last support foot for the HAC
    hac_last_support_foot_ = labrob::Foot::LEFT;


    // ONLINE PLANNER INIT
    labrob::FootstepPlannerCoop::Params coop_fp;
    coop_fp.F                    = 4;
    coop_fp.T_step_ms            = 2000.0;
    coop_fp.double_support_ratio = 0.4;
    coop_fp.step_height          = 0.06;
    // Calcola ell dalla posizione iniziale misurata
    const double foot_separation = std::abs( T_lsole_init.translation().y() - T_rsole_init.translation().y());  // ≈ 0.276m
    coop_fp.ell = foot_separation;
    coop_fp.kp_x = 0.8;  coop_fp.kp_y = 0.8;
    coop_fp.kd_x = 0.1;  coop_fp.kd_y = 0.1;
    coop_fp.ki_x = 0.05; coop_fp.ki_y = 0.05;
    coop_fp.da_x = 0.20; coop_fp.da_y = 0.10;

    coop_planner_ptr_ = std::make_unique<labrob::FootstepPlannerCoop>(coop_fp);
    coop_planner_ptr_->init(
        labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
        labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
        labrob::Foot::LEFT
    );

    prev_support_foot_ = labrob::Foot::LEFT;

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
    

    // Init Perturbed LIP (PLIP) dynamics
    L_dot_ = Eigen::Vector3d::Zero();           // angular momentum derivative

    discrete_plip_dynamics_ptr_ = std::make_unique<labrob::DiscretePLIPDynamics>(
        mass, com_target_height, 0.001 * controller_timestep_msec_,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        p_rhand_W, p_rhand_W
    );

    

    discrete_lip_dynamics_ptr_mpc_ = std::make_unique<labrob::DiscreteLIPDynamics>(
        std::sqrt(eta2),
        0.001 * mpc_timestep_msec
    );

    // WRIST FORCE ESTIMATOR BASED ON FULL MODEL AND ALL EXTERNAL WRENCHES
    wrist_force_estimator_ptr_ = std::make_unique<labrob::WristForceEstimator>(
        robot_model,
        armatures,
        500.0,                      // Ki = 500
        "right_wrist_yaw_link",
        "left_wrist_yaw_link",
        "right_foot_link",
        "left_foot_link"
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

    // SET FORCE ESTIMATION (1 step causal delay)

    double residual_vector_norm = 0;
    
    Eigen::Vector3d f_left_wrist  = estimated_force_wrist.head(3);
    Eigen::Vector3d f_right_wrist = estimated_force_wrist.tail(3);
    

    Eigen::Vector3d left_foot_force = estimated_force_sole.head(3);
    Eigen::Vector3d right_foot_force = estimated_force_sole.tail(3);

    Eigen::Vector3d total_force = left_foot_force + right_foot_force + f_left_wrist + f_right_wrist;

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

    const auto& T_lwrist = robot_data.oMf[lwrist_idx_];
    Eigen::MatrixXd J_lwrist = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        robot_data,
        lwrist_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_lwrist
    );
    const auto& v_lwrist = J_lwrist * qdot;

    const auto& T_rwrist = robot_data.oMf[rwrist_idx_];
    Eigen::MatrixXd J_rwrist = Eigen::MatrixXd::Zero(6, njnt + 6);
    pinocchio::getFrameJacobian(
        robot_model,
        robot_data,
        rwrist_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_rwrist
    );
    const auto& v_rwrist = J_rwrist * qdot;

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

    

    // =========================================================
    // ZMP RECONSTRUCTION
    // =========================================================

    // From LIP
    Eigen::Vector3d zmp_3d;


    zmp_3d.z() = p_CoM.z() - (a_CoM_drift.z() + 9.81) / eta2;
    zmp_3d.x() = p_CoM.x() - a_CoM_drift.x() / eta2;
    zmp_3d.y() = p_CoM.y() - a_CoM_drift.y() / eta2;
    

    // From RW-BO    
    Eigen::Vector3d ef_zmp_3d = Eigen::Vector3d::Zero();

    if (total_force.z() > 1e-5) {

        // SECOND FORMULA FOR ZMP POSITION WITH FORCE ESTIMATION WITH 1 CONTACT POINT PER FOOT

        ef_zmp_3d.x() =
            ( left_foot_force.z()  * T_lsole.translation().x() +
            right_foot_force.z() * T_rsole.translation().x()+
            f_right_wrist.z() * T_lwrist.translation().x() +
            f_left_wrist.z() * T_rwrist.translation().x() ) / total_force.z();


        ef_zmp_3d.y() =
            ( left_foot_force.z()  * T_lsole.translation().y() +
            right_foot_force.z() * T_rsole.translation().y() +
            f_right_wrist.z() * T_lwrist.translation().y() +
            f_left_wrist.z() * T_rwrist.translation().y() ) / total_force.z();

        ef_zmp_3d.z() = zmp_3d.z() + (f_right_wrist.z() + f_left_wrist.z() ) / (robot_data.mass[0] * eta2);

        
    } else {
        ef_zmp_3d.z() = zmp_3d.z();
        ef_zmp_3d.x() = zmp_3d.x();
        ef_zmp_3d.y() = zmp_3d.y();
    }
    
    // =========================================================
    // END ZMP RECONSTRUCTION
    // =========================================================
    
    

    walking_data_.updateWalkingState(t_msec_);


    // =========================================================
    // HAC — Hand Admittance Controller
    // =========================================================

    // Start HAC timer
    auto start_hac = std::chrono::system_clock::now();
 
    // Compute current local frame F:
    // Origin = midpoint of ground projections of feet
    // Orientation = current support foot
    T_lsole_ = robot_data.oMf[lsole_idx_];
    T_rsole_ = robot_data.oMf[rsole_idx_];
    updateHACLocalFrame();
    
    // Integrate HAC dynamics
    hac_ptr_->integrate(hac_f_l_W, hac_f_r_W, p_F_hac_, R_F_hac_);

    // End HAC timer
    auto end_hac = std::chrono::system_clock::now();

    // Print HAC info
    if ((t_msec_ % 1000 == 0) && verbose_coop_) {
        std::cout << "[HAC] f_l_W = " << hac_f_l_W.transpose()
                << "  eh = " << hac_ptr_->getEh().transpose() << "\n";
    }
 
    // =========================================================
    // End of HAC block
    // =========================================================

    //=========================================================
    // KF FUNCTION CALL
    //=========================================================

    auto start_kf = std::chrono::high_resolution_clock::now();

    if (t_msec_ < 2000)
        LipState = LIPState(p_CoM, J_CoM * qdot, zmp_3d);
    else
        LipState = LIPState(p_CoM, J_CoM * qdot, ef_zmp_3d);

    // kf_LipState = LipState; // --> use this to disable the CoM KF
    kf_LipState = com_kf_step(kf_LipState, LipState, ismpc_ptr_->getInput());

    auto end_kf = std::chrono::high_resolution_clock::now();


    //=========================================================
    // END KF FUNCTION CALL
    //=========================================================



    // =========================================================
    // COOP WALKING STANDING --> WALKING transition
    // =========================================================

    // Declare coop planner timer placeholders
    std::chrono::time_point<std::chrono::system_clock> start_coop_planner;
    std::chrono::time_point<std::chrono::system_clock> end_coop_planner;
    bool coop_planner_ran = false;

    if (!coop_walking_triggered_ &&
        walking_data_.getWalkingState() == WalkingState::Standing &&
        hac_ptr_->isEhAboveThreshold())
    {
        std::cout << "[COOP] Walking triggered at t=" << t_msec_
                << " eh=" << hac_ptr_->getEh().transpose() << "\n";
        
        const labrob::SE3 T_l(T_lsole.rotation(), T_lsole.translation());
        const labrob::SE3 T_r(T_rsole.rotation(), T_rsole.translation());

        // Initialize planner on current pose
        const labrob::Foot init_sf = walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot();
        coop_planner_ptr_->init(T_l, T_r, init_sf);
        coop_planner_ptr_->resetIntegral();
        integral_eh_.setZero();

        // Reset LIP state at current measured value
        des_LipState = kf_LipState;

         // Standing --> Walking transition: remove infinite standing step, reset t0, pre-fill deque
        if (reactive_standing_) {
            std::cout << "[COOP PLANNER] Reactive standing: keeping infinite standing step\n";
        } else {

            std::cout << "[COOP PLANNER] Reactive walking: removing infinite standing step and resetting t0\n";
            walking_data_.footstep_plan.pop_front();
            walking_data_.t0 = t_msec_;
        
        
            // Append starting sequence + plan
            const labrob::Foot first_swing_foot = (init_sf == labrob::Foot::LEFT) ? labrob::Foot::RIGHT : labrob::Foot::LEFT;
            walking_data_.startWalkingCoop(T_l, T_r, first_swing_foot,
                                        coop_T_ds_ms_, coop_T_ss_ms_,
                                        coop_step_height_);

            
            // Start coop planner timer
            start_coop_planner = std::chrono::system_clock::now();

            // Pre-filling: plan F footsteps based on current e_h
            auto result = coop_planner_ptr_->computeNextSteps(
                T_l, T_r, init_sf,
                hac_ptr_->getEh(), hac_ptr_->getEhDot(),
                hac_ptr_->getLeftHandPos(), hac_ptr_->getRightHandPos(),
                R_F_hac_, controller_timestep_msec_ * 0.001
            );
            auto pre_steps = coop_planner_ptr_->buildDequeElements(result);

            // End coop planner timer
            end_coop_planner = std::chrono::system_clock::now();
            coop_planner_ran = true;
            
            // Print info
            if (verbose_coop_) {
                
                // Show first plan
                std::cout << "First QP solution:\n";
                showPlan(result);

                // Show deque at first trigger before inserting presteps
                std::cout << "[DEQUE at trigger before inserting pre-steps] size=" << walking_data_.footstep_plan.size() << "\n";
                showDeque(walking_data_);
                
            }

            // Update deque after pre-filling
            for (size_t i = 1; i < pre_steps.size(); ++i)
                walking_data_.footstep_plan.push_back(pre_steps[i]);

            prev_support_foot_ = init_sf;
            coop_walking_triggered_ = true;
            coop_walking_active_    = true;
            
            // Print info
            if (verbose_coop_) {
                // Show deque after pre-filling
                std::cout << "[DEQUE at trigger after pre-filling] size=" << walking_data_.footstep_plan.size() << "\n";
                showDeque(walking_data_);

            }
        }
    }

    // =========================================================
    // END COOP WALKING STANDING --> WALKING transition
    // =========================================================

    // =========================================================
    // COOP ROLLING PLANNER — replan at every DS
    // =========================================================
    if (coop_walking_active_) {

        const labrob::Foot curr_sf =
            walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot();

        
        const bool support_switched =
            (curr_sf != prev_support_foot_) &&
            (walking_data_.getWalkingState() == WalkingState::DoubleSupport ||
            walking_data_.getWalkingState() == WalkingState::Starting);

        prev_support_foot_ = curr_sf;

        

        if (support_switched) {
            waiting_for_rolling_ = true;   // armed, waiting for landing
            
            // Print
            if (verbose_coop_) {
                std::cout << "Support foot switched to " << (curr_sf == Foot::LEFT ? "LEFT" : "RIGHT") << " at t=" << t_msec_ << "\n";
                std::cout << "Waiting for rolling ..." << "\n";
            }
        }


        if (waiting_for_rolling_) {
            
            const labrob::SE3 T_l(T_lsole.rotation(), T_lsole.translation());
            const labrob::SE3 T_r(T_rsole.rotation(), T_rsole.translation());

            // Flag to make sure swing foot has landed before replanning
            const double new_support_z = (curr_sf == Foot::LEFT)
            ? T_lsole.translation().z()
            : T_rsole.translation().z();

            if (new_support_z < 0.015) {
                waiting_for_rolling_ = false;

                // Print
                if (verbose_coop_) {
                    std::cout << "Foot landed and rolling started" << "\n";
                }
                

                // Check that deque has more than one element
                if (walking_data_.footstep_plan.size() < 2) {
                    std::cerr << "[WARN] stopWalkingCoop: deque too short, skipping rolling\n";
                } else {

                    if (hac_ptr_->stoppingRequested()) {
                        std::cout << "Stopping requested!" << std::endl;

                        
                        // Read planned position from last stored element (SS, deque[1])
                        const auto& last_ss = walking_data_.footstep_plan[1];
                        const auto& planned_l = last_ss.getFeetPlacement().getLeftFootConfiguration();
                        const auto& planned_r = last_ss.getFeetPlacement().getRightFootConfiguration();

                        // Support foot: use planned position (continuity with committed plan)
                        // Swing foot: use measure (real position from which the swing starts)
                        const labrob::SE3 T_l_anchor = (curr_sf == labrob::Foot::LEFT)
                            ? labrob::SE3(planned_l.R, planned_l.p)
                            : labrob::SE3(T_lsole.rotation(), T_lsole.translation());

                        const labrob::SE3 T_r_anchor = (curr_sf == labrob::Foot::RIGHT)
                            ? labrob::SE3(planned_r.R, planned_r.p)
                            : labrob::SE3(T_rsole.rotation(), T_rsole.translation());

                        // Rolling: keep only current front (first DS-SS pair) and replace the rest
                        while (walking_data_.footstep_plan.size() > 2)
                            walking_data_.footstep_plan.pop_back();

                        walking_data_.stopWalkingCoop(T_l_anchor, T_r_anchor, curr_sf);
                        coop_walking_active_ = false;
                    
                    } else {
                        integral_eh_ = coop_planner_ptr_->updateErrorIntegral(hac_ptr_->getEh(), controller_timestep_msec_ * 0.001);

                        // Read planned position from last stored element (SS, deque[1])
                        const auto& last_ss = walking_data_.footstep_plan[1];
                        const auto& planned_l = last_ss.getFeetPlacement().getLeftFootConfiguration();
                        const auto& planned_r = last_ss.getFeetPlacement().getRightFootConfiguration();

                        // Support foot: use planned position (continuity with committed plan)
                        // Swing foot: use measure (real position from which the swing starts)
                        const labrob::SE3 T_l_anchor = (curr_sf == labrob::Foot::LEFT)
                            ? labrob::SE3(planned_l.R, planned_l.p)
                            : labrob::SE3(T_lsole.rotation(), T_lsole.translation());

                        const labrob::SE3 T_r_anchor = (curr_sf == labrob::Foot::RIGHT)
                            ? labrob::SE3(planned_r.R, planned_r.p)
                            : labrob::SE3(T_rsole.rotation(), T_rsole.translation());
                        

                        // Start coop planner timer
                        start_coop_planner = std::chrono::system_clock::now();

                        auto result = coop_planner_ptr_->computeNextSteps(
                            T_l_anchor, T_r_anchor, curr_sf,
                            hac_ptr_->getEh(), hac_ptr_->getEhDot(),
                            hac_ptr_->getLeftHandPos(), hac_ptr_->getRightHandPos(),
                            R_F_hac_, controller_timestep_msec_ * 0.001
                        );
                        auto new_steps = coop_planner_ptr_->buildDequeElements(result);

                        
                        // Print
                        if (verbose_coop_) {
                            
                            // Show regular plan
                            std::cout << "[COOP PLANNER] t=" << t_msec_
                                    << " sf=" << (curr_sf == Foot::LEFT ? "L" : "R")
                                    << " eh=" << hac_ptr_->getEh().transpose()
                                    << " dp=" << coop_planner_ptr_->getLastDeltaP().transpose()
                                    << " dth=" << coop_planner_ptr_->getLastDeltaTheta()*180/M_PI << "deg\n";

                            std::cout << "Regular QP solution:\n";
                            showPlan(result);
                            

                            
                            // Visualize deque before rolling
                            std::cout << "[DEQUE STATE] t=" << t_msec_
                                    << " ws=" << static_cast<int>(walking_data_.getWalkingState())
                                    << " deque=" << walking_data_.footstep_plan.size()
                                    << " current_lsole=" << T_lsole.translation().transpose()
                                    << " current_rsole=" << T_rsole.translation().transpose() << "\n";
                            showDeque(walking_data_);

                        }
                        
                        
                    
                        // Rolling: keep only current front (first DS-SS pair) and replace the rest
                        while (walking_data_.footstep_plan.size() > 2)
                            walking_data_.footstep_plan.pop_back();
                        for (size_t i = 2; i < new_steps.size(); ++i)
                            walking_data_.footstep_plan.push_back(new_steps[i]);

                        
                        // End coop planner timer
                        end_coop_planner = std::chrono::system_clock::now();
                        coop_planner_ran = true;
                        
                        // Print info
                        if (verbose_coop_) {

                            // Visualize deque after rolling
                            std::cout << "[DEQUE STATE AFTER ROLLING] t=" << t_msec_
                                    << " ws=" << static_cast<int>(walking_data_.getWalkingState())
                                    << " deque=" << walking_data_.footstep_plan.size()
                                    << " lsole_z=" << T_lsole_.translation().z()
                                    << " rsole_z=" << T_rsole_.translation().z() << "\n";
                            showDeque(walking_data_);
                            
                        }
                        

                    }


                }
                                
            }

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
    current_gait_configuration.lsole.pos = labrob::SE3(robot_data.oMf[lsole_idx_].rotation(), robot_data.oMf[lsole_idx_].translation());
    current_gait_configuration.lsole.vel = J_lsole * qdot;
    current_gait_configuration.rsole.pos = labrob::SE3(robot_data.oMf[rsole_idx_].rotation(), robot_data.oMf[rsole_idx_].translation());
    current_gait_configuration.rsole.vel = J_rsole * qdot;
    current_gait_configuration.lwrist.pos = robot_data.oMf[lwrist_idx_].translation();
    current_gait_configuration.lwrist.vel = J_lwrist.topRows<3>() * qdot;
    current_gait_configuration.rwrist.pos = robot_data.oMf[rwrist_idx_].translation();
    current_gait_configuration.rwrist.vel = J_rwrist.topRows<3>() * qdot;

    /////////////////////////////////////
    // MPC FUNCTION CALL
    /////////////////////////////////////

    LIPState mpc_LipState_prec = des_LipState;

    auto start_mpc = std::chrono::system_clock::now();

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            if(isMPCLoopClosed){


                // LIP
                
                /*
                // ismpc_ptr_->solve(t_msec_, walking_data_, kf_LipState);
                ismpc_ptr_->solve(t_msec_, walking_data_, kf_LipState, Eigen::Vector3d::Zero());

                
                // CoM reference generation without considering external disturbance
                des_LipState = discrete_lip_dynamics_ptr_->integrate(
                    kf_LipState,
                    ismpc_ptr_->getInput(),
                    Eigen::Vector3d::Zero()
                );
                */
                
                
    
                // PLIP
                
                
                // Extract disturbance term from wrist forces and angular momentum derivative
                discrete_plip_dynamics_ptr_->updateDisturbanceTerm(kf_LipState,
                    f_right_wrist, f_left_wrist,
                    Eigen::Vector3d::Zero(),            //L_dot_ --> should be strongly filtered
                    T_rwrist.translation(), T_lwrist.translation()
                );

                Eigen::Vector3d current_disturbance; 

                // Cut off transient
                if (t_msec_ < 2000) {
                    current_disturbance.x() = 0.0;
                    current_disturbance.y() = 0.0;
                    current_disturbance.z() = -9.81;
                } else {
                    current_disturbance = discrete_plip_dynamics_ptr_->get_disturbance();
                }


                ismpc_ptr_->solve(t_msec_, walking_data_, kf_LipState, current_disturbance);

                // Log disturbance term right before it enters the PLIP integration
                logger_.log("current_disturbance", current_disturbance);

                // CoM reference generation while considering external disturbance (overwrite)
                des_LipState = discrete_plip_dynamics_ptr_->integrate(
                    kf_LipState,
                    ismpc_ptr_->getInput(),
                    current_disturbance
                );
                
                
                
                
            }
            else{

                
                // ismpc_ptr_->solve(t_msec_, walking_data_, des_LipState);
                ismpc_ptr_->solve(t_msec_, walking_data_, des_LipState, Eigen::Vector3d::Zero());
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

    // Wrist task — active fron Standing phase on (not during PostureRegulation/Init)
    const bool wrist_task_active =
        (walking_data_.getWalkingState() != WalkingState::Init &&
        walking_data_.getWalkingState() != WalkingState::PostureRegulation);


    // --- Re-anchor HAC rest positions at task activation edge ---
    // The HAC was initialized at startup posture, but posture regulation may have
    // moved the hands before the wrist task switches on. Re-fix r_i_bar to the
    // hands' CURRENT measured pose so that e_h starts at ~0 (no spurious transient).
    if (wrist_task_active && !hac_wrist_task_was_active_) {
        const Eigen::Vector3d p_lhand_W = robot_data.oMf[lwrist_idx_].translation();
        const Eigen::Vector3d p_rhand_W = robot_data.oMf[rwrist_idx_].translation();
        hac_ptr_->reset(p_lhand_W, p_rhand_W, p_F_hac_, R_F_hac_);
    }
    hac_wrist_task_was_active_ = wrist_task_active;

    
    
    if (wrist_task_active) {
        desired_gait_configuration.lwrist.pos = hac_ptr_->getLeftHandRef();
        desired_gait_configuration.rwrist.pos = hac_ptr_->getRightHandRef();

    } else {
        // Errore zero: desired = current, il task non perturba la postura
        desired_gait_configuration.lwrist.pos = robot_data.oMf[lwrist_idx_].translation();
        desired_gait_configuration.rwrist.pos = robot_data.oMf[rwrist_idx_].translation();    
    }
    desired_gait_configuration.lwrist.vel.setZero();
    desired_gait_configuration.lwrist.acc.setZero();
    desired_gait_configuration.rwrist.vel.setZero();
    desired_gait_configuration.rwrist.acc.setZero();


    /////////////////////////////////////
    // START WHOLE BODY CONTROLLER FUNCTION CALL
    /////////////////////////////////////

        // Print WBC input
    if ((t_msec_ % 100 == 0) && verbose_coop_) {
        std::cout << "[WBC INPUT] t=" << t_msec_
          << " des_lsole=" << desired_gait_configuration.lsole.pos.p.transpose()
          << " des_rsole=" << desired_gait_configuration.rsole.pos.p.transpose()
          << " des_rwrist=" << desired_gait_configuration.rwrist.pos.transpose()
          << " des_CoM=" << desired_gait_configuration.com.pos.transpose()
          << "\n";
    }

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

    // =========================================================================
    // WristForceEstimator — AFTER the WBC, use torques of the current torques
    // =========================================================================
    
    // Start wrench observer timer
    auto start_res_obs = std::chrono::system_clock::now();

    if (isObserverActive) 
        {   

            // Collect torques
            Eigen::VectorXd wbc_torques(njnt);
            Eigen::VectorXd motor_torques = Eigen::VectorXd::Zero(njnt);
            std::lock_guard<std::mutex> lock(stateMutex);
            for (pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) robot_model.njoints; ++joint_id) {
                const auto& joint_name = robot_model.names[joint_id];

                int torque_idx = joint_id - 2;

                // Fill wbc_torques with torques from WBC
                wbc_torques(torque_idx) = joint_command[joint_name];

                // Fill motor_torques with estimated and published torques
                int sdk_idx = joint_name_to_index.at(joint_name);
                motor_torques(torque_idx) = motor_state_data.tau_est[sdk_idx];
            }


            // Select which torques to use for the observer: 
            Eigen::VectorXd torques(robot_model.nv - 6);
            {
                // 1. In simulation --> WBC output 
                torques = wbc_torques;

                // 2. In real experiments --> estimates from motor's firmware + EMA filter to reduce noise
                if (useRobot) {

                    // Filter
                    torques_filt_ = 0.1 * torques + 0.9 * torques_filt_;

                    // Save for logs
                    torques = torques_filt_;

                    logger_.log("motor_torque_filt", torques_filt_);
                }

            }
            
            // TEST
            // Parti dalle misure grezze del robot
            labrob::RobotState raw_robot_state = robot_state;

            // EMA sulle velocità di giunto (le posizioni restano grezze)
            const double a_vel = 0.1;   // taratura: più basso = più filtraggio

            if (!joint_vel_filt_init_) {
                for (pinocchio::JointIndex jid = 2; jid < (pinocchio::JointIndex) robot_model.njoints; ++jid) {
                    joint_vel_filt_(jid-2) = measured_joint_velocity(jid-2);
                }
                joint_vel_filt_init_ = true;
            } else {
                for (pinocchio::JointIndex jid = 2; jid < (pinocchio::JointIndex) robot_model.njoints; ++jid) {
                    const auto& name = robot_model.names[jid];
                    joint_vel_filt_(jid-2) = a_vel * robot_state.joint_state[name].vel
                                    + (1.0 - a_vel) * joint_vel_filt_(jid-2);
                }
            }

            // Scrivi le velocità filtrate nello stato grezzo
            for (pinocchio::JointIndex jid = 2; jid < (pinocchio::JointIndex) robot_model.njoints; ++jid) {
                raw_robot_state.joint_state[robot_model.names[jid]].vel = joint_vel_filt_(jid-2);
            }

            // Update observer
            wrist_force_estimator_ptr_->update(
                raw_robot_state, 
                robot_data,
                torques,
                controller_timestep_msec_ * 0.001
            );
    
            // Save observations
            estimated_force_sole.head<3>() = wrist_force_estimator_ptr_->getLeftFootWrench().head<3>();
            estimated_force_sole.tail<3>() = wrist_force_estimator_ptr_->getRightFootWrench().head<3>();

            estimated_force_wrist.head<3>() = wrist_force_estimator_ptr_->getWeightedLeftWristForce();
            estimated_force_wrist.tail<3>() = wrist_force_estimator_ptr_->getWeightedRightWristForce();
            
            residual_vector_norm = wrist_force_estimator_ptr_->getResidual().norm();
    
            // Update forces for the HAC at next step (1 step causal delay)
            static constexpr int64_t WFE_TRANSIENT_MS = 2000;
            if (t_msec_ >= WFE_TRANSIENT_MS) {
                hac_f_l_W = f_left_wrist;
                hac_f_r_W = f_right_wrist;
            } else {
                hac_f_l_W = Eigen::Vector3d::Zero();
                hac_f_r_W = Eigen::Vector3d::Zero();
            }

        }

    // End wrench observer timer
    auto end_res_obs = std::chrono::system_clock::now();

    
    // Print estimated wrist forces
    if ((t_msec_ % 500 == 0) && verbose_coop_ && isObserverActive) 
    {
        std::cout << "[WRIST FORCE] right: " << estimated_force_wrist.head<3>().transpose() << "\n";
        std::cout << "[WRIST FORCE] left:  " << estimated_force_wrist.tail<3>().transpose()  << "\n";
    }
    
    // =========================================================================
    // END WristForceEstimator — AFTER the WBC, use torques of the current torques
    // =========================================================================


    // Update timing in milliseconds.
    // NOTE: assuming update() is actually called every controller_timestep_msec_
    //       milliseconds.
    t_msec_ += controller_timestep_msec_;

    // Upadate angular momentum rate
    L_dot_ = (angular_momentum - prev_angular_momentum_) / (0.001 * controller_timestep_msec_);
    
    // Store angular momentum for next iteration
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
    logger_.log("ef_zmp_position", ef_zmp_3d);

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

    logger_.log("hac_eh",     hac_ptr_->getEh());
    logger_.log("hac_eh_dot", hac_ptr_->getEhDot());
    logger_.log("estimated_force_lsole", estimated_force_sole.head<3>());
    logger_.log("estimated_force_rsole", estimated_force_sole.tail<3>());
    logger_.log("estimated_force_lwrist", estimated_force_wrist.tail<3>());
    logger_.log("estimated_force_rwrist", estimated_force_wrist.head<3>());
    logger_.log("residual_vector_norm", residual_vector_norm);

    {
        static const std::vector<std::string> left_arm_joints = {
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
        };
        static const std::vector<std::string> right_arm_joints = {
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        };
        const Eigen::VectorXd& r_full = wrist_force_estimator_ptr_->getResidual();
        const Eigen::VectorXd tau_minus_g = wrist_force_estimator_ptr_->getTauMinusG();

        Eigen::VectorXd left_arm_res = Eigen::VectorXd::Zero(left_arm_joints.size());
        Eigen::VectorXd right_arm_res = Eigen::VectorXd::Zero(right_arm_joints.size());
        Eigen::VectorXd left_arm_tg = Eigen::VectorXd::Zero(left_arm_joints.size());
        Eigen::VectorXd right_arm_tg = Eigen::VectorXd::Zero(right_arm_joints.size());
        for (size_t i = 0; i < left_arm_joints.size(); ++i) {
            if (robot_model.existJointName(left_arm_joints[i])) {
                int vidx = robot_model.idx_vs[robot_model.getJointId(left_arm_joints[i])];
                left_arm_res(i) = r_full(vidx);
                left_arm_tg(i) = tau_minus_g(vidx);
            }
            if (robot_model.existJointName(right_arm_joints[i])) {
                int vidx = robot_model.idx_vs[robot_model.getJointId(right_arm_joints[i])];
                right_arm_res(i) = r_full(vidx);
                right_arm_tg(i) = tau_minus_g(vidx);
            }
        }
        logger_.log("left_arm_residual", left_arm_res);
        logger_.log("right_arm_residual", right_arm_res);
        logger_.log("left_arm_tau_g", left_arm_tg);
        logger_.log("right_arm_tau_g", right_arm_tg);
    }


    logger_.log("generalized_momentum", wrist_force_estimator_ptr_->getGeneralizedMomentum());
    logger_.log("initialized_generalized_momentum", wrist_force_estimator_ptr_->getInitialGeneralizedMomentum());

    logger_.log("wbc_force_lsole",       whole_body_controller_ptr_->getLeftFootWrench());
    logger_.log("wbc_force_rsole",       whole_body_controller_ptr_->getRightFootWrench());
    logger_.log("wbc_accelerations",     whole_body_controller_ptr_->get_q_ddot());
    logger_.log("angular_momentum",      angular_momentum);

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
    auto res_obs_duration    = std::chrono::duration_cast<std::chrono::microseconds>(end_res_obs   - start_res_obs).count();
    auto hac_duration    = std::chrono::duration_cast<std::chrono::microseconds>(end_hac   - start_hac).count();
    auto coop_planner_duration    = coop_planner_ran ? std::chrono::duration_cast<std::chrono::microseconds>(end_coop_planner   - start_coop_planner).count() : 0;

    logger_.log("execution_time_update", static_cast<double>(update_duration));
    logger_.log("execution_time_ekf",    static_cast<double>(ekf_duration));
    logger_.log("execution_time_kf",     static_cast<double>(kf_duration));
    logger_.log("execution_time_mpc",    static_cast<double>(mpc_duration));
    logger_.log("execution_time_wbc",    static_cast<double>(wbc_duration));
    logger_.log("execution_time_res_obs",    static_cast<double>(res_obs_duration));
    logger_.log("execution_time_hac",    static_cast<double>(hac_duration));
    logger_.log("execution_time_coop_planner",    static_cast<double>(coop_planner_duration));
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



void WalkingManager::updateHACLocalFrame() {
    const Eigen::Vector3d p_lsole = T_lsole_.translation();
    const Eigen::Vector3d p_rsole = T_rsole_.translation();


    // Compute position of the origin of the HAC local frame F
    Eigen::Vector3d p_F;
    p_F.x() = 0.5 * (p_lsole.x() + p_rsole.x());
    p_F.y() = 0.5 * (p_lsole.y() + p_rsole.y());
    p_F.z() = 0.5 * (p_lsole.z() + p_rsole.z());
    
    
    Eigen::Matrix3d R_F;
    const WalkingState ws = walking_data_.getWalkingState();
    if ((ws == WalkingState::SingleSupport ||
         ws == WalkingState::Starting      ||
         ws == WalkingState::Stopping)     &&
        !walking_data_.footstep_plan.empty())
    {
        const Foot sf = walking_data_.footstep_plan.front()
                            .getFeetPlacement().getSupportFoot();
        if (sf != hac_last_support_foot_) {
            hac_ptr_->onSupportSwitch(p_F,
                sf == Foot::LEFT ? T_lsole_.rotation()
                                 : T_rsole_.rotation());
            hac_last_support_foot_ = sf;
        }
        R_F = (sf == Foot::LEFT) ? T_lsole_.rotation()
                                 : T_rsole_.rotation();
    } else {
        R_F = T_lsole_.rotation();
    }

    p_F_hac_ = p_F;
    R_F_hac_ = R_F;
    
    
    
}

void WalkingManager::showPlan(const labrob::FootstepPlannerCoop::QP2DResult res) {
    for (int j = 0; j < res.F; ++j) {
            const Eigen::Vector2d pos = res.getFootPos2D(j);
            const std::string sf = (res.getSwingFoot(j) == Foot::LEFT) ? "L" : "R";
            std::cout << "  step[" << j << "] swing=" << sf
                    << " xy=(" << pos.x() << ", " << pos.y() << ")\n";
        }
}


void WalkingManager::showDeque(const labrob::WalkingData wd) {

    for (size_t k = 0; k < wd.footstep_plan.size(); ++k) {
            const auto& e = wd.footstep_plan[k];
            const auto& fp = e.getFeetPlacement();
            const std::string ws =
                e.getWalkingState() == WalkingState::Standing      ? "STA" :
                e.getWalkingState() == WalkingState::Starting      ? "STR" :
                e.getWalkingState() == WalkingState::DoubleSupport ? "DS"  :
                e.getWalkingState() == WalkingState::SingleSupport ? "SS"  : "?";
            std::cout << "  [" << k << "] " << ws
                    << " sup=" << (fp.getSupportFoot()==Foot::LEFT?"L":"R")
                    << " dur=" << e.getDuration() << "ms"
                    << " lsole_pos=" << fp.getLeftFootConfiguration().p.x() << " " 
                        << fp.getLeftFootConfiguration().p.y() << " " 
                        << fp.getLeftFootConfiguration().p.z()
                    << " rsole_pos=" << fp.getRightFootConfiguration().p.x() << " " 
                        << fp.getRightFootConfiguration().p.y() << " " 
                        << fp.getRightFootConfiguration().p.z()
                    << "\n";
    }
    
}

} // end namespace labrob
