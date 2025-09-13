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
#include <hrp4_locomotion/DdpSolver.hpp>

namespace labrob {

WalkingManager::WalkingManager() :
    sim_filt_LIPstate(Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero()),
    sim_filt_LIPstate2(Eigen::Vector3d::Zero(),
                     Eigen::Vector3d::Zero(),
                     Eigen::Vector3d::Zero())
{

}

bool
WalkingManager::init(const labrob::RobotState& initial_robot_state,
                     std::map<std::string, double> &armatures, bool useRobot) {
    cov_x = Eigen::Matrix3d::Identity();
    cov_y = Eigen::Matrix3d::Identity();
    cov_z = Eigen::Matrix3d::Identity();

    cov_meas_pos = 1.0e1;
    cov_meas_vel = 1.0e2;
    cov_meas_zmp = 1.0e6;

    cov_mod_pos = 1.0;
    cov_mod_vel = 1.0;
    cov_mod_zmp = 1.0;

    // Pre-allocation of arrays (maximum size is 50000)

    int64_t max_steps = 50000;

    mpc_timings_log_.reserve(max_steps);
    mpc_com_log_.reserve(max_steps);
    mpc_zmp_log_.reserve(max_steps);
    com_log_.reserve(max_steps);
    p_lsole_log_.reserve(max_steps);
    p_rsole_log_.reserve(max_steps);
    v_lsole_log_.reserve(max_steps);
    v_rsole_log_.reserve(max_steps);
    p_lsole_des_log_.reserve(max_steps);
    p_rsole_des_log_.reserve(max_steps);
    v_lsole_des_log_.reserve(max_steps);
    v_rsole_des_log_.reserve(max_steps);
    angular_momentum_log_.reserve(max_steps);
    cop_computed_log_.reserve(max_steps);
    mpc_predictions_log_.reserve(max_steps);
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
    fb_base_position_log_.reserve(max_steps);
    fb_base_velocity_log_.reserve(max_steps);
    fb_base_orientation_log_.reserve(max_steps);
    fb_base_angular_velocity_log_.reserve(max_steps);
    fb_joint_position_log_.reserve(max_steps);
    fb_joint_velocity_log_.reserve(max_steps);
    real_com_log_.reserve(max_steps);
    predicted_imu_accelerometer_log_.reserve(max_steps);
    predicted_imu_angular_velocity_log_.reserve(max_steps);
    predicted_imu_orientation_log_.reserve(max_steps);

    execution_time_wbc_log_.reserve(max_steps);
    execution_time_mpc_log_.reserve(max_steps);
    execution_time_ekf_log_.reserve(max_steps);
    execution_time_kf_log_.reserve(max_steps);
    execution_time_kalman_gain_log_.reserve(max_steps);

    kalman_gain_matrix_log_.reserve(max_steps);

    // Read URDF from file:
    std::string robot_description_filename = "../g1_description/unitreeg1.urdf";

    // Build Pinocchio model and data from URDF:
    pinocchio::Model full_robot_model;
    
    pinocchio::JointModelFreeFlyer root_joint;
    pinocchio::urdf::buildModel(
        robot_description_filename,
        root_joint,
        full_robot_model
    );
    const std::vector<std::string> joint_to_lock_names{};
    std::vector<pinocchio::JointIndex> joint_ids_to_lock;
    for (const auto& joint_name : joint_to_lock_names) {
        if (full_robot_model.existJointName(joint_name)) {
        joint_ids_to_lock.push_back(full_robot_model.getJointId(joint_name));
        }
    }

    sim_robot_model = pinocchio::buildReducedModel(
        full_robot_model,
        joint_ids_to_lock,
        pinocchio::neutral(full_robot_model)
    );
    sim_robot_data = pinocchio::Data(sim_robot_model);

    // Init desired lsole and rsole poses:
    auto q_init = robot_state_to_pinocchio_joint_configuration(
        sim_robot_model,
        initial_robot_state
    );
    auto qdot_init = robot_state_to_pinocchio_joint_velocity(
        sim_robot_model,
        initial_robot_state
    );
    pinocchio::forwardKinematics(sim_robot_model, sim_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(sim_robot_model, sim_robot_data, q_init);
    pinocchio::framesForwardKinematics(sim_robot_model, sim_robot_data, q_init);


    if (useRobot) {
    
        fb_robot_model = pinocchio::buildReducedModel(
            full_robot_model,
            joint_ids_to_lock,
            pinocchio::neutral(full_robot_model)
        );
        fb_robot_data = pinocchio::Data(fb_robot_model);

        pinocchio::forwardKinematics(fb_robot_model, fb_robot_data, q_init);
        pinocchio::jacobianCenterOfMass(fb_robot_model, fb_robot_data, q_init);
        pinocchio::framesForwardKinematics(fb_robot_model, fb_robot_data, q_init);

    } else {
        fb_robot_model = sim_robot_model;
        fb_robot_data = sim_robot_data;
    }


    n_ekf_output = fb_robot_model.nv - 3 + fb_robot_model.nv - 3 + 3 + 6 + 6;

    P_ = Eigen::MatrixXd::Identity(2 * fb_robot_model.nv, 2 * fb_robot_model.nv) * 1;
    P_.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-6;
    P_.block(fb_robot_model.nv, fb_robot_model.nv, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-6;
    P_.block(3,3,3,3) = Eigen::MatrixXd::Identity(3, 3) * 1e-2;
    P_.block(fb_robot_model.nv + 3, fb_robot_model.nv + 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-2;
    x_estimate = Eigen::VectorXd::Zero(2 * fb_robot_model.nv);
    x_estimate.head(3) = q_init.head(3);
    x_estimate.segment(3, 3) = Eigen::AngleAxisd(
        Eigen::Quaterniond(q_init[6], q_init[3], q_init[4], q_init[5])
    ).axis() * Eigen::AngleAxisd(Eigen::Quaterniond(q_init[6], q_init[3], q_init[4], q_init[5])).angle();
    x_estimate.segment(3 + 3, fb_robot_model.nv - 6) = q_init.tail(fb_robot_model.nv - 6);
    x_estimate.tail(fb_robot_model.nv) = qdot_init;
    std::cout << "Initial EKF state: " << x_estimate.transpose() << std::endl;
    y_pred = Eigen::VectorXd::Zero(n_ekf_output);
    y_actual = Eigen::VectorXd::Zero(n_ekf_output);
    y_estimate = Eigen::VectorXd::Zero(n_ekf_output);

    lsole_idx_ = sim_robot_model.getFrameId("left_foot_link");
    rsole_idx_ = sim_robot_model.getFrameId("right_foot_link");
    torso_idx_ = sim_robot_model.getFrameId("torso_link");
    const auto& T_lsole_init = sim_robot_data.oMf[lsole_idx_];
    const auto& T_rsole_init = sim_robot_data.oMf[rsole_idx_];

    Eigen::Quaterniond imu_orientation(
        fb_robot_data.oMf[fb_robot_model.getFrameId("imu_in_torso")].rotation()
    );

    // std::cout << "imu orientation: " << imu_orientation.coeffs().transpose() << std::endl;
    // std::cout << "imu orientation: " << imu_orientation.vec() << std::endl;

    Eigen::AngleAxisd axis_angle_init = Eigen::AngleAxisd(
        imu_orientation.w(),
        imu_orientation.vec()
    );

    Eigen::MatrixXd J_imu = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
    pinocchio::getFrameJacobian(
        fb_robot_model,
        fb_robot_data,
        fb_robot_model.getFrameId("imu_in_torso"),
        pinocchio::LOCAL_WORLD_ALIGNED,
        J_imu
    );

    //compute angular velocity of imu
    Eigen::Vector3d imu_angular_velocity = J_imu.block(3, 0, 3, fb_robot_model.nv) * qdot_init;

    //get feet position
    Eigen::Vector3d left_foot_position = fb_robot_data.oMf[lsole_idx_].translation();
    Eigen::Vector3d right_foot_position = fb_robot_data.oMf[rsole_idx_].translation();

    // //divide angle by 3.14 and take the integer part and subtract it from the angle
    // axis_angle_init.angle() = std::fmod(axis_angle_init.angle(), 2 * M_PI);
    if (axis_angle_init.angle() > M_PI) {
        axis_angle_init.angle() -= 2 * M_PI;
    } else if (axis_angle_init.angle() < -M_PI) {
        axis_angle_init.angle() += 2 * M_PI;
    }

    y_estimate.head(3) = Eigen::Vector3d(
        axis_angle_init.axis().x() * axis_angle_init.angle(),
        axis_angle_init.axis().y() * axis_angle_init.angle(),
        axis_angle_init.axis().z() * axis_angle_init.angle()
    );
    y_estimate.segment(3, fb_robot_model.nv - 6) = q_init.tail(fb_robot_model.nv - 6);
    y_estimate.segment(fb_robot_model.nv - 3, 3) = imu_angular_velocity;
    y_estimate.segment(fb_robot_model.nv - 3 + 3, fb_robot_model.nv - 6) = qdot_init.tail(fb_robot_model.nv - 6);
    // y_estimate.segment(fb_robot_model.nv - 3 + fb_robot_model.nv - 3, 3) = J_imu_dot.block(0, 0, 3, fb_robot_model.nv) * qdot_init;
    y_estimate.segment(fb_robot_model.nv - 3 + fb_robot_model.nv - 3 + 6 + 3, 3) = left_foot_position;
    y_estimate.segment(fb_robot_model.nv - 3 + fb_robot_model.nv - 3 + 6 + 3 + 3, 3) = right_foot_position;
    

    input = Eigen::VectorXd::Zero(fb_robot_model.nv);
    // convert quaternion to axis-angle representation
    Eigen::Quaterniond q_init_quat(
        q_init[6], q_init[3], q_init[4], q_init[5]
    );
    Eigen::AngleAxisd axis_angle(q_init_quat);

    if (axis_angle.angle() > M_PI) {
        axis_angle.angle() -= 2 * M_PI;
    } else if (axis_angle.angle() < -M_PI) {
        axis_angle.angle() += 2 * M_PI;
    }

    x_estimate.head(3) = q_init.head(3);
    x_estimate.segment(3, 3) = Eigen::Vector3d(
        axis_angle.axis().x() * axis_angle.angle(),
        axis_angle.axis().y() * axis_angle.angle(),
        axis_angle.axis().z() * axis_angle.angle()
    );
    x_estimate.segment(3 + 3, fb_robot_model.nv - 6) = q_init.tail(fb_robot_model.nv - 6);
    x_estimate.tail(fb_robot_model.nv) = qdot_init;
    Q = Eigen::MatrixXd::Identity(2 * fb_robot_model.nv, 2 * fb_robot_model.nv) * 1e-2;
    R = Eigen::MatrixXd::Identity(n_ekf_output, n_ekf_output) * 1e-2;

    int njnt = sim_robot_model.nv - 6;

    M_armature_ = Eigen::VectorXd::Zero(njnt);
    for(pinocchio::JointIndex joint_id = 2;
        joint_id < (pinocchio::JointIndex) sim_robot_model.njoints;
        ++joint_id) {
        std::string joint_name = sim_robot_model.names[joint_id];
        M_armature_(joint_id - 2) = armatures[joint_name];
    }

    double waist_p_des = 0.0;
    double waist_y_des = 0.0;
    double waist_r_des = 0.0;
    double r_hip_y_des = 0.0;
    double r_hip_r_des = -0.05;
    double r_hip_p_des = -0.44;
    double r_knee_p_des = 0.95;
    double r_ankle_p_des = -0.49;
    double r_ankle_r_des = 0.07;
    double l_hip_y_des = 0.0;
    double l_hip_r_des = -r_hip_r_des;
    double l_hip_p_des = r_hip_p_des;
    double l_knee_p_des = r_knee_p_des;
    double l_ankle_p_des = r_ankle_p_des;
    double l_ankle_r_des = -r_ankle_r_des;
    double r_shoulder_p_des = 0.07;
    double r_shoulder_r_des = -0.14;
    double r_shoulder_y_des = 0.0;
    double r_elbow_p_des = -0.44;
    double l_shoulder_p_des = r_shoulder_p_des;
    double l_shoulder_r_des = -r_shoulder_r_des;
    double l_shoulder_y_des = 0.0;
    double l_elbow_p_des = r_elbow_p_des;

    q_jnt_des_ = q_init.tail(njnt);

    // TODO: init using node handle.
    controller_frequency_ = 500;
    controller_timestep_msec_ = 1000 / controller_frequency_;

    double swing_foot_trajectory_height = 0.1;
    double step_length_x = 0.0;
    double step_length_y = 0.0;
    double step_rotation = 0.0;
    int n_steps = 10;
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
            labrob::Foot::RIGHT
        ),
        0.0,
        2000,
        labrob::WalkingState::Init
    ));
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
            labrob::Foot::RIGHT
        ),
        0.0,
        2000,
        labrob::WalkingState::Standing
    ));

    double double_support_duration = 20000;
    double single_support_duration = 20000;
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
            labrob::Foot::RIGHT
        ),
        0.0,
        double_support_duration,
        labrob::WalkingState::Starting
    ));

    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
            labrob::Foot::RIGHT
        ),
        swing_foot_trajectory_height,
        single_support_duration,
        labrob::WalkingState::SingleSupport
    ));
    for (int n = 0; n < n_steps; n += 2) {
        walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                labrob::SE3(labrob::Rz((n + 1) * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::SE3(labrob::Rz(n * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + n * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::Foot::RIGHT
            ),
            0.0,
            double_support_duration,
            labrob::WalkingState::DoubleSupport
        ));
        walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                labrob::SE3(labrob::Rz((n + 1) * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::SE3(labrob::Rz(n * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + n * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::Foot::LEFT
            ),
            swing_foot_trajectory_height,
            single_support_duration,
            labrob::WalkingState::SingleSupport
        ));
        walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                labrob::SE3(labrob::Rz((n + 1) * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::SE3(labrob::Rz((n + 2) * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + (n + 2) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::Foot::LEFT
            ),
            0.0,
            double_support_duration,
            labrob::WalkingState::DoubleSupport
        ));
        walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                labrob::SE3(labrob::Rz((n + 1) * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::SE3(labrob::Rz((n + 2) * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + (n + 2) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::Foot::RIGHT
            ),
            swing_foot_trajectory_height,
            single_support_duration,
            labrob::WalkingState::SingleSupport
        ));
    }
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(labrob::Rz(n_steps * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::SE3(labrob::Rz(n_steps * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::Foot::RIGHT
        ),
        0.0,
        0,
        labrob::WalkingState::Stopping
    ));
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(labrob::Rz(n_steps * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::SE3(labrob::Rz(n_steps * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::Foot::RIGHT
        ),
        0.0,
        2000,
        labrob::WalkingState::Standing
    ));

    // Save and read again footstep plan to double check it's working:
    //std::string footstep_plan_path = "/tmp/ditch-footstep-plan-argos.txt";
    //labrob::saveFootstepPlan(walking_data_.footstep_plan, footstep_plan_path);
    //labrob::readFootstepPlan(footstep_plan_path, walking_data_.footstep_plan);
    //labrob::readArgosFootstepPlan(footstep_plan_path, walking_data_.footstep_plan);

    // Init MPC:
    Eigen::Vector3d p_CoM = sim_robot_data.com[0];
    // std::cout << "CoM position: " << p_CoM.transpose() << std::endl;
    int64_t mpc_prediction_horizon_msec = 2000;
    int64_t mpc_timestep_msec = 100;
    double com_target_height = p_CoM.z() - T_lsole_init.translation().z();
    // std::cout << "CoM target height: " << com_target_height << std::endl;
    double foot_constraint_square_length = 100; //0.20;
    double foot_constraint_square_width = 100; //0.07;
    Eigen::Vector3d p_ZMP = p_CoM - Eigen::Vector3d(0.0, 0.0, com_target_height);
    sim_filt_LIPstate = labrob::LIPState(
        p_CoM,
        Eigen::Vector3d::Zero(),
        p_ZMP
    );
    sim_filt_LIPstate2 = labrob::LIPState(
        p_CoM,
        Eigen::Vector3d::Zero(),
        p_ZMP
    );
    ismpc_ptr_ = std::make_unique<labrob::ISMPC>(
        mpc_prediction_horizon_msec,
        mpc_timestep_msec,
        std::sqrt(9.81 / com_target_height),
        foot_constraint_square_length,
        foot_constraint_square_width
    );

    // DdpSolver ddpsolver = DdpSolver();

    // // set x0 as initial state of CoM_pos CoM_vel and ZMP_pos
    // Eigen::Vector<double, NX> x0;
    // x0 <<
    //     sim_filt_LIPstate.com_pos_(0),
    //     sim_filt_LIPstate.com_pos_(1),
    //     sim_filt_LIPstate.com_pos_(2),
    //     sim_filt_LIPstate.com_vel_(0),
    //     sim_filt_LIPstate.com_vel_(1),
    //     sim_filt_LIPstate.com_vel_(2),
    //     sim_filt_LIPstate.zmp_pos_(0),
    //     sim_filt_LIPstate.zmp_pos_(1),
    //     sim_filt_LIPstate.zmp_pos_(2);
    // std::array<Eigen::Vector<double, NX>, NH+1> x_traj;
    // x_traj[0] = x0;
    // std::array<Eigen::Vector<double, NU>, NH> u_traj;

    // // set warm-start trajectories
    // std::array<Eigen::Vector<double, NX>, NH+1> x_guess;
    // for (int i = 0; i < NH+1; ++i)
    //     x_guess[i] = x0;
    // std::array<Eigen::Vector<double, NU>, NH> u_guess;
    // for (int i = 0; i < NH; ++i)
    //     u_guess[i].setZero();
    // ddpsolver.set_initial_state(x0);
    // ddpsolver.set_x_warmstart(x_guess);
    // ddpsolver.set_u_warmstart(u_guess);

    auto params = WholeBodyControllerParams::getDefaultParams();
    whole_body_controller_ptr_ = std::make_shared<WholeBodyController>(
        params,
        sim_robot_model,
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

    model = fb_robot_model;
    data = fb_robot_data;

    K = Eigen::MatrixXd::Zero(2 * fb_robot_model.nv, n_ekf_output);
    std::ifstream kalman_gain_file("../mean_kalman_gain.txt");
    if (kalman_gain_file.is_open()) {
        for (int i = 0; i < K.rows(); i++) {
            for (int j = 0; j < K.cols(); j++) {
                kalman_gain_file >> K(i, j);
            }
        }
        kalman_gain_file.close();
    } else {
        std::cerr << "Unable to open file mean_kalman_gain.txt";
    }

    return true;
} 

void WalkingManager::updateEKF(RobotState current_state, bool useRobot, Eigen::VectorXd actual_output) {

    Eigen::Quaterniond q_orientation;
    Eigen::Vector3d q_rot_vec = x_estimate.segment<3>(3);  // x_estimate(3), (4), (5)
    double q_angle = q_rot_vec.norm();

    if (q_angle > M_PI) {
        q_angle -= 2 * M_PI;
    } else if (q_angle < -M_PI) {
        q_angle += 2 * M_PI;
    }

    if (std::abs(q_angle) < 1e-4) {
        q_orientation = Eigen::Quaterniond(1,0,0,0);  // nessuna rotazione
    } else {
        Eigen::Vector3d axis = q_rot_vec.normalized();
        q_orientation = Eigen::Quaterniond(Eigen::AngleAxisd(q_angle, axis));
    }

    Eigen::VectorXd q_estimate = Eigen::VectorXd::Zero(fb_robot_model.nq);
    q_estimate.head(3) = x_estimate.head(3);
    q_estimate.segment(3, 4) = Eigen::Vector4d(
        q_orientation.w(), q_orientation.x(), q_orientation.y(), q_orientation.z()
    );
    q_estimate.tail(fb_robot_model.nv - 6) = x_estimate.segment(3 + 3, fb_robot_model.nv - 6);

    pinocchio::forwardKinematics(model, data, q_estimate);
    pinocchio::framesForwardKinematics(model, data, q_estimate);
    pinocchio::jacobianCenterOfMass(model, data, q_estimate);
    pinocchio::computeJointJacobians(model, data, q_estimate);
    pinocchio::computeCentroidalMomentum(model, data, q_estimate, x_estimate.tail(fb_robot_model.nv));
    pinocchio::updateFramePlacements(model, data);

    Eigen::MatrixXd J_imu = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
    pinocchio::getFrameJacobian(
        model, 
        data, 
        fb_robot_model.getFrameId("imu_in_torso"), 
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
        J_imu
    );
    Eigen::MatrixXd J_imu_dot = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
    pinocchio::getFrameJacobianTimeVariation(
        model, 
        data, 
        fb_robot_model.getFrameId("imu_in_torso"), 
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
        J_imu_dot
    );

    Eigen::MatrixXd J_left_foot = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
    pinocchio::getFrameJacobian(
        model,
        data,
        fb_robot_model.getFrameId("left_foot_link"),
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_left_foot
    );

    Eigen::MatrixXd J_right_foot = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
    pinocchio::getFrameJacobian(
        model,
        data,
        fb_robot_model.getFrameId("right_foot_link"),
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_right_foot
    );

    double left_support_check = 1.0;
    double right_support_check = 1.0;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport){
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT){
            right_support_check = 0.0;
        }
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT){
            left_support_check = 0.0;
        }
    }

    //MATRICE C:

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_ekf_output, 2 * fb_robot_model.nv);
    C.block(0, 0, 3, fb_robot_model.nv) = J_imu.block(3, 0, 3, fb_robot_model.nv);
    C.block(3, 6, fb_robot_model.nv - 6, fb_robot_model.nv - 6) = Eigen::MatrixXd::Identity(fb_robot_model.nv - 6, fb_robot_model.nv - 6);
    C.block(fb_robot_model.nv - 3, fb_robot_model.nv, 3, fb_robot_model.nv) = J_imu.block(3, 0, 3, fb_robot_model.nv);
    C.block(fb_robot_model.nv, fb_robot_model.nv + 6, fb_robot_model.nv - 6, fb_robot_model.nv - 6) = Eigen::MatrixXd::Identity(fb_robot_model.nv - 6, fb_robot_model.nv - 6);
    C.block(2 * fb_robot_model.nv - 6, fb_robot_model.nv, 3, fb_robot_model.nv) = J_imu_dot.block(0, 0, 3, fb_robot_model.nv);
    C.block(2 * fb_robot_model.nv - 3, fb_robot_model.nv, 3, fb_robot_model.nv) = J_left_foot.block(0,0,3,fb_robot_model.nv);
    C.block(2 * fb_robot_model.nv, fb_robot_model.nv, 3, fb_robot_model.nv) = J_right_foot.block(0,0,3,fb_robot_model.nv);
    C.block(2 * fb_robot_model.nv + 3, 0, 3, fb_robot_model.nv) = J_left_foot.block(0, 0, 3, fb_robot_model.nv);
    C.block(2 * fb_robot_model.nv + 6, 0, 3, fb_robot_model.nv) = J_right_foot.block(0, 0, 3, fb_robot_model.nv);

    //MATRICE D:

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_ekf_output, fb_robot_model.nv);
    D.block(2*fb_robot_model.nv - 6, 0, 3, fb_robot_model.nv) = J_imu.block(0, 0, 3, fb_robot_model.nv);

    //MATRICE A:

    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2 * fb_robot_model.nv, 2 * fb_robot_model.nv);
    A.block(0, fb_robot_model.nv, fb_robot_model.nv, fb_robot_model.nv) = controller_timestep_msec_ * 0.001 * Eigen::MatrixXd::Identity(fb_robot_model.nv, fb_robot_model.nv);

    //PREDICTION COVARIANCE E KALMAN GAIN
    // Eigen::MatrixXd Lambda_ = A * P_ * A.transpose() + Q;
    // Eigen::MatrixXd K = Lambda_ * C.transpose() * (C * Lambda_ * C.transpose() + R).inverse();

    // Eigen::LLT<Eigen::MatrixXd> llt(C * Lambda_ * C.transpose() + R);
    // Eigen::MatrixXd MatInv = llt.solve(Eigen::MatrixXd::Identity(n_ekf_output, n_ekf_output));
    // Eigen::MatrixXd K = Lambda_ * C.transpose() * MatInv;

    // Eigen::MatrixXd S = C * Lambda_ * C.transpose() + R;   // innovation covariance
    // Eigen::MatrixXd K = Lambda_ * C.transpose() * S.ldlt().solve(Eigen::MatrixXd::Identity(S.rows(), S.cols()));

    // Eigen::MatrixXd S = C * Lambda_ * C.transpose() + R;
    // Eigen::MatrixXd K = Lambda_ * C.transpose();
    // K = S.ldlt().solve(K.transpose()).transpose();

    //save kalman gain matrix in a file
    // kalman_gain_matrix_log_file_ << K << std::endl;

    //take K from the file "mean_kalman_gain"
    


    // P_ = (Eigen::MatrixXd::Identity(2 * fb_robot_model.nv, 2 * fb_robot_model.nv) - K * C) * Lambda_;

    if (useRobot) {
        y_actual = actual_output;
    }
    else{
        // compute y_actual from current_state, y_actual is composed by 1) orientation of imu in axis angle 
        // 2) joint position 3) angular velocity of the imu 4) joint velocity of the robot
        // 5) accelerometer of the imu 6) velocity of feet 7) feet position

        Eigen::VectorXd q = robot_state_to_pinocchio_joint_configuration(
            fb_robot_model,
            current_state
        );

        Eigen::VectorXd qdot = robot_state_to_pinocchio_joint_velocity(
            fb_robot_model,
            current_state
        );

        Eigen::Quaterniond imu_orientation(
            fb_robot_data.oMf[fb_robot_model.getFrameId("imu_in_torso")].rotation()
        );

        Eigen::AngleAxisd axis_angle = Eigen::AngleAxisd(
            imu_orientation.w(),
            imu_orientation.vec()
        );

        if (axis_angle.angle() > M_PI) {
            axis_angle.angle() -= 2 * M_PI;
        } else if (axis_angle.angle() < -M_PI) {
            axis_angle.angle() += 2 * M_PI;
        }

        Eigen::MatrixXd J_imu = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
        pinocchio::getFrameJacobian(
            fb_robot_model, 
            fb_robot_data, 
            fb_robot_model.getFrameId("imu_in_torso"), 
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
            J_imu
        );

        Eigen::MatrixXd J_imu_dot = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
        pinocchio::getFrameJacobianTimeVariation(
            fb_robot_model, 
            fb_robot_data, 
            fb_robot_model.getFrameId("imu_in_torso"), 
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
            J_imu_dot
        );

        //compute angular velocity of imu
        Eigen::Vector3d imu_angular_velocity = J_imu.block(3, 0, 3, fb_robot_model.nv) * qdot;

        //get feet position
        Eigen::Vector3d left_foot_position = fb_robot_data.oMf[lsole_idx_].translation();
        Eigen::Vector3d right_foot_position = fb_robot_data.oMf[rsole_idx_].translation();

        y_actual.head(3) = Eigen::Vector3d(
            axis_angle.axis().x() * axis_angle.angle(),
            axis_angle.axis().y() * axis_angle.angle(),
            axis_angle.axis().z() * axis_angle.angle()
        );
        y_actual.segment(3, fb_robot_model.nv - 6) = q.tail(fb_robot_model.nv - 6);
        y_actual.segment(fb_robot_model.nv - 3, 3) = imu_angular_velocity;
        y_actual.segment(fb_robot_model.nv - 3 + 3, fb_robot_model.nv - 6) = qdot.tail(fb_robot_model.nv - 6);
        y_actual.segment(fb_robot_model.nv - 3 + fb_robot_model.nv - 3, 3) = J_imu.block(0, 0, 3, fb_robot_model.nv) * whole_body_controller_ptr_->get_q_ddot() + J_imu_dot.block(0, 0, 3, fb_robot_model.nv) * qdot;
        y_actual.segment(fb_robot_model.nv - 3 + fb_robot_model.nv - 3 + 6 + 3, 3) = left_foot_position*left_support_check;
        y_actual.segment(fb_robot_model.nv - 3 + fb_robot_model.nv - 3 + 6 + 3 + 3, 3) = right_foot_position*right_support_check;
    }


    if (!input_initialized){
        input = whole_body_controller_ptr_->get_q_ddot();
        input_initialized = true;
    }
    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(2 * fb_robot_model.nv);
    //compute x_pred using the pinocchio function
    Eigen::VectorXd integrated_state = pinocchio::integrate(
        fb_robot_model,
        q_estimate.head(fb_robot_model.nq),
        x_estimate.tail(fb_robot_model.nv) * 0.001 * controller_timestep_msec_ + 0.5 * (0.001 * controller_timestep_msec_) * (0.001 * controller_timestep_msec_) * whole_body_controller_ptr_->get_q_ddot()
    );
    x_pred.head(3) = integrated_state.head(3);
    x_pred.segment(3,3) = Eigen::AngleAxisd(
        Eigen::Quaterniond(integrated_state[3], integrated_state[4], integrated_state[5], integrated_state[6])
    ).axis() * Eigen::AngleAxisd(Eigen::Quaterniond(integrated_state[3], integrated_state[4], integrated_state[5], integrated_state[6])).angle();
    x_pred.segment(6, fb_robot_model.nv - 6) = integrated_state.tail(fb_robot_model.nv - 6);
    // x_pred.head(fb_robot_model.nv) = x_estimate.head(fb_robot_model.nv) + x_estimate.tail(fb_robot_model.nv)*0.001 * controller_timestep_msec_ + 0.5*(0.001 * controller_timestep_msec_)*(0.001 * controller_timestep_msec_) * whole_body_controller_ptr_->get_q_ddot();
    x_pred.tail(fb_robot_model.nv) = x_estimate.tail(fb_robot_model.nv) + whole_body_controller_ptr_->get_q_ddot() * controller_timestep_msec_ * 0.001;
    y_pred = y_estimate + C * (x_pred - x_estimate) + D * (whole_body_controller_ptr_->get_q_ddot() - input);
    y_pred.segment(n_ekf_output - 12, 3) = y_pred.segment(n_ekf_output - 12, 3)*left_support_check;
    y_pred.segment(n_ekf_output - 9, 3) = y_pred.segment(n_ekf_output - 9, 3)*right_support_check;
    y_pred.segment(n_ekf_output - 6, 3) = y_pred.segment(n_ekf_output - 6, 3)*left_support_check;
    y_pred.segment(n_ekf_output - 3, 3) = y_pred.segment(n_ekf_output - 3, 3)*right_support_check;

    //X ESTIMATE 
    Eigen::VectorXd x_estimate_prec = x_estimate;
    x_estimate = x_pred + K * (y_actual - y_pred);
    y_estimate = y_estimate + C*(x_estimate - x_estimate_prec) + D*(whole_body_controller_ptr_->get_q_ddot() - input);
    input = whole_body_controller_ptr_->get_q_ddot();
    current_state.position = x_estimate.head(3);
    
    //convert angle axis representation to quaternion representation
    //if the angle is too small, use identity quaternion

    Eigen::Quaterniond orientation;
    Eigen::Vector3d rot_vec = x_estimate.segment<3>(3);  // x_estimate(3), (4), (5)
    double angle = rot_vec.norm();

    if (angle > M_PI) {
        angle -= 2 * M_PI;
    } else if (angle < -M_PI) {
        angle += 2 * M_PI;
    }

    // std::cout << "Angle: " << angle << std::endl;

    if (std::abs(angle) < 1e-4) {
        orientation = Eigen::Quaterniond(1,0,0,0);  // nessuna rotazione
    } else {
        Eigen::Vector3d axis = rot_vec.normalized();
        orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
    }

    current_state.orientation = Eigen::Quaterniond(
        orientation.w(),
        orientation.x(),
        orientation.y(),
        orientation.z()
    );

    current_state.linear_velocity = x_estimate.segment(fb_robot_model.nv, 3);
    current_state.angular_velocity = x_estimate.segment(fb_robot_model.nv + 3, 3);

    // assign position and velocity for each joint
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) fb_robot_model.njoints; ++joint_id) {
        std::string joint_name = fb_robot_model.names[joint_id];
        current_state.joint_state[joint_name].pos = x_estimate(joint_id - 2 + 6);
        current_state.joint_state[joint_name].vel = x_estimate(fb_robot_model.njoints + joint_id - 2 + 6 + 6);
    }


    Eigen::Quaterniond predicted_imu_orientation;
    rot_vec = y_pred.head(3);  
    angle = rot_vec.norm();
    if (angle > M_PI) {
        angle -= 2 * M_PI;
    } else if (angle < -M_PI) {
        angle += 2 * M_PI;
    }
    if (std::abs(angle) < 1e-4) {
        predicted_imu_orientation = Eigen::Quaterniond(1,0,0,0);  // nessuna rotazione
    } else {
        Eigen::Vector3d axis = rot_vec.normalized();
        predicted_imu_orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
    }


    Eigen::Quaterniond fb_imu_orientation;
    rot_vec = y_actual.head(3);  
    angle = rot_vec.norm();
    if (angle > M_PI) {
        angle -= 2 * M_PI;
    } else if (angle < -M_PI) {
        angle += 2 * M_PI;
    }
    if (std::abs(angle) < 1e-4) {
        fb_imu_orientation = Eigen::Quaterniond(1,0,0,0);  // nessuna rotazione
    } else {
        Eigen::Vector3d axis = rot_vec.normalized();
        fb_imu_orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
    }

    //save values into arrays
    predicted_imu_orientation_log_.push_back(Eigen::Vector4d(
        predicted_imu_orientation.w(),
        predicted_imu_orientation.x(),
        predicted_imu_orientation.y(),
        predicted_imu_orientation.z()
    ));
    predicted_imu_angular_velocity_log_.push_back(y_pred.segment(3, 3));
    predicted_imu_accelerometer_log_.push_back(y_pred.segment(2*fb_robot_model.nv - 6, 3));
    fb_imu_orientation_log_.push_back(Eigen::Vector4d(
        fb_imu_orientation.w(),
        fb_imu_orientation.x(),
        fb_imu_orientation.y(),
        fb_imu_orientation.z()
    ));
    fb_imu_angular_velocity_log_.push_back(y_actual.segment(3, 3));
    fb_imu_accelerometer_log_.push_back(y_actual.segment(2*fb_robot_model.nv - 6, 3));

    // return current_state;
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
    labrob::RobotState& fb_robot_state,
    bool useRobot,
    Eigen::VectorXd actual_output
) {

    int njnt = sim_robot_model.nv - 6; // size of configuration space without floating base

    auto q = robot_state_to_pinocchio_joint_configuration(sim_robot_model, sim_robot_state);
    auto qdot = robot_state_to_pinocchio_joint_velocity(sim_robot_model, sim_robot_state);

    // Perform forward kinematics on the whole tree and update robot data:
    pinocchio::forwardKinematics(sim_robot_model, sim_robot_data, q);

    // // NOTE: jacobianCenterOfMass calls forwardKinematics and
    //       computeJointJacobians.
    pinocchio::jacobianCenterOfMass(sim_robot_model, sim_robot_data, q);
    pinocchio::computeJointJacobiansTimeVariation(sim_robot_model, sim_robot_data, q, qdot);
    pinocchio::framesForwardKinematics(sim_robot_model, sim_robot_data, q);
    pinocchio::centerOfMass(sim_robot_model, sim_robot_data, q, qdot, 0.0 * qdot); // This is used to compute the CoM drift (J_com_dot * qdot)
    const auto& centroidal_momentum_matrix = pinocchio::ccrba(
        sim_robot_model,
        sim_robot_data,
        q,
        qdot
    );

    auto angular_momentum = (centroidal_momentum_matrix * qdot).tail<3>();
    

    if (useRobot) {

        auto q_fb = robot_state_to_pinocchio_joint_configuration(fb_robot_model, fb_robot_state);
        auto qdot_fb = robot_state_to_pinocchio_joint_velocity(fb_robot_model, fb_robot_state);

        // Perform forward kinematics on the whole tree and update robot data:
        pinocchio::forwardKinematics(fb_robot_model, fb_robot_data, q_fb);

        // // NOTE: jacobianCenterOfMass calls forwardKinematics and
        //       computeJointJacobians.
        pinocchio::jacobianCenterOfMass(fb_robot_model, fb_robot_data, q_fb);
        pinocchio::computeJointJacobiansTimeVariation(fb_robot_model, fb_robot_data, q_fb, qdot_fb);
        pinocchio::framesForwardKinematics(fb_robot_model, fb_robot_data, q_fb);
        pinocchio::centerOfMass(fb_robot_model, fb_robot_data, q_fb, qdot_fb, 0.0 * qdot_fb); // This is used to compute the CoM drift (J_com_dot * qdot)
        const auto& centroidal_momentum_matrix = pinocchio::ccrba(
            fb_robot_model,
            fb_robot_data,
            q_fb,
            qdot_fb
        );

        const auto& p_CoM_fb = fb_robot_data.com[0];
        const auto& a_CoM_drift_fb = fb_robot_data.acom[0];
        const auto& J_CoM_fb = fb_robot_data.Jcom;
        const auto& T_torso_fb = fb_robot_data.oMf[torso_idx_];
        auto torso_orientation_fb = T_torso_fb.rotation();
        Eigen::MatrixXd J_torso_fb = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
        pinocchio::getFrameJacobian(
            fb_robot_model,
            fb_robot_data,
            torso_idx_,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
            J_torso_fb
        );

        const auto& T_lsole_fb = fb_robot_data.oMf[lsole_idx_];
        Eigen::MatrixXd J_lsole_fb = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
        pinocchio::getFrameJacobian(
            fb_robot_model,
            fb_robot_data,
            lsole_idx_,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
            J_lsole_fb
        );

        const auto& v_lsole_fb = J_lsole_fb * qdot_fb;

        const auto& T_rsole_fb = fb_robot_data.oMf[rsole_idx_];
        Eigen::MatrixXd J_rsole_fb = Eigen::MatrixXd::Zero(6, fb_robot_model.nv);
        pinocchio::getFrameJacobian(
            fb_robot_model,
            fb_robot_data,
            rsole_idx_,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
            J_rsole_fb
        );
        const auto& v_rsole_fb = J_rsole_fb * qdot_fb;

        // Update walking state:
        walking_data_.updateWalkingState(t_msec_);
        
        // Fill current gait configuration:
        labrob::GaitConfiguration real_current_gait_configuration;
        real_current_gait_configuration.qjnt = q_fb.tail(njnt);
        real_current_gait_configuration.qjntdot = qdot_fb.tail(njnt);

        real_current_gait_configuration.is_left_foot_support = true;
        real_current_gait_configuration.is_right_foot_support = true;
        if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
        if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT) real_current_gait_configuration.is_right_foot_support = false;
        else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT) real_current_gait_configuration.is_left_foot_support = false;
        }

        real_current_gait_configuration.com.pos = fb_robot_data.com[0];
        real_current_gait_configuration.com.vel = fb_robot_data.vcom[0];

        real_current_gait_configuration.torso.pos = fb_robot_data.oMf[torso_idx_].rotation();
        real_current_gait_configuration.torso.vel = J_torso_fb.bottomRows<3>() * qdot_fb;

        real_current_gait_configuration.lsole.pos = labrob::SE3(fb_robot_data.oMf[lsole_idx_].rotation(), fb_robot_data.oMf[lsole_idx_].translation());
        real_current_gait_configuration.lsole.vel = J_lsole_fb * qdot_fb;

        real_current_gait_configuration.rsole.pos = labrob::SE3(fb_robot_data.oMf[rsole_idx_].rotation(), fb_robot_data.oMf[rsole_idx_].translation());
        real_current_gait_configuration.rsole.vel = J_rsole_fb * qdot_fb;

        //log real com
        Eigen::Vector3d real_com_pos = fb_robot_data.com[0];
        real_com_log_.push_back(real_com_pos);

    }

    const auto& T_torso = sim_robot_data.oMf[torso_idx_];
    auto torso_orientation = T_torso.rotation();
    Eigen::MatrixXd J_torso = Eigen::MatrixXd::Zero(6, sim_robot_model.nv);
    pinocchio::getFrameJacobian(
        sim_robot_model,
        sim_robot_data,
        torso_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_torso
    );

    const auto& T_lsole = sim_robot_data.oMf[lsole_idx_];
    Eigen::MatrixXd J_lsole = Eigen::MatrixXd::Zero(6, sim_robot_model.nv);
    pinocchio::getFrameJacobian(
        sim_robot_model,
        sim_robot_data,
        lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_lsole
    );

    const auto& v_lsole = J_lsole * qdot;

    const auto& T_rsole = sim_robot_data.oMf[rsole_idx_];
    Eigen::MatrixXd J_rsole = Eigen::MatrixXd::Zero(6, sim_robot_model.nv);
    pinocchio::getFrameJacobian(
        sim_robot_model,
        sim_robot_data,
        rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_rsole
    );
    const auto& v_rsole = J_rsole * qdot;

    // save left and right foot position in last 6 places of actual output
    actual_output.segment(fb_robot_model.nv - 3 + fb_robot_model.nv - 3 + 3 + 6, 3) = T_lsole.translation();
    actual_output.segment(fb_robot_model.nv - 3 + fb_robot_model.nv - 3 + 3 + 6 + 3, 3) = T_rsole.translation();

    // Update walking state:
    walking_data_.updateWalkingState(t_msec_);

    // Fill current gait configuration:
    labrob::GaitConfiguration current_gait_configuration;
    current_gait_configuration.qjnt = q.tail(njnt);
    current_gait_configuration.qjntdot = qdot.tail(njnt);

    current_gait_configuration.is_left_foot_support = true;
    current_gait_configuration.is_right_foot_support = true;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
    if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT) current_gait_configuration.is_right_foot_support = false;
    else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT) current_gait_configuration.is_left_foot_support = false;
    }

    current_gait_configuration.com.pos = sim_robot_data.com[0];
    current_gait_configuration.com.vel = sim_robot_data.vcom[0];

    current_gait_configuration.torso.pos = sim_robot_data.oMf[torso_idx_].rotation();
    current_gait_configuration.torso.vel = J_torso.bottomRows<3>() * qdot;

    current_gait_configuration.lsole.pos = labrob::SE3(sim_robot_data.oMf[lsole_idx_].rotation(), sim_robot_data.oMf[lsole_idx_].translation());
    current_gait_configuration.lsole.vel = J_lsole * qdot;

    current_gait_configuration.rsole.pos = labrob::SE3(sim_robot_data.oMf[rsole_idx_].rotation(), sim_robot_data.oMf[rsole_idx_].translation());
    current_gait_configuration.rsole.vel = J_rsole * qdot;

    double eta2 = std::pow(ismpc_ptr_->getOmega(), 2.0);
    double mass = pinocchio::computeTotalMass(sim_robot_model);
    // Eigen::Vector3d lip_zmp = p_CoM - robot_state.total_force / (mass * eta2);

    if (!useRobot) {
        fb_robot_model = sim_robot_model;
        fb_robot_data = sim_robot_data;
    }
    
    RobotState fb_filt_robot_state;

    LIPState sim_LIPstate;

    //start measuring time
    auto t_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        {
            // EKF
            fb_filt_robot_state = fb_robot_state;
            updateEKF(fb_filt_robot_state, useRobot, actual_output);
        }
        #pragma omp section
        {
        }
    } // end of parallel sections
    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    execution_time_ekf_log_.push_back(t_duration);

    
    if (useRobot && t_msec_ > 6000) {
        if(t_msec_ == 6002){
            std::cout << "STARTING CLOSED-LOOP SIM"<< std::endl;
        }

        const auto& fb_p_CoM = fb_robot_data.com[0];
        const auto& J_CoM = fb_robot_data.Jcom;
        Eigen::Vector3d zmp_3d;
        zmp_3d.z() = fb_filt_robot_state.position(2) - fb_filt_robot_state.total_force.z() / (mass * eta2);
        zmp_3d.x() = 0.0;
        zmp_3d.y() = 0.0;
        for (int i = 0; i < fb_filt_robot_state.contact_points.size(); ++i) {
            auto &pi = fb_filt_robot_state.contact_points[i];
            auto &fi = fb_filt_robot_state.contact_forces[i];
            zmp_3d.x() += (pi.x() * fi.z() / fb_filt_robot_state.total_force.z() + (zmp_3d.z() - pi.z()) * fi.x() / fb_filt_robot_state.total_force.z());
            zmp_3d.y() += (pi.y() * fi.z() / fb_filt_robot_state.total_force.z() + (zmp_3d.z() - pi.z()) * fi.y() / fb_filt_robot_state.total_force.z());
        }
    
        auto qdot_fb = robot_state_to_pinocchio_joint_velocity(fb_robot_model, fb_filt_robot_state);
    
        fb_LIPstate = LIPState(fb_p_CoM, J_CoM * qdot_fb, zmp_3d);
    
        // fb_filt_LIPstate = updateKF(fb_filt_LIPstate, fb_LIPstate, ismpc_ptr_->getInput());
    }

    const auto& p_CoM = sim_robot_data.com[0];
    const auto& J_CoM = sim_robot_data.Jcom;
    Eigen::Vector3d zmp_3d;
    zmp_3d.z() = sim_robot_state.position(2) - sim_robot_state.total_force.z() / (mass * eta2);
    zmp_3d.x() = 0.0;
    zmp_3d.y() = 0.0;
    for (int i = 0; i < sim_robot_state.contact_points.size(); ++i) {
        auto &pi = sim_robot_state.contact_points[i];
        auto &fi = sim_robot_state.contact_forces[i];
        zmp_3d.x() += (pi.x() * fi.z() / sim_robot_state.total_force.z() + (zmp_3d.z() - pi.z()) * fi.x() / sim_robot_state.total_force.z());
        zmp_3d.y() += (pi.y() * fi.z() / sim_robot_state.total_force.z() + (zmp_3d.z() - pi.z()) * fi.y() / sim_robot_state.total_force.z());
    }

    t_start = std::chrono::high_resolution_clock::now();
    if(!useRobot || t_msec_ >= 0) {
        sim_LIPstate = LIPState(p_CoM, J_CoM * robot_state_to_pinocchio_joint_velocity(sim_robot_model, sim_robot_state), zmp_3d);

        sim_filt_LIPstate = updateKF(sim_filt_LIPstate, sim_LIPstate, ismpc_ptr_->getInput());
    }
    t_end = std::chrono::high_resolution_clock::now();
    t_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
    execution_time_kf_log_.push_back(t_duration);

    // CoM task:
    auto mpc_t0_ms = std::chrono::system_clock::now();
    ismpc_ptr_->solve(t_msec_, walking_data_, sim_filt_LIPstate); //change sim to fb to close the loop
    auto mpc_tf_ms = std::chrono::system_clock::now();
    auto mpc_duration = std::chrono::duration_cast<std::chrono::microseconds>(mpc_tf_ms - mpc_t0_ms).count();
    execution_time_mpc_log_.push_back(mpc_duration);

    // DdpSolver ddpsolver = DdpSolver();

    // // set x0 as initial state of CoM_pos CoM_vel and ZMP_pos
    // Eigen::Vector<double, NX> x0;
    // x0 <<
    //     sim_filt_LIPstate.com_pos_(0),
    //     sim_filt_LIPstate.com_pos_(1),
    //     sim_filt_LIPstate.com_pos_(2),
    //     sim_filt_LIPstate.com_vel_(0),
    //     sim_filt_LIPstate.com_vel_(1),
    //     sim_filt_LIPstate.com_vel_(2),
    //     sim_filt_LIPstate.zmp_pos_(0),
    //     sim_filt_LIPstate.zmp_pos_(1),
    //     sim_filt_LIPstate.zmp_pos_(2);
    // std::array<Eigen::Vector<double, NX>, NH+1> x_traj;
    // x_traj[0] = x0;
    // std::array<Eigen::Vector<double, NU>, NH> u_traj;
  
    // // set warm-start trajectories
    // std::array<Eigen::Vector<double, NX>, NH+1> x_guess;
    // for (int i = 0; i < NH+1; ++i)
    //   x_guess[i] = x0;
    // std::array<Eigen::Vector<double, NU>, NH> u_guess;
    // for (int i = 0; i < NH; ++i)
    //   u_guess[i].setZero();
    // ddpsolver.set_initial_state(x0);
    // ddpsolver.set_x_warmstart(x_guess);
    // ddpsolver.set_u_warmstart(u_guess);
    // ddpsolver.solve();


    // Update the state based on the result of the QP:
    auto lip_state = discrete_lip_dynamics_ptr_->integrate(sim_filt_LIPstate, ismpc_ptr_->getInput());
    // substitute with ddpsolver.u[0]

    Eigen::VectorXd inputSequenceX = ismpc_ptr_->getInputSequenceX();
    Eigen::VectorXd inputSequenceY = ismpc_ptr_->getInputSequenceY();
    Eigen::VectorXd inputSequenceZ = ismpc_ptr_->getInputSequenceZ();

    LIPState sim_LIPstate_mpc = sim_filt_LIPstate;

    // for (int i = 0; i < 20; ++i) {
    //     sim_LIPstate_mpc = discrete_lip_dynamics_ptr_mpc_->integrate(
    //         sim_LIPstate_mpc, 
    //         Eigen::Vector3d(inputSequenceX(i), inputSequenceY(i), inputSequenceZ(i))
    //     );

    //     mpc_predictions_log_.push_back(sim_LIPstate_mpc.com_pos_);
    //     mpc_predictions_vel_log_.push_back(sim_LIPstate_mpc.com_vel_);
    //     mpc_predictions_zmp_log_.push_back(sim_LIPstate_mpc.zmp_pos_);
    // }


    Eigen::Vector3d v_CoM_des = lip_state.com_vel_;
    Eigen::Vector3d p_CoM_des = lip_state.com_pos_;
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

    // if (t_msec_ == 2000) {

    //     initial_gait_configuration = desired_gait_configuration;

    // }
    // else if (t_msec_ < 2000) {
    //     //initial_gait_configuration = desired_gait_configuration;
    // }
    // else {
    //     desired_gait_configuration = initial_gait_configuration;
    //     double A = 0.05;
    //     double f = 3;
    //     desired_gait_configuration.com.pos = initial_gait_configuration.com.pos + Eigen::Vector3d(
    //         0.0,
    //         A * std::sin(f * 0.001 * (t_msec_ - 2000)),
    //         0.0
    //     );
    //     desired_gait_configuration.com.vel = initial_gait_configuration.com.vel + Eigen::Vector3d(
    //         0.0,
    //         f * A * std::cos(f *0.001 * (t_msec_ - 2000)),
    //         0.0
    //     );
    //     desired_gait_configuration.com.acc = initial_gait_configuration.com.acc + Eigen::Vector3d(
    //         0.0,
    //         - f * f* A * std::sin(f * 0.001 * (t_msec_ - 2000)),
    //         0.0
    //     );
    // }



    auto start = std::chrono::system_clock::now();
    if (t_msec_ > 20000 && false) {
        // Use the robot feedback to compute the joint command:
        if(t_msec_ == 20002){
            std::cout << "SWITCHING TO FEEDBACK CONTROL"<< std::endl;
        }
        joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
            fb_robot_model,
            fb_filt_robot_state,
            fb_robot_data,
            current_gait_configuration,
            desired_gait_configuration
        );
    } else {
        // Use the MPC to compute the joint command:
        joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
            sim_robot_model,
            sim_robot_state,
            sim_robot_data,
            current_gait_configuration,
            desired_gait_configuration
        );
    }
    auto end = std::chrono::system_clock::now();
    auto compute_id_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    execution_time_wbc_log_.push_back(compute_id_duration);


    // Update timing in milliseconds.
    // NOTE: assuming update() is actually called every controller_timestep_msec_
    //       milliseconds.
    t_msec_ += controller_timestep_msec_;
    prev_angular_momentum_ = angular_momentum;

    mpc_timings_log_.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(mpc_tf_ms - mpc_t0_ms).count()
    );
    mpc_com_log_.push_back(p_CoM_des);
    mpc_zmp_log_.push_back(p_ZMP_des);
    com_log_.push_back(p_CoM);
    p_lsole_log_.push_back(T_lsole.translation().transpose());
    p_rsole_log_.push_back(T_rsole.translation().transpose());
    v_lsole_log_.push_back(v_lsole.head<3>().transpose());
    v_rsole_log_.push_back(v_rsole.head<3>().transpose());
    p_lsole_des_log_.push_back(desired_gait_configuration.lsole.pos.p.transpose());
    p_rsole_des_log_.push_back(desired_gait_configuration.rsole.pos.p.transpose());
    v_lsole_des_log_.push_back(desired_gait_configuration.lsole.vel.head<3>().transpose());
    v_rsole_des_log_.push_back(desired_gait_configuration.rsole.vel.head<3>().transpose());
    angular_momentum_log_.push_back(angular_momentum.transpose());
    cop_computed_log_.push_back(Eigen::VectorXd(9));
    cop_computed_log_.back().segment(0, 3) = sim_LIPstate.zmp_pos_.transpose();
    cop_computed_log_.back().segment(3, 3) = sim_filt_LIPstate.zmp_pos_.transpose();
    cop_computed_log_.back().segment(6, 3) = zmp_3d.transpose();
    // Log the filtered state:
    ekf_base_position_log_.push_back(fb_filt_robot_state.position.transpose());
    ekf_base_velocity_log_.push_back(fb_filt_robot_state.linear_velocity.transpose());
    ekf_base_orientation_log_.push_back(Eigen::Vector4d(
        fb_filt_robot_state.orientation.w(),
        fb_filt_robot_state.orientation.x(),
        fb_filt_robot_state.orientation.y(),
        fb_filt_robot_state.orientation.z()
    ).transpose());
    ekf_base_angular_velocity_log_.push_back(fb_filt_robot_state.angular_velocity.transpose());
    ekf_joint_position_log_.push_back(Eigen::VectorXd(njnt).transpose());
    ekf_joint_velocity_log_.push_back(Eigen::VectorXd(njnt).transpose());   
    for (pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) fb_robot_model.njoints; ++joint_id) {
        std::string joint_name = fb_robot_model.names[joint_id];
        ekf_joint_position_log_.back()(joint_id - 2) = fb_filt_robot_state.joint_state[joint_name].pos;
        ekf_joint_velocity_log_.back()(joint_id - 2) = fb_filt_robot_state.joint_state[joint_name].vel;
    }
    // Log the simulated state:
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
    for (pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) sim_robot_model.njoints; ++joint_id) {
        std::string joint_name = sim_robot_model.names[joint_id];
        sim_joint_position_log_.back()(joint_id - 2) = sim_robot_state.joint_state[joint_name].pos;
        sim_joint_velocity_log_.back()(joint_id - 2) = sim_robot_state.joint_state[joint_name].vel;
    }
    // Log the actual state:
    fb_base_position_log_.push_back(fb_robot_state.position.transpose());
    fb_base_velocity_log_.push_back(fb_robot_state.linear_velocity.transpose());
    fb_base_orientation_log_.push_back(Eigen::Vector4d(
        fb_robot_state.orientation.w(),
        fb_robot_state.orientation.x(),
        fb_robot_state.orientation.y(),
        fb_robot_state.orientation.z()
    ).transpose());
    fb_base_angular_velocity_log_.push_back(fb_robot_state.angular_velocity.transpose()); 
    fb_joint_position_log_.push_back(Eigen::VectorXd(njnt).transpose());
    fb_joint_velocity_log_.push_back(Eigen::VectorXd(njnt).transpose());
    for (pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) fb_robot_model.njoints; ++joint_id) {
        std::string joint_name = fb_robot_model.names[joint_id];
        fb_joint_position_log_.back()(joint_id - 2) = fb_robot_state.joint_state[joint_name].pos;
        fb_joint_velocity_log_.back()(joint_id - 2) = fb_robot_state.joint_state[joint_name].vel;
    }
}

void WalkingManager::saveLogs() {

    std::cout << "Saving logs..." << std::endl;

    std::ofstream mpc_timings_file("/tmp/mpc_timings.txt");
    for (auto& t : mpc_timings_log_) {
        mpc_timings_file << t << "\n";
    }

    std::ofstream mpc_com_file("/tmp/mpc_com.txt");
    for (auto& v : mpc_com_log_) {
        mpc_com_file << v.transpose() << "\n";
    }

    std::ofstream mpc_zmp_file("/tmp/mpc_zmp.txt");
    for (auto& v : mpc_zmp_log_) {
        mpc_zmp_file << v.transpose() << "\n";
    }

    std::ofstream com_file("/tmp/com.txt");
    for (auto& v : com_log_) {
        com_file << v.transpose() << "\n";
    }

    std::ofstream p_lsole_file("/tmp/p_lsole.txt");
    for (auto& v : p_lsole_log_) {
        p_lsole_file << v.transpose() << "\n";
    }
    std::ofstream p_rsole_file("/tmp/p_rsole.txt");
    for (auto& v : p_rsole_log_) {
        p_rsole_file << v.transpose() << "\n";
    }
    
    std::ofstream v_lsole_file("/tmp/v_lsole.txt");
    for (auto& v : v_lsole_log_) {
        v_lsole_file << v.transpose() << "\n";
    }

    std::ofstream v_rsole_file("/tmp/v_rsole.txt");
    for (auto& v : v_rsole_log_) {
        v_rsole_file << v.transpose() << "\n";
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

    std::ofstream angular_momentum_file("/tmp/angular_momentum.txt");
    for (auto& v : angular_momentum_log_) {
        angular_momentum_file << v.transpose() << "\n";
    }

    std::ofstream cop_computed_file("/tmp/cop_computed.txt");
    for (auto& v : cop_computed_log_) {
        cop_computed_file << v.transpose() << "\n";
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

    std::ofstream predicted_imu_accelerometer_file("/tmp/predicted_imu_accelerometer.txt");
    for (auto& v : predicted_imu_accelerometer_log_) {
        predicted_imu_accelerometer_file << v.transpose() << "\n";
    }

    std::ofstream predicted_imu_angular_velocity_file("/tmp/predicted_imu_angular_velocity.txt");
    for (auto& v : predicted_imu_angular_velocity_log_) {
        predicted_imu_angular_velocity_file << v.transpose() << "\n";
    }

    std::ofstream predicted_imu_orientation_file("/tmp/predicted_imu_orientation.txt");
    for (auto& v : predicted_imu_orientation_log_) {
        predicted_imu_orientation_file << v.transpose() << "\n";
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

void
WalkingManager::swingFootTrajectoryBezier(
    pinocchio::SE3& swing_foot_pose,
    pinocchio::Motion& swing_foot_velocity
) const {
  // NOTE: assuming there are at least two elements in the footstep plan.
  // NOTE: assuming roll and pitch are always zero for the swing foot.
  double t = 0.001 * static_cast<double>(t_msec_ - walking_data_.t0 + controller_timestep_msec_);
  double swing_duration = 0.001 * static_cast<double>(walking_data_.footstep_plan.front().getDuration());
  labrob::QuinticPolynomialTimingLaw timing_law(swing_duration);
  double s = timing_law.eval(t);
  double s_dot = timing_law.eval_dt(t);

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

  double h_z = walking_data_.footstep_plan[0].getSwingFootTrajectoryHeight();
  double z_max = support_foot_configuration.p.z() + h_z;
  auto p1 = Eigen::Vector3d(p0.x(), p0.y(), z_max);
  auto p2 = Eigen::Vector3d(pf.x(), pf.y(), z_max);

  pinocchio::SE3 desired_swing_foot_pose;
  desired_swing_foot_pose.translation() =
      std::pow(1.0 - s, 3.0) * p0 +
      3.0 * std::pow(1.0 - s, 2.0) * s * p1 +
      3.0 * (1.0 - s) * std::pow(s, 2.0) * p2 +
      std::pow(s, 3.0) * pf;
  double desired_swing_foot_yaw = yaw0 + angle_difference(yawf, yaw0) * s;
  desired_swing_foot_pose.rotation() = Rz(desired_swing_foot_yaw);

  pinocchio::Motion desired_swing_foot_velocity(
      (3.0 * std::pow(1.0 - s, 2.0) * (p1 - p0) + 6.0 * (1.0 - s) * s * (p2 - p1) + 3.0 * std::pow(t, 2.0) * (pf - p2)) * s_dot,
      Eigen::Vector3d(0.0, 0.0, angle_difference(yawf, yaw0)) * s_dot
  );

  swing_foot_pose = desired_swing_foot_pose;
  swing_foot_velocity = desired_swing_foot_velocity;
}

} // end namespace labrob
