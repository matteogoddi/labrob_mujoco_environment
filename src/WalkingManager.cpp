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
    filtered_state_(Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero(),
                    Eigen::Vector3d::Zero()),
    filtered_state2_(Eigen::Vector3d::Zero(),
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

    robot_model_ = pinocchio::buildReducedModel(
        full_robot_model,
        joint_ids_to_lock,
        pinocchio::neutral(full_robot_model)
    );
    robot_data_ = pinocchio::Data(robot_model_);

    // Init desired lsole and rsole poses:
    auto q_init = robot_state_to_pinocchio_joint_configuration(
        robot_model_,
        initial_robot_state
    );
    auto qdot_init = robot_state_to_pinocchio_joint_velocity(
        robot_model_,
        initial_robot_state
    );
    pinocchio::forwardKinematics(robot_model_, robot_data_, q_init);
    pinocchio::jacobianCenterOfMass(robot_model_, robot_data_, q_init);
    pinocchio::framesForwardKinematics(robot_model_, robot_data_, q_init);


    if (useRobot) {
    
        real_model_ = pinocchio::buildReducedModel(
            full_robot_model,
            joint_ids_to_lock,
            pinocchio::neutral(full_robot_model)
        );
        real_data_ = pinocchio::Data(real_model_);

        pinocchio::forwardKinematics(real_model_, real_data_, q_init);
        pinocchio::jacobianCenterOfMass(real_model_, real_data_, q_init);
        pinocchio::framesForwardKinematics(real_model_, real_data_, q_init);

    } else {
        real_model_ = robot_model_;
        real_data_ = robot_data_;
    }


    n_ekf_output = real_model_.nv - 3 + real_model_.nv - 3 + 3 + 6 + 6;

    P_ = Eigen::MatrixXd::Identity(2 * real_model_.nv, 2 * real_model_.nv) * 1;
    P_.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-6;
    P_.block(real_model_.nv, real_model_.nv, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-6;
    P_.block(3,3,3,3) = Eigen::MatrixXd::Identity(3, 3) * 1e-2;
    P_.block(real_model_.nv + 3, real_model_.nv + 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 1e-2;
    x_estimate = Eigen::VectorXd::Zero(2 * real_model_.nv);
    x_estimate.head(3) = q_init.head(3);
    x_estimate.segment(3, 3) = Eigen::AngleAxisd(
        Eigen::Quaterniond(q_init[6], q_init[3], q_init[4], q_init[5])
    ).axis() * Eigen::AngleAxisd(Eigen::Quaterniond(q_init[6], q_init[3], q_init[4], q_init[5])).angle();
    x_estimate.segment(3 + 3, real_model_.nv - 6) = q_init.tail(real_model_.nv - 6);
    x_estimate.tail(real_model_.nv) = qdot_init;
    y_pred = Eigen::VectorXd::Zero(n_ekf_output);
    y_actual = Eigen::VectorXd::Zero(n_ekf_output);
    y_estimate = Eigen::VectorXd::Zero(n_ekf_output);

    lsole_idx_ = robot_model_.getFrameId("left_foot_link");
    rsole_idx_ = robot_model_.getFrameId("right_foot_link");
    torso_idx_ = robot_model_.getFrameId("torso_link");
    const auto& T_lsole_init = robot_data_.oMf[lsole_idx_];
    const auto& T_rsole_init = robot_data_.oMf[rsole_idx_];

    Eigen::Quaterniond imu_orientation(
        real_data_.oMf[real_model_.getFrameId("imu_in_torso")].rotation()
    );

    std::cout << "imu orientation: " << imu_orientation.coeffs().transpose() << std::endl;
    std::cout << "imu orientation: " << imu_orientation.vec() << std::endl;

    Eigen::AngleAxisd axis_angle_init = Eigen::AngleAxisd(
        imu_orientation.w(),
        imu_orientation.vec()
    );

    Eigen::MatrixXd J_imu = Eigen::MatrixXd::Zero(6, real_model_.nv);
    pinocchio::getFrameJacobian(
        real_model_,
        real_data_,
        real_model_.getFrameId("imu_in_torso"),
        pinocchio::LOCAL_WORLD_ALIGNED,
        J_imu
    );

    //compute angular velocity of imu
    Eigen::Vector3d imu_angular_velocity = J_imu.block(3, 0, 3, real_model_.nv) * qdot_init;

    //get feet position
    Eigen::Vector3d left_foot_position = real_data_.oMf[lsole_idx_].translation();
    Eigen::Vector3d right_foot_position = real_data_.oMf[rsole_idx_].translation();

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
    y_estimate.segment(3, real_model_.nv - 6) = q_init.tail(real_model_.nv - 6);
    y_estimate.segment(real_model_.nv - 3, 3) = imu_angular_velocity;
    y_estimate.segment(real_model_.nv - 3 + 3, real_model_.nv - 6) = qdot_init.tail(real_model_.nv - 6);
    // y_estimate.segment(real_model_.nv - 3 + real_model_.nv - 3, 3) = J_imu_dot.block(0, 0, 3, real_model_.nv) * qdot_init;
    y_estimate.segment(real_model_.nv - 3 + real_model_.nv - 3 + 6 + 3, 3) = left_foot_position;
    y_estimate.segment(real_model_.nv - 3 + real_model_.nv - 3 + 6 + 3 + 3, 3) = right_foot_position;
    

    input = Eigen::VectorXd::Zero(real_model_.nv);
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
    x_estimate.segment(3 + 3, real_model_.nv - 6) = q_init.tail(real_model_.nv - 6);
    x_estimate.tail(real_model_.nv) = qdot_init;
    Q = Eigen::MatrixXd::Identity(2 * real_model_.nv, 2 * real_model_.nv) * 1e-2;
    R = Eigen::MatrixXd::Identity(n_ekf_output, n_ekf_output) * 1e-2;

    int njnt = robot_model_.nv - 6;

    M_armature_ = Eigen::VectorXd::Zero(njnt);
    for(pinocchio::JointIndex joint_id = 2;
        joint_id < (pinocchio::JointIndex) robot_model_.njoints;
        ++joint_id) {
        std::string joint_name = robot_model_.names[joint_id];
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
    controller_frequency_ = 1000;
    controller_timestep_msec_ = 1000 / controller_frequency_;

    double swing_foot_trajectory_height = 0.05;
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

    double double_support_duration = 8000;
    double single_support_duration = 8000;
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
    Eigen::Vector3d p_CoM = robot_data_.com[0];
    int64_t mpc_prediction_horizon_msec = 2000;
    int64_t mpc_timestep_msec = 100;
    double com_target_height = p_CoM.z() - T_lsole_init.translation().z();
    double foot_constraint_square_length = 100; //0.20;
    double foot_constraint_square_width = 100; //0.07;
    Eigen::Vector3d p_ZMP = p_CoM - Eigen::Vector3d(0.0, 0.0, com_target_height);
    filtered_state_ = labrob::LIPState(
        p_CoM,
        Eigen::Vector3d::Zero(),
        p_ZMP
    );
    filtered_state2_ = labrob::LIPState(
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

    DdpSolver ddpsolver = DdpSolver();

    // set x0 as initial state of CoM_pos CoM_vel and ZMP_pos
    Eigen::Vector<double, NX> x0;
    x0 <<
        filtered_state_.com_pos_(0),
        filtered_state_.com_pos_(1),
        filtered_state_.com_pos_(2),
        filtered_state_.com_vel_(0),
        filtered_state_.com_vel_(1),
        filtered_state_.com_vel_(2),
        filtered_state_.zmp_pos_(0),
        filtered_state_.zmp_pos_(1),
        filtered_state_.zmp_pos_(2);
    std::array<Eigen::Vector<double, NX>, NH+1> x_traj;
    x_traj[0] = x0;
    std::array<Eigen::Vector<double, NU>, NH> u_traj;

    // set warm-start trajectories
    std::array<Eigen::Vector<double, NX>, NH+1> x_guess;
    for (int i = 0; i < NH+1; ++i)
        x_guess[i] = x0;
    std::array<Eigen::Vector<double, NU>, NH> u_guess;
    for (int i = 0; i < NH; ++i)
        u_guess[i].setZero();
    ddpsolver.set_initial_state(x0);
    ddpsolver.set_x_warmstart(x_guess);
    ddpsolver.set_u_warmstart(u_guess);

    auto params = WholeBodyControllerParams::getDefaultParams();
    whole_body_controller_ptr_ = std::make_shared<WholeBodyController>(
        params,
        robot_model_,
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
        0.1
    );

    // Init log files:
    // TODO: may be better to use a proper logging system such as glog.
    mpc_timings_log_file_.open("/tmp/mpc_timings.txt");
    mpc_timings_log_file_.open("/tmp/mpc_timings.txt");
    mpc_com_log_file_.open("/tmp/mpc_com.txt");
    mpc_zmp_log_file_.open("/tmp/mpc_zmp.txt");
    //configuration_log_file_.open("/tmp/configuration.txt");
    com_log_file_.open("/tmp/com.txt");
    p_lsole_log_file_.open("/tmp/p_lsole.txt");
    p_rsole_log_file_.open("/tmp/p_rsole.txt");
    v_lsole_log_file_.open("/tmp/v_lsole.txt");
    v_rsole_log_file_.open("/tmp/v_rsole.txt");
    p_lsole_des_log_file_.open("/tmp/p_lsole_des.txt");
    p_rsole_des_log_file_.open("/tmp/p_rsole_des.txt");
    v_lsole_des_log_file_.open("/tmp/v_lsole_des.txt");
    v_rsole_des_log_file_.open("/tmp/v_rsole_des.txt");
    angular_momentum_log_file_.open("/tmp/angular_momentum.txt");
    //   fl_log_file_.open("/tmp/fl.txt");
    //   fr_log_file_.open("/tmp/fr.txt");
    cop_computed_log_file_.open("/tmp/cop_computed.txt");
    mpc_predictions_log_file_.open("/tmp/mpc_predictions.txt");
    ekf_base_position_log_file_.open("/tmp/ekf_base_position.txt");
    ekf_base_velocity_log_file_.open("/tmp/ekf_base_velocity.txt");
    ekf_base_orientation_log_file_.open("/tmp/ekf_base_orientation.txt");
    ekf_base_angular_velocity_log_file_.open("/tmp/ekf_base_angular_velocity.txt");
    ekf_joint_position_log_file_.open("/tmp/ekf_joint_position.txt");
    ekf_joint_velocity_log_file_.open("/tmp/ekf_joint_velocity.txt");
    base_position_log_file_.open("/tmp/base_position.txt");
    base_velocity_log_file_.open("/tmp/base_velocity.txt");
    base_orientation_log_file_.open("/tmp/base_orientation.txt");
    base_angular_velocity_log_file_.open("/tmp/base_angular_velocity.txt");
    real_com_log_file_.open("/tmp/real_com.txt");
    predicted_imu_accelerometer_log_file_.open("/tmp/predicted_imu_accelerometer.txt");
    predicted_imu_angular_velocity_log_file_.open("/tmp/predicted_imu_angular_velocity.txt");
    predicted_imu_orientation_log_file_.open("/tmp/predicted_imu_orientation.txt");

    return true;
} 

RobotState WalkingManager::updateEKF(RobotState current_state, bool useRobot, Eigen::VectorXd actual_output) {

    Eigen::Quaterniond q_orientation;
    Eigen::Vector3d q_rot_vec = x_estimate.segment<3>(3);  // x_estimate(3), (4), (5)
    double q_angle = q_rot_vec.norm();

    std::cout << "q_angle: " << q_angle << std::endl;

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

    Eigen::VectorXd q_estimate = Eigen::VectorXd::Zero(real_model_.nq);
    q_estimate.head(3) = x_estimate.head(3);
    q_estimate.segment(3, 4) = Eigen::Vector4d(
        q_orientation.w(), q_orientation.x(), q_orientation.y(), q_orientation.z()
    );
    q_estimate.tail(real_model_.nv - 6) = x_estimate.segment(3 + 3, real_model_.nv - 6);
    //create a pinocchio robot model based on x_estimate and data
    pinocchio::Model model(real_model_);
    pinocchio::Data data(model);
    pinocchio::forwardKinematics(model, data, q_estimate);
    pinocchio::framesForwardKinematics(model, data, q_estimate);
    pinocchio::computeJointJacobians(model, data, q_estimate);
    pinocchio::updateFramePlacements(model, data);

    Eigen::MatrixXd J_imu = Eigen::MatrixXd::Zero(6, real_model_.nv);
    pinocchio::getFrameJacobian(
        model, 
        data, 
        real_model_.getFrameId("imu_in_torso"), 
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
        J_imu
    );
    Eigen::MatrixXd J_imu_dot = Eigen::MatrixXd::Zero(6, real_model_.nv);
    pinocchio::getFrameJacobianTimeVariation(
        model, 
        data, 
        real_model_.getFrameId("imu_in_torso"), 
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
        J_imu_dot
    );

    Eigen::MatrixXd J_left_foot = Eigen::MatrixXd::Zero(6, real_model_.nv);
    pinocchio::getFrameJacobian(
        model,
        data,
        real_model_.getFrameId("left_foot_link"),
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_left_foot
    );

    Eigen::MatrixXd J_right_foot = Eigen::MatrixXd::Zero(6, real_model_.nv);
    pinocchio::getFrameJacobian(
        model,
        data,
        real_model_.getFrameId("right_foot_link"),
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

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n_ekf_output, 2 * real_model_.nv);
    C.block(0, 0, 3, real_model_.nv) = J_imu.block(3, 0, 3, real_model_.nv);
    C.block(3, 6, real_model_.nv - 6, real_model_.nv - 6) = Eigen::MatrixXd::Identity(real_model_.nv - 6, real_model_.nv - 6);
    C.block(real_model_.nv - 3, real_model_.nv, 3, real_model_.nv) = J_imu.block(3, 0, 3, real_model_.nv);
    C.block(real_model_.nv, real_model_.nv + 6, real_model_.nv - 6, real_model_.nv - 6) = Eigen::MatrixXd::Identity(real_model_.nv - 6, real_model_.nv - 6);
    C.block(2 * real_model_.nv - 6, real_model_.nv, 3, real_model_.nv) = J_imu_dot.block(0, 0, 3, real_model_.nv);
    C.block(2 * real_model_.nv - 3, real_model_.nv, 3, real_model_.nv) = J_left_foot.block(0,0,3,real_model_.nv)*left_support_check;
    C.block(2 * real_model_.nv, real_model_.nv, 3, real_model_.nv) = J_right_foot.block(0,0,3,real_model_.nv)*right_support_check;
    C.block(2 * real_model_.nv + 3, 0, 3, real_model_.nv) = J_left_foot.block(0, 0, 3, real_model_.nv)*left_support_check;
    C.block(2 * real_model_.nv + 6, 0, 3, real_model_.nv) = J_right_foot.block(0, 0, 3, real_model_.nv)*right_support_check;

    //MATRICE D:

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n_ekf_output, real_model_.nv);
    D.block(2*real_model_.nv - 6, 0, 3, real_model_.nv) = J_imu.block(0, 0, 3, real_model_.nv);

    //MATRICE A:

    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(2 * real_model_.nv, 2 * real_model_.nv);
    A.block(0, real_model_.nv, real_model_.nv, real_model_.nv) = controller_timestep_msec_ * 0.001 * Eigen::MatrixXd::Identity(real_model_.nv, real_model_.nv);

    //PREDICTION COVARIANCE E KALMAN GAIN

    Eigen::MatrixXd Lambda_ = A * P_ * A.transpose() + Q;
    Eigen::MatrixXd K = Lambda_ * C.transpose() * (C * Lambda_ * C.transpose() + R).inverse();
    P_ = (Eigen::MatrixXd::Identity(2 * real_model_.nv, 2 * real_model_.nv) - K * C) * Lambda_;

    if (useRobot) {
        y_actual = actual_output;
    }
    else{
        // compute y_actual from current_state, y_actual is composed by 1) orientation of imu in axis angle 
        // 2) joint position 3) angular velocity of the imu 4) joint velocity of the robot
        // 5) accelerometer of the imu 6) velocity of feet 7) feet position

        Eigen::VectorXd q = robot_state_to_pinocchio_joint_configuration(
            real_model_,
            current_state
        );

        Eigen::VectorXd qdot = robot_state_to_pinocchio_joint_velocity(
            real_model_,
            current_state
        );

        Eigen::Quaterniond imu_orientation(
            real_data_.oMf[real_model_.getFrameId("imu_in_torso")].rotation()
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

        Eigen::MatrixXd J_imu = Eigen::MatrixXd::Zero(6, real_model_.nv);
        pinocchio::getFrameJacobian(
            real_model_, 
            real_data_, 
            real_model_.getFrameId("imu_in_torso"), 
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
            J_imu
        );

        Eigen::MatrixXd J_imu_dot = Eigen::MatrixXd::Zero(6, real_model_.nv);
        pinocchio::getFrameJacobianTimeVariation(
            real_model_, 
            real_data_, 
            real_model_.getFrameId("imu_in_torso"), 
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
            J_imu_dot
        );

        //compute angular velocity of imu
        Eigen::Vector3d imu_angular_velocity = J_imu.block(3, 0, 3, real_model_.nv) * qdot;

        //get feet position
        Eigen::Vector3d left_foot_position = real_data_.oMf[lsole_idx_].translation();
        Eigen::Vector3d right_foot_position = real_data_.oMf[rsole_idx_].translation();

        y_actual.head(3) = Eigen::Vector3d(
            axis_angle.axis().x() * axis_angle.angle(),
            axis_angle.axis().y() * axis_angle.angle(),
            axis_angle.axis().z() * axis_angle.angle()
        );
        y_actual.segment(3, real_model_.nv - 6) = q.tail(real_model_.nv - 6);
        y_actual.segment(real_model_.nv - 3, 3) = imu_angular_velocity;
        y_actual.segment(real_model_.nv - 3 + 3, real_model_.nv - 6) = qdot.tail(real_model_.nv - 6);
        y_actual.segment(real_model_.nv - 3 + real_model_.nv - 3, 3) = J_imu.block(0, 0, 3, real_model_.nv) * whole_body_controller_ptr_->get_q_ddot() + J_imu_dot.block(0, 0, 3, real_model_.nv) * qdot;
        y_actual.segment(real_model_.nv - 3 + real_model_.nv - 3 + 6 + 3, 3) = left_foot_position*left_support_check;
        y_actual.segment(real_model_.nv - 3 + real_model_.nv - 3 + 6 + 3 + 3, 3) = right_foot_position*right_support_check;
    }

    //PREDICTED OUTPUT E PREDICTED X

    Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(2 * real_model_.nv);
    x_pred.head(real_model_.nv) = x_estimate.head(real_model_.nv) + x_estimate.tail(real_model_.nv) * controller_timestep_msec_ * 0.001; 
    x_pred.tail(real_model_.nv) = x_estimate.tail(real_model_.nv) + whole_body_controller_ptr_->get_q_ddot() * controller_timestep_msec_ * 0.001;
    y_pred = y_estimate + C * (x_pred - x_estimate) + D * (whole_body_controller_ptr_->get_q_ddot() - input);
    std::cout << "y_pred: " << y_pred.tail(6).transpose() << std::endl;

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

    std::cout << "Angle: " << angle << std::endl;

    if (std::abs(angle) < 1e-4) {
        orientation = Eigen::Quaterniond(1,0,0,0);  // nessuna rotazione
    } else {
        Eigen::Vector3d axis = rot_vec.normalized();
        orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
    }

    std::cout << "Orientation: " << orientation.w() << std::endl;

    current_state.orientation = Eigen::Quaterniond(
        orientation.w(),
        orientation.x(),
        orientation.y(),
        orientation.z()
    );

    current_state.linear_velocity = x_estimate.segment(real_model_.nv, 3);
    current_state.angular_velocity = x_estimate.segment(real_model_.nv + 3, 3);

    // assign position and velocity for each joint
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) real_model_.njoints; ++joint_id) {
        std::string joint_name = real_model_.names[joint_id];
        current_state.joint_state[joint_name].pos = x_estimate(joint_id - 2 + 6);
        current_state.joint_state[joint_name].vel = x_estimate(real_model_.nv + joint_id - 2 + 6);
    }

    rot_vec = y_pred.head(3);  
    angle = rot_vec.norm();

    if (angle > M_PI) {
        angle -= 2 * M_PI;
    } else if (angle < -M_PI) {
        angle += 2 * M_PI;
    }

    Eigen::Quaterniond predicted_imu_orientation;
    if (std::abs(angle) < 1e-4) {
        predicted_imu_orientation = Eigen::Quaterniond(1,0,0,0);  // nessuna rotazione
    } else {
        Eigen::Vector3d axis = rot_vec.normalized();
        predicted_imu_orientation = Eigen::Quaterniond(Eigen::AngleAxisd(angle, axis));
    }

    std::cout << "Predicted IMU Orientation: " << predicted_imu_orientation.coeffs().transpose() << std::endl;

    // log the predicted state
    predicted_imu_accelerometer_log_file_ << y_pred.segment(real_model_.nv - 3 + real_model_.nv - 3, 3).transpose() << std::endl;
    predicted_imu_angular_velocity_log_file_ << y_pred.segment(real_model_.nv - 3, 3).transpose() << std::endl;
    predicted_imu_orientation_log_file_ << predicted_imu_orientation.w() << " "
        << predicted_imu_orientation.x() << " "
        << predicted_imu_orientation.y() << " "
        << predicted_imu_orientation.z() << std::endl;

    return current_state;
}

// RobotState WalkingManager::updateEKF(RobotState current_state, bool useRobot, Eigen::VectorXd actual_output) {

//     auto q = robot_state_to_pinocchio_joint_configuration(real_model_, current_state);

//     //compute M matrix from pinocchio model
//     Eigen::MatrixXd M = pinocchio::crba(real_model_, real_data_, q);
//     // Eigen::MatrixXd M_inv = M.inverse();
//     Eigen::MatrixXd J_imu = Eigen::MatrixXd::Zero(6, real_model_.nv);
//     pinocchio::getFrameJacobian(
//         real_model_, 
//         real_data_, 
//         real_model_.getFrameId("imu_in_torso"), 
//         pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
//         J_imu
//     );
//     Eigen::MatrixXd J_imu_dot = Eigen::MatrixXd::Zero(6, real_model_.nv);
//     pinocchio::getFrameJacobianTimeVariation(
//         real_model_, 
//         real_data_, 
//         real_model_.getFrameId("imu_in_torso"), 
//         pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, 
//         J_imu_dot
//     );

//     Eigen::MatrixXd J_left_foot = Eigen::MatrixXd::Zero(6, real_model_.nv);
//     pinocchio::getFrameJacobian(
//         real_model_,
//         real_data_,
//         real_model_.getFrameId("left_foot_link"),
//         pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
//         J_left_foot
//     );

//     Eigen::MatrixXd J_right_foot = Eigen::MatrixXd::Zero(6, real_model_.nv);
//     pinocchio::getFrameJacobian(
//         real_model_,
//         real_data_,
//         real_model_.getFrameId("right_foot_link"),
//         pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
//         J_right_foot
//     );

//     double left_support_check = 1.0;
//     double right_support_check = 1.0;
//     if (walking_data_.getWalkingState() == WalkingState::SingleSupport){
//         if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT){
//             right_support_check = 0.0;
//         }
//         if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT){
//             left_support_check = 0.0;
//         }
//     }

//     // get imu orientation from pinocchio
//     Eigen::Quaterniond imu_orientation(
//         real_data_.oMf[real_model_.getFrameId("imu_in_torso")].rotation()
//     );

//     Eigen::Quaterniond imu_orientation_in_base = imu_orientation * current_state.orientation.inverse();

//     Eigen::MatrixXd relative_orientation = Eigen::MatrixXd::Zero(4, 4);
//     relative_orientation(0, 0) = imu_orientation_in_base.w();
//     relative_orientation(0, 1) = -imu_orientation_in_base.x();
//     relative_orientation(0, 2) = -imu_orientation_in_base.y();
//     relative_orientation(0, 3) = -imu_orientation_in_base.z();
//     relative_orientation(1, 0) = imu_orientation_in_base.x();
//     relative_orientation(1, 1) = imu_orientation_in_base.w();
//     relative_orientation(1, 2) = imu_orientation_in_base.z();
//     relative_orientation(1, 3) = -imu_orientation_in_base.y();
//     relative_orientation(2, 0) = imu_orientation_in_base.y();
//     relative_orientation(2, 1) = -imu_orientation_in_base.z();
//     relative_orientation(2, 2) = imu_orientation_in_base.w();
//     relative_orientation(2, 3) = imu_orientation_in_base.x();
//     relative_orientation(3, 0) = imu_orientation_in_base.z();
//     relative_orientation(3, 1) = imu_orientation_in_base.y();
//     relative_orientation(3, 2) = -imu_orientation_in_base.x();
//     relative_orientation(3, 3) = imu_orientation_in_base.w();
    
//     // create quaterniond using base orientation
//     Eigen::Quaterniond base_orientation(
//         current_state.orientation.w(),
//         current_state.orientation.x(),
//         current_state.orientation.y(),
//         current_state.orientation.z()
//     );
//     Eigen::Matrix3d R_qb = base_orientation.toRotationMatrix();

//     Eigen::Matrix<double, 3, 4> dR_v_dq_left;
//     Eigen::Matrix<double, 3, 4> dR_v_dq_right;

//     double eps = 1e-6;
//     //forward kinematics to left foot
//     Eigen::Vector3d v_left = real_data_.oMf[real_model_.getFrameId("left_foot_link")].translation();
//     Eigen::Vector3d v_right = real_data_.oMf[real_model_.getFrameId("right_foot_link")].translation();


//     for (int i = 0; i < 4; ++i) {
//         Eigen::Vector4d dq = Eigen::Vector4d::Zero();
//         dq(i) = eps;

//         //create a vector 4d with base orientation
//         Eigen::Vector4d q_base = Eigen::Vector4d(
//             base_orientation.w(),
//             base_orientation.x(),
//             base_orientation.y(),
//             base_orientation.z()
//         );

//         Eigen::Vector4d q_plus = q_base + dq;
//         Eigen::Vector4d q_minus = q_base - dq;
//         q_plus.normalize();
//         q_minus.normalize();

//         Eigen::Quaterniond q_plus_eigen(q_plus(0), q_plus(1), q_plus(2), q_plus(3));
//         Eigen::Quaterniond q_minus_eigen(q_minus(0), q_minus(1), q_minus(2), q_minus(3));

//         Eigen::Vector3d Rv_plus_left = q_plus_eigen.toRotationMatrix() * v_left;
//         Eigen::Vector3d Rv_minus_left = q_minus_eigen.toRotationMatrix() * v_left;

//         Eigen::Vector3d Rv_plus_right = q_plus_eigen.toRotationMatrix() * v_right;
//         Eigen::Vector3d Rv_minus_right = q_minus_eigen.toRotationMatrix() * v_right;

//         dR_v_dq_left.col(i) = (Rv_plus_left - Rv_minus_left) / (2.0 * eps);
//         dR_v_dq_right.col(i) = (Rv_plus_right - Rv_minus_right) / (2.0 * eps);
//     }

//     Eigen::MatrixXd C = Eigen::MatrixXd::Zero(real_model_.nq - 3 + real_model_.nv - 3 + 6 + 3 + 6, real_model_.nq + real_model_.nv);
//     C.block(0, 3, 4, 4) = relative_orientation;
//     C.block(4, 7, real_model_.nq - 7, real_model_.nq - 7) = Eigen::MatrixXd::Identity(real_model_.nq - 7, real_model_.nq - 7);
//     // C.block(real_model_.nq - 3, real_model_.nq + 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
//     C.block(real_model_.nq - 3, real_model_.nq, 3, real_model_.nv) = J_imu.block(3, 0, 3, real_model_.nv);
//     C.block(real_model_.nq , real_model_.nq + 6, real_model_.nv - 6, real_model_.nv - 6) = Eigen::MatrixXd::Identity(real_model_.nv - 6, real_model_.nv - 6);
//     C.block(real_model_.nq + real_model_.nv - 6, real_model_.nq, 3, real_model_.nv) = J_imu_dot.block(0, 0, 3, real_model_.nv);
//     C.block(real_model_.nq + real_model_.nv - 3, real_model_.nq, 3, real_model_.nv) = J_left_foot.block(0,0,3,real_model_.nv)*left_support_check;
//     C.block(real_model_.nq + real_model_.nv, real_model_.nq, 3, real_model_.nv) = J_right_foot.block(0,0,3,real_model_.nv)*right_support_check;
//     C.block(real_model_.nq + real_model_.nv + 3, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
//     C.block(real_model_.nq + real_model_.nv + 3, 3, 3, 4) = dR_v_dq_left;
//     C.block(real_model_.nq + real_model_.nv + 3, 7, 3, real_model_.nq - 7) = R_qb * J_left_foot.block(0, 6, 3, real_model_.nv - 6);
//     C.block(real_model_.nq + real_model_.nv + 6, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
//     C.block(real_model_.nq + real_model_.nv + 6, 3, 3, 4) = dR_v_dq_right;
//     C.block(real_model_.nq + real_model_.nv + 6, 7, 3, real_model_.nq - 7) = R_qb * J_right_foot.block(0, 6, 3, real_model_.nv - 6);
//     Eigen::VectorXd input = whole_body_controller_ptr_->get_q_ddot();

//     Eigen::MatrixXd D = Eigen::MatrixXd::Zero(real_model_.nq - 3 + real_model_.nv - 3 + 6 + 3 + 6, real_model_.nv);
//     D.block(real_model_.nq - 6 + real_model_.nv, 0, 3, real_model_.nv) = J_imu.block(0, 0, 3, real_model_.nv);
//     //compute skewsimm matrix with omega
//     Eigen::MatrixXd skew_4x4 = Eigen::MatrixXd::Zero(4, 4);
//     skew_4x4(0, 1) = -current_state.angular_velocity(0);
//     skew_4x4(0, 2) = -current_state.angular_velocity(1);
//     skew_4x4(0, 3) = -current_state.angular_velocity(2);
//     skew_4x4(1, 0) = current_state.angular_velocity(0);
//     skew_4x4(1, 2) = current_state.angular_velocity(2);
//     skew_4x4(1, 3) = -current_state.angular_velocity(1);
//     skew_4x4(2, 0) = current_state.angular_velocity(1);
//     skew_4x4(2, 1) = -current_state.angular_velocity(2);
//     skew_4x4(2, 3) = current_state.angular_velocity(0);
//     skew_4x4(3, 0) = current_state.angular_velocity(2);
//     skew_4x4(3, 1) = current_state.angular_velocity(1);
//     skew_4x4(3, 2) = -current_state.angular_velocity(0);

//     Eigen::MatrixXd skew_4x3 = Eigen::MatrixXd::Zero(4, 3);
//     skew_4x3(0, 0) = -current_state.orientation.y(); // q2 x y z w -> w x y z
//     skew_4x3(0, 1) = -current_state.orientation.z(); // q3
//     skew_4x3(0, 2) = -current_state.orientation.w(); // q4
//     skew_4x3(1, 0) = current_state.orientation.x(); // q1
//     skew_4x3(1, 1) = -current_state.orientation.w();
//     skew_4x3(1, 2) = current_state.orientation.z();
//     skew_4x3(2, 0) = current_state.orientation.w();
//     skew_4x3(2, 1) = current_state.orientation.x();
//     skew_4x3(2, 2) = -current_state.orientation.y();
//     skew_4x3(3, 0) = -current_state.orientation.z();
//     skew_4x3(3, 1) = current_state.orientation.y();
//     skew_4x3(3, 2) = current_state.orientation.x();

//     Eigen::MatrixXd A = Eigen::MatrixXd::Zero(real_model_.nq + real_model_.nv, real_model_.nq + real_model_.nv);
//     A.block(0, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
//     A.block(0, real_model_.nq, 3, 3) = controller_timestep_msec_ * 0.001 * Eigen::MatrixXd::Identity(3, 3);
//     A.block(3, 3, 4, 4) = Eigen::MatrixXd::Identity(4, 4) + controller_timestep_msec_ * 0.001 * 0.5 * skew_4x4;
//     A.block(3, real_model_.nq + 3, 4, 3) = controller_timestep_msec_ * 0.001 * 0.5 * skew_4x3;
//     A.block(7, 7, real_model_.nq - 7, real_model_.nq - 7) = Eigen::MatrixXd::Identity(real_model_.nq - 7, real_model_.nq - 7);
//     A.block(7, real_model_.nq + 6, real_model_.nv- 6, real_model_.nv- 6) = controller_timestep_msec_ * 0.001 * Eigen::MatrixXd::Identity(real_model_.nv- 6, real_model_.nv- 6);
//     A.block(real_model_.nq, real_model_.nq, real_model_.nv, real_model_.nv) = Eigen::MatrixXd::Identity(real_model_.nv, real_model_.nv);
//     Eigen::MatrixXd Lambda_ = A * P_ * A.transpose() + Q;
//     Eigen::MatrixXd K = Lambda_ * C.transpose() * (C * Lambda_ * C.transpose() + R).inverse();
//     P_ = (Eigen::MatrixXd::Identity(real_model_.nq + real_model_.nv, real_model_.nq + real_model_.nv) - K * C) * Lambda_;
//     Eigen::VectorXd a = whole_body_controller_ptr_->get_q_ddot();
//     Eigen::VectorXd x_pred = Eigen::VectorXd::Zero(real_model_.nq + real_model_.nv);
//     Eigen::VectorXd y_pred = Eigen::VectorXd::Zero(real_model_.nq - 3 + real_model_.nv - 3 + 6 + 3);
//     x_pred.head(real_model_.nq) = pinocchio::integrate(real_model_, x_estimate.head(real_model_.nq), x_estimate.tail(real_model_.nv) * controller_timestep_msec_ * 0.001); 
//     x_pred.tail(real_model_.nv) = x_estimate.tail(real_model_.nv) + a * controller_timestep_msec_ * 0.001;
//     y_pred = C * x_pred + D * input;
//     Eigen::VectorXd y = Eigen::VectorXd::Zero(real_model_.nq - 3 + real_model_.nv - 3 + 6 + 3);
//     if (useRobot) {
//         y = actual_output;
//     }
//     else{
//         Eigen::VectorXd x = Eigen::VectorXd::Zero(real_model_.nq + real_model_.nv);
//         x.head(real_model_.nq) = robot_state_to_pinocchio_joint_configuration(
//             real_model_,
//             current_state
//         );
//         x.tail(real_model_.nv) = robot_state_to_pinocchio_joint_velocity(
//             real_model_,
//             current_state
//         );

//         y = C * x + D * input; 
//     }
//     x_estimate = x_pred + K * (y - y_pred);

//     current_state.position = x_estimate.head(3);
//     current_state.orientation = Eigen::Quaterniond(
//         x_estimate(6),
//         x_estimate(3),
//         x_estimate(4),
//         x_estimate(5)
//     );
//     current_state.linear_velocity = x_estimate.segment(real_model_.nq, 3);
//     current_state.angular_velocity = x_estimate.segment(real_model_.nq + 3, 3);

//     // assign position and velocity for each joint
//     for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) real_model_.njoints; ++joint_id) {
//         std::string joint_name = real_model_.names[joint_id];
//         current_state.joint_state[joint_name].pos = x_estimate(joint_id - 2 + 7);
//         current_state.joint_state[joint_name].vel = x_estimate(real_model_.nq + joint_id - 2 + 6);
//     }

//     // log the predicted state
//     predicted_imu_accelerometer_log_file_ << y_pred.segment(real_model_.nq - 3 + real_model_.nv - 3, 3).transpose() << std::endl;
//     predicted_imu_angular_velocity_log_file_ << y_pred.segment(real_model_.nq - 3, 3).transpose() << std::endl;
//     predicted_imu_orientation_log_file_ << y_pred.head(4).transpose() << std::endl;

//     return current_state;
// }

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
    const labrob::RobotState& robot_state,
    labrob::JointCommand& joint_command,
    labrob::RobotState& fb_robot_state,
    bool useRobot,
    Eigen::VectorXd actual_output
) {

    int njnt = robot_model_.nv - 6; // size of configuration space without floating base

    auto q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
    auto qdot = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);

    // Perform forward kinematics on the whole tree and update robot data:
    pinocchio::forwardKinematics(robot_model_, robot_data_, q);

    // // NOTE: jacobianCenterOfMass calls forwardKinematics and
    //       computeJointJacobians.
    pinocchio::jacobianCenterOfMass(robot_model_, robot_data_, q);
    pinocchio::computeJointJacobiansTimeVariation(robot_model_, robot_data_, q, qdot);
    pinocchio::framesForwardKinematics(robot_model_, robot_data_, q);
    pinocchio::centerOfMass(robot_model_, robot_data_, q, qdot, 0.0 * qdot); // This is used to compute the CoM drift (J_com_dot * qdot)
    const auto& centroidal_momentum_matrix = pinocchio::ccrba(
        robot_model_,
        robot_data_,
        q,
        qdot
    );

    auto angular_momentum = (centroidal_momentum_matrix * qdot).tail<3>();
    

    if (useRobot) {

        auto q_fb = robot_state_to_pinocchio_joint_configuration(real_model_, fb_robot_state);
        auto qdot_fb = robot_state_to_pinocchio_joint_velocity(real_model_, fb_robot_state);

        // Perform forward kinematics on the whole tree and update robot data:
        pinocchio::forwardKinematics(real_model_, real_data_, q_fb);

        // // NOTE: jacobianCenterOfMass calls forwardKinematics and
        //       computeJointJacobians.
        pinocchio::jacobianCenterOfMass(real_model_, real_data_, q_fb);
        pinocchio::computeJointJacobiansTimeVariation(real_model_, real_data_, q_fb, qdot_fb);
        pinocchio::framesForwardKinematics(real_model_, real_data_, q_fb);
        pinocchio::centerOfMass(real_model_, real_data_, q_fb, qdot_fb, 0.0 * qdot_fb); // This is used to compute the CoM drift (J_com_dot * qdot)
        const auto& centroidal_momentum_matrix = pinocchio::ccrba(
            real_model_,
            real_data_,
            q_fb,
            qdot_fb
        );

        const auto& p_CoM_fb = real_data_.com[0];
        const auto& a_CoM_drift_fb = real_data_.acom[0];
        const auto& J_CoM_fb = real_data_.Jcom;
        const auto& T_torso_fb = real_data_.oMf[torso_idx_];
        auto torso_orientation_fb = T_torso_fb.rotation();
        Eigen::MatrixXd J_torso_fb = Eigen::MatrixXd::Zero(6, real_model_.nv);
        pinocchio::getFrameJacobian(
            real_model_,
            real_data_,
            torso_idx_,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
            J_torso_fb
        );

        const auto& T_lsole_fb = real_data_.oMf[lsole_idx_];
        Eigen::MatrixXd J_lsole_fb = Eigen::MatrixXd::Zero(6, real_model_.nv);
        pinocchio::getFrameJacobian(
            real_model_,
            real_data_,
            lsole_idx_,
            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
            J_lsole_fb
        );

        const auto& v_lsole_fb = J_lsole_fb * qdot_fb;

        const auto& T_rsole_fb = real_data_.oMf[rsole_idx_];
        Eigen::MatrixXd J_rsole_fb = Eigen::MatrixXd::Zero(6, real_model_.nv);
        pinocchio::getFrameJacobian(
            real_model_,
            real_data_,
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

        real_current_gait_configuration.com.pos = real_data_.com[0];
        real_current_gait_configuration.com.vel = real_data_.vcom[0];

        real_current_gait_configuration.torso.pos = real_data_.oMf[torso_idx_].rotation();
        real_current_gait_configuration.torso.vel = J_torso_fb.bottomRows<3>() * qdot_fb;

        real_current_gait_configuration.lsole.pos = labrob::SE3(real_data_.oMf[lsole_idx_].rotation(), real_data_.oMf[lsole_idx_].translation());
        real_current_gait_configuration.lsole.vel = J_lsole_fb * qdot_fb;

        real_current_gait_configuration.rsole.pos = labrob::SE3(real_data_.oMf[rsole_idx_].rotation(), real_data_.oMf[rsole_idx_].translation());
        real_current_gait_configuration.rsole.vel = J_rsole_fb * qdot_fb;

        //log real com
        Eigen::Vector3d real_com_pos = real_data_.com[0];
        real_com_log_file_ << real_com_pos.transpose() << t_msec_ << std::endl;

    }

    const auto& T_torso = robot_data_.oMf[torso_idx_];
    auto torso_orientation = T_torso.rotation();
    Eigen::MatrixXd J_torso = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    pinocchio::getFrameJacobian(
        robot_model_,
        robot_data_,
        torso_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_torso
    );

    const auto& T_lsole = robot_data_.oMf[lsole_idx_];
    Eigen::MatrixXd J_lsole = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    pinocchio::getFrameJacobian(
        robot_model_,
        robot_data_,
        lsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_lsole
    );

    const auto& v_lsole = J_lsole * qdot;

    const auto& T_rsole = robot_data_.oMf[rsole_idx_];
    Eigen::MatrixXd J_rsole = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    pinocchio::getFrameJacobian(
        robot_model_,
        robot_data_,
        rsole_idx_,
        pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
        J_rsole
    );
    const auto& v_rsole = J_rsole * qdot;

    // save left and right foot position in last 6 places of actual output
    actual_output.segment(real_model_.nv - 3 + real_model_.nv - 3 + 3 + 6, 3) = T_lsole.translation();
    actual_output.segment(real_model_.nv - 3 + real_model_.nv - 3 + 3 + 6 + 3, 3) = T_rsole.translation();
    //print T_lsole and T_rsole
    std::cout << "T_lsole: " << T_lsole.translation().transpose() << std::endl;
    std::cout << "T_rsole: " << T_rsole.translation().transpose() << std::endl;

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

    current_gait_configuration.com.pos = robot_data_.com[0];
    current_gait_configuration.com.vel = robot_data_.vcom[0];

    current_gait_configuration.torso.pos = robot_data_.oMf[torso_idx_].rotation();
    current_gait_configuration.torso.vel = J_torso.bottomRows<3>() * qdot;

    current_gait_configuration.lsole.pos = labrob::SE3(robot_data_.oMf[lsole_idx_].rotation(), robot_data_.oMf[lsole_idx_].translation());
    current_gait_configuration.lsole.vel = J_lsole * qdot;

    current_gait_configuration.rsole.pos = labrob::SE3(robot_data_.oMf[rsole_idx_].rotation(), robot_data_.oMf[rsole_idx_].translation());
    current_gait_configuration.rsole.vel = J_rsole * qdot;

    double eta2 = std::pow(ismpc_ptr_->getOmega(), 2.0);
    double mass = pinocchio::computeTotalMass(robot_model_);
    // Eigen::Vector3d lip_zmp = p_CoM - robot_state.total_force / (mass * eta2);

    if (!useRobot) {
        real_model_ = robot_model_;
        real_data_ = robot_data_;
    }
    
    RobotState fb_filtered_state_;

    LIPState measured_state;

    if(!useRobot) {
        fb_filtered_state_ = updateEKF(robot_state, useRobot, actual_output);
    }
    else{
        fb_filtered_state_ = updateEKF(fb_robot_state, useRobot, actual_output);
        fb_robot_state = fb_filtered_state_;
    }
    
    if (useRobot && t_msec_ > 6000) {
        if(t_msec_ == 6002){
            std::cout << "STARTING CLOSED-LOOP SIM"<< std::endl;
        }

        const auto& fb_p_CoM = real_data_.com[0];
        const auto& J_CoM = real_data_.Jcom;
        Eigen::Vector3d zmp_3d;
        zmp_3d.z() = fb_filtered_state_.position(2) - fb_filtered_state_.total_force.z() / (mass * eta2);
        zmp_3d.x() = 0.0;
        zmp_3d.y() = 0.0;
        for (int i = 0; i < fb_filtered_state_.contact_points.size(); ++i) {
            auto &pi = fb_filtered_state_.contact_points[i];
            auto &fi = fb_filtered_state_.contact_forces[i];
            zmp_3d.x() += (pi.x() * fi.z() / fb_filtered_state_.total_force.z() + (zmp_3d.z() - pi.z()) * fi.x() / fb_filtered_state_.total_force.z());
            zmp_3d.y() += (pi.y() * fi.z() / fb_filtered_state_.total_force.z() + (zmp_3d.z() - pi.z()) * fi.y() / fb_filtered_state_.total_force.z());
        }
    
        auto qdot_fb = robot_state_to_pinocchio_joint_velocity(real_model_, fb_filtered_state_);
    
        measured_state = LIPState(fb_p_CoM, J_CoM * qdot_fb, zmp_3d);
    
        // filtered_state_ = updateKF(filtered_state_, measured_state, ismpc_ptr_->getInput());
    }


    const auto& p_CoM = robot_data_.com[0];
    const auto& J_CoM = robot_data_.Jcom;
    Eigen::Vector3d zmp_3d;
    zmp_3d.z() = robot_state.position(2) - robot_state.total_force.z() / (mass * eta2);
    zmp_3d.x() = 0.0;
    zmp_3d.y() = 0.0;
    for (int i = 0; i < robot_state.contact_points.size(); ++i) {
        auto &pi = robot_state.contact_points[i];
        auto &fi = robot_state.contact_forces[i];
        zmp_3d.x() += (pi.x() * fi.z() / robot_state.total_force.z() + (zmp_3d.z() - pi.z()) * fi.x() / robot_state.total_force.z());
        zmp_3d.y() += (pi.y() * fi.z() / robot_state.total_force.z() + (zmp_3d.z() - pi.z()) * fi.y() / robot_state.total_force.z());
    }
    if(!useRobot || t_msec_ >= 0) {
        measured_state = LIPState(p_CoM, J_CoM * robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state), zmp_3d);

        filtered_state_ = updateKF(filtered_state_, measured_state, ismpc_ptr_->getInput());
    }



    // CoM task:
    auto mpc_t0_ms = std::chrono::system_clock::now();
    ismpc_ptr_->solve(t_msec_, walking_data_, filtered_state_);
    auto mpc_tf_ms = std::chrono::system_clock::now();
    auto mpc_duration = std::chrono::duration_cast<std::chrono::microseconds>(mpc_tf_ms - mpc_t0_ms).count();

    DdpSolver ddpsolver = DdpSolver();

    // set x0 as initial state of CoM_pos CoM_vel and ZMP_pos
    Eigen::Vector<double, NX> x0;
    x0 <<
        filtered_state_.com_pos_(0),
        filtered_state_.com_pos_(1),
        filtered_state_.com_pos_(2),
        filtered_state_.com_vel_(0),
        filtered_state_.com_vel_(1),
        filtered_state_.com_vel_(2),
        filtered_state_.zmp_pos_(0),
        filtered_state_.zmp_pos_(1),
        filtered_state_.zmp_pos_(2);
    std::array<Eigen::Vector<double, NX>, NH+1> x_traj;
    x_traj[0] = x0;
    std::array<Eigen::Vector<double, NU>, NH> u_traj;
  
    // set warm-start trajectories
    std::array<Eigen::Vector<double, NX>, NH+1> x_guess;
    for (int i = 0; i < NH+1; ++i)
      x_guess[i] = x0;
    std::array<Eigen::Vector<double, NU>, NH> u_guess;
    for (int i = 0; i < NH; ++i)
      u_guess[i].setZero();
    ddpsolver.set_initial_state(x0);
    ddpsolver.set_x_warmstart(x_guess);
    ddpsolver.set_u_warmstart(u_guess);

    auto start = std::chrono::system_clock::now();
    // ddpsolver.solve();
    auto end = std::chrono::system_clock::now();
    auto solve_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Update the state based on the result of the QP:
    auto lip_state = discrete_lip_dynamics_ptr_->integrate(filtered_state_, ismpc_ptr_->getInput());
    // substitute with ddpsolver.u[0]

    Eigen::VectorXd inputSequenceX = ismpc_ptr_->getInputSequenceX();
    Eigen::VectorXd inputSequenceY = ismpc_ptr_->getInputSequenceY();
    Eigen::VectorXd inputSequenceZ = ismpc_ptr_->getInputSequenceZ();

    LIPState measured_state_mpc = filtered_state_;

    for (int i = 0; i < 20; ++i) {
        measured_state_mpc = discrete_lip_dynamics_ptr_mpc_->integrate(
            measured_state_mpc, 
            Eigen::Vector3d(inputSequenceX(i), inputSequenceY(i), inputSequenceZ(i))
        );

        mpc_predictions_log_file_ << measured_state_mpc.com_pos_.transpose() << " "
            << measured_state_mpc.com_vel_.transpose() << " "
            << measured_state_mpc.zmp_pos_.transpose() << std::endl;
    }


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



    start = std::chrono::system_clock::now();
    if (useRobot && t_msec_ > 2000 && false) {
        // Use the robot feedback to compute the joint command:
        joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
            real_model_,
            fb_filtered_state_,
            real_data_,
            current_gait_configuration,
            desired_gait_configuration
        );
    } else {
        // Use the MPC to compute the joint command:
        joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
            robot_model_,
            fb_filtered_state_,
            robot_data_,
            current_gait_configuration,
            desired_gait_configuration
        );
    }
    end = std::chrono::system_clock::now();
    auto compute_id_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    // std::cout << "compute_inverse_dynamics took " << compute_id_duration << " microseconds." << std::endl;


    // Update timing in milliseconds.
    // NOTE: assuming update() is actually called every controller_timestep_msec_
    //       milliseconds.
    t_msec_ += controller_timestep_msec_;
    prev_angular_momentum_ = angular_momentum;


    // Log:
    mpc_timings_log_file_ << std::chrono::duration_cast<std::chrono::microseconds>(mpc_tf_ms - mpc_t0_ms).count() << std::endl;
    mpc_com_log_file_ << p_CoM_des.transpose() << std::endl;
    mpc_zmp_log_file_ << p_ZMP_des.transpose() << std::endl;
    com_log_file_ << p_CoM.transpose() << std::endl;
    p_lsole_log_file_ << T_lsole.translation().transpose() << std::endl;
    p_rsole_log_file_ << T_rsole.translation().transpose() << std::endl;
    v_lsole_log_file_ << v_lsole.head<3>().transpose() << std::endl;
    v_rsole_log_file_ << v_rsole.head<3>().transpose() << std::endl;
    p_lsole_des_log_file_ << desired_gait_configuration.lsole.pos.p.transpose() << std::endl;
    p_rsole_des_log_file_ << desired_gait_configuration.rsole.pos.p.transpose() << std::endl;
    v_lsole_des_log_file_ << desired_gait_configuration.lsole.vel.head<3>().transpose() << std::endl;
    v_rsole_des_log_file_ << desired_gait_configuration.rsole.vel.head<3>().transpose() << std::endl;
    angular_momentum_log_file_ << angular_momentum.transpose() << std::endl;
    cop_computed_log_file_ << measured_state.zmp_pos_.transpose() << " " << filtered_state_.zmp_pos_.transpose() << " " << zmp_3d.transpose() << std::endl;

    // Log the filtered state:
    ekf_base_position_log_file_ << fb_filtered_state_.position.transpose() << std::endl;
    ekf_base_velocity_log_file_ << fb_filtered_state_.linear_velocity.transpose() << std::endl;
    ekf_base_orientation_log_file_ << fb_filtered_state_.orientation.w() << " "
        << fb_filtered_state_.orientation.x() << " "
        << fb_filtered_state_.orientation.y() << " "
        << fb_filtered_state_.orientation.z() << std::endl;
    ekf_base_angular_velocity_log_file_ << fb_filtered_state_.angular_velocity.transpose() << std::endl;
    for (pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) real_model_.njoints; ++joint_id) {
        std::string joint_name = real_model_.names[joint_id];
        ekf_joint_position_log_file_ << fb_filtered_state_.joint_state[joint_name].pos << " ";
        ekf_joint_velocity_log_file_ << fb_filtered_state_.joint_state[joint_name].vel << " ";
    }
    ekf_joint_position_log_file_ << std::endl;
    ekf_joint_velocity_log_file_ << std::endl;

    base_position_log_file_ << robot_state.position.transpose() << std::endl;
    base_velocity_log_file_ << robot_state.linear_velocity.transpose() << std::endl;
    base_orientation_log_file_ << robot_state.orientation.w() << " "
        << robot_state.orientation.x() << " "
        << robot_state.orientation.y() << " "
        << robot_state.orientation.z() << std::endl;
    base_angular_velocity_log_file_ << robot_state.angular_velocity.transpose() << std::endl;
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
