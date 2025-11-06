#include <hrp4_locomotion/WalkingManager.hpp>

// STL
#include <algorithm>
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
    std::cout << "Number of joints: " << njnt << std::endl;

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
    fb_robot_state = initial_robot_state;

    pinocchio::forwardKinematics(robot_model, fb_robot_data, q_init);
    pinocchio::jacobianCenterOfMass(robot_model, fb_robot_data, q_init);
    pinocchio::framesForwardKinematics(robot_model, fb_robot_data, q_init);

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

    double swing_foot_trajectory_height = 0.05;
    double step_length_x = 0.0;
    double step_length_y = 0.0;
    double step_rotation = 0.0;
    int n_steps = 10;
    walking_data_.footstep_plan.push_back(labrob::Footstep(
        T_lsole_init.rotation(), T_lsole_init.translation(),
        T_rsole_init.rotation(), T_rsole_init.translation(),
        labrob::Foot::RIGHT,
        labrob::WalkingState::Init,
        2000,
        0.0
    ));
    walking_data_.footstep_plan.push_back(labrob::Footstep(
        T_lsole_init.rotation(), T_lsole_init.translation(),
        T_rsole_init.rotation(), T_rsole_init.translation(),
        labrob::Foot::RIGHT,
        labrob::WalkingState::Standing,
        2000,
        0.0
    ));

    double double_support_duration = 6000;
    double single_support_duration = 6000;
    walking_data_.footstep_plan.push_back(labrob::Footstep(
        T_lsole_init.rotation(), T_lsole_init.translation(),
        T_rsole_init.rotation(), T_rsole_init.translation(),
        labrob::Foot::RIGHT,
        labrob::WalkingState::Starting,
        double_support_duration,
        0.0
    ));

    walking_data_.footstep_plan.push_back(labrob::Footstep(
        T_lsole_init.rotation(), T_lsole_init.translation(),
        T_rsole_init.rotation(), T_rsole_init.translation(),
        labrob::Foot::RIGHT,
        labrob::WalkingState::SingleSupport,
        single_support_duration,
        swing_foot_trajectory_height
    ));
    for (int n = 0; n < n_steps; n += 2) {
        walking_data_.footstep_plan.push_back(labrob::Footstep(
            labrob::Rz((n + 1) * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
            labrob::Rz(n * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + n * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
            labrob::Foot::RIGHT,
            labrob::WalkingState::DoubleSupport,
            double_support_duration,
            0.0
        ));
        walking_data_.footstep_plan.push_back(labrob::Footstep(
            labrob::Rz((n + 1) * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
            labrob::Rz(n * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + n * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
            labrob::Foot::LEFT,
            labrob::WalkingState::SingleSupport,
            single_support_duration,
            swing_foot_trajectory_height
        ));
        walking_data_.footstep_plan.push_back(labrob::Footstep(
            labrob::Rz((n + 1) * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
            labrob::Rz((n + 2) * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + (n + 2) * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
            labrob::Foot::LEFT,
            labrob::WalkingState::DoubleSupport,
            double_support_duration,
            0.0
        ));
        walking_data_.footstep_plan.push_back(labrob::Footstep(
            labrob::Rz((n + 1) * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
            labrob::Rz((n + 2) * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + (n + 2) * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
            labrob::Foot::RIGHT,
            labrob::WalkingState::SingleSupport,
            single_support_duration,
            swing_foot_trajectory_height
        ));
    }
    walking_data_.footstep_plan.push_back(labrob::Footstep(
        labrob::Rz(n_steps * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
        labrob::Rz(n_steps * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
        labrob::Foot::RIGHT,
        labrob::WalkingState::Stopping,
        0,
        0.0
    ));
    walking_data_.footstep_plan.push_back(labrob::Footstep(
        labrob::Rz(n_steps * step_rotation) * T_lsole_init.rotation(), T_lsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
        labrob::Rz(n_steps * step_rotation) * T_rsole_init.rotation(), T_rsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0),
        labrob::Foot::RIGHT,
        labrob::WalkingState::Standing,
        2000,
        0.0
    ));

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

    residual_estimator_ptr_ = std::make_unique<ResidualEstimator>(robot_model, 1.0, armatures);

    return true;
}


LIPState WalkingManager::updateKF(LIPState filtered, LIPState current, const Eigen::Vector3d &input) {
  // Static local variables to maintain state between calls (replaces removed member variables)
  static Eigen::Matrix3d cov_x = Eigen::Matrix3d::Identity();
  static Eigen::Matrix3d cov_y = Eigen::Matrix3d::Identity();
  static Eigen::Matrix3d cov_z = Eigen::Matrix3d::Identity();
  static const double cov_meas_pos = 1.0e1;
  static const double cov_meas_vel = 1.0e2;
  static const double cov_meas_zmp = 1.0e8;
  static const double cov_mod_pos = 1.0;
  static const double cov_mod_vel = 1.0;
  static const double cov_mod_zmp = 1.0;

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


void
WalkingManager::update(
    const labrob::RobotState& sim_robot_state,
    labrob::JointCommand& joint_command,
    Eigen::VectorXd actual_output
) {


    // Update walking state:
    walking_data_.updateWalkingState(t_msec_);

    double eta2 = std::pow(ismpc_ptr_->getOmega(), 2.0);
    double mass = pinocchio::computeTotalMass(robot_model);

    Eigen::Vector3d left_foot_force = estimated_force.head(3);
    Eigen::Vector3d right_foot_force = estimated_force.tail(3);
    Eigen::Vector3d total_force = left_foot_force + right_foot_force;

    auto q = robot_state_to_pinocchio_joint_configuration(robot_model, sim_robot_state);
    auto qdot = robot_state_to_pinocchio_joint_velocity(robot_model, sim_robot_state);

    integrated_state_pos.head(3) = sim_robot_state.position;
    integrated_state_pos.segment<3>(3) = rotVecFromQuaternion(sim_robot_state.orientation);
    integrated_state_vel.head(3) = sim_robot_state.linear_velocity;
    integrated_state_vel.segment<3>(3) = sim_robot_state.angular_velocity;
    for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex) njnt; ++joint_id) {
        std::string joint_name = robot_model.names[joint_id + 2];
        integrated_state_pos(6 + joint_id) = sim_robot_state.joint_state[joint_name].pos;
        integrated_state_vel(6 + joint_id) = sim_robot_state.joint_state[joint_name].vel;
    }

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
    



    ////////////////////////
    // BASE ESTIMATION
    ////////////////////////
    // WORK IN PROGRESS
    
    // In simulation mode, feedback state is same as sim state
    fb_robot_state = sim_robot_state;

    Eigen::Vector3d left_foot_position;
    Eigen::Vector3d right_foot_position;

    pinocchio::SE3 desired_lsole_pose_base_est;
    pinocchio::Motion desired_lsole_vel_base_est;
    pinocchio::Motion desired_lsole_acc_base_est;
    pinocchio::SE3 desired_rsole_pose_base_est;
    pinocchio::Motion desired_rsole_vel_base_est;
    pinocchio::Motion desired_rsole_acc_base_est;

    left_foot_position = walking_data_.footstep_plan.front().left_foot_position.transpose();
    right_foot_position = walking_data_.footstep_plan.front().right_foot_position.transpose();
    // if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
    //     if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT){

    //         pinocchio::SE3 desired_rsole_pose;
    //         pinocchio::Motion desired_rsole_vel;
    //         pinocchio::Motion desired_rsole_acc;
    //         swingFootTrajectory(desired_rsole_pose, desired_rsole_vel, desired_rsole_acc);
    //         right_foot_position = desired_rsole_pose.translation().transpose();
            
    //     }
    //     else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == Foot::RIGHT){
    //         pinocchio::SE3 desired_lsole_pose;
    //         pinocchio::Motion desired_lsole_vel;
    //         pinocchio::Motion desired_lsole_acc;
    //         swingFootTrajectory(desired_lsole_pose, desired_lsole_vel, desired_lsole_acc);
    //         left_foot_position = desired_lsole_pose.translation().transpose();
    //     }
    // }

    RobotState base_estimation_robot_state = fb_robot_state;
    base_estimation_robot_state.position = Eigen::Vector3d(0,0,0);

    pinocchio::Data base_estimation_robot_data(robot_model);
    auto q_base_est = robot_state_to_pinocchio_joint_configuration(robot_model, base_estimation_robot_state);
    pinocchio::forwardKinematics(robot_model, base_estimation_robot_data, q_base_est);
    pinocchio::framesForwardKinematics(robot_model, base_estimation_robot_data, q_base_est);


    double foot_line_angle;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
        if (walking_data_.footstep_plan.front().support_foot == Foot::LEFT){

            Eigen::Vector3d left_foot_orientation = base_estimation_robot_data.oMf[lsole_idx_].rotation() * Eigen::Vector3d::UnitX();
            double left_foot_yaw = atan2(left_foot_orientation.y(), left_foot_orientation.x());
            foot_line_angle = left_foot_yaw;
            // foot_line_angle -= M_PI/2;
            
        }
        else if (walking_data_.footstep_plan.front().support_foot == Foot::RIGHT){
            Eigen::Vector3d right_foot_orientation = base_estimation_robot_data.oMf[rsole_idx_].rotation() * Eigen::Vector3d::UnitX();
            double right_foot_yaw = atan2(right_foot_orientation.y(), right_foot_orientation.x());
            foot_line_angle = right_foot_yaw;
            // foot_line_angle -= M_PI/2;
        }
    }else{
        // compute left foot yaw angle relative to base frame of simulation
        Eigen::Vector3d left_foot_orientation = base_estimation_robot_data.oMf[lsole_idx_].rotation() * Eigen::Vector3d::UnitX();
        double left_foot_yaw = atan2(left_foot_orientation.y(), left_foot_orientation.x());
        // compute right foot yaw angle relative to base frame of simulation
        Eigen::Vector3d right_foot_orientation = base_estimation_robot_data.oMf[rsole_idx_].rotation() * Eigen::Vector3d::UnitX();
        double right_foot_yaw = atan2(right_foot_orientation.y(), right_foot_orientation.x());
        foot_line_angle = 0.5 * (left_foot_yaw + right_foot_yaw);
        // foot_line_angle -= M_PI/2; // to be aligned with the foot line
    }

    Eigen::Vector3d base_estimate;

    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
        if (walking_data_.footstep_plan.front().support_foot == Foot::LEFT) base_estimate =  (left_foot_position - base_estimation_robot_data.oMf[lsole_idx_].translation());
        else if (walking_data_.footstep_plan.front().support_foot == Foot::RIGHT) base_estimate =  (right_foot_position - base_estimation_robot_data.oMf[rsole_idx_].translation());
        }
    else{
        Eigen::Vector3d mean_des_feet = 0.5 * (left_foot_position + right_foot_position);
        Eigen::Vector3d mean_fb_feet = 0.5 * (base_estimation_robot_data.oMf[lsole_idx_].translation() + base_estimation_robot_data.oMf[rsole_idx_].translation());
        base_estimate = (mean_des_feet - mean_fb_feet);
        base_estimate =  (left_foot_position - base_estimation_robot_data.oMf[lsole_idx_].translation());
    }

    Eigen::AngleAxisd yaw_correction(-foot_line_angle, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond q_yaw(yaw_correction);
    base_estimation_robot_state.orientation = q_yaw * base_estimation_robot_state.orientation;

    base_estimation_robot_state.position = base_estimate;

    q_base_est = robot_state_to_pinocchio_joint_configuration(robot_model, base_estimation_robot_state);
    pinocchio::forwardKinematics(robot_model, base_estimation_robot_data, q_base_est);
    pinocchio::framesForwardKinematics(robot_model, base_estimation_robot_data, q_base_est);

    auto left_foot_position_base_estimation = base_estimation_robot_data.oMf[lsole_idx_].translation();
    auto right_foot_position_base_estimation = base_estimation_robot_data.oMf[rsole_idx_].translation();

    if (true) {
        fb_robot_state = base_estimation_robot_state;
    }

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
    //get rotation matrix of imu
    Eigen::Matrix3d R_imu_fb = fb_robot_data.oMf[imu_idx_].rotation();
    imu_accelerometer = R_imu_fb * imu_accelerometer; //convert to world frame
    imu_accelerometer[2] -= 9.81; //remove gravity
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


    // zmp_3d_fb.z() = p_CoM_fb.z() - (a_CoM_drift_fb.z() + 9.81) / eta2;
    // zmp_3d_fb.x() = p_CoM_fb.x() - a_CoM_drift_fb.x() / eta2;
    // zmp_3d_fb.y() = p_CoM_fb.y() - a_CoM_drift_fb.y() / eta2;

    /////////////////////////////////////
    // 
    // START KF
    //
    /////////////////////////////////////

    // Use feedback CoM after 15 seconds (startTimeCoMCL = 15000.0)
    if (t_msec_ >= 15000.0) {
        if (t_msec_ == 15000.0) {
            std::cout << "Using feedback Center of Mass" << std::endl;
        }
        LipState = LIPState(p_CoM_fb, J_CoM_fb * qdot_fb_filt, zmp_3d_fb);
    } else {
        LipState = LIPState(p_CoM_sim, J_CoM_sim * qdot, zmp_3d_sim);
    }
    kf_LipState = updateKF(kf_LipState, LipState, ismpc_ptr_->getInput());

    // Fill current gait configuration (always uses sim_robot_data in simulation mode)
    labrob::GaitConfiguration current_gait_configuration;
    current_gait_configuration.qjnt = q.tail(njnt);
    current_gait_configuration.qjntdot = qdot.tail(njnt);

    current_gait_configuration.is_left_foot_support = true;
    current_gait_configuration.is_right_foot_support = true;
    if (walking_data_.getWalkingState() == WalkingState::SingleSupport) {
        if (walking_data_.footstep_plan.front().support_foot == Foot::LEFT) current_gait_configuration.is_right_foot_support = false;
        else if (walking_data_.footstep_plan.front().support_foot == Foot::RIGHT) current_gait_configuration.is_left_foot_support = false;
    }

    current_gait_configuration.com.pos = kf_LipState.com_pos_;
    current_gait_configuration.com.vel = kf_LipState.com_vel_;

    current_gait_configuration.torso.pos = sim_robot_data.oMf[torso_idx_].rotation();
    current_gait_configuration.torso.vel = J_torso_sim.bottomRows<3>() * qdot;

    current_gait_configuration.lsole.pos = labrob::SE3(sim_robot_data.oMf[lsole_idx_].rotation(), sim_robot_data.oMf[lsole_idx_].translation());
    current_gait_configuration.lsole.vel = J_lsole_sim * qdot;

    current_gait_configuration.rsole.pos = labrob::SE3(sim_robot_data.oMf[rsole_idx_].rotation(), sim_robot_data.oMf[rsole_idx_].translation());
    current_gait_configuration.rsole.vel = J_rsole_sim * qdot;

    /////////////////////////////////////
    // 
    // START MPC
    //
    /////////////////////////////////////

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

    LIPState lip_state;
    lip_state = discrete_lip_dynamics_ptr_->integrate(kf_LipState, ismpc_ptr_->getInput());

    // Eigen::VectorXd inputSequenceX = ismpc_ptr_->getInputSequenceX();
    // Eigen::VectorXd inputSequenceY = ismpc_ptr_->getInputSequenceY();
    // Eigen::VectorXd inputSequenceZ = ismpc_ptr_->getInputSequenceZ();

    // LIPState LipState_mpc = kf_LipState;

    // for (int i = 0; i < 20; ++i) {
    //     LipState_mpc = discrete_lip_dynamics_ptr_mpc_->integrate(
    //         LipState_mpc,
    //         Eigen::Vector3d(inputSequenceX(i), inputSequenceY(i), inputSequenceZ(i))
    //     );

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
        desired_gait_configuration.lsole.pos = labrob::SE3(walking_data_.footstep_plan.front().left_foot_rotation, walking_data_.footstep_plan.front().left_foot_position);
        desired_gait_configuration.lsole.vel = Eigen::VectorXd::Zero(6);
        desired_gait_configuration.lsole.acc = Eigen::VectorXd::Zero(6);
        desired_gait_configuration.rsole.pos = labrob::SE3(walking_data_.footstep_plan.front().right_foot_rotation, walking_data_.footstep_plan.front().right_foot_position);
        desired_gait_configuration.rsole.vel = Eigen::VectorXd::Zero(6);
        desired_gait_configuration.rsole.acc = Eigen::VectorXd::Zero(6);
    } else if (walking_data_.footstep_plan.front().support_foot == Foot::LEFT) {
        desired_gait_configuration.lsole.pos = labrob::SE3(walking_data_.footstep_plan.front().left_foot_rotation, walking_data_.footstep_plan.front().left_foot_position);
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
        desired_gait_configuration.rsole.pos = labrob::SE3(walking_data_.footstep_plan.front().right_foot_rotation, walking_data_.footstep_plan.front().right_foot_position);
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

    // Compute inverse dynamics (simulation mode uses sim_robot_state for kinematics, fb_robot_state for CoM)
    joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
        robot_model,
        sim_robot_state,
        fb_robot_state,
        sim_robot_data,
        fb_robot_data,
        current_gait_configuration,
        desired_gait_configuration
    );

    // Get measured joint torques from the joint command
    Eigen::VectorXd measured_torques(robot_model.nv - 6);  // Exclude floating base
    int idx = 0;
    for(pinocchio::JointIndex joint_id = 2; joint_id < (pinocchio::JointIndex) robot_model.njoints; ++joint_id) {
        const auto& joint_name = robot_model.names[joint_id];
        measured_torques(idx++) = joint_command[joint_name];
    }


    // Compute residual-based force estimation
    Eigen::VectorXd wbc_left_wrench = whole_body_controller_ptr_->getLeftFootWrench();
    Eigen::VectorXd wbc_right_wrench = whole_body_controller_ptr_->getRightFootWrench();

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





}

int64_t
WalkingManager::get_controller_frequency() const {
  return controller_frequency_;
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
  double swing_duration = 0.001 * static_cast<double>(walking_data_.footstep_plan.front().duration_ms);
  labrob::QuinticPolynomialTimingLaw timing_law(swing_duration);
  double s = timing_law.eval(t);
  double s_dot = timing_law.eval_dt(t);
  double s_ddot = timing_law.eval_dt_dt(t);

  const auto& current_footstep = walking_data_.footstep_plan[0];
  const auto& target_footstep = walking_data_.footstep_plan[1];
  const auto& support_foot_identity = current_footstep.support_foot;
  const auto& support_foot_rotation = current_footstep.get_support_foot_rotation();
  const auto& support_foot_position = current_footstep.get_support_foot_position();
  const auto& starting_swing_foot_rotation = current_footstep.get_swing_foot_rotation();
  const auto& starting_swing_foot_position = current_footstep.get_swing_foot_position();
  const Eigen::Matrix3d& target_swing_foot_rotation =
      (support_foot_identity == labrob::Foot::LEFT ?
             target_footstep.right_foot_rotation :
             target_footstep.left_foot_rotation
      );
  const Eigen::Vector3d& target_swing_foot_position =
      (support_foot_identity == labrob::Foot::LEFT ?
             target_footstep.right_foot_position :
             target_footstep.left_foot_position
      );
  const auto& p0 = starting_swing_foot_position;
  const auto& R0 = starting_swing_foot_rotation;
  const auto& pf = target_swing_foot_position;
  const auto& Rf = target_swing_foot_rotation;
  double yaw0 = std::atan2(R0(1, 0), R0(0, 0));
  double yawf = std::atan2(Rf(1, 0), Rf(0, 0));

  pinocchio::SE3 desired_swing_foot_pose;
  desired_swing_foot_pose.translation().x() = p0.x() + (pf.x() - p0.x()) * s;
  desired_swing_foot_pose.translation().y() = p0.y() + (pf.y() - p0.y()) * s;
  double zs = support_foot_position.z();
  double z0 = p0.z();
  double zf = pf.z();
  double h_z = walking_data_.footstep_plan[0].swing_height;
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
