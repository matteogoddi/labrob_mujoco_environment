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
// #include <hrp4_locomotion/TimingLaw.hpp>
#include <hrp4_locomotion/utils.hpp>

namespace labrob {

WalkingManager::WalkingManager(){}

bool
WalkingManager::init(const labrob::RobotState& initial_robot_state,
                     std::map<std::string, double> &armatures) {

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
  auto q_init = robot_state_to_pinocchio_joint_configuration(robot_model_, initial_robot_state);

  pinocchio::forwardKinematics(robot_model_, robot_data_, q_init);
  pinocchio::jacobianCenterOfMass(robot_model_, robot_data_, q_init);
  pinocchio::framesForwardKinematics(robot_model_, robot_data_, q_init);

  lsole_idx_ = robot_model_.getFrameId("left_foot_link");
  rsole_idx_ = robot_model_.getFrameId("right_foot_link");
  torso_idx_ = robot_model_.getFrameId("torso_link");
  pelvis_idx_ = robot_model_.getFrameId("pelvis");
  const auto& T_lsole_init = robot_data_.oMf[lsole_idx_];
  const auto& T_rsole_init = robot_data_.oMf[rsole_idx_];

  Eigen::Vector3d pcom_init = robot_data_.com[0];
  Eigen::Matrix3d R_pelvis_init = robot_data_.oMf[pelvis_idx_].rotation();




  int njnt = robot_model_.nv - 6;

  M_armature_ = Eigen::VectorXd::Zero(njnt);
  for(pinocchio::JointIndex joint_id = 2;
      joint_id < (pinocchio::JointIndex) robot_model_.njoints;
      ++joint_id) {
    std::string joint_name = robot_model_.names[joint_id];
    M_armature_(joint_id - 2) = armatures[joint_name];
  }


  
  // TODO: init using node handle.
  controller_frequency_ = 500;
  controller_timestep_msec_ = 1000 / controller_frequency_;
  
  auto params = WholeBodyControllerParams::getDefaultParams();
  whole_body_controller_ptr_ = std::make_shared<WholeBodyController>(
    params,
    robot_model_,
    q_jnt_des_,
    0.001 * controller_timestep_msec_,
    armatures
  );

  
  q_jnt_des_ = q_init.tail(njnt);


  // Fill desired gait configuration:
  desired_gait_configuration_.qjnt = q_jnt_des_;
  desired_gait_configuration_.qjntdot = Eigen::VectorXd::Zero(njnt);
  desired_gait_configuration_.qjntddot = Eigen::VectorXd::Zero(njnt);

  desired_gait_configuration_.com.pos = pcom_init;
  desired_gait_configuration_.com.vel = Eigen::Vector3d(0,0,0);
  desired_gait_configuration_.com.acc = Eigen::Vector3d(0,0,0);

  // Feet tasks
  desired_gait_configuration_.lsole.pos.p = T_lsole_init.translation();
  desired_gait_configuration_.lsole.pos.R = T_lsole_init.rotation();
  desired_gait_configuration_.lsole.vel = Eigen::VectorXd::Zero(6);
  desired_gait_configuration_.lsole.acc = Eigen::VectorXd::Zero(6);
  desired_gait_configuration_.rsole.pos.p = T_rsole_init.translation();
  desired_gait_configuration_.rsole.pos.R = T_rsole_init.rotation();
  desired_gait_configuration_.rsole.vel = Eigen::VectorXd::Zero(6);
  desired_gait_configuration_.rsole.acc = Eigen::VectorXd::Zero(6);

  

  // Torso task
  double left_foot_yaw = std::atan2(desired_gait_configuration_.lsole.pos.R(1, 0), desired_gait_configuration_.lsole.pos.R(0, 0));
  double right_foot_yaw = std::atan2(desired_gait_configuration_.rsole.pos.R(1, 0), desired_gait_configuration_.rsole.pos.R(0, 0));
  desired_gait_configuration_.torso.pos = Rz((left_foot_yaw + right_foot_yaw) / 2.0);
  desired_gait_configuration_.torso.vel = (desired_gait_configuration_.lsole.vel.tail(3) + desired_gait_configuration_.rsole.vel.tail(3)) / 2.0;
  desired_gait_configuration_.torso.acc = (desired_gait_configuration_.lsole.acc.tail(3) + desired_gait_configuration_.rsole.acc.tail(3)) / 2.0;




  // Eigen::Matrix3d R;
  // double pitch = -0.1;
  // R <<  cos(pitch), 0, sin(pitch),
  //       0,          1, 0,
  //     -sin(pitch), 0, cos(pitch);

  // desired_gait_configuration_.torso.pos = R;
  // desired_gait_configuration_.torso.vel = Eigen::Vector3d::Zero();
  // desired_gait_configuration_.torso.acc = Eigen::Vector3d::Zero();





  // Pelvis task
  desired_gait_configuration_.pelvis.pos = Rz((left_foot_yaw + right_foot_yaw) / 2.0);
  desired_gait_configuration_.pelvis.vel = (desired_gait_configuration_.lsole.vel.tail(3) + desired_gait_configuration_.rsole.vel.tail(3)) / 2.0;
  desired_gait_configuration_.pelvis.acc = (desired_gait_configuration_.lsole.acc.tail(3) + desired_gait_configuration_.rsole.acc.tail(3)) / 2.0;







  // Init logger
  logger_.reserve(20000);

  return true;
}

void
WalkingManager::update(
    const labrob::RobotState& robot_state,
    labrob::JointCommand& joint_command
) {

    auto start_time = std::chrono::system_clock::now();

    double controller_timestep = 0.001 * static_cast<double>(controller_timestep_msec_);

    int njnt = robot_model_.nv - 6; // size of configuration space without floating base

    auto q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);
    auto qdot = robot_state_to_pinocchio_joint_velocity(robot_model_, robot_state);
    
    const auto& centroidal_momentum_matrix = pinocchio::ccrba(
        robot_model_,
        robot_data_,
        q,
        qdot
    );
    auto angular_momentum = (centroidal_momentum_matrix * qdot).tail<3>();

    const auto& p_CoM = robot_data_.com[0];
    const auto& v_CoM = robot_data_.vcom[0];
    const auto& a_CoM_drift = robot_data_.acom[0];
    const auto& J_CoM = robot_data_.Jcom;
    const auto& T_torso = robot_data_.oMf[torso_idx_];
    const auto& T_pelvis = robot_data_.oMf[pelvis_idx_];
    auto torso_orientation = T_torso.rotation();
    auto pelvis_orientation = T_pelvis.rotation();
    Eigen::MatrixXd J_torso = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    pinocchio::getFrameJacobian(robot_model_, robot_data_, torso_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_torso);

    Eigen::MatrixXd J_pelvis = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    pinocchio::getFrameJacobian(robot_model_, robot_data_, pelvis_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_pelvis);

    const auto& T_lsole = robot_data_.oMf[lsole_idx_];
    Eigen::MatrixXd J_lsole = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    pinocchio::getFrameJacobian(robot_model_, robot_data_, lsole_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lsole);

    const auto& v_lsole = J_lsole * qdot;
    // Eigen::MatrixXd J_lsole_dot = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    // pinocchio::getFrameJacobianTimeVariation(robot_model_, robot_data_, lsole_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_lsole_dot);
    
    const auto& T_rsole = robot_data_.oMf[rsole_idx_];
    Eigen::MatrixXd J_rsole = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    pinocchio::getFrameJacobian(robot_model_, robot_data_, rsole_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rsole);
    const auto& v_rsole = J_rsole * qdot;
    // Eigen::MatrixXd J_rsole_dot = Eigen::MatrixXd::Zero(6, robot_model_.nv);
    // pinocchio::getFrameJacobianTimeVariation(robot_model_, robot_data_, rsole_idx_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_rsole_dot);

    
    // Fill current gait configuration:
    labrob::GaitConfiguration current_gait_configuration;
    current_gait_configuration.qjnt = q.tail(njnt);
    current_gait_configuration.qjntdot = qdot.tail(njnt);

    current_gait_configuration.is_left_foot_support = true;
    current_gait_configuration.is_right_foot_support = true;
   
    current_gait_configuration.com.pos = robot_data_.com[0];
    current_gait_configuration.com.vel = robot_data_.vcom[0];

    current_gait_configuration.torso.pos = robot_data_.oMf[torso_idx_].rotation();
    current_gait_configuration.torso.vel = J_torso.bottomRows<3>() * qdot;

    current_gait_configuration.pelvis.pos = robot_data_.oMf[pelvis_idx_].rotation();
    current_gait_configuration.pelvis.vel = J_pelvis.bottomRows<3>() * qdot;

    current_gait_configuration.lsole.pos = labrob::SE3(robot_data_.oMf[lsole_idx_].rotation(), robot_data_.oMf[lsole_idx_].translation());
    current_gait_configuration.lsole.vel = J_lsole * qdot;

    current_gait_configuration.rsole.pos = labrob::SE3(robot_data_.oMf[rsole_idx_].rotation(), robot_data_.oMf[rsole_idx_].translation());
    current_gait_configuration.rsole.vel = J_rsole * qdot;

    

    // std::cout << "Current CoM pos: " << robot_data_.com[0] << std::endl;
    // std::cout << "Current left foot pos: " << robot_data_.oMf[lsole_idx_].translation().transpose() << std::endl;
    // std::cout << "Current right foot pos: " << robot_data_.oMf[rsole_idx_].translation().transpose() << std::endl;


    // Eigen::Vector3d v_CoM_des = Eigen::Vector3d(0,0,0);
    // Eigen::Vector3d p_CoM_des = Eigen::Vector3d(0.04,0,0.53);

    // if (t_msec_ > 5000) {
    //   desired_gait_configuration_.com.pos.x() = 0.035;
    //     // p_CoM_des.z() = 0.55;
    // }

    // if (t_msec_ >6000) {
    //     p_CoM_des.z() = 0.7;
    // }

    // if (t_msec_ >8000) {
    //     p_CoM_des.z() = 0.55;
    // }

    // if (t_msec_ >9000) {
    //     p_CoM_des.z() = 0.7;
    // }



    // // Fill desired gait configuration:
    // desired_gait_configuration_.com.pos = p_CoM_des;
    // desired_gait_configuration_.com.vel = v_CoM_des;
    // desired_gait_configuration_.com.acc = Eigen::Vector3d(0,0,0);

    

    auto start_wbc_time = std::chrono::system_clock::now();
    joint_command = whole_body_controller_ptr_->compute_inverse_dynamics(
        robot_model_,
        robot_state,
        robot_data_,
        current_gait_configuration,
        desired_gait_configuration_
    );
    auto end_wbc_time = std::chrono::system_clock::now();
    auto time_wbc = std::chrono::duration_cast<std::chrono::microseconds>(end_wbc_time - start_wbc_time).count();


    auto end_time = std::chrono::system_clock::now();
    auto controller_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // std::cout << "WalkingManager::update() took " << elapsed_time << " us" << std::endl;


    // Update timing in milliseconds.
    // NOTE: assuming update() is actually called every controller_timestep_msec_
    //       milliseconds.
    t_msec_ += controller_timestep_msec_;
    prev_angular_momentum_ = angular_momentum;


    // log WBC data
    labrob::WBCEntry wbc_entry;
    wbc_entry.time_ms = t_msec_;

    wbc_entry.com_x = p_CoM(0); 
    wbc_entry.com_y = p_CoM(1); 
    wbc_entry.com_z = p_CoM(2);

    wbc_entry.com_x_des = desired_gait_configuration_.com.pos(0); 
    wbc_entry.com_y_des = desired_gait_configuration_.com.pos(1); 
    wbc_entry.com_z_des = desired_gait_configuration_.com.pos(2);

    wbc_entry.com_vel_x = v_CoM(0); 
    wbc_entry.com_vel_y = v_CoM(1); 
    wbc_entry.com_vel_z = v_CoM(2);

    wbc_entry.com_vel_x_des = desired_gait_configuration_.com.vel(0); 
    wbc_entry.com_vel_y_des = desired_gait_configuration_.com.vel(1); 
    wbc_entry.com_vel_z_des = desired_gait_configuration_.com.vel(2);


    wbc_entry.l_sole_x = current_gait_configuration.lsole.pos.p(0);
    wbc_entry.l_sole_y = current_gait_configuration.lsole.pos.p(1);
    wbc_entry.l_sole_z = current_gait_configuration.lsole.pos.p(2);
   
    wbc_entry.l_sole_x_des = desired_gait_configuration_.lsole.pos.p(0);
    wbc_entry.l_sole_y_des = desired_gait_configuration_.lsole.pos.p(1);
    wbc_entry.l_sole_z_des = desired_gait_configuration_.lsole.pos.p(2);

    wbc_entry.r_sole_x = current_gait_configuration.rsole.pos.p(0);
    wbc_entry.r_sole_y = current_gait_configuration.rsole.pos.p(1);
    wbc_entry.r_sole_z = current_gait_configuration.rsole.pos.p(2);

    wbc_entry.r_sole_x_des = desired_gait_configuration_.rsole.pos.p(0);
    wbc_entry.r_sole_y_des = desired_gait_configuration_.rsole.pos.p(1);
    wbc_entry.r_sole_z_des = desired_gait_configuration_.rsole.pos.p(2);

    Eigen::Quaterniond q_pelvis(T_pelvis.rotation());
    Eigen::Quaterniond q_torso(T_torso.rotation());

    wbc_entry.q_w_pelvis = q_pelvis.w();
    wbc_entry.q_x_pelvis = q_pelvis.x();
    wbc_entry.q_y_pelvis = q_pelvis.y();
    wbc_entry.q_z_pelvis = q_pelvis.z();

    wbc_entry.q_w_torso = q_torso.w();
    wbc_entry.q_x_torso = q_torso.x();
    wbc_entry.q_y_torso = q_torso.y();
    wbc_entry.q_z_torso = q_torso.z();




    logger_.log_wbc_data(std::move(wbc_entry));

    // log timings data
    labrob::TimingEntry timing;
    timing.time_wbc_us = time_wbc;
    timing.total_time_us = controller_time;

    logger_.log_timing_data(std::move(timing));


}


void WalkingManager::save_data(){
    logger_.save_log_data();
}


} // end namespace labrob
