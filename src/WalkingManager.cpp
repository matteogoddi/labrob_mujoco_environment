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
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <hrp4_locomotion/JointCommand.hpp>
#include <hrp4_locomotion/TimingLaw.hpp>
#include <hrp4_locomotion/utils.hpp>

namespace labrob {

bool
WalkingManager::init(const labrob::RobotState& initial_robot_state) {
  // Read URDF from file:
  std::string robot_description_filename = "/home/michele/repos/labrob_mujoco_environment/jvrc_description/urdf/jvrc1.urdf";

  // Build Pinocchio model and data from URDF:
  pinocchio::Model full_robot_model;
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::urdf::buildModel(
    robot_description_filename,
    root_joint,
    full_robot_model
  );
  const std::vector<std::string> joint_to_lock_names{
    "R_LTHUMB",
    "R_UTHUMB",
    "R_LINDEX",
    "R_UINDEX",
    "R_LLITTLE",
    "L_LTHUMB",
    "L_UTHUMB",
    "L_LINDEX",
    "L_UINDEX",
    "L_LLITTLE"
  };
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
  pinocchio::forwardKinematics(robot_model_, robot_data_, q_init);
  pinocchio::jacobianCenterOfMass(robot_model_, robot_data_, q_init);
  pinocchio::framesForwardKinematics(robot_model_, robot_data_, q_init);
  lsole_idx_ = robot_model_.getFrameId("L_ANKLE_P_S");
  rsole_idx_ = robot_model_.getFrameId("R_ANKLE_P_S");
  torso_idx_ = robot_model_.getFrameId("base_link");
  const auto& T_lsole_init = robot_data_.oMf[lsole_idx_];
  const auto& T_rsole_init = robot_data_.oMf[rsole_idx_];

  int njnt = robot_model_.nv - 6;

  double waist_p_des = 0.425;
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

  q_jnt_des_ = Eigen::VectorXd::Zero(njnt);
  q_jnt_des_(robot_model_.getJointId("WAIST_P") - 2) = waist_p_des;
  q_jnt_des_(robot_model_.getJointId("R_HIP_Y") - 2) = r_hip_y_des;
  q_jnt_des_(robot_model_.getJointId("R_HIP_R") - 2) = r_hip_r_des;
  q_jnt_des_(robot_model_.getJointId("R_HIP_P") - 2) = r_hip_p_des;
  q_jnt_des_(robot_model_.getJointId("R_KNEE") - 2) = r_knee_p_des;
  q_jnt_des_(robot_model_.getJointId("R_ANKLE_P") - 2) = r_ankle_p_des;
  q_jnt_des_(robot_model_.getJointId("R_ANKLE_R") - 2) = r_ankle_r_des;
  q_jnt_des_(robot_model_.getJointId("L_HIP_Y") - 2) = l_hip_y_des;
  q_jnt_des_(robot_model_.getJointId("L_HIP_R") - 2) = l_hip_r_des;
  q_jnt_des_(robot_model_.getJointId("L_HIP_P") - 2) = l_hip_p_des;
  q_jnt_des_(robot_model_.getJointId("L_KNEE") - 2) = l_knee_p_des;
  q_jnt_des_(robot_model_.getJointId("L_ANKLE_P") - 2) = l_ankle_p_des;
  q_jnt_des_(robot_model_.getJointId("L_ANKLE_R") - 2) = l_ankle_r_des;
  q_jnt_des_(robot_model_.getJointId("R_SHOULDER_P") - 2) = r_shoulder_p_des;
  q_jnt_des_(robot_model_.getJointId("R_SHOULDER_R") - 2) = r_shoulder_r_des;
  q_jnt_des_(robot_model_.getJointId("R_SHOULDER_Y") - 2) = r_shoulder_y_des;
  q_jnt_des_(robot_model_.getJointId("R_ELBOW_P") - 2) = r_elbow_p_des;
  q_jnt_des_(robot_model_.getJointId("L_SHOULDER_P") - 2) = l_shoulder_p_des;
  q_jnt_des_(robot_model_.getJointId("L_SHOULDER_R") - 2) = l_shoulder_r_des;
  q_jnt_des_(robot_model_.getJointId("L_SHOULDER_Y") - 2) = l_shoulder_y_des;
  q_jnt_des_(robot_model_.getJointId("L_ELBOW_P") - 2) = l_elbow_p_des;

  // TODO: init using node handle.
  int64_t controller_frequency = 100;
  controller_timestep_msec_ = 1000 / controller_frequency;

  double swing_foot_trajectory_height = 0.02;
  double step_length_x = 0.25;
  double step_length_y = 0.00;
  int n_steps = 50;
  walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
      labrob::DoubleSupportConfiguration(
          labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
          labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
          labrob::Foot::RIGHT
      ),
      0.0,
      10000,
      labrob::WalkingState::Init
  ));
  walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
      labrob::DoubleSupportConfiguration(
          labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
          labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
          labrob::Foot::RIGHT
      ),
      0.0,
      10000,
      labrob::WalkingState::Standing
  ));
/*
  walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
      labrob::DoubleSupportConfiguration(
          labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
          labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
          labrob::Foot::RIGHT
      ),
      0.0,
      600,
      labrob::WalkingState::Starting
  ));
  walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
      labrob::DoubleSupportConfiguration(
          labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation()),
          labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation()),
          labrob::Foot::RIGHT
      ),
      swing_foot_trajectory_height,
      600,
      labrob::WalkingState::SingleSupport
  ));
  for (int n = 0; n < n_steps; n += 2) {
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation() + n * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::Foot::RIGHT
        ),
        0.0,
        400,
        labrob::WalkingState::DoubleSupport
    ));
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation() + n * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::Foot::LEFT
        ),
        swing_foot_trajectory_height,
        600,
        labrob::WalkingState::SingleSupport
    ));
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation() + (n + 2) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::Foot::LEFT
        ),
        0.0,
        400,
        labrob::WalkingState::DoubleSupport
    ));
    walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation() + (n + 2) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::Foot::RIGHT
        ),
        swing_foot_trajectory_height,
        600,
        labrob::WalkingState::SingleSupport
    ));
  }
  walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
      labrob::DoubleSupportConfiguration(
          labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
          labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
          labrob::Foot::RIGHT
      ),
      0.0,
      300,
      labrob::WalkingState::Stopping
  ));
  walking_data_.footstep_plan.push_back(labrob::FootstepPlanElement(
      labrob::DoubleSupportConfiguration(
          labrob::SE3(T_lsole_init.rotation(), T_lsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
          labrob::SE3(T_rsole_init.rotation(), T_rsole_init.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
          labrob::Foot::RIGHT
      ),
      0.0,
      2000,
      labrob::WalkingState::Standing
  ));
*/
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
  std::cerr << "CoM target height: " << com_target_height << std::endl;
  double foot_constraint_square_width = 0.05;
  Eigen::Vector3d p_ZMP = p_CoM - Eigen::Vector3d(0.0, 0.0, com_target_height);
  labrob::ISMPCState mpc_init_state(
      p_CoM,
      Eigen::Vector3d::Zero(),
      p_ZMP
  );
  ismpc_ptr_ = std::make_unique<labrob::ISMPC>(
      mpc_prediction_horizon_msec,
      mpc_timestep_msec,
      controller_timestep_msec_,
      com_target_height,
      foot_constraint_square_width,
      mpc_init_state
  );

  // Init QP solver:
  qp_solver_ptr_ = std::make_unique<labrob::qpsolvers::QPSolverEigenWrapper<double>>(
      std::make_shared<labrob::qpsolvers::HPIPMQPSolver>(
          6 + njnt, 12, 2 * njnt
      )
  );

  // Init log files:
  // TODO: may be better to use a proper logging system such as glog.
  mpc_timings_log_file_.open("/tmp/mpc_timings.txt");
  com_log_file_.open("/tmp/com.txt");
  zmp_log_file_.open("/tmp/zmp.txt");
  //configuration_log_file_.open("/tmp/configuration.txt");
  lsole_log_file_.open("/tmp/lsole.txt");
  rsole_log_file_.open("/tmp/rsole.txt");
  lsole_des_log_file_.open("/tmp/lsole_des.txt");
  rsole_des_log_file_.open("/tmp/rsole_des.txt");

  return true;
}

void
WalkingManager::update(
    const labrob::RobotState& robot_state,
    labrob::JointCommand& joint_command) {

  double controller_timestep = 0.001 * static_cast<double>(controller_timestep_msec_);

  int njnt = robot_model_.nv - 6; // size of configuration space without floating base

  auto q = robot_state_to_pinocchio_joint_configuration(robot_model_, robot_state);

  // Perform forward kinematics on the whole tree and update robot data:
  pinocchio::forwardKinematics(robot_model_, robot_data_, q);

  // NOTE: jacobianCenterOfMass calls forwardKinematics and
  //       computeJointJacobians.
  pinocchio::jacobianCenterOfMass(robot_model_, robot_data_, q);
  pinocchio::framesForwardKinematics(robot_model_, robot_data_, q);

  const auto& p_CoM = robot_data_.com[0];
  const auto& J_CoM = robot_data_.Jcom;
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
  auto J_torso_orientation = J_torso.bottomRows<3>();
  const auto& T_lsole = robot_data_.oMf[lsole_idx_];
  Eigen::MatrixXd J_lsole = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  pinocchio::getFrameJacobian(
      robot_model_,
      robot_data_,
      lsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_lsole
  );
  const auto& T_rsole = robot_data_.oMf[rsole_idx_];
  Eigen::MatrixXd J_rsole = Eigen::MatrixXd::Zero(6, robot_model_.nv);
  pinocchio::getFrameJacobian(
      robot_model_,
      robot_data_,
      rsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_rsole
  );

  std::cerr << "lsole position: " << T_lsole.translation().transpose() << std::endl;
  std::cerr << "rsole position: " << T_rsole.translation().transpose() << std::endl;

  // Update walking state:
  walking_data_.updateWalkingState(t_msec_);
  
  // Update CoM target height in IS-MPC depending on current configuration
  // in case of PostureRegulation walking state:
  /*if (walking_data_.getWalkingState() == labrob::WalkingState::PostureRegulation) {
      const auto& p_support = walking_data_.footstep_plan.front().getFeetPlacement().getSupportFootConfiguration().p;
      auto com_target_height = p_CoM.z() - p_support.z();
      Eigen::Vector3d p_ZMP = p_CoM - Eigen::Vector3d(0.0, 0.0, com_target_height);
      ismpc_ptr_->setCOMTargetHeight(com_target_height);
      ismpc_ptr_->setState(labrob::ISMPCState(
          p_CoM,
          Eigen::Vector3d::Zero(),
          p_ZMP
      ));
  }*/

  // Update first element of footstep plan to make it consistent with 
  // state estimation module:
  //walking_data_.updateFootstepPlanWithCurrentStance(
  //    labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
  //    labrob::SE3(T_rsole.rotation(), T_rsole.translation())
  //);

  std::cerr << "t (msec):" << t_msec_ << std::endl;
  std::cerr << "\tWalkingState::" << to_string(walking_data_.getWalkingState()) << std::endl;
  std::cerr << "\tFootstep plan size: " << walking_data_.footstep_plan.size() << std::endl;
  for (const auto& footstep_plan_element : walking_data_.footstep_plan) {
    const auto& feet_placement = footstep_plan_element.getFeetPlacement();
    auto duration = footstep_plan_element.getDuration();
    auto h_z = footstep_plan_element.getSwingFootTrajectoryHeight();
    auto walking_state = footstep_plan_element.getWalkingState();
    auto support_foot = footstep_plan_element.getFeetPlacement().getSupportFoot();
    const auto& left_foot_configuration = feet_placement.getLeftFootConfiguration();
    const auto& right_foot_configuration = feet_placement.getRightFootConfiguration();
    std::cerr << "qL=" << left_foot_configuration.p.transpose()
        << ", qR=" << right_foot_configuration.p.transpose()
        << ", T=" << duration
        << ", h_z=" << h_z
        << ", support foot=" << (support_foot == labrob::Foot::LEFT ? "LEFT" : "RIGHT")
        << ", walking state="
        << labrob::to_string(walking_state)
        << std::endl;
  }

  // CoM task:
  auto mpc_t0_ms = std::chrono::system_clock::now();
  ismpc_ptr_->solve(t_msec_, walking_data_);
  auto mpc_tf_ms = std::chrono::system_clock::now();
  const auto& ismpc_state = ismpc_ptr_->getState();
  Eigen::Vector3d v_CoM_des = ismpc_state.com_vel_;
  Eigen::Vector3d p_CoM_des = ismpc_state.com_pos_;
  Eigen::Vector3d p_ZMP_des = ismpc_state.zmp_pos_;

  std::cerr << "p_CoM_des: " << p_CoM_des.transpose() << std::endl;

  // CoM task error:
  /*p_CoM_des << 0.05, 0.0, ismpc_ptr_->getCOMTargetHeight();
  v_CoM_des.setZero();
  p_ZMP_des.setZero();*/
  auto err_CoM = p_CoM_des - p_CoM;

  // Torso orientation task (NOTE: choose orientation as the average of
  // orientations of feet configuration in the footstep plan, assuming WoS-like
  // rotation matrix, hence roll=pitch=0.0):
  double left_foot_orientation = std::atan2(
      walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().R(1, 0),
      walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().R(0, 0)
  );
  double right_foot_orientation = std::atan2(
      walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().R(1, 0),
      walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().R(0, 0)
  );
  Eigen::Matrix3d torso_orientation_des =
      labrob::Rz((left_foot_orientation + right_foot_orientation) / 2.0);
  Eigen::Vector3d v_torso_orientation_des = Eigen::Vector3d::Zero();
  auto err_torso_orientation = err_rotation(torso_orientation_des, torso_orientation);

  // Support foot and swing foot tasks:
  pinocchio::SE3 T_lsole_des(
      walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().R,
      walking_data_.footstep_plan.front().getFeetPlacement().getLeftFootConfiguration().p
  );
  Eigen::VectorXd v_lsole_des = Eigen::VectorXd::Zero(6);
  pinocchio::SE3 T_rsole_des(
      walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().R,
      walking_data_.footstep_plan.front().getFeetPlacement().getRightFootConfiguration().p
  );
  Eigen::VectorXd v_rsole_des = Eigen::VectorXd::Zero(6);

  if (walking_data_.getWalkingState() == labrob::WalkingState::SingleSupport) {
    pinocchio::SE3 T_support_des(
        walking_data_.footstep_plan.front().getFeetPlacement().getSupportFootConfiguration().R,
        walking_data_.footstep_plan.front().getFeetPlacement().getSupportFootConfiguration().p
    );
    pinocchio::Motion v_support_des(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    pinocchio::SE3 T_swing_des;
    pinocchio::Motion v_swing_des;
    swingFootTrajectoryBezier(T_swing_des, v_swing_des);

    if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == labrob::Foot::LEFT) {
      // lsole is support:
      T_lsole_des = T_support_des;
      v_lsole_des << v_support_des.linear(), v_support_des.angular();
      // rsole is swinging:
      T_rsole_des = T_swing_des;
      v_rsole_des << v_swing_des.linear(), v_swing_des.angular();
    } else {
      // lsole is swinging:
      T_lsole_des = T_swing_des;
      v_lsole_des << v_swing_des.linear(), v_swing_des.angular();
      // rsole is support:
      T_rsole_des = T_support_des;
      v_rsole_des << v_support_des.linear(), v_support_des.angular();
    }
  }

  // lsole task error:
  auto err_lsole = err_frameplacement(T_lsole_des, T_lsole);

  // rsole task error:
  auto err_rsole = err_frameplacement(T_rsole_des, T_rsole);

  // Posture regulation task error (deactivate legs if not doing posture regulation):
  Eigen::MatrixXd err_posture_selection_matrix = Eigen::MatrixXd::Identity(njnt, njnt);
  if (walking_data_.getWalkingState() != labrob::WalkingState::PostureRegulation) {
    const std::vector<std::string> legs_joint_name = {
        "R_HIP_Y", "R_HIP_R", "R_HIP_P", "R_KNEE", "R_ANKLE_P", "R_ANKLE_R",
        "L_HIP_Y", "L_HIP_R", "L_HIP_P", "L_KNEE", "L_ANKLE_P", "L_ANKLE_R"
    };
    for (const auto& joint_name : legs_joint_name) {
      auto idx = robot_model_.getJointId(joint_name) - 2;
      err_posture_selection_matrix(idx, idx) = 0.0;
    }
  }
  Eigen::VectorXd err_posture(6 + njnt);
  err_posture << Eigen::VectorXd::Zero(6), err_posture_selection_matrix * (q_jnt_des_ - q.tail(njnt));

  bool is_left_foot_support = true;
  bool is_right_foot_support = true;
  if (walking_data_.getWalkingState() == labrob::WalkingState::SingleSupport) {
    if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == labrob::Foot::LEFT) is_right_foot_support = false;
    else if (walking_data_.footstep_plan.front().getFeetPlacement().getSupportFoot() == labrob::Foot::RIGHT) is_left_foot_support = false;
  }

  // CLIK-weights:
  Eigen::Matrix3d K_CoM = 50.0 * Eigen::Matrix3d::Identity();
  Eigen::Matrix3d K_torso_orientation = Eigen::Vector3d(50.0, 50.0, 50.0).asDiagonal().toDenseMatrix();
  Eigen::MatrixXd K_lsole = Eigen::MatrixXd::Identity(6, 6);
  K_lsole.diagonal().head<3>().setConstant(50.0);
  K_lsole.diagonal().tail<3>().setConstant(50.0);
  Eigen::MatrixXd K_rsole = Eigen::MatrixXd::Identity(6, 6);
  K_rsole.diagonal().head<3>().setConstant(50.0);
  K_rsole.diagonal().tail<3>().setConstant(50.0);

  // QP-based task priority IK:
  Eigen::MatrixXd cost_function_H = Eigen::MatrixXd::Zero(6 + njnt, 6 + njnt);
  Eigen::VectorXd cost_function_f = Eigen::VectorXd::Zero(6 + njnt);
  Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(12, 6 + njnt);
  Eigen::VectorXd b_eq = Eigen::VectorXd::Zero(12);
  Eigen::MatrixXd C_ineq = Eigen::MatrixXd::Zero(2 * njnt, 6 + njnt);
  Eigen::VectorXd d_min_ineq(2 * njnt);
  Eigen::VectorXd d_max_ineq(2 * njnt);

  // Set weights depending on walking state:
  double weight_q_dot = 0.0;
  double weight_com_task = 0.1;//1.0;
  double weight_lsole_task = 1.0;
  double weight_rsole_task = 1.0;
  double weight_torso_orientation_task = 0.0;
  double weight_posture_regulation_task = 0.0;

  if (is_left_foot_support) weight_lsole_task = 100.0;
  if (is_right_foot_support) weight_rsole_task = 100.0;

  if (walking_data_.footstep_plan.front().getWalkingState() == WalkingState::PostureRegulation) {
    weight_posture_regulation_task = 1.0;
  } else {
    weight_q_dot = 1e-4;
    weight_torso_orientation_task = 0.25; // 1e-3
    weight_posture_regulation_task = 0.25; // 0.01; // (legs not considered)
  }

  cost_function_H += weight_q_dot * Eigen::MatrixXd::Identity(6 + njnt, 6 + njnt);
  cost_function_H += weight_com_task * (J_CoM.transpose() * J_CoM);
  cost_function_H += weight_lsole_task * (J_lsole.transpose() * J_lsole);
  cost_function_H += weight_rsole_task * (J_rsole.transpose() * J_rsole);
  cost_function_H += weight_torso_orientation_task * (J_torso_orientation.transpose() * J_torso_orientation);
  cost_function_H += weight_posture_regulation_task * std::pow(controller_timestep, 2.0) * Eigen::MatrixXd::Identity(6 + njnt, 6 + njnt);
  cost_function_f += -weight_com_task * J_CoM.transpose() * (v_CoM_des + K_CoM * err_CoM);
  cost_function_f += -weight_lsole_task * J_lsole.transpose() * (v_lsole_des + K_lsole * err_lsole);
  cost_function_f += -weight_rsole_task * J_rsole.transpose() * (v_rsole_des + K_rsole * err_rsole);
  cost_function_f += -weight_torso_orientation_task * J_torso_orientation.transpose() * (v_torso_orientation_des + K_torso_orientation * err_torso_orientation);
  cost_function_f += -weight_posture_regulation_task * controller_timestep * err_posture;

  auto q_jnt_dot_min = -robot_model_.velocityLimit.tail(njnt);
  auto q_jnt_dot_max =  robot_model_.velocityLimit.tail(njnt);
  auto q_jnt_min = robot_model_.lowerPositionLimit.tail(njnt);
  auto q_jnt_max = robot_model_.upperPositionLimit.tail(njnt);
  if (is_left_foot_support) {
    A_eq.topRows(6) = J_lsole;
    // NOTE: the following is useful to correct kinematic simulation errors.
    b_eq.topRows(6) = K_lsole * err_lsole;
  }
  if (is_right_foot_support) {
    A_eq.bottomRows(6) = J_rsole;
    // NOTE: the following is useful to correct kinematic simulation errors.
    b_eq.bottomRows(6) = K_rsole * err_rsole;
  }
  C_ineq.rightCols(njnt).topRows(njnt).diagonal().setConstant(1.0);
  C_ineq.rightCols(njnt).bottomRows(njnt).diagonal().setConstant(controller_timestep);
  d_min_ineq << q_jnt_dot_min, q_jnt_min - q.tail(njnt);
  d_max_ineq << q_jnt_dot_max, q_jnt_max - q.tail(njnt);

  qp_solver_ptr_->solve(
      cost_function_H,
      cost_function_f,
      A_eq,
      b_eq,
      C_ineq,
      d_min_ineq,
      d_max_ineq
  );

  Eigen::VectorXd q_dot = qp_solver_ptr_->get_solution();

  // Pinocchio representation to labrob::RobotState representation:
  // TODO: is there a less error-prone way to convert representation?
  for(pinocchio::JointIndex joint_id = 2;
      joint_id < (pinocchio::JointIndex) robot_model_.njoints;
      ++joint_id) {
    const auto& joint_name = robot_model_.names[joint_id];
    joint_command[joint_name] = q_dot[joint_id + 4];
  }

  // Update timing in milliseconds.
  // NOTE: assuming update() is actually called every controller_timestep_msec_
  //       milliseconds.
  t_msec_ += controller_timestep_msec_;

  // Log:
  mpc_timings_log_file_ << std::chrono::duration_cast<std::chrono::microseconds>(mpc_tf_ms - mpc_t0_ms).count() << std::endl;
  com_log_file_ << p_CoM_des.transpose() << std::endl;
  zmp_log_file_ << p_ZMP_des.transpose() << std::endl;
  lsole_log_file_ << T_lsole.translation().transpose() << std::endl;
  rsole_log_file_ << T_rsole.translation().transpose() << std::endl;
  lsole_des_log_file_ << T_lsole_des.translation().transpose() << std::endl;
  rsole_des_log_file_ << T_rsole_des.translation().transpose() << std::endl;
}

Eigen::VectorXd
WalkingManager::robot_state_to_pinocchio_joint_configuration(
    const pinocchio::Model& robot_model,
    const labrob::RobotState& robot_state
) {
  // labrob::RobotState representation to Pinocchio representation:
  // TODO: RobotState also has information about the velocity of the floating base.
  // TODO: is there a less error-prone way to convert representation?
  Eigen::VectorXd q(robot_model.nq);
  q.head<3>() = robot_state.position;
  q.segment<4>(3) = robot_state.orientation.coeffs();
  // NOTE: start from joint id (2) to skip frames "universe" and "root_joint".
  for(pinocchio::JointIndex joint_id = 2;
      joint_id < (pinocchio::JointIndex) robot_model.njoints;
      ++joint_id) {
    const auto& joint_name = robot_model.names[joint_id];
    q[joint_id + 5] = robot_state.joint_state[joint_name].pos;
  }

  return q;
}

Eigen::MatrixXd
WalkingManager::pseudoinverse(const Eigen::MatrixXd& J, double damp) const {
  auto J_T = J.transpose();
  auto Id = Eigen::MatrixXd::Identity(J.cols(), J.cols());
  return (J_T * J + damp * Id).inverse() * J_T;
}

Eigen::Matrix<double, 6, 1>
WalkingManager::err_frameplacement(const pinocchio::SE3& Ta, const pinocchio::SE3& Tb) {
  // TODO: how do you use pinocchio::log6?
  Eigen::Matrix<double, 6, 1> err;
  err << err_translation(Ta.translation(), Tb.translation()),
         err_rotation(Ta.rotation(), Tb.rotation());
  return err;
}

Eigen::Vector3d
WalkingManager::err_translation(
    const Eigen::Vector3d& pa,
    const Eigen::Vector3d& pb) {
  return pa - pb;
}

Eigen::Vector3d
WalkingManager::err_rotation(
    const Eigen::Matrix3d& Ra,
    const Eigen::Matrix3d& Rb) {
  // TODO: how do you use pinocchio::log3?
  Eigen::Matrix3d Rdiff = Rb.transpose() * Ra;
  auto aa = Eigen::AngleAxisd(Rdiff);
  return aa.angle() * Ra * aa.axis();
}

void
WalkingManager::swingFootTrajectory(
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

  swing_foot_pose = desired_swing_foot_pose;
  swing_foot_velocity = desired_swing_foot_velocity;
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