#ifndef LABROB_WALKING_MANAGER_HPP_
#define LABROB_WALKING_MANAGER_HPP_

#include <cmath>
#include <string>
#include <vector>

// Pinocchio
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

// #include <hrp4_locomotion/DiscreteLIPDynamics.hpp>
// #include <hrp4_locomotion/ISMPC.hpp>
#include <hrp4_locomotion/JointCommand.hpp>
#include <hrp4_locomotion/JointState.hpp>
#include <hrp4_locomotion/RobotState.hpp>
// #include <hrp4_locomotion/WalkingData.hpp>
#include <hrp4_locomotion/utils.hpp>
#include <hrp4_locomotion/WholeBodyController.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>

#include <hrp4_locomotion/Logger.hpp>


namespace labrob {

class WalkingManager {
 public:
  WalkingManager();

  bool init(const labrob::RobotState& initial_robot_state, std::map<std::string, double> &armatures);

  void update(
      const labrob::RobotState& robot_state,
      labrob::JointCommand& joint_command
  );


  void save_data();
  
 protected:
  pinocchio::Model robot_model_;
  pinocchio::Data robot_data_;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;
  pinocchio::FrameIndex torso_idx_;
  pinocchio::FrameIndex pelvis_idx_;

  Eigen::VectorXd q_jnt_des_;

  int64_t controller_timestep_msec_;


  Eigen::VectorXd M_armature_;
  Eigen::VectorXd q_next_prev_;
  Eigen::VectorXd v_next_prev_;


  std::shared_ptr<WholeBodyController> whole_body_controller_ptr_;

private:

  labrob::GaitConfiguration desired_gait_configuration_;

  int64_t controller_frequency_;
  int64_t t_msec_ = 0;


  Eigen::Vector3d prev_angular_momentum_ = Eigen::Vector3d::Zero();

  // Log files:
  std::ofstream com_log_file_;
  std::ofstream p_lsole_log_file_;
  std::ofstream p_rsole_log_file_;
  std::ofstream v_lsole_log_file_;
  std::ofstream v_rsole_log_file_;
  std::ofstream p_lsole_des_log_file_;
  std::ofstream p_rsole_des_log_file_;
  std::ofstream v_lsole_des_log_file_;
  std::ofstream v_rsole_des_log_file_;
  std::ofstream angular_momentum_log_file_;
  //std::ofstream fl_log_file_;
  //std::ofstream fr_log_file_;
  std::ofstream cop_computed_log_file_;



  // Logger:
  labrob::Logger logger_;

}; // end class WalkingManager

} // end namespace labrob

#endif // LABROB_WALKING_MANAGER_HPP_