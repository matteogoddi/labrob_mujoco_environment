//
// Created by mmaximo on 20/02/24.
//

#ifndef LABROB_WHOLE_BODY_CONTROLLER_H_
#define LABROB_WHOLE_BODY_CONTROLLER_H_

#include <array>
#include <map>
#include <memory>
#include <string>

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

#include <GaitConfiguration.hpp>
#include <JointCommand.hpp>
#include <RobotState.hpp>

#include <QpSolver.hpp>

namespace labrob {

struct WholeBodyControllerParams {
  double Kp_motion;
  double Kd_motion;
  double Kp_regulation;
  double Kd_regulation;
  double Kp_orientation;
  double Kd_orientation;
  double Kp_foot;
  double Kd_foot;
  double Kp_wrist;
  double Kd_wrist;


  Eigen::MatrixXd Kp_joint_matrix;
  Eigen::MatrixXd Kd_joint_matrix;

  double weight_q_ddot;
  double weight_com;
  double weight_lsole;
  double weight_rsole;
  double weight_lwrist;
  double weight_rwrist;
  double weight_torso;
  double weight_pelvis;
  double weight_regulation;
  double weight_angular_momentum;
  Eigen::MatrixXd weight_regulation_matrix;

  double cmm_selection_matrix_x;
  double cmm_selection_matrix_y;
  double cmm_selection_matrix_z;

  double gamma;
  double beta;

  double mu;

  double foot_length;
  double foot_width;

  static WholeBodyControllerParams getDefaultParams();
};

class WholeBodyController {
 public:
  WholeBodyController(const WholeBodyControllerParams& params,
                      const pinocchio::Model& robot_model,
                      const Eigen::VectorXd& q_jnt_reg,
                      double sample_time,
                      std::map<std::string, double>& armature);

  labrob::JointCommand
  compute_inverse_dynamics(
      const pinocchio::Model& robot_model,
      const labrob::RobotState& robot_state,
      pinocchio::Data& robot_data,
      const labrob::GaitConfiguration& current,
      const labrob::GaitConfiguration& desired
  );

  const Eigen::VectorXd& get_q_ddot() const { return q_ddot_; }
  const Eigen::VectorXd& get_flr()    const { return flr_; }
  const Eigen::VectorXd& getLeftFootWrench()  const { return left_foot_wrench_; }
  const Eigen::VectorXd& getRightFootWrench() const { return right_foot_wrench_; }

  // INFO
  int  wbc_solver_status()      const { return wbc_solver_ptr_->get_status(); }
  
  

 private:
  pinocchio::Model robot_model_;
  pinocchio::Data  robot_data_;

  pinocchio::FrameIndex lsole_idx_, rsole_idx_, lwrist_idx_, rwrist_idx_, torso_idx_, pelvis_idx_;

  Eigen::MatrixXd J_torso_,     J_pelvis_,     J_lsole_,     J_rsole_,     J_lwrist_,     J_rwrist_;
  Eigen::MatrixXd J_torso_dot_, J_pelvis_dot_, J_lsole_dot_, J_rsole_dot_, J_lwrist_dot_, J_rwrist_dot_;

  Eigen::VectorXd q_jnt_reg_;
  Eigen::VectorXd q_ddot_;
  Eigen::VectorXd flr_;

  double sample_time_;
  WholeBodyControllerParams params_;
  Eigen::VectorXd M_armature_;

  int n_joints_, n_contacts_, n_wbc_variables_, n_wbc_equalities_, n_wbc_inequalities_;

  std::unique_ptr<labrob::QpSolver> wbc_solver_ptr_;
  Eigen::VectorXd left_foot_wrench_, right_foot_wrench_;

  
  // ── pre-allocated buffers — no malloc in the 1kHz hot path ──────────────

  // Constant: computed once in constructor, never change:
  Eigen::MatrixXd err_posture_sel_;      // (6+nj)×(6+nj), lower-right block = I_nj
  Eigen::MatrixXd cmm_sel_;              // 3×6
  Eigen::VectorXd f_force_;             // 3*nc zeros
  Eigen::MatrixXd C_force_block_;        // 4×3 friction cone
  Eigen::VectorXd d_min_force_;          // 4*nc  -10000*ones
  Eigen::VectorXd d_max_force_;          // 4*nc  zeros
  Eigen::VectorXd b_no_contact_;         // 3*nc  zeros
  std::array<Eigen::Vector3d, 4> pcis_;  // foot corner offsets
  Eigen::MatrixXd C_acc_;               // 2*nj×(6+nj), diagonal structure
  Eigen::VectorXd zero_nv_;             // (6+nj) zeros for pinocchio drift calls

  // Per-call: pre-allocated, refilled each solve without malloc:
  Eigen::VectorXd err_posture_, err_posture_vel_, desired_qddot_;
  Eigen::MatrixXd H_acc_;
  Eigen::VectorXd f_acc_;
  Eigen::VectorXd d_min_acc_, d_max_acc_;
  std::array<Eigen::Vector3d, 4> pcis_l_, pcis_r_;
  Eigen::MatrixXd T_l_, T_r_;
  Eigen::MatrixXd M_inertia_;
  Eigen::MatrixXd Mu_, Ma_;
  Eigen::VectorXd cu_, ca_;
  Eigen::MatrixXd Jlu_, Jla_, Jru_, Jra_;
  Eigen::MatrixXd H_wbc_;
  Eigen::VectorXd f_wbc_;
  Eigen::MatrixXd A_acc_wbc_;
  Eigen::VectorXd b_acc_wbc_;
  Eigen::MatrixXd A_no_contact_;
  Eigen::MatrixXd A_dyn_;
  Eigen::MatrixXd A_wbc_;
  Eigen::VectorXd b_wbc_;
  Eigen::MatrixXd C_force_left_, C_force_right_;
  Eigen::MatrixXd C_wbc_;
  Eigen::VectorXd d_min_wbc_, d_max_wbc_;
};

} // namespace labrob

#endif // LABROB_WHOLE_BODY_CONTROLLER_H_