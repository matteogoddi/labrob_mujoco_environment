//
// Created by mmaximo on 20/02/24.
//

#ifndef LABROB_WHOLE_BODY_CONTROLLER_H_
#define LABROB_WHOLE_BODY_CONTROLLER_H_

#include <array>
#include <vector>

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
#include <hrp4_locomotion/RobotState.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>

namespace labrob {

struct WholeBodyControllerParams {
  double Kp_motion;
  double Kd_motion;
  Eigen::Vector3d Kp_torso_motion;
  Eigen::Vector3d Kd_torso_motion;
  Eigen::Matrix<double, 6, 1> Kp_hand_compliance;
  Eigen::Matrix<double, 6, 1> Kd_hand_compliance;
  // Cartesian components used by the residual-arm task. The force
  // experiments select translation, consistently with the CRG admittance and
  // allocation selectors.
  Eigen::Matrix<double, 6, 6> S_hand_compliance;
  double Kp_regulation;
  double Kd_regulation;

  double weight_q_ddot;
  double weight_com;
  double weight_lsole;
  double weight_rsole;
  double weight_torso;
  double weight_lhand_compliance;
  double weight_rhand_compliance;
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
      const labrob::RobotState& fb_filt_robot_state,
      pinocchio::Data& robot_data,
      pinocchio::Data& fb_robot_data,
      const labrob::GaitConfiguration& current,
      const labrob::GaitConfiguration& desired,
      const Eigen::Matrix<double, 6, 1>& left_interaction_wrench,
      const Eigen::Matrix<double, 6, 1>& right_interaction_wrench
  );

  Eigen::VectorXd get_q_ddot() const;
  Eigen::VectorXd get_flr() const;
  const Eigen::VectorXd& getLeftFootWrench() const { return left_foot_wrench_; }
  const Eigen::VectorXd& getRightFootWrench() const { return right_foot_wrench_; }
  const Eigen::Matrix<double, 6, 1>&
  getLeftHandComplianceAccelerationReference() const {
    return left_hand_compliance_acceleration_reference_;
  }
  const Eigen::Matrix<double, 6, 1>&
  getRightHandComplianceAccelerationReference() const {
    return right_hand_compliance_acceleration_reference_;
  }
  const Eigen::Matrix<double, 6, 1>&
  getLeftHandComplianceAccelerationAchieved() const {
    return left_hand_compliance_acceleration_achieved_;
  }
  const Eigen::Matrix<double, 6, 1>&
  getRightHandComplianceAccelerationAchieved() const {
    return right_hand_compliance_acceleration_achieved_;
  }
  const Eigen::Vector3d& getLeftHandCompliancePositionReference() const {
    return left_hand_compliance_position_reference_;
  }
  const Eigen::Vector3d& getRightHandCompliancePositionReference() const {
    return right_hand_compliance_position_reference_;
  }
  const Eigen::Vector3d& getLeftHandCompliancePositionAchieved() const {
    return left_hand_compliance_position_achieved_;
  }
  const Eigen::Vector3d& getRightHandCompliancePositionAchieved() const {
    return right_hand_compliance_position_achieved_;
  }
  int getSolverStatus() const { return solver_status_; }
  double getEqualityResidualInfinityNorm() const {
    return equality_residual_infinity_norm_;
  }
  double getInequalityViolationInfinityNorm() const {
    return inequality_violation_infinity_norm_;
  }
  const Eigen::MatrixXd& getLeftFootUnderactuatedJacobian() const;
  const Eigen::MatrixXd& getRightFootUnderactuatedJacobian() const;

 private:
  pinocchio::Model robot_model_;
  pinocchio::Data robot_data_;
  pinocchio::Data hand_nominal_data_;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;
  pinocchio::FrameIndex torso_idx_;
  pinocchio::FrameIndex lhand_idx_;
  pinocchio::FrameIndex rhand_idx_;

  Eigen::MatrixXd J_torso_;
  Eigen::MatrixXd J_lsole_;
  Eigen::MatrixXd J_rsole_;
  Eigen::MatrixXd J_lhand_;
  Eigen::MatrixXd J_rhand_;

  Eigen::MatrixXd J_torso_dot_;
  Eigen::MatrixXd J_lsole_dot_;
  Eigen::MatrixXd J_rsole_dot_;
  Eigen::MatrixXd J_lhand_dot_;
  Eigen::MatrixXd J_rhand_dot_;

  std::vector<int> left_arm_velocity_indices_;
  std::vector<int> right_arm_velocity_indices_;
  std::vector<int> left_arm_configuration_indices_;
  std::vector<int> right_arm_configuration_indices_;

  Eigen::VectorXd q_jnt_reg_;

  Eigen::VectorXd q_ddot_;
  Eigen::VectorXd flr;

  double sample_time_;

  WholeBodyControllerParams params_;

  Eigen::VectorXd M_armature_;

  int n_joints_;
  int n_contacts_;
  int n_wbc_variables_;
  int n_wbc_inequalities_;
  int n_slack_;

  // One fixed-dimension HPIPM instance per support count. This keeps every
  // equality row meaningful: double support has 18 rows, single support 24,
  // and flight 30.
  std::array<
      std::unique_ptr<qpsolvers::QPSolverEigenWrapper<double>>, 3>
      wbc_solver_by_support_count_;

  Eigen::VectorXd left_foot_wrench_;
  Eigen::VectorXd right_foot_wrench_;
  Eigen::Matrix<double, 6, 1> left_hand_compliance_acceleration_reference_ =
      Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> right_hand_compliance_acceleration_reference_ =
      Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> left_hand_compliance_acceleration_achieved_ =
      Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> right_hand_compliance_acceleration_achieved_ =
      Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Vector3d left_hand_compliance_position_reference_ =
      Eigen::Vector3d::Zero();
  Eigen::Vector3d right_hand_compliance_position_reference_ =
      Eigen::Vector3d::Zero();
  Eigen::Vector3d left_hand_compliance_position_achieved_ =
      Eigen::Vector3d::Zero();
  Eigen::Vector3d right_hand_compliance_position_achieved_ =
      Eigen::Vector3d::Zero();
  int solver_status_ = -1;
  double equality_residual_infinity_norm_ = 0.0;
  double inequality_violation_infinity_norm_ = 0.0;
  std::size_t solver_failure_count_ = 0;
  Eigen::MatrixXd Jlu_;
  Eigen::MatrixXd Jru_;

};

}

#endif //LABROB_WHOLE_BODY_CONTROLLER_H_
