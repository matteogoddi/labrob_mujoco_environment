//
// Created by mmaximo on 20/02/24.
//

#ifndef LABROB_TORQUEWHOLEBODYCONTROLLER_H_
#define LABROB_TORQUEWHOLEBODYCONTROLLER_H_

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
#include <hrp4_locomotion/WholeBodyController.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>

namespace labrob {

struct TorqueWholeBodyControllerParams {
  double Kp_motion;
  double Kd_motion;
  double Kp_regulation;
  double Kd_regulation;

  double weight_q_ddot;
  double weight_com;
  double weight_lsole;
  double weight_rsole;
  double weight_torso;
  double weight_regulation;
  double weight_angular_momentum;

  double cmm_selection_matrix_x;
  double cmm_selection_matrix_y;
  double cmm_selection_matrix_z;

  double gamma;

  double mu;

  double foot_length;
  double foot_width;

  static TorqueWholeBodyControllerParams getDefaultParams();
};

class TorqueWholeBodyController : public WholeBodyController {
 public:
  TorqueWholeBodyController(const TorqueWholeBodyControllerParams& params,
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

 private:
  TorqueWholeBodyControllerParams params_;

  Eigen::VectorXd M_armature_;

  int n_joints_;
  int n_contacts_;
  int n_wbc_variables_;
  int n_wbc_equalities_;
  int n_wbc_inequalities_;

  std::unique_ptr<qpsolvers::QPSolverEigenWrapper<double>> wbc_solver_ptr_;

};

}

#endif //LABROB_WHOLEBODYCONTROLLER_H_
