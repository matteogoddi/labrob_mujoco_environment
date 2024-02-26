//
// Created by mmaximo on 20/02/24.
//

#ifndef LABROB_TORQUEWHOLEBODYCONTROLLER_H_
#define LABROB_TORQUEWHOLEBODYCONTROLLER_H_

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

  double gamma;

  double mu;

  double foot_length;
  double foot_width;

  static TorqueWholeBodyControllerParams getDefaultParams();
};

class TorqueWholeBodyController : public WholeBodyController {
 public:
  TorqueWholeBodyController(const TorqueWholeBodyControllerParams &params,
                            const pinocchio::Model &robot_model,
                            const Eigen::VectorXd &q_jnt_reg,
                            double sample_time,
                            std::map<std::string, double> &armatures);

  virtual WBCOutput control(const CoMMotion &desired_com_motion,
                          const FootMotion &desired_left_foot_motion,
                          const FootMotion &desired_right_foot_motion,
                          const Eigen::VectorXd &q,
                          const Eigen::VectorXd &q_dot,
                          bool is_left_support,
                          bool is_right_support) override;

 private:
  TorqueWholeBodyControllerParams params_;

  Eigen::VectorXd M_armature_;

  Eigen::MatrixXd Kp_com_;
  Eigen::MatrixXd Kd_com_;
  Eigen::Matrix3d Kp_torso_orientation_;
  Eigen::Matrix3d Kd_torso_orientation_;
  Eigen::MatrixXd Kp_lsole_;
  Eigen::MatrixXd Kd_lsole_;
  Eigen::MatrixXd Kp_rsole_;
  Eigen::MatrixXd Kd_rsole_;

  double Kp_jnt_;
  double Kd_jnt_;

  int num_joints_;
  int num_contacts_;

  int num_variables_single_;
  int num_equalities_single_;
  int num_inequalities_single_;

  int num_variables_double_;
  int num_equalities_double_;
  int num_inequalities_double_;

  std::unique_ptr<labrob::qpsolvers::QPSolverEigenWrapper<double>> single_support_solver_ptr_;
  std::unique_ptr<labrob::qpsolvers::QPSolverEigenWrapper<double>> double_support_solver_ptr_;

};

}

#endif //LABROB_WHOLEBODYCONTROLLER_H_
