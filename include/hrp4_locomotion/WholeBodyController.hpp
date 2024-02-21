//
// Created by mmaximo on 19/02/24.
//

#ifndef LABROB_WHOLE_BODY_CONTROLLER_HPP_
#define LABROB_WHOLE_BODY_CONTROLLER_HPP_

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <hrp4_locomotion/RobotState.hpp>

namespace labrob {

struct CoMMotion {
  Eigen::Vector3d position;
  Eigen::Vector3d velocity;
  Eigen::Vector3d acceleration;
};

struct FootMotion {
  pinocchio::SE3 pose;
  Eigen::VectorXd velocity;
  Eigen::VectorXd acceleration;
};

class WholeBodyController {
 public:
  WholeBodyController(const pinocchio::Model &robot_model,
                      const Eigen::VectorXd &q_jnt_reg,
                      double sample_time);

  virtual Eigen::VectorXd control(const CoMMotion &desired_com_motion,
                          const FootMotion &desired_left_foot_motion,
                          const FootMotion &desired_right_foot_motion,
                          const RobotState &robot_state,
                          bool is_left_support,
                          bool is_right_support
                          ) = 0;

 protected:

  void computePinocchio(const Eigen::VectorXd &q, const Eigen::VectorXd &q_dot);

  void getJacobians(const Eigen::VectorXd &q, const Eigen::VectorXd &q_dot, Eigen::MatrixXd &J_torso, Eigen::MatrixXd &J_lsole, Eigen::MatrixXd &J_rsole);

  void getJacobiansTimeVariation(const Eigen::VectorXd &q, const Eigen::VectorXd &q_dot, Eigen::MatrixXd &J_torso_dot, Eigen::MatrixXd &J_lsole_dot, Eigen::MatrixXd &J_rsole_dot);

  Eigen::Matrix<double, 6, 1> err_frameplacement(const pinocchio::SE3& Ta, const pinocchio::SE3& Tb);

  Eigen::Vector3d err_translation(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb);

  Eigen::Vector3d err_rotation(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb);

  pinocchio::Model robot_model_;
  pinocchio::Data robot_data_;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;
  pinocchio::FrameIndex torso_idx_;

  Eigen::VectorXd q_jnt_reg_;

  double sample_time_;
};

}

#endif //LABROB_WHOLE_BODY_CONTROLLER_HOPP_
