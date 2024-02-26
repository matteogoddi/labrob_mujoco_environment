//
// Created by mmaximo on 20/02/24.
//

#include <hrp4_locomotion/WholeBodyController.hpp>

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

namespace labrob {

WholeBodyController::WholeBodyController(const pinocchio::Model &robot_model,
                                         const Eigen::VectorXd &q_jnt_reg,
                                         double sample_time)
  : robot_model_(robot_model), q_jnt_reg_(q_jnt_reg), sample_time_(sample_time) {
  robot_data_ = pinocchio::Data(robot_model_);

  lsole_idx_ = robot_model_.getFrameId("L_ANKLE_P_S");
  rsole_idx_ = robot_model_.getFrameId("R_ANKLE_P_S");
  torso_idx_ = robot_model_.getFrameId("base_link");
}

void WholeBodyController::computePinocchio(const Eigen::VectorXd &q, const Eigen::VectorXd &q_dot) {
  // Perform forward kinematics on the whole tree and update robot data:
  pinocchio::forwardKinematics(robot_model_, robot_data_, q);

  // NOTE: jacobianCenterOfMass calls forwardKinematics and
  //       computeJointJacobians.
  pinocchio::jacobianCenterOfMass(robot_model_, robot_data_, q);
  pinocchio::computeJointJacobiansTimeVariation(robot_model_, robot_data_, q, q_dot);
  pinocchio::framesForwardKinematics(robot_model_, robot_data_, q);
  pinocchio::centerOfMass(robot_model_, robot_data_, q, q_dot, 0.0 * q_dot); // This is used to compute the CoM drift (J_com_dot * qdot)
}

void WholeBodyController::getJacobians(Eigen::MatrixXd &J_torso, Eigen::MatrixXd &J_lsole, Eigen::MatrixXd &J_rsole) {
  pinocchio::getFrameJacobian(
      robot_model_,
      robot_data_,
      torso_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_torso
  );
  pinocchio::getFrameJacobian(
      robot_model_,
      robot_data_,
      lsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_lsole
  );
  pinocchio::getFrameJacobian(
      robot_model_,
      robot_data_,
      rsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_rsole
  );
}

void WholeBodyController::getJacobiansTimeVariation(Eigen::MatrixXd &J_torso_dot, Eigen::MatrixXd &J_lsole_dot, Eigen::MatrixXd &J_rsole_dot) {
  pinocchio::getFrameJacobianTimeVariation(
      robot_model_,
      robot_data_,
      torso_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_torso_dot
  );
  pinocchio::getFrameJacobianTimeVariation(
      robot_model_,
      robot_data_,
      lsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_lsole_dot
  );
  pinocchio::getFrameJacobianTimeVariation(
      robot_model_,
      robot_data_,
      rsole_idx_,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_rsole_dot
  );
}

Eigen::Matrix<double, 6, 1> WholeBodyController::err_frameplacement(const pinocchio::SE3& Ta, const pinocchio::SE3& Tb) {
  // TODO: how do you use pinocchio::log6?
  Eigen::Matrix<double, 6, 1> err;
  err << err_translation(Ta.translation(), Tb.translation()),
      err_rotation(Ta.rotation(), Tb.rotation());
  return err;
}

Eigen::Vector3d WholeBodyController::err_translation(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb) {
  return pa - pb;
}

Eigen::Vector3d WholeBodyController::err_rotation(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb) {
  // TODO: how do you use pinocchio::log3?
  Eigen::Matrix3d Rdiff = Rb.transpose() * Ra;
  auto aa = Eigen::AngleAxisd(Rdiff);
  return aa.angle() * Ra * aa.axis();
}

}