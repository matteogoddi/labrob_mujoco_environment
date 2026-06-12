#ifndef LABROB_JISMPC_WHOLE_BODY_CONTROLLER_HPP_
#define LABROB_JISMPC_WHOLE_BODY_CONTROLLER_HPP_

#include <array>
#include <map>
#include <memory>
#include <string>

#include <Eigen/Core>

#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/spatial/skew.hpp>

#include <GaitConfiguration.hpp>
#include <JointCommand.hpp>
#include <RobotState.hpp>

#include <QpSolver.hpp>

namespace labrob {

struct JISMPCWholeBodyControllerParams {
  double weight_q_ddot = 1e-9;
  double gamma         = 50.0;  // contact velocity damping
  double mu            = 0.5;   // friction coefficient
  double foot_length   = 0.20;  // [m] (G1 foot)
  double foot_width    = 0.07;  // [m] (G1 foot)

  static JISMPCWholeBodyControllerParams getDefaultParams();
};

// Simplified whole-body controller ported from jl2.
// Variables: [q̈ (nv); f_l (3*4); f_r (3*4)]
// Equalities: contact foot zero-acc + no-contact zero-force + base dynamics
// Inequalities: joint vel/pos limits + friction cones
class JISMPCWholeBodyController {
 public:
  JISMPCWholeBodyController(
      const JISMPCWholeBodyControllerParams& params,
      const pinocchio::Model& robot_model,
      const Eigen::VectorXd& q_jnt_reg,
      double sample_time,
      std::map<std::string, double>& armatures
  );

  // desired.qjntddot and/or a full desired nu_dot can be provided.
  // nu_dot_des: nv-dimensional desired joint acceleration from JIS-MPC.
  labrob::JointCommand compute_inverse_dynamics(
      const pinocchio::Model& robot_model,
      const labrob::RobotState& robot_state,
      pinocchio::Data& robot_data,
      const labrob::GaitConfiguration& current,
      const labrob::GaitConfiguration& desired,
      const Eigen::VectorXd& nu_dot_des  // nv-dim from JIS-MPC (or zeros)
  );

 private:
  pinocchio::Model robot_model_;
  pinocchio::Data  robot_data_;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;

  Eigen::MatrixXd J_lsole_;
  Eigen::MatrixXd J_rsole_;
  Eigen::MatrixXd J_lsole_dot_;
  Eigen::MatrixXd J_rsole_dot_;

  Eigen::VectorXd q_jnt_reg_;
  double sample_time_;
  JISMPCWholeBodyControllerParams params_;
  Eigen::VectorXd M_armature_;

  Eigen::VectorXd q_;
  Eigen::VectorXd qdot_;
  Eigen::VectorXd zero_nv_;

  Eigen::MatrixXd M_;
  Eigen::VectorXd c_;

  Eigen::MatrixXd H_acc_;
  Eigen::VectorXd f_acc_;
  Eigen::MatrixXd C_acc_;
  Eigen::VectorXd d_min_acc_;
  Eigen::VectorXd d_max_acc_;

  static constexpr int kFootContactsPerFoot = 4;
  std::array<Eigen::Vector3d, kFootContactsPerFoot> pcis_;
  std::array<Eigen::Vector3d, kFootContactsPerFoot> pcis_l_;
  std::array<Eigen::Vector3d, kFootContactsPerFoot> pcis_r_;

  Eigen::MatrixXd T_l_;
  Eigen::MatrixXd T_r_;

  Eigen::MatrixXd H_force_one_;
  Eigen::VectorXd f_force_one_;
  Eigen::VectorXd b_dyn_;
  Eigen::Matrix<double, 4, 3> C_force_block_;
  Eigen::VectorXd d_min_force_one_;
  Eigen::VectorXd d_max_force_one_;

  Eigen::MatrixXd H_;
  Eigen::VectorXd f_;

  Eigen::MatrixXd A_acc_;
  Eigen::VectorXd b_acc_;
  Eigen::MatrixXd A_no_contact_;
  Eigen::VectorXd b_no_contact_;
  Eigen::MatrixXd A_dyn_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;

  Eigen::MatrixXd C_force_left_;
  Eigen::MatrixXd C_force_right_;
  Eigen::MatrixXd C_;
  Eigen::VectorXd d_min_;
  Eigen::VectorXd d_max_;

  Eigen::VectorXd tau_;

  int n_joints_;
  int n_contacts_;
  int n_wbc_variables_;
  int n_wbc_equalities_;
  int n_wbc_inequalities_;

  std::unique_ptr<labrob::QpSolver> wbc_solver_ptr_;
};

} // namespace labrob

#endif // LABROB_JISMPC_WHOLE_BODY_CONTROLLER_HPP_