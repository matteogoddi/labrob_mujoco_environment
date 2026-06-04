#ifndef LABROB_JISMPC_HPP_
#define LABROB_JISMPC_HPP_

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <hrp4_locomotion/GaitConfiguration.hpp>
#include <hrp4_locomotion/RobotState.hpp>
#include <hrp4_locomotion/WalkingData.hpp>

#include <labrob_qpsolvers/qpsolvers.hpp>

namespace labrob {

// JIS-MPC: Joint-level Intrinsically Stable MPC.
// Optimizes joint accelerations nu_dot over a horizon C, with:
//   - Hard ZMP-box inequality constraints
//   - Hard DCM terminal stability equality constraint
//   - Hard zero-velocity contact equality constraints
//   - Soft cost: swing-foot tracking + CoM height + regularization
class JISMPC {
 public:
  JISMPC(
      const pinocchio::Model& model,
      int horizon,                   // C (number of MPC steps)
      double dt,                     // MPC timestep [s]
      double eta,                    // sqrt(g / h_com)
      double robot_mass,
      double foot_constraint_x,      // ZMP box half-length [m]
      double foot_constraint_y,      // ZMP box half-width  [m]
      int64_t jismpc_timestep_msec
  );

  // Run one MPC iteration.
  // current: current robot state (FK already computed externally).
  // walking_data: labrob footstep plan (used for ZMP refs and contact flags).
  // t_msec: current controller time in milliseconds.
  void solve(
      const labrob::RobotState& robot_state,
      const labrob::GaitConfiguration& current,
      const labrob::WalkingData& walking_data,
      int64_t t_msec
  );

  // First-step joint acceleration (nv-dimensional, including floating base).
  const Eigen::VectorXd& get_nu_dot_0() const { return nu_dot_0_; }

  bool get_left_in_contact()  const { return l_contact_[0]; }
  bool get_right_in_contact() const { return r_contact_[0]; }

 private:
  // ZMP reference at a given millisecond time from WalkingData.
  Eigen::Vector2d zmpReference(const labrob::WalkingData& wd, int64_t t_ms) const;

  pinocchio::Model model_;
  pinocchio::Data  data_;

  int    C_;     // horizon
  double dt_;    // MPC timestep [s]
  double eta_;   // sqrt(g/h)
  double robot_mass_;
  double foot_x_; // ZMP box half-length
  double foot_y_; // ZMP box half-width
  int64_t jismpc_timestep_msec_;

  pinocchio::FrameIndex lsole_idx_;
  pinocchio::FrameIndex rsole_idx_;

  int nv_;   // model_.nv
  int njnt_; // model_.nv - 6
  int n_var_;  // C * nv
  int n_eq_;   // 2 (DCM) + 12*C (contact)
  int n_ineq_; // 4*C (ZMP box)

  // Auxiliary trajectory (warm-started from previous solution)
  std::vector<Eigen::VectorXd> q_aux_;    // [C+1] configuration
  std::vector<Eigen::VectorXd> qdot_aux_; // [C+1] velocity
  std::vector<Eigen::VectorXd> nu_dot_;   // [C]   controls

  // Kinematics along auxiliary trajectory
  struct StepData {
    Eigen::MatrixXd J_com;        // 3 x nv
    Eigen::MatrixXd Ag;           // 6 x nv (centroidal momentum matrix)
    Eigen::MatrixXd J_lsole;      // 6 x nv
    Eigen::MatrixXd J_rsole;      // 6 x nv
    Eigen::VectorXd Jdot_qdot_lsole; // 6
    Eigen::VectorXd Jdot_qdot_rsole; // 6
    Eigen::Vector3d com_pos;
    Eigen::Vector3d com_vel;
    Eigen::Vector3d Jdot_qdot_com; // 3  (drift of com velocity)
    bool l_contact;
    bool r_contact;
  };
  std::vector<StepData> steps_; // [C]

  std::vector<bool> l_contact_; // [C]
  std::vector<bool> r_contact_; // [C]

  // QP matrices
  Eigen::MatrixXd H_;
  Eigen::VectorXd f_;
  Eigen::MatrixXd A_eq_;
  Eigen::VectorXd b_eq_;
  Eigen::MatrixXd A_ineq_;
  Eigen::VectorXd lg_;
  Eigen::VectorXd ug_;

  // Solution
  Eigen::VectorXd nu_dot_0_;

  std::shared_ptr<labrob::qpsolvers::QPSolverEigenWrapper<double>> qp_solver_ptr_;
};

} // namespace labrob

#endif // LABROB_JISMPC_HPP_