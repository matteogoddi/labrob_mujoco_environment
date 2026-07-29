#ifndef LABROB_COM_COMPLIANCE_CONTROLLER_HPP_
#define LABROB_COM_COMPLIANCE_CONTROLLER_HPP_

#include <Eigen/Core>

namespace labrob {

// CoM position compliance stabilizer, in the style of Nagasaka, Inaba, Inoue
// (1999), "Stabilization of Dynamic Walk on a Humanoid Using Torso Position
// Compliance Control", RSJ'99 -- ported here to act on the CoM instead of the
// torso, since our feet are rigid (no ankle compliance) and the CoM task is
// already an unconstrained soft task in the WBC QP, unlike the feet (which
// carry the contact constraints).
//
// The CoM is treated as if connected to the measured ZMP through a virtual
// spring-damper: the ZMP tracking error (measured - desired) drives a leaky
// integrator whose *state* is the CoM position correction itself:
//
//   e        = zmp_measured - zmp_desired
//   ddelta/dt = Kp_c*e + Kd_c*de/dt - delta/T_leak
//   delta    += ddelta/dt * dt
//
// The leakage term (-delta/T_leak) is what turns this into a compliance
// (bounded, spring-like) response instead of a plain integrator: a constant
// ZMP error settles to a bounded delta = Kp_c*T_leak*e instead of growing
// without limit, and the correction decays back to zero once the error
// disappears. This is deliberately a *position*-level correction (not an
// acceleration one): closing the loop at the acceleration level runs into
// the ZMP/CoM-acceleration algebraic coupling (zmp = p - a/eta^2), which
// makes naive proportional feedback on acceleration delicate to stabilize;
// correcting the position reference avoids that entirely.
class ComComplianceController {
 public:
  struct Params {
    bool enabled = false;
    double Kp_c = 0.0;     // proportional gain on ZMP error [1/s]
    double Kd_c = 0.0;     // derivative gain on ZMP error [dimensionless]
    double T_leak = 1.0;   // leakage time constant [s]

    static Params getDefaultParams() { return Params{}; }
  };

  explicit ComComplianceController(const Params& params) : params_(params) {}

  // zmp_measured/zmp_desired are 2D (x,y), in meters. Updates the internal
  // compliance state and returns the CoM position correction (x,y) to add
  // to the nominal CoM position reference.
  Eigen::Vector2d update(
      const Eigen::Vector2d& zmp_measured,
      const Eigen::Vector2d& zmp_desired,
      double dt) {
    const Eigen::Vector2d e = zmp_measured - zmp_desired;
    const Eigen::Vector2d e_dot = (e - e_prev_) / dt;

    delta_com_dot_ = params_.Kp_c * e + params_.Kd_c * e_dot
                     - delta_com_ / params_.T_leak;
    delta_com_ += delta_com_dot_ * dt;
    e_prev_ = e;
    last_zmp_error_ = e;

    return params_.enabled ? delta_com_ : Eigen::Vector2d::Zero();
  }

  // Rate of change of the position correction: optional feedforward for the
  // CoM velocity task, if position-only compliance proves too stiff/laggy.
  const Eigen::Vector2d& getVelocityCorrection() const { return delta_com_dot_; }

  // Resets the compliance state (call on Init/PostureRegulation/Standing
  // transitions, where there is no LIP walking dynamics to stabilize).
  void reset() {
    delta_com_.setZero();
    delta_com_dot_.setZero();
    e_prev_.setZero();
  }

  void setGains(const Params& params) { params_ = params; }
  const Params& getGains() const { return params_; }

  const Eigen::Vector2d& getLastZmpError() const { return last_zmp_error_; }
  const Eigen::Vector2d& getDeltaCom() const { return delta_com_; }

 private:
  Params params_;
  Eigen::Vector2d delta_com_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d delta_com_dot_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d e_prev_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d last_zmp_error_ = Eigen::Vector2d::Zero();
};

} // namespace labrob

#endif // LABROB_COM_COMPLIANCE_CONTROLLER_HPP_
