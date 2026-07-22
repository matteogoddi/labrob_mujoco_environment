#ifndef LABROB_DCM_FEEDBACK_CONTROLLER_HPP_
#define LABROB_DCM_FEEDBACK_CONTROLLER_HPP_

#include <Eigen/Core>

namespace labrob {

// DCM (Divergent Component of Motion) feedback stabilizer, in the style of
// Kajita et al. 2010 / Englsberger et al. 2015 / Caron et al. 2021: corrects
// the commanded ZMP (x,y only — z is handled separately by the constant-height
// LIPM assumption) based on the error between the nominal (pattern/MPC) DCM
// and the estimated DCM, using a proportional + leaky-integral (anti-windup,
// no plain integrator) + ZMP-error derivative-like term:
//
//   z_cmd = z_nom - (1 + kp/eta)*(xi_nom - xi_meas)
//                 - (ki/eta)*leaky_integral(xi_nom - xi_meas)
//                 + (kz/eta)*(z_nom - z_meas)
//
// The leaky integrator follows dI/dt = e - I/Ti (forward Euler), so unlike a
// plain integrator it cannot wind up on a persistent bias.
class DcmFeedbackController {
 public:
  struct Params {
    // NOTE: even at kp=ki=kz=0 the underlying DCM feedback law still applies
    // a baseline unity correction (the "1 +" term below is structural, not a
    // tunable gain — see Kajita/Caron derivation). So kp=ki=kz=0 is NOT a
    // no-op. `enabled=false` is the actual safe-rollout bypass: it returns
    // zmp_nominal unchanged and leaves the leaky integrator at rest.
    bool enabled = false;
    double kp = 0.0;   // proportional gain on DCM error
    double ki = 0.0;   // leaky-integral gain on DCM error
    double kz = 0.0;   // proportional gain on ZMP error
    double Ti = 20.0;  // leaky integrator time constant [s]

    // Disabled: zero behavior change until explicitly turned on (safe rollout).
    // NOTE: kp can't default to a fraction of eta here — eta is a per-instance
    // construction argument of DcmFeedbackController, not available in this
    // nested struct's own default-member-initializers. Set kp relative to eta
    // at the call site instead (see WalkingManager::init()).
    static Params getDefaultParams() { return Params{}; }
  };

  DcmFeedbackController(double eta, const Params& params)
      : eta_(eta), params_(params) {}

  // dcm_nominal/dcm_measured/zmp_nominal/zmp_measured are 2D (x,y), in meters.
  // Updates the internal leaky-integrator state and returns the corrected ZMP.
  Eigen::Vector2d computeCorrectedZmp(
      const Eigen::Vector2d& dcm_nominal,
      const Eigen::Vector2d& dcm_measured,
      const Eigen::Vector2d& zmp_nominal,
      const Eigen::Vector2d& zmp_measured,
      double dt) {
    const Eigen::Vector2d dcm_error = dcm_nominal - dcm_measured;

    if (!params_.enabled) {
      // True bypass: no correction, integrator held at rest (no windup while
      // disabled).
      last_dcm_error_ = dcm_error;
      last_zmp_cmd_ = zmp_nominal;
      return zmp_nominal;
    }

    integral_state_ += dt * (dcm_error - integral_state_ / params_.Ti);

    const Eigen::Vector2d zmp_error = zmp_nominal - zmp_measured;
    const Eigen::Vector2d zmp_cmd =
        zmp_nominal
        - (1.0 + params_.kp / eta_) * dcm_error
        - (params_.ki / eta_) * integral_state_
        + (params_.kz / eta_) * zmp_error;

    last_dcm_error_ = dcm_error;
    last_zmp_cmd_ = zmp_cmd;
    return zmp_cmd;
  }

  // Zeroes the leaky-integrator state (call on Init/PostureRegulation/Standing
  // transitions, where there is no LIP walking dynamics to stabilize).
  void reset() { integral_state_.setZero(); }

  void setEta(double eta) { eta_ = eta; }
  double getEta() const { return eta_; }

  void setGains(const Params& params) { params_ = params; }
  const Params& getGains() const { return params_; }

  const Eigen::Vector2d& getLastDcmError() const { return last_dcm_error_; }
  const Eigen::Vector2d& getIntegralState() const { return integral_state_; }
  const Eigen::Vector2d& getLastZmpCommand() const { return last_zmp_cmd_; }

 private:
  double eta_;
  Params params_;
  Eigen::Vector2d integral_state_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d last_dcm_error_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d last_zmp_cmd_ = Eigen::Vector2d::Zero();
};

} // namespace labrob

#endif // LABROB_DCM_FEEDBACK_CONTROLLER_HPP_
