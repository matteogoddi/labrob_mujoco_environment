#ifndef LABROB_DISCRETE_LIP_DYNAMICS
#define LABROB_DISCRETE_LIP_DYNAMICS

#include <LIPState.hpp>
#include <Eigen/Geometry>
#include <cmath>

namespace labrob {

typedef Eigen::Vector3d LIPControlInput;

// Discrete (forward Euler) LIP dynamics, per axis:
//   s = [p_c, p_c_dot, p_z],  s+ = A s + B z_dot + E w
//
// External perturbations are handled inside the model. With LIP_TYPE::PLIP,
// updateDisturbanceTerm() maps the estimated wrist forces and the angular
// momentum rate into the disturbance w and into eta^2 (through the average
// vertical wrist force); with LIP_TYPE::LIP they are ignored and the nominal
// values w = [0, 0, -g], eta^2 = g / z_c are kept. Users of the model
// (integrate(), ISMPC, WalkingManager) read w and eta^2 from here and never
// handle the perturbations themselves.
class DiscreteLIPDynamics {
 public:
  DiscreteLIPDynamics(double m, double z_c, double timestep_sec, LIP_TYPE type);

  LIPState integrate(
      const LIPState& lip_state,
      const LIPControlInput& zmp_vel
  ) const;

  // Update disturbance w and eta^2 from wrist forces and angular momentum rate.
  // In LIP mode the inputs are ignored and the nominal values are kept.
  void updateDisturbanceTerm(const LIPState& lip_state,
    const Eigen::Vector3d& f_right, const Eigen::Vector3d& f_left,
    const Eigen::Vector3d& L_dot,
    const Eigen::Vector3d& p_right, const Eigen::Vector3d& p_left);

  // Reset w and eta^2 to the nominal LIP values (e.g. to cut off transients)
  void resetDisturbance();

  // Model type getter and setter
  LIP_TYPE get_type() const { return type_; }
  void set_type(LIP_TYPE type);

  // Eta getters
  double getEta2() const { return eta2_; }
  double getEta() const { return std::sqrt(eta2_); }

  const Eigen::Vector3d& get_disturbance() const { return w_; }

 private:
  Eigen::Vector3d updateState(
      const LIPState& lip_state,
      double zmpDot,
      int dim
  ) const;

  void setEta2(double f_z_nom);

  LIP_TYPE type_;

  double timestep_;

  // Model params
  double m_;                                // mass
  double z_c_;                              // fixed CoM height
  double g_;                                // gravity constant
  double eta2_;                             // eta^2

  // Complete disturbance entering the model
  Eigen::Vector3d w_;

  // Model matrices
  Eigen::Matrix3d A_;  // I + T * A_c
  Eigen::Vector3d B_;  // T * B_c = [0, 0, T]
  Eigen::Vector3d E_;  // T * E_c = [0, T, 0]
};

} // namespace labrob

#endif
