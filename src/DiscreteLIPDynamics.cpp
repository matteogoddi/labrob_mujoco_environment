// LIP discrete dynamics


#include <DiscreteLIPDynamics.hpp>

namespace labrob {

DiscreteLIPDynamics::DiscreteLIPDynamics(double m, double z_c, double timestep_sec, LIP_TYPE type)
    : type_(type), timestep_(timestep_sec), m_(m), z_c_(z_c), g_(9.81)
{
  B_ << 0.0, 0.0, timestep_;
  E_ << 0.0, timestep_, 0.0;

  // Start from the nominal LIP (no external forces): sets w_, eta2_ and A_
  resetDisturbance();
}


// Integration
LIPState
DiscreteLIPDynamics::integrate(
    const LIPState& lip_state,
    const LIPControlInput& zmp_vel
) const {
  // Integrate each component
  Eigen::Vector3d nextStateX = updateState(lip_state, zmp_vel.x(), 0);
  Eigen::Vector3d nextStateY = updateState(lip_state, zmp_vel.y(), 1);
  Eigen::Vector3d nextStateZ = updateState(lip_state, zmp_vel.z(), 2);

  return LIPState(
      Eigen::Vector3d(nextStateX(0), nextStateY(0), nextStateZ(0)),
      Eigen::Vector3d(nextStateX(1), nextStateY(1), nextStateZ(1)),
      Eigen::Vector3d(nextStateX(2), nextStateY(2), nextStateZ(2))
  );
}


// Single coordinate state dynamics: s+ = A s + B z_dot + E w
Eigen::Vector3d
DiscreteLIPDynamics::updateState(
    const LIPState& lip_state,
    double zmpDot,
    int dim
) const {
  Eigen::Vector3d s(
      lip_state.com_pos_(dim),
      lip_state.com_vel_(dim),
      lip_state.zmp_pos_(dim)
  );

  // In LIP mode w_ = [0, 0, -g], so gravity still enters the z component
  return A_ * s + B_ * zmpDot + E_ * w_(dim);
}


// Update disturbance term
void
DiscreteLIPDynamics::updateDisturbanceTerm(const LIPState& lip_state,
  const Eigen::Vector3d& f_right, const Eigen::Vector3d& f_left,
  const Eigen::Vector3d& L_dot,
  const Eigen::Vector3d& p_right, const Eigen::Vector3d& p_left)
{
  // Standard LIP: external forces are not modelled
  if (type_ == LIP_TYPE::LIP) return;

  // Wrist positions relative to the CoM
  const Eigen::Vector3d r_right = p_right - lip_state.com_pos_;
  const Eigen::Vector3d r_left  = p_left  - lip_state.com_pos_;

  // Moment of wrist forces about the CoM: n = sum r x f
  const Eigen::Vector3d n = r_right.cross(f_right) + r_left.cross(f_left);
  const Eigen::Vector3d f_sum = f_right + f_left;

  // Disturbances
  w_(0) = ( n.y() - L_dot.y() + z_c_ * f_sum.x()) / (m_ * z_c_);
  w_(1) = (-n.x() + L_dot.x() + z_c_ * f_sum.y()) / (m_ * z_c_);
  w_(2) = -g_ + f_sum.z() / m_;

  // eta^2 from average vertical wrist force
  setEta2(0.5 * f_sum.z());
}


void
DiscreteLIPDynamics::resetDisturbance() {
  w_ << 0.0, 0.0, -g_;
  setEta2(0.0);
}


void
DiscreteLIPDynamics::set_type(LIP_TYPE type) {
  type_ = type;
  if (type_ == LIP_TYPE::LIP) resetDisturbance();
}


void
DiscreteLIPDynamics::setEta2(double f_z_nom) {

  // eta^2
  eta2_ = (m_ * g_ - f_z_nom) / (m_ * z_c_);

  // Dynamic matrix A
  A_ << 1.0, timestep_, 0.0,
        eta2_ * timestep_, 1.0, -eta2_ * timestep_,
        0.0, 0.0, 1.0;
}



} // namespace labrob
