// PLIP discrete dynamics


#include <DiscretePLIPDynamics.hpp>

namespace labrob {

DiscretePLIPDynamics::DiscretePLIPDynamics(double m, double z_c, double timestep_msec,
                        const Eigen::Vector3d& f_right, const Eigen::Vector3d& f_left, const Eigen::Vector3d& L_dot,
                        const Eigen::Vector3d& p_right, const Eigen::Vector3d& p_left)
    : timestep_(timestep_msec), m_(m), z_c_(z_c), g_(9.81)
    
{

  // Init disturbance terms
  f_x_right_ = 0;                        // x component of estimated right wrist force
  f_y_right_ = 0;                        // y component of estimated right wrist force
  f_z_right_ = 0;                        // z component of estimated right wrist force
  f_x_left_ = 0;                         // x component of estimated left wrist force
  f_y_left_ = 0;                         // y component of estimated left wrist force
  f_z_left_ = 0;                         // z component of estimated left wrist force
  L_dot_x_ = 0;                          // x component of angular momentum rate
  L_dot_y_ = 0;                          // y component of angular momentum rate
  p_x_right_ = 0;                        // x component of right wrist
  p_y_right_ = 0;                        // y component of right wrist
  p_z_right_ = 0;                        // z component of right wrist
  p_x_left_ = 0;                         // x component of left wrist
  p_y_left_ = 0;                         // y component of left wrist
  p_z_left_ = 0;                         // z component of left wrist

  
  w_ = Eigen::Vector3d::Zero();          // Member variable for getter

  // Init PLIP model params
  f_z_nom_ = (f_z_right_ + f_z_left_)/2;
  eta2_ = (m_ * g_ - f_z_nom_)/(m_ * z_c_);

  // Init LIP model matrices
  A_ << 1.0, timestep_, 0.0,
        eta2_ * timestep_, 1.0, -eta2_ * timestep_,
        0.0, 0.0,  1.0;
  B_ << 0.0, 0.0, timestep_;
  E_ << 0.0, timestep_, 0.0;


}


// Integration
LIPState
DiscretePLIPDynamics::integrate(
    const LIPState& lip_state,
    const PLIPControlInput& zmp_vel,
    const Eigen::Vector3d& f_right, const Eigen::Vector3d& f_left,
    const Eigen::Vector3d& L_dot,
    const Eigen::Vector3d& p_right, const Eigen::Vector3d& p_left
) {

  // Read disturbance
  Eigen::Vector3d w_xyz = updateDisturbanceTerm(lip_state, f_right, f_left, L_dot, p_right, p_left);

  // Update eta^2 and time-varying matrix A
  setEta2();

  Eigen::Vector3d nextStateX = updateState(lip_state, zmp_vel.x(), w_xyz.x(), 0);
  Eigen::Vector3d nextStateY = updateState(lip_state, zmp_vel.y(), w_xyz.y(), 1);
  Eigen::Vector3d nextStateZ = updateState(lip_state, zmp_vel.z(), w_xyz.z(), 2);

  return LIPState(
      Eigen::Vector3d(nextStateX(0), nextStateY(0), nextStateZ(0)),
      Eigen::Vector3d(nextStateX(1), nextStateY(1), nextStateZ(1)),
      Eigen::Vector3d(nextStateX(2), nextStateY(2), nextStateZ(2))
  );
}

// Single coordinate state dynamics
Eigen::Vector3d
DiscretePLIPDynamics::updateState(
    const LIPState& lip_state,
    double zmpDot,
    double w,
    int dim
) {
  Eigen::Vector3d s(
      lip_state.com_pos_(dim),
      lip_state.com_vel_(dim),
      lip_state.zmp_pos_(dim)
  );

  
  // Thanks to the initializations \dot p^z_c(0)=0, p^z_z(0)=p^z_c(0)-\bar{p}^z_c and \bar{p}^z_c=p^z_(0)-p^z_{sole}(0)
  // the z-coordinate p^z_c is forced to stay equal to \bar{p}^z_c+p^z_{sole}
  // if (dim == 2) return A_ * (s + Eigen::Vector3d(0.0, 0.0, z_c_)) + B_ * zmpDot - Eigen::Vector3d(0.0, 0.0, z_c_);
  
  

  // if (dim == 2) return A_ * s + timestep_ * Eigen::Vector3d(0.0, -g_, 0.0) + B_ * zmpDot;

  return A_ * s + B_ * zmpDot + E_ * w;
}


// Update disturbance term
Eigen::Vector3d
DiscretePLIPDynamics::updateDisturbanceTerm(const LIPState& lip_state,
  const Eigen::Vector3d& f_right, const Eigen::Vector3d& f_left,
  const Eigen::Vector3d& L_dot,
  const Eigen::Vector3d& p_right, const Eigen::Vector3d& p_left)
{

  // Wrist forces
  f_x_right_ = f_right(0);
  f_y_right_ = f_right(1);
  f_z_right_ = f_right(2);

  f_x_left_ = f_left(0);
  f_y_left_ = f_left(1);
  f_z_left_ = f_left(2);


  // Angular momentum rate
  L_dot_x_ = L_dot(0);
  L_dot_y_ = L_dot(1);


  // Wrist xy coordinates
  p_x_right_ = p_right(0);
  p_y_right_ = p_right(1);
  p_z_right_ = p_right(2);
  p_x_left_ = p_left(0);
  p_y_left_ = p_left(1);
  p_z_left_ = p_left(2);

  
  // CoM position from current LIP state
  double p_cx = lip_state.com_pos_(0);
  double p_cy = lip_state.com_pos_(1);
  double p_cz = lip_state.com_pos_(2);


  // Compute nx and ny
  double nx = -(p_z_right_ - p_cz)*f_y_right_ + (p_y_right_ - p_cy)*f_z_right_ - (p_z_left_ - p_cz)*f_y_left_ + (p_y_left_ - p_cy)*f_z_left_;
  double ny = (p_z_right_ - p_cz)*f_x_right_ - (p_x_right_ - p_cx)*f_z_right_ + (p_z_left_ - p_cz)*f_x_left_ - (p_x_left_ - p_cx)*f_z_left_;

  // Compute disturbances
  Eigen::Vector3d w;
  w(0) = (ny - L_dot_y_ + z_c_*(f_x_right_ + f_x_left_))/(m_*z_c_);
  w(1) = (-nx + L_dot_x_ + z_c_*(f_y_right_ + f_y_left_))/(m_*z_c_);
  w(2) = - g_ + (f_z_right_ + f_z_left_)/m_;

  // Update member variable
  w_ = w;

  return w;
}


void
DiscretePLIPDynamics::setEta2() {

  // Average vertical force
  f_z_nom_ = (f_z_right_ + f_z_left_)/2;
  
  // eta^2
  eta2_ = (m_ * g_ - f_z_nom_)/(m_ * z_c_);

  // PLIP dynamic matrix A
  A_ << 1.0, timestep_, 0.0,
        eta2_ * timestep_, 1.0, -eta2_ * timestep_,
        0.0, 0.0, 1.0;
}



} // namespace labrob