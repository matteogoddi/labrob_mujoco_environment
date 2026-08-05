#ifndef LABROB_DISCRETE_PLIP_DYNAMICS
#define LABROB_DISCRETE_PLIP_DYNAMICS

#include <LIPState.hpp>
#include <cmath>

namespace labrob {

typedef Eigen::Vector3d PLIPControlInput;

class DiscretePLIPDynamics {
 public:
  DiscretePLIPDynamics(double m, double z_c, double timestep_msec, 
                        const Eigen::Vector3d& f_right, const Eigen::Vector3d& f_left, const Eigen::Vector3d& L_dot,
                        const Eigen::Vector3d& p_right, const Eigen::Vector3d& p_left);

  LIPState integrate(
      const LIPState& lip_state,
      const PLIPControlInput& zmp_vel,
     const Eigen::Vector3d& w
  );

  void setEta2();
  double getEta2() const { return eta2_; }
  double getEta() const { return std::sqrt(eta2_); }

  void updateDisturbanceTerm(const LIPState& lip_state,
    const Eigen::Vector3d& f_right, const Eigen::Vector3d& f_left,
    const Eigen::Vector3d& L_dot,
    const Eigen::Vector3d& p_right, const Eigen::Vector3d& p_left);

  Eigen::Vector3d& get_disturbance() { return w_;}

 private:
  Eigen::Vector3d updateState(
      const LIPState& lip_state,
      double zmpDot,
      double w,
      int dim
  );
  
  

  double timestep_;

  // PLIP model params
  double eta2_;                             // eta^2
  double m_;                                // mass
  double z_c_;                              // fixed CoM height
  double g_;                                // gravity constant

  // Disturbance terms
  double f_x_right_;                        // x component of estimated right wrist force
  double f_y_right_;                        // y component of estimated right wrist force
  double f_z_right_;                        // z component of estimated right wrist force
  double f_x_left_;                         // x component of estimated left wrist force
  double f_y_left_;                         // y component of estimated left wrist force
  double f_z_left_;                         // z component of estimated left wrist force

  double f_z_nom_;                          // average vertical force acting on wrists
  
  double L_dot_x_;                          // x component of angular momentum rate
  double L_dot_y_;                          // y component of angular momentum rate


  double p_x_right_;                        // x component of right wrist
  double p_y_right_;                        // y component of right wrist
  double p_z_right_;                        // z component of right wrist
  double p_x_left_;                         // x component of left wrist
  double p_y_left_;                         // y component of left wrist
  double p_z_left_;                         // z component of left wrist

  // Complete disturbance entering the PLIP
  Eigen::Vector3d w_;


  // PLIP model matrices
  Eigen::Matrix3d A_;  // I + T * A_c
  Eigen::Vector3d B_;  // T * B_c = [0, 0, T]
  Eigen::Vector3d E_;  // T * E_c = [0, T, 0]
};

} // namespace labrob

#endif