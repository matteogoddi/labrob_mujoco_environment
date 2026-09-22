#ifndef LABROB_LIP_STATE_HPP_
#define LABROB_LIP_STATE_HPP_

#include <Eigen/Core>

namespace labrob {

class LIPState {
 public:
  LIPState() = default;
  LIPState(
      const Eigen::Vector3d& com_pos,
      const Eigen::Vector3d& com_vel,
      const Eigen::Vector3d& zmp_pos
  ) : com_pos_(com_pos),
    com_vel_(com_vel),
    zmp_pos_(zmp_pos) {}

  LIPState operator+(const LIPState& other) const {
    return LIPState(
        com_pos_ + other.com_pos_,
        com_vel_ + other.com_vel_,
        zmp_pos_ + other.zmp_pos_
    );
  }

  LIPState operator-(const LIPState& other) const {
    return LIPState(
        com_pos_ - other.com_pos_,
        com_vel_ - other.com_vel_,
        zmp_pos_ - other.zmp_pos_
    );
  }

  LIPState operator*(double scalar) const {
    return LIPState(
        com_pos_ * scalar,
        com_vel_ * scalar,
        zmp_pos_ * scalar
    );
  }

  Eigen::Vector3d com_pos_;
  Eigen::Vector3d com_vel_;
  Eigen::Vector3d zmp_pos_;
}; // end class LIPState

inline LIPState operator*(double scalar, const LIPState& state) {
  return state * scalar;
}


} // end namespace labrob

#endif // LABROB_LIP_STATE_HPP_