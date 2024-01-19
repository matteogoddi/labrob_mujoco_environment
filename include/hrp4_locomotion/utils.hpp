#ifndef LABROB_UTILS_HPP_
#define LABROB_UTILS_HPP_

// STL
#include <cmath>
#include <deque>

// Eigen
#include <Eigen/Core>

// hrp4_locomotion
#include <hrp4_locomotion/FootstepPlanElement.hpp>
#include <hrp4_locomotion/WalkingState.hpp>

namespace labrob {

template <class T>
Eigen::Matrix<T, 3, 3>
Rz(T theta) {
  T c = std::cos(theta);
  T s = std::sin(theta);
  Eigen::Matrix<T, 3, 3> R;
  R <<   c,   -s, 0.0,
        s,    c, 0.0,
      0.0,  0.0, 1.0;
  return R;
}

template <class T>
Eigen::Matrix<T, 2, 2>
Rz_planar(T theta) {
  T c = std::cos(theta);
  T s = std::sin(theta);
  Eigen::Matrix<T, 2, 2> R;
  R << c, -s,
      s,  c;
  return R;
}

template <class T>
T
wrap_angle(T alpha) {
  return std::atan2(std::sin(alpha), std::cos(alpha));
}

template <class T>
T
angle_difference(T alpha, T beta) {
  return wrap_angle(alpha - beta);
}

labrob::WalkingState
walkingStateFromString(
    const std::string& walking_state_str
);

bool readArgosFootstepPlan(
    const std::string& file_path,
    std::deque<labrob::FootstepPlanElement>& footstep_plan
);

bool readHumanoids2023FootstepPlan(
    const std::string& file_path,
    const labrob::DoubleSupportConfiguration& current_feet_placement,
    const labrob::WalkingState& current_walking_state,
    std::deque<labrob::FootstepPlanElement>& footstep_plan
);

bool readFootstepPlan(
    const std::string& file_path,
    std::deque<labrob::FootstepPlanElement>& footstep_plan
);

void saveFootstepPlan(
    const std::deque<labrob::FootstepPlanElement>& footstep_plan,
    const std::string& file_path
);

} // end namespace labrob

#endif // LABROB_UTILS_HPP_