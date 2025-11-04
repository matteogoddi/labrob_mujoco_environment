#ifndef LABROB_WALKING_DATA_HPP_
#define LABROB_WALKING_DATA_HPP_

#include <vector>
#include <Eigen/Core>

#include <hrp4_locomotion/Footstep.hpp>
#include <hrp4_locomotion/WalkingState.hpp>

namespace labrob {

class WalkingData {
 public:
  std::vector<labrob::Footstep> footstep_plan;
  int64_t t0; // Starting time of footstep_plan.front()

  /**!
   * Returns walking state of first element of the footstep plan, which
   * corresponds to current walking state.
   */
  const labrob::WalkingState& getWalkingState() const;

  /**!
   * Update footstep plan with current stance to make it consistent with
   * sensor data.
  */
  void updateFootstepPlanWithCurrentStance(
      const Eigen::Matrix3d& left_foot_rotation,
      const Eigen::Vector3d& left_foot_position,
      const Eigen::Matrix3d& right_foot_rotation,
      const Eigen::Vector3d& right_foot_position
  );

  /**!
   * Update walking state and footstep plan with current time.
  */
  void updateWalkingState(int64_t t);

}; // end class WalkingData

} // end namespace labrob

#endif // LABROB_WALKING_DATA_HPP_