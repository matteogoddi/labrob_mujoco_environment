#ifndef LABROB_WALKING_DATA_HPP_
#define LABROB_WALKING_DATA_HPP_

#include <deque>

#include <FootstepPlanElement.hpp>
#include <WalkingState.hpp>
#include <SE3.hpp>

namespace labrob {

class WalkingData {
 public:
  std::deque<labrob::FootstepPlanElement> footstep_plan;
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
      const labrob::SE3& leftFootConfiguration,
      const labrob::SE3& rightFootConfiguration
  );

  /**!
   * Update walking state and footstep plan with current time.
  */
  void updateWalkingState(int64_t t);

  void initializeWalkingData(
    double controller_timestep_msec,
    const labrob::SE3& T_lsole,
    const labrob::SE3& T_rsole
  );

  void addSteps(
    const labrob::SE3& T_lsole,
    const labrob::SE3& T_rsole,
    double step_length_x,
    double step_length_y,
    const double yaw_angle,
    double double_support_duration,
    double single_support_duration
  );

  void swapStanding(
    const labrob::SE3& T_lsole,
    const labrob::SE3& T_rsole
  );

    void startWalkingCoop(
      const labrob::SE3& T_lsole,
      const labrob::SE3& T_rsole,
      labrob::Foot first_swing_foot,
      int64_t T_ds_ms,
      int64_t T_ss_ms,
      double step_height
  );


  void stopWalkingCoop(
    const labrob::SE3& T_lsole,
    const labrob::SE3& T_rsole,
    labrob::Foot support_foot
  );

  void removeSteps();

}; // end class WalkingData

} // end namespace labrob

#endif // LABROB_WALKING_DATA_HPP_