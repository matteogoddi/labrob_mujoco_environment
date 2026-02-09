#ifndef LABROB_FOOTSTEP_PLAN_ELEMENT_HPP_
#define LABROB_FOOTSTEP_PLAN_ELEMENT_HPP_

#include <vector>

#include <hrp4_locomotion/DoubleSupportConfiguration.hpp>
#include <hrp4_locomotion/WalkingState.hpp>

namespace labrob {

class FootstepPlanElement {
 public:
  FootstepPlanElement(
      const labrob::DoubleSupportConfiguration& feet_placement,
      double swing_foot_trajectory_height,
      int64_t duration,
      const labrob::WalkingState& walking_state
  ) : feet_placement_(feet_placement),
    walking_state_(walking_state),
    swing_foot_trajectory_height_(swing_foot_trajectory_height),
    duration_(duration)
     { }
  
  labrob::DoubleSupportConfiguration& getFeetPlacement() {return feet_placement_;}
  const labrob::DoubleSupportConfiguration& getFeetPlacement() const {return feet_placement_;}

  double getSwingFootTrajectoryHeight() const {return swing_foot_trajectory_height_;}

  int64_t getDuration() const {return duration_;}

  const labrob::WalkingState& getWalkingState() const {return walking_state_;}

  void
  setFeetPlacement(const labrob::DoubleSupportConfiguration& feet_placement) {feet_placement_ = feet_placement;}

  void setSwingFootTrajectoryHeight(double swing_foot_trajectory_height) {swing_foot_trajectory_height_ = swing_foot_trajectory_height;}

  void setDuration(int64_t duration) {duration_ = duration;}

  void setWalkingState(const labrob::WalkingState& walking_state) {walking_state_ = walking_state;}


 protected:
  labrob::DoubleSupportConfiguration feet_placement_;
  labrob::WalkingState walking_state_;
  double swing_foot_trajectory_height_;
  int64_t duration_;
}; // end class FootstepPlanElement

} // end namespace labrob

#endif // LABROB_FOOTSTEP_PLAN_ELEMENT_HPP_