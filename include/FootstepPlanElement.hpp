#ifndef LABROB_FOOTSTEP_PLAN_ELEMENT_HPP_
#define LABROB_FOOTSTEP_PLAN_ELEMENT_HPP_

#include <vector>

#include <SE3.hpp>
#include <WalkingState.hpp>

namespace labrob {

enum class Foot { LEFT, RIGHT };

class DoubleSupportConfiguration {
 public:
  DoubleSupportConfiguration(
      const labrob::SE3& qLeft,
      const labrob::SE3& qRight,
      const labrob::Foot& support_foot
  ) : qLeft_(qLeft),
      qRight_(qRight),
      support_foot_(support_foot) { }

  const labrob::SE3& getLeftFootConfiguration() const {return qLeft_;}
  const labrob::SE3& getRightFootConfiguration() const {return qRight_;}
  const labrob::SE3& getSupportFootConfiguration() const {
    if (support_foot_ == labrob::Foot::LEFT) return qLeft_;
    else return qRight_;
  }
  const labrob::SE3& getSwingFootConfiguration() const {
    if (support_foot_ == labrob::Foot::LEFT) return qRight_;
    else return qLeft_;
  }
  const labrob::Foot& getSupportFoot() const {return support_foot_;}

  void setFeetConfiguration(const labrob::SE3& qLeft, const labrob::SE3& qRight) {
    qLeft_ = qLeft;
    qRight_ = qRight;
  }
  void setLeftFootConfiguration(const labrob::SE3& qLeft) {qLeft_ = qLeft;}
  void setRightFootConfiguration(const labrob::SE3& qRight) {qRight_ = qRight;}
  void setSupportFootConfiguration(const labrob::SE3& qSupport) {
    if (support_foot_ == labrob::Foot::LEFT) qLeft_ = qSupport;
    else qRight_ = qSupport;
  }
  void setSwingFootConfiguration(const labrob::SE3& qSwing) {
    if (support_foot_ == labrob::Foot::LEFT) qRight_ = qSwing;
    else qLeft_ = qSwing;
  }
  void setSupportFoot(const labrob::Foot& support_foot) {support_foot_ = support_foot;}

 protected:
  labrob::SE3 qLeft_;
  labrob::SE3 qRight_;
  labrob::Foot support_foot_;

}; // end class DoubleSupportConfiguration

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