#ifndef LABROB_WALKING_STATE_HPP_
#define LABROB_WALKING_STATE_HPP_

#include <string>

namespace labrob {
/*! State of the robot regarding walking phases. */
enum class WalkingState {
  Init, /*!< The robot is in initial posture. */
  PostureRegulation, /*!< The robot is performing a posture regulation to go to Standing state. */
  Standing, /*!< The robot is standing, no footstep plan to execute. */
  Starting, /*!< A footstep plan is available, moving equilibrium from static to dynamic. */
  SingleSupport, /*!< Single support phase, keep ZMP within support polygon. */
  DoubleSupport, /*!< Double support phase, keep ZMP within support polygon. */
  Stopping, /*!< Similar to double support phase, but executing last element of footstep plan. */
  AbortStarting, /*!< Abort starting phase in order to go back to Standing without swing motion. */
  AbortWalking /*! Abort walking phase in order to go back to Standing without swing motion. */
};

inline std::string to_string(const WalkingState& walking_state) {
  std::string walking_state_str;
  if (walking_state == WalkingState::Init) {
    walking_state_str = "Init";
  } else if (walking_state == WalkingState::PostureRegulation) {
    walking_state_str = "PostureRegulation";
  } else if (walking_state == WalkingState::Standing) {
    walking_state_str = "Standing";
  } else if (walking_state == WalkingState::Starting) {
    walking_state_str = "Starting";
  } else if (walking_state == WalkingState::SingleSupport) {
    walking_state_str = "SingleSupport";
  } else if (walking_state == WalkingState::DoubleSupport) {
    walking_state_str = "DoubleSupport";
  } else if (walking_state == WalkingState::Stopping) {
    walking_state_str = "Stopping";
  } else if (walking_state == WalkingState::AbortStarting) {
    walking_state_str = "AbortStarting";
  } else if (walking_state == WalkingState::AbortWalking) {
    walking_state_str = "AbortWalking";
  } else {
    // NOTE: execution should never get here.
    walking_state_str = "Unknown";
  }
  return walking_state_str;
}

} // end namespace labrob

#endif // LABROB_WALKING_STATE_HPP_