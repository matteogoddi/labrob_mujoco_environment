#include <hrp4_locomotion/WalkingData.hpp>

#include <iostream>

namespace labrob {

const labrob::WalkingState&
WalkingData::getWalkingState() const {
  return footstep_plan.front().walking_state;
}

void
WalkingData::updateFootstepPlanWithCurrentStance(
    const Eigen::Matrix3d& left_foot_rotation,
    const Eigen::Vector3d& left_foot_position,
    const Eigen::Matrix3d& right_foot_rotation,
    const Eigen::Vector3d& right_foot_position
) {
  size_t idx = 0;

  // Update all elements up to the first single support (sharing both feet):
  bool first_single_support_found = false;
  while (idx < footstep_plan.size() && !first_single_support_found) {
    if (footstep_plan[idx].walking_state == labrob::WalkingState::SingleSupport) {
      first_single_support_found = true;
    }
    footstep_plan[idx].left_foot_rotation = left_foot_rotation;
    footstep_plan[idx].left_foot_position = left_foot_position;
    footstep_plan[idx].right_foot_rotation = right_foot_rotation;
    footstep_plan[idx].right_foot_position = right_foot_position;
    ++idx;
  }

  // Update all elements up to the second single support (sharing only one foot):
  while (idx < footstep_plan.size()) {
    if (footstep_plan[idx].walking_state != labrob::WalkingState::SingleSupport) {
      if (footstep_plan[idx].support_foot == labrob::Foot::LEFT) {
        footstep_plan[idx].left_foot_rotation = left_foot_rotation;
        footstep_plan[idx].left_foot_position = left_foot_position;
      } else {
        footstep_plan[idx].right_foot_rotation = right_foot_rotation;
        footstep_plan[idx].right_foot_position = right_foot_position;
      }
    } else {
      break;
    }
    ++idx;
  }

}

void
WalkingData::updateWalkingState(int64_t t) {
  if (getWalkingState() == labrob::WalkingState::Init) {
    footstep_plan.front().walking_state = labrob::WalkingState::PostureRegulation;
    t0 = t;
  } else if (getWalkingState() == labrob::WalkingState::Standing && footstep_plan.size() == 1) {
    // Update t0 to keep robot in standing position.
    t0 = t;
  }
  else if (t >= t0 + footstep_plan.front().duration_ms) {
    footstep_plan.erase(footstep_plan.begin());
    t0 = t;
  }
}

} // end namespace labrob