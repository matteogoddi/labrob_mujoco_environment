#include <hrp4_locomotion/WalkingData.hpp>
#include <hrp4_locomotion/globals.h>
#include <hrp4_locomotion/utils.hpp>

#include <iostream>

namespace labrob {

const labrob::WalkingState&
WalkingData::getWalkingState() const {
  return footstep_plan.front().getWalkingState();  
}

void
WalkingData::updateFootstepPlanWithCurrentStance(
    const labrob::SE3& leftFootConfiguration,
    const labrob::SE3& rightFootConfiguration
) {
  size_t idx = 0;

  // Update all elements up to the first single support (sharing both feet):
  bool first_single_support_found = false;
  while (idx < footstep_plan.size() && !first_single_support_found) {
    if (footstep_plan[idx].getWalkingState() == labrob::WalkingState::SingleSupport) {
      first_single_support_found = true;
    }
    footstep_plan[idx].getFeetPlacement().setFeetConfiguration(
        leftFootConfiguration,
        rightFootConfiguration
    );
    ++idx;
  }

  // Update all elements up to the second single support (sharing only one foot):
  while (idx < footstep_plan.size()) {
    if (footstep_plan[idx].getWalkingState() != labrob::WalkingState::SingleSupport) {
      if (footstep_plan[idx].getFeetPlacement().getSupportFoot() == labrob::Foot::LEFT) {
        footstep_plan[idx].getFeetPlacement().setLeftFootConfiguration(
            leftFootConfiguration
        );
      } else {
        footstep_plan[idx].getFeetPlacement().setRightFootConfiguration(
            rightFootConfiguration
        );
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
    footstep_plan.front().setWalkingState(labrob::WalkingState::PostureRegulation);
    t0 = t;
  } else if (getWalkingState() == labrob::WalkingState::Standing && footstep_plan.size() == 1) {
    // Update t0 to keep robot in standing position.
    t0 = t;
  }else if (t >= t0 + footstep_plan.front().getDuration()) {
    footstep_plan.pop_front();
    t0 = t;
  } 
}

//&& footstep_plan.size() == 1

void
WalkingData::initializeWalkingData(
  double controller_timestep_msec,
  const labrob::SE3& T_lsole,
  const labrob::SE3& T_rsole
){
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
            labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
            labrob::Foot::RIGHT
        ),
        0.0,
        2000,
        labrob::WalkingState::Init
    ));
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
            labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
            labrob::Foot::RIGHT
        ),
        0.0,
        2000,
        labrob::WalkingState::Standing
    ));
}

void
WalkingData::addSteps(
  const labrob::SE3& T_lsole,
  const labrob::SE3& T_rsole
){
    double swing_foot_trajectory_height = 0.00;//0.05
    double step_length_x = 0.0;//0.1
    double step_length_y = 0.0;
    double step_rotation = 0.0;
    int n_steps = 0;//20

    double double_support_duration = 10;//800
    double single_support_duration = 10;//600
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
            labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
            labrob::Foot::RIGHT
        ),
        0.0,
        double_support_duration,
        labrob::WalkingState::Starting
    ));

    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
            labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
            labrob::Foot::RIGHT
        ),
        swing_foot_trajectory_height,
        single_support_duration,
        labrob::WalkingState::SingleSupport
    ));
    for (int n = 0; n < n_steps; n += 2) {
        footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                labrob::SE3(labrob::Rz((n + 1) * step_rotation) * T_lsole.rotation(), T_lsole.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::SE3(labrob::Rz(n * step_rotation) * T_rsole.rotation(), T_rsole.translation() + n * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::Foot::RIGHT
            ),
            0.0,
            double_support_duration,
            labrob::WalkingState::DoubleSupport
        ));
        footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                labrob::SE3(labrob::Rz((n + 1) * step_rotation) * T_lsole.rotation(), T_lsole.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::SE3(labrob::Rz(n * step_rotation) * T_rsole.rotation(), T_rsole.translation() + n * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::Foot::LEFT
            ),
            swing_foot_trajectory_height,
            single_support_duration,
            labrob::WalkingState::SingleSupport
        ));
        footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                labrob::SE3(labrob::Rz((n + 1) * step_rotation) * T_lsole.rotation(), T_lsole.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::SE3(labrob::Rz((n + 2) * step_rotation) * T_rsole.rotation(), T_rsole.translation() + (n + 2) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::Foot::LEFT
            ),
            0.0,
            double_support_duration,
            labrob::WalkingState::DoubleSupport
        ));
        footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                labrob::SE3(labrob::Rz((n + 1) * step_rotation) * T_lsole.rotation(), T_lsole.translation() + (n + 1) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::SE3(labrob::Rz((n + 2) * step_rotation) * T_rsole.rotation(), T_rsole.translation() + (n + 2) * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
                labrob::Foot::RIGHT
            ),
            swing_foot_trajectory_height,
            single_support_duration,
            labrob::WalkingState::SingleSupport
        ));
    }
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(labrob::Rz(n_steps * step_rotation) * T_lsole.rotation(), T_lsole.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::SE3(labrob::Rz(n_steps * step_rotation) * T_rsole.rotation(), T_rsole.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::Foot::RIGHT
        ),
        0.0,
        0,
        labrob::WalkingState::Stopping
    ));
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(labrob::Rz(n_steps * step_rotation) * T_lsole.rotation(), T_lsole.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::SE3(labrob::Rz(n_steps * step_rotation) * T_rsole.rotation(), T_rsole.translation() + n_steps * Eigen::Vector3d(step_length_x, step_length_y, 0.0)),
            labrob::Foot::RIGHT
        ),
        0.0,
        2000,
        labrob::WalkingState::Standing
    ));
  }

void
WalkingData::removeSteps(){
  // Remove all steps except the first one (current stance) and the last one (standing).
  while (footstep_plan.size() > 2) {
    footstep_plan.erase(footstep_plan.begin() + 1);
  }
}
} // end namespace labrob