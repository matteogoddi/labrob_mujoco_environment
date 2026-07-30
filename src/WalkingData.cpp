#include <WalkingData.hpp>
#include <globals.h>
#include <utils.hpp>

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

void
WalkingData::initializeWalkingData(
  double controller_timestep_msec,
  const labrob::SE3& T_lsole,
  const labrob::SE3& T_rsole
){
    footstep_plan.clear();
    t0 = 0;
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
  const labrob::SE3& T_rsole,
  const double yaw_angle
){
    double swing_foot_trajectory_height = 0.01;
    double step_length_x = 0.1;
    double step_length_y = 0.0;
    int n_steps = 20;
    double double_support_duration = 2000;
    double single_support_duration = 2000;

    // Decompose the initial stance into a centerline point and each foot's
    // lateral offset from it (expressed in the initial heading frame). As
    // the gait turns, the centerline advances step by step while each
    // foot's lateral offset gets re-rotated by the heading accumulated so
    // far. This alone makes the outer foot trace a longer arc than the
    // inner one (same reasoning as the outer/inner wheel of a turning
    // vehicle) as an emergent geometric consequence, with no separate
    // ad-hoc per-foot step-length correction needed -- and keeps the two
    // feet parallel/symmetric about the centerline at the end of every
    // stride, since both are expressed via the same rotated frame instead
    // of a fixed-world-frame lateral offset that never turns with heading.
    const Eigen::Vector3d center0 = 0.5 * (T_lsole.translation() + T_rsole.translation());
    const Eigen::Vector3d lateral_left0  = T_lsole.translation() - center0;
    const Eigen::Vector3d lateral_right0 = T_rsole.translation() - center0;

    // yaw_angle is the per-STRIDE incremental turn (one left step + one right
    // step = one stride), not a per-single-step turn and not a total/final
    // heading: it accumulates stride after stride, so the whole sequence
    // curves progressively (an arc/spiral) instead of jumping to one fixed
    // final heading. Both feet share the SAME heading at the end of a given
    // stride (they must end up parallel again after every double support,
    // like a real turning gait) — only the stride index j = ceil(k/2)
    // matters for R_at[k], not k itself. R_at[k] is therefore the heading
    // after ceil(k/2) full strides; offset_at[k] is the cumulative
    // centerline offset after k steps, each taken along the heading
    // accumulated so far.
    std::vector<Eigen::Matrix3d> R_at(n_steps + 1);
    std::vector<Eigen::Vector3d> offset_at(n_steps + 1);
    R_at[0] = Eigen::Matrix3d::Identity();
    offset_at[0] = Eigen::Vector3d::Zero();
    for (int k = 1; k <= n_steps; ++k) {
        const int stride = (k + 1) / 2; // ceil(k / 2), integer arithmetic
        R_at[k] = labrob::Rz(stride * yaw_angle);
        offset_at[k] = offset_at[k - 1]
            + R_at[k - 1] * Eigen::Vector3d(step_length_x, step_length_y, 0.0);
    }

    // Foot pose at step k: centerline point plus this foot's lateral offset
    // rotated by the heading reached at that step.
    auto leftFootPoseAt = [&](int k) {
        return labrob::SE3(R_at[k] * T_lsole.rotation(),
                            center0 + offset_at[k] + R_at[k] * lateral_left0);
    };
    auto rightFootPoseAt = [&](int k) {
        return labrob::SE3(R_at[k] * T_rsole.rotation(),
                            center0 + offset_at[k] + R_at[k] * lateral_right0);
    };

    // footstep_plan.push_back(labrob::FootstepPlanElement(
    //     labrob::DoubleSupportConfiguration(
    //         labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
    //         labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
    //         labrob::Foot::RIGHT
    //     ),
    //     0.0,
    //     2000,
    //     labrob::WalkingState::Standing
    // ));

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
                leftFootPoseAt(n + 1),
                rightFootPoseAt(n),
                labrob::Foot::LEFT
            ),
            0.0,
            double_support_duration,
            labrob::WalkingState::DoubleSupport
        ));
        footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                leftFootPoseAt(n + 1),
                rightFootPoseAt(n),
                labrob::Foot::LEFT
            ),
            swing_foot_trajectory_height,
            single_support_duration,
            labrob::WalkingState::SingleSupport
        ));
        footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                leftFootPoseAt(n + 1),
                rightFootPoseAt(n + 2),
                labrob::Foot::RIGHT
            ),
            0.0,
            double_support_duration,
            labrob::WalkingState::DoubleSupport
        ));
        footstep_plan.push_back(labrob::FootstepPlanElement(
            labrob::DoubleSupportConfiguration(
                leftFootPoseAt(n + 1),
                rightFootPoseAt(n + 2),
                labrob::Foot::RIGHT
            ),
            swing_foot_trajectory_height,
            single_support_duration,
            labrob::WalkingState::SingleSupport
        ));
    }
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            leftFootPoseAt(n_steps),
            rightFootPoseAt(n_steps),
            labrob::Foot::RIGHT
        ),
        0.0,
        0,
        labrob::WalkingState::Stopping
    ));
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            leftFootPoseAt(n_steps),
            rightFootPoseAt(n_steps),
            labrob::Foot::RIGHT
        ),
        0.0,
        1000,
        labrob::WalkingState::Standing
    ));
  }

void
WalkingData::swapStanding(
  const labrob::SE3& T_lsole,
  const labrob::SE3& T_rsole
){
    footstep_plan.pop_front();
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
        4000,
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

void
WalkingData::startWalkingCoop(
    const labrob::SE3& T_lsole,
    const labrob::SE3& T_rsole,
    labrob::Foot first_swing_foot,
    int64_t T_ds_ms,
    int64_t T_ss_ms,
    double step_height)
{
    // Breve pausa in Standing prima di partire
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
            labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
            first_swing_foot == labrob::Foot::RIGHT ? labrob::Foot::LEFT
                                                     : labrob::Foot::RIGHT),
        0.0, 500,
        labrob::WalkingState::Standing
    ));

    // Starting: prima DS, entrambi i piedi fermi
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
            labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
            first_swing_foot == labrob::Foot::RIGHT ? labrob::Foot::LEFT
                                                     : labrob::Foot::RIGHT),
        0.0, T_ds_ms,
        labrob::WalkingState::Starting
    ));
}

void
WalkingData::stopWalkingCoop(
    const labrob::SE3& T_lsole,
    const labrob::SE3& T_rsole,
    labrob::Foot support_foot
)
{
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
            labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
            support_foot
        ),
        0.0,
        0,
        labrob::WalkingState::Stopping
    ));
    footstep_plan.push_back(labrob::FootstepPlanElement(
        labrob::DoubleSupportConfiguration(
            labrob::SE3(T_lsole.rotation(), T_lsole.translation()),
            labrob::SE3(T_rsole.rotation(), T_rsole.translation()),
            support_foot
        ),
        0.0,
        10000,
        labrob::WalkingState::Standing
    ));
}
} // end namespace labrob