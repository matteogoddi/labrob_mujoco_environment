#pragma once
#include <array>

namespace labrob {
// Left pitch, roll, yaw and elbow in radians. Right roll and yaw are mirrored.
inline constexpr std::array<double, 4> initial_arm_pose{-0.0473, 0.20, 1.3954, 1.7231};
inline constexpr std::array<double, 4> open_arm_pose{-0.0554, 0.42, 1.4655, 1.9414};
}  // namespace labrob
