#pragma once

#include <mujoco/mujoco.h>
#include <vector>

namespace labrob {
class MujocoUI;

struct PdGains {
  double kp;
  double kd;
};

// Prepares the simulated body and hands, then holds double support until key 5.
bool prepareSimulation(mjModel& m, mjData& d, MujocoUI& ui,
                       const std::vector<PdGains>& motor_gains);
}  // namespace labrob
