#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <tuple>
#include <utility>


constexpr std::string_view robotUrdfPath  = "../../labrob_mujoco_environment/robot/g1/g1_description/g1_29dof_rev_1_0.urdf";
constexpr std::string_view robotScenePath = "../../labrob_mujoco_environment/robot/g1/g1_mj_description/stair_steps.xml";

extern bool isWBCLoopClosed;
extern bool isEKFactive;
extern bool isMPCLoopClosed;
extern bool useSim;
extern bool useRobot;
extern bool switchWalkingState;
extern bool oneTimepress;
extern bool loopClosed;

extern double startTimeMPCCL;
extern double startTimeWBCCL;

