#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <tuple>
#include <utility>


constexpr std::string_view robotUrdfPath  = "../../labrob_mujoco_environment/robot/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf";
//constexpr std::string_view robotUrdfPath  = "../../labrob_mujoco_environment/robot/g1/g1_description/g1_29dof_dex3.urdf";
constexpr std::string_view robotScenePath = "../../labrob_mujoco_environment/robot/g1/g1_mj_description/stair_steps.xml";

extern bool isWBCLoopClosed;
extern bool isEKFactive;
extern bool isMPCLoopClosed;
extern bool isObserverActive;
extern bool useSim;
extern bool useRobot;
extern bool switchWalkingState;
extern bool switchCoopState;
extern bool loopClosed;

extern bool forward;
extern bool lateral;
extern bool curve;

extern Eigen::VectorXd measured_joint_velocity;


