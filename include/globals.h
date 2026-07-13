#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

static constexpr int    G1_NUM_MOTOR     = 29;
static constexpr int    G1_CONTROLLER_HZ = 500;
static constexpr double G1_CONTROLLER_DT = 1.0 / G1_CONTROLLER_HZ;

constexpr std::string_view robotUrdfPath  = "../robot/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf";
constexpr std::string_view robotScenePath = "../robot/g1/g1_mj_description/stair_steps.xml";

inline const std::map<std::string, double> joint_initial_positions = {
    {"left_hip_pitch_joint",       -0.44},
    {"left_hip_roll_joint",         0.04},
    {"left_hip_yaw_joint",          0.0},
    {"left_knee_joint",             0.95},
    {"left_ankle_pitch_joint",     -0.50},
    {"left_ankle_roll_joint",       0.0},
    {"right_hip_pitch_joint",      -0.44},
    {"right_hip_roll_joint",       -0.04},
    {"right_hip_yaw_joint",         0.0},
    {"right_knee_joint",            0.95},
    {"right_ankle_pitch_joint",    -0.50},
    {"right_ankle_roll_joint",      0.0},
    {"waist_yaw_joint",             0.0},
    {"waist_roll_joint",            0.0},
    {"waist_pitch_joint",           0.0},
    {"left_shoulder_pitch_joint",   0.07},
    {"left_shoulder_roll_joint",    0.25},
    {"left_shoulder_yaw_joint",     0.0},
    {"left_elbow_joint",            1.13},
    {"left_wrist_roll_joint",       0.0},
    {"left_wrist_pitch_joint",      0.0},
    {"left_wrist_yaw_joint",        0.0},
    {"right_shoulder_pitch_joint",  0.07},
    {"right_shoulder_roll_joint",  -0.25},
    {"right_shoulder_yaw_joint",    0.0},
    {"right_elbow_joint",           1.13},
    {"right_wrist_roll_joint",      0.0},
    {"right_wrist_pitch_joint",     0.0},
    {"right_wrist_yaw_joint",       0.0},
};

extern bool isMPCLoopClosed;
extern bool isObserverActive;
extern bool switchWalkingState;
extern bool switchCoopState;

extern Eigen::VectorXd measured_joint_velocity;