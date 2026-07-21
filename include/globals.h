#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

// ── MPC dimensionality switch ─────────────────────────────────────────────────
// Uncomment to use the 2D ISMPC (x/y only, constant CoM height assumed).
#define ISMPC_USE_2D

// ── Joint configuration switch ────────────────────────────────────────────────
// true  = 29 DOF (waist yaw + roll + pitch active)
// false = 27 DOF (waist roll and pitch locked)
static constexpr bool G1_29DOF = true;

// Returns false for SDK motor indices that are physically absent in 27-DOF mode
// (waist_roll=13, waist_pitch=14 are locked and must not be commanded).
inline constexpr bool g1_motor_active(int sdk_idx) {
    if constexpr (G1_29DOF) return true;
    return sdk_idx != 13 && sdk_idx != 14;
}

static constexpr int    G1_NUM_MOTOR     = 29;  // physical DDS motor slots, always 29
static constexpr int    G1_CONTROLLER_HZ = 500;
static constexpr double G1_CONTROLLER_DT = 1.0 / G1_CONTROLLER_HZ;

constexpr std::string_view robotUrdfPath = G1_29DOF
    ? "../robot/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
    : "../robot/g1/g1_description/g1_29dof_lock_waist_with_hand_rev_1_0.urdf";

constexpr std::string_view robotScenePath = G1_29DOF
    ? "../robot/g1/g1_mj_description/stair_steps.xml"
    : "../robot/g1/g1_mj_description/stair_steps_lock_waist.xml";

// Deeper-squat standing pose (kept for reference / easy revert):
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

// More upright standing pose: same hip/ankle-pitch ≈ -knee/2 ratio (keeps the
// foot flat and torso vertical) as above, just with a much shallower knee bend.
// inline const std::map<std::string, double> joint_initial_positions = {
//     {"left_hip_pitch_joint",       -0.15},
//     {"left_hip_roll_joint",         0.04},
//     {"left_hip_yaw_joint",          0.0},
//     {"left_knee_joint",             0.30},
//     {"left_ankle_pitch_joint",     -0.15},
//     {"left_ankle_roll_joint",       0.0},
//     {"right_hip_pitch_joint",      -0.15},
//     {"right_hip_roll_joint",       -0.04},
//     {"right_hip_yaw_joint",         0.0},
//     {"right_knee_joint",            0.30},
//     {"right_ankle_pitch_joint",    -0.15},
//     {"right_ankle_roll_joint",      0.0},
//     {"waist_yaw_joint",             0.0},
//     {"waist_roll_joint",            0.0},
//     {"waist_pitch_joint",           0.0},
//     {"left_shoulder_pitch_joint",   0.07},
//     {"left_shoulder_roll_joint",    0.25},
//     {"left_shoulder_yaw_joint",     0.0},
//     {"left_elbow_joint",            1.13},
//     {"left_wrist_roll_joint",       0.0},
//     {"left_wrist_pitch_joint",      0.0},
//     {"left_wrist_yaw_joint",        0.0},
//     {"right_shoulder_pitch_joint",  0.07},
//     {"right_shoulder_roll_joint",  -0.25},
//     {"right_shoulder_yaw_joint",    0.0},
//     {"right_elbow_joint",           1.13},
//     {"right_wrist_roll_joint",      0.0},
//     {"right_wrist_pitch_joint",     0.0},
//     {"right_wrist_yaw_joint",       0.0},
// };

extern bool isMPCLoopClosed;
extern bool isObserverActive;
extern bool switchWalkingState;
extern bool switchCoopState;
extern int  ZMP_TYPE;  // 1=force-based, 2=wrench-based, 3=CoM-accel-based

extern Eigen::VectorXd measured_joint_velocity;