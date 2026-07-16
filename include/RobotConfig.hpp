#pragma once

#include <array>
#include <map>
#include <string>

static constexpr int    G1_NUM_MOTOR          = 29;
static constexpr int    G1_CONTROLLER_HZ      = 500;
static constexpr double G1_CONTROLLER_DT      = 1.0 / G1_CONTROLLER_HZ;

const std::array<float, G1_NUM_MOTOR> Kp_cl{
    0, 0, 0, 0, 0, 0,      // legs
    0, 0, 0, 0, 0, 0,      // legs
    0, 0, 0,                     // waist
    0, 0, 0, 0, 0, 0, 0,   // arms
    0, 0, 0, 0, 0, 0, 0    // arms
};

const std::array<float, G1_NUM_MOTOR> Kd_cl{
    8, 8, 8, 10, 8, 8,
    8, 8, 8, 10, 8, 8,
    2, 2, 2,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0
};

const std::array<float, G1_NUM_MOTOR> Kp_reg{
    400, 400, 400, 600, 400, 300,      // legs
    400, 400, 400, 600, 400, 300,      // legs
    250, 250, 150,                     // waist
    120, 120, 120, 70,  40, 40, 40,   // arms
    120, 120, 120, 70,  40, 40, 40    // arms
};

const std::array<float, G1_NUM_MOTOR> Kd_reg{
    2, 2, 2, 3, 2, 2,
    2, 2, 2, 3, 2, 2,
    2, 2, 2,
    2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2
};

// Pinocchio joint index (motor order from Unitree SDK)
const std::map<std::string, int> joint_name_to_index = {
    {"left_hip_pitch_joint",       0},
    {"left_hip_roll_joint",        1},
    {"left_hip_yaw_joint",         2},
    {"left_knee_joint",            3},
    {"left_ankle_pitch_joint",     4},
    {"left_ankle_roll_joint",      5},
    {"right_hip_pitch_joint",      6},
    {"right_hip_roll_joint",       7},
    {"right_hip_yaw_joint",        8},
    {"right_knee_joint",           9},
    {"right_ankle_pitch_joint",   10},
    {"right_ankle_roll_joint",    11},
    {"waist_yaw_joint",           12},
    {"waist_roll_joint",          13},
    {"waist_pitch_joint",         14},
    {"left_shoulder_pitch_joint", 15},
    {"left_shoulder_roll_joint",  16},
    {"left_shoulder_yaw_joint",   17},
    {"left_elbow_joint",          18},
    {"left_wrist_roll_joint",     19},
    {"left_wrist_pitch_joint",    20},
    {"left_wrist_yaw_joint",      21},
    {"right_shoulder_pitch_joint",22},
    {"right_shoulder_roll_joint", 23},
    {"right_shoulder_yaw_joint",  24},
    {"right_elbow_joint",         25},
    {"right_wrist_roll_joint",    26},
    {"right_wrist_pitch_joint",   27},
    {"right_wrist_yaw_joint",     28},
};

// Initial joint configuration (radians)
const std::map<std::string, double> joint_initial_positions = {
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