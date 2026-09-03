#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

// ── MPC dimensionality switch ─────────────────────────────────────────────────
// Uncomment to use the 2D ISMPC (x/y only, constant CoM height assumed).
// #define ISMPC_USE_2D

// ── Joint configuration switch (g1) ───────────────────────────────────────────
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

// ── Robot selection ───────────────────────────────────────────────────────────
// Single switch to change which robot's paths/link/joint names WholeBodyController,
// WalkingManager, StateEstimator and the main_* entry points use. To add a new
// robot: add an enumerator here, add a `robot_type::kXxx` RobotConfig below
// with every field filled in from that robot's URDF, and extend the
// kRobotConfig selection at the bottom of this section.
enum class RobotId { G1, Gene01 };
constexpr RobotId kRobotId = RobotId::G1;

struct RobotConfig {
    std::string_view urdf_path;
    std::string_view scene_path;   // MuJoCo scene/model XML
    std::string left_foot_link;    // pinocchio frame at the foot sole/contact point
    std::string right_foot_link;
    std::string left_wrist_link;   // pinocchio frame at the last wrist DOF, before the hand
    std::string right_wrist_link;
    std::string torso_link;        // pinocchio frame the arms/head attach to
    std::string pelvis_link;
    double foot_length;             // support-polygon foot size [m], for ZMP/WBC contact constraints
    double foot_width;
    double initial_pelvis_height;   // floating-base z [m] so the feet land exactly on the ground
                                     // given initial_joint_positions below -- robot- (and pose-)
                                     // specific, computed via pinocchio forward kinematics at that
                                     // pose with the pelvis at the origin (see the sole frame's z).
    std::vector<std::string> left_arm_joints;   // shoulder->wrist joint chain, for per-joint logging
    std::vector<std::string> right_arm_joints;
    std::map<std::string, double> initial_joint_positions;  // startup/standing pose, keyed by joint name
};

namespace robot_type {

// g1 (Unitree), robot/g1/g1_description/*.urdf
// initial_joint_positions always lists all 29 joint names, regardless of
// G1_29DOF: even when the loaded model/URDF only has 27 movable joints, the
// real DDS hardware always has 29 motors, and the 2 not present in the model
// (waist_roll/waist_pitch) still need a regulation setpoint (see their
// dedicated handling in main_g1.cpp's send_dds_command). Dropping them here
// would break that lookup -- kNumControlledMotors (below) is what tracks
// G1_29DOF, not this map's size.
inline const RobotConfig G1{
    G1_29DOF ? "../robot/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
             : "../robot/g1/g1_description/g1_29dof_lock_waist_with_hand_rev_1_0.urdf",
    G1_29DOF ? "../robot/g1/g1_mj_description/g1_scene.xml"
             : "../robot/g1/g1_mj_description/g1_scene_lock_waist.xml",
    "left_foot_link",
    "right_foot_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
    "pelvis",
    0.20,
    0.05,
    0.725112,  // deeper-squat pose matching initial_joint_positions below
               // (hip_pitch=-0.44, knee=0.95, ankle_pitch=-0.50)
    {"left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
     "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"},
    {"right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
     "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"},
    {
        {"left_hip_pitch_joint",       -0.44},
        {"left_hip_roll_joint",         0.0},
        {"left_hip_yaw_joint",          0.0},
        {"left_knee_joint",             0.95},
        {"left_ankle_pitch_joint",     -0.50},
        {"left_ankle_roll_joint",       0.0},
        {"right_hip_pitch_joint",      -0.44},
        {"right_hip_roll_joint",        0.0},
        {"right_hip_yaw_joint",         0.0},
        {"right_knee_joint",            0.95},
        {"right_ankle_pitch_joint",    -0.50},
        {"right_ankle_roll_joint",      0.0},
        {"waist_yaw_joint",             0.0},
        {"waist_roll_joint",            0.0},
        {"waist_pitch_joint",           0.0},
        {"left_shoulder_pitch_joint",   0.07},
        {"left_shoulder_roll_joint",    0.35},
        {"left_shoulder_yaw_joint",     0.0},
        {"left_elbow_joint",            1.25},
        {"left_wrist_roll_joint",       0.0},
        {"left_wrist_pitch_joint",      0.0},
        {"left_wrist_yaw_joint",        0.0},
        {"right_shoulder_pitch_joint",  0.07},
        {"right_shoulder_roll_joint",  -0.35},
        {"right_shoulder_yaw_joint",    0.0},
        {"right_elbow_joint",           1.25},
        {"right_wrist_roll_joint",      0.0},
        {"right_wrist_pitch_joint",     0.0},
        {"right_wrist_yaw_joint",       0.0},
    },
};

// gene01, robot/gene01/gene01_description/gene01.urdf
inline const RobotConfig Gene01{
    "../robot/gene01/gene01_description/gene01.urdf",
    "../robot/gene01/gene01_mj_description/gene_scene.xml",
    "l_sole",
    "r_sole",
    "l_wrist_3",
    "r_wrist_3",
    "chest",
    "pelvis",
    0.26,
    0.09,
    1.0,  // computed via pinocchio FK at the initial_joint_positions
               // pose below (pelvis at origin, sole frame z negated), same
               // methodology as the CoM-centering comment above
    {"l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
     "l_elbow", "l_wrist_yaw", "l_wrist_pitch", "l_wrist_roll"},
    {"r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
     "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"},
    // hip_pitch/ankle_pitch = +-0.057 rad keeps the feet flat (hip_pitch +
    // knee + ankle_pitch == 0) while centering the CoM over the support
    // polygon at rest: verified with pinocchio at q=neutral, the all-zero
    // pose had CoM ~4.1cm behind mid-foot (a likely cause of falling
    // backward); this brings that offset to <1mm.
    {
        {"l_hip_pitch",       0.057}, {"l_hip_roll",        0.0}, {"l_hip_yaw",         0.0},
        {"l_knee",            0.0}, {"l_ankle_pitch",     -0.057}, {"l_ankle_roll",      0.0},
        {"r_hip_pitch",       0.057}, {"r_hip_roll",        0.0}, {"r_hip_yaw",         0.0},
        {"r_knee",            0.0}, {"r_ankle_pitch",     -0.057}, {"r_ankle_roll",      0.0},
        {"torso_yaw",         0.0}, /*{"torso_roll",        0.0},*/
        {"l_shoulder_pitch",  0.0}, {"l_shoulder_roll",   0.0}, {"l_shoulder_yaw",    0.0},
        {"l_elbow",           0.0}, /*{"l_wrist_yaw",       0.0}, {"l_wrist_pitch",     0.0}, {"l_wrist_roll", 0.0},*/
        {"r_shoulder_pitch",  0.0}, {"r_shoulder_roll",   0.0}, {"r_shoulder_yaw",    0.0},
        {"r_elbow",           0.0}, /*{"r_wrist_yaw",       0.0}, {"r_wrist_pitch",     0.0}, {"r_wrist_roll", 0.0},*/
    },
};

}  // namespace robot_type

inline const RobotConfig& kRobotConfig =
    (kRobotId == RobotId::G1) ? robot_type::G1 : robot_type::Gene01;

// Number of movable joints in the currently loaded robot MODEL (URDF/MuJoCo
// scene) -- NOT the same as initial_joint_positions.size(), which for G1
// always lists all 29 real motor names regardless of G1_29DOF (see above).
// For G1 this must track G1_29DOF explicitly, since the model literally has
// fewer DOF in the locked-waist URDF variant. Use this instead of
// G1_NUM_MOTOR wherever code is meant to work for any kRobotId/URDF variant,
// not just the real G1 DDS hardware path (which always has 29 physical
// motors no matter which URDF is loaded for kinematics/WBC).
inline const int kNumControlledMotors =
    (kRobotId == RobotId::G1)
        ? (G1_29DOF ? 29 : 27)
        : static_cast<int>(kRobotConfig.initial_joint_positions.size());

extern bool isMPCLoopClosed;
extern bool isObserverActive;
extern bool switchWalkingState;
extern bool switchCoopState;
extern int  ZMP_TYPE;  // 1=force-based, 2=wrench-based, 3=CoM-accel-based

// Gait parameters set from the MuJoCo UI control panel (see MujocoUI.hpp),
// read at the switchWalkingState trigger site in WalkingManager.cpp.
extern double uiStepLengthX;
extern double uiStepLengthY;
extern double uiYawIncrement;
extern double uiDoubleSupportDuration;  // milliseconds
extern double uiSingleSupportDuration;  // milliseconds

extern Eigen::VectorXd measured_joint_velocity;