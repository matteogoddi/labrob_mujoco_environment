#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

#include <Eigen/Core>

#include <gamepad.hpp>

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

#include <globals.h>

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace unitree_go::msg::dds_;
using namespace unitree::common;
using namespace unitree::robot::b2;

// ── DDS topics ───────────────────────────────────────────────────────────────
static const std::string HG_CMD_TOPIC  = "rt/lowcmd";
static const std::string HG_IMU_TORSO  = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string GO_STATE_TOPIC = "rt/odommodestate";

// ── Motor mode ───────────────────────────────────────────────────────────────
enum class Mode { PR = 0, AB = 1 };

// ── Data structs ─────────────────────────────────────────────────────────────
struct ImuState {
    Eigen::Vector3d accelerometer = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega        = Eigen::Vector3d::Zero();
    Eigen::Vector4d quaternion        = Eigen::Vector4d::Zero();
    Eigen::Vector3d rpy        = Eigen::Vector3d::Zero();
};

struct MotorState {
    std::array<float, G1_NUM_MOTOR> q  = {};
    std::array<float, G1_NUM_MOTOR> dq = {};
    std::array<float, G1_NUM_MOTOR> tau_est = {};
};

struct MotorCommand {
    std::array<float, G1_NUM_MOTOR> q_target  = {};
    std::array<float, G1_NUM_MOTOR> dq_target = {};
    std::array<float, G1_NUM_MOTOR> kp        = {};
    std::array<float, G1_NUM_MOTOR> kd        = {};
    std::array<float, G1_NUM_MOTOR> tau_ff    = {};
};

struct SportModeState {
    Eigen::Vector3d position = {};
    Eigen::Vector3d velocity = {};
    Eigen::Vector4d quaternion   = Eigen::Vector4d(1, 0, 0, 0);
    Eigen::Vector3d rpy          = Eigen::Vector3d::Zero();
};

// ── Thread-safe single-value buffer ──────────────────────────────────────────
template <typename T>
class DataBuffer {
public:
    void SetData(const T& newData) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        data_ = std::make_shared<T>(newData);
    }
    std::shared_ptr<const T> GetData() {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return data_ ? data_ : nullptr;
    }
    void Clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        data_ = nullptr;
    }
private:
    std::shared_ptr<T> data_;
    std::shared_mutex  mutex_;
};

// ── CRC utility ──────────────────────────────────────────────────────────────
inline uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
    uint32_t xbit = 0, data = 0, CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    for (uint32_t i = 0; i < len; ++i) {
        xbit = 1u << 31;
        data = ptr[i];
        for (uint32_t bits = 0; bits < 32; ++bits) {
            if (CRC32 & 0x80000000) { CRC32 <<= 1; CRC32 ^= dwPolynomial; }
            else CRC32 <<= 1;
            if (data & xbit) CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }
    return CRC32;
}

// ── Low-level gains (body joints, in MuJoCo actuator order: legs, waist, arms) ─
inline constexpr std::array<float, 29> Kp_cl{
    400, 400, 400, 600, 400, 300,      // left leg
    400, 400, 400, 600, 400, 300,      // right leg
    250,  250,  250,                     // waist yaw/roll/pitch
    120, 120, 120, 70,  40, 40, 40,   // left arm
    120, 120, 120, 70,  40, 40, 40    // right arm
};
inline constexpr std::array<float, 29> Kd_cl{
    2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2,
    2, 2, 2,
    2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2
};
inline constexpr std::array<float, 29> Kp_reg{
    400, 400, 400, 600, 400, 300,
    400, 400, 400, 600, 400, 300,
    250,  250,  250,
    120, 120, 120, 20,  40, 40, 40,
    120, 120, 120, 20,  40, 40, 40
};
inline constexpr std::array<float, 29> Kd_reg{
    2, 2, 2, 3, 2, 2,
    2, 2, 2, 3, 2, 2,
    2, 2, 2,
    2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2
};

// ── SDK motor index map (joint name → Unitree SDK motor index 0-28) ──────────
inline const std::map<std::string, int> joint_name_to_index = {
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

// ── Shared state populated by SDK callbacks (defined in RobotInterface.cpp) ──
extern std::mutex    stateMutex;
extern uint8_t       mode_machine_;
extern MotorState    motor_state_data;
extern ImuState      imu_pelvis_data;
extern ImuState      imu_torso_data;
extern SportModeState odometry_data;

extern Gamepad       gamepad_;
extern REMOTE_DATA_RX rx_;

// Timestamp (steady_clock, nanoseconds) of the last valid LowState_ message.
// Used to detect a dead DDS link (e.g. ethernet cable pulled) from the main loop.
extern std::atomic<int64_t> last_lowstate_recv_ns;

// ── SDK callbacks ─────────────────────────────────────────────────────────────
void LowStateHandler(const void* msg);
void imuTorsoHandler(const void* msg);
void SportModeStateHandler(const void* msg);

// ── Motion switcher helpers ───────────────────────────────────────────────────
std::string queryServiceName(const std::string& form, const std::string& name);
int         queryMotionStatus(std::shared_ptr<MotionSwitcherClient> msc);