#pragma once

#include <array>
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

#include <RobotConfig.hpp>

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
};

struct MotorState {
    std::array<float, G1_NUM_MOTOR> q  = {};
    std::array<float, G1_NUM_MOTOR> dq = {};
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

// ── Shared state populated by SDK callbacks (defined in RobotInterface.cpp) ──
extern std::mutex    stateMutex;
extern uint8_t       mode_machine_;
extern MotorState    motor_state_data;
extern ImuState      imu_pelvis_data;
extern ImuState      imu_torso_data;
extern SportModeState odometry_data;

extern Gamepad       gamepad_;
extern REMOTE_DATA_RX rx_;

// ── SDK callbacks ─────────────────────────────────────────────────────────────
void LowStateHandler(const void* msg);
void imuTorsoHandler(const void* msg);
void SportModeStateHandler(const void* msg);

// ── Motion switcher helpers ───────────────────────────────────────────────────
std::string queryServiceName(const std::string& form, const std::string& name);
int         queryMotionStatus(std::shared_ptr<MotionSwitcherClient> msc);