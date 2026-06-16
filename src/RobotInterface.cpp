#include <RobotInterface.hpp>

#include <cstring>
#include <iostream>

// ── Global definitions ────────────────────────────────────────────────────────
std::mutex     stateMutex;
uint8_t        mode_machine_ = 0;
MotorState     motor_state_data;
ImuState       imu_pelvis_data;
ImuState       imu_torso_data;
SportModeState odometry_data;
Gamepad        gamepad_;
REMOTE_DATA_RX rx_;

// ── SDK callbacks ─────────────────────────────────────────────────────────────
void LowStateHandler(const void* msg) {
    LowState_ low_state = *(const LowState_*)msg;
    uint32_t crc_calc = Crc32Core(
        (uint32_t*)&low_state,
        (sizeof(LowState_) >> 2) - 1
    );
    if (low_state.crc() != crc_calc) {
        std::cerr << "CRC32 mismatch in LowState message!" << std::endl;
        return;
    }

    std::lock_guard<std::mutex> lock(stateMutex);
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        motor_state_data.q[i]  = low_state.motor_state()[i].q();
        motor_state_data.dq[i] = low_state.motor_state()[i].dq();
    }
    imu_pelvis_data.quaternion    = Eigen::Vector4d(
        low_state.imu_state().quaternion()[0], low_state.imu_state().quaternion()[1],
        low_state.imu_state().quaternion()[2], low_state.imu_state().quaternion()[3]);
    imu_pelvis_data.rpy           = Eigen::Vector3d(
        low_state.imu_state().rpy()[0], low_state.imu_state().rpy()[1],
        low_state.imu_state().rpy()[2]);
    imu_pelvis_data.omega         = Eigen::Vector3d(
        low_state.imu_state().gyroscope()[0], low_state.imu_state().gyroscope()[1],
        low_state.imu_state().gyroscope()[2]);
    imu_pelvis_data.accelerometer = Eigen::Vector3d(
        low_state.imu_state().accelerometer()[0], low_state.imu_state().accelerometer()[1],
        low_state.imu_state().accelerometer()[2]);

    memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
    gamepad_.update(rx_.RF_RX);

    if (mode_machine_ != low_state.mode_machine()) {
        if (mode_machine_ == 0)
            std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
        mode_machine_ = low_state.mode_machine();
    }
}

void imuTorsoHandler(const void* msg) {
    auto imu_state = *(const unitree_hg::msg::dds_::IMUState_*)msg;

    std::lock_guard<std::mutex> lock(stateMutex);
    imu_torso_data.quaternion    = Eigen::Vector4d(
        imu_state.quaternion()[0], imu_state.quaternion()[1],
        imu_state.quaternion()[2], imu_state.quaternion()[3]);
    imu_torso_data.rpy           = Eigen::Vector3d(
        imu_state.rpy()[0], imu_state.rpy()[1], imu_state.rpy()[2]);
    imu_torso_data.omega         = Eigen::Vector3d(
        imu_state.gyroscope()[0], imu_state.gyroscope()[1], imu_state.gyroscope()[2]);
    imu_torso_data.accelerometer = Eigen::Vector3d(
        imu_state.accelerometer()[0], imu_state.accelerometer()[1],
        imu_state.accelerometer()[2]);
}

void SportModeStateHandler(const void* msg) {
    auto sportmodestate = *(const SportModeState_*)msg;

    std::lock_guard<std::mutex> lock(stateMutex);
    odometry_data.position[0] = sportmodestate.position()[0];
    odometry_data.position[1] = sportmodestate.position()[1];
    odometry_data.position[2] = sportmodestate.position()[2];
    odometry_data.velocity[0] = sportmodestate.velocity()[0];
    odometry_data.velocity[1] = sportmodestate.velocity()[1];
    odometry_data.velocity[2] = sportmodestate.velocity()[2];

    odometry_data.imu_state.quaternion[0] = sportmodestate.imu_state().quaternion()[0];
    odometry_data.imu_state.quaternion[1] = sportmodestate.imu_state().quaternion()[1];
    odometry_data.imu_state.quaternion[2] = sportmodestate.imu_state().quaternion()[2];
    odometry_data.imu_state.quaternion[3] = sportmodestate.imu_state().quaternion()[3];
    odometry_data.imu_state.omega[0]        = sportmodestate.imu_state().gyroscope()[0];
    odometry_data.imu_state.omega[1]        = sportmodestate.imu_state().gyroscope()[1];
    odometry_data.imu_state.omega[2]        = sportmodestate.imu_state().gyroscope()[2];
    odometry_data.imu_state.accelerometer[0] = sportmodestate.imu_state().accelerometer()[0];
    odometry_data.imu_state.accelerometer[1] = sportmodestate.imu_state().accelerometer()[1];
    odometry_data.imu_state.accelerometer[2] = sportmodestate.imu_state().accelerometer()[2];
    odometry_data.imu_state.rpy[0]          = sportmodestate.imu_state().rpy()[0];
    odometry_data.imu_state.rpy[1]          = sportmodestate.imu_state().rpy()[1];
    odometry_data.imu_state.rpy[2]          = sportmodestate.imu_state().rpy()[2];
}

// ── Motion switcher helpers ───────────────────────────────────────────────────
std::string queryServiceName(const std::string& form, const std::string& name) {
    if (form == "0") {
        if (name == "normal")   return "sport_mode";
        if (name == "ai")       return "ai_sport";
        if (name == "advanced") return "advanced_sport";
    } else {
        if (name == "ai-w")     return "wheeled_sport(go2W)";
        if (name == "normal-w") return "wheeled_sport(b2W)";
    }
    return "";
}

int queryMotionStatus(std::shared_ptr<MotionSwitcherClient> msc) {
    std::string robotForm, motionName;
    int32_t ret = msc->CheckMode(robotForm, motionName);
    if (ret == 0)
        std::cout << "CheckMode succeeded." << std::endl;
    else
        std::cout << "CheckMode failed. Error code: " << ret << std::endl;

    if (motionName.empty()) {
        std::cout << "The motion control-related service is deactivated." << std::endl;
        return 0;
    }
    std::cout << "Service: " << queryServiceName(robotForm, motionName) << " is active" << std::endl;
    return 1;
}