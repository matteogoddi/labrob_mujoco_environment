#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include <chrono>
#include <csignal>

#include <Eigen/Core>

#include <hrp4_locomotion/gamepad.hpp>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>


#include <hrp4_locomotion/StateEstimator.hpp>
#include <hrp4_locomotion/WalkingManager.hpp>

// Path to the URDF used solely to build the pinocchio model for the state
// estimator, relative to the build/ directory where g1_controller is
// expected to run from.
static const std::string kUrdfPath = "../g1_description/unitreeg1.urdf";

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string GO_STATE_TOPIC = "rt/odommodestate";

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace unitree_go::msg::dds_;



enum ControlMode {
  LISTEN,
  REGULATION,
  WBC
};
static std::atomic<ControlMode> ctrl_mode{REGULATION};
static std::atomic<bool> g_running{true};

void signalHandler(int signum)
{
    std::cout << "\n[INFO] SIGINT received → stopping...\n";

    g_running = false;
    ctrl_mode = LISTEN;
}




template <typename T>
class DataBuffer {
 public:
  void SetData(const T &newData) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = std::make_shared<T>(newData);
  }

  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data ? data : nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = nullptr;
  }

 private:
  std::shared_ptr<T> data;
  std::shared_mutex mutex;
};

const int G1_NUM_MOTOR = 29;
struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
  std::array<float, 3> accelerometer = {};
};
struct SportModeState {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  Eigen::Vector4d quaternion = Eigen::Vector4d(1, 0, 0, 0);
};
struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};
struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
};



// REGULATION KP (stiff position hold; same values as main_g1.cpp's Kp_reg)
std::array<float, G1_NUM_MOTOR> Kp_reg{
    400, 400, 400, 600, 400, 300,      // legs
    400, 400, 400, 600, 400, 300,      // legs
    250, 250, 150,                     // waist
    120, 120, 120, 70,  40, 40, 40,    // arms
    120, 120, 120, 70,  40, 40, 40     // arms
};

// REGULATION KD
std::array<float, G1_NUM_MOTOR> Kd_reg{
    2, 2, 2, 3, 2, 2,     // legs
    2, 2, 2, 3, 2, 2,     // legs
    2, 2, 2,              // waist
    2, 2, 2, 2, 2, 2, 2,  // arms
    2, 2, 2, 2, 2, 2, 2   // arms
};



// WBC KP (low stiffness: the WBC's torque solution does most of the work,
// same values as main_g1.cpp's Kp_cl)
std::array<float, G1_NUM_MOTOR> Kp_wbc{
    120, 120, 120, 120, 120, 120,     // legs
    120, 120, 120, 120, 120, 120,     // legs
    60, 60, 60,                       // waist
    80, 80, 80, 80,  40, 40, 40,      // arms
    80, 80, 80, 80,  40, 40, 40       // arms
};

// WBC KD (mostly damping, same values as main_g1.cpp's Kd_cl)
std::array<float, G1_NUM_MOTOR> Kd_wbc{
    20, 20, 20, 25, 20, 20,     // legs
    20, 20, 20, 25, 20, 20,     // legs
    15, 15, 15,                 // waist
    10, 10, 10, 10, 2, 2, 2,    // arms
    10, 10, 10, 10, 2, 2, 2     // arms
};

enum class Mode {
  PR = 0,  // Series Control for Ptich/Roll Joints
  AB = 1   // Parallel Control for A/B Joints
};

enum G1JointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13,        // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistA = 13,           // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistPitch = 14,       // NOTE INVALID for g1 23dof/29dof with waist locked
  WaistB = 14,           // NOTE INVALID for g1 23dof/29dof with waist locked
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,   // NOTE INVALID for g1 23dof
  LeftWristYaw = 21,     // NOTE INVALID for g1 23dof
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,  // NOTE INVALID for g1 23dof
  RightWristYaw = 28     // NOTE INVALID for g1 23dof
};

// URDF/pinocchio joint names in G1JointIndex order. No MuJoCo model is
// available in this environment to derive these at runtime (unlike
// main_g1.cpp), so they are hardcoded here instead.
static const std::array<std::string, G1_NUM_MOTOR> kJointNames{
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
};

inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else
        CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;

      xbit >>= 1;
    }
  }
  return CRC32;
};

// Minimal named-channel logger for raw sensor feedback (joint_pos, odom_pos,
// ...). Kept local to this file: labrob::Logger in this branch is a
// different, fixed-schema WBC logger already used internally by
// WalkingManager, not a generic log(name, value)/save(dir) API.
class SensorLogger {
 public:
  template <typename Derived>
  void log(const std::string& name, const Eigen::MatrixBase<Derived>& v) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_[name].emplace_back(v);
  }

  void save(const std::string& dir) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [name, entries] : data_) {
      if (entries.empty()) continue;
      std::ofstream f(dir + "/" + name + ".txt");
      for (const auto& v : entries) f << v.transpose() << "\n";
    }
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::vector<Eigen::VectorXd>> data_;
};

class G1 {
 private:
  int wbc_ctrl_cycle_;
  double control_dt_;  // [2ms]
  Mode mode_pr_;
  uint8_t mode_machine_;
  std::chrono::steady_clock::time_point start_time_;  // for the REGULATION Kp ramp

  Gamepad gamepad_;
  REMOTE_DATA_RX rx_;

  bool init_walking_manager_ = false;
  std::atomic<bool> ekf_active_{false};        // set on A press
  labrob::WalkingManager walking_manager_;
  std::unique_ptr<labrob::StateEstimator> state_estimator_ptr_;

  SensorLogger sensor_logger_;  // raw feedback, for saveExperimentLogs()

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;
  DataBuffer<SportModeState> sportmode_buffer_;

  ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
  ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_> imutorso_subscriber_;
  ChannelSubscriberPtr<SportModeState_> sportmodestate_subscriber_;
  ThreadPtr command_writer_ptr_, control_thread_ptr_;

  std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;

 public:
  G1(std::string networkInterface)
      : wbc_ctrl_cycle_(0),
        control_dt_(0.002),
        mode_pr_(Mode::PR),
        mode_machine_(0),
        start_time_(std::chrono::steady_clock::now()) {
    state_estimator_ptr_ = std::make_unique<labrob::StateEstimator>(
        kUrdfPath, control_dt_, labrob::StateEstimator::Filter::RightInvariantEKF);

    ChannelFactory::Instance()->Init(0, networkInterface);

    // try to shutdown motion control-related service
    msc_ = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
    msc_->SetTimeout(5.0f);
    msc_->Init();
    std::string form, name;
    while (msc_->CheckMode(form, name), !name.empty()) {        // SLOW DOWN
      if (msc_->ReleaseMode())
        std::cout << "Failed to switch to Release Mode\n";
      // sleep(5);
    }

    // create publisher
    lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();
    // create subscriber
    lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(std::bind(&G1::LowStateHandler, this, std::placeholders::_1), 1);
    imutorso_subscriber_.reset(new ChannelSubscriber<unitree_hg::msg::dds_::IMUState_>(HG_IMU_TORSO));
    imutorso_subscriber_->InitChannel(std::bind(&G1::imuTorsoHandler, this, std::placeholders::_1), 1);
    sportmodestate_subscriber_.reset(new ChannelSubscriber<SportModeState_>(GO_STATE_TOPIC));
    sportmodestate_subscriber_->InitChannel(std::bind(&G1::SportModeStateHandler, this, std::placeholders::_1), 1);
    // create threads
    command_writer_ptr_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 2000, &G1::LowCommandWriter, this);
    control_thread_ptr_ = CreateRecurrentThreadEx("control", UT_CPU_ID_NONE, 2000, &G1::Control, this);
  }


  void imuTorsoHandler(const void *message) {
    unitree_hg::msg::dds_::IMUState_ imu_torso = *(const unitree_hg::msg::dds_::IMUState_ *)message;
    auto &rpy = imu_torso.rpy();
  }

  void SportModeStateHandler(const void *message) {
    SportModeState_ sportmodestate = *(const SportModeState_ *)message;

    SportModeState sm_tmp;
    sm_tmp.position = Eigen::Vector3d(
        sportmodestate.position()[0], sportmodestate.position()[1], sportmodestate.position()[2]);
    sm_tmp.velocity = Eigen::Vector3d(
        sportmodestate.velocity()[0], sportmodestate.velocity()[1], sportmodestate.velocity()[2]);
    sm_tmp.quaternion = Eigen::Vector4d(
        sportmodestate.imu_state().quaternion()[0], sportmodestate.imu_state().quaternion()[1],
        sportmodestate.imu_state().quaternion()[2], sportmodestate.imu_state().quaternion()[3]);
    sportmode_buffer_.SetData(sm_tmp);

    sensor_logger_.log("odom_pos", sm_tmp.position);
    sensor_logger_.log("odom_vel", sm_tmp.velocity);
    sensor_logger_.log("odom_quat", sm_tmp.quaternion);
  }

  void LowStateHandler(const void *message) {
    LowState_ low_state = *(const LowState_ *)message;
    if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1)) {
      std::cout << "[ERROR] CRC Error" << std::endl;
      return;
    }

    // get motor state
    MotorState ms_tmp;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = low_state.motor_state()[i].q();
      ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
      if (low_state.motor_state()[i].motorstate() && i <= RightAnkleRoll)
        std::cout << "[ERROR] motor " << i << " with code " << low_state.motor_state()[i].motorstate() << "\n";
    }
    motor_state_buffer_.SetData(ms_tmp);
    sensor_logger_.log("joint_pos", Eigen::Map<const Eigen::Matrix<float, G1_NUM_MOTOR, 1>>(ms_tmp.q.data()).cast<double>());
    sensor_logger_.log("joint_vel", Eigen::Map<const Eigen::Matrix<float, G1_NUM_MOTOR, 1>>(ms_tmp.dq.data()).cast<double>());

    // get imu state
    ImuState imu_tmp;
    imu_tmp.omega = low_state.imu_state().gyroscope();
    imu_tmp.accelerometer = low_state.imu_state().accelerometer();
    imu_tmp.rpy = low_state.imu_state().rpy();
    imu_state_buffer_.SetData(imu_tmp);
    sensor_logger_.log("pelvis_gyro", Eigen::Map<const Eigen::Vector3f>(imu_tmp.omega.data()).cast<double>());
    sensor_logger_.log("pelvis_acc", Eigen::Map<const Eigen::Vector3f>(imu_tmp.accelerometer.data()).cast<double>());

    // update gamepad
    memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
    gamepad_.update(rx_.RF_RX);


    // security stop
    static bool yPressedPrev = false;
    if (gamepad_.Y.pressed && !yPressedPrev) {
        printf("[GAMEPAD] Y -> stopping\n");
        signalHandler(SIGINT);
    }
    yPressedPrev = gamepad_.Y.pressed;

    // request EKF activation; actually activated in Control() (needs robot_state)
    static bool aPressedPrev = false;
    if (gamepad_.A.pressed && !aPressedPrev) {
        printf("[GAMEPAD] A -> EKF start requested\n");
        ekf_active_ = true;
    }
    aPressedPrev = gamepad_.A.pressed;

    // reserved, no action yet
    static bool bPressedPrev = false;
    if (gamepad_.B.pressed && !bPressedPrev) {
        printf("[GAMEPAD] B pressed (no action assigned)\n");
    }
    bPressedPrev = gamepad_.B.pressed;

    // switch REGULATION -> WBC (init_walking_manager_ makes Control() run
    // walking_manager_.init() exactly once on the first WBC tick)
    static bool xPressedPrev = false;
    if (gamepad_.X.pressed && !xPressedPrev && ctrl_mode == REGULATION) {
        printf("[GAMEPAD] X -> switching to WBC\n");
        ctrl_mode = WBC;
    }
    xPressedPrev = gamepad_.X.pressed;

    // update mode machine
    if (mode_machine_ != low_state.mode_machine()) {
      if (mode_machine_ == 0) std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
      mode_machine_ = low_state.mode_machine();
    }


  }

  void LowCommandWriter() {
    LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
    dds_low_command.mode_machine() = mode_machine_;

    const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();
    if (mc) {
      for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
        dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
        dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
        dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
        dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
        dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);
      }

      dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
      if (ctrl_mode != LISTEN)
        lowcmd_publisher_->Write(dds_low_command);
    }
  }


  void Control() {

    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      motor_command_tmp.tau_ff.at(i) = 0.0;
      motor_command_tmp.q_target.at(i) = 0.0;
      motor_command_tmp.dq_target.at(i) = 0.0;
      motor_command_tmp.kp.at(i) = 0.0;
      motor_command_tmp.kd.at(i) = 0.0;
    }
    
    if (!ms) {
      motor_command_buffer_.SetData(motor_command_tmp);
      return;
    }

    // ── State estimation: runs every tick regardless of ctrl_mode, so the
    // filter can be started (gamepad A) and converge while still in
    // LISTEN/REGULATION, before switching to WBC (gamepad X). ──────────────
    labrob::RobotState robot_state;
    bool have_robot_state = false;
    {
      const std::shared_ptr<const ImuState> imu = imu_state_buffer_.GetData();
      const std::shared_ptr<const SportModeState> odom = sportmode_buffer_.GetData();
      if (imu && odom) {
        const Eigen::Vector3d imu_gyro(imu->omega[0], imu->omega[1], imu->omega[2]);
        const Eigen::Vector3d imu_acc(
            imu->accelerometer[0], imu->accelerometer[1], imu->accelerometer[2]);

        // Order matches main_g1.cpp: linear_velocity is rotated by the
        // *previous* tick's orientation, updated right after.
        robot_state.angular_velocity = imu_gyro;
        robot_state.position = odom->position;
        robot_state.orientation = Eigen::Quaterniond(
            odom->quaternion(0), odom->quaternion(1), odom->quaternion(2), odom->quaternion(3));
        robot_state.linear_velocity =
            robot_state.orientation.toRotationMatrix().transpose() * odom->velocity;

        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          robot_state.joint_state[kJointNames[i]].pos = ms->q.at(i);
          robot_state.joint_state[kJointNames[i]].vel = ms->dq.at(i);
        }
        have_robot_state = true;

        // No feet force/torque sensing available here: assume double support.
        const std::array<bool, 2> contact{true, true};
        if (ekf_active_ && !state_estimator_ptr_->is_active()) {
          Eigen::VectorXd q_joints_pin(G1_NUM_MOTOR);
          for (int i = 0; i < G1_NUM_MOTOR; ++i) q_joints_pin(i) = ms->q.at(i);
          state_estimator_ptr_->activate(robot_state, q_joints_pin);
        }
        if (state_estimator_ptr_->is_active())
          state_estimator_ptr_->update(robot_state, imu_gyro, imu_acc, contact);

        // Log robot_state generically: before the filter is active this is
        // just the odometry-derived state, same channel name throughout.
        sensor_logger_.log("filtered_base_position", robot_state.position);
        sensor_logger_.log("filtered_base_velocity", robot_state.linear_velocity);
        sensor_logger_.log("filtered_base_quat", Eigen::Vector4d(
            robot_state.orientation.w(), robot_state.orientation.x(),
            robot_state.orientation.y(), robot_state.orientation.z()));
        sensor_logger_.log("filtered_base_rpy",
            robot_state.orientation.toRotationMatrix().eulerAngles(0, 1, 2));
        sensor_logger_.log("filtered_base_ang_vel", robot_state.angular_velocity);
      }
    }

    // listen only: nothing to send
    if (ctrl_mode == LISTEN) {
      motor_command_buffer_.SetData(motor_command_tmp);
      return;
    }

    switch (ctrl_mode) {

      case REGULATION: {

        std::array<float, G1_NUM_MOTOR> q_target{
          -0.44, 0.05, 0.0, 0.95, -0.49, -0.07,     // Left leg
          -0.44, -0.05, 0.0, 0.95, -0.49, 0.07,     // Right leg
          0.0, 0.0, 0.0,                            // Waist
          0.07, 0.14, 0.0, 1.1308, 0.0, 0.0, 0.0,   // Left arm
          0.07, -0.14, 0.0, 1.1308, 0.0, 0.0, 0.0   // Right arm
        };

        // Ramp Kp from 0 to Kp_reg over the first 5s since startup, so the
        // robot doesn't snap to the regulation posture at full stiffness.
        // Kd is applied at full value immediately (same as main_g1.cpp).
        constexpr float kRampDuration = 5.0f;
        const float elapsed_s = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - start_time_).count();
        const float kp_ratio = std::clamp(elapsed_s / kRampDuration, 0.0f, 1.0f);

        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          motor_command_tmp.tau_ff.at(i) = 0.0;
          motor_command_tmp.q_target.at(i) = q_target[i];
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = Kp_reg[i] * kp_ratio;
          motor_command_tmp.kd.at(i) = Kd_reg[i];
        }

        break;
      }

      case WBC: {
        if (!have_robot_state) break;  // wait for at least one imu+odom sample

        if (!init_walking_manager_) {
          std::map<std::string, double> armatures;  // TODO: calibrate real values
          walking_manager_.init(robot_state, armatures);
          init_walking_manager_ = true;
        }

        labrob::JointCommand joint_command;
        walking_manager_.update(robot_state, joint_command);

        // Position/velocity references from the WBC's solved acceleration
        // (see main_g1.cpp), on top of the torque solution. With Kp_wbc/Kd_wbc
        // still at 0 these have no effect on the real command until tuned.
        constexpr double kCmdDt = 0.002;
        const Eigen::VectorXd& jddot = walking_manager_.get_wbc_q_ddot();
        const Eigen::VectorXd jddot_joints = jddot.tail(G1_NUM_MOTOR);
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          const auto& jstate = robot_state.joint_state[kJointNames[i]];
          motor_command_tmp.tau_ff.at(i) = joint_command[kJointNames[i]];
          motor_command_tmp.q_target.at(i) =
              jstate.pos + jstate.vel * kCmdDt + 0.5 * jddot_joints[i] * kCmdDt * kCmdDt;
          motor_command_tmp.dq_target.at(i) = jstate.vel + jddot_joints[i] * kCmdDt;
          motor_command_tmp.kp.at(i) = Kp_wbc[i];
          motor_command_tmp.kd.at(i) = Kd_wbc[i];
        }

        ++wbc_ctrl_cycle_;

        break;
      }

    }

    // Safety check: refuse to send a command that looks unsafe, on either the
    // measured state or the just-computed command (same thresholds as
    // main_g1.cpp's send_dds_command()).
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      if (std::abs(ms->q.at(i)) > 3.14 ||
          std::abs(ms->dq.at(i)) > 3.0 ||
          std::abs(motor_command_tmp.tau_ff.at(i)) > 60.0) {
        std::cout << "[SAFETY] limit exceeded on " << kJointNames[i]
                  << ": q=" << ms->q.at(i) << " dq=" << ms->dq.at(i)
                  << " tau=" << motor_command_tmp.tau_ff.at(i) << std::endl;
        signalHandler(SIGINT);
        for (int j = 0; j < G1_NUM_MOTOR; ++j) {
          motor_command_tmp.tau_ff.at(j) = 0.0;
          motor_command_tmp.kp.at(j) = 0.0;
          motor_command_tmp.kd.at(j) = 0.0;
        }
        break;
      }
    }

    motor_command_buffer_.SetData(motor_command_tmp);





    //   time_ += control_dt_;
    //   if (time_ < duration_) {
    //     // [Stage 1]: set robot to zero posture
    //     for (int i = 0; i < G1_NUM_MOTOR; ++i) {
    //       double ratio = std::clamp(time_ / duration_, 0.0, 1.0);
    //       motor_command_tmp.q_target.at(i) = (1.0 - ratio) * ms->q.at(i);
    //     }
    //   } else if (time_ < duration_ * 2) {
    //     // [Stage 2]: swing ankle using PR mode
    //     mode_pr_ = Mode::PR;
    //     double max_P = M_PI * 30.0 / 180.0;
    //     double max_R = M_PI * 10.0 / 180.0;
    //     double t = time_ - duration_;
    //     double L_P_des = max_P * std::sin(2.0 * M_PI * t);
    //     double L_R_des = max_R * std::sin(2.0 * M_PI * t);
    //     double R_P_des = max_P * std::sin(2.0 * M_PI * t);
    //     double R_R_des = -max_R * std::sin(2.0 * M_PI * t);

    //     motor_command_tmp.q_target.at(LeftAnklePitch) = L_P_des;
    //     motor_command_tmp.q_target.at(LeftAnkleRoll) = L_R_des;
    //     motor_command_tmp.q_target.at(RightAnklePitch) = R_P_des;
    //     motor_command_tmp.q_target.at(RightAnkleRoll) = R_R_des;
    //   } else {
    //     // [Stage 3]: swing ankle using AB mode
    //     mode_pr_ = Mode::AB;
    //     double max_A = M_PI * 30.0 / 180.0;
    //     double max_B = M_PI * 10.0 / 180.0;
    //     double t = time_ - duration_ * 2;
    //     double L_A_des = +max_A * std::sin(M_PI * t);
    //     double L_B_des = +max_B * std::sin(M_PI * t + M_PI);
    //     double R_A_des = -max_A * std::sin(M_PI * t);
    //     double R_B_des = -max_B * std::sin(M_PI * t + M_PI);

    //     motor_command_tmp.q_target.at(LeftAnkleA) = L_A_des;
    //     motor_command_tmp.q_target.at(LeftAnkleB) = L_B_des;
    //     motor_command_tmp.q_target.at(RightAnkleA) = R_A_des;
    //     motor_command_tmp.q_target.at(RightAnkleB) = R_B_des;
    //   }












    //   motor_command_buffer_.SetData(motor_command_tmp);
    // }



  }

  // Saves accumulated sensor_logger_ data + joint_names.txt to /tmp/robot_logs,
  // then copies it into experiments/experiment_N/, mirroring
  // save_experiment_logs() in main_g1.cpp.
  void saveExperimentLogs() {
    std::filesystem::remove_all("/tmp/robot_logs");
    std::filesystem::create_directories("/tmp/robot_logs");

    sensor_logger_.save("/tmp/robot_logs");
    walking_manager_.save_data();

    {
      std::ofstream joint_names_file("/tmp/robot_logs/joint_names.txt");
      for (const auto& name : kJointNames)
        joint_names_file << name << "\n";
    }  // flush + close before the directory gets copied below

    std::cout << "Logs saved." << std::endl;

    std::string experiment_folder;
    int n = 1;
    while (true) {
      if (!std::filesystem::exists("../experiments"))
        std::filesystem::create_directory("../experiments");
      experiment_folder = "../experiments/experiment_" + std::to_string(n);
      if (!std::filesystem::exists(experiment_folder)) {
        std::filesystem::create_directory(experiment_folder);
        break;
      }
      ++n;
    }
    if (std::filesystem::exists("/tmp/robot_logs"))
      std::filesystem::copy("/tmp/robot_logs",
          std::filesystem::path(experiment_folder) / "robot_logs",
          std::filesystem::copy_options::recursive |
          std::filesystem::copy_options::overwrite_existing);

    std::cout << "Experiment saved in folder experiment_" << n << std::endl;
  }
};

int main(int argc, char const *argv[]) {
  std::string networkInterface;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--listen") ctrl_mode = LISTEN;
    else if (networkInterface.empty()) networkInterface = a;
  }

  if (networkInterface.empty()) {
    std::cout << "Usage: g1_controller network_interface [--listen]" << std::endl;
    exit(0);
  }


  std::signal(SIGINT, signalHandler);



  
  auto start = std::chrono::steady_clock::now();

  G1 custom(networkInterface);

  auto end = std::chrono::steady_clock::now();

  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "G1 constructor took "
            << elapsed_ms.count()
            << " ms\n";

  // while (g_running) sleep(5);

  while (g_running)
  {
    // std::cout<< "in the loop " << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::cout << "Do you want to save logs? [y/n] ";
  std::string input;
  std::getline(std::cin, input);
  if (input == "y" || input == "Y" || input == "yes")
    custom.saveExperimentLogs();
  else
    std::cout << "Logs not saved." << std::endl;

  return 0;
}