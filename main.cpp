// std
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <csignal>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <filesystem>
#include <algorithm>
#include <array>

#include <thread>
#include <iomanip>
#include <cstdlib>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Pinocchio
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

// Labrob
#include <hrp4_locomotion/JointCommand.hpp>
#include <hrp4_locomotion/RobotState.hpp>
#include <hrp4_locomotion/WalkingManager.hpp>
#include <hrp4_locomotion/utils.hpp>
#include <hrp4_locomotion/gamepad.hpp>


#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

#include <hrp4_locomotion/globals.h>

#include <hrp4_locomotion/EstimateForce.hpp>

bool isTotalBodyLoopClosed = false;
bool isCoMLoopClosed = false;
bool isEKFactive = false;
bool useSim = false;
bool useRobot = false;
bool oneTimepress = true;
bool isIMUcalibrating = false;


double startTimeTotalBodyCL = 15000.0;
double startTimeCoMCL = 15000.0;
double startTimeEKF = 0.0;
double startTimeIMUcalibrating = 0.0;

Eigen::Vector3d imu_accelerometer = Eigen::Vector3d::Zero();

#include "MujocoUI.hpp"

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace unitree::common;
using namespace unitree::robot::b2;

bool switchWalkingState = false;
bool IMUcalibrated = false;

Gamepad gamepad_;
REMOTE_DATA_RX rx_;

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
labrob::WalkingManager walking_manager;
labrob::EstimateForce estimate_force;

std::vector<Eigen::VectorXd> left_wrist_wrench_log;
std::vector<Eigen::VectorXd> right_wrist_wrench_log;
std::vector<Eigen::VectorXd> left_wrist_wrench_filtered_log;
std::vector<Eigen::VectorXd> right_wrist_wrench_filtered_log;
std::vector<double> wrist_force_time_log;
std::vector<Eigen::Vector3d> left_wrist_force_gt_log;
std::vector<Eigen::Vector3d> right_wrist_force_gt_log;
std::vector<Eigen::Vector3d> left_wrist_torque_gt_log;
std::vector<Eigen::Vector3d> right_wrist_torque_gt_log;
std::vector<Eigen::Vector3d> left_wrist_force_point_log;
std::vector<Eigen::Vector3d> right_wrist_force_point_log;
std::vector<int> left_wrist_force_enabled_log;
std::vector<int> right_wrist_force_enabled_log;
std::vector<double> all_joint_motor_time_log;
std::map<std::string, std::vector<std::array<float, 5>>> all_joint_motor_log;

std::vector<std::pair<std::string, int>> getValidJointNameIndexPairs();

void saveEstimateForceLogs() {
  std::ofstream left_wrist_file("/tmp/estimated_force_left_wrist.txt");
  for (const auto& v : left_wrist_wrench_log) {
    left_wrist_file << v.transpose() << "\n";
  }

  std::ofstream right_wrist_file("/tmp/estimated_force_right_wrist.txt");
  for (const auto& v : right_wrist_wrench_log) {
    right_wrist_file << v.transpose() << "\n";
  }

  std::ofstream left_wrist_filtered_file("/tmp/estimated_force_left_wrist_filtered.txt");
  for (const auto& v : left_wrist_wrench_filtered_log) {
    left_wrist_filtered_file << v.transpose() << "\n";
  }

  std::ofstream right_wrist_filtered_file("/tmp/estimated_force_right_wrist_filtered.txt");
  for (const auto& v : right_wrist_wrench_filtered_log) {
    right_wrist_filtered_file << v.transpose() << "\n";
  }

  std::ofstream validation_file("/tmp/wrist_force_validation.txt");
  validation_file << "time "
                  << "l_gt_fx l_gt_fy l_gt_fz "
                  << "l_gt_tx l_gt_ty l_gt_tz "
                  << "l_est_fx l_est_fy l_est_fz "
                  << "l_est_tx l_est_ty l_est_tz "
                  << "l_estf_fx l_estf_fy l_estf_fz "
                  << "l_estf_tx l_estf_ty l_estf_tz "
                  << "r_gt_fx r_gt_fy r_gt_fz "
                  << "r_gt_tx r_gt_ty r_gt_tz "
                  << "r_est_fx r_est_fy r_est_fz "
                  << "r_est_tx r_est_ty r_est_tz "
                  << "r_estf_fx r_estf_fy r_estf_fz "
                  << "r_estf_tx r_estf_ty r_estf_tz\n";

  const std::size_t n = std::min(
      wrist_force_time_log.size(),
      std::min(
          left_wrist_force_gt_log.size(),
          std::min(
              right_wrist_force_gt_log.size(),
              std::min(
                left_wrist_torque_gt_log.size(),
                std::min(
                  right_wrist_torque_gt_log.size(),
                  std::min(
                    left_wrist_wrench_log.size(),
                    std::min(
                      right_wrist_wrench_log.size(),
                      std::min(left_wrist_wrench_filtered_log.size(), right_wrist_wrench_filtered_log.size())
                    )
                  )
                )
              )
            )
          )
        );

        for (std::size_t i = 0; i < n; ++i) {
        validation_file << wrist_force_time_log[i] << " "
                << left_wrist_force_gt_log[i].x() << " "
                << left_wrist_force_gt_log[i].y() << " "
                << left_wrist_force_gt_log[i].z() << " "
                << left_wrist_torque_gt_log[i].x() << " "
                << left_wrist_torque_gt_log[i].y() << " "
                << left_wrist_torque_gt_log[i].z() << " "
                << left_wrist_wrench_log[i](0) << " "
                << left_wrist_wrench_log[i](1) << " "
                << left_wrist_wrench_log[i](2) << " "
                << left_wrist_wrench_log[i](3) << " "
                << left_wrist_wrench_log[i](4) << " "
                << left_wrist_wrench_log[i](5) << " "
                << left_wrist_wrench_filtered_log[i](0) << " "
                << left_wrist_wrench_filtered_log[i](1) << " "
                << left_wrist_wrench_filtered_log[i](2) << " "
                << left_wrist_wrench_filtered_log[i](3) << " "
                << left_wrist_wrench_filtered_log[i](4) << " "
                << left_wrist_wrench_filtered_log[i](5) << " "
                << right_wrist_force_gt_log[i].x() << " "
                << right_wrist_force_gt_log[i].y() << " "
                << right_wrist_force_gt_log[i].z() << " "
                << right_wrist_torque_gt_log[i].x() << " "
                << right_wrist_torque_gt_log[i].y() << " "
                << right_wrist_torque_gt_log[i].z() << " "
                << right_wrist_wrench_log[i](0) << " "
                << right_wrist_wrench_log[i](1) << " "
                << right_wrist_wrench_log[i](2) << " "
                << right_wrist_wrench_log[i](3) << " "
                << right_wrist_wrench_log[i](4) << " "
                << right_wrist_wrench_log[i](5) << " "
                << right_wrist_wrench_filtered_log[i](0) << " "
                << right_wrist_wrench_filtered_log[i](1) << " "
                << right_wrist_wrench_filtered_log[i](2) << " "
                << right_wrist_wrench_filtered_log[i](3) << " "
                << right_wrist_wrench_filtered_log[i](4) << " "
                << right_wrist_wrench_filtered_log[i](5) << "\n";
        }

            std::ofstream applied_force_file("/tmp/applied_external_wrist_force.txt");
            applied_force_file << "time "
                     << "l_enabled l_px l_py l_pz l_fx l_fy l_fz l_tx l_ty l_tz "
                     << "r_enabled r_px r_py r_pz r_fx r_fy r_fz r_tx r_ty r_tz\n";

            const std::size_t n_applied = std::min(
              wrist_force_time_log.size(),
              std::min(
                left_wrist_force_point_log.size(),
                std::min(
                  right_wrist_force_point_log.size(),
                  std::min(
                    left_wrist_force_gt_log.size(),
                    std::min(
                      right_wrist_force_gt_log.size(),
                      std::min(
                        left_wrist_torque_gt_log.size(),
                        right_wrist_torque_gt_log.size()
                      )
                    )
                  )
                )
              )
            );

            for (std::size_t i = 0; i < n_applied; ++i) {
            applied_force_file << wrist_force_time_log[i] << " "
                       << left_wrist_force_enabled_log[i] << " "
                       << left_wrist_force_point_log[i].x() << " "
                       << left_wrist_force_point_log[i].y() << " "
                       << left_wrist_force_point_log[i].z() << " "
                       << left_wrist_force_gt_log[i].x() << " "
                       << left_wrist_force_gt_log[i].y() << " "
                       << left_wrist_force_gt_log[i].z() << " "
                       << left_wrist_torque_gt_log[i].x() << " "
                       << left_wrist_torque_gt_log[i].y() << " "
                       << left_wrist_torque_gt_log[i].z() << " "
                       << right_wrist_force_enabled_log[i] << " "
                       << right_wrist_force_point_log[i].x() << " "
                       << right_wrist_force_point_log[i].y() << " "
                       << right_wrist_force_point_log[i].z() << " "
                       << right_wrist_force_gt_log[i].x() << " "
                       << right_wrist_force_gt_log[i].y() << " "
                       << right_wrist_force_gt_log[i].z() << " "
                       << right_wrist_torque_gt_log[i].x() << " "
                       << right_wrist_torque_gt_log[i].y() << " "
                       << right_wrist_torque_gt_log[i].z() << "\n";
            }

  const auto valid_joint_pairs = getValidJointNameIndexPairs();
  std::vector<std::pair<std::string, int>> available_joint_pairs;
  available_joint_pairs.reserve(valid_joint_pairs.size());
  for (const auto& [joint_name, idx] : valid_joint_pairs) {
    auto it = all_joint_motor_log.find(joint_name);
    if (it != all_joint_motor_log.end() && !it->second.empty()) {
      available_joint_pairs.emplace_back(joint_name, idx);
    }
  }

  std::ofstream motor_source_file("/tmp/all_joint_motor_source.txt");
  std::ofstream motor_joint_names_file("/tmp/all_joint_motor_names.txt");
  std::ofstream motor_time_file("/tmp/all_joint_motor_time.txt");
  std::ofstream motor_q_file("/tmp/all_joint_motor_q.txt");
  std::ofstream motor_dq_file("/tmp/all_joint_motor_dq.txt");
  std::ofstream motor_ddq_file("/tmp/all_joint_motor_ddq.txt");
  std::ofstream motor_tau_est_file("/tmp/all_joint_motor_tau_est.txt");
  std::ofstream motor_tau_applied_file("/tmp/all_joint_motor_tau_applied.txt");

  if (!available_joint_pairs.empty()) {
    std::size_t n_motor = all_joint_motor_time_log.size();
    for (const auto& [joint_name, idx] : available_joint_pairs) {
      auto it = all_joint_motor_log.find(joint_name);
      if (it == all_joint_motor_log.end()) {
        n_motor = 0;
        break;
      }
      n_motor = std::min(n_motor, it->second.size());
    }

    motor_source_file << "source=sdk_low_state\n";
    motor_source_file << "samples=" << n_motor << "\n";
    motor_source_file << "joints=" << available_joint_pairs.size() << "\n";

    for (const auto& [joint_name, idx] : available_joint_pairs) {
      motor_joint_names_file << joint_name << "\n";
    }

    for (std::size_t i = 0; i < n_motor; ++i) {
      motor_time_file << all_joint_motor_time_log[i] << "\n";

      for (std::size_t j = 0; j < available_joint_pairs.size(); ++j) {
        const auto& [joint_name, idx] = available_joint_pairs[j];
        const auto& sample = all_joint_motor_log[joint_name][i];
        motor_q_file << sample[0];
        motor_dq_file << sample[1];
        motor_ddq_file << sample[2];
        motor_tau_est_file << sample[3];
        motor_tau_applied_file << sample[4];
        if (j + 1 < available_joint_pairs.size()) {
          motor_q_file << " ";
          motor_dq_file << " ";
          motor_ddq_file << " ";
          motor_tau_est_file << " ";
          motor_tau_applied_file << " ";
        }
      }

      motor_q_file << "\n";
      motor_dq_file << "\n";
      motor_ddq_file << "\n";
      motor_tau_est_file << "\n";
      motor_tau_applied_file << "\n";
    }
  } else {
    motor_source_file << "source=sdk_low_state\n";
    motor_source_file << "samples=0\n";
    motor_source_file << "joints=0\n";
    motor_source_file << "note=no lowstate motor samples captured in this run\n";
  }

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
  std::array<float, 4> quaternion = {};
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
  std::array<float, 3> accelerometer = {};
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
  std::array<float, G1_NUM_MOTOR> ddq = {};
  std::array<float, G1_NUM_MOTOR> tau_est = {};
};


// Stiffness for all G1 Joints
std::array<float, G1_NUM_MOTOR> Kp{
  700, 700, 700, 1000, 900, 500,      // legs sx
  700, 700, 700, 1000, 900, 500,      // legs dx
  400,                   // waist
  300, 300, 300, 300,  200, 200, 200,  // arms sx
  300, 300, 300, 300,  200, 200, 200,  // arms dx
  200, 200                              // right wrist pitch/yaw (29-dof)
};

// std::array<float, G1_NUM_MOTOR> Kp = {
//   25, 25, 25, 25, 25, 25, // legs sx
//   25, 25, 25, 25, 25, 25, // legs dx
//   25, 25, 25,                   // waist
//   25, 25, 25, 25, 25, 25, 25,   // arms sx
//   25, 25, 25, 25, 25, 25, 25    // arms dx
// };

// Damping for all G1 Joints
std::array<float, G1_NUM_MOTOR> Kd{
  10, 10, 10, 10, 10, 10,     // legs sx
  10, 10, 10, 10, 10, 10,     // legs dx
  10,             // waist
  10, 10, 10, 10, 10, 10, 10,  // arms sx
  10, 10, 10, 10, 10, 10, 10,  // arms dx
  10, 10                        // right wrist pitch/yaw (29-dof)
};

//assign at each value of kd twice the square root of the corresponding kp value
// std::array<float, G1_NUM_MOTOR> Kd = {
//   2*sqrt(Kp[0]), 2*sqrt(Kp[1]), 2*sqrt(Kp[2]), 2*sqrt(Kp[3]), 2*sqrt(Kp[4]), 2*sqrt(Kp[5]), // legs sx
//   2*sqrt(Kp[6]), 2*sqrt(Kp[7]), 2*sqrt(Kp[8]), 2*sqrt(Kp[9]), 2*sqrt(Kp[10]), 2*sqrt(Kp[11]), // legs dx
//   2*sqrt(Kp[12]), 2*sqrt(Kp[13]), 2*sqrt(Kp[14]), // waist
//   2*sqrt(Kp[15]), 2*sqrt(Kp[16]), 2*sqrt(Kp[17]), 2*sqrt(Kp[18]), 2*sqrt(Kp[19]), 2*sqrt(Kp[20]), 2*sqrt(Kp[21]), // arms sx
//   2*sqrt(Kp[22]), 2*sqrt(Kp[23]), 2*sqrt(Kp[24]), 2*sqrt(Kp[25]), 2*sqrt(Kp[26]), 2*sqrt(Kp[27]), 2*sqrt(Kp[28]) // arms dx
// };

std::mutex stateMutex;
DataBuffer<MotorState> motor_state_buffer_;
DataBuffer<MotorCommand> motor_command_buffer_;
uint8_t mode_machine_ = 0;

enum class Mode {
PR = 0,  // Series Control for Ptich/Roll Joints
AB = 1   // Parallel Control for A/B Joints
};

// make a map between joint name and robot joint index
std::map<std::string, int> joint_name_to_index = {
  {"left_hip_pitch_joint", 0},
  {"left_hip_roll_joint", 1},
  {"left_hip_yaw_joint", 2},
  {"left_knee_joint", 3},
  {"left_ankle_pitch_joint", 4},
  {"left_ankle_roll_joint", 5},
  {"right_hip_pitch_joint", 6},
  {"right_hip_roll_joint", 7},
  {"right_hip_yaw_joint", 8},
  {"right_knee_joint", 9},
  {"right_ankle_pitch_joint", 10},
  {"right_ankle_roll_joint", 11},
  {"waist_yaw_joint", 12},
  // {"waist_roll_joint", 13},        // NOTE INVALID for g1 23dof/29dof with waist locked
  // {"waist_pitch_joint", 14},      // NOTE INVALID for g1 23dof/29dof with waist locked
  {"left_shoulder_pitch_joint", 15},
  {"left_shoulder_roll_joint", 16},
  {"left_shoulder_yaw_joint", 17},
  {"left_elbow_joint", 18},
  {"left_wrist_roll_joint", 19},
  {"left_wrist_pitch_joint", 20},   // NOTE INVALID for g1 23dof
  {"left_wrist_yaw_joint", 21},       // NOTE INVALID for g1 23dof
  {"right_shoulder_pitch_joint", 22},
  {"right_shoulder_roll_joint", 23},
  {"right_shoulder_yaw_joint", 24},
  {"right_elbow_joint", 25},
  {"right_wrist_roll_joint", 26},
  {"right_wrist_pitch_joint", 27}, // NOTE INVALID for g1 23dof
  {"right_wrist_yaw_joint", 28}      // NOTE INVALID for g1 23dof
};

std::vector<std::pair<std::string, int>> getValidJointNameIndexPairs() {
  std::vector<std::pair<std::string, int>> valid_joint_pairs;
  valid_joint_pairs.reserve(joint_name_to_index.size());

  for (const auto& [joint_name, idx] : joint_name_to_index) {
    if (idx >= 0 && idx < G1_NUM_MOTOR) {
      valid_joint_pairs.emplace_back(joint_name, idx);
    }
  }

  std::sort(valid_joint_pairs.begin(), valid_joint_pairs.end(),
            [](const auto& a, const auto& b) {
              return a.second < b.second;
            });

  return valid_joint_pairs;
}

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

MotorState motor_state_data;
bool has_lowstate_data = false;
void LowStateHandler(const void* msg){
  LowState_ low_state = *(const LowState_*)msg;
  uint32_t crc_calc = Crc32Core((uint32_t*)&low_state, ((sizeof(LowState_) >> 2) -1));

  if (low_state.crc() != crc_calc) {
    std::cerr << "CRC32 mismatch in LowState message!" << std::endl;
    return;
  }

  std::lock_guard<std::mutex> lock(stateMutex);
  for (int i = 0; i < G1_NUM_MOTOR + 2; ++i) {
    if (i == 13 || i == 14) continue; // skip waist roll and pitch
    else if (i < 13) {
      motor_state_data.q[i] = low_state.motor_state()[i].q();
      motor_state_data.dq[i] = low_state.motor_state()[i].dq();
      motor_state_data.ddq[i] = low_state.motor_state()[i].ddq();
      motor_state_data.tau_est[i] = low_state.motor_state()[i].tau_est();
    }
    else if (i > 14) {
      motor_state_data.q[i - 2] = low_state.motor_state()[i].q();
      motor_state_data.dq[i - 2] = low_state.motor_state()[i].dq();
      motor_state_data.ddq[i - 2] = low_state.motor_state()[i].ddq();
      motor_state_data.tau_est[i - 2] = low_state.motor_state()[i].tau_est();
    }
  }
  has_lowstate_data = true;

  // update gamepad
  memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
  gamepad_.update(rx_.RF_RX);
  
  if (mode_machine_ != low_state.mode_machine()) {
    if (mode_machine_ == 0) {
      std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
    }
    mode_machine_ = low_state.mode_machine();
  }
}

ImuState imu_state_data;
void imuTorsoHandler(const void* msg) {
  IMUState_ imu_state = *(const IMUState_*)msg;

  std::lock_guard<std::mutex> lock(stateMutex);
  imu_state_data.quaternion[0] = imu_state.quaternion()[0];
  imu_state_data.quaternion[1] = imu_state.quaternion()[1];
  imu_state_data.quaternion[2] = imu_state.quaternion()[2];
  imu_state_data.quaternion[3] = imu_state.quaternion()[3];

  imu_state_data.rpy[0] = imu_state.rpy()[0];
  imu_state_data.rpy[1] = imu_state.rpy()[1];
  imu_state_data.rpy[2] = imu_state.rpy()[2];

  imu_state_data.omega[0] = imu_state.gyroscope()[0];
  imu_state_data.omega[1] = imu_state.gyroscope()[1];
  imu_state_data.omega[2] = imu_state.gyroscope()[2];

  imu_state_data.accelerometer[0] = imu_state.accelerometer()[0];
  imu_state_data.accelerometer[1] = imu_state.accelerometer()[1];
  imu_state_data.accelerometer[2] = imu_state.accelerometer()[2];
}

std::string queryServiceName(std::string form,std::string name)
{
    if(form == "0")
    {
        if(name == "normal" ) return "sport_mode"; 
        if(name == "ai" ) return "ai_sport"; 
        if(name == "advanced" ) return "advanced_sport"; 
    }
    else
    {
        if(name == "ai-w" ) return "wheeled_sport(go2W)"; 
        if(name == "normal-w" ) return "wheeled_sport(b2W)";
    }
    return "";
}

int queryMotionStatus(std::shared_ptr<MotionSwitcherClient> msc)
{
    std::string robotForm,motionName;
    int motionStatus;
    int32_t ret = msc->CheckMode(robotForm,motionName);
    if (ret == 0) {
        std::cout << "CheckMode succeeded." << std::endl;
    } else {
        std::cout << "CheckMode failed. Error code: " << ret << std::endl;
    }
    if(motionName.empty())
    {
        std::cout << "The motion control-related service is deactivated." << std::endl;
        motionStatus = 0;
    }
    else
    {
        std::string serviceName = queryServiceName(robotForm,motionName);
        std::cout << "Service: "<< serviceName<< " is activate" << std::endl;
        motionStatus = 1;
    }
    return motionStatus;
};

void StandStillInfinete(const labrob::RobotState& infi_robot_state, mjModel* mj_model_ptr, mjData* mj_data_ptr){
      //   // update mujoco state with robot_state
        mj_data_ptr->qpos[0] = infi_robot_state.position.x();
        mj_data_ptr->qpos[1] = infi_robot_state.position.y();
        mj_data_ptr->qpos[2] = infi_robot_state.position.z();
        mj_data_ptr->qpos[3] = infi_robot_state.orientation.w();
        mj_data_ptr->qpos[4] = infi_robot_state.orientation.x();
        mj_data_ptr->qpos[5] = infi_robot_state.orientation.y();
        mj_data_ptr->qpos[6] = infi_robot_state.orientation.z();
        //rotate the linear velocity from world to body frame
        Eigen::Vector3d lin_vel_body = infi_robot_state.orientation.toRotationMatrix() * infi_robot_state.linear_velocity;
        mj_data_ptr->qvel[0] = lin_vel_body.x();
        mj_data_ptr->qvel[1] = lin_vel_body.y();
        mj_data_ptr->qvel[2] = lin_vel_body.z();
        mj_data_ptr->qvel[3] = infi_robot_state.angular_velocity.x();
        mj_data_ptr->qvel[4] = infi_robot_state.angular_velocity.y();
        mj_data_ptr->qvel[5] = infi_robot_state.angular_velocity.z();
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[joint_id]] = infi_robot_state.joint_state[joint_name].pos;
          mj_data_ptr->qvel[mj_model_ptr->jnt_dofadr[joint_id]] = infi_robot_state.joint_state[joint_name].vel;
        }
}

void signalHandler(int signum) {
  std::cerr << "Received signal " << signum << ", exiting..." << std::endl;

  std::cout << "Exiting simulation loop." << std::endl;
  std::cout << "Do you want to save logs? [y/n]" << std::endl;
  std::string user_input;
  std::getline(std::cin, user_input);
  if(user_input == "y" || user_input == "Y" || user_input == "yes" || user_input == "Yes" || user_input == "YES"){
    std::cout << "Saving logs..." << std::endl;
    walking_manager.saveLogs();
    saveEstimateForceLogs();
    std::cout << "Logs saved." << std::endl;

    if(useRobot){
      std::string experiment_folder;
      bool experiment_folder_exists = true;
      int experiment_counter = 1;
      while (experiment_folder_exists) {
        if (!std::filesystem::exists("../experiments")) {
          std::filesystem::create_directory("../experiments");
          std::cout << "Created experiments directory." << std::endl;
        }
        experiment_folder = "../experiments/experiment_" + std::to_string(experiment_counter);
        experiment_folder_exists = std::filesystem::exists(experiment_folder);
        if (!experiment_folder_exists) {
          std::filesystem::create_directory(experiment_folder);
          std::cout << "Created experiment folder: " << experiment_folder << std::endl;
          break;
        }
        ++experiment_counter;
      }
      for (const auto& entry : std::filesystem::directory_iterator("/tmp")) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::filesystem::path destination = experiment_folder / entry.path().filename();
            std::filesystem::copy_file(entry.path(), destination, std::filesystem::copy_options::overwrite_existing);
        }
      }
      // create a README file in the experiment folder
      std::ofstream readme_file(experiment_folder + "/README.txt");
      if (readme_file.is_open()) {
        readme_file << "This folder contains the results of the experiment.\n";
        readme_file << "The gains used for the experiment are:\n";
        readme_file << "Kp: ";
        for (const auto& kp : Kp) {
          readme_file << kp << " ";
        }
        readme_file << "\nKd: ";
        for (const auto& kd : Kd) {
          readme_file << kd << " ";
        }
        readme_file << "\n\n";
  
        //request text input from terminal and write the text on the readme file
        std::string user_input;
        std::cout << "Please enter a description of the experiment: ";
        std::getline(std::cin, user_input);
        if (user_input == "delete" || user_input == "remove" || user_input == "erase" || user_input == "trash") {
          std::cout << "Deleting experiment folder: " << experiment_folder << std::endl;
          std::filesystem::remove_all(experiment_folder);
          readme_file.close();
        }
        else{
          std::cout << "Experiment description: " << user_input << std::endl;
          readme_file << "Experiment description: " << user_input << "\n\n";
        }
  
        readme_file.close();
      }
    }
  }
  else{
    std::cout << "Logs not saved." << std::endl;
  }

  exit(signum);
}

labrob::RobotState
robot_state_from_mujoco(mjModel* m, mjData* d) {
  labrob::RobotState robot_state;

  robot_state.position = Eigen::Vector3d(
    d->qpos[0], d->qpos[1], d->qpos[2]
  );

  robot_state.orientation = Eigen::Quaterniond(
      d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]
  );

  robot_state.linear_velocity = robot_state.orientation.toRotationMatrix().transpose() *
      Eigen::Vector3d(
          d->qvel[0], d->qvel[1], d->qvel[2]
      );

  robot_state.angular_velocity = Eigen::Vector3d(
    d->qvel[3], d->qvel[4], d->qvel[5]
  );

  for (int i = 1; i < m->njnt; ++i) {
    const char* name = mj_id2name(m, mjOBJ_JOINT, i);
    robot_state.joint_state[name].pos = d->qpos[m->jnt_qposadr[i]];
    robot_state.joint_state[name].vel = d->qvel[m->jnt_dofadr[i]];
    robot_state.joint_state[name].acc = d->qacc[m->jnt_dofadr[i]];
    robot_state.joint_state[name].eff = d->qfrc_actuator[m->jnt_dofadr[i]];
  }

  static double force[6];
  static double result[3];
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  robot_state.contact_points.resize(d->ncon);
  robot_state.contact_forces.resize(d->ncon);
  for (int i = 0; i < d->ncon; ++i) {
    mj_contactForce(m, d, i, force);
    //mju_rotVecMatT(result, force, d->contact[i].frame);
    mju_mulMatVec(result, d->contact[i].frame, force, 3, 3);
    for (int row = 0; row < 3; ++row) {
        result[row] = 0;
        for (int col = 0; col < 3; ++col) {
            result[row] += d->contact[i].frame[3 * col + row] * force[col];
        }
    }
    sum += Eigen::Vector3d(result);
    for (int j = 0; j < 3; ++j) {
      robot_state.contact_points[i](j) = d->contact[i].pos[j];
      robot_state.contact_forces[i](j) = result[j];
    }
  }

  robot_state.total_force = sum;

  return robot_state;
}

int main(const int argc, const char* argv[]) {

  std::string netInterface;
  std::string sdkInterface;
  bool enable_external_left_force_test = false;
  double external_left_fx_newton = 0.0;
  double external_left_fy_newton = 0.0;
  double external_left_fz_newton = 0.0;
  bool enable_external_left_torque_test = false;
  double external_left_tx_newton_meter = 0.0;
  double external_left_ty_newton_meter = 0.0;
  double external_left_tz_newton_meter = 0.0;
  bool enable_external_right_force_test = false;
  double external_right_fx_newton = 0.0;
  double external_right_fy_newton = 0.0;
  double external_right_fz_newton = 0.0;
  bool enable_external_right_torque_test = false;
  double external_right_tx_newton_meter = 0.0;
  double external_right_ty_newton_meter = 0.0;
  double external_right_tz_newton_meter = 0.0;
  double external_force_start_sec = 5.0;
  double external_force_duration_sec = 2.0;
  double external_force_ramp_sec = 0.2;
  bool use_mujoco_step = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--sim") {
        useSim = true;
    } else if (a == "--robot" && i + 1 < argc) {
        useRobot = true;
        useSim = true;
        netInterface = argv[++i];
    } else if (a == "--sdk-interface" && i + 1 < argc) {
      sdkInterface = argv[++i];
    } else if (a == "--external-left-fz" && i + 1 < argc) {
      enable_external_left_force_test = true;
      external_left_fz_newton = std::atof(argv[++i]);
    } else if (a == "--external-left-fx" && i + 1 < argc) {
      enable_external_left_force_test = true;
      external_left_fx_newton = std::atof(argv[++i]);
    } else if (a == "--external-left-fy" && i + 1 < argc) {
      enable_external_left_force_test = true;
      external_left_fy_newton = std::atof(argv[++i]);
    } else if (a == "--external-left-tx" && i + 1 < argc) {
      enable_external_left_torque_test = true;
      external_left_tx_newton_meter = std::atof(argv[++i]);
    } else if (a == "--external-left-ty" && i + 1 < argc) {
      enable_external_left_torque_test = true;
      external_left_ty_newton_meter = std::atof(argv[++i]);
    } else if (a == "--external-left-tz" && i + 1 < argc) {
      enable_external_left_torque_test = true;
      external_left_tz_newton_meter = std::atof(argv[++i]);
    } else if (a == "--external-right-fz" && i + 1 < argc) {
      enable_external_right_force_test = true;
      external_right_fz_newton = std::atof(argv[++i]);
    } else if (a == "--external-right-fx" && i + 1 < argc) {
      enable_external_right_force_test = true;
      external_right_fx_newton = std::atof(argv[++i]);
    } else if (a == "--external-right-fy" && i + 1 < argc) {
      enable_external_right_force_test = true;
      external_right_fy_newton = std::atof(argv[++i]);
    } else if (a == "--external-right-tx" && i + 1 < argc) {
      enable_external_right_torque_test = true;
      external_right_tx_newton_meter = std::atof(argv[++i]);
    } else if (a == "--external-right-ty" && i + 1 < argc) {
      enable_external_right_torque_test = true;
      external_right_ty_newton_meter = std::atof(argv[++i]);
    } else if (a == "--external-right-tz" && i + 1 < argc) {
      enable_external_right_torque_test = true;
      external_right_tz_newton_meter = std::atof(argv[++i]);
    } else if (a == "--external-force-start" && i + 1 < argc) {
      external_force_start_sec = std::atof(argv[++i]);
    } else if (a == "--external-force-duration" && i + 1 < argc) {
      external_force_duration_sec = std::atof(argv[++i]);
    } else if (a == "--external-force-ramp" && i + 1 < argc) {
      external_force_ramp_sec = std::atof(argv[++i]);
    } else if (a == "--use-mujoco-step") {
      use_mujoco_step = true;
    } else if (a == "--help" || a == "-h") {
      std::cout << "Usage:\n"
            << "  --sim\n"
            << "  --robot <network_interface>\n"
        << "  --sdk-interface <network_interface>\n"
            << "Optional external-force test params:\n"
            << "  --external-left-fz <newton>\n"
            << "  --external-left-fx <newton>\n"
            << "  --external-left-fy <newton>\n"
            << "  --external-left-tx <newton_meter>\n"
            << "  --external-left-ty <newton_meter>\n"
            << "  --external-left-tz <newton_meter>\n"
            << "  --external-right-fz <newton>\n"
            << "  --external-right-fx <newton>\n"
            << "  --external-right-fy <newton>\n"
            << "  --external-right-tx <newton_meter>\n"
            << "  --external-right-ty <newton_meter>\n"
            << "  --external-right-tz <newton_meter>\n"
            << "  --external-force-start <sec>\n"
            << "  --external-force-duration <sec>\n"
            << "  --external-force-ramp <sec>\n"
            << "Other:\n"
            << "  --use-mujoco-step\n";
      return 0;
    }
  }

        if ((enable_external_left_force_test || enable_external_right_force_test ||
          enable_external_left_torque_test || enable_external_right_torque_test) && !useSim) {
    std::cerr << "External-force test requires simulation mode (--sim)." << std::endl;
    return -1;
    }

  if(!useRobot && !useSim) {
    std::cerr << "Please specify either --sim or --robot <network_interface>" << std::endl;
    return -1;
  }

 signal(SIGINT, signalHandler);

  // Load MJCF (for Mujoco):
  const int kErrorLength = 1024;          // load error string length
  char loadError[kErrorLength] = "";
  const char* mjcf_filepath = "../g1_mj_description/stair_steps.xml";
  mjModel* mj_model_ptr = mj_loadXML(mjcf_filepath, nullptr, loadError, kErrorLength);
  if (!mj_model_ptr) {
    std::cerr << "Error loading model: " << loadError << std::endl;
    return -1;
  }
  mjData* mj_data_ptr = mj_makeData(mj_model_ptr);

  if (useRobot) {
    std::cout << "Press 'X' on the GAMEPAD to toggle CoM closed loop." << std::endl;
    std::cout << "Press 'Y' on the GAMEPAD to end the program." << std::endl;
    std::cout << "Press 'B' on the GAMEPAD to switch walking state." << std::endl;
    std::cout << "If GAMEPAD is not used, select now which loops to close:" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "1. Center of Mass (CoM)" << std::endl;
    std::cout << "2. Total Body" << std::endl;
    std::cout << "3. Extended Kalman Filter (EKF)" << std::endl;
    std::cout << "You can select multiple options by entering their numbers separated by spaces (e.g., '1 3' for CoM and EKF)." << std::endl;
    std::cout << "Enter your choice: " << std::endl;
    std::string user_input;
    std::getline(std::cin, user_input);
    std::istringstream iss(user_input);
    std::string token;
    while (iss >> token) {
      if (token == "1") {
        isCoMLoopClosed = true;
      } else if (token == "2") {
        isTotalBodyLoopClosed = true;
      } else if (token == "3") {
        isEKFactive = true;
      } 
    }
  } else {
    isTotalBodyLoopClosed = true;
    isCoMLoopClosed = true;
    isEKFactive = true;
  }
  

  // Init robot posture:
  mjtNum waist_y_init = 0.0;
  mjtNum r_hip_y_init = -0.005;
  mjtNum r_hip_r_init = -0.04;
  mjtNum r_hip_p_init = -0.44;
  mjtNum r_knee_init = 0.95;
  mjtNum r_ankle_p_init = -0.49;
  mjtNum r_ankle_r_init = 0.07;
  mjtNum l_hip_y_init = 0.0;
  mjtNum l_hip_r_init = -r_hip_r_init;
  mjtNum l_hip_p_init = r_hip_p_init;
  mjtNum l_knee_init = r_knee_init;
  mjtNum l_ankle_p_init = r_ankle_p_init;
  mjtNum l_ankle_r_init = -r_ankle_r_init;
  mjtNum r_shoulder_p_init = 0.07;
  mjtNum r_shoulder_r_init = -0.12;
  mjtNum r_shoulder_y_init = 0.0;
  mjtNum r_elbow_p_init = 3.14 / 2.0 - 0.44;
  mjtNum l_shoulder_p_init = r_shoulder_p_init;
  mjtNum l_shoulder_r_init = -r_shoulder_r_init;
  mjtNum l_shoulder_y_init = 0.0;
  mjtNum l_elbow_p_init = r_elbow_p_init;

  for (int i = 0; i < mj_model_ptr->nq; ++i) {
    mj_data_ptr->qpos[i] = 0.0;
  }
  // mj_data_ptr->qpos[0] = 10.0;
  // mj_data_ptr->qpos[1] = 10.0;

  mj_data_ptr->qpos[2] = 0.792151-0.125+0.0263 - 0.071 + 0.105 - 0.010526;
  mj_data_ptr->qpos[3] = 1.0;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "waist_yaw_joint")]] = waist_y_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_hip_yaw_joint")]] = r_hip_y_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_hip_roll_joint")]] = r_hip_r_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_hip_pitch_joint")]] = r_hip_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_knee_joint")]] = r_knee_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_ankle_pitch_joint")]] = r_ankle_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_ankle_roll_joint")]] = r_ankle_r_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_hip_yaw_joint")]] = l_hip_y_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_hip_roll_joint")]] = l_hip_r_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_hip_pitch_joint")]] = l_hip_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_knee_joint")]] = l_knee_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_ankle_pitch_joint")]] = l_ankle_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_ankle_roll_joint")]] = l_ankle_r_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_shoulder_pitch_joint")]] = r_shoulder_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_shoulder_roll_joint")]] = r_shoulder_r_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_shoulder_yaw_joint")]] = r_shoulder_y_init;
  // mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_elbow_joint")]] = r_elbow_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_shoulder_pitch_joint")]] = l_shoulder_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_shoulder_roll_joint")]] = l_shoulder_r_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_shoulder_yaw_joint")]] = l_shoulder_y_init;
  // mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_elbow_joint")]] = l_elbow_p_init;


  std::map<std::string, double> armatures;
  for (int i = 0; i < mj_model_ptr->nu; ++i) {
    int joint_id = mj_model_ptr->actuator_trnid[i * 2];
    std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
    int dof_id = mj_model_ptr->jnt_dofadr[joint_id];
    armatures[joint_name] = mj_model_ptr->dof_armature[dof_id];
  }

  // Walking Manager:
  labrob::RobotState robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
  
  walking_manager.init(robot_state, armatures);
  estimate_force.initialize(walking_manager.getRobotModel());
  const int left_wrist_body_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "left_wrist_yaw_link");
  const int right_wrist_body_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "right_wrist_yaw_link");
  if (enable_external_left_force_test || enable_external_left_torque_test) {
    std::cout << "External MuJoCo wrench test enabled on left wrist: F = ["
              << external_left_fx_newton << ", "
              << external_left_fy_newton << ", "
              << external_left_fz_newton << "] N, Tau = ["
              << external_left_tx_newton_meter << ", "
              << external_left_ty_newton_meter << ", "
              << external_left_tz_newton_meter << "] N*m" << std::endl;
    if (left_wrist_body_id < 0) {
      std::cout << "Warning: left_wrist_yaw_link body not found, external wrench will be ignored." << std::endl;
    }
  }
  if (enable_external_right_force_test || enable_external_right_torque_test) {
    std::cout << "External MuJoCo wrench test enabled on right wrist: F = ["
              << external_right_fx_newton << ", "
              << external_right_fy_newton << ", "
              << external_right_fz_newton << "] N, Tau = ["
              << external_right_tx_newton_meter << ", "
              << external_right_ty_newton_meter << ", "
              << external_right_tz_newton_meter << "] N*m" << std::endl;
    if (right_wrist_body_id < 0) {
      std::cout << "Warning: right_wrist_yaw_link body not found, external wrench will be ignored." << std::endl;
    }
  }
  if (enable_external_left_force_test || enable_external_right_force_test ||
      enable_external_left_torque_test || enable_external_right_torque_test) {
    std::cout << "External force window: start=" << external_force_start_sec
              << " s, duration=" << external_force_duration_sec << " s" << std::endl;
    std::cout << "External force ramp: " << external_force_ramp_sec << " s" << std::endl;
  }

  auto& mujoco_ui = *labrob::MujocoUI::getInstance(mj_model_ptr, mj_data_ptr);

  static int framerate = 60.0;

  ChannelPublisherPtr<LowCmd_> lowcmd_publisher;
  ChannelSubscriberPtr<LowState_> lowstate_subscriber;
  ChannelSubscriberPtr<IMUState_> imutorso_subscriber;
  std::shared_ptr<MotionSwitcherClient> msc;
  bool sdk_lowstate_stream_enabled = false;

  if(useRobot) {
    std::cout << "Using robot with network interface: " << netInterface << std::endl;
    ChannelFactory::Instance()->Init(0, netInterface);
    std::cout << "ChannelFactory initialized with interface: " << netInterface << std::endl;

    msc.reset(new MotionSwitcherClient());
    msc->SetTimeout(5.0f);
    msc->Init();

    while(queryMotionStatus(msc)){
      std::cout << "try to deactivate the motion control - related service" << std::endl;
      int32_t ret = msc->ReleaseMode();
      if (ret == 0) {
        std::cout << "Motion control service deactivated successfully." << std::endl;
      } else {
        std::cerr << "Failed to deactivate motion control service, retrying..." << std::endl;
        sleep(5);
      }
    }

    lowcmd_publisher.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher->InitChannel();
    lowstate_subscriber.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber->InitChannel(std::bind(&LowStateHandler, std::placeholders::_1), 1);
    imutorso_subscriber.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
    imutorso_subscriber->InitChannel(std::bind(&imuTorsoHandler, std::placeholders::_1), 1);
    sdk_lowstate_stream_enabled = true;
  } else {
    if (!sdkInterface.empty()) {
      std::cout << "[INFO] Running in simulation mode with SDK interface: " << sdkInterface << std::endl;
      ChannelFactory::Instance()->Init(0, sdkInterface);
      lowstate_subscriber.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
      lowstate_subscriber->InitChannel(std::bind(&LowStateHandler, std::placeholders::_1), 1);
      imutorso_subscriber.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
      imutorso_subscriber->InitChannel(std::bind(&imuTorsoHandler, std::placeholders::_1), 1);
      sdk_lowstate_stream_enabled = true;
    } else {
      std::cout << "[INFO] Running in simulation-only mode without SDK interface: "
                << "pass --sdk-interface <iface> to capture LowState tau_est logs." << std::endl;
    }
  }

  auto next_tick = std::chrono::steady_clock::now();

  constexpr double hand_force_min_newton = 0.0;
  constexpr double hand_force_max_newton = 5.0;
  constexpr double hand_force_start_threshold_newton = 0.5;
  constexpr double step_length_x_min_meter = 0.0;
  constexpr double step_length_x_max_meter = 0.1;
  constexpr double step_length_x_start_floor_meter = 0.06;
  bool walk_started_by_force = false;
  double latched_step_length_x = 0.0;

  // robot_state = walking_manager.getNewRobotState(robot_state);
  // StandStillInfinete(robot_state, mj_model_ptr, mj_data_ptr);

  // Simulation loop:
  while (!mujoco_ui.windowShouldClose()) {

    mjtNum simstart = mj_data_ptr->time;
    while( mj_data_ptr->time - simstart < 1.0/framerate ) {

      auto start_sleep = std::chrono::steady_clock::now();

      Eigen::VectorXd actual_output = Eigen::VectorXd::Zero(3 + mj_model_ptr->nu + 3 + mj_model_ptr->nu + 6 + 6);
      MotorState measured_motor_state;
      bool has_measured_motor_state = false;

      // If SDK lowstate is available, use it for measured joint states/torques.
      if (sdk_lowstate_stream_enabled) {

        if (useRobot && gamepad_.Y.pressed) {
          std::cout << "[GAMEPAD] Y pressed -> Deactivating motors..." << std::endl;
          signalHandler(SIGINT);
        }

        if (useRobot && gamepad_.X.on_press && isEKFactive) {
          isCoMLoopClosed = !isCoMLoopClosed;
          // isTotalBodyLoopClosed = !isTotalBodyLoopClosed;
          startTimeCoMCL = 1000 * mj_data_ptr->time;
          // startTimeTotalBodyCL = 1000 * mj_data_ptr->time;
          if(isCoMLoopClosed)
            std::cout << "[GAMEPAD] X pressed -> Closed loop activated." << std::endl;
          else
            std::cout << "[GAMEPAD] X pressed -> Closed loop deactivated." << std::endl;
        }

        if (useRobot && gamepad_.B.on_press) {
          switchWalkingState = true;
          std::cout << "[GAMEPAD] B pressed -> Walking state switched." << std::endl;
        }

        if (useRobot && gamepad_.A.on_press) {
          if(oneTimepress){
            std::cout << "[GAMEPAD] A pressed -> Starting IMU calibration routine..." << std::endl;
            isIMUcalibrating = true;
            oneTimepress = false;
            startTimeIMUcalibrating = 1000 * mj_data_ptr->time;
          }
          else{
            if(isCoMLoopClosed == true && isEKFactive == true){
              isTotalBodyLoopClosed = !isTotalBodyLoopClosed;
              if(isTotalBodyLoopClosed)
                std::cout << "[GAMEPAD] A pressed -> Total Body closed loop activated." << std::endl;
              else
                std::cout << "[GAMEPAD] A pressed -> Total Body closed loop deactivated." << std::endl;
            }
          }
        }

        std::lock_guard<std::mutex> lock(stateMutex);

        // save in actual_output: 1) imu orientation in quaternions, 2) joint positions, 3) imu angular velocity 4) joint velocities 5) imu accelerometer
        actual_output.head<3>() = labrob::rotVecFromQuaternion(Eigen::Quaterniond(
          imu_state_data.quaternion[0],
          imu_state_data.quaternion[1],
          imu_state_data.quaternion[2],
          imu_state_data.quaternion[3]
        ));
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          actual_output[3 + i] = motor_state_data.q[i];
          actual_output[3 + mj_model_ptr->nu + i + 3] = motor_state_data.dq[i];
        }
        measured_motor_state = motor_state_data;
        has_measured_motor_state = has_lowstate_data;
        actual_output[3 + mj_model_ptr->nu] = imu_state_data.omega[0];
        actual_output[3 + mj_model_ptr->nu + 1] = imu_state_data.omega[1];
        actual_output[3 + mj_model_ptr->nu + 2] = imu_state_data.omega[2];
        // actual_output[3 + 3 + 2 * mj_model_ptr->nu] = imu_state_data.accelerometer[0];
        // actual_output[3 + 3 + 2 * mj_model_ptr->nu + 1] = imu_state_data.accelerometer[1];
        // actual_output[3 + 3 + 2 * mj_model_ptr->nu + 2] = imu_state_data.accelerometer[2];

        imu_accelerometer = Eigen::Vector3d(
          imu_state_data.accelerometer[0],
          imu_state_data.accelerometer[1],
          imu_state_data.accelerometer[2]
        );

      } else {
        actual_output.head<3>() = labrob::rotVecFromQuaternion(robot_state.orientation);
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          const int qpos_adr = mj_model_ptr->jnt_qposadr[joint_id];
          const int dof_adr = mj_model_ptr->jnt_dofadr[joint_id];
          actual_output[3 + i] = mj_data_ptr->qpos[qpos_adr];
          actual_output[3 + mj_model_ptr->nu + i + 3] = mj_data_ptr->qvel[dof_adr];

          auto it = joint_name_to_index.find(joint_name);
          if (it == joint_name_to_index.end()) {
            continue;
          }
          const int idx = it->second;
          if (idx < 0 || idx >= G1_NUM_MOTOR) {
            continue;
          }

          measured_motor_state.q[idx] = static_cast<float>(mj_data_ptr->qpos[qpos_adr]);
          measured_motor_state.dq[idx] = static_cast<float>(mj_data_ptr->qvel[dof_adr]);
          measured_motor_state.ddq[idx] = static_cast<float>(mj_data_ptr->qacc[dof_adr]);
          measured_motor_state.tau_est[idx] = static_cast<float>(
              mj_data_ptr->qfrc_actuator[dof_adr]
          );
        }

        has_measured_motor_state = true;
        motor_state_data = measured_motor_state;
        actual_output[3 + mj_model_ptr->nu] = robot_state.angular_velocity.x();
        actual_output[3 + mj_model_ptr->nu + 1] = robot_state.angular_velocity.y();
        actual_output[3 + mj_model_ptr->nu + 2] = robot_state.angular_velocity.z();
        imu_accelerometer = Eigen::Vector3d::Zero();
      }

      // std::cout << imu_state_data.rpy[0] << " " << imu_state_data.rpy[1] << " " << imu_state_data.rpy[2] << std::endl;
      estimate_force.update(robot_state);

      const Eigen::VectorXd& left_wrist_wrench_filtered = estimate_force.getLeftWristWrenchFiltered();
      const Eigen::VectorXd& right_wrist_wrench_filtered = estimate_force.getRightWristWrenchFiltered();
      const double left_hand_force_x = std::clamp(std::abs(left_wrist_wrench_filtered(0)), hand_force_min_newton, hand_force_max_newton);
      const double right_hand_force_x = std::clamp(std::abs(right_wrist_wrench_filtered(0)), hand_force_min_newton, hand_force_max_newton);
      const double average_hand_force_x = 0.5 * (left_hand_force_x + right_hand_force_x);
      const double normalized_force = (average_hand_force_x - hand_force_min_newton) /
                  (hand_force_max_newton - hand_force_min_newton);
      const bool external_force_detected = (average_hand_force_x >= hand_force_start_threshold_newton);
      const double desired_step_length_x_realtime = step_length_x_min_meter +
             std::clamp(normalized_force, 0.0, 1.0) *
             (step_length_x_max_meter - step_length_x_min_meter);
      if (!walk_started_by_force) {
        if (external_force_detected) {
          latched_step_length_x = std::max(
              desired_step_length_x_realtime,
              step_length_x_start_floor_meter
          );
          switchWalkingState = true;
          walk_started_by_force = true;
        }
      }
      walking_manager.setDesiredStepLengthX(walk_started_by_force ? latched_step_length_x : desired_step_length_x_realtime);
      walking_manager.setDesiredStepCount(walk_started_by_force ? 20 : 0);

      // Update walking manager:
      labrob::JointCommand joint_command;
      // #pragma omp parallel sections num_threads(2)
      // {
      //   #pragma omp section
      //   {
      //     walking_manager.update(robot_state, joint_command, actual_output);
      //   }
      //   #pragma omp section
      //   {
      //   }
      // } // end of parallel sections

      walking_manager.update(robot_state, joint_command, actual_output);

      if (has_measured_motor_state) {
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          auto it = joint_name_to_index.find(joint_name);
          if (it == joint_name_to_index.end()) {
            continue;
          }
          const int idx = it->second;
          if (idx < 0 || idx >= G1_NUM_MOTOR) {
            continue;
          }
          // robot_state.joint_state[joint_name].eff = measured_motor_state.tau_est[idx];
        }
      }

        mju_zero(mj_data_ptr->qfrc_applied, mj_model_ptr->nv);
        mju_zero(mj_data_ptr->xfrc_applied, 6 * mj_model_ptr->nbody);

        const double force_window_start = external_force_start_sec;
        const double force_window_end = external_force_start_sec + external_force_duration_sec;
        const bool external_force_active =
          (mj_data_ptr->time >= force_window_start) &&
          (mj_data_ptr->time <= force_window_end);

        double external_force_scale = 0.0;
        if (external_force_active) {
          if (external_force_ramp_sec > 0.0) {
            const double elapsed = mj_data_ptr->time - force_window_start;
            const double remaining = force_window_end - mj_data_ptr->time;
            const double ramp_up = std::min(1.0, elapsed / external_force_ramp_sec);
            const double ramp_down = std::min(1.0, remaining / external_force_ramp_sec);
            external_force_scale = std::min(ramp_up, ramp_down);
          } else {
            external_force_scale = 1.0;
          }
        }

        const bool left_wrench_enabled = (enable_external_left_force_test || enable_external_left_torque_test);
        const bool right_wrench_enabled = (enable_external_right_force_test || enable_external_right_torque_test);

        if (external_force_active &&
          ((left_wrench_enabled && left_wrist_body_id >= 0) ||
           (right_wrench_enabled && right_wrist_body_id >= 0))) {
        if (left_wrench_enabled && left_wrist_body_id >= 0) {
          const mjtNum force_world_left[3] = {
            static_cast<mjtNum>(external_left_fx_newton * external_force_scale),
            static_cast<mjtNum>(external_left_fy_newton * external_force_scale),
            static_cast<mjtNum>(external_left_fz_newton * external_force_scale)
          };
          const mjtNum torque_world_left[3] = {
            static_cast<mjtNum>(external_left_tx_newton_meter * external_force_scale),
            static_cast<mjtNum>(external_left_ty_newton_meter * external_force_scale),
            static_cast<mjtNum>(external_left_tz_newton_meter * external_force_scale)
          };
          mj_data_ptr->xfrc_applied[6 * left_wrist_body_id + 0] = force_world_left[0];
          mj_data_ptr->xfrc_applied[6 * left_wrist_body_id + 1] = force_world_left[1];
          mj_data_ptr->xfrc_applied[6 * left_wrist_body_id + 2] = force_world_left[2];
          mj_data_ptr->xfrc_applied[6 * left_wrist_body_id + 3] = torque_world_left[0];
          mj_data_ptr->xfrc_applied[6 * left_wrist_body_id + 4] = torque_world_left[1];
          mj_data_ptr->xfrc_applied[6 * left_wrist_body_id + 5] = torque_world_left[2];
        }

        if (right_wrench_enabled && right_wrist_body_id >= 0) {
          const mjtNum force_world_right[3] = {
            static_cast<mjtNum>(external_right_fx_newton * external_force_scale),
            static_cast<mjtNum>(external_right_fy_newton * external_force_scale),
            static_cast<mjtNum>(external_right_fz_newton * external_force_scale)
          };
          const mjtNum torque_world_right[3] = {
            static_cast<mjtNum>(external_right_tx_newton_meter * external_force_scale),
            static_cast<mjtNum>(external_right_ty_newton_meter * external_force_scale),
            static_cast<mjtNum>(external_right_tz_newton_meter * external_force_scale)
          };
          mj_data_ptr->xfrc_applied[6 * right_wrist_body_id + 0] = force_world_right[0];
          mj_data_ptr->xfrc_applied[6 * right_wrist_body_id + 1] = force_world_right[1];
          mj_data_ptr->xfrc_applied[6 * right_wrist_body_id + 2] = force_world_right[2];
          mj_data_ptr->xfrc_applied[6 * right_wrist_body_id + 3] = torque_world_right[0];
          mj_data_ptr->xfrc_applied[6 * right_wrist_body_id + 4] = torque_world_right[1];
          mj_data_ptr->xfrc_applied[6 * right_wrist_body_id + 5] = torque_world_right[2];
        }
      }

      mjtNum left_force_point_world[3] = {0.0, 0.0, 0.0};
      mjtNum left_force_world[3] = {0.0, 0.0, 0.0};
      mjtNum left_torque_world[3] = {0.0, 0.0, 0.0};
      bool left_force_enabled = false;
      if ((enable_external_left_force_test || enable_external_left_torque_test) && left_wrist_body_id >= 0 && external_force_active) {
        left_force_point_world[0] = mj_data_ptr->xpos[3 * left_wrist_body_id + 0];
        left_force_point_world[1] = mj_data_ptr->xpos[3 * left_wrist_body_id + 1];
        left_force_point_world[2] = mj_data_ptr->xpos[3 * left_wrist_body_id + 2];
        left_force_world[0] = static_cast<mjtNum>(external_left_fx_newton * external_force_scale);
        left_force_world[1] = static_cast<mjtNum>(external_left_fy_newton * external_force_scale);
        left_force_world[2] = static_cast<mjtNum>(external_left_fz_newton * external_force_scale);
        left_torque_world[0] = static_cast<mjtNum>(external_left_tx_newton_meter * external_force_scale);
        left_torque_world[1] = static_cast<mjtNum>(external_left_ty_newton_meter * external_force_scale);
        left_torque_world[2] = static_cast<mjtNum>(external_left_tz_newton_meter * external_force_scale);
        left_force_enabled = true;
      }

      mjtNum right_force_point_world[3] = {0.0, 0.0, 0.0};
      mjtNum right_force_world[3] = {0.0, 0.0, 0.0};
      mjtNum right_torque_world[3] = {0.0, 0.0, 0.0};
      bool right_force_enabled = false;
      if ((enable_external_right_force_test || enable_external_right_torque_test) && right_wrist_body_id >= 0 && external_force_active) {
        right_force_point_world[0] = mj_data_ptr->xpos[3 * right_wrist_body_id + 0];
        right_force_point_world[1] = mj_data_ptr->xpos[3 * right_wrist_body_id + 1];
        right_force_point_world[2] = mj_data_ptr->xpos[3 * right_wrist_body_id + 2];
        right_force_world[0] = static_cast<mjtNum>(external_right_fx_newton * external_force_scale);
        right_force_world[1] = static_cast<mjtNum>(external_right_fy_newton * external_force_scale);
        right_force_world[2] = static_cast<mjtNum>(external_right_fz_newton * external_force_scale);
        right_torque_world[0] = static_cast<mjtNum>(external_right_tx_newton_meter * external_force_scale);
        right_torque_world[1] = static_cast<mjtNum>(external_right_ty_newton_meter * external_force_scale);
        right_torque_world[2] = static_cast<mjtNum>(external_right_tz_newton_meter * external_force_scale);
        right_force_enabled = true;
      }

      mujoco_ui.setExternalWristWrenches(
          left_force_point_world,
          left_force_world,
          left_torque_world,
          left_force_enabled,
          right_force_point_world,
          right_force_world,
          right_torque_world,
          right_force_enabled
      );



      Eigen::Vector3d left_force_gt = Eigen::Vector3d::Zero();
      Eigen::Vector3d right_force_gt = Eigen::Vector3d::Zero();
      Eigen::Vector3d left_torque_gt = Eigen::Vector3d::Zero();
      Eigen::Vector3d right_torque_gt = Eigen::Vector3d::Zero();
      if (external_force_active) {
        if (enable_external_left_force_test && left_wrist_body_id >= 0) {
          left_force_gt.x() = external_left_fx_newton * external_force_scale;
          left_force_gt.y() = external_left_fy_newton * external_force_scale;
          left_force_gt.z() = external_left_fz_newton * external_force_scale;
        }
        if (enable_external_right_force_test && right_wrist_body_id >= 0) {
          right_force_gt.x() = external_right_fx_newton * external_force_scale;
          right_force_gt.y() = external_right_fy_newton * external_force_scale;
          right_force_gt.z() = external_right_fz_newton * external_force_scale;
        }
        if (enable_external_left_torque_test && left_wrist_body_id >= 0) {
          left_torque_gt.x() = external_left_tx_newton_meter * external_force_scale;
          left_torque_gt.y() = external_left_ty_newton_meter * external_force_scale;
          left_torque_gt.z() = external_left_tz_newton_meter * external_force_scale;
        }
        if (enable_external_right_torque_test && right_wrist_body_id >= 0) {
          right_torque_gt.x() = external_right_tx_newton_meter * external_force_scale;
          right_torque_gt.y() = external_right_ty_newton_meter * external_force_scale;
          right_torque_gt.z() = external_right_tz_newton_meter * external_force_scale;
        }
      }
      wrist_force_time_log.push_back(mj_data_ptr->time);
      left_wrist_force_gt_log.push_back(left_force_gt);
      right_wrist_force_gt_log.push_back(right_force_gt);
      left_wrist_torque_gt_log.push_back(left_torque_gt);
      right_wrist_torque_gt_log.push_back(right_torque_gt);
        left_wrist_force_point_log.emplace_back(
          left_force_point_world[0],
          left_force_point_world[1],
          left_force_point_world[2]
        );
        right_wrist_force_point_log.emplace_back(
          right_force_point_world[0],
          right_force_point_world[1],
          right_force_point_world[2]
        );
        left_wrist_force_enabled_log.push_back(left_force_enabled ? 1 : 0);
        right_wrist_force_enabled_log.push_back(right_force_enabled ? 1 : 0);


      if (true){
        auto start_integration = std::chrono::steady_clock::now();
        mj_step1(mj_model_ptr, mj_data_ptr);
  
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          mj_data_ptr->ctrl[i] = joint_command[joint_name];
        }
  
        mj_step2(mj_model_ptr, mj_data_ptr);

        auto end_integration = std::chrono::steady_clock::now();
        auto integration_duration = end_integration - start_integration;
        if(integration_duration > std::chrono::milliseconds(1))
          std::cout << "Warning: integration took too long: " << std::chrono::duration_cast<std::chrono::microseconds>(integration_duration).count() << " us" << std::endl;
        robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);

      }else{
        auto start_integration = std::chrono::steady_clock::now();

        robot_state = walking_manager.getNewRobotState(robot_state);
      //   // update mujoco state with robot_state
        mj_data_ptr->qpos[0] = robot_state.position.x();
        mj_data_ptr->qpos[1] = robot_state.position.y();
        mj_data_ptr->qpos[2] = robot_state.position.z();
        mj_data_ptr->qpos[3] = robot_state.orientation.w();
        mj_data_ptr->qpos[4] = robot_state.orientation.x();
        mj_data_ptr->qpos[5] = robot_state.orientation.y();
        mj_data_ptr->qpos[6] = robot_state.orientation.z();
        //rotate the linear velocity from world to body frame
        Eigen::Vector3d lin_vel_body = robot_state.orientation.toRotationMatrix() * robot_state.linear_velocity;
        mj_data_ptr->qvel[0] = lin_vel_body.x();
        mj_data_ptr->qvel[1] = lin_vel_body.y();
        mj_data_ptr->qvel[2] = lin_vel_body.z();
        mj_data_ptr->qvel[3] = robot_state.angular_velocity.x();
        mj_data_ptr->qvel[4] = robot_state.angular_velocity.y();
        mj_data_ptr->qvel[5] = robot_state.angular_velocity.z();
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[joint_id]] = robot_state.joint_state[joint_name].pos;
          mj_data_ptr->qvel[mj_model_ptr->jnt_dofadr[joint_id]] = robot_state.joint_state[joint_name].vel;
        }
      }

      for (int i = 0; i < mj_model_ptr->nu; ++i) {
        int joint_id = mj_model_ptr->actuator_trnid[i * 2];
        std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
        mj_data_ptr->ctrl[i] = joint_command[joint_name];

        // auto end_integration = std::chrono::steady_clock::now();
        // auto integration_duration = end_integration - start_integration;
        // if(integration_duration > std::chrono::milliseconds(1))
        //   std::cout << "Warning: integration took too long: " << std::chrono::duration_cast<std::chrono::microseconds>(integration_duration).count() << " us" << std::endl;

      }
      //Run Dynamics Mujoco:
      //mj_step(mj_model_ptr, mj_data_ptr);

      //robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);

      if (!sdk_lowstate_stream_enabled) {
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          auto it = joint_name_to_index.find(joint_name);
          if (it == joint_name_to_index.end()) {
            continue;
          }
          const int idx = it->second;
          if (idx < 0 || idx >= G1_NUM_MOTOR) {
            continue;
          }

          const int qpos_adr = mj_model_ptr->jnt_qposadr[joint_id];
          const int dof_adr = mj_model_ptr->jnt_dofadr[joint_id];
          measured_motor_state.q[idx] = static_cast<float>(mj_data_ptr->qpos[qpos_adr]);
          measured_motor_state.dq[idx] = static_cast<float>(mj_data_ptr->qvel[dof_adr]);
          measured_motor_state.ddq[idx] = static_cast<float>(mj_data_ptr->qacc[dof_adr]);
          measured_motor_state.tau_est[idx] = static_cast<float>(
              mj_data_ptr->qfrc_actuator[dof_adr] //关节力矩，随控制输入变化
              //mj_data_ptr->qfrc_applied[dof_adr] //外力力矩投影变化
              //mj_data_ptr->qfrc_constraint[dof_adr] //约束力矩，通常在有接触时非零
              //mj_data_ptr->qfrc_passive[dof_adr] //被动力矩，通常与速度相关，如摩擦力矩
          );
        }
        has_measured_motor_state = true;
        motor_state_data = measured_motor_state;
      }

      left_wrist_wrench_log.push_back(estimate_force.getLeftWristWrench());
      right_wrist_wrench_log.push_back(estimate_force.getRightWristWrench());
      left_wrist_wrench_filtered_log.push_back(estimate_force.getLeftWristWrenchFiltered());
      right_wrist_wrench_filtered_log.push_back(estimate_force.getRightWristWrenchFiltered());
      
      if (has_measured_motor_state) {
        std::vector<mjtNum> qfrc_from_xfrc(mj_model_ptr->nv, 0.0);
        auto accumulate_body_xfrc = [&](int body_id) {
          if (body_id < 0) {
            return;
          }
          const mjtNum force_world[3] = {
            mj_data_ptr->xfrc_applied[6 * body_id + 0],
            mj_data_ptr->xfrc_applied[6 * body_id + 1],
            mj_data_ptr->xfrc_applied[6 * body_id + 2]
          };
          const mjtNum torque_world[3] = {
            mj_data_ptr->xfrc_applied[6 * body_id + 3],
            mj_data_ptr->xfrc_applied[6 * body_id + 4],
            mj_data_ptr->xfrc_applied[6 * body_id + 5]
          };
          const mjtNum point_world[3] = {
            mj_data_ptr->xpos[3 * body_id + 0],
            mj_data_ptr->xpos[3 * body_id + 1],
            mj_data_ptr->xpos[3 * body_id + 2]
          };
          mj_applyFT(
            mj_model_ptr,
            mj_data_ptr,
            force_world,
            torque_world,
            point_world,
            body_id,
            qfrc_from_xfrc.data()
          );
        };
        accumulate_body_xfrc(left_wrist_body_id);
        accumulate_body_xfrc(right_wrist_body_id);

        all_joint_motor_time_log.push_back(mj_data_ptr->time);
        auto append_joint_from_motor = [&](const char* joint_name, std::vector<std::array<float, 5>>& log) {
          auto it = joint_name_to_index.find(joint_name);
          if (it == joint_name_to_index.end()) {
            return;
          }
          const int idx = it->second;
          if (idx < 0 || idx >= G1_NUM_MOTOR) {
            return;
          }

          const int joint_id = mj_name2id(mj_model_ptr, mjOBJ_JOINT, joint_name);
          if (joint_id < 0) {
            return;
          }
          const int dof_adr = mj_model_ptr->jnt_dofadr[joint_id];

          log.push_back({
            measured_motor_state.q[idx],
            measured_motor_state.dq[idx],
            measured_motor_state.ddq[idx],
            measured_motor_state.tau_est[idx],
            static_cast<float>(qfrc_from_xfrc[dof_adr])
          });
        };

        for (const auto& [joint_name, idx] : joint_name_to_index) {
          auto& log = all_joint_motor_log[joint_name];
          append_joint_from_motor(joint_name.c_str(), log);
        }
      }

      if (useRobot) {

        MotorCommand motor_command;
        const int num_controlled_joints = std::min(G1_NUM_MOTOR, mj_model_ptr->nu);
      
        // Impostazioni di base
        motor_command.tau_ff.fill(0.0f);
        motor_command.q_target.fill(0.0f);
        motor_command.dq_target.fill(0.0f);

        // impose kp and kd to increase linearly with time
        if (mj_data_ptr->time < 5.0f) {
          for (int i = 0; i < num_controlled_joints; ++i) {
            int joint_id = mj_model_ptr->actuator_trnid[i * 2];
            std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
            motor_command.kp[i] = 0.0f + Kp[i] * (mj_data_ptr->time / 5.0f);
            motor_command.kd[i] = Kd[i];
          }
        }
        else{
          motor_command.kp = Kp;
          motor_command.kd = Kd;
        }

        // assegna i valori di controllo per i giunti
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));

          // if the values are too big in module, turn off the robot
          if (std::abs(robot_state.joint_state[joint_name].pos) > 2 || std::abs(robot_state.joint_state[joint_name].vel) > 15 || std::abs(joint_command[joint_name]) > 100.0)  {
            std::cout << "Warning: motor command values too high for joint " << joint_name << ": "
                      << "q_target = " << robot_state.joint_state[joint_name].pos << ", "
                      << "dq_target = " << robot_state.joint_state[joint_name].vel << ", "
                      << "tau_ff = " << joint_command[joint_name] << std::endl;
            std::cout << "Disabling robot for safety." << std::endl;

            signalHandler(SIGINT);
          }else {
            motor_command.q_target[i] = robot_state.joint_state[joint_name].pos;
            motor_command.dq_target[i] = robot_state.joint_state[joint_name].vel;
            motor_command.tau_ff[i] = joint_command[joint_name];
          }
        }
      
        // Costruisci comando DDS
        LowCmd_ dds_low_command;
        dds_low_command.mode_pr() = static_cast<uint8_t>(Mode::PR);
        dds_low_command.mode_machine() = mode_machine_;
      
        for (int i = 0; i < num_controlled_joints; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id)); 
          int robot_idx = joint_name_to_index[joint_name];
          auto &cmd = dds_low_command.motor_cmd().at(robot_idx);
          cmd.mode() = 1;
          cmd.q()    = motor_command.q_target[i];
          cmd.dq()   = motor_command.dq_target[i];
          cmd.tau()  = motor_command.tau_ff[i];
          cmd.kp()   = motor_command.kp[i];
          cmd.kd()   = motor_command.kd[i];
        }
      
        dds_low_command.crc() = Crc32Core((uint32_t*)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
        lowcmd_publisher->Write(dds_low_command);
      }

      next_tick += std::chrono::milliseconds(2);

      // Calcola quanto dormire
      auto end_sleep = std::chrono::steady_clock::now();
      if ( end_sleep - start_sleep < std::chrono::milliseconds(2)) {
          std::this_thread::sleep_until(next_tick);
      }
      else {
          // std::cout << "Warning: walking manager update took too long: " << std::chrono::duration_cast<std::chrono::microseconds>(end_sleep - start_sleep).count() << " us" << std::endl;
          next_tick = end_sleep;
      }
    }

    auto start_render =  std::chrono::steady_clock::now();

    mujoco_ui.render();

    auto end_render =  std::chrono::steady_clock::now();
    auto render_duration = end_render - start_render;
    if(render_duration > std::chrono::milliseconds(5))
      std::cout << "Warning: rendering took too long: " << std::chrono::duration_cast<std::chrono::microseconds>(render_duration).count() << " us" << std::endl;

  }

  // Free memory (Mujoco):
  mj_deleteData(mj_data_ptr);
  mj_deleteModel(mj_model_ptr);

  return 0;
}

