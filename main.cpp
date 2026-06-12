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

#include <thread>
#include <iomanip>
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
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

#include <hrp4_locomotion/globals.h>
#include "MujocoUI.hpp"

bool isWBCLoopClosed = false;
bool isMPCLoopClosed = false;
bool isEKFactive = false;
bool useSim = false;
bool useRobot = false;
bool oneTimepress = true;
bool xPressed = false;
bool loopClosed = true;
bool switchWalkingState = false;
bool imuCalibration = false;

double startTimeWBCCL = 1000.0;
double startTimeMPCCL = 1000.0;
double startTimeEKF = 0.0;

Eigen::Vector3d odometry_base_position = Eigen::Vector3d::Zero();
Eigen::Vector3d odometry_base_velocity = Eigen::Vector3d::Zero();
Eigen::Vector4d odometry_imu_quaternion = Eigen::Vector4d(1,0,0,0);
Eigen::Vector3d odometry_imu_rpy = Eigen::Vector3d::Zero();

Eigen::VectorXd measured_joint_position = Eigen::VectorXd::Zero(29);
Eigen::VectorXd measured_joint_velocity = Eigen::VectorXd::Zero(29);
Eigen::Vector4d measured_imu_quaternion = Eigen::Vector4d(1,0,0,0);
Eigen::Vector3d measured_imu_rpy = Eigen::Vector3d::Zero();
Eigen::Vector3d measured_imu_angular_velocity = Eigen::Vector3d::Zero();
Eigen::Vector3d measured_imu_accelerometer= Eigen::Vector3d::Zero();
Eigen::Vector4d measured_imu_pelvis_quaternion = Eigen::Vector4d(1,0,0,0);
Eigen::Vector3d measured_imu_pelvis_rpy = Eigen::Vector3d::Zero();
Eigen::Vector3d measured_imu_pelvis_angular_velocity = Eigen::Vector3d::Zero();
Eigen::Vector3d measured_imu_pelvis_accelerometer= Eigen::Vector3d::Zero();

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace unitree_go::msg::dds_;
using namespace unitree::common;
using namespace unitree::robot::b2;

Gamepad gamepad_;
REMOTE_DATA_RX rx_;

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string GO_STATE_TOPIC = "rt/odommodestate";
labrob::WalkingManager walking_manager;

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
  Eigen::Vector3d rpy = {};
  Eigen::Vector3d omega = {};
  Eigen::Vector3d accelerometer = {};
  Eigen::Vector4d quaternion = {};
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
struct SportModeState {
  std::array<float, 3> position = {};
  std::array<float, 3> velocity = {};
  ImuState imu_state = {};
};


// Stiffness for all G1 Joints
// const std::array<float, G1_NUM_MOTOR> Kp_reg{
//   700, 700, 700, 1000, 900, 600,      // legs sx
//   700, 700, 700, 1000, 900, 600,      // legs dx
//   400, 400, 400,                  // waist
//   400, 400, 400, 400,  200, 200, 200,  // arms sx
//   400, 400, 400, 400,  200, 200, 200   // arms dx
// };

// // Damping for all G1 Joints
// const std::array<float, G1_NUM_MOTOR> Kd_reg{
//   15, 15, 15, 15, 15, 15,     // legs sx
//   15, 15, 15, 15, 15, 15,     // legs dx
//   15, 15, 15,         // waist
//   15, 15, 15, 15, 15, 15, 15,  // arms sx
//   15, 15, 15, 15, 15, 15, 15   // arms dx
// };

const std::array<float, G1_NUM_MOTOR> Kp_cl{
    400, 400, 400, 700, 400, 300,      // legs
    400, 400, 400, 700, 400, 300,      // legs
    400, 400, 150,                   // waist
    200, 200, 120, 70,  40, 40, 40,  // arms
    200, 200, 120, 70,  40, 40, 40   // arms
};

// Damping for all G1 Joints
const std::array<float, G1_NUM_MOTOR> Kd_cl{
    2, 2, 2, 3, 2, 2,     // legs
    2, 2, 2, 3, 2, 2,     // legs
    2, 2, 2,              // waist
    2, 2, 2, 2, 2, 2, 2,  // arms
    2, 2, 2, 2, 2, 2, 2   // arms
};

const std::array<float, G1_NUM_MOTOR> Kp_reg{
    400, 400, 400, 600, 400, 300,      // legs
    400, 400, 400, 600, 400, 300,      // legs
    250, 250, 150,                   // waist
    120, 120, 120, 70,  40, 40, 40,  // arms
    120, 120, 120, 70,  40, 40, 40   // arms
};

// Damping for all G1 Joints
const std::array<float, G1_NUM_MOTOR> Kd_reg{
    2, 2, 2, 3, 2, 2,     // legs
    2, 2, 2, 3, 2, 2,     // legs
    2, 2, 2,              // waist
    2, 2, 2, 2, 2, 2, 2,  // arms
    2, 2, 2, 2, 2, 2, 2   // arms
};



//assign at each value of kd twice the square root of the corresponding kp value
// std::array<float, G1_NUM_MOTOR> Kd = {
//   2*sqrt(Kp[0]), 2*sqrt(Kp[1]), 2*sqrt(Kp[2]), 2*sqrt(Kp[3]), 2*sqrt(Kp[4]), 2*sqrt(Kp[5]), // legs sx
//   2*sqrt(Kp[6]), 2*sqrt(Kp[7]), 2*sqrt(Kp[8]), 2*sqrt(Kp[9]), 2*sqrt(Kp[10]), 2*sqrt(Kp[11]), // legs dx
//   2*sqrt(Kp[12]), 2*sqrt(Kp[13]), 2*sqrt(Kp[14]), // waist
//   2*sqrt(Kp[15]), 2*sqrt(Kp[16]), 2*sqrt(Kp[17]), 2*sqrt(Kp[18]), 2*sqrt(Kp[19]), 2*sqrt(Kp[20]), 2*sqrt(Kp[21]), // arms sx
//   2*sqrt(Kp[22]), 2*sqrt(Kp[23]), 2*sqrt(Kp[24]), 2*sqrt(Kp[25]), 2*sqrt(Kp[26]), 2*sqrt(Kp[27]), 2*sqrt(Kp[28]) // arms dx
// };

//TODO is it better to use DataBuffers?
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
  {"waist_roll_joint", 13},        // NOTE INVALID for g1 23dof/29dof with waist locked
  {"waist_pitch_joint", 14},      // NOTE INVALID for g1 23dof/29dof with waist locked
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

// create a map assigning each joint name to its initial position 
// recall that pitch and yaw should be the same, while roll should be opposite for left and right joints
std::map<std::string, mjtNum> joint_initial_positions = {
  {"left_hip_pitch_joint", -0.44},
  {"left_hip_roll_joint", 0.04}, 
  {"left_hip_yaw_joint", 0.0},
  {"left_knee_joint", 0.95},
  {"left_ankle_pitch_joint", -0.50},
  {"left_ankle_roll_joint", 0.0},
  {"right_hip_pitch_joint", -0.44},
  {"right_hip_roll_joint", -0.04},
  {"right_hip_yaw_joint", 0.0},
  {"right_knee_joint", 0.95},
  {"right_ankle_pitch_joint", -0.50},
  {"right_ankle_roll_joint", 0.0},
  {"waist_yaw_joint", 0.0},
  {"waist_roll_joint", 0.0},
  {"waist_pitch_joint", 0.0},
  {"left_shoulder_pitch_joint", 0.07},
  {"left_shoulder_roll_joint", 0.25},
  {"left_shoulder_yaw_joint", 0.0},
  {"left_elbow_joint", 1.13},
  {"left_wrist_roll_joint", 0.0},
  {"left_wrist_pitch_joint", 0.0},
  {"left_wrist_yaw_joint", 0.0},
  {"right_shoulder_pitch_joint", 0.07},
  {"right_shoulder_roll_joint", -0.25},
  {"right_shoulder_yaw_joint", 0.0},
  {"right_elbow_joint", 1.13},
  {"right_wrist_roll_joint", 0.0},
  {"right_wrist_pitch_joint", 0.0},
  {"right_wrist_yaw_joint", 0.0},
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

MotorState motor_state_data;
ImuState imu_pelvis_data;
void LowStateHandler(const void* msg){
  LowState_ low_state = *(const LowState_*)msg;
  uint32_t crc_calc = Crc32Core((uint32_t*)&low_state, ((sizeof(LowState_) >> 2) -1));

  if (low_state.crc() != crc_calc) {
    std::cerr << "CRC32 mismatch in LowState message!" << std::endl;
    return;
  }

  std::lock_guard<std::mutex> lock(stateMutex);
  for (int i = 0; i < G1_NUM_MOTOR; ++i) {
    motor_state_data.q[i] = low_state.motor_state()[i].q();
    motor_state_data.dq[i] = low_state.motor_state()[i].dq();
  }

  imu_pelvis_data.quaternion = Eigen::Vector4d(low_state.imu_state().quaternion()[0],
                                      low_state.imu_state().quaternion()[1],
                                      low_state.imu_state().quaternion()[2],
                                      low_state.imu_state().quaternion()[3]);
  imu_pelvis_data.rpy = Eigen::Vector3d(low_state.imu_state().rpy()[0],
                              low_state.imu_state().rpy()[1],
                              low_state.imu_state().rpy()[2]);
  imu_pelvis_data.omega = Eigen::Vector3d(low_state.imu_state().gyroscope()[0],
                                  low_state.imu_state().gyroscope()[1],
                                  low_state.imu_state().gyroscope()[2]);
  imu_pelvis_data.accelerometer = Eigen::Vector3d(low_state.imu_state().accelerometer()[0],
                                          low_state.imu_state().accelerometer()[1],
                                          low_state.imu_state().accelerometer()[2]);

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

ImuState imu_torso_data;
void imuTorsoHandler(const void* msg) {
  unitree_hg::msg::dds_::IMUState_ imu_state = *(const unitree_hg::msg::dds_::IMUState_*)msg;

  std::lock_guard<std::mutex> lock(stateMutex);
  imu_torso_data.quaternion = imu_torso_data.quaternion;
  imu_torso_data.rpy = imu_torso_data.rpy;
  imu_torso_data.omega = imu_torso_data.omega;
  imu_torso_data.accelerometer = imu_torso_data.accelerometer;
}

SportModeState odometry_data;
void SportModeStateHandler(const void* msg){
  SportModeState_ sportmodestate = *(const SportModeState_*)msg;

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

  odometry_data.imu_state.omega[0] = sportmodestate.imu_state().gyroscope()[0];
  odometry_data.imu_state.omega[1] = sportmodestate.imu_state().gyroscope()[1];
  odometry_data.imu_state.omega[2] = sportmodestate.imu_state().gyroscope()[2];

  odometry_data.imu_state.accelerometer[0] = sportmodestate.imu_state().accelerometer()[0];
  odometry_data.imu_state.accelerometer[1] = sportmodestate.imu_state().accelerometer()[1];
  odometry_data.imu_state.accelerometer[2] = sportmodestate.imu_state().accelerometer()[2];
  
  odometry_data.imu_state.rpy[0] = sportmodestate.imu_state().rpy()[0];
  odometry_data.imu_state.rpy[1] = sportmodestate.imu_state().rpy()[1];
  odometry_data.imu_state.rpy[2] = sportmodestate.imu_state().rpy()[2];
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

void signalHandler(int signum) {
  std::cerr << "Received signal " << signum << ", exiting..." << std::endl;

  std::cout << "Exiting simulation loop." << std::endl;
  std::cout << "Do you want to save logs? [y/n]" << std::endl;
  std::string user_input;
  std::getline(std::cin, user_input);
  if(user_input == "y" || user_input == "Y" || user_input == "yes" || user_input == "Yes" || user_input == "YES"){
    std::cout << "Saving logs..." << std::endl;
    walking_manager.saveLogs();
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
      if (std::filesystem::exists("/tmp/mpc_data")) {
        std::filesystem::path destination = std::filesystem::path(experiment_folder) / "mpc_data";
        std::filesystem::copy("/tmp/mpc_data", destination, std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing);
      }
      // create a README file in the experiment folder
      std::ofstream readme_file(experiment_folder + "/README.txt");
      if (readme_file.is_open()) {
        readme_file << "This folder contains the results of the experiment.\n";
        readme_file << "The gains used for the experiment are:\n";
        readme_file << "Kp: ";
        for (const auto& kp : Kp_cl) {
          readme_file << kp << " ";
        }
        readme_file << "\nKd: ";
        for (const auto& kd : Kd_cl) {
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

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--sim") {
        useSim = true;
    } else if (a == "--robot" && i + 1 < argc) {
        useRobot = true;
        useSim = true;
        netInterface = argv[++i];
    }
  }

  if(!useRobot && !useSim) {
    std::cerr << "Please specify either --sim or --robot <network_interface>" << std::endl;
    return -1;
  }

 signal(SIGINT, signalHandler);

  // Load MJCF (for Mujoco):
  const int kErrorLength = 1024;          // load error string length
  char loadError[kErrorLength] = "";
  const char* mjcf_filepath = robotScenePath.data();
  mjModel* mj_model_ptr = mj_loadXML(mjcf_filepath, nullptr, loadError, kErrorLength);
  mjData* mj_data_ptr = mj_makeData(mj_model_ptr);

  if (useRobot) {
    std::cout << "Press 'Y' on the GAMEPAD to end the program." << std::endl;
    std::cout << "Press 'A' on the GAMEPAD to start EKF." << std::endl;
    std::cout << "Press 'X' on the GAMEPAD to start Closed Loop on Whole Body." << std::endl;
    std::cout << "Press 'B' on the GAMEPAD to switch walking state." << std::endl;
    std::cout << "If GAMEPAD is not used, select now which loops to close:" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "1. MPC (starting in 20 seconds)" << std::endl;
    std::cout << "2. WBC (starting in 15 seconds)" << std::endl;
    std::cout << "You can select multiple options by entering their numbers separated by spaces (e.g., '1 2' for MPC and WBC)." << std::endl;
    std::cout << "Enter your choice: " << std::endl;
    std::string user_input;
    std::getline(std::cin, user_input);
    std::istringstream iss(user_input);
    std::string token;
    while (iss >> token) {
      if (token == "1") {
        isMPCLoopClosed = true;
        startTimeMPCCL = 20000;
      } else if (token == "2") {
        isWBCLoopClosed = true;
        startTimeWBCCL = 15000;
      } 
    }
  } else {
    isWBCLoopClosed = true;
    isMPCLoopClosed = true;
    isEKFactive = false;
  }

  ChannelPublisherPtr<LowCmd_> lowcmd_publisher;
  ChannelSubscriberPtr<LowState_> lowstate_subscriber;
  ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_> imutorso_subscriber;
  ChannelSubscriberPtr<SportModeState_> sportmodestate_subscriber;
  std::shared_ptr<MotionSwitcherClient> msc;

  if(useRobot) {
    std::cout << "Using robot with network interface: " << netInterface << std::endl;
    ChannelFactory::Instance()->Init(0, netInterface);
    std::cout << "ChannelFactory initialized with interface: " << netInterface << std::endl;

    msc.reset(new MotionSwitcherClient());
    msc->SetTimeout(5.0f);
    msc->Init();
    std::cout << "ciao" << std::endl;

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
    imutorso_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::IMUState_>(HG_IMU_TORSO));
    imutorso_subscriber->InitChannel(std::bind(&imuTorsoHandler, std::placeholders::_1), 1);
    sportmodestate_subscriber.reset(new ChannelSubscriber<SportModeState_>(GO_STATE_TOPIC));
    sportmodestate_subscriber->InitChannel(std::bind(&SportModeStateHandler, std::placeholders::_1), 1);
  }

  for (int i = 0; i < mj_model_ptr->nq; ++i) {
    mj_data_ptr->qpos[i] = 0.0;
  }
  mj_data_ptr->qpos[2] = 0.728112;
  mj_data_ptr->qpos[3] = 1;

  for (int i = 0; i < mj_model_ptr->njnt; ++i) {
    const char* name = mj_id2name(mj_model_ptr, mjOBJ_JOINT, i);
    if (joint_initial_positions.find(name) != joint_initial_positions.end()) {
      int qpos_addr = mj_model_ptr->jnt_qposadr[i];
      mj_data_ptr->qpos[qpos_addr] = joint_initial_positions[name];
    }
  }

  std::map<std::string, double> armatures;
  for (int i = 0; i < mj_model_ptr->nu; ++i) {
    int joint_id = mj_model_ptr->actuator_trnid[i * 2];
    std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
    int dof_id = mj_model_ptr->jnt_dofadr[joint_id];
    armatures[joint_name] = mj_model_ptr->dof_armature[dof_id];
  }

  // Walking Manager:
  labrob::RobotState robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
  walking_manager.enableJISMPC(false);
  walking_manager.init(robot_state, armatures);

  auto& mujoco_ui = *labrob::MujocoUI::getInstance(mj_model_ptr, mj_data_ptr);

  static int framerate = 60.0;

  auto next_tick = std::chrono::steady_clock::now();

  // Simulation loop:
  while (!mujoco_ui.windowShouldClose()) {

    mjtNum simstart = mj_data_ptr->time;
    while( mj_data_ptr->time - simstart < 1.0/framerate ) {

      auto start_sleep = std::chrono::steady_clock::now();

      // if userobot is true, update the robot state from the real robot
      if (useRobot) {

        if (gamepad_.Y.pressed) {
          std::cout << "[GAMEPAD] Y pressed -> Deactivating motors..." << std::endl;
          signalHandler(SIGINT);
        }

        if (gamepad_.X.pressed) {
          if (isEKFactive && !xPressed){
            xPressed = true;
            loopClosed = true;
            isMPCLoopClosed = !isMPCLoopClosed;
            isWBCLoopClosed = !isWBCLoopClosed;
            startTimeMPCCL = 1000 * mj_data_ptr->time + 5000;
            startTimeWBCCL = 1000 * mj_data_ptr->time;
            if(isMPCLoopClosed)
              std::cout << "[GAMEPAD] X pressed -> Closed loop activated." << std::endl;
            else
              std::cout << "[GAMEPAD] X pressed -> Closed loop deactivated." << std::endl;
          } else if (!isEKFactive){
              std::cout << "[GAMEPAD] X pressed -> Closed loop did not activate. Activate EKF first" << std::endl;
          }
        } else {
          xPressed = false;
        }

        if (gamepad_.B.on_press) {
          switchWalkingState = true;
          std::cout << "[GAMEPAD] B pressed -> Walking state switched." << std::endl;
        }

        if (gamepad_.A.pressed) {
          if(oneTimepress && !imuCalibration){
            std::cout << "[GAMEPAD] A pressed -> EKF started." << std::endl;
            isEKFactive = true;
            oneTimepress = false;
            startTimeEKF = 1000 * mj_data_ptr->time;
          } else if(imuCalibration){
            std::cout << "[GAMEPAD] A pressed -> EKF not started. IMU not calibrated" << std::endl;
          }
        }

        std::lock_guard<std::mutex> lock(stateMutex);
        measured_imu_accelerometer = Eigen::Vector3d(imu_pelvis_data.accelerometer[0], imu_pelvis_data.accelerometer[1], imu_pelvis_data.accelerometer[2]);
        measured_imu_angular_velocity = Eigen::Vector3d(imu_pelvis_data.omega[0], imu_pelvis_data.omega[1], imu_pelvis_data.omega[2]);
        measured_imu_quaternion = Eigen::Vector4d(imu_torso_data.quaternion[0], imu_torso_data.quaternion[1], imu_torso_data.quaternion[2], imu_torso_data.quaternion[3]);
        measured_imu_rpy = Eigen::Vector3d(imu_torso_data.rpy[0], imu_torso_data.rpy[1], imu_torso_data.rpy[2]);
        measured_imu_pelvis_accelerometer = Eigen::Vector3d(imu_pelvis_data.accelerometer[0], imu_pelvis_data.accelerometer[1], imu_pelvis_data.accelerometer[2]);
        measured_imu_pelvis_angular_velocity = Eigen::Vector3d(imu_pelvis_data.omega[0], imu_pelvis_data.omega[1], imu_pelvis_data.omega[2]);
        measured_imu_pelvis_quaternion = Eigen::Vector4d(imu_pelvis_data.quaternion[0], imu_pelvis_data.quaternion[1], imu_pelvis_data.quaternion[2], imu_pelvis_data.quaternion[3]);
        measured_imu_pelvis_rpy = Eigen::Vector3d(imu_pelvis_data.rpy[0], imu_pelvis_data.rpy[1], imu_pelvis_data.rpy[2]);
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          measured_joint_position[i] = motor_state_data.q[i];
          measured_joint_velocity[i] = motor_state_data.dq[i];
        }
        odometry_base_position = Eigen::Vector3d(
          odometry_data.position[0],
          odometry_data.position[1],
          odometry_data.position[2] + 0.061273
        );
        odometry_base_velocity = Eigen::Vector3d(odometry_data.velocity[0], odometry_data.velocity[1], odometry_data.velocity[2]);
        odometry_imu_quaternion = Eigen::Vector4d(odometry_data.imu_state.quaternion[0], odometry_data.imu_state.quaternion[1], odometry_data.imu_state.quaternion[2], odometry_data.imu_state.quaternion[3]);
        odometry_imu_rpy = Eigen::Vector3d(odometry_data.imu_state.rpy[0], odometry_data.imu_state.rpy[1], odometry_data.imu_state.rpy[2]);
      }
      
      // Update walking manager:
      labrob::JointCommand joint_command;
      walking_manager.update(robot_state, joint_command);

      if (!useRobot){
        auto start_integration = std::chrono::steady_clock::now();

        mj_step1(mj_model_ptr, mj_data_ptr);

        // ad disturbance to robot
        double point[3]{0.0, 0.0, 0.0};
        double force[3]{0.0, 0.0, -100.0};
        double torque[3]{0.0, 0.0, 0.0};
        // std::cout << "Time: " << mj_data_ptr->time << std::k;
        // cast in double to avoid precision issues with the comparison
        if (std::abs(mj_data_ptr->time - 5.0) < 1e-6) {
          int torso_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "torso_link");
          mj_applyFT(mj_model_ptr, mj_data_ptr, force, torque, point, torso_id, mj_data_ptr->qfrc_applied);
          std::cout << "Disturbance applied at time: " << mj_data_ptr->time << std::endl;
        }
        //remove force on button pressed
        if (mj_data_ptr->time > 5.2) {
          // mujoco_ui.getMousePosition(point[0], point[1]);
          mju_zero(mj_data_ptr->qfrc_applied, mj_model_ptr->nv);
        }
  
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          mj_data_ptr->ctrl[i] = joint_command[joint_name];
        }
  
        mj_step2(mj_model_ptr, mj_data_ptr);

        auto end_integration = std::chrono::steady_clock::now();
        auto integration_duration = end_integration - start_integration;
        if(integration_duration > std::chrono::milliseconds(1))
          std::cout << "Warning: mujoco render took too long: " << std::chrono::duration_cast<std::chrono::microseconds>(integration_duration).count() << " us" << std::endl;
        robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);

        // read sensors from mujoco and save them
        
        int acc_id  = mj_name2id(mj_model_ptr, mjOBJ_SENSOR, "imu-pelvis-linear-acceleration");
        int gyro_id = mj_name2id(mj_model_ptr, mjOBJ_SENSOR, "imu-pelvis-angular-velocity");

        // measured_imu_accelerometer = Eigen::Vector3d(mj_data_ptr->sensordata[acc_id], mj_data_ptr->sensordata[acc_id + 1], mj_data_ptr->sensordata[acc_id + 2]);
        // measured_imu_angular_velocity = Eigen::Vector3d(mj_data_ptr->sensordata[gyro_id], mj_data_ptr->sensordata[gyro_id + 1], mj_data_ptr->sensordata[gyro_id + 2]);

        int adr = mj_model_ptr->sensor_adr[acc_id];
        measured_imu_accelerometer =
            Eigen::Vector3d(
                mj_data_ptr->sensordata[adr],
                mj_data_ptr->sensordata[adr + 1],
                mj_data_ptr->sensordata[adr + 2]
            );
        
        adr = mj_model_ptr->sensor_adr[gyro_id];
        measured_imu_angular_velocity =
            Eigen::Vector3d(
                mj_data_ptr->sensordata[adr],
                mj_data_ptr->sensordata[adr + 1],
                mj_data_ptr->sensordata[adr + 2]
            );
      } else {
        auto start_integration = std::chrono::steady_clock::now();
        robot_state = walking_manager.getNewRobotState();
        labrob::RobotState fb_robot_state = walking_manager.getActualRobotState();
        // update mujoco state with robot_state
        mj_data_ptr->qpos[0] = fb_robot_state.position.x();
        mj_data_ptr->qpos[1] = fb_robot_state.position.y();
        mj_data_ptr->qpos[2] = fb_robot_state.position.z();
        mj_data_ptr->qpos[3] = fb_robot_state.orientation.w();
        mj_data_ptr->qpos[4] = fb_robot_state.orientation.x();
        mj_data_ptr->qpos[5] = fb_robot_state.orientation.y();
        mj_data_ptr->qpos[6] = fb_robot_state.orientation.z();
        //rotate the linear velocity from world to body frame
        Eigen::Vector3d lin_vel_body = fb_robot_state.orientation.toRotationMatrix() * fb_robot_state.linear_velocity;
        mj_data_ptr->qvel[0] = lin_vel_body.x();
        mj_data_ptr->qvel[1] = lin_vel_body.y();
        mj_data_ptr->qvel[2] = lin_vel_body.z();
        mj_data_ptr->qvel[3] = fb_robot_state.angular_velocity.x();
        mj_data_ptr->qvel[4] = fb_robot_state.angular_velocity.y();
        mj_data_ptr->qvel[5] = fb_robot_state.angular_velocity.z();
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[joint_id]] = fb_robot_state.joint_state[joint_name].pos;
          mj_data_ptr->qvel[mj_model_ptr->jnt_dofadr[joint_id]] = fb_robot_state.joint_state[joint_name].vel;
        }
        mj_forward(mj_model_ptr, mj_data_ptr);

        mju_zero(mj_data_ptr->ctrl, mj_model_ptr->nu);
        mju_zero(mj_data_ptr->qfrc_applied, mj_model_ptr->nv);
        mju_zero(mj_data_ptr->qacc, mj_model_ptr->nv);
        mju_zero(mj_data_ptr->act, mj_model_ptr->nu);

        mj_data_ptr->time += 0.002;

        auto end_integration = std::chrono::steady_clock::now();
        auto integration_duration = end_integration - start_integration;
        if(integration_duration > std::chrono::milliseconds(1))
          std::cout << "Warning: integration took too long: " << std::chrono::duration_cast<std::chrono::microseconds>(integration_duration).count() << " us" << std::endl;

      }

      if (useRobot) {

        MotorCommand motor_command;
      
        // Impostazioni di base
        motor_command.tau_ff.fill(0.0f);
        motor_command.q_target.fill(0.0f);
        motor_command.dq_target.fill(0.0f);

        // impose kp and kd to increase linearly with time
        if (mj_data_ptr->time < 5.0f) {
          for (int i = 0; i < G1_NUM_MOTOR; ++i) {
            int joint_id = mj_model_ptr->actuator_trnid[i * 2];
            std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
            motor_command.kp[i] = 0.0f + Kp_reg[i] * (mj_data_ptr->time / 5.0f);
            motor_command.kd[i] = Kd_reg[i];
          }
        }
        else{
          motor_command.kp = Kp_reg;
          motor_command.kd = Kd_reg;
          if (isWBCLoopClosed && mj_data_ptr->time >= startTimeWBCCL / 1000.0f) {
            motor_command.kp = Kp_cl;
            motor_command.kd = Kd_cl;    
          }
        }

        // assegna i valori di controllo per i giunti
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));

          // if the values are too big in module, turn off the robot
          if (std::abs(robot_state.joint_state[joint_name].pos) > 1.3 || std::abs(robot_state.joint_state[joint_name].vel) > 3 || std::abs(joint_command[joint_name]) > 105.0)  {
            std::cout << "Warning: motor command values too high for joint " << joint_name << ": "
                      << "q_target = " << robot_state.joint_state[joint_name].pos << ", "
                      << "dq_target = " << robot_state.joint_state[joint_name].vel << ", "
                      << "tau_ff = " << joint_command[joint_name] << std::endl;
            std::cout << "Disabling robot for safety." << std::endl;

            signalHandler(SIGINT);
          }else {
            motor_command.q_target[i] = robot_state.joint_state[joint_name].pos;
            motor_command.dq_target[i] = robot_state.joint_state[joint_name].vel;
            
            // std::cout << "Joint: " << joint_name << ", q_target: " << motor_command.q_target[i] << ", dq_target: " << motor_command.dq_target[i] << ", tau_ff: " << joint_command[joint_name] << std::endl;

            if (isWBCLoopClosed && mj_data_ptr->time >= startTimeWBCCL / 1000.0f) {
              motor_command.tau_ff[i] = joint_command[joint_name];

            } else {
              motor_command.tau_ff[i] = 0.0;
            }
          }
        }
      
        // Costruisci comando DDS
        LowCmd_ dds_low_command;
        dds_low_command.mode_pr() = static_cast<uint8_t>(Mode::PR);
        dds_low_command.mode_machine() = mode_machine_;
      
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
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

