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

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include "MujocoUI.hpp"

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

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


// Stiffness for all G1 Joints
// std::array<float, G1_NUM_MOTOR> Kp{
//   60, 60, 60, 100, 40, 40,      // legs
//   60, 60, 60, 100, 40, 40,      // legs
//   60, 40, 40,                   // waist
//   40, 40, 40, 40,  40, 40, 40,  // arms
//   40, 40, 40, 40,  40, 40, 40   // arms
// };

std::array<float, G1_NUM_MOTOR> Kp{
  0, 0, 0, 0, 0, 0,      // legs
  0, 0, 0, 0, 0, 0,     // legs
  0, 0, 0,                  // waist
  0, 0, 0, 0, 0, 0, 0,  // arms
  0, 0, 0, 5,  0, 0, 0  // arms
};

// Damping for all G1 Joints
std::array<float, G1_NUM_MOTOR> Kd{
  1, 1, 1, 2, 1, 1,     // legs
  1, 1, 1, 2, 1, 1,     // legs
  1, 1, 1,              // waist
  1, 1, 1, 1, 1, 1, 1,  // arms
  1, 1, 1, 1, 1, 1, 1   // arms
};

std::mutex stateMutex;
DataBuffer<MotorState> motor_state_buffer_;
DataBuffer<MotorCommand> motor_command_buffer_;
std::atomic<int> counter_{0};

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

inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < G1_NUM_MOTOR; bits++) {
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

void LowStateHandler(const void* msg){
  LowState_ low_state = *(const LowState_*)msg;
  uint32_t crc_calc = Crc32Core((uint32_t*)&low_state, ((sizeof(LowState_) >> 2) -1));

  if (low_state.crc() != crc_calc) {
    std::cerr << "CRC32 mismatch in LowState message!" << std::endl;
    return;
  }

  MotorState ms;
  for (int i = 0; i < G1_NUM_MOTOR; ++i) {
    ms.q[i] = low_state.motor_state()[i].q();
    ms.dq[i] = low_state.motor_state()[i].dq();
  }

  motor_state_buffer_.SetData(ms);  // thread-safe update
}

void imuTorsoHandler(const void *message) {
  IMUState_ imu_torso = *(const IMUState_ *)message;
  const auto &rpy = imu_torso.rpy();

  if (++counter_ % 500 == 0)
    printf("IMU.torso.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);
}

void signalHandler(int signum) {
  std::cerr << "Received signal " << signum << ", exiting..." << std::endl;
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

Eigen::MatrixXd convert_matrix_mujoco_to_eigen(mjtNum *matrix, int num_rows, int num_cols) {
  Eigen::MatrixXd result(num_rows, num_cols);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      result(i, j) = matrix[i * num_cols + j];
    }
  }
  return result;
}

int main(const int argc, const char* argv[]) {

  bool useRobot = false;
  bool useSim = false;
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
  const char* mjcf_filepath = "../g1_mj_description/stair_steps.xml";
  mjModel* mj_model_ptr = mj_loadXML(mjcf_filepath, nullptr, loadError, kErrorLength);
  if (!mj_model_ptr) {
    std::cerr << "Error loading model: " << loadError << std::endl;
    return -1;
  }
  mjData* mj_data_ptr = mj_makeData(mj_model_ptr);

  std::ofstream joint_pos_log_file("/tmp/joint_pos.txt");
  std::ofstream joint_vel_log_file("/tmp/joint_vel.txt");
  std::ofstream joint_eff_log_file("/tmp/joint_eff.txt");
  std::ofstream joint_names_log_file("/tmp/joint_names.txt");

  // Init robot posture:
  mjtNum waist_p_init = 0.0;
  mjtNum waist_y_init = 0.0;
  mjtNum waist_r_init = 0.0;
  mjtNum r_hip_y_init = 0.0;
  mjtNum r_hip_r_init = -0.05;
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
  mjtNum r_shoulder_r_init = -0.14;
  mjtNum r_shoulder_y_init = 0.0;
  mjtNum r_elbow_p_init = 3.14 / 2.0 - 0.44;
  mjtNum l_shoulder_p_init = r_shoulder_p_init;
  mjtNum l_shoulder_r_init = -r_shoulder_r_init;
  mjtNum l_shoulder_y_init = 0.0;
  mjtNum l_elbow_p_init = r_elbow_p_init;

  for (int i = 0; i < mj_model_ptr->nq; ++i) {
    mj_data_ptr->qpos[i] = 0.0;
  }

  mj_data_ptr->qpos[2] = 0.792151-0.125+0.0263 - 0.071;
  mj_data_ptr->qpos[3] = 1.0;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "waist_pitch_joint")]] = waist_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "waist_yaw_joint")]] = waist_y_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "waist_roll_joint")]] = waist_r_init;
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
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "right_elbow_joint")]] = r_elbow_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_shoulder_pitch_joint")]] = l_shoulder_p_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_shoulder_roll_joint")]] = l_shoulder_r_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_shoulder_yaw_joint")]] = l_shoulder_y_init;
  mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[mj_name2id(mj_model_ptr, mjOBJ_JOINT, "left_elbow_joint")]] = l_elbow_p_init;

  mjtNum* qpos0 = (mjtNum*) malloc(sizeof(mjtNum) * mj_model_ptr->nq);
  memcpy(qpos0, mj_data_ptr->qpos, mj_model_ptr->nq * sizeof(mjtNum));

  std::map<std::string, double> armatures;
  for (int i = 0; i < mj_model_ptr->nu; ++i) {
    int joint_id = mj_model_ptr->actuator_trnid[i * 2];
    std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
    int dof_id = mj_model_ptr->jnt_dofadr[joint_id];
    armatures[joint_name] = mj_model_ptr->dof_armature[dof_id];
  }

  // Walking Manager:
  labrob::RobotState initial_robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
  labrob::WalkingManager walking_manager;
  walking_manager.init(initial_robot_state, armatures);

  auto& mujoco_ui = *labrob::MujocoUI::getInstance(mj_model_ptr, mj_data_ptr);

  for (int i = 0; i < mj_model_ptr->nu; ++i) {
    int joint_id = mj_model_ptr->actuator_trnid[i * 2];
    std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
    joint_names_log_file << joint_name << std::endl;
  }

  joint_names_log_file.flush();
  joint_names_log_file.close();

  static int framerate = 60.0;

  ChannelPublisherPtr<LowCmd_> lowcmd_publisher;
  ChannelSubscriberPtr<LowState_> lowstate_subscriber;
  ChannelSubscriberPtr<IMUState_> imutorso_subscriber;

  if(useRobot) {
    std::cout << "Using robot with network interface: " << netInterface << std::endl;
    ChannelFactory::Instance()->Init(0, netInterface);
    lowcmd_publisher.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher->InitChannel();
    lowstate_subscriber.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber->InitChannel(std::bind(&LowStateHandler, std::placeholders::_1), 1);
    imutorso_subscriber.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
    imutorso_subscriber->InitChannel(std::bind(&imuTorsoHandler, std::placeholders::_1), 1);
  }

  // Simulation loop:
  while (!mujoco_ui.windowShouldClose()) {

    auto start_time = std::chrono::high_resolution_clock::now();

    mjtNum simstart = mj_data_ptr->time;
    while( mj_data_ptr->time - simstart < 1.0/framerate ) {

      labrob::RobotState robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);

      // Update walking manager:
      labrob::JointCommand joint_command;
      // measure the time taken by the walking manager update
      auto update_start = std::chrono::high_resolution_clock::now();
      walking_manager.update(robot_state, joint_command);
      auto update_end = std::chrono::high_resolution_clock::now();
      auto update_duration = std::chrono::duration_cast<std::chrono::microseconds>(update_end - update_start).count();
      std::cout << "WalkingManager::update() took " << update_duration << " us" << std::endl;
      
      mj_step1(mj_model_ptr, mj_data_ptr);

      for (int i = 0; i < mj_model_ptr->nu; ++i) {
        int joint_id = mj_model_ptr->actuator_trnid[i * 2];
        std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
        int jnt_qvel_idx = mj_model_ptr->jnt_dofadr[joint_id];
        mj_data_ptr->ctrl[i] = joint_command[joint_name];

        joint_pos_log_file << mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[joint_id]] << " ";
        joint_vel_log_file << mj_data_ptr->qvel[jnt_qvel_idx] << " ";
        joint_eff_log_file << mj_data_ptr->ctrl[i] << " ";
      }

      mj_step2(mj_model_ptr, mj_data_ptr);

      if (useRobot) {
        MotorCommand motor_command;
      
        // Impostazioni di base
        motor_command.tau_ff.fill(0.0f);
        motor_command.q_target.fill(0.0f);
        motor_command.dq_target.fill(0.0f);
        motor_command.kp = Kp;
        motor_command.kd = Kd;

        // assegna i valori di controllo per i giunti
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id));
          if (joint_name == "right_elbow_joint"){
            motor_command.q_target[joint_id] = 3.14/2;
            motor_command.dq_target[joint_id] = 0;
          }
        }
      
        // Costruisci comando DDS
        LowCmd_ dds_low_command;
        dds_low_command.mode_pr() = static_cast<uint8_t>(Mode::PR);
        dds_low_command.mode_machine() = 0;
      
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          auto &cmd = dds_low_command.motor_cmd().at(i);
          cmd.mode() = 1;
          cmd.q()    = motor_command.q_target[i];
          cmd.dq()   = motor_command.dq_target[i];
          cmd.tau()  = motor_command.tau_ff[i];
          cmd.kp()   = motor_command.kp[i];
          cmd.kd()   = motor_command.kd[i];
        }
      
        dds_low_command.crc() = Crc32Core((uint32_t*)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
        lowcmd_publisher->Write(dds_low_command);
      
        // Salva per usi futuri
        motor_command_buffer_.SetData(motor_command);
      }
      
      joint_pos_log_file << std::endl;
      joint_vel_log_file << std::endl;
      joint_eff_log_file << std::endl;
    
    }

    // Fine misurazione del tempo
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Stampa del tempo di esecuzione
    std::cout << "Tempo di esecuzione del main: " << duration << " millisecondi" << std::endl;


    mujoco_ui.render();
  }

  // Free memory (Mujoco):
  mj_deleteData(mj_data_ptr);
  mj_deleteModel(mj_model_ptr);

  joint_pos_log_file.close();
  joint_vel_log_file.close();
  joint_eff_log_file.close();

  return 0;
}

