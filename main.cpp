// std
#include <fstream>

// GLFW
#include <GLFW/glfw3.h>

// Mujoco
#include <mujoco/mujoco.h>

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

labrob::RobotState
robot_state_from_mujoco(mjModel* m, mjData* d) {
  labrob::RobotState robot_state;

  robot_state.position = Eigen::Vector3d(
    d->qpos[0], d->qpos[1], d->qpos[2]
  );

  robot_state.orientation = Eigen::Quaterniond(
    d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]
  );

  robot_state.linear_velocity = Eigen::Vector3d(
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

  return robot_state;
}

int main() {

  // Paths (to be read from local configuration file):
  const std::string pd_gains_filepath = "../config/pd_gains.txt";

  // Load MJCF (for Mujoco):
  const int kErrorLength = 1024;          // load error string length
  char loadError[kErrorLength] = "";
  const char* jvrc1_mjcf_filepath = "../jvrc_mj_description/scene.xml";
  mjModel* jvrc1_mj_model_ptr = mj_loadXML(jvrc1_mjcf_filepath, nullptr, loadError, kErrorLength);
  mjData* jvrc1_mj_data_ptr = mj_makeData(jvrc1_mj_model_ptr);

  std::ofstream joint_vel_log_file("/tmp/joint_vel.txt");
  std::ofstream joint_vel_des_log_file("/tmp/joint_vel_des.txt");
  std::ofstream joint_eff_log_file("/tmp/joint_eff.txt");
  std::ofstream joint_names_log_file("/tmp/joint_names.txt");

  // Init robot posture:
  mjtNum waist_p_init = 0.0;
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
  mjtNum r_elbow_p_init = -0.44;
  mjtNum l_shoulder_p_init = r_shoulder_p_init;
  mjtNum l_shoulder_r_init = -r_shoulder_r_init;
  mjtNum l_shoulder_y_init = 0.0;
  mjtNum l_elbow_p_init = r_elbow_p_init;

  for (int i = 0; i < jvrc1_mj_model_ptr->nq; ++i) {
    jvrc1_mj_data_ptr->qpos[i] = 0.0;
  }
  jvrc1_mj_data_ptr->qpos[2] = 0.792151-0.125;
  jvrc1_mj_data_ptr->qpos[3] = 1.0;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "WAIST_P")]] = waist_p_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_HIP_Y")]] = r_hip_y_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_HIP_R")]] = r_hip_r_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_HIP_P")]] = r_hip_p_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_KNEE")]] = r_knee_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_ANKLE_P")]] = r_ankle_p_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_ANKLE_R")]] = r_ankle_r_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_HIP_Y")]] = l_hip_y_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_HIP_R")]] = l_hip_r_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_HIP_P")]] = l_hip_p_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_KNEE")]] = l_knee_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_ANKLE_P")]] = l_ankle_p_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_ANKLE_R")]] = l_ankle_r_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_SHOULDER_P")]] = r_shoulder_p_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_SHOULDER_R")]] = r_shoulder_r_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_SHOULDER_Y")]] = r_shoulder_y_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "R_ELBOW_P")]] = r_elbow_p_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_SHOULDER_P")]] = l_shoulder_p_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_SHOULDER_R")]] = l_shoulder_r_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_SHOULDER_Y")]] = l_shoulder_y_init;
  jvrc1_mj_data_ptr->qpos[jvrc1_mj_model_ptr->jnt_qposadr[mj_name2id(jvrc1_mj_model_ptr, mjOBJ_JOINT, "L_ELBOW_P")]] = l_elbow_p_init;

  mjtNum* qpos0 = (mjtNum*) malloc(sizeof(mjtNum) * jvrc1_mj_model_ptr->nq);
  memcpy(qpos0, jvrc1_mj_data_ptr->qpos, jvrc1_mj_model_ptr->nq * sizeof(mjtNum));

  // Walking Manager:
  labrob::RobotState initial_robot_state = robot_state_from_mujoco(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);
  labrob::WalkingManager walking_manager;
  walking_manager.init(initial_robot_state);

  // Mujoco visualization:
  mjvCamera cam;                      // abstract camera
  mjvOption opt;                      // visualization options
  mjvScene scn;                       // abstract scene
  mjrContext con;                     // custom GPU context

  // init GLFW, create window, make OpenGL context current, request v-sync
  glfwInit();
  GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  // initialize visualization data structures
  mjv_defaultCamera(&cam);
  //mjv_defaultPerturb(&pert);
  mjv_defaultOption(&opt);
  mjr_defaultContext(&con);

  // create scene and context
  mjv_makeScene(jvrc1_mj_model_ptr, &scn, 1000);
  mjr_makeContext(jvrc1_mj_model_ptr, &con, mjFONTSCALE_100);

  for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
    int joint_id = jvrc1_mj_model_ptr->actuator_trnid[i * 2];
    std::string joint_name = std::string(mj_id2name(jvrc1_mj_model_ptr, mjOBJ_JOINT, joint_id));
    joint_names_log_file << joint_name << std::endl;
  }

  joint_names_log_file.flush();
  joint_names_log_file.close();

  std::vector<double> position_gains(jvrc1_mj_model_ptr->nu);
  std::vector<double> velocity_gains(jvrc1_mj_model_ptr->nu);

  // Read PD gains from file (assuming the size matches):
  std::ifstream pd_gains_file(pd_gains_filepath);
  for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
    pd_gains_file >> position_gains[i] >> velocity_gains[i];
  }

  // Simulation loop:
  while (!glfwWindowShouldClose(window)) {
    mjtNum simstart = jvrc1_mj_data_ptr->time;
    while( jvrc1_mj_data_ptr->time - simstart < 1.0/60.0 ) {
      labrob::RobotState robot_state = robot_state_from_mujoco(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);

      // Update walking manager:
      labrob::JointCommand joint_command;
      walking_manager.update(robot_state, joint_command);

      for (const auto& joint_command_pair : joint_command) {
        std::cerr << "joint_command[" << joint_command_pair.first << "]: " << joint_command_pair.second << std::endl;
      }

      for (int k = 0; k < 10; ++k) {
//        mj_step1(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);

        for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
          int joint_id = jvrc1_mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(jvrc1_mj_model_ptr, mjOBJ_JOINT, joint_id));
          int jnt_qpos_idx = jvrc1_mj_model_ptr->jnt_qposadr[joint_id];
          int jnt_qvel_idx = jvrc1_mj_model_ptr->jnt_dofadr[joint_id];
          mjtNum err_q = 0.001 * joint_command[joint_name];//labrob::wrap_angle(qpos0[jnt_qpos_idx] - jvrc1_mj_data_ptr->qpos[jnt_qpos_idx]);
          mjtNum err_v = joint_command[joint_name] - jvrc1_mj_data_ptr->qvel[jnt_qvel_idx];
          //printf("jnt_qpos_idx=%d\n", jnt_qpos_idx);
          //printf("qpos0[%d]=%f\n", jnt_qpos_idx, qpos0[jnt_qpos_idx]);
          //printf("qpos[%d]=%f\n", jnt_qpos_idx, jvrc1_mj_data_ptr->qpos[jnt_qpos_idx]);
          //printf("err_q=%f\n", err_q);
          jvrc1_mj_data_ptr->ctrl[i] = position_gains[i] * err_q + velocity_gains[i] * err_v;
          //jvrc1_mj_data_ptr->ctrl[i - 1] = 10.0 * err_q;
          //jvrc1_mj_data_ptr->qvel[jnt_qvel_idx] = 10.0 * err_q;

          joint_vel_log_file << jvrc1_mj_data_ptr->qvel[jnt_qvel_idx] << " ";
          joint_vel_des_log_file << joint_command[joint_name] << " ";
          joint_eff_log_file << jvrc1_mj_data_ptr->ctrl[i] << " ";
        }

//        mj_step2(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);

        mj_step(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);

        joint_vel_log_file << std::endl;
        joint_vel_des_log_file << std::endl;
        joint_eff_log_file << std::endl;
      }
    }

    joint_vel_log_file.flush();
    joint_vel_des_log_file.flush();
    joint_eff_log_file.flush();

    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
  }

  // close GLFW, free visualization storage
  glfwTerminate();
  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  // Free memory (Mujoco):
  mj_deleteData(jvrc1_mj_data_ptr);
  mj_deleteModel(jvrc1_mj_model_ptr);

  joint_vel_log_file.close();
  joint_vel_des_log_file.close();
  joint_eff_log_file.close();

  return 0;
}

