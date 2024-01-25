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
#include <hrp4_locomotion/utils.hpp>

mjtNum* qpos0;

void jvrc1_controller(mjModel* m, mjData* d) {
  for (int i = 0; i < m->nu; ++i) {
    int joint_id = m->actuator_trnid[i * 2];
    std::string joint_name = std::string(mj_id2name(m, mjOBJ_JOINT, joint_id));
    int jnt_qpos_idx = m->jnt_qposadr[joint_id];
    int jnt_qvel_idx = m->jnt_dofadr[joint_id];
    mjtNum err_q = labrob::wrap_angle(qpos0[jnt_qpos_idx] - d->qpos[jnt_qpos_idx]);
    mjtNum err_v = -d->qvel[jnt_qvel_idx];
    printf("jnt_qpos_idx=%d\n", jnt_qpos_idx);
    printf("qpos0[%d]=%f\n", jnt_qpos_idx, qpos0[jnt_qpos_idx]);
    printf("qpos[%d]=%f\n", jnt_qpos_idx, d->qpos[jnt_qpos_idx]);
    printf("err_q=%f\n", err_q);
    d->ctrl[i] = 2000.0 * err_q + 50.0 * err_v;
    //d->ctrl[i - 1] = 10.0 * err_q;
    //d->qvel[jnt_qvel_idx] = 10.0 * err_q;
  }
}

int main() {

  // Load MJCF (for Mujoco):
  const int kErrorLength = 1024;          // load error string length
  char loadError[kErrorLength] = "";
  const char* jvrc1_mjcf_filepath = "/home/michele/repos/labrob_mujoco_environment/jvrc_mj_description/scene.xml";
  mjModel* jvrc1_mj_model_ptr = mj_loadXML(jvrc1_mjcf_filepath, nullptr, loadError, kErrorLength);
  mjData* jvrc1_mj_data_ptr = mj_makeData(jvrc1_mj_model_ptr);

  // Init robot posture:
  mjtNum waist_p_init = 0.425;
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
  jvrc1_mj_data_ptr->qpos[2] = 0.792151;
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

  qpos0 = (mjtNum*) malloc(sizeof(mjtNum) * jvrc1_mj_model_ptr->nq);
  memcpy(qpos0, jvrc1_mj_data_ptr->qpos, jvrc1_mj_model_ptr->nq * sizeof(mjtNum));

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

  // Load URDF (for Pinocchio):
  std::string robot_description_filename = "/home/michele/repos/labrob_mujoco_environment/jvrc_description/urdf/jvrc1.urdf";
  pinocchio::Model full_robot_model;
  pinocchio::JointModelFreeFlyer root_joint;
  pinocchio::urdf::buildModel(
    robot_description_filename,
    root_joint,
    full_robot_model
  );
  pinocchio::Model robot_model = full_robot_model;
  pinocchio::Data robot_data(robot_model);

  // Simulation loop:
  while (!glfwWindowShouldClose(window)) {
    mjtNum simstart = jvrc1_mj_data_ptr->time;
    while( jvrc1_mj_data_ptr->time - simstart < 1.0/60.0 ) {
      jvrc1_controller(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);
      mj_step(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);
    }
      
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

  return 0;
}

