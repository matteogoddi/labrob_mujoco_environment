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

  double force[6];
  double result[3];
  double sum[3]{0.0, 0.0, 0.0};
  double zmp[3]{0.0, 0.0, 0.0};
  for (int i = 0; i < d->ncon; ++i) {
    mj_contactForce(m, d, i, force);
    mju_rotVecMatT(result, force, d->contact[i].frame);
//    std::cout << "force " << i << " = ";
//    for (int j = 0; j < 3; ++j)
//      std::cout << result[j] << " ";
//    std::cout << std::endl;
//    std::cout << "contact position " << i << " = ";
//    for (int j = 0; j < 3; ++j)
//      std::cout << d->contact[i].pos[j] << " ";
//    std::cout << std::endl;
    sum[0] += result[0];
    sum[1] += result[1];
    sum[2] += result[2];
    zmp[0] += d->contact[i].pos[0] * result[2];
    zmp[1] += d->contact[i].pos[1] * result[2];
  }
  zmp[0] /= sum[2];
  zmp[1] /= sum[2];
//  std::cout << "sum = ";
//  for (int i = 0; i < 3; ++i) {
//    std::cout << sum[i] << " ";
//  }
//  std::cout << std::endl;
  std::cout << "zmp = ";
  for (int i = 0; i < 2; ++i) {
    std::cout << zmp[i] << " ";
  }
  std::cout << std::endl;
  robot_state.zmp(0) = zmp[0];
  robot_state.zmp(1) = zmp[1];
  robot_state.zmp(2) = 0.0;

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
  jvrc1_mj_data_ptr->qpos[2] = 0.792151-0.125+0.0263;
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

  std::map<std::string, double> armatures;
  for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
    int joint_id = jvrc1_mj_model_ptr->actuator_trnid[i * 2];
    std::string joint_name = std::string(mj_id2name(jvrc1_mj_model_ptr, mjOBJ_JOINT, joint_id));
    int dof_id = jvrc1_mj_model_ptr->jnt_dofadr[joint_id];
    armatures[joint_name] = jvrc1_mj_model_ptr->dof_armature[dof_id];
  }

  // Walking Manager:
  labrob::RobotState initial_robot_state = robot_state_from_mujoco(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);
  labrob::WalkingManager walking_manager;
  walking_manager.init(initial_robot_state, armatures);

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

  for (int i = 0; i < jvrc1_mj_model_ptr->nv; ++i) {
    std::cout << jvrc1_mj_model_ptr->dof_armature[i] << " ";
  }

  // Simulation loop:
  while (!glfwWindowShouldClose(window)) {
    mjtNum simstart = jvrc1_mj_data_ptr->time;
    while( jvrc1_mj_data_ptr->time - simstart < 1.0/60.0 ) {
      labrob::RobotState robot_state = robot_state_from_mujoco(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);
      labrob::JointState desired_joint_state;

      // Update walking manager:
      labrob::JointCommand joint_command;
      Eigen::VectorXd desired_base_velocity;
      Eigen::VectorXd desired_base_acceleration;
      Eigen::Vector3d zmp_position;
      walking_manager.update(robot_state, joint_command, desired_joint_state, desired_base_velocity, desired_base_acceleration, zmp_position);
      desired_base_velocity.block(0, 0, 3, 1) = robot_state.orientation.toRotationMatrix() * desired_base_velocity.block(0, 0, 3, 1);
      desired_base_acceleration.block(0, 0, 3, 1) = robot_state.orientation.toRotationMatrix() * desired_base_acceleration.block(0, 0, 3, 1);

      labrob::RobotState robot_state_mujoco = robot_state;
      robot_state_mujoco.linear_velocity = robot_state.orientation.toRotationMatrix() * robot_state.linear_velocity;

      double desired_joint_pos[jvrc1_mj_model_ptr->nu];
      for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
        int joint_id = jvrc1_mj_model_ptr->actuator_trnid[i * 2];
        std::string joint_name = std::string(mj_id2name(jvrc1_mj_model_ptr, mjOBJ_JOINT, joint_id));
        int jnt_qpos_idx = jvrc1_mj_model_ptr->jnt_qposadr[joint_id];
        desired_joint_pos[i] = jvrc1_mj_data_ptr->qpos[jnt_qpos_idx] + (1.0 / walking_manager.get_controller_frequency()) * joint_command[joint_name];
      }

//      for (const auto& joint_command_pair : joint_command) {
//        std::cerr << "joint_command[" << joint_command_pair.first << "]: " << joint_command_pair.second << std::endl;
//      }

      Eigen::MatrixXd Jl_prev;
      Eigen::MatrixXd Jr_prev;

      mjtNum JTf[jvrc1_mj_model_ptr->nv];

      for (int k = 0; k < 1; ++k) {
        mj_step1(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);

        mjtNum jacp_left[3 * jvrc1_mj_model_ptr->nv];
        mjtNum jacp_right[3 * jvrc1_mj_model_ptr->nv];
        mjtNum jacr_left[3 * jvrc1_mj_model_ptr->nv];
        mjtNum jacr_right[3 * jvrc1_mj_model_ptr->nv];
        mjtNum M[jvrc1_mj_model_ptr->nv * jvrc1_mj_model_ptr->nv];

        int left_foot_id = mj_name2id(jvrc1_mj_model_ptr, mjOBJ_BODY, "L_ANKLE_P_S");
        int right_foot_id = mj_name2id(jvrc1_mj_model_ptr, mjOBJ_BODY, "R_ANKLE_P_S");
        mj_jacBody(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr, jacp_left, jacr_left, left_foot_id);
        mj_jacBody(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr, jacp_right, jacr_right, right_foot_id);
        mj_fullM(jvrc1_mj_model_ptr, M, jvrc1_mj_data_ptr->qM);
        Eigen::MatrixXd jacp_left_eigen = convert_matrix_mujoco_to_eigen(jacp_left, 3, jvrc1_mj_model_ptr->nv);
        Eigen::MatrixXd jacp_right_eigen = convert_matrix_mujoco_to_eigen(jacp_right, 3, jvrc1_mj_model_ptr->nv);
        Eigen::MatrixXd jacr_left_eigen = convert_matrix_mujoco_to_eigen(jacr_left, 3, jvrc1_mj_model_ptr->nv);
        Eigen::MatrixXd jacr_right_eigen = convert_matrix_mujoco_to_eigen(jacr_right, 3, jvrc1_mj_model_ptr->nv);
        Eigen::MatrixXd Jl(2 * jacp_left_eigen.rows(), jacp_left_eigen.cols());
        Eigen::MatrixXd Jr(2 * jacp_right_eigen.rows(), jacp_right_eigen.cols());
        Eigen::MatrixXd dJl = Eigen::MatrixXd::Zero(2 * jacp_left_eigen.rows(), jacp_left_eigen.cols());
        Eigen::MatrixXd dJr = Eigen::MatrixXd::Zero(2 * jacp_left_eigen.rows(), jacp_left_eigen.cols());

        Jl << jacp_left_eigen,
            jacr_left_eigen;
        Jr << jacp_right_eigen,
            jacr_right_eigen;
        if (Jl_prev.size() != 0) {
          dJl = (Jl - Jl_prev) / 0.001;
          dJr = (Jr - Jr_prev) / 0.001;
        }
        Jl_prev = Jl;
        Jr_prev = Jr;

        int n_under = 6;
        int n_act = jvrc1_mj_model_ptr->nv - n_under;
        Eigen::MatrixXd Jlu = Jl.block(0, 0, Jl.rows(), 6);
        Eigen::MatrixXd Jla = Jl.block(0, 6, Jl.rows(), n_act);
        Eigen::MatrixXd Jru = Jr.block(0, 0, Jr.rows(), 6);
        Eigen::MatrixXd Jra = Jr.block(0, 6, Jr.rows(), n_act);

        mj_mulJacTVec(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr, JTf, jvrc1_mj_data_ptr->efc_force);

//        std::cout << "ncon = " << jvrc1_mj_data_ptr->ncon << std::endl;
//        double force[6];
//        double position[3];
//        double frame[9];
//        double result[3];
//        double sum[3]{0.0, 0.0, 0.0};
//        double zmp[3]{0.0, 0.0, 0.0};
//        for (int i = 0; i < jvrc1_mj_data_ptr->ncon; ++i) {
//          int address = jvrc1_mj_data_ptr->contact[i].efc_address;
//          mj_contactForce(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr, i, force);
//          mju_rotVecMatT(result, force, jvrc1_mj_data_ptr->contact[i].frame);
//          std::cout << "force " << i << " = ";
//          for (int j = 0; j < 3; ++j)
//            std::cout << result[j] << " ";
//          std::cout << std::endl;
//          std::cout << "contact position " << i << " = ";
//          for (int j = 0; j < 3; ++j)
//            std::cout << jvrc1_mj_data_ptr->contact[i].pos[j] << " ";
//          std::cout << std::endl;
//          sum[0] += result[0];
//          sum[1] += result[1];
//          sum[2] += result[2];
//          zmp[0] += jvrc1_mj_data_ptr->contact[i].pos[0] * result[2];
//          zmp[1] += jvrc1_mj_data_ptr->contact[i].pos[1] * result[2];
//        }
//        zmp[0] /= sum[2];
//        zmp[1] /= sum[2];
//        std::cout << "sum = ";
//        for (int i = 0; i < 3; ++i) {
//          std::cout << sum[i] << " ";
//        }
//        std::cout << std::endl;
//        std::cout << "zmp = ";
//        for (int i = 0; i < 2; ++i) {
//          std::cout << zmp[i] << " ";
//        }
//        std::cout << std::endl;
//        std::cout << "efc_force = " << std::endl;
//        for (int i = 0; i < jvrc1_mj_data_ptr->nefc; ++i) {
//          std::cout << jvrc1_mj_data_ptr->efc_force[i] << " ";
//        }
//        std::cout << std::endl;

        Eigen::MatrixXd M_eigen = convert_matrix_mujoco_to_eigen(M, jvrc1_mj_model_ptr->nv, jvrc1_mj_model_ptr->nv);
//        std::cout << "M_mujoco = " << std::endl << M_eigen << std::endl;

        Eigen::VectorXd c_eigen(jvrc1_mj_model_ptr->nv);

        Eigen::VectorXd dq = Eigen::VectorXd::Zero(jvrc1_mj_model_ptr->nv);
        for (int i = 0; i < jvrc1_mj_model_ptr->nv; ++i) {
          dq(i) = jvrc1_mj_data_ptr->qvel[i];
          c_eigen(i) = jvrc1_mj_data_ptr->qfrc_bias[i];
        }
//        std::cout << "qadd = " << qadd << std::endl;
        Eigen::VectorXd qdd = Eigen::VectorXd::Zero(jvrc1_mj_model_ptr->nv);
        qdd.block(0, 0, 6, 1) = desired_base_acceleration;
        for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
          int joint_id = jvrc1_mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(jvrc1_mj_model_ptr, mjOBJ_JOINT, joint_id));
          int jnt_qvel_idx = jvrc1_mj_model_ptr->jnt_dofadr[joint_id];
          qdd(jnt_qvel_idx) = desired_joint_state[joint_name].acc;
        }

//        std::cout << "M = " << std::endl << M_eigen << std::endl;

        Eigen::MatrixXd Mu = M_eigen.block(0, 0, 6, jvrc1_mj_model_ptr->nv);
        Eigen::MatrixXd Ma = M_eigen.block(6, 0, n_act, jvrc1_mj_model_ptr->nv);

        Eigen::VectorXd cu = c_eigen.block(0, 0, 6, 1);
        Eigen::VectorXd ca = c_eigen.block(6, 0, n_act, 1);

//        qdd.setZero();
        for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
          int joint_id = jvrc1_mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(jvrc1_mj_model_ptr, mjOBJ_JOINT, joint_id));
          int jnt_qpos_idx = jvrc1_mj_model_ptr->jnt_qposadr[joint_id];
          int jnt_qvel_idx = jvrc1_mj_model_ptr->jnt_dofadr[joint_id];
          mjtNum err_q = labrob::wrap_angle(qpos0[jnt_qpos_idx] - jvrc1_mj_data_ptr->qpos[jnt_qpos_idx]);
          mjtNum err_v = -jvrc1_mj_data_ptr->qvel[jnt_qvel_idx];
//          qdd(jnt_qvel_idx) = 50.0 * err_q + 10.0 * err_v;
        }

        double alpha = walking_manager.get_alpha();
        std::cout << "alpha = " << alpha << std::endl;
        Eigen::VectorXd fl = Jlu.transpose().completeOrthogonalDecomposition().pseudoInverse() * (Mu * qdd + cu);
        Eigen::VectorXd fr = Jru.transpose().completeOrthogonalDecomposition().pseudoInverse() * (Mu * qdd + cu);
//        Eigen::VectorXd left_foot_pos = convert_matrix_mujoco_to_eigen(jvrc1_mj_data_ptr->xpos[left_foot_id], 3, 1);
//        Eigen::VectorXd right_foot_pos = convert_matrix_mujoco_to_eigen(jvrc1_mj_data_ptr->xpos[right_foot_id], 3, 1);
        const double xl = jvrc1_mj_data_ptr->xpos[3 * left_foot_id + 0];
        const double yl = jvrc1_mj_data_ptr->xpos[3 * left_foot_id + 1];
        const double xr = jvrc1_mj_data_ptr->xpos[3 * right_foot_id + 0];
        const double yr = jvrc1_mj_data_ptr->xpos[3 * right_foot_id + 1];
        const double xc = zmp_position(0);
        const double yc = zmp_position(1);
//        std::cout << "(xl, yl) = " << xl << ", " << yl << std::endl;
//        std::cout << "(xr, yr) = " << xr << ", " << yr << std::endl;
//        std::cout << "(xc, yc) = " << xc << ", " << yc << std::endl;
        if (alpha >= -0.05 and alpha <= 1.05) {
          Eigen::MatrixXd A(6 + 2, 2 * 6);
          Eigen::VectorXd b(6 + 2);
          A.block(0, 0, 6, 6) = Jlu.transpose();
          A.block(0, 6, 6, 6) = Jru.transpose();
          A.block(6, 0, 1, 12) =
              Eigen::MatrixXd{{0.0, 0.0, xc - xl,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, xc - xr,
                              0.0, 1.0, 0.0}};
          A.block(7, 0, 1, 12) =
              Eigen::MatrixXd{{0.0, 0.0, yc - yl,
                              -1.0, 0.0, 0.0,
                              0.0, 0.0, yc - yr,
                              -1.0, 0.0, 0.0}};
          b.block(0, 0, 6, 1) = Mu * qdd + cu;
          b(6) = 0.0;
          b(7) = 0.0;
//          std::cout << A << std::endl;
//          std::cout << b << std::endl;
          Eigen::VectorXd f = A.completeOrthogonalDecomposition().pseudoInverse() * b;
          Eigen::VectorXd f_mix(12);
          f_mix.block(0, 0, 6, 1) = 0.5 * fl;
          f_mix.block(6, 0, 6, 1) = 0.5 * fr;
          fl = f.block(0, 0, 6, 1);
          fr = f.block(6, 0, 6, 1);
//          std::cout << "f = " << f.transpose() << std::endl;
//          std::cout << "f_mix = " << f_mix.transpose() << std::endl;
//          std::cout << "f.norm() = " << f.norm() << std::endl;
//          std::cout << "f_mix.norm() = " << f_mix.norm() << std::endl;
        } else if (alpha < 0.5) {
          fr.setZero();
        } else {
          fl.setZero();
        }
//        std::cout << "fl = " << fl.transpose() << std::endl;
//        std::cout << "fr = " << fr.transpose() << std::endl;
        Eigen::VectorXd tau_ext = Jla.transpose() * fl + Jra.transpose() * fr;
        Eigen::VectorXd tau_a = Ma * qdd + ca - tau_ext;
//        std::cout << "tau_a = " << tau_a.transpose() << std::endl;

        Eigen::VectorXd tau_a_expanded(6 + n_act);
        tau_a_expanded << Eigen::VectorXd::Zero(6), tau_a;

//        Eigen::MatrixXd A_f(12, 12);
//        Eigen::VectorXd f = A_f.colPivHouseholderQr().solve(Mu * qadd + cu);
//        Eigen::VectorXd fl = f.block(0, 0, 6, 1);
//        Eigen::VectorXd fr = f.block(6, 0, 6, 1);

        for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
          int joint_id = jvrc1_mj_model_ptr->actuator_trnid[i * 2];
          std::string joint_name = std::string(mj_id2name(jvrc1_mj_model_ptr, mjOBJ_JOINT, joint_id));
          int jnt_qpos_idx = jvrc1_mj_model_ptr->jnt_qposadr[joint_id];
          int jnt_qvel_idx = jvrc1_mj_model_ptr->jnt_dofadr[joint_id];
          mjtNum err_q = labrob::wrap_angle(desired_joint_state[joint_name].pos - jvrc1_mj_data_ptr->qpos[jnt_qpos_idx]);
          mjtNum err_v = desired_joint_state[joint_name].vel - jvrc1_mj_data_ptr->qvel[jnt_qvel_idx];
          //printf("jnt_qpos_idx=%d\n", jnt_qpos_idx);
          //printf("qpos0[%d]=%f\n", jnt_qpos_idx, qpos0[jnt_qpos_idx]);
          //printf("qpos[%d]=%f\n", jnt_qpos_idx, jvrc1_mj_data_ptr->qpos[jnt_qpos_idx]);
          //printf("err_q=%f\n", err_q);
//          jvrc1_mj_data_ptr->ctrl[i] = position_gains[i] * err_q + velocity_gains[i] * err_v;
          jvrc1_mj_data_ptr->ctrl[i] = joint_command[joint_name];//tau_a_expanded(jnt_qvel_idx);// - jvrc1_mj_data_ptr->qfrc_applied[jnt_qvel_idx] - JTf[jnt_qvel_idx];// + joint_command[joint_name] + position_gains[i] * err_q + velocity_gains[i] * err_v;
//          jvrc1_mj_data_ptr->ctrl[i] = 0.0;// position_gains[i] * err_q + velocity_gains[i] * err_v;
          //jvrc1_mj_data_ptr->ctrl[i - 1] = 10.0 * err_q;
          //jvrc1_mj_data_ptr->qvel[jnt_qvel_idx] = 10.0 * err_q;
//          jvrc1_mj_data_ptr->ctrl[i] = 0.0;

          joint_vel_log_file << jvrc1_mj_data_ptr->qvel[jnt_qvel_idx] << " ";
          joint_vel_des_log_file << desired_joint_state[joint_name].vel << " ";
          joint_eff_log_file << jvrc1_mj_data_ptr->ctrl[i] << " ";
        }

        mj_step2(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);

//        for (int i = 0; i < 6; ++i) {
//          jvrc1_mj_data_ptr->qvel[i] = desired_base_velocity(i);
////          jvrc1_mj_data_ptr->qvel[i] += 0.001 * qdd(i);
//        }
//        for (int i = 0; i < jvrc1_mj_model_ptr->nu; ++i) {
//          int joint_id = jvrc1_mj_model_ptr->actuator_trnid[i * 2];
//          std::string joint_name = std::string(mj_id2name(jvrc1_mj_model_ptr, mjOBJ_JOINT, joint_id));
//          int jnt_qvel_idx = jvrc1_mj_model_ptr->jnt_dofadr[joint_id];
//          jvrc1_mj_data_ptr->qvel[jnt_qvel_idx] = desired_joint_state[joint_name].vel;
////          jvrc1_mj_data_ptr->qvel[jnt_qvel_idx] += 0.001 * qdd(jnt_qvel_idx);
//        }
//        mj_step(jvrc1_mj_model_ptr, jvrc1_mj_data_ptr);

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

