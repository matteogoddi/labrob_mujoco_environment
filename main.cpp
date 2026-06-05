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
#include <limits>
#include <stdexcept>

#include <thread>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Pinocchio
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
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
#include <hrp4_locomotion/ComplianceReferenceGenerator.hpp>

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

double torso_spring_kp = 50.0;
double torso_spring_kd = 10.0;
double torso_spring_roll_kp = 50.0;
double torso_spring_roll_kd = 10.0;
double torso_spring_pitch_kp = 50.0;
double torso_spring_pitch_kd = 10.0;
double torso_spring_yaw_kp = 50.0;
double torso_spring_yaw_kd = 10.0;
double torso_spring_weight = 5e-2;
double waist_yaw_compliance_kp = 1.2;
double waist_yaw_compliance_kd = 0.0;
double waist_roll_compliance_kp = 0.0;
double waist_roll_compliance_kd = 0.0;
double waist_pitch_compliance_kp = 0.0;
double waist_pitch_compliance_kd = 0.0;

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

using ComplianceVector6d = labrob::ComplianceReferenceGenerator::Vector6d;
using ComplianceMatrix6d = labrob::ComplianceReferenceGenerator::Matrix6d;

std::filesystem::path resolveProjectPath(const std::filesystem::path& relative_path) {
  const std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();
  const std::vector<std::filesystem::path> candidates{
      std::filesystem::current_path() / relative_path,
      std::filesystem::current_path() / ".." / relative_path,
      source_dir / relative_path,
  };

  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return std::filesystem::weakly_canonical(candidate);
    }
  }

  throw std::runtime_error("Cannot find project file: " + relative_path.string());
}

std::vector<Eigen::VectorXd> left_wrist_wrench_log;
std::vector<Eigen::VectorXd> right_wrist_wrench_log;
std::vector<Eigen::VectorXd> left_wrist_wrench_filtered_log;
std::vector<Eigen::VectorXd> right_wrist_wrench_filtered_log;
std::vector<Eigen::Matrix<double, 6, 1>> compliance_delta_x_left_log;
std::vector<Eigen::Matrix<double, 6, 1>> compliance_delta_x_right_log;
std::vector<Eigen::Matrix<double, 6, 1>> compliance_delta_dx_left_log;
std::vector<Eigen::Matrix<double, 6, 1>> compliance_delta_dx_right_log;
std::vector<Eigen::Matrix<double, 6, 1>> compliance_delta_ddx_left_log;
std::vector<Eigen::Matrix<double, 6, 1>> compliance_delta_ddx_right_log;
std::vector<double> compliance_time_log;
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
std::vector<double> compliance_torso_time_log;
std::vector<ComplianceVector6d> compliance_torso_wrench_left_log;
std::vector<ComplianceVector6d> compliance_torso_wrench_right_log;
std::vector<ComplianceVector6d> compliance_torso_manual_delta_xc_left_log;
std::vector<ComplianceVector6d> compliance_torso_manual_delta_xc_right_log;
std::vector<ComplianceVector6d> compliance_torso_delta_xc_left_log;
std::vector<ComplianceVector6d> compliance_torso_delta_xc_right_log;
std::vector<ComplianceVector6d> compliance_torso_delta_xc_left_filtered_log;
std::vector<ComplianceVector6d> compliance_torso_delta_xc_right_filtered_log;
std::vector<ComplianceVector6d> compliance_torso_delta_xb_log;
std::vector<ComplianceVector6d> compliance_torso_delta_xb_filtered_log;
std::vector<ComplianceVector6d> compliance_torso_delta_xb_final_log;
std::vector<int> compliance_torso_qp_solved_log;
std::vector<ComplianceMatrix6d> compliance_torso_Jb_left_log;
std::vector<ComplianceMatrix6d> compliance_torso_Jb_right_log;
std::vector<int> compliance_torso_Jb_valid_log;

std::vector<std::pair<std::string, int>> getValidJointNameIndexPairs();
int getControlledJointIndex(const std::string& joint_name);

ComplianceVector6d computeQuasiStaticComplianceDelta(
    const ComplianceVector6d& wrench,
    const ComplianceMatrix6d& stiffness,
    const ComplianceMatrix6d& selection) {
  ComplianceMatrix6d stiffness_reg = stiffness;
  stiffness_reg += 1e-10 * ComplianceMatrix6d::Identity();

  const ComplianceVector6d rhs = selection * wrench;
  ComplianceVector6d delta = stiffness_reg.ldlt().solve(rhs);
  if (!delta.allFinite()) {
    delta.setZero();
  }

  return delta;
}

Eigen::MatrixXd dampedPseudoInverse(
    const Eigen::MatrixXd& A,
    double damping = 1e-6) {
  const Eigen::MatrixXd regularized =
      A * A.transpose() +
      damping * damping * Eigen::MatrixXd::Identity(A.rows(), A.rows());
  return A.transpose() * regularized.ldlt().solve(
      Eigen::MatrixXd::Identity(A.rows(), A.rows()));
}

bool computeTorsoToWristJacobians(
    const pinocchio::Model& robot_model,
    const labrob::RobotState& robot_state,
    ComplianceMatrix6d& Jb_left,
    ComplianceMatrix6d& Jb_right) {
  const pinocchio::FrameIndex torso_frame_id =
      robot_model.existFrame("torso_link")
          ? robot_model.getFrameId("torso_link")
          : robot_model.nframes;
  const pinocchio::FrameIndex left_wrist_frame_id =
      robot_model.existFrame("left_wrist_yaw_joint")
          ? robot_model.getFrameId("left_wrist_yaw_joint")
          : robot_model.nframes;
  const pinocchio::FrameIndex right_wrist_frame_id =
      robot_model.existFrame("right_wrist_yaw_joint")
          ? robot_model.getFrameId("right_wrist_yaw_joint")
          : robot_model.nframes;

  if (torso_frame_id >= robot_model.nframes ||
      left_wrist_frame_id >= robot_model.nframes ||
      right_wrist_frame_id >= robot_model.nframes) {
    Jb_left.setZero();
    Jb_right.setZero();
    return false;
  }

  pinocchio::Data robot_data(robot_model);
  const Eigen::VectorXd q =
      labrob::robot_state_to_pinocchio_joint_configuration(robot_model, robot_state);

  pinocchio::computeJointJacobians(robot_model, robot_data, q);
  pinocchio::framesForwardKinematics(robot_model, robot_data, q);

  Eigen::MatrixXd J_torso = Eigen::MatrixXd::Zero(6, robot_model.nv);
  Eigen::MatrixXd J_left = Eigen::MatrixXd::Zero(6, robot_model.nv);
  Eigen::MatrixXd J_right = Eigen::MatrixXd::Zero(6, robot_model.nv);

  pinocchio::getFrameJacobian(
      robot_model,
      robot_data,
      torso_frame_id,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_torso);
  pinocchio::getFrameJacobian(
      robot_model,
      robot_data,
      left_wrist_frame_id,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_left);
  pinocchio::getFrameJacobian(
      robot_model,
      robot_data,
      right_wrist_frame_id,
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
      J_right);

  const Eigen::MatrixXd J_torso_pinv = dampedPseudoInverse(J_torso);
  Jb_left = J_left * J_torso_pinv;
  Jb_right = J_right * J_torso_pinv;

  if (!Jb_left.allFinite() || !Jb_right.allFinite()) {
    Jb_left.setZero();
    Jb_right.setZero();
    return false;
  }

  return true;
}

std::size_t torsoComplianceLogSampleCount() {
  std::size_t n = compliance_torso_time_log.size();
  n = std::min(n, compliance_torso_wrench_left_log.size());
  n = std::min(n, compliance_torso_wrench_right_log.size());
  n = std::min(n, compliance_torso_manual_delta_xc_left_log.size());
  n = std::min(n, compliance_torso_manual_delta_xc_right_log.size());
  n = std::min(n, compliance_torso_delta_xc_left_log.size());
  n = std::min(n, compliance_torso_delta_xc_right_log.size());
  n = std::min(n, compliance_torso_delta_xc_left_filtered_log.size());
  n = std::min(n, compliance_torso_delta_xc_right_filtered_log.size());
  n = std::min(n, compliance_torso_delta_xb_log.size());
  n = std::min(n, compliance_torso_delta_xb_filtered_log.size());
  n = std::min(n, compliance_torso_delta_xb_final_log.size());
  n = std::min(n, compliance_torso_qp_solved_log.size());
  n = std::min(n, compliance_torso_Jb_left_log.size());
  n = std::min(n, compliance_torso_Jb_right_log.size());
  n = std::min(n, compliance_torso_Jb_valid_log.size());
  return n;
}

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

  std::ofstream compliance_file("/tmp/compliance_hand_state.txt");
  compliance_file << "time "
                  << "l_dx0 l_dx1 l_dx2 l_dx3 l_dx4 l_dx5 "
                  << "l_ddx0 l_ddx1 l_ddx2 l_ddx3 l_ddx4 l_ddx5 "
                  << "l_dddx0 l_dddx1 l_dddx2 l_dddx3 l_dddx4 l_dddx5 "
                  << "r_dx0 r_dx1 r_dx2 r_dx3 r_dx4 r_dx5 "
                  << "r_ddx0 r_ddx1 r_ddx2 r_ddx3 r_ddx4 r_ddx5 "
                  << "r_dddx0 r_dddx1 r_dddx2 r_dddx3 r_dddx4 r_dddx5\n";

  const std::size_t n_compliance = std::min(
      compliance_time_log.size(),
      std::min(
          compliance_delta_x_left_log.size(),
          std::min(
              compliance_delta_x_right_log.size(),
              std::min(
                  compliance_delta_dx_left_log.size(),
                  std::min(
                      compliance_delta_dx_right_log.size(),
                      std::min(
                          compliance_delta_ddx_left_log.size(),
                          compliance_delta_ddx_right_log.size()))))));

  for (std::size_t i = 0; i < n_compliance; ++i) {
    compliance_file << compliance_time_log[i] << " "
                    << compliance_delta_x_left_log[i].transpose() << " "
                    << compliance_delta_dx_left_log[i].transpose() << " "
                    << compliance_delta_ddx_left_log[i].transpose() << " "
                    << compliance_delta_x_right_log[i].transpose() << " "
                    << compliance_delta_dx_right_log[i].transpose() << " "
                    << compliance_delta_ddx_right_log[i].transpose() << "\n";
  }

  std::ofstream torso_compliance_file("/tmp/compliance_torso_state.txt");
  torso_compliance_file << "time "
                        << "l_w0 l_w1 l_w2 l_w3 l_w4 l_w5 "
                        << "r_w0 r_w1 r_w2 r_w3 r_w4 r_w5 "
                        << "l_manual_dx0 l_manual_dx1 l_manual_dx2 l_manual_dx3 l_manual_dx4 l_manual_dx5 "
                        << "r_manual_dx0 r_manual_dx1 r_manual_dx2 r_manual_dx3 r_manual_dx4 r_manual_dx5 "
                        << "l_dx0 l_dx1 l_dx2 l_dx3 l_dx4 l_dx5 "
                        << "r_dx0 r_dx1 r_dx2 r_dx3 r_dx4 r_dx5 "
                        << "l_dxf0 l_dxf1 l_dxf2 l_dxf3 l_dxf4 l_dxf5 "
                        << "r_dxf0 r_dxf1 r_dxf2 r_dxf3 r_dxf4 r_dxf5 "
                        << "xb0 xb1 xb2 xb3 xb4 xb5 "
                        << "xbf0 xbf1 xbf2 xbf3 xbf4 xbf5 "
                        << "xbfinal0 xbfinal1 xbfinal2 xbfinal3 xbfinal4 xbfinal5 "
                        << "qp_solved\n";

  const std::size_t n_torso_compliance = torsoComplianceLogSampleCount();
  for (std::size_t i = 0; i < n_torso_compliance; ++i) {
    torso_compliance_file << compliance_torso_time_log[i] << " "
                          << compliance_torso_wrench_left_log[i].transpose() << " "
                          << compliance_torso_wrench_right_log[i].transpose() << " "
                          << compliance_torso_manual_delta_xc_left_log[i].transpose() << " "
                          << compliance_torso_manual_delta_xc_right_log[i].transpose() << " "
                          << compliance_torso_delta_xc_left_log[i].transpose() << " "
                          << compliance_torso_delta_xc_right_log[i].transpose() << " "
                          << compliance_torso_delta_xc_left_filtered_log[i].transpose() << " "
                          << compliance_torso_delta_xc_right_filtered_log[i].transpose() << " "
                          << compliance_torso_delta_xb_log[i].transpose() << " "
                          << compliance_torso_delta_xb_filtered_log[i].transpose() << " "
                          << compliance_torso_delta_xb_final_log[i].transpose() << " "
                          << compliance_torso_qp_solved_log[i] << "\n";
  }

  std::ofstream torso_jacobian_file("/tmp/compliance_torso_jacobian.txt");
  torso_jacobian_file << "time valid ";
  for (int side = 0; side < 2; ++side) {
    const char prefix = (side == 0) ? 'l' : 'r';
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        torso_jacobian_file << prefix << "_Jb_" << r << "_" << c;
        if (!(side == 1 && r == 5 && c == 5)) {
          torso_jacobian_file << " ";
        }
      }
    }
  }
  torso_jacobian_file << "\n";
  for (std::size_t i = 0; i < n_torso_compliance; ++i) {
    torso_jacobian_file << compliance_torso_time_log[i] << " "
                        << compliance_torso_Jb_valid_log[i] << " ";
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        torso_jacobian_file << compliance_torso_Jb_left_log[i](r, c) << " ";
      }
    }
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        torso_jacobian_file << compliance_torso_Jb_right_log[i](r, c);
        if (!(r == 5 && c == 5)) {
          torso_jacobian_file << " ";
        }
      }
    }
    torso_jacobian_file << "\n";
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
  400, 400, 400,                       // waist yaw/roll/pitch
  300, 300, 300, 300,  200, 200, 200,  // arms sx
  300, 300, 300, 300,  200, 200, 200   // arms dx
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
  10, 10, 10,                  // waist yaw/roll/pitch
  10, 10, 10, 10, 10, 10, 10,  // arms sx
  10, 10, 10, 10, 10, 10, 10   // arms dx
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
  {"waist_roll_joint", 13},
  {"waist_pitch_joint", 14},
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

int getControlledJointIndex(const std::string& joint_name) {
  auto it = joint_name_to_index.find(joint_name);
  if (it == joint_name_to_index.end()) {
    return -1;
  }

  const int idx = it->second;
  return (idx >= 0 && idx < G1_NUM_MOTOR) ? idx : -1;
}

bool isHandJoint(const std::string& joint_name) {
  return joint_name.find("_hand_") != std::string::npos;
}

double nominalTorqueLimitNm(const std::string& joint_name) {
  if (joint_name.find("_knee_joint") != std::string::npos) {
    return 139.0;
  }
  if (joint_name.find("_hip_") != std::string::npos) {
    return 88.0;
  }
  if (joint_name == "waist_yaw_joint") {
    return 88.0;
  }
  if (joint_name == "waist_roll_joint" || joint_name == "waist_pitch_joint") {
    return 50.0;
  }
  if (joint_name.find("_ankle_") != std::string::npos) {
    return 50.0;
  }
  if (joint_name.find("_wrist_pitch_joint") != std::string::npos ||
      joint_name.find("_wrist_yaw_joint") != std::string::npos) {
    return 5.0;
  }
  if (joint_name.find("_shoulder_") != std::string::npos ||
      joint_name.find("_elbow_joint") != std::string::npos ||
      joint_name.find("_wrist_roll_joint") != std::string::npos) {
    return 25.0;
  }
  if (joint_name.find("_hand_thumb_0_joint") != std::string::npos) {
    return 2.45;
  }
  if (isHandJoint(joint_name)) {
    return 1.4;
  }
  return 0.0;
}

double clampJointTorque(
    const std::string& joint_name,
    double tau,
    double limit_scale) {
  if (limit_scale <= 0.0) {
    return tau;
  }
  const double limit = nominalTorqueLimitNm(joint_name) * limit_scale;
  if (limit <= 0.0) {
    return tau;
  }
  return std::clamp(tau, -limit, limit);
}

struct FootContactPointDiag {
  std::string side;
  int geom_id;
  Eigen::Vector3d center;
  double radius;
  double surface_z;
};

std::vector<FootContactPointDiag> getFootContactPointDiagnostics(
    const mjModel* model,
    const mjData* data) {
  std::vector<FootContactPointDiag> points;
  const int left_body_id = mj_name2id(model, mjOBJ_BODY, "left_ankle_roll_link");
  const int right_body_id = mj_name2id(model, mjOBJ_BODY, "right_ankle_roll_link");

  for (int geom_id = 0; geom_id < model->ngeom; ++geom_id) {
    const int body_id = model->geom_bodyid[geom_id];
    std::string side;
    if (body_id == left_body_id) {
      side = "left";
    } else if (body_id == right_body_id) {
      side = "right";
    } else {
      continue;
    }

    if (model->geom_type[geom_id] != mjGEOM_SPHERE) {
      continue;
    }

    const double radius = model->geom_size[3 * geom_id];
    const Eigen::Vector3d center(
        data->geom_xpos[3 * geom_id + 0],
        data->geom_xpos[3 * geom_id + 1],
        data->geom_xpos[3 * geom_id + 2]);
    points.push_back({
        side,
        geom_id,
        center,
        radius,
        center.z() - radius
    });
  }

  return points;
}

double computeMujocoTotalMass(const mjModel* model) {
  double total_mass = 0.0;
  for (int body_id = 1; body_id < model->nbody; ++body_id) {
    total_mass += std::max(0.0, static_cast<double>(model->body_mass[body_id]));
  }
  return total_mass;
}

Eigen::Vector3d computeMujocoCom(const mjModel* model, const mjData* data) {
  Eigen::Vector3d weighted_sum = Eigen::Vector3d::Zero();
  const double total_mass = computeMujocoTotalMass(model);
  if (total_mass <= 0.0) {
    return Eigen::Vector3d::Zero();
  }

  for (int body_id = 1; body_id < model->nbody; ++body_id) {
    const double mass = model->body_mass[body_id];
    if (mass <= 0.0) {
      continue;
    }
    weighted_sum += mass * Eigen::Vector3d(
        data->xipos[3 * body_id + 0],
        data->xipos[3 * body_id + 1],
        data->xipos[3 * body_id + 2]);
  }
  return weighted_sum / total_mass;
}

bool getFootSurfaceHeightRange(
    const mjModel* model,
    const mjData* data,
    double& min_surface_z,
    double& max_surface_z) {
  const auto points = getFootContactPointDiagnostics(model, data);
  if (points.empty()) {
    return false;
  }

  min_surface_z = std::numeric_limits<double>::infinity();
  max_surface_z = -std::numeric_limits<double>::infinity();
  for (const auto& point : points) {
    min_surface_z = std::min(min_surface_z, point.surface_z);
    max_surface_z = std::max(max_surface_z, point.surface_z);
  }
  return true;
}

double computeTotalContactForceZ(const mjModel* model, const mjData* data) {
  double total_fz = 0.0;
  double force_contact[6];
  for (int contact_id = 0; contact_id < data->ncon; ++contact_id) {
    mj_contactForce(model, data, contact_id, force_contact);
    double force_world_z = 0.0;
    for (int col = 0; col < 3; ++col) {
      force_world_z += data->contact[contact_id].frame[3 * col + 2] * force_contact[col];
    }
    total_fz += force_world_z;
  }
  return total_fz;
}

void printInitialStanceDiagnostics(
    const mjModel* model,
    const mjData* data,
    const std::string& label) {
  const auto points = getFootContactPointDiagnostics(model, data);
  const Eigen::Vector3d com = computeMujocoCom(model, data);
  const double total_mass = computeMujocoTotalMass(model);

  if (points.empty()) {
    std::cout << "[stance-diag] " << label
              << " no foot contact sphere geoms found on ankle roll bodies."
              << std::endl;
    return;
  }

  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double min_surface_z = std::numeric_limits<double>::infinity();
  double max_surface_z = -std::numeric_limits<double>::infinity();

  for (const auto& point : points) {
    min_x = std::min(min_x, point.center.x());
    max_x = std::max(max_x, point.center.x());
    min_y = std::min(min_y, point.center.y());
    max_y = std::max(max_y, point.center.y());
    min_surface_z = std::min(min_surface_z, point.surface_z);
    max_surface_z = std::max(max_surface_z, point.surface_z);
  }

  const bool com_in_bbox =
      com.x() >= min_x && com.x() <= max_x &&
      com.y() >= min_y && com.y() <= max_y;
  const double support_margin = std::min({
      com.x() - min_x,
      max_x - com.x(),
      com.y() - min_y,
      max_y - com.y()
  });

  std::cout << "[stance-diag] " << label
            << " base_z=" << data->qpos[2]
            << " mass=" << total_mass
            << " weight=" << total_mass * 9.81
            << " com=[" << com.x() << ", " << com.y() << ", " << com.z() << "]"
            << " support_x=[" << min_x << ", " << max_x << "]"
            << " support_y=[" << min_y << ", " << max_y << "]"
            << " com_in_support_bbox=" << (com_in_bbox ? "yes" : "no")
            << " support_margin=" << support_margin
            << " min_foot_surface_z=" << min_surface_z
            << " max_foot_surface_z=" << max_surface_z
            << " foot_points=" << points.size()
            << std::endl;

  for (const auto& point : points) {
    std::cout << "[stance-foot] " << label
              << " side=" << point.side
              << " geom_id=" << point.geom_id
              << " center=[" << point.center.x()
              << ", " << point.center.y()
              << ", " << point.center.z() << "]"
              << " radius=" << point.radius
              << " surface_z=" << point.surface_z
              << std::endl;
  }
}

void printRuntimeStanceDiagnostics(
    const mjModel* model,
    const mjData* data) {
  double min_surface_z = 0.0;
  double max_surface_z = 0.0;
  const bool has_foot_height =
      getFootSurfaceHeightRange(model, data, min_surface_z, max_surface_z);
  const Eigen::Vector3d com = computeMujocoCom(model, data);
  const double total_mass = computeMujocoTotalMass(model);
  const Eigen::Quaterniond base_quat(
      data->qpos[3],
      data->qpos[4],
      data->qpos[5],
      data->qpos[6]);
  const Eigen::Vector3d base_rpy =
      base_quat.normalized().toRotationMatrix().eulerAngles(0, 1, 2);
  const auto points = getFootContactPointDiagnostics(model, data);
  bool has_support_bbox = !points.empty();
  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  for (const auto& point : points) {
    min_x = std::min(min_x, point.center.x());
    max_x = std::max(max_x, point.center.x());
    min_y = std::min(min_y, point.center.y());
    max_y = std::max(max_y, point.center.y());
  }
  const bool com_in_bbox =
      has_support_bbox &&
      com.x() >= min_x && com.x() <= max_x &&
      com.y() >= min_y && com.y() <= max_y;
  const double support_margin = has_support_bbox
      ? std::min({
            com.x() - min_x,
            max_x - com.x(),
            com.y() - min_y,
            max_y - com.y()})
      : 0.0;

  std::cout << "[runtime-stance] t=" << data->time
            << " base_z=" << data->qpos[2]
            << " base_rpy=[" << base_rpy.x()
            << ", " << base_rpy.y()
            << ", " << base_rpy.z() << "]"
            << " weight=" << total_mass * 9.81
            << " com=[" << com.x() << ", " << com.y() << ", " << com.z() << "]"
            << " com_in_support_bbox=" << (com_in_bbox ? "yes" : "no")
            << " support_margin=" << support_margin
            << " ncon=" << data->ncon
            << " contact_fz=" << computeTotalContactForceZ(model, data);
  if (has_foot_height) {
    std::cout << " foot_surface_z=[" << min_surface_z
              << ", " << max_surface_z << "]";
  }
  std::cout << std::endl;
}

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
  for (int i = 0; i < G1_NUM_MOTOR; ++i) {
    motor_state_data.q[i] = low_state.motor_state()[i].q();
    motor_state_data.dq[i] = low_state.motor_state()[i].dq();
    motor_state_data.ddq[i] = low_state.motor_state()[i].ddq();
    motor_state_data.tau_est[i] = low_state.motor_state()[i].tau_est();
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
  bool enable_waist_yaw_sine_test = false;
  double waist_yaw_test_amp_nm = 0.0;
  double waist_yaw_test_freq_hz = 1.0;
  double waist_yaw_test_start_sec = 2.0;
  double waist_yaw_test_duration_sec = 5.0;
  double waist_yaw_test_max_tau_nm = 12.0;
  bool use_mujoco_step = false;
  bool auto_save_logs = false;
  double stop_time_sec = -1.0;
  bool enable_torso_compliance_numeric_test = false;
  bool enable_torso_compliance_zero_input_test = false;
  bool enable_torso_compliance_wbc_test = false;
  bool waist_yaw_compliance_kp_user_set = false;
  bool waist_yaw_compliance_kd_user_set = false;
  bool waist_roll_compliance_kp_user_set = false;
  bool waist_roll_compliance_kd_user_set = false;
  bool waist_pitch_compliance_kp_user_set = false;
  bool waist_pitch_compliance_kd_user_set = false;
  bool disable_feedback_loops = false;
  bool feedback_loop_option_user_set = false;
  bool enable_joint_pd_hold_test = false;
  double mujoco_torque_limit_scale = 1.0;
  bool auto_initial_base_height = false;
  double initial_foot_clearance = 0.001;

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
    } else if (a == "--torso-spring-kp" && i + 1 < argc) {
      torso_spring_kp = std::atof(argv[++i]);
      torso_spring_roll_kp = torso_spring_kp;
      torso_spring_pitch_kp = torso_spring_kp;
      torso_spring_yaw_kp = torso_spring_kp;
    } else if (a == "--torso-spring-kd" && i + 1 < argc) {
      torso_spring_kd = std::atof(argv[++i]);
      torso_spring_roll_kd = torso_spring_kd;
      torso_spring_pitch_kd = torso_spring_kd;
      torso_spring_yaw_kd = torso_spring_kd;
    } else if (a == "--torso-spring-roll-kp" && i + 1 < argc) {
      torso_spring_roll_kp = std::atof(argv[++i]);
    } else if (a == "--torso-spring-roll-kd" && i + 1 < argc) {
      torso_spring_roll_kd = std::atof(argv[++i]);
    } else if (a == "--torso-spring-pitch-kp" && i + 1 < argc) {
      torso_spring_pitch_kp = std::atof(argv[++i]);
    } else if (a == "--torso-spring-pitch-kd" && i + 1 < argc) {
      torso_spring_pitch_kd = std::atof(argv[++i]);
    } else if (a == "--torso-spring-yaw-kp" && i + 1 < argc) {
      torso_spring_yaw_kp = std::atof(argv[++i]);
    } else if (a == "--torso-spring-yaw-kd" && i + 1 < argc) {
      torso_spring_yaw_kd = std::atof(argv[++i]);
    } else if (a == "--torso-spring-weight" && i + 1 < argc) {
      torso_spring_weight = std::atof(argv[++i]);
    } else if ((a == "--waist-yaw-compliance-kp" || a == "--waist-yaw-compliance-gain" || a == "--waist-yaw-spring-gain") && i + 1 < argc) {
      waist_yaw_compliance_kp = std::atof(argv[++i]);
      waist_yaw_compliance_kp_user_set = true;
    } else if (a == "--waist-yaw-compliance-kd" && i + 1 < argc) {
      waist_yaw_compliance_kd = std::atof(argv[++i]);
      waist_yaw_compliance_kd_user_set = true;
    } else if ((a == "--waist-roll-compliance-kp" || a == "--waist-roll-compliance-gain" || a == "--waist-roll-spring-gain") && i + 1 < argc) {
      waist_roll_compliance_kp = std::atof(argv[++i]);
      waist_roll_compliance_kp_user_set = true;
    } else if (a == "--waist-roll-compliance-kd" && i + 1 < argc) {
      waist_roll_compliance_kd = std::atof(argv[++i]);
      waist_roll_compliance_kd_user_set = true;
    } else if ((a == "--waist-pitch-compliance-kp" || a == "--waist-pitch-compliance-gain" || a == "--waist-pitch-spring-gain") && i + 1 < argc) {
      waist_pitch_compliance_kp = std::atof(argv[++i]);
      waist_pitch_compliance_kp_user_set = true;
    } else if (a == "--waist-pitch-compliance-kd" && i + 1 < argc) {
      waist_pitch_compliance_kd = std::atof(argv[++i]);
      waist_pitch_compliance_kd_user_set = true;
    } else if (a == "--waist-yaw-test-amp" && i + 1 < argc) {
      enable_waist_yaw_sine_test = true;
      waist_yaw_test_amp_nm = std::atof(argv[++i]);
    } else if (a == "--waist-yaw-test-freq" && i + 1 < argc) {
      enable_waist_yaw_sine_test = true;
      waist_yaw_test_freq_hz = std::atof(argv[++i]);
    } else if (a == "--waist-yaw-test-start" && i + 1 < argc) {
      enable_waist_yaw_sine_test = true;
      waist_yaw_test_start_sec = std::atof(argv[++i]);
    } else if (a == "--waist-yaw-test-duration" && i + 1 < argc) {
      enable_waist_yaw_sine_test = true;
      waist_yaw_test_duration_sec = std::atof(argv[++i]);
    } else if (a == "--waist-yaw-test-max-tau" && i + 1 < argc) {
      enable_waist_yaw_sine_test = true;
      waist_yaw_test_max_tau_nm = std::atof(argv[++i]);
    } else if (a == "--use-mujoco-step") {
      use_mujoco_step = true;
    } else if (a == "--disable-feedback-loops") {
      disable_feedback_loops = true;
      feedback_loop_option_user_set = true;
    } else if (a == "--enable-feedback-loops") {
      disable_feedback_loops = false;
      feedback_loop_option_user_set = true;
    } else if (a == "--auto-save-logs") {
      auto_save_logs = true;
    } else if (a == "--stop-time" && i + 1 < argc) {
      stop_time_sec = std::atof(argv[++i]);
    } else if (a == "--torso-compliance-numeric-test") {
      enable_torso_compliance_numeric_test = true;
    } else if (a == "--torso-compliance-zero-input-test") {
      enable_torso_compliance_numeric_test = true;
      enable_torso_compliance_zero_input_test = true;
    } else if (a == "--torso-compliance-wbc-test") {
      enable_torso_compliance_numeric_test = true;
      enable_torso_compliance_wbc_test = true;
    } else if (a == "--joint-pd-hold-test") {
      enable_joint_pd_hold_test = true;
      disable_feedback_loops = true;
      feedback_loop_option_user_set = true;
    } else if (a == "--mujoco-torque-limit-scale" && i + 1 < argc) {
      mujoco_torque_limit_scale = std::atof(argv[++i]);
    } else if (a == "--auto-initial-base-height") {
      auto_initial_base_height = true;
    } else if (a == "--initial-foot-clearance" && i + 1 < argc) {
      initial_foot_clearance = std::atof(argv[++i]);
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
            << "Torso/waist tracking options:\n"
            << "  --torso-spring-kp <value> (sets roll/pitch/yaw kp)\n"
            << "  --torso-spring-kd <value> (sets roll/pitch/yaw kd)\n"
            << "  --torso-spring-roll-kp <value>\n"
            << "  --torso-spring-roll-kd <value>\n"
            << "  --torso-spring-pitch-kp <value>\n"
            << "  --torso-spring-pitch-kd <value>\n"
            << "  --torso-spring-yaw-kp <value>\n"
            << "  --torso-spring-yaw-kd <value>\n"
            << "  --torso-spring-weight <value>\n"
            << "  --waist-yaw-compliance-kp <value> (deprecated alias for --torso-spring-yaw-kp in torso WBC test)\n"
            << "  --waist-yaw-compliance-kd <value> (deprecated alias for --torso-spring-yaw-kd in torso WBC test)\n"
            << "  --waist-roll-compliance-kp <value> (deprecated alias for --torso-spring-roll-kp in torso WBC test)\n"
            << "  --waist-roll-compliance-kd <value> (deprecated alias for --torso-spring-roll-kd in torso WBC test)\n"
            << "  --waist-pitch-compliance-kp <value> (deprecated alias for --torso-spring-pitch-kp in torso WBC test)\n"
            << "  --waist-pitch-compliance-kd <value> (deprecated alias for --torso-spring-pitch-kd in torso WBC test)\n"
            << "    Legacy kp aliases: --waist-*-compliance-gain, --waist-*-spring-gain\n"
            << "  --waist-yaw-test-amp <N*m>\n"
            << "  --waist-yaw-test-freq <Hz>\n"
            << "  --waist-yaw-test-start <sec>\n"
            << "  --waist-yaw-test-duration <sec>\n"
            << "  --waist-yaw-test-max-tau <N*m>\n"
            << "Torso compliance tests:\n"
            << "  --torso-compliance-numeric-test\n"
            << "    Uses existing --external-left/right-* inputs, EstimateForce filtered wrist wrench,\n"
            << "    and current Pinocchio torso-to-wrist Jacobians.\n"
            << "  --torso-compliance-zero-input-test\n"
            << "    Uses zero torso generator input with current Pinocchio torso-to-wrist Jacobians.\n"
            << "  --torso-compliance-wbc-test\n"
            << "    Applies torso QP angular output as WBC torso orientation reference offset.\n"
            << "Other:\n"
            << "  --joint-pd-hold-test\n"
            << "    Bypasses WBC and holds the initial joint pose with bounded joint PD torques.\n"
            << "  --mujoco-torque-limit-scale <scale>\n"
            << "    Optional MuJoCo torque clamp before writing ctrl. Use 1 for nominal joint limits.\n"
            << "  --auto-initial-base-height\n"
            << "    Shifts the initial floating-base z so the lowest foot contact sphere is near the floor.\n"
            << "  --initial-foot-clearance <m>\n"
            << "    Target lowest foot sphere clearance used with --auto-initial-base-height.\n"
            << "  --disable-feedback-loops\n"
            << "  --enable-feedback-loops\n"
            << "  --auto-save-logs\n"
            << "  --stop-time <sec>\n"
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

  if (useSim && !useRobot && !feedback_loop_option_user_set) {
    disable_feedback_loops = true;
  }

  if (enable_torso_compliance_wbc_test) {
    if (waist_roll_compliance_kp_user_set) {
      torso_spring_roll_kp = waist_roll_compliance_kp;
      std::cout << "Compatibility: --waist-roll-compliance-kp is treated as --torso-spring-roll-kp in torso compliance WBC test." << std::endl;
    }
    if (waist_roll_compliance_kd_user_set) {
      torso_spring_roll_kd = waist_roll_compliance_kd;
      std::cout << "Compatibility: --waist-roll-compliance-kd is treated as --torso-spring-roll-kd in torso compliance WBC test." << std::endl;
    }
    if (waist_pitch_compliance_kp_user_set) {
      torso_spring_pitch_kp = waist_pitch_compliance_kp;
      std::cout << "Compatibility: --waist-pitch-compliance-kp is treated as --torso-spring-pitch-kp in torso compliance WBC test." << std::endl;
    }
    if (waist_pitch_compliance_kd_user_set) {
      torso_spring_pitch_kd = waist_pitch_compliance_kd;
      std::cout << "Compatibility: --waist-pitch-compliance-kd is treated as --torso-spring-pitch-kd in torso compliance WBC test." << std::endl;
    }
    if (waist_yaw_compliance_kp_user_set) {
      torso_spring_yaw_kp = waist_yaw_compliance_kp;
      std::cout << "Compatibility: --waist-yaw-compliance-kp is treated as --torso-spring-yaw-kp in torso compliance WBC test." << std::endl;
    }
    if (waist_yaw_compliance_kd_user_set) {
      torso_spring_yaw_kd = waist_yaw_compliance_kd;
      std::cout << "Compatibility: --waist-yaw-compliance-kd is treated as --torso-spring-yaw-kd in torso compliance WBC test." << std::endl;
    }

    waist_yaw_compliance_kp = 0.0;
    waist_yaw_compliance_kd = 0.0;
    waist_roll_compliance_kp = 0.0;
    waist_roll_compliance_kd = 0.0;
    waist_pitch_compliance_kp = 0.0;
    waist_pitch_compliance_kd = 0.0;
  }
  if (enable_torso_compliance_wbc_test && !feedback_loop_option_user_set) {
    disable_feedback_loops = true;
  }

  if (torso_spring_kp < 0.0 || torso_spring_kd < 0.0 ||
      torso_spring_roll_kp < 0.0 || torso_spring_roll_kd < 0.0 ||
      torso_spring_pitch_kp < 0.0 || torso_spring_pitch_kd < 0.0 ||
      torso_spring_yaw_kp < 0.0 || torso_spring_yaw_kd < 0.0 ||
      torso_spring_weight < 0.0 ||
      waist_yaw_compliance_kp < 0.0 || waist_yaw_compliance_kd < 0.0 ||
      waist_roll_compliance_kp < 0.0 || waist_roll_compliance_kd < 0.0 ||
      waist_pitch_compliance_kp < 0.0 || waist_pitch_compliance_kd < 0.0) {
    std::cerr << "Torso/waist spring parameters must be non-negative." << std::endl;
    return -1;
  }

  if (enable_waist_yaw_sine_test && !useSim) {
    std::cerr << "Waist yaw sine test requires simulation mode (--sim)." << std::endl;
    return -1;
  }
  if (waist_yaw_test_freq_hz < 0.0 || waist_yaw_test_duration_sec < 0.0 || waist_yaw_test_max_tau_nm < 0.0) {
    std::cerr << "Waist yaw sine test frequency, duration and max tau must be non-negative." << std::endl;
    return -1;
  }
  if (enable_torso_compliance_numeric_test && !useSim) {
    std::cerr << "Torso compliance numeric test requires simulation mode (--sim)." << std::endl;
    return -1;
  }
  if (enable_joint_pd_hold_test && (!useSim || useRobot)) {
    std::cerr << "Joint PD hold test is simulation-only; use --sim without --robot." << std::endl;
    return -1;
  }
  if (enable_torso_compliance_numeric_test &&
      !enable_torso_compliance_zero_input_test &&
      !(enable_external_left_force_test || enable_external_right_force_test ||
        enable_external_left_torque_test || enable_external_right_torque_test)) {
    std::cerr << "Torso compliance numeric test requires at least one existing --external-left/right-* wrench input." << std::endl;
    return -1;
  }
  if (stop_time_sec == 0.0 || stop_time_sec < -1.0) {
    std::cerr << "Stop time must be positive, or negative to disable it." << std::endl;
    return -1;
  }
  if (mujoco_torque_limit_scale == 0.0 || mujoco_torque_limit_scale < -1.0) {
    std::cerr << "MuJoCo torque limit scale must be positive, or negative to disable it." << std::endl;
    return -1;
  }
  if (initial_foot_clearance < 0.0) {
    std::cerr << "Initial foot clearance must be non-negative." << std::endl;
    return -1;
  }

  std::cout << "Torso spring (WBC): kp(r,p,y)=("
            << torso_spring_roll_kp << ", "
            << torso_spring_pitch_kp << ", "
            << torso_spring_yaw_kp << "), kd(r,p,y)=("
            << torso_spring_roll_kd << ", "
            << torso_spring_pitch_kd << ", "
            << torso_spring_yaw_kd << ")"
            << ", weight=" << torso_spring_weight << std::endl;
  std::cout << "Waist compliance gains: yaw(kp,kd)=("
            << waist_yaw_compliance_kp << ", " << waist_yaw_compliance_kd
            << "), roll(kp,kd)=(" << waist_roll_compliance_kp << ", " << waist_roll_compliance_kd
            << "), pitch(kp,kd)=(" << waist_pitch_compliance_kp << ", " << waist_pitch_compliance_kd
            << ")" << std::endl;
  if (enable_waist_yaw_sine_test) {
    std::cout << "Waist yaw sine test: amp=" << waist_yaw_test_amp_nm
              << " N*m, freq=" << waist_yaw_test_freq_hz
              << " Hz, start=" << waist_yaw_test_start_sec
              << " s, duration=" << waist_yaw_test_duration_sec
              << " s, max_tau=" << waist_yaw_test_max_tau_nm << " N*m" << std::endl;
  }
  if (enable_joint_pd_hold_test) {
    std::cout << "Joint PD hold test enabled: WBC update is bypassed and the initial joint pose is held." << std::endl;
  }
  if (mujoco_torque_limit_scale > 0.0) {
    std::cout << "MuJoCo torque clamp enabled with scale=" << mujoco_torque_limit_scale << std::endl;
  }
  if (auto_initial_base_height) {
    std::cout << "Auto initial base height enabled: target foot clearance="
              << initial_foot_clearance << " m" << std::endl;
  }
  if (enable_torso_compliance_numeric_test) {
    if (enable_torso_compliance_wbc_test) {
      std::cout << "Torso compliance WBC test enabled. Output is logged and angular delta_xb is applied to WBC." << std::endl;
    } else {
      std::cout << "Torso compliance numeric test enabled. Output is logged only and is not applied to WBC." << std::endl;
    }
    if (enable_torso_compliance_zero_input_test) {
      std::cout << "  Source wrench: zero input injected only at the torso generator input." << std::endl;
    } else {
      std::cout << "  Source wrench: EstimateForce filtered wrist wrench after applying existing external-force test inputs." << std::endl;
    }
    std::cout << "  Jb source: current Pinocchio wrist Jacobian times damped pseudoinverse of torso Jacobian." << std::endl;
    if (enable_torso_compliance_wbc_test) {
      std::cout << "  WBC coupling: delta_xb_final angular part is applied to the WBC torso orientation reference." << std::endl;
      std::cout << "  WBC coupling: delta_xb_final linear part is logged but not applied by the current orientation-only torso task." << std::endl;
      if (disable_feedback_loops) {
        std::cout << "  Feedback isolation: TotalBody/CoM/EKF feedback loops are disabled for this run." << std::endl;
      }
    }
  }

 signal(SIGINT, signalHandler);

  // Load MJCF (for Mujoco):
  const int kErrorLength = 1024;          // load error string length
  char loadError[kErrorLength] = "";
  const std::string mjcf_filepath =
      resolveProjectPath("g1_mj_description/g1_29dof_with_hand_scene.xml").string();
  mjModel* mj_model_ptr = mj_loadXML(mjcf_filepath.c_str(), nullptr, loadError, kErrorLength);
  if (!mj_model_ptr) {
    std::cerr << "Error loading model: " << loadError << std::endl;
    return -1;
  }
  mjData* mj_data_ptr = mj_makeData(mj_model_ptr);

  const int waist_yaw_joint_id = mj_name2id(mj_model_ptr, mjOBJ_JOINT, "waist_yaw_joint");
  const int waist_yaw_qpos_adr = (waist_yaw_joint_id >= 0) ? mj_model_ptr->jnt_qposadr[waist_yaw_joint_id] : -1;
  const int waist_yaw_dof_adr = (waist_yaw_joint_id >= 0) ? mj_model_ptr->jnt_dofadr[waist_yaw_joint_id] : -1;
  int waist_yaw_actuator_idx = -1;
  for (int i = 0; i < mj_model_ptr->nu; ++i) {
    if (mj_model_ptr->actuator_trnid[i * 2] == waist_yaw_joint_id) {
      waist_yaw_actuator_idx = i;
      break;
    }
  }
  if (waist_yaw_joint_id >= 0) {
    std::cout << "Waist yaw diagnostic: joint_id=" << waist_yaw_joint_id
              << ", actuator_idx=" << waist_yaw_actuator_idx
              << ", qpos_adr=" << waist_yaw_qpos_adr
              << ", dof_adr=" << waist_yaw_dof_adr << std::endl;
  } else {
    std::cout << "Waist yaw diagnostic: joint not found in MuJoCo model." << std::endl;
  }
  double last_waist_diag_time = -1.0;
  double last_ctrl_diag_time = -1.0;
  double last_runtime_stance_diag_time = -1.0;

  if (useRobot && !disable_feedback_loops) {
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
  } else if (!disable_feedback_loops) {
    isTotalBodyLoopClosed = true;
    isCoMLoopClosed = true;
    isEKFactive = true;
  } else {
    isTotalBodyLoopClosed = false;
    isCoMLoopClosed = false;
    isEKFactive = false;
    std::cout << "Feedback loops disabled: WBC will use MuJoCo simulation state without EKF/feedback state switching." << std::endl;
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
  auto set_joint_qpos = [&](const char* joint_name, mjtNum value) {
    const int joint_id = mj_name2id(mj_model_ptr, mjOBJ_JOINT, joint_name);
    if (joint_id >= 0) {
      mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[joint_id]] = value;
    }
  };
  auto set_hand_joint_hold_qpos = [&](const char* joint_name, mjtNum value) {
    const int joint_id = mj_name2id(mj_model_ptr, mjOBJ_JOINT, joint_name);
    if (joint_id < 0) {
      return;
    }

    mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[joint_id]] = value;
    const int dof_id = mj_model_ptr->jnt_dofadr[joint_id];
    mj_model_ptr->dof_damping[dof_id] = std::max<mjtNum>(mj_model_ptr->dof_damping[dof_id], 0.08);
    mj_model_ptr->dof_frictionloss[dof_id] = std::max<mjtNum>(mj_model_ptr->dof_frictionloss[dof_id], 0.002);
    mj_model_ptr->dof_armature[dof_id] = std::max<mjtNum>(mj_model_ptr->dof_armature[dof_id], 1e-4);
  };

  set_joint_qpos("waist_yaw_joint", waist_y_init);
  set_joint_qpos("waist_roll_joint", 0.0);
  set_joint_qpos("waist_pitch_joint", 0.0);
  set_joint_qpos("right_hip_yaw_joint", r_hip_y_init);
  set_joint_qpos("right_hip_roll_joint", r_hip_r_init);
  set_joint_qpos("right_hip_pitch_joint", r_hip_p_init);
  set_joint_qpos("right_knee_joint", r_knee_init);
  set_joint_qpos("right_ankle_pitch_joint", r_ankle_p_init);
  set_joint_qpos("right_ankle_roll_joint", r_ankle_r_init);
  set_joint_qpos("left_hip_yaw_joint", l_hip_y_init);
  set_joint_qpos("left_hip_roll_joint", l_hip_r_init);
  set_joint_qpos("left_hip_pitch_joint", l_hip_p_init);
  set_joint_qpos("left_knee_joint", l_knee_init);
  set_joint_qpos("left_ankle_pitch_joint", l_ankle_p_init);
  set_joint_qpos("left_ankle_roll_joint", l_ankle_r_init);
  set_joint_qpos("right_shoulder_pitch_joint", r_shoulder_p_init);
  set_joint_qpos("right_shoulder_roll_joint", r_shoulder_r_init);
  set_joint_qpos("right_shoulder_yaw_joint", r_shoulder_y_init);
  // set_joint_qpos("right_elbow_joint", r_elbow_p_init);
  set_joint_qpos("left_shoulder_pitch_joint", l_shoulder_p_init);
  set_joint_qpos("left_shoulder_roll_joint", l_shoulder_r_init);
  set_joint_qpos("left_shoulder_yaw_joint", l_shoulder_y_init);
  // set_joint_qpos("left_elbow_joint", l_elbow_p_init);

  set_hand_joint_hold_qpos("left_hand_thumb_0_joint", 0.0);
  set_hand_joint_hold_qpos("left_hand_thumb_1_joint", 0.25);
  set_hand_joint_hold_qpos("left_hand_thumb_2_joint", 0.35);
  set_hand_joint_hold_qpos("left_hand_middle_0_joint", -0.25);
  set_hand_joint_hold_qpos("left_hand_middle_1_joint", -0.35);
  set_hand_joint_hold_qpos("left_hand_index_0_joint", -0.25);
  set_hand_joint_hold_qpos("left_hand_index_1_joint", -0.35);
  set_hand_joint_hold_qpos("right_hand_thumb_0_joint", 0.0);
  set_hand_joint_hold_qpos("right_hand_thumb_1_joint", -0.25);
  set_hand_joint_hold_qpos("right_hand_thumb_2_joint", -0.35);
  set_hand_joint_hold_qpos("right_hand_middle_0_joint", 0.25);
  set_hand_joint_hold_qpos("right_hand_middle_1_joint", 0.35);
  set_hand_joint_hold_qpos("right_hand_index_0_joint", 0.25);
  set_hand_joint_hold_qpos("right_hand_index_1_joint", 0.35);

  mj_forward(mj_model_ptr, mj_data_ptr);
  printInitialStanceDiagnostics(mj_model_ptr, mj_data_ptr, "initial");
  if (auto_initial_base_height) {
    double min_foot_surface_z = 0.0;
    double max_foot_surface_z = 0.0;
    if (getFootSurfaceHeightRange(
            mj_model_ptr,
            mj_data_ptr,
            min_foot_surface_z,
            max_foot_surface_z)) {
      const double base_z_shift = initial_foot_clearance - min_foot_surface_z;
      mj_data_ptr->qpos[2] += base_z_shift;
      mj_forward(mj_model_ptr, mj_data_ptr);
      std::cout << "[stance-diag] auto_initial_base_height shift_z="
                << base_z_shift << std::endl;
      printInitialStanceDiagnostics(mj_model_ptr, mj_data_ptr, "after-auto-height");
    } else {
      std::cout << "[stance-diag] auto_initial_base_height skipped: no foot contact sphere geoms found."
                << std::endl;
    }
  }

  std::map<std::string, double> armatures;
  std::map<std::string, double> initial_joint_targets;
  std::map<std::string, double> fixed_hand_joint_targets;
  int controlled_actuator_count = 0;
  int fixed_hand_actuator_count = 0;
  for (int i = 0; i < mj_model_ptr->nu; ++i) {
    int joint_id = mj_model_ptr->actuator_trnid[i * 2];
    const char* joint_name_cstr = mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id);
    if (joint_name_cstr == nullptr) {
      continue;
    }
    std::string joint_name = std::string(joint_name_cstr);
    int dof_id = mj_model_ptr->jnt_dofadr[joint_id];
    armatures[joint_name] = mj_model_ptr->dof_armature[dof_id];
    initial_joint_targets[joint_name] = mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[joint_id]];
    if (getControlledJointIndex(joint_name) >= 0) {
      ++controlled_actuator_count;
    } else if (isHandJoint(joint_name)) {
      fixed_hand_joint_targets[joint_name] = mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[joint_id]];
      ++fixed_hand_actuator_count;
    }
  }
  std::cout << "MuJoCo actuator summary: total=" << mj_model_ptr->nu
            << ", WBC controlled=" << controlled_actuator_count
            << ", fixed hand=" << fixed_hand_actuator_count << std::endl;

  // Walking Manager:
  labrob::RobotState robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);

  labrob::ComplianceReferenceGenerator compliance_reference_generator;
  labrob::ComplianceReferenceGenerator::Parameters compliance_params;
  compliance_params.compliance_mode = labrob::ComplianceReferenceGenerator::ComplianceMode::TORSO_ONLY;
  compliance_params.use_admittance_dynamics = false;

  compliance_params.S_left.setZero();
  compliance_params.S_right.setZero();
  // compliance_params.S_left(5, 5) = 1.0; // enable left hand pitch compliance

  compliance_params.Ma_left.setZero();
  compliance_params.Da_left.setZero();
  compliance_params.Ka_left.setZero();
  compliance_params.Ma_right.setZero();
  compliance_params.Da_right.setZero();
  compliance_params.Ka_right.setZero();

  // left hand: only z channel really matters, but keep all diagonals positive
  compliance_params.Ma_left.diagonal() << 5.0, 5.0, 5.0, 0.1, 0.1, 0.1;
  compliance_params.Ka_left.diagonal() << 10.0, 10.0, 10.0, 0.5, 0.5, 0.5;

  // critical damping: D = 2 * sqrt(M*K)
  // for M=5, K=100, D≈44.7
  compliance_params.Da_left.diagonal() << 45.0, 45.0, 45.0, 1.0, 1.0, 1.0;

  // right hand disabled by S_right=0, but still give positive matrices
  compliance_params.Ma_right.diagonal() << 5.0, 5.0, 5.0, 0.1, 0.1, 0.1;
  compliance_params.Ka_right.diagonal() << 10.0, 10.0, 10.0, 0.5, 0.5, 0.5;
  compliance_params.Da_right.diagonal() << 45.0, 45.0, 45.0, 1.0, 1.0, 1.0;

  compliance_params.Kb.diagonal() << 100.0, 100.0, 100.0, 0.001, 0.001, 0.001;
  std::cout << "Torso compliance QP params: Kb_diag=("
            << compliance_params.Kb.diagonal().transpose()
            << "), Ka_left_diag=("
            << compliance_params.Ka_left.diagonal().transpose()
            << "), Ka_right_diag=("
            << compliance_params.Ka_right.diagonal().transpose()
            << ")" << std::endl;

  // safety limit: do not comment this out
  compliance_params.delta_xc_left_limit  << 0.03, 0.03, 0.03, 0.08, 0.08, 0.08;
  compliance_params.delta_xc_right_limit << 0.03, 0.03, 0.03, 0.08, 0.08, 0.08;

  compliance_params.filter_alpha = 0.98;

  if (enable_torso_compliance_numeric_test) {
    compliance_params.compliance_mode = labrob::ComplianceReferenceGenerator::ComplianceMode::TORSO_ONLY;
    compliance_params.use_admittance_dynamics = false;
    compliance_params.S_left.setIdentity();
    compliance_params.S_right.setIdentity();
    compliance_params.delta_xc_left_limit = ComplianceVector6d::Constant(1e9);
    compliance_params.delta_xc_right_limit = ComplianceVector6d::Constant(1e9);
  }

  compliance_reference_generator.setParameters(compliance_params);
  compliance_reference_generator.reset();   // clear workspace shift and velocity history
  double previous_compliance_time = -1.0;
  
  walking_manager.init(robot_state, armatures);
  const int wbc_num_joints = walking_manager.getRobotModel().nv - 6;
  if (wbc_num_joints != G1_NUM_MOTOR) {
    std::cerr << "WBC model joint count mismatch: expected " << G1_NUM_MOTOR
              << ", got " << wbc_num_joints
              << ". Check the reduced URDF joint lock list." << std::endl;
    return -1;
  }
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

  // robot_state = walking_manager.getNewRobotState(robot_state);
  // StandStillInfinete(robot_state, mj_model_ptr, mj_data_ptr);

  bool stop_requested = false;

  // Simulation loop:
  while (!mujoco_ui.windowShouldClose() && !stop_requested) {

    mjtNum simstart = mj_data_ptr->time;
    while( mj_data_ptr->time - simstart < 1.0/framerate && !stop_requested ) {

      auto start_sleep = std::chrono::steady_clock::now();

      Eigen::VectorXd actual_output = Eigen::VectorXd::Zero(3 + G1_NUM_MOTOR + 3 + G1_NUM_MOTOR + 6 + 6);
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
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          actual_output[3 + i] = motor_state_data.q[i];
          actual_output[3 + G1_NUM_MOTOR + 3 + i] = motor_state_data.dq[i];
        }
        measured_motor_state = motor_state_data;
        has_measured_motor_state = has_lowstate_data;
        actual_output[3 + G1_NUM_MOTOR] = imu_state_data.omega[0];
        actual_output[3 + G1_NUM_MOTOR + 1] = imu_state_data.omega[1];
        actual_output[3 + G1_NUM_MOTOR + 2] = imu_state_data.omega[2];
        // actual_output[3 + 3 + 2 * G1_NUM_MOTOR] = imu_state_data.accelerometer[0];
        // actual_output[3 + 3 + 2 * G1_NUM_MOTOR + 1] = imu_state_data.accelerometer[1];
        // actual_output[3 + 3 + 2 * G1_NUM_MOTOR + 2] = imu_state_data.accelerometer[2];

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
          const int idx = getControlledJointIndex(joint_name);
          if (idx < 0) {
            continue;
          }

          actual_output[3 + idx] = mj_data_ptr->qpos[qpos_adr];
          actual_output[3 + G1_NUM_MOTOR + 3 + idx] = mj_data_ptr->qvel[dof_adr];
          measured_motor_state.q[idx] = static_cast<float>(mj_data_ptr->qpos[qpos_adr]);
          measured_motor_state.dq[idx] = static_cast<float>(mj_data_ptr->qvel[dof_adr]);
          measured_motor_state.ddq[idx] = static_cast<float>(mj_data_ptr->qacc[dof_adr]);
          measured_motor_state.tau_est[idx] = static_cast<float>(
              mj_data_ptr->qfrc_actuator[dof_adr]
          );
        }

        has_measured_motor_state = true;
        motor_state_data = measured_motor_state;
        actual_output[3 + G1_NUM_MOTOR] = robot_state.angular_velocity.x();
        actual_output[3 + G1_NUM_MOTOR + 1] = robot_state.angular_velocity.y();
        actual_output[3 + G1_NUM_MOTOR + 2] = robot_state.angular_velocity.z();
        imu_accelerometer = Eigen::Vector3d::Zero();
      }

      // std::cout << imu_state_data.rpy[0] << " " << imu_state_data.rpy[1] << " " << imu_state_data.rpy[2] << std::endl;
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

      if (enable_joint_pd_hold_test) {
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          const int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          const char* joint_name_cstr = mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id);
          if (joint_name_cstr == nullptr) {
            continue;
          }
          const std::string joint_name(joint_name_cstr);
          if (getControlledJointIndex(joint_name) < 0) {
            continue;
          }
          const int qpos_adr = mj_model_ptr->jnt_qposadr[joint_id];
          const int dof_adr = mj_model_ptr->jnt_dofadr[joint_id];
          const double q_target = initial_joint_targets[joint_name];

          double kp_hold = 80.0;
          double kd_hold = 5.0;
          if (joint_name.find("_knee_joint") != std::string::npos) {
            kp_hold = 120.0;
            kd_hold = 7.0;
          } else if (joint_name.find("_ankle_") != std::string::npos) {
            kp_hold = 60.0;
            kd_hold = 4.0;
          } else if (joint_name.find("waist_") != std::string::npos) {
            kp_hold = 40.0;
            kd_hold = 3.0;
          } else if (joint_name.find("_shoulder_") != std::string::npos ||
                     joint_name.find("_elbow_joint") != std::string::npos ||
                     joint_name.find("_wrist_") != std::string::npos) {
            kp_hold = 20.0;
            kd_hold = 1.5;
          }

          const double tau_hold =
              kp_hold * (q_target - mj_data_ptr->qpos[qpos_adr]) -
              kd_hold * mj_data_ptr->qvel[dof_adr];
          joint_command[joint_name] = clampJointTorque(joint_name, tau_hold, 1.0);
        }
      } else {
        walking_manager.update(robot_state, joint_command, actual_output);
      }

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

      if (enable_waist_yaw_sine_test) {
        const double test_end = waist_yaw_test_start_sec + waist_yaw_test_duration_sec;
        if (mj_data_ptr->time >= waist_yaw_test_start_sec && mj_data_ptr->time <= test_end) {
          const double t = mj_data_ptr->time - waist_yaw_test_start_sec;
          const double two_pi = 6.28318530717958647692;
          const double tau_test = waist_yaw_test_amp_nm * std::sin(two_pi * waist_yaw_test_freq_hz * t);
          const double tau_cmd = std::clamp(
              tau_test,
              -waist_yaw_test_max_tau_nm,
              waist_yaw_test_max_tau_nm
          );
          joint_command["waist_yaw_joint"] = tau_cmd;
        }
      }


      if (true){
        auto start_integration = std::chrono::steady_clock::now();
        mj_step1(mj_model_ptr, mj_data_ptr);
  
        double max_abs_controlled_ctrl = 0.0;
        double max_abs_raw_controlled_ctrl = 0.0;
        double max_abs_hand_hold_ctrl = 0.0;
        int nonzero_controlled_ctrl_count = 0;
        int nonzero_hand_hold_ctrl_count = 0;
        int saturated_controlled_ctrl_count = 0;
        double waist_yaw_ctrl_diag = 0.0;
        double waist_roll_ctrl_diag = 0.0;
        double waist_pitch_ctrl_diag = 0.0;
        std::string max_raw_controlled_joint = "NA";
        const double hand_hold_kp = 8.0;
        const double hand_hold_kd = 0.35;
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          const char* joint_name_cstr = mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id);
          if (joint_name_cstr == nullptr) {
            mj_data_ptr->ctrl[i] = 0.0;
            continue;
          }
          std::string joint_name = std::string(joint_name_cstr);
          const int controlled_idx = getControlledJointIndex(joint_name);
          double tau_cmd = 0.0;
          double tau_cmd_raw = 0.0;
          if (controlled_idx >= 0) {
            tau_cmd_raw = joint_command[joint_name];
            tau_cmd = clampJointTorque(joint_name, tau_cmd_raw, mujoco_torque_limit_scale);
          } else {
            auto fixed_hand_it = fixed_hand_joint_targets.find(joint_name);
            if (fixed_hand_it != fixed_hand_joint_targets.end()) {
              const int qpos_adr = mj_model_ptr->jnt_qposadr[joint_id];
              const int dof_adr = mj_model_ptr->jnt_dofadr[joint_id];
              tau_cmd = hand_hold_kp * (fixed_hand_it->second - mj_data_ptr->qpos[qpos_adr])
                        - hand_hold_kd * mj_data_ptr->qvel[dof_adr];
              tau_cmd = clampJointTorque(joint_name, tau_cmd, 1.0);
              const double abs_tau = std::abs(tau_cmd);
              max_abs_hand_hold_ctrl = std::max(max_abs_hand_hold_ctrl, abs_tau);
              if (abs_tau > 1e-6) {
                ++nonzero_hand_hold_ctrl_count;
              }
            }
          }
          mj_data_ptr->ctrl[i] = tau_cmd;

          if (controlled_idx >= 0) {
            const double abs_raw_tau = std::abs(tau_cmd_raw);
            const double abs_tau = std::abs(tau_cmd);
            if (abs_raw_tau > max_abs_raw_controlled_ctrl) {
              max_abs_raw_controlled_ctrl = abs_raw_tau;
              max_raw_controlled_joint = joint_name;
            }
            if (std::abs(tau_cmd_raw - tau_cmd) > 1e-9) {
              ++saturated_controlled_ctrl_count;
            }
            max_abs_controlled_ctrl = std::max(max_abs_controlled_ctrl, abs_tau);
            if (abs_tau > 1e-6) {
              ++nonzero_controlled_ctrl_count;
            }
            if (joint_name == "waist_yaw_joint") {
              waist_yaw_ctrl_diag = tau_cmd;
            } else if (joint_name == "waist_roll_joint") {
              waist_roll_ctrl_diag = tau_cmd;
            } else if (joint_name == "waist_pitch_joint") {
              waist_pitch_ctrl_diag = tau_cmd;
            }
          }
        }
        if (last_ctrl_diag_time < 0.0 || (mj_data_ptr->time - last_ctrl_diag_time) >= 1.0) {
          last_ctrl_diag_time = mj_data_ptr->time;
          std::cout << "[ctrl-diag] t=" << mj_data_ptr->time
                    << " nonzero_controlled=" << nonzero_controlled_ctrl_count
                    << "/" << G1_NUM_MOTOR
                    << " max_abs_ctrl=" << max_abs_controlled_ctrl
                    << " max_abs_raw_ctrl=" << max_abs_raw_controlled_ctrl
                    << " max_raw_joint=" << max_raw_controlled_joint
                    << " saturated=" << saturated_controlled_ctrl_count
                    << " waist_ctrl=[" << waist_yaw_ctrl_diag
                    << ", " << waist_roll_ctrl_diag
                    << ", " << waist_pitch_ctrl_diag << "]"
                    << " hand_hold_nonzero=" << nonzero_hand_hold_ctrl_count
                    << "/" << fixed_hand_joint_targets.size()
                    << " hand_hold_max_abs=" << max_abs_hand_hold_ctrl
                    << std::endl;
        }
  
        mj_step2(mj_model_ptr, mj_data_ptr);

        auto end_integration = std::chrono::steady_clock::now();
        auto integration_duration = end_integration - start_integration;
        if(integration_duration > std::chrono::milliseconds(1))
          std::cout << "Warning: integration took too long: " << std::chrono::duration_cast<std::chrono::microseconds>(integration_duration).count() << " us" << std::endl;
        robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
        if (last_runtime_stance_diag_time < 0.0 ||
            (mj_data_ptr->time - last_runtime_stance_diag_time) >= 1.0) {
          last_runtime_stance_diag_time = mj_data_ptr->time;
          printRuntimeStanceDiagnostics(mj_model_ptr, mj_data_ptr);
        }

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

      if (waist_yaw_joint_id >= 0 && waist_yaw_actuator_idx >= 0 &&
          (last_waist_diag_time < 0.0 || (mj_data_ptr->time - last_waist_diag_time) >= 1.0)) {
        last_waist_diag_time = mj_data_ptr->time;
        double waist_tau_wbc = 0.0;
        bool waist_tau_wbc_available = false;
        try {
          const auto& joint_command_const = static_cast<const labrob::JointCommand&>(joint_command);
          waist_tau_wbc = joint_command_const["waist_yaw_joint"];
          waist_tau_wbc_available = true;
        } catch (const std::exception&) {
          waist_tau_wbc_available = false;
        }

        const double waist_q = (waist_yaw_qpos_adr >= 0) ? mj_data_ptr->qpos[waist_yaw_qpos_adr] : 0.0;
        const double waist_dq = (waist_yaw_dof_adr >= 0) ? mj_data_ptr->qvel[waist_yaw_dof_adr] : 0.0;
        const double waist_tau_actuator = (waist_yaw_dof_adr >= 0) ? mj_data_ptr->qfrc_actuator[waist_yaw_dof_adr] : 0.0;
        const double waist_ctrl = mj_data_ptr->ctrl[waist_yaw_actuator_idx];

        std::cout << "[waist-diag] t=" << mj_data_ptr->time
                  << " q=" << waist_q
                  << " dq=" << waist_dq
                  << " ctrl=" << waist_ctrl
                  << " tau_act=" << waist_tau_actuator;
        if (waist_tau_wbc_available) {
          std::cout << " tau_wbc=" << waist_tau_wbc;
        } else {
          std::cout << " tau_wbc=NA";
        }
        std::cout << std::endl;
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

	      estimate_force.update(robot_state);
	      left_wrist_wrench_log.push_back(estimate_force.getLeftWristWrench());
	      right_wrist_wrench_log.push_back(estimate_force.getRightWristWrench());
	      left_wrist_wrench_filtered_log.push_back(estimate_force.getLeftWristWrenchFiltered());
	      right_wrist_wrench_filtered_log.push_back(estimate_force.getRightWristWrenchFiltered());

	      {
	        labrob::ComplianceReferenceGenerator::Input compliance_input;
	        const double current_time = mj_data_ptr->time;
	        if (previous_compliance_time < 0.0) {
	          const auto controller_frequency = walking_manager.get_controller_frequency();
	          compliance_input.dt = (controller_frequency > 0)
	              ? 1.0 / static_cast<double>(controller_frequency)
	              : 0.002;
	        } else {
	          compliance_input.dt = std::max(1e-6, current_time - previous_compliance_time);
	        }
	        previous_compliance_time = current_time;

	        ComplianceVector6d torso_numeric_wrench_left = ComplianceVector6d::Zero();
	        ComplianceVector6d torso_numeric_wrench_right = ComplianceVector6d::Zero();
	        ComplianceVector6d torso_numeric_manual_delta_left = ComplianceVector6d::Zero();
	        ComplianceVector6d torso_numeric_manual_delta_right = ComplianceVector6d::Zero();
	        ComplianceMatrix6d torso_numeric_Jb_left = ComplianceMatrix6d::Zero();
	        ComplianceMatrix6d torso_numeric_Jb_right = ComplianceMatrix6d::Zero();
	        bool torso_numeric_Jb_valid = false;

	        if (enable_torso_compliance_numeric_test) {
	          if (!enable_torso_compliance_zero_input_test) {
	            const Eigen::VectorXd& left_wrench_filtered =
	                estimate_force.getLeftWristWrenchFiltered();
	            const Eigen::VectorXd& right_wrench_filtered =
	                estimate_force.getRightWristWrenchFiltered();
	            if (left_wrench_filtered.size() >= 6) {
	              torso_numeric_wrench_left = left_wrench_filtered.head<6>();
	            }
	            if (right_wrench_filtered.size() >= 6) {
	              torso_numeric_wrench_right = right_wrench_filtered.head<6>();
	            }

	            torso_numeric_manual_delta_left = computeQuasiStaticComplianceDelta(
	                torso_numeric_wrench_left,
	                compliance_params.Ka_left,
	                compliance_params.S_left);
	            torso_numeric_manual_delta_right = computeQuasiStaticComplianceDelta(
	                torso_numeric_wrench_right,
	                compliance_params.Ka_right,
	                compliance_params.S_right);
	          }

	          compliance_input.use_manual_delta_xc = true;
	          compliance_input.manual_delta_xc_left = torso_numeric_manual_delta_left;
	          compliance_input.manual_delta_xc_right = torso_numeric_manual_delta_right;
	          torso_numeric_Jb_valid = computeTorsoToWristJacobians(
	              walking_manager.getRobotModel(),
	              robot_state,
	              torso_numeric_Jb_left,
	              torso_numeric_Jb_right);
	          compliance_input.Jb_left = torso_numeric_Jb_left;
	          compliance_input.Jb_right = torso_numeric_Jb_right;
	          if (!torso_numeric_Jb_valid) {
	            static bool warned_jacobian_failure = false;
	            if (!warned_jacobian_failure) {
	              warned_jacobian_failure = true;
	              std::cout << "Warning: failed to compute torso-to-wrist Jacobians; using zero Jb for torso compliance numeric test." << std::endl;
	            }
	          }
	          compliance_input.wrench_left = torso_numeric_wrench_left;
	          compliance_input.wrench_right = torso_numeric_wrench_right;
	        } else {
	          compliance_input.wrench_left = estimate_force.getLeftWristWrenchFiltered();
	          compliance_input.wrench_right = estimate_force.getRightWristWrenchFiltered();
	        }
	        compliance_input.wrench_left_ref.setZero();
	        compliance_input.wrench_right_ref.setZero();

	        const auto compliance_output = compliance_reference_generator.update(compliance_input);

	        compliance_time_log.push_back(current_time);
	        compliance_delta_x_left_log.push_back(compliance_output.delta_xc_left);
	        compliance_delta_x_right_log.push_back(compliance_output.delta_xc_right);
	        compliance_delta_dx_left_log.push_back(compliance_output.delta_dxc_left);
	        compliance_delta_dx_right_log.push_back(compliance_output.delta_dxc_right);
	        compliance_delta_ddx_left_log.push_back(compliance_output.delta_ddxc_left);
	        compliance_delta_ddx_right_log.push_back(compliance_output.delta_ddxc_right);

	        if (enable_torso_compliance_numeric_test) {
	          compliance_torso_time_log.push_back(current_time);
	          compliance_torso_wrench_left_log.push_back(torso_numeric_wrench_left);
	          compliance_torso_wrench_right_log.push_back(torso_numeric_wrench_right);
	          compliance_torso_manual_delta_xc_left_log.push_back(torso_numeric_manual_delta_left);
	          compliance_torso_manual_delta_xc_right_log.push_back(torso_numeric_manual_delta_right);
	          compliance_torso_delta_xc_left_log.push_back(compliance_output.delta_xc_left);
	          compliance_torso_delta_xc_right_log.push_back(compliance_output.delta_xc_right);
	          compliance_torso_delta_xc_left_filtered_log.push_back(compliance_output.delta_xc_left_filtered);
	          compliance_torso_delta_xc_right_filtered_log.push_back(compliance_output.delta_xc_right_filtered);
	          compliance_torso_delta_xb_log.push_back(compliance_output.delta_xb);
	          compliance_torso_delta_xb_filtered_log.push_back(compliance_output.delta_xb_filtered);
	          compliance_torso_delta_xb_final_log.push_back(compliance_output.delta_xb_final);
	          compliance_torso_qp_solved_log.push_back(compliance_output.qp_solved ? 1 : 0);
	          compliance_torso_Jb_left_log.push_back(torso_numeric_Jb_left);
	          compliance_torso_Jb_right_log.push_back(torso_numeric_Jb_right);
	          compliance_torso_Jb_valid_log.push_back(torso_numeric_Jb_valid ? 1 : 0);

	          if (enable_torso_compliance_wbc_test) {
	            Eigen::Vector3d torso_rpy_offset = Eigen::Vector3d::Zero();
	            if (compliance_output.qp_solved) {
	              torso_rpy_offset = compliance_output.delta_xb_final.tail<3>();
	            }
	            walking_manager.setTorsoComplianceReference(
	                torso_rpy_offset,
	                Eigen::Vector3d::Zero(),
	                Eigen::Vector3d::Zero());
	          }
	        }
	      }
      
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
      
        // Impostazioni di base
        motor_command.tau_ff.fill(0.0f);
        motor_command.q_target.fill(0.0f);
        motor_command.dq_target.fill(0.0f);
        motor_command.kp.fill(0.0f);
        motor_command.kd.fill(0.0f);

        // impose kp and kd to increase linearly with time
        if (mj_data_ptr->time < 5.0f) {
          for (int i = 0; i < mj_model_ptr->nu; ++i) {
            const int joint_id = mj_model_ptr->actuator_trnid[i * 2];
            const char* joint_name_cstr = mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id);
            if (joint_name_cstr == nullptr) {
              continue;
            }
            const int robot_idx = getControlledJointIndex(joint_name_cstr);
            if (robot_idx < 0) {
              continue;
            }
            motor_command.kp[robot_idx] = Kp[robot_idx] * (mj_data_ptr->time / 5.0f);
            motor_command.kd[robot_idx] = Kd[robot_idx];
          }
        }
        else{
          motor_command.kp = Kp;
          motor_command.kd = Kd;
        }

        // assegna i valori di controllo per i giunti
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          const int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          const char* joint_name_cstr = mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id);
          if (joint_name_cstr == nullptr) {
            continue;
          }
          std::string joint_name = std::string(joint_name_cstr);
          const int robot_idx = getControlledJointIndex(joint_name);
          if (robot_idx < 0) {
            continue;
          }

          // if the values are too big in module, turn off the robot
          if (std::abs(robot_state.joint_state[joint_name].pos) > 2 || std::abs(robot_state.joint_state[joint_name].vel) > 15 || std::abs(joint_command[joint_name]) > 100.0)  {
            std::cout << "Warning: motor command values too high for joint " << joint_name << ": "
                      << "q_target = " << robot_state.joint_state[joint_name].pos << ", "
                      << "dq_target = " << robot_state.joint_state[joint_name].vel << ", "
                      << "tau_ff = " << joint_command[joint_name] << std::endl;
            std::cout << "Disabling robot for safety." << std::endl;

            signalHandler(SIGINT);
          }else {
            motor_command.q_target[robot_idx] = robot_state.joint_state[joint_name].pos;
            motor_command.dq_target[robot_idx] = robot_state.joint_state[joint_name].vel;
            motor_command.tau_ff[robot_idx] = joint_command[joint_name];
          }
        }
      
        // Costruisci comando DDS
        LowCmd_ dds_low_command;
        dds_low_command.mode_pr() = static_cast<uint8_t>(Mode::PR);
        dds_low_command.mode_machine() = mode_machine_;
      
        for (int i = 0; i < mj_model_ptr->nu; ++i) {
          const int joint_id = mj_model_ptr->actuator_trnid[i * 2];
          const char* joint_name_cstr = mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id);
          if (joint_name_cstr == nullptr) {
            continue;
          }
          const int robot_idx = getControlledJointIndex(joint_name_cstr);
          if (robot_idx < 0) {
            continue;
          }
          auto &cmd = dds_low_command.motor_cmd().at(robot_idx);
          cmd.mode() = 1;
          cmd.q()    = motor_command.q_target[robot_idx];
          cmd.dq()   = motor_command.dq_target[robot_idx];
          cmd.tau()  = motor_command.tau_ff[robot_idx];
          cmd.kp()   = motor_command.kp[robot_idx];
          cmd.kd()   = motor_command.kd[robot_idx];
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

	      if (stop_time_sec > 0.0 && mj_data_ptr->time >= stop_time_sec) {
	        stop_requested = true;
	      }
	    }

	    if (stop_requested) {
	      break;
	    }

	    auto start_render =  std::chrono::steady_clock::now();

    mujoco_ui.render();

    auto end_render =  std::chrono::steady_clock::now();
    auto render_duration = end_render - start_render;
    if(render_duration > std::chrono::milliseconds(5))
      std::cout << "Warning: rendering took too long: " << std::chrono::duration_cast<std::chrono::microseconds>(render_duration).count() << " us" << std::endl;

	  }

	  if (auto_save_logs) {
	    std::cout << "Auto-saving logs..." << std::endl;
	    walking_manager.saveLogs();
	    saveEstimateForceLogs();
	    std::cout << "Logs saved." << std::endl;
	  }

	  // Free memory (Mujoco):
	  mj_deleteData(mj_data_ptr);
  mj_deleteModel(mj_model_ptr);

  return 0;
}
