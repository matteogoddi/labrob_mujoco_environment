#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <tuple>
#include <utility>


constexpr std::string_view robotUrdfPath  = "../../labrob_mujoco_environment/robot/g1/g1_description/g1_29dof_rev_1_0.urdf";
constexpr std::string_view robotScenePath = "../../labrob_mujoco_environment/robot/g1/g1_mj_description/stair_steps.xml";

extern bool isWBCLoopClosed;
extern bool isEKFactive;
extern bool isMPCLoopClosed;
extern bool useSim;
extern bool useRobot;
extern bool switchWalkingState;
extern bool oneTimepress;
extern bool loopClosed;
extern bool imuCalibration;

extern double startTimeMPCCL;
extern double startTimeWBCCL;
extern double startTimeEKF;

extern Eigen::Vector3d odometry_base_position;
extern Eigen::Vector3d odometry_base_velocity;
extern Eigen::Vector4d odometry_imu_quaternion;
extern Eigen::Vector3d odometry_imu_rpy;

extern Eigen::VectorXd measured_joint_position;
extern Eigen::VectorXd measured_joint_velocity;

extern Eigen::Vector4d measured_imu_quaternion;
extern Eigen::Vector3d measured_imu_rpy;
extern Eigen::Vector3d measured_imu_angular_velocity;
extern Eigen::Vector3d measured_imu_accelerometer;

extern Eigen::Vector4d measured_imu_pelvis_quaternion;
extern Eigen::Vector3d measured_imu_pelvis_rpy;
extern Eigen::Vector3d measured_imu_pelvis_angular_velocity;
extern Eigen::Vector3d measured_imu_pelvis_accelerometer;

