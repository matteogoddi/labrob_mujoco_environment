#ifndef GLOBALS_H
#define GLOBALS_H

extern bool isWBCLoopClosed;
extern bool isEKFactive;
extern bool isMPCLoopClosed;
extern bool useSim;
extern bool useRobot;
extern bool switchWalkingState;
extern bool oneTimepress;
extern bool loopClosed;

extern double startTimeMPCCL;
extern double startTimeWBCCL;
extern double startTimeEKF;

extern Eigen::Vector3d go_base_position;
extern Eigen::Vector3d go_base_velocity;
extern Eigen::Vector4d go_imu_quaternion;
extern Eigen::Vector3d go_imu_rpy;
extern Eigen::Vector3d go_imu_omega;
extern Eigen::Vector3d go_imu_accelerometer;

extern Eigen::VectorXd measured_joint_position;
extern Eigen::VectorXd measured_joint_velocity;
extern Eigen::Vector4d measured_imu_quaternion;
extern Eigen::Vector3d measured_imu_rpy;
extern Eigen::Vector3d measured_imu_omega;
extern Eigen::Vector3d measured_imu_accelerometer;

#endif // GLOBALS_H