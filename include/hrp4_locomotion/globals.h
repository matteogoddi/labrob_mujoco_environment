#ifndef GLOBALS_H
#define GLOBALS_H

extern bool isTotalBodyLoopClosed;
extern bool isEKFactive;
extern bool isCoMLoopClosed;
extern bool useSim;
extern bool useRobot;
extern bool switchWalkingState;
extern bool isIMUcalibrating;
extern bool oneTimepress;
extern bool loopClosed;

extern double startTimeCoMCL;
extern double startTimeTotalBodyCL;
extern double startTimeEKF;
extern double startTimeIMUcalibrating;

extern Eigen::Vector3d go_base_position;
extern Eigen::Vector3d go_base_velocity;
extern Eigen::Vector4d go_imu_quaternion;
extern Eigen::Vector3d go_imu_rpy;
extern Eigen::Vector3d go_imu_gyroscope;
extern Eigen::Vector3d go_imu_accelerometer;

extern Eigen::Vector3d imu_accelerometer;
#endif // GLOBALS_H