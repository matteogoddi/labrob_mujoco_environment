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

extern double startTimeCoMCL;
extern double startTimeTotalBodyCL;
extern double startTimeEKF;
extern double startTimeIMUcalibrating;

extern Eigen::Vector3d imu_accelerometer;
#endif // GLOBALS_H