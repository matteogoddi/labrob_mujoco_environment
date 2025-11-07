#ifndef GLOBALS_H
#define GLOBALS_H

extern bool isTotalBodyLoopClosed;
extern bool isEKFLoopClosed;
extern bool isCoMLoopClosed;
extern bool useSim;
extern bool useRobot;
extern bool switchWalkingState;
extern bool calibratingImu;

extern double startTimeCoMCL;
extern double startTimeTotalBodyCL;
extern double startTimeEKFCL;

extern Eigen::Vector3d imu_accelerometer;
#endif // GLOBALS_H