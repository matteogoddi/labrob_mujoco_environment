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

extern double torso_spring_kp;
extern double torso_spring_kd;
extern double torso_spring_weight;
extern double waist_yaw_compliance_kp;
extern double waist_yaw_compliance_kd;
extern double waist_roll_compliance_kp;
extern double waist_roll_compliance_kd;
extern double waist_pitch_compliance_kp;
extern double waist_pitch_compliance_kd;

extern Eigen::Vector3d imu_accelerometer;
#endif // GLOBALS_H
