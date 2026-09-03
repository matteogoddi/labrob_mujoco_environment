// std
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <csignal>
#include <chrono>

// Pinocchio (must come before RobotState.hpp)
#include <pinocchio/multibody/model.hpp>

// Labrob
#include <JointCommand.hpp>
#include <Logger.hpp>
#include <RobotState.hpp>
#include <StateEstimator.hpp>
#include <WalkingManager.hpp>
#include <utils.hpp>

#include <globals.h>
#include "MujocoUI.hpp"

// ── Globals required by WalkingManager (via globals.h) ───────────────────────
bool isMPCLoopClosed    = true;
bool isObserverActive   = false;
bool switchWalkingState = false;  // set from the MuJoCo UI "Start steps" button
int  ZMP_TYPE           = 3;

double uiStepLengthX  = 0.1;
double uiStepLengthY  = 0.0;
double uiYawIncrement = 0.0;
double uiDoubleSupportDuration = 2000;
double uiSingleSupportDuration = 2000;

Eigen::VectorXd measured_joint_velocity = Eigen::VectorXd::Zero(kNumControlledMotors);

using Clock = std::chrono::steady_clock;

alignas(EIGEN_MAX_ALIGN_BYTES) labrob::WalkingManager walking_manager;
labrob::Logger logger_;

static volatile sig_atomic_t g_signal_received = 0;

// ── External force schedule ───────────────────────────────────────────────────
struct ForceEvent {
    double          t_start;
    double          t_end;
    const char*     body;
    Eigen::Vector3d force;        // world frame [N]
    Eigen::Vector3d torque;       // world frame [Nm]
    Eigen::Vector3d local_offset; // offset from body CoM in body frame [m]
};

// Edit this list to change the disturbance scenario.
static const std::vector<ForceEvent> force_schedule = {
    // { t_start, t_end, body,              force [N],        torque,    local offset [m] }
    // {  5.15,  5.5,  "torso_link",           { 0,  0, -80}, {0,0,0}, {0, 0, 0.1} },
    // {  8.0,  18.0,  "left_wrist_yaw_link",  { 3,  0,   0}, {0,0,0}, {0, 0, 0  } },
    // {  8.0,  18.0,  "right_wrist_yaw_link", { 3,  0,   0}, {0,0,0}, {0, 0, 0  } },
};

// Must be called after mj_step1 and before mj_step2.
static void apply_force_schedule(mjModel* m, mjData* d,
                                 const std::vector<ForceEvent>& events)
{
    mju_zero(d->qfrc_applied, m->nv);
    const double t = d->time;
    for (const auto& e : events) {
        if (t < e.t_start || t >= e.t_end) continue;
        const int bid = mj_name2id(m, mjOBJ_BODY, e.body);
        if (bid < 0) continue;
        // world-frame application point = body CoM + R_body * local_offset
        double point[3];
        mju_rotVecMat(point, e.local_offset.data(), d->xmat + 9 * bid);
        point[0] += d->xpos[3 * bid + 0];
        point[1] += d->xpos[3 * bid + 1];
        point[2] += d->xpos[3 * bid + 2];
        mj_applyFT(m, d, e.force.data(), e.torque.data(), point, bid, d->qfrc_applied);
    }
}

// Returns the total force currently active on a given body (for logging/viz).
static Eigen::Vector3d active_force_on(const char* body, double t,
                                       const std::vector<ForceEvent>& events)
{
    Eigen::Vector3d total = Eigen::Vector3d::Zero();
    for (const auto& e : events)
        if (std::string(e.body) == body && t >= e.t_start && t < e.t_end)
            total += e.force;
    return total;
}

void signalHandler(int signum) {
    g_signal_received = signum;  // only async-signal-safe operation
}

labrob::RobotState robot_state_from_mujoco(mjModel* m, mjData* d) {
    labrob::RobotState rs;
    rs.position    = Eigen::Vector3d(d->qpos[0], d->qpos[1], d->qpos[2]);
    rs.orientation = Eigen::Quaterniond(d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]);
    rs.linear_velocity  = rs.orientation.toRotationMatrix().transpose() *
                          Eigen::Vector3d(d->qvel[0], d->qvel[1], d->qvel[2]);
    rs.angular_velocity = Eigen::Vector3d(d->qvel[3], d->qvel[4], d->qvel[5]);

    for (int i = 1; i < m->njnt; ++i) {
        const char* name = mj_id2name(m, mjOBJ_JOINT, i);
        rs.joint_state[name].pos = d->qpos[m->jnt_qposadr[i]];
        rs.joint_state[name].vel = d->qvel[m->jnt_dofadr[i]];
    }

    static double force[6], result[3];
    rs.contact_points.resize(d->ncon);
    rs.contact_forces.resize(d->ncon);
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (int i = 0; i < d->ncon; ++i) {
        mj_contactForce(m, d, i, force);
        for (int r = 0; r < 3; ++r) {
            result[r] = 0;
            for (int c = 0; c < 3; ++c)
                result[r] += d->contact[i].frame[3 * c + r] * force[c];
        }
        sum += Eigen::Vector3d(result);
        for (int j = 0; j < 3; ++j) {
            rs.contact_points[i](j) = d->contact[i].pos[j];
            rs.contact_forces[i](j) = result[j];
        }
    }
    rs.total_force = sum;
    return rs;
}

// Orientation of a MuJoCo site, as read from its rotation matrix (site_xmat).
Eigen::Quaterniond site_orientation(mjData* d, int site_id) {
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = d->site_xmat[9 * site_id + 3 * r + c];
    return Eigen::Quaterniond(R);
}

int main(const int argc, const char* argv[]) {
    bool reactiveStanding = false;
    bool verboseCoop      = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--walk")    reactiveStanding = false;
        else if (a == "--verbose") verboseCoop = true;
    }

    signal(SIGINT, signalHandler);

    mj_loadAllPluginLibraries("/usr/local/lib/mujoco", nullptr);
    const int kErrorLength = 1024;
    char loadError[kErrorLength] = "";
    mjModel* mj_model_ptr = mj_loadXML(kRobotConfig.scene_path.data(), nullptr, loadError, kErrorLength);
    if (!mj_model_ptr) {
        std::cerr << "mj_loadXML failed: " << loadError << std::endl;
        return -1;
    }
    mjData* mj_data_ptr = mj_makeData(mj_model_ptr);

    // Initial joint configuration
    for (int i = 0; i < mj_model_ptr->nq; ++i) mj_data_ptr->qpos[i] = 0.0;
    mj_data_ptr->qpos[2] = kRobotConfig.initial_pelvis_height;
    // More upright pose: legs are ~6.29cm taller hip-to-ankle (computed from
    // the URDF leg chain for knee 0.95->0.30, hip/ankle pitch -0.44/-0.50->
    // -0.15/-0.15), so the pelvis must start that much higher to keep the
    // feet on the ground.
    // mj_data_ptr->qpos[2] = 1.0;
    mj_data_ptr->qpos[3] = 1;
    for (int i = 0; i < mj_model_ptr->njnt; ++i) {
        const char* name = mj_id2name(mj_model_ptr, mjOBJ_JOINT, i);
        auto it = kRobotConfig.initial_joint_positions.find(name);
        if (it != kRobotConfig.initial_joint_positions.end())
            mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[i]] = it->second;
    }

    std::map<std::string, double> armatures;
    std::vector<std::string> ordered_joint_names(mj_model_ptr->nu);
    for (int i = 0; i < mj_model_ptr->nu; ++i) {
        int joint_id = mj_model_ptr->actuator_trnid[i * 2];
        std::string joint_name = mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id);
        armatures[joint_name] = mj_model_ptr->dof_armature[mj_model_ptr->jnt_dofadr[joint_id]];
        ordered_joint_names[i] = joint_name;
    }

    labrob::RobotState robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
    walking_manager.setReactiveStanding(reactiveStanding);
    walking_manager.setVerboseCoop(verboseCoop);
    walking_manager.init(robot_state, armatures);

    // Pelvis + torso IMU sensor indices (gyro then accelerometer) and their
    // sites, used to read back orientation (quat/rpy) as seen by the IMU.
    const int gyro_sid       = mj_name2id(mj_model_ptr, mjOBJ_SENSOR, "imu-pelvis-angular-velocity");
    const int acc_sid        = mj_name2id(mj_model_ptr, mjOBJ_SENSOR, "imu-pelvis-linear-acceleration");
    const int torso_gyro_sid = mj_name2id(mj_model_ptr, mjOBJ_SENSOR, "imu-torso-angular-velocity");
    const int torso_acc_sid  = mj_name2id(mj_model_ptr, mjOBJ_SENSOR, "imu-torso-linear-acceleration");
    const int pelvis_site_id = mj_name2id(mj_model_ptr, mjOBJ_SITE,   "imu_in_pelvis");
    const int torso_site_id  = mj_name2id(mj_model_ptr, mjOBJ_SITE,   "imu_in_torso");

    labrob::NoiseParams ri_noise;
    ri_noise.gyro_noise    = 0.01;
    ri_noise.accel_noise   = 0.01;
    ri_noise.contact_noise = 0.01;
    ri_noise.gyro_bias_rw  = 0.001;
    ri_noise.accel_bias_rw = 0.001;
    ri_noise.encoder_noise = 0.001;

    labrob::StateEstimator state_estimator(
        walking_manager.get_robot_model(),
        1.0 / walking_manager.get_controller_frequency(),
        ri_noise
    );

    // Activate immediately so the filter warms up from t=0.
    // Its output is only applied to robot_state after 5 s.
    {
        const auto& model = walking_manager.get_robot_model();
        const int njnt = model.nv - 6;
        Eigen::VectorXd q_joints(njnt);
        for (int i = 0; i < njnt; ++i)
            q_joints(i) = robot_state.joint_state.at(model.names[i + 2]).pos;
        state_estimator.activate(robot_state, q_joints);
    }

    labrob::MujocoUI* mujoco_ui_ptr = labrob::MujocoUI::getInstance(mj_model_ptr, mj_data_ptr);
    static constexpr int framerate = 60;

    while (!(g_signal_received || mujoco_ui_ptr->windowShouldClose())) {
        if (mujoco_ui_ptr->windowShouldClose()) break;

        mjtNum simstart = mj_data_ptr->time;
        while (!g_signal_received &&
               std::isfinite(mj_data_ptr->time) &&
               mj_data_ptr->time - simstart < 1.0 / framerate) {

            logger_.log("gt_left_wrist_force",  active_force_on("left_wrist_yaw_link",  mj_data_ptr->time, force_schedule));
            logger_.log("gt_right_wrist_force", active_force_on("right_wrist_yaw_link", mj_data_ptr->time, force_schedule));

            Eigen::VectorXd measured_joint_position(mj_model_ptr->nu);
            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                const std::string jname = mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid);
                measured_joint_position[i] = robot_state.joint_state.at(jname).pos;
                measured_joint_velocity[i] = robot_state.joint_state.at(jname).vel;
            }
            logger_.log("joint_pos", measured_joint_position);
            logger_.log("joint_vel", measured_joint_velocity);
            // The state estimator has no joint-position estimate, only the joint
            // velocity it takes as input: log that same feedback as "filtered".
            logger_.log("filtered_joint_velocity", measured_joint_velocity);

            labrob::JointCommand joint_command;
            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)] = 0.0;
            }

            // Read pelvis + torso IMU from MuJoCo sensor data.
            // MuJoCo's accelerometer already outputs specific force (like a real IMU):
            // at rest it reads {0, 0, 9.81} m/s² — no gravity correction needed.
            const Eigen::Vector3d imu_gyro(mj_data_ptr->sensordata + mj_model_ptr->sensor_adr[gyro_sid]);
            const Eigen::Vector3d imu_acc(mj_data_ptr->sensordata  + mj_model_ptr->sensor_adr[acc_sid]);
            const Eigen::Vector3d torso_imu_gyro(mj_data_ptr->sensordata + mj_model_ptr->sensor_adr[torso_gyro_sid]);
            const Eigen::Vector3d torso_imu_acc(mj_data_ptr->sensordata  + mj_model_ptr->sensor_adr[torso_acc_sid]);

            // Odometry: ground-truth base pose/velocity from the robot state (no
            // separate odometry estimator exists in simulation).
            logger_.log("odom_pos",  robot_state.position);
            logger_.log("odom_vel",  robot_state.linear_velocity);
            logger_.log("odom_quat", Eigen::Vector4d(
                robot_state.orientation.w(), robot_state.orientation.x(),
                robot_state.orientation.y(), robot_state.orientation.z()));
            logger_.log("odom_rpy",  labrob::rpyFromQuaternion(robot_state.orientation));

            // IMU orientation, read from the pelvis/torso sites as seen by MuJoCo.
            const Eigen::Quaterniond pelvis_quat = site_orientation(mj_data_ptr, pelvis_site_id);
            const Eigen::Quaterniond torso_quat  = site_orientation(mj_data_ptr, torso_site_id);
            logger_.log("pelvis_gyro", imu_gyro);
            logger_.log("pelvis_acc",  imu_acc);
            logger_.log("pelvis_quat", Eigen::Vector4d(pelvis_quat.w(), pelvis_quat.x(), pelvis_quat.y(), pelvis_quat.z()));
            logger_.log("pelvis_rpy",  labrob::rpyFromQuaternion(pelvis_quat));
            logger_.log("torso_gyro",  torso_imu_gyro);
            logger_.log("torso_acc",   torso_imu_acc);
            logger_.log("torso_quat",  Eigen::Vector4d(torso_quat.w(), torso_quat.x(), torso_quat.y(), torso_quat.z()));
            logger_.log("torso_rpy",   labrob::rpyFromQuaternion(torso_quat));

            // Always run one EKF step so the filter stays warm.
            // Before 5 s we propagate on a throw-away copy; after 5 s we update robot_state.
            labrob::RobotState rs_ekf = robot_state;
            if (mj_data_ptr->time >= 0.0) {
                const auto start_ekf = std::chrono::high_resolution_clock::now();
                state_estimator.update(
                    rs_ekf, imu_gyro, imu_acc,
                    walking_manager.get_contact()
                );
                const auto end_ekf = std::chrono::high_resolution_clock::now();
                const auto ekf_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_ekf - start_ekf).count();
                logger_.log("execution_time_ekf", static_cast<double>(ekf_duration));
            }

            // Print EKF vs GT comparison every 0.5 s (not 1 kHz)
            {
                static double last_print = -1.0;
                if (mj_data_ptr->time - last_print >= 0.5 && false) {
                    last_print = mj_data_ptr->time;
                    const Eigen::Vector3d pos_err = rs_ekf.position - robot_state.position;
                    const Eigen::Vector3d rpy_ekf = rs_ekf.orientation.toRotationMatrix().eulerAngles(2,1,0);
                    const Eigen::Vector3d rpy_gt  = robot_state.orientation.toRotationMatrix().eulerAngles(2,1,0);
                    std::printf("[t=%.2f]  pos_err=[%.4f %.4f %.4f]  "
                                "rpy_ekf=[%.3f %.3f %.3f]  rpy_gt=[%.3f %.3f %.3f]  "
                                "vel_ekf=[%.4f %.4f %.4f]\n",
                        mj_data_ptr->time,
                        pos_err.x(), pos_err.y(), pos_err.z(),
                        rpy_ekf.z(), rpy_ekf.y(), rpy_ekf.x(),
                        rpy_gt.z(),  rpy_gt.y(),  rpy_gt.x(),
                        rs_ekf.linear_velocity.x(), rs_ekf.linear_velocity.y(), rs_ekf.linear_velocity.z());
                }
            }
            static bool ekf_took_over = false;
            if (mj_data_ptr->time >= 3.0 && mujoco_ui_ptr->isFilterEnabled()) {
                robot_state = rs_ekf;
                if (!ekf_took_over) {
                    ekf_took_over = true;
                    walking_manager.init(robot_state, armatures);
                }
            }

            // Read gait parameters from the MuJoCo UI control panel.
            uiStepLengthX  = mujoco_ui_ptr->getStepLengthX();
            uiStepLengthY  = mujoco_ui_ptr->getStepLengthY();
            uiYawIncrement = mujoco_ui_ptr->getYawIncrement();
            uiDoubleSupportDuration = mujoco_ui_ptr->getDoubleSupportDuration();
            uiSingleSupportDuration = mujoco_ui_ptr->getSingleSupportDuration();
            if (mujoco_ui_ptr->consumeStartStepsPressed()) {
                switchWalkingState = true;
            }

            walking_manager.update(robot_state, joint_command);

            mj_step1(mj_model_ptr, mj_data_ptr);
            apply_force_schedule(mj_model_ptr, mj_data_ptr, force_schedule);

            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                mj_data_ptr->ctrl[i] = joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)];
            }
            // mujoco_ui_ptr->render();
            mj_step2(mj_model_ptr, mj_data_ptr);

            robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
        }

        // Build force arrows from all active events
        std::vector<labrob::MujocoUI::ForceArrow> arrows;
        const double t = mj_data_ptr->time;
        for (const auto& e : force_schedule) {
            if (t < e.t_start || t >= e.t_end) continue;
            const int bid = mj_name2id(mj_model_ptr, mjOBJ_BODY, e.body);
            if (bid < 0) continue;
            double point[3];
            mju_rotVecMat(point, e.local_offset.data(), mj_data_ptr->xmat + 9 * bid);
            labrob::MujocoUI::ForceArrow a;
            a.from  = Eigen::Vector3d(
                mj_data_ptr->xpos[3*bid+0] + point[0],
                mj_data_ptr->xpos[3*bid+1] + point[1],
                mj_data_ptr->xpos[3*bid+2] + point[2]);
            a.force = e.force;
            a.rgba[0] = 1.0f; a.rgba[1] = 0.3f; a.rgba[2] = 0.0f; a.rgba[3] = 0.9f;
            arrows.push_back(a);
        }
        mujoco_ui_ptr->renderWithForces(arrows);
    }

    mj_deleteData(mj_data_ptr);
    mj_deleteModel(mj_model_ptr);

    if (g_signal_received) {
        std::cout << "Do you want to save logs? [y/n] ";
        std::string input;
        std::getline(std::cin, input);
        if (input == "y" || input == "Y" || input == "yes") {
            walking_manager.saveLogs();
            logger_.save("/tmp/robot_logs");

            std::ofstream joint_names_file("/tmp/robot_logs/joint_names.txt");
            for (const auto& name : ordered_joint_names)
                joint_names_file << name << "\n";

            std::cout << "Logs saved." << std::endl;
        }
    }
    return 0;
}