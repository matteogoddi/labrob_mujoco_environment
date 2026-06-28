// std
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
#include <WalkingManager.hpp>

#include <globals.h>
#include <RobotConfig.hpp>
#include "MujocoUI.hpp"

// ── Globals required by WalkingManager (via globals.h) ───────────────────────
bool isMPCLoopClosed    = true;
bool isObserverActive   = true;
bool switchWalkingState = false;  // never set in sim, WalkingManager reads it

Eigen::VectorXd measured_joint_velocity = Eigen::VectorXd::Zero(29);

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
    // {  5.15,  5.2,  "torso_link",           { 0,  0, -100}, {0,0,0}, {0, 0, 0.1} },
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
    mjModel* mj_model_ptr = mj_loadXML(robotScenePath.data(), nullptr, loadError, kErrorLength);
    if (!mj_model_ptr) {
        std::cerr << "mj_loadXML failed: " << loadError << std::endl;
        return -1;
    }
    mjData* mj_data_ptr = mj_makeData(mj_model_ptr);

    // Initial joint configuration
    for (int i = 0; i < mj_model_ptr->nq; ++i) mj_data_ptr->qpos[i] = 0.0;
    mj_data_ptr->qpos[2] = 0.728112;
    mj_data_ptr->qpos[3] = 1;
    for (int i = 0; i < mj_model_ptr->njnt; ++i) {
        const char* name = mj_id2name(mj_model_ptr, mjOBJ_JOINT, i);
        auto it = joint_initial_positions.find(name);
        if (it != joint_initial_positions.end())
            mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[i]] = it->second;
    }

    std::map<std::string, double> armatures;
    for (int i = 0; i < mj_model_ptr->nu; ++i) {
        int joint_id = mj_model_ptr->actuator_trnid[i * 2];
        std::string joint_name = mj_id2name(mj_model_ptr, mjOBJ_JOINT, joint_id);
        armatures[joint_name] = mj_model_ptr->dof_armature[mj_model_ptr->jnt_dofadr[joint_id]];
    }

    labrob::RobotState robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
    walking_manager.setReactiveStanding(reactiveStanding);
    walking_manager.setVerboseCoop(verboseCoop);
    walking_manager.init(robot_state, armatures);

    labrob::MujocoUI* mujoco_ui_ptr = labrob::MujocoUI::getInstance(mj_model_ptr, mj_data_ptr);
    static constexpr int framerate = 60;

    while (true) {
        if (g_signal_received || mujoco_ui_ptr->windowShouldClose()) break;

        mjtNum simstart = mj_data_ptr->time;
        while (!g_signal_received &&
               std::isfinite(mj_data_ptr->time) &&
               mj_data_ptr->time - simstart < 1.0 / framerate) {

            logger_.log("gt_left_wrist_force",  active_force_on("left_wrist_yaw_link",  mj_data_ptr->time, force_schedule));
            logger_.log("gt_right_wrist_force", active_force_on("right_wrist_yaw_link", mj_data_ptr->time, force_schedule));

            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                measured_joint_velocity[i] = robot_state.joint_state.at(
                    mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)).vel;
            }

            labrob::JointCommand joint_command;
            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)] = 0.0;
            }

            walking_manager.update(robot_state, joint_command);

            mj_step1(mj_model_ptr, mj_data_ptr);
            apply_force_schedule(mj_model_ptr, mj_data_ptr, force_schedule);

            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                mj_data_ptr->ctrl[i] = joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)];
            }
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
            std::cout << "Logs saved." << std::endl;
        }
    }
    return 0;
}