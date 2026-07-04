// std
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <csignal>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Pinocchio (must come before RobotState.hpp)
#include <pinocchio/multibody/model.hpp>

// Unitree SDK
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// Labrob
#include <JointCommand.hpp>
#include <Logger.hpp>
#include <RobotState.hpp>
#include <StateEstimator.hpp>
#include <WalkingManager.hpp>

#include <globals.h>
#include "MujocoUI.hpp"
#include <RobotConfig.hpp>
#include <RobotInterface.hpp>

// ── Globals required by WalkingManager (via globals.h) ───────────────────────
bool isWBCLoopClosed  = false;
bool isMPCLoopClosed  = false;
bool isObserverActive = false;
bool isEKFactive      = false;
bool switchWalkingState = false;
bool switchCoopState  = false;

bool pendingWBCInit   = false;   // set on X press, consumed after EKF update
bool useViz           = true;
bool reactiveStanding = false;
bool verboseCoop      = false;

bool useRobot = true;

Eigen::VectorXd measured_joint_velocity = Eigen::VectorXd::Zero(29);

using Clock = std::chrono::steady_clock;

enum class ExperimentMode { Listening, Regulation, WBC };
ExperimentMode experiment_mode = ExperimentMode::Regulation;

alignas(EIGEN_MAX_ALIGN_BYTES) labrob::WalkingManager walking_manager;
labrob::Logger sensor_logger;

static volatile sig_atomic_t g_signal_received = 0;

void signalHandler(int signum) {
    g_signal_received = signum;
}

static void save_experiment_logs() {
    walking_manager.saveLogs();
    sensor_logger.save("/tmp/robot_logs");
    std::cout << "Logs saved." << std::endl;

    std::string experiment_folder;
    int n = 1;
    while (true) {
        if (!std::filesystem::exists("../experiments"))
            std::filesystem::create_directory("../experiments");
        experiment_folder = "../experiments/experiment_" + std::to_string(n);
        if (!std::filesystem::exists(experiment_folder)) {
            std::filesystem::create_directory(experiment_folder);
            break;
        }
        ++n;
    }
    for (const char* src : {"/tmp/robot_logs", "/tmp/mpc_data"}) {
        const char* dst_name = (std::string(src) == "/tmp/robot_logs") ? "robot_logs" : "mpc_logs";
        if (std::filesystem::exists(src))
            std::filesystem::copy(src,
                std::filesystem::path(experiment_folder) / dst_name,
                std::filesystem::copy_options::recursive |
                std::filesystem::copy_options::overwrite_existing);
    }
    std::ofstream readme(experiment_folder + "/README.txt");
    if (readme.is_open()) {
        readme << "Gains Kp_cl: ";
        for (auto v : Kp_cl) readme << v << " ";
        readme << "\nGains Kd_cl: ";
        for (auto v : Kd_cl) readme << v << " ";
        readme << "\n\n";
        std::cout << "Please enter a description of the experiment: ";
        std::string desc;
        std::getline(std::cin, desc);
        if (desc == "delete" || desc == "remove" || desc == "erase" || desc == "trash")
            std::filesystem::remove_all(experiment_folder);
        else
            readme << "Experiment description: " << desc << "\n";
        readme.close();
    }
    std::cout << "Experiment saved in folder experiment_" << n << std::endl;
}

// ── Gamepad handler ───────────────────────────────────────────────────────────
static void handle_gamepad(
    labrob::WalkingManager&      wm,
    labrob::StateEstimator&      se,
    const labrob::RobotState&    robot_state,
    std::map<std::string, double>& armatures,
    const Eigen::VectorXd&       measured_joint_pos,
    double                       current_sim_ms)
{
    if (gamepad_.Y.pressed) {
        std::cout << "[GAMEPAD] Y -> Deactivating motors..." << std::endl;
        signalHandler(SIGINT);
    }
    static bool xPressed = false;
    if (gamepad_.X.pressed) {
        if (!xPressed && experiment_mode == ExperimentMode::Regulation) {
            xPressed        = true;
            experiment_mode = ExperimentMode::WBC;
            pendingWBCInit  = true;
            std::cout << "[GAMEPAD] X -> Switching to WBC mode." << std::endl;
        }
    } else {
        xPressed = false;
    }
    static bool bPressed = false;
    if (gamepad_.B.pressed) {
        if (!bPressed) {
            bPressed = true;
            /*
            switchWalkingState = true;
            std::cout << "[GAMEPAD] B -> Walking state switched." << std::endl;
            */
            isObserverActive = true;
            std::cout << "[GAMEPAD] B pressed -> Wrench observer activated." << std::endl;
        }
    } else {
        bPressed = false;
    }
    static bool aPressed = false;
    if (gamepad_.A.pressed) {
        if (!aPressed) {
            aPressed = true;
            if (!isEKFactive) {
                std::cout << "[GAMEPAD] A -> EKF started." << std::endl;
                isEKFactive = true;
                se.activate(robot_state, measured_joint_pos);
            }
        }
    } else {
        aPressed = false;
    }
}

// ── DDS command sender ────────────────────────────────────────────────────────
static void send_dds_command(
    ChannelPublisherPtr<LowCmd_>&   publisher,
    mjModel*                        m,
    const labrob::RobotState&       robot_state,
    const labrob::JointCommand&     joint_command,
    Clock::duration                 elapsed,
    const Eigen::VectorXd&          q_ref,
    const Eigen::VectorXd&          dq_ref)
{
    MotorCommand motor_command;
    motor_command.tau_ff.fill(0.0f);
    motor_command.q_target.fill(0.0f);
    motor_command.dq_target.fill(0.0f);

    const float t_s        = std::chrono::duration<float>(elapsed).count();
    const bool  in_ramp    = elapsed < std::chrono::seconds(5);

    if (in_ramp) {
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
            motor_command.kp[i] = Kp_reg[i] * (t_s / 5.0f);
            motor_command.kd[i] = Kd_reg[i];
        }
    } else if (experiment_mode == ExperimentMode::WBC) {
        motor_command.kp = Kp_cl;
        motor_command.kd = Kd_cl;
    } else if (experiment_mode == ExperimentMode::Regulation) {
        motor_command.kp = Kp_reg;
        motor_command.kd = Kd_reg;
    }

    for (int i = 0; i < m->nu; ++i) {
        int jid = m->actuator_trnid[i * 2];
        std::string jname = mj_id2name(m, mjOBJ_JOINT, jid);
        if (experiment_mode == ExperimentMode::WBC) {
            if (std::abs(robot_state.joint_state.at(jname).pos) > 3.14 ||
                std::abs(robot_state.joint_state.at(jname).vel) > 3   ||
                std::abs(joint_command[jname]) > 100.0) {
                std::cout << "Safety limit exceeded on " << jname << ": "
                          << "q="   << robot_state.joint_state.at(jname).pos
                          << " dq=" << robot_state.joint_state.at(jname).vel
                          << " tau=" << joint_command[jname] << std::endl;
                signalHandler(SIGINT);
            }
            motor_command.q_target[i]  = static_cast<float>(q_ref[i]);
            motor_command.dq_target[i] = static_cast<float>(dq_ref[i]);
            motor_command.tau_ff[i]    = joint_command[jname];
        } else if (experiment_mode == ExperimentMode::Regulation){
            motor_command.q_target[i]  = static_cast<float>(joint_initial_positions.at(jname));
        }
    }

    LowCmd_ dds_cmd;
    dds_cmd.mode_pr()      = static_cast<uint8_t>(Mode::PR);
    dds_cmd.mode_machine() = mode_machine_;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        int jid   = m->actuator_trnid[i * 2];
        std::string jname = mj_id2name(m, mjOBJ_JOINT, jid);
        int ridx  = joint_name_to_index.at(jname);
        auto& cmd = dds_cmd.motor_cmd().at(ridx);
        cmd.mode() = 1;
        cmd.q()    = motor_command.q_target[i];
        cmd.dq()   = motor_command.dq_target[i];
        cmd.tau()  = motor_command.tau_ff[i];
        cmd.kp()   = motor_command.kp[i];
        cmd.kd()   = motor_command.kd[i];
    }
    dds_cmd.crc() = Crc32Core((uint32_t*)&dds_cmd, (sizeof(dds_cmd) >> 2) - 1);
    publisher->Write(dds_cmd);
}

// ─────────────────────────────────────────────────────────────────────────────

int main(const int argc, const char* argv[]) {

    std::string netInterface;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--no-viz")  useViz = false;
        else if (a == "--walk")    reactiveStanding = false;
        else if (a == "--verbose") verboseCoop = true;
        else if (a == "--listen")  experiment_mode = ExperimentMode::Listening;
        else if (netInterface.empty()) netInterface = a;
    }

    std::cout << netInterface << std::endl;

    if (netInterface.empty()) {
        std::cerr << "Usage: g1 <network_interface> [--no-viz] [--walk] [--verbose]" << std::endl;
        return -1;
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

    std::cout << "Gamepad controls:\n";
    std::cout << "  A -> start EKF (optional, can be done at any time)\n";
    std::cout << "  X -> switch Regulation -> WBC\n";
    // std::cout << "  B -> switch walking state\n";
    std::cout << "  B -> enable RW-BO\n";
    std::cout << "  Y -> exit\n";
    std::cout << "Starting in Regulation mode. Press Enter to start..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    // SDK setup
    ChannelPublisherPtr<LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_> imutorso_subscriber;
    ChannelSubscriberPtr<SportModeState_> sportmodestate_subscriber;
    std::shared_ptr<MotionSwitcherClient> msc;

    std::cout << "Using robot with network interface: " << netInterface << std::endl;
    ChannelFactory::Instance()->Init(0, netInterface);

    msc.reset(new MotionSwitcherClient());
    msc->SetTimeout(5.0f);
    msc->Init();

    while (queryMotionStatus(msc)) {
        std::cout << "Trying to deactivate motion control service..." << std::endl;
        if (msc->ReleaseMode() == 0)
            std::cout << "Motion control service deactivated." << std::endl;
        else {
            std::cerr << "Failed to deactivate, retrying..." << std::endl;
            sleep(5);
        }
    }

    lowcmd_publisher.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher->InitChannel();
    lowstate_subscriber.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber->InitChannel(std::bind(&LowStateHandler, std::placeholders::_1), 1);
    imutorso_subscriber.reset(new ChannelSubscriber<unitree_hg::msg::dds_::IMUState_>(HG_IMU_TORSO));
    imutorso_subscriber->InitChannel(std::bind(&imuTorsoHandler, std::placeholders::_1), 1);
    sportmodestate_subscriber.reset(new ChannelSubscriber<SportModeState_>(GO_STATE_TOPIC));
    sportmodestate_subscriber->InitChannel(std::bind(&SportModeStateHandler, std::placeholders::_1), 1);

    // Initial MuJoCo state (used for joint name mapping only in robot mode)
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

    labrob::RobotState robot_state;
    for (int i = 0; i < mj_model_ptr->nu; ++i) {
        int jid = mj_model_ptr->actuator_trnid[i * 2];
        std::string jname = mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid);
        robot_state.joint_state[jname].pos = 0.0;
        robot_state.joint_state[jname].vel = 0.0;
    }

    walking_manager.setReactiveStanding(reactiveStanding);
    walking_manager.setVerboseCoop(verboseCoop);

    labrob::NoiseParams ri_noise;
    ri_noise.gyro_noise    = 0.01;
    ri_noise.accel_noise   = 0.1;
    ri_noise.contact_noise = 0.01;
    ri_noise.gyro_bias_rw  = 0.0001;
    ri_noise.accel_bias_rw = 0.001;
    ri_noise.encoder_noise = 0.01;

    labrob::StateEstimator state_estimator(
        "../robot/g1/g1_description/g1_29dof_with_hand_rev_1_0.urdf",
        G1_CONTROLLER_DT,
        labrob::StateEstimator::Filter::RightInvariantEKF,
        ri_noise
    );

    labrob::MujocoUI* mujoco_ui_ptr = useViz
        ? labrob::MujocoUI::getInstance(mj_model_ptr, mj_data_ptr)
        : nullptr;
    static constexpr int framerate = 60;
    const Clock::time_point t_start = Clock::now();

    Eigen::VectorXd ema_joint_vel = Eigen::VectorXd::Zero(29);
    constexpr double ema_alpha = 0.1;
    Eigen::Vector3d ema_imu_gyro = Eigen::Vector3d::Zero();
    constexpr double ema_alpha_gyro = 0.15;

    Eigen::VectorXd q_ref_joints  = Eigen::VectorXd::Zero(29);
    Eigen::VectorXd dq_ref_joints = Eigen::VectorXd::Zero(29);

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (!g_signal_received) {
        if (useViz && mujoco_ui_ptr->windowShouldClose()) break;

        mjtNum simstart = mj_data_ptr->time;

        while (!g_signal_received &&
               std::isfinite(mj_data_ptr->time) &&
               mj_data_ptr->time - simstart < 1.0 / framerate) {

            const auto tick_start = Clock::now();

            Eigen::VectorXd measured_joint_pos = Eigen::VectorXd::Zero(29);
            Eigen::VectorXd measured_joint_vel = Eigen::VectorXd::Zero(29);
            Eigen::Vector3d imu_acc            = Eigen::Vector3d::Zero();
            Eigen::Vector3d imu_gyro           = Eigen::Vector3d::Zero();

            // ── Read sensors from SDK callbacks ───────────────────────────────
            {
                std::lock_guard<std::mutex> lock(stateMutex);
                imu_acc  = imu_pelvis_data.accelerometer;
                imu_gyro = imu_pelvis_data.omega;
                ema_imu_gyro = ema_alpha_gyro * imu_gyro
                             + (1.0 - ema_alpha_gyro) * ema_imu_gyro;
                for (int i = 0; i < mj_model_ptr->nu; ++i) {
                    measured_joint_pos[i] = motor_state_data.q[i];
                    measured_joint_vel[i] = motor_state_data.dq[i];
                    ema_joint_vel[i] = ema_alpha * measured_joint_vel[i]
                                     + (1.0 - ema_alpha) * ema_joint_vel[i];
                }
                robot_state.angular_velocity = imu_gyro;

                robot_state.position = Eigen::Vector3d(
                    odometry_data.position[0],
                    odometry_data.position[1],
                    odometry_data.position[2]
                );
                robot_state.linear_velocity = robot_state.orientation.toRotationMatrix().transpose() *
                    Eigen::Vector3d(
                        odometry_data.velocity[0],
                        odometry_data.velocity[1],
                        odometry_data.velocity[2]
                    );
                robot_state.orientation = Eigen::Quaterniond(
                    odometry_data.quaternion[0],
                    odometry_data.quaternion[1],
                    odometry_data.quaternion[2],
                    odometry_data.quaternion[3]
                );
                for (int i = 0; i < mj_model_ptr->nu; ++i) {
                    int jid = mj_model_ptr->actuator_trnid[i * 2];
                    std::string jname = mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid);
                    robot_state.joint_state[jname].pos = measured_joint_pos[i];
                    robot_state.joint_state[jname].vel = ema_joint_vel[i];
                    measured_joint_velocity[i] = ema_joint_vel[i];
                }

                sensor_logger.log("pelvis_acc",  imu_pelvis_data.accelerometer);
                sensor_logger.log("pelvis_gyro", imu_pelvis_data.omega);
                sensor_logger.log("pelvis_quat", imu_pelvis_data.quaternion);
                sensor_logger.log("pelvis_rpy",  imu_pelvis_data.rpy);
                sensor_logger.log("torso_quat",  imu_torso_data.quaternion);
                sensor_logger.log("torso_rpy",   imu_torso_data.rpy);
                sensor_logger.log("torso_acc",   imu_torso_data.accelerometer);
                sensor_logger.log("torso_gyro",  imu_torso_data.omega);
                sensor_logger.log("odom_pos",    odometry_data.position);
                sensor_logger.log("odom_vel",    odometry_data.velocity);
                sensor_logger.log("odom_quat",   odometry_data.quaternion);
                sensor_logger.log("odom_rpy",    odometry_data.rpy);
                sensor_logger.log("joint_pos",   measured_joint_pos);
                sensor_logger.log("joint_vel",   measured_joint_vel);
            }

            handle_gamepad(walking_manager, state_estimator, robot_state,
                           armatures, measured_joint_pos,
                           1000.0 * mj_data_ptr->time);

            // ── State estimator ───────────────────────────────────────────────
            if (isEKFactive && state_estimator.is_active()) {
                state_estimator.update(
                    robot_state, imu_gyro, imu_acc,
                    walking_manager.get_contact()
                );
            }

            if (pendingWBCInit) {
                pendingWBCInit  = false;
                walking_manager.init(robot_state, armatures);
                isWBCLoopClosed = true;
                isMPCLoopClosed = true;
            }

            sensor_logger.log("filtered_base_position", robot_state.position);
            sensor_logger.log("filtered_base_velocity", robot_state.linear_velocity);
            sensor_logger.log("filtered_base_quat",
                Eigen::Vector4d(
                    robot_state.orientation.w(), robot_state.orientation.x(),
                    robot_state.orientation.y(), robot_state.orientation.z()));
            sensor_logger.log("filtered_base_rpy",     labrob::rpyFromQuaternion(robot_state.orientation));
            sensor_logger.log("filtered_base_ang_vel", robot_state.angular_velocity);
            {
                Eigen::VectorXd filt_jpos(mj_model_ptr->nu);
                Eigen::VectorXd filt_jvel(mj_model_ptr->nu);
                for (int i = 0; i < mj_model_ptr->nu; ++i) {
                    int jid = mj_model_ptr->actuator_trnid[i * 2];
                    std::string jname = mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid);
                    filt_jpos[i] = robot_state.joint_state.at(jname).pos;
                    filt_jvel[i] = robot_state.joint_state.at(jname).vel;
                }
                sensor_logger.log("filtered_joint_position", filt_jpos);
                sensor_logger.log("filtered_joint_velocity", filt_jvel);
            }

            // ── Walking Manager ───────────────────────────────────────────────
            labrob::JointCommand joint_command;
            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)] = 0.0;
            }

            switch (experiment_mode) {
                case ExperimentMode::Regulation:
                case ExperimentMode::Listening:
                    break;

                case ExperimentMode::WBC:
                    walking_manager.update(robot_state, joint_command);
                    {
                        constexpr double cmd_dt = 0.002;
                        const Eigen::VectorXd& jddot = walking_manager.get_wbc_q_ddot();
                        const Eigen::VectorXd jddot_joints = jddot.tail(mj_model_ptr->nu);
                        for (int i = 0; i < mj_model_ptr->nu; ++i) {
                            int jid = mj_model_ptr->actuator_trnid[i * 2];
                            std::string jname = mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid);
                            q_ref_joints[i]  = robot_state.joint_state.at(jname).pos + robot_state.joint_state.at(jname).vel * cmd_dt + 0.5 * jddot_joints[i] * cmd_dt * cmd_dt;
                            dq_ref_joints[i] = robot_state.joint_state.at(jname).vel + jddot_joints[i] * cmd_dt;
                        }
                    }
                    break;
            }

            // ── Visualization: push robot state to MuJoCo ────────────────────
            if (useViz) {
                mj_data_ptr->qpos[0] = robot_state.position.x();
                mj_data_ptr->qpos[1] = robot_state.position.y();
                mj_data_ptr->qpos[2] = robot_state.position.z();
                mj_data_ptr->qpos[3] = robot_state.orientation.w();
                mj_data_ptr->qpos[4] = robot_state.orientation.x();
                mj_data_ptr->qpos[5] = robot_state.orientation.y();
                mj_data_ptr->qpos[6] = robot_state.orientation.z();
                Eigen::Vector3d lv = robot_state.orientation.toRotationMatrix() * robot_state.linear_velocity;
                mj_data_ptr->qvel[0] = lv.x();
                mj_data_ptr->qvel[1] = lv.y();
                mj_data_ptr->qvel[2] = lv.z();
                mj_data_ptr->qvel[3] = robot_state.angular_velocity.x();
                mj_data_ptr->qvel[4] = robot_state.angular_velocity.y();
                mj_data_ptr->qvel[5] = robot_state.angular_velocity.z();
                for (int i = 0; i < mj_model_ptr->nu; ++i) {
                    int jid = mj_model_ptr->actuator_trnid[i * 2];
                    std::string jname = mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid);
                    mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[jid]] = robot_state.joint_state[jname].pos;
                    mj_data_ptr->qvel[mj_model_ptr->jnt_dofadr[jid]]  = robot_state.joint_state[jname].vel;
                }
                mj_forward(mj_model_ptr, mj_data_ptr);
                mju_zero(mj_data_ptr->ctrl,         mj_model_ptr->nu);
                mju_zero(mj_data_ptr->qfrc_applied, mj_model_ptr->nv);
                mju_zero(mj_data_ptr->qacc,         mj_model_ptr->nv);
                mju_zero(mj_data_ptr->act,          mj_model_ptr->nu);
            }
            mj_data_ptr->time += 0.002;

            // ── Timing: pad tick to 2 ms ──────────────────────────────────────
            const auto remaining = std::chrono::milliseconds(2) - (Clock::now() - tick_start);
            if (remaining > Clock::duration::zero())
                std::this_thread::sleep_for(remaining);

            // ── Send motor commands via DDS (not in Listening mode) ──────────
            if (experiment_mode != ExperimentMode::Listening)
                send_dds_command(lowcmd_publisher, mj_model_ptr, robot_state,
                                 joint_command, Clock::now() - t_start,
                                 q_ref_joints, dq_ref_joints);
        }

        if (useViz) {
            mujoco_ui_ptr->render();
        }
    }

    mj_deleteData(mj_data_ptr);
    mj_deleteModel(mj_model_ptr);

    if (g_signal_received) {
        std::cout << "Do you want to save logs? [y/n] ";
        std::string input;
        std::getline(std::cin, input);
        if (input == "y" || input == "Y" || input == "yes")
            save_experiment_logs();
        else
            std::cout << "Logs not saved." << std::endl;
    }
    return 0;
}