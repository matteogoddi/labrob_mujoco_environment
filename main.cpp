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

// Unitree SDK (channel types used in main for subscriber/publisher setup)
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// Labrob
#include <JointCommand.hpp>
#include <Logger.hpp>
#include <RobotState.hpp>
#include <StateEstimator.hpp>
#include <WalkingManager.hpp>
#include <HandAdmittanceController.hpp>

#include <globals.h>
#include "MujocoUI.hpp"
#include <RobotConfig.hpp>
#include <RobotInterface.hpp>

// ── Control-loop flags (main-thread only) ────────────────────────────────────
bool running          = true;
bool isWBCLoopClosed  = false;
bool isMPCLoopClosed  = false;
bool isObserverActive = false;
bool isEKFactive      = false;
bool useSim           = false;
bool useRobot         = false;
bool useViz           = true;
bool switchWalkingState = false;
bool reactiveStanding = true;
bool verboseCoop = false;
bool forward = false;
bool lateral = false;
bool curve = false;

Eigen::VectorXd measured_joint_velocity = Eigen::VectorXd::Zero(29);

using Clock = std::chrono::steady_clock;


enum class ExperimentMode { Regulation, WBC };
ExperimentMode experiment_mode = ExperimentMode::Regulation;

alignas(EIGEN_MAX_ALIGN_BYTES) labrob::WalkingManager walking_manager;
labrob::Logger sensor_logger;

// ── Signal handler ────────────────────────────────────────────────────────────
void signalHandler(int signum) {
    std::cerr << "Received signal " << signum << ", exiting..." << std::endl;
    std::cout << "Do you want to save logs? [y/n]" << std::endl;
    std::string user_input;
    std::getline(std::cin, user_input);

    running = false;
    if (user_input == "y" || user_input == "Y" || user_input == "yes" ||
        user_input == "Yes" || user_input == "YES") {
        std::cout << "Saving logs..." << std::endl;
        walking_manager.saveLogs();
        sensor_logger.save("/tmp/robot_logs");
        std::cout << "Logs saved." << std::endl;

        if (useRobot) {
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
            for (const char* src : {"/tmp/robot_logs", "/tmp/mpc_data", "/tmp/ofp_data"}) {
                const char* dst_name = (std::string(src) == "/tmp/robot_logs") ? "robot_logs"
                    : (std::string(src) == "/tmp/mpc_data") ? "mpc_logs" : "ofp_logs";
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
                std::getline(std::cin, user_input);
                if (user_input == "delete" || user_input == "remove" ||
                    user_input == "erase"  || user_input == "trash") {
                    std::filesystem::remove_all(experiment_folder);
                } else {
                    readme << "Experiment description: " << user_input << "\n";
                }
                readme.close();
            }
            std::cout << "Experiment saved in folder experiment_" << n << std::endl; 
        }
    } else {
        std::cout << "Logs not saved." << std::endl;
    }
    exit(signum);
}

// ── MuJoCo → RobotState ───────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────

static void handle_gamepad(
    labrob::WalkingManager&             wm,
    labrob::StateEstimator&             se,
    const labrob::RobotState&           robot_state,
    std::map<std::string, double>&        armatures,
    const Eigen::VectorXd&               measured_joint_pos,
    double                               current_sim_ms)
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
            wm.init(robot_state, armatures);
            isWBCLoopClosed = true;
            isMPCLoopClosed = true;
            std::cout << "[GAMEPAD] X -> Switching to WBC mode." << std::endl;
        }
    } else {
        xPressed = false;
    }
    static bool bPressed = false;
    if (gamepad_.B.pressed) {
        if (!bPressed) {
            bPressed = true;
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

static void send_dds_command(
    ChannelPublisherPtr<LowCmd_>&   publisher,
    mjModel*                        m,
    labrob::RobotState&             robot_state,
    labrob::JointCommand&           joint_command,
    Clock::duration                 elapsed,
    const Eigen::VectorXd&          q_ref,
    const Eigen::VectorXd&          dq_ref)
{
    MotorCommand motor_command;
    motor_command.tau_ff.fill(0.0f);
    motor_command.q_target.fill(0.0f);
    motor_command.dq_target.fill(0.0f);

    const bool  wbc_active = (experiment_mode == ExperimentMode::WBC);
    const float t_s        = std::chrono::duration<float>(elapsed).count();
    const bool  in_ramp    = elapsed < std::chrono::seconds(5);

    if (in_ramp) {
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
            motor_command.kp[i] = Kp_reg[i] * (t_s / 5.0f);
            motor_command.kd[i] = Kd_reg[i];
        }
    } else if (wbc_active) {
        motor_command.kp = Kp_cl;
        motor_command.kd = Kd_cl;
    } else {
        motor_command.kp = Kp_reg;
        motor_command.kd = Kd_reg;
    }

    for (int i = 0; i < m->nu; ++i) {
        int jid = m->actuator_trnid[i * 2];
        std::string jname = mj_id2name(m, mjOBJ_JOINT, jid);
        if (wbc_active) {
            if (std::abs(robot_state.joint_state[jname].pos) > 1.5 ||
                std::abs(robot_state.joint_state[jname].vel) > 4   ||
                std::abs(joint_command[jname]) > 60.0) {
                std::cout << "Safety limit exceeded on " << jname << ": "
                          << "q="   << robot_state.joint_state[jname].pos
                          << " dq=" << robot_state.joint_state[jname].vel
                          << " tau=" << joint_command[jname] << std::endl;
                signalHandler(SIGINT);
            }
            motor_command.q_target[i]  = static_cast<float>(q_ref[i]);
            motor_command.dq_target[i] = static_cast<float>(dq_ref[i]);
            motor_command.tau_ff[i]    = joint_command[jname];
        } else {
            motor_command.q_target[i]  = static_cast<float>(joint_initial_positions.at(jname));
            motor_command.dq_target[i] = 0.0f;
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


int main(const int argc, const char* argv[]) {

    bool needInterface = false;

    std::string netInterface;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sim") {
            useSim = true;
        } else if (a == "--robot") {
            useRobot = true;
            useSim   = true;
            needInterface = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                netInterface = argv[++i];
        } else if (a == "--no-viz") {
            useViz = false;
        } else if (a == "--walk") {
            reactiveStanding = false;
        } else if (a == "--verbose") {
            verboseCoop = true;
        } else if (useSim == true && a == "--forward") {
            forward = true;
        } else if (useSim == true && a == "--lateral") {
            lateral = true;
        } else if (useSim == true && a == "--curve") {
            curve = true;
        } else if (needInterface && netInterface.empty() && a[0] != '-') {
            netInterface = a;
        }
    }
    
    if (!useRobot && !useSim) {
        std::cerr << "Please specify either --sim or --robot <network_interface>" << std::endl;
        return -1;
    }

    signal(SIGINT, signalHandler);

    // Load MJCF:
    mj_loadAllPluginLibraries("/usr/local/lib/mujoco", nullptr);
    const int kErrorLength = 1024;
    char loadError[kErrorLength] = "";
    mjModel* mj_model_ptr = mj_loadXML(robotScenePath.data(), nullptr, loadError, kErrorLength);
    if (!mj_model_ptr) {
        std::cerr << "mj_loadXML failed: " << loadError << std::endl;
        return -1;
    }
    mjData*  mj_data_ptr  = mj_makeData(mj_model_ptr);

    if (useRobot) {
        std::cout << "Gamepad controls:\n";
        std::cout << "  A -> start EKF (optional, can be done at any time)\n";
        std::cout << "  X -> switch Regulation -> WBC\n";
        std::cout << "  B -> activate RW-BO\n";
        std::cout << "  Y -> exit\n";
        std::cout << "Starting in Regulation mode. Press Enter to start..." << std::endl;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    // isWBCLoopClosed stays false for sim (set on first tick) and robot (set on X press)

    // SDK setup (robot mode only):
    ChannelPublisherPtr<LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_hg::msg::dds_::IMUState_> imutorso_subscriber;
    ChannelSubscriberPtr<SportModeState_> sportmodestate_subscriber;
    std::shared_ptr<MotionSwitcherClient> msc;

    if (useRobot) {
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
    }

    // MuJoCo initial state:
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

    labrob::RobotState init_robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
    labrob::RobotState robot_state      = init_robot_state;
    walking_manager.setReactiveStanding(reactiveStanding);
    walking_manager.setVerboseCoop(verboseCoop);
    walking_manager.init(robot_state, armatures);

    labrob::StateEstimator state_estimator(
        walking_manager.get_robot_model(),
        1.0 / walking_manager.get_controller_frequency(),
        labrob::StateEstimator::Filter::RightInvariantEKF // SimpleEKF
    );

    labrob::MujocoUI* mujoco_ui_ptr = useViz
        ? labrob::MujocoUI::getInstance(mj_model_ptr, mj_data_ptr)
        : nullptr;
    static constexpr int framerate = 60;
    const Clock::time_point t_start = Clock::now();

    // EMA on raw motor dq — smoothing factor alpha: weight on the new measurement.
    // alpha = 0.15 → ~12 Hz cutoff at 500 Hz. Increase for less lag, decrease for more smoothing.
    Eigen::VectorXd ema_joint_vel = Eigen::VectorXd::Zero(29);
    constexpr double ema_alpha = 0.15;

    // EMA on pelvis IMU gyroscope.
    Eigen::Vector3d ema_imu_gyro = Eigen::Vector3d::Zero();
    constexpr double ema_alpha_gyro = 0.15;

    // Joint position/velocity references sent as q_target / dq_target to the motor PD.
    // Propagated each tick via Euler integration of the WBC desired joint accelerations.
    Eigen::VectorXd q_ref_joints  = Eigen::VectorXd::Zero(29);
    Eigen::VectorXd dq_ref_joints = Eigen::VectorXd::Zero(29);

    // Hand forces
    Eigen::Vector3d f_l_test = Eigen::Vector3d::Zero();
    Eigen::Vector3d f_r_test = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_lhand  = Eigen::Vector3d::Zero();
    Eigen::Vector3d p_rhand  = Eigen::Vector3d::Zero();
    int lhand_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "left_wrist_yaw_link");
    int rhand_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "right_wrist_yaw_link");

    // ── Main loop ─────────────────────────────────────────────────────────────
    while (running) {
        if (useViz && mujoco_ui_ptr->windowShouldClose()) break;

        mjtNum simstart = mj_data_ptr->time;
        while (mj_data_ptr->time - simstart < 1.0 / framerate) {

            const auto tick_start = Clock::now();

            // Per-tick sensor snapshot (populated below, robot or sim):
            Eigen::VectorXd measured_joint_pos = Eigen::VectorXd::Zero(29);
            Eigen::VectorXd measured_joint_vel = Eigen::VectorXd::Zero(29);
            Eigen::Vector3d imu_acc            = Eigen::Vector3d::Zero();
            Eigen::Vector3d imu_gyro           = Eigen::Vector3d::Zero();

            if (useRobot) {

                // ── Read sensors from SDK callbacks ───────────────────────────
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

                    // Fill robot_state for EKF:
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

                    

                    // ── Log sensor feedback ───────────────────────────────────
                    sensor_logger.log("pelvis_acc",  imu_pelvis_data.accelerometer);
                    sensor_logger.log("pelvis_gyro", imu_pelvis_data.omega);
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
            }

            // ── State estimator (robot mode only) ────────────────────────────
            if (useRobot && isEKFactive) {
                if (state_estimator.is_active()) {
                    state_estimator.update(
                        robot_state, imu_gyro, imu_acc,
                        walking_manager.get_contact(),
                        walking_manager.get_wbc_q_ddot()
                    );
                }
            }

            // Log robot_state base quantities (odometry before EKF activation,
            // filtered estimates after pressing A).
            if (useRobot) {
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
            }

            // ── Walking Manager ───────────────────────────────────────────────
            labrob::JointCommand joint_command;
            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)] = 0.0;
            }

            if (!useRobot) {
                // ── Simulation ────────────────────────────────────────────────
                // First tick: init walking manager with ground-truth MuJoCo state
                if (!isWBCLoopClosed) {
                    robot_state     = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
                    isWBCLoopClosed = true;
                    isMPCLoopClosed = true;
                    isObserverActive = true;
                }
                f_l_test = Eigen::Vector3d::Zero();
                f_r_test = Eigen::Vector3d::Zero();

                
                // Forward walk
                if (forward == true) {
                    if (mj_data_ptr->time >= 8.0 && mj_data_ptr->time < 30.0) {

                        f_l_test = Eigen::Vector3d(3.0, 0.0, 0.0);
                        f_r_test = Eigen::Vector3d(3.0, 0.0, 0.0);

                    }
                }
                

                // Lateral walk
                if (lateral == true) {
                    if (mj_data_ptr->time >= 8.0 && mj_data_ptr->time < 30.0) {

                        f_l_test = Eigen::Vector3d(0.0, 3.0, 0.0);
                        f_r_test = Eigen::Vector3d(0.0, 0.0, 0.0);

                    }
                }


                // Curve walk
                if (curve == true) {               
                    if (mj_data_ptr->time >= 10.0 && mj_data_ptr->time < 33.0) {

                        f_l_test = walking_manager.get_R_F_hac() * Eigen::Vector3d(4.0, 0.0, -4.0);
                        f_r_test = walking_manager.get_R_F_hac() * Eigen::Vector3d(2.0, 0.0, -4.0);

                    } else if (mj_data_ptr->time >= 33.0 && mj_data_ptr->time < 35.0) {
                        
                        f_l_test = Eigen::Vector3d(0.0, 0.0, -4.0);
                        f_r_test = Eigen::Vector3d(0.0, 0.0, -4.0);

                    } else if (mj_data_ptr->time >= 35.0 && mj_data_ptr->time < 55.0) {

                        f_l_test = walking_manager.get_R_F_hac() * Eigen::Vector3d(4.0, 0.0, -4.0);
                        f_r_test = walking_manager.get_R_F_hac() * Eigen::Vector3d(4.0, 0.0, -4.0);
                    }
                }
                

                // Apply forces physically in MuJoCo
                int l_wrist_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "left_wrist_yaw_link");
                int r_wrist_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "right_wrist_yaw_link");

                if (l_wrist_id >= 0) {
                    mj_data_ptr->xfrc_applied[l_wrist_id * 6 + 0] = f_l_test.x();
                    mj_data_ptr->xfrc_applied[l_wrist_id * 6 + 1] = f_l_test.y();
                    mj_data_ptr->xfrc_applied[l_wrist_id * 6 + 2] = f_l_test.z();
                }
                if (r_wrist_id >= 0) {
                    mj_data_ptr->xfrc_applied[r_wrist_id * 6 + 0] = f_r_test.x();
                    mj_data_ptr->xfrc_applied[r_wrist_id * 6 + 1] = f_r_test.y();
                    mj_data_ptr->xfrc_applied[r_wrist_id * 6 + 2] = f_r_test.z();
                }


                // Ground forces right and left wrists
                static std::ofstream gt_right("/tmp/gt_right_wrist.txt");   // truncate at startup
                static std::ofstream gt_left ("/tmp/gt_left_wrist.txt");
                gt_right << f_r_test.transpose() << "\n";
                gt_left  << f_l_test.transpose() << "\n";

                for (int i = 0; i < mj_model_ptr->nu; ++i) {
                    int jid = mj_model_ptr->actuator_trnid[i * 2];
                    std::string jname = mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid);
                    measured_joint_velocity[i] = robot_state.joint_state.at(jname).vel;
                }


                walking_manager.update(robot_state, joint_command);

                auto t0 = std::chrono::steady_clock::now();
                mj_step1(mj_model_ptr, mj_data_ptr);

                if (mj_data_ptr->time > 5.2)
                    mju_zero(mj_data_ptr->qfrc_applied, mj_model_ptr->nv);

                for (int i = 0; i < mj_model_ptr->nu; ++i) {
                    int jid = mj_model_ptr->actuator_trnid[i * 2];
                    mj_data_ptr->ctrl[i] = joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)];
                }
                mj_step2(mj_model_ptr, mj_data_ptr);

                auto dt = std::chrono::steady_clock::now() - t0;
                if (dt > std::chrono::milliseconds(1))
                    std::cout << "Warning: mujoco step took "
                              << std::chrono::duration_cast<std::chrono::microseconds>(dt).count()
                              << " us" << std::endl;

                robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);

            } else {
                // ── Experiment ────────────────────────────────────────────────
                switch (experiment_mode) {
                    case ExperimentMode::Regulation:
                        // joint_command stays zero; PD holds measured position
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

                // ── Push robot state to MuJoCo for visualization (optional) ──
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

                // ── Timing: pad tick to 2 ms before sending ──────────────────
                const auto remaining = std::chrono::milliseconds(2) - (Clock::now() - tick_start);
                if (remaining > Clock::duration::zero())
                    std::this_thread::sleep_for(remaining);

                // ── Send motor commands via DDS ───────────────────────────────
                send_dds_command(lowcmd_publisher, mj_model_ptr, robot_state,
                                 joint_command, Clock::now() - t_start,
                                 q_ref_joints, dq_ref_joints);
            }

        }

        if (useViz) {
            auto t0 = std::chrono::steady_clock::now();
            // ####################### //
            // Rendering with hand forces
            int lhand_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "left_wrist_yaw_link");
            int rhand_id = mj_name2id(mj_model_ptr, mjOBJ_BODY, "right_wrist_yaw_link");

            Eigen::Vector3d p_lhand(
                mj_data_ptr->xpos[3 * lhand_id + 0],
                mj_data_ptr->xpos[3 * lhand_id + 1],
                mj_data_ptr->xpos[3 * lhand_id + 2]
            );
            Eigen::Vector3d p_rhand(
                mj_data_ptr->xpos[3 * rhand_id + 0],
                mj_data_ptr->xpos[3 * rhand_id + 1],
                mj_data_ptr->xpos[3 * rhand_id + 2]
            );

            mujoco_ui_ptr->renderWithHandForces(p_lhand, f_l_test, p_rhand, f_r_test);

            
            auto render_dt = std::chrono::steady_clock::now() - t0;
            if (render_dt > std::chrono::milliseconds(20))
                std::cout << "Warning: render took "
                          << std::chrono::duration_cast<std::chrono::microseconds>(render_dt).count()
                          << " us" << std::endl;
            
        }
    }

    mj_deleteData(mj_data_ptr);
    mj_deleteModel(mj_model_ptr);
    return 0;
}