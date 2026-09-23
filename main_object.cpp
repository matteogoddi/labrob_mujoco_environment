// ─────────────────────────────────────────────────────────────────────────────
// Carried-object experiment (simulation only).
//
// The robot holds an object of given size and weight with both hands; a force
// is applied to the OBJECT, and the robot has to move accordingly. The control
// architecture is exactly the one of main.cpp — residual wrench observer → hand
// admittance (HAC) → cooperative footstep planner → IS-MPC → WBC — the only
// difference being where the excitation enters: instead of pushing the wrists
// directly, the push is applied to the object and reaches the hands through the
// grasp, so the object is what carries the motion command to the robot.
//
// Being simulation-only, none of the code that talks to the real robot (DDS
// publishers/subscribers, gamepad, motion-switcher) is replicated here.
//
// Usage:
//   ./main_object [--forward | --lateral | --curve] [--no-viz] [--stand] [--verbose]
// ─────────────────────────────────────────────────────────────────────────────

// std
#include <cmath>
#include <csignal>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <map>
#include <string>
#include <thread>
#include <vector>

// Pinocchio (must come before RobotState.hpp)
#include <pinocchio/multibody/model.hpp>

// Labrob
#include <JointCommand.hpp>
#include <Logger.hpp>
#include <RobotState.hpp>
#include <WalkingManager.hpp>

#include <globals.h>
#include "MujocoUI.hpp"
#include <RobotConfig.hpp>

// ── Control-loop flags ───────────────────────────────────────────────────────
// Same globals declared in globals.h and used inside the controller library:
// this executable provides its own definitions, with the values that make sense
// for a pure-simulation experiment (no robot, no EKF, no gamepad).
bool running            = true;
bool isWBCLoopClosed    = false;
bool isMPCLoopClosed    = false;
bool isObserverActive   = false;
bool isEKFactive        = false;
bool useSim             = true;
bool useRobot           = false;
bool useViz             = true;
bool switchWalkingState = false;
bool reactiveStanding   = false;   // the demo is meant to walk: reactive standing off by default
bool verboseCoop        = false;
// These three select the push tests of main.cpp inside the controller library,
// and they are deliberately left false here: `lateral` in particular makes
// WalkingManager build the desired CoM acceleration from the plain LIP model,
// dropping the disturbance term. That is fine when nothing is held, but with a
// payload in the hands the disturbance carries the weight of the object, and
// ignoring it makes the robot stand ~10 cm too tall and fall. The direction of
// the push in this experiment is chosen by the local kPushMode below instead.
bool forward            = false;
bool lateral            = false;
bool curve              = false;

Eigen::VectorXd measured_joint_velocity = Eigen::VectorXd::Zero(29);

using Clock = std::chrono::steady_clock;

// ─────────────────────────────────────────────────────────────────────────────
// Experiment parameters — this is where the object is defined
// ─────────────────────────────────────────────────────────────────────────────

// Half-extents of the box along x (depth), y (width) and z (height) [m].
// The y half-extent sets how far the side faces of the box are from the grip
// points, so it has to be consistent with the distance between the two palms in
// the carrying posture (printed at startup) and with the finger angles below:
// too narrow and the fingers close on nothing, too wide and they sink into the
// box. The startup print reports how deep each fingertip sits inside the face.
//
// Narrowing the box also lowers its inertia about the axis joining the two grip
// points, and the constraint solver scales the weld stiffness from the inertia
// at the centre of mass rather than at the anchors: too narrow a box and the
// grasp goes unstable within the first second. That is what the soft solref of
// the welds buys back (see the scene file); if this box is made much smaller,
// check that a standing run stays put before trusting it.
static const Eigen::Vector3d kObjectHalfSize(0.09, 0.12, 0.09);

// Mass of the object [kg]. It is carried by the two hands, so each hand feels
// roughly half of its weight.
static constexpr double kObjectMass = 2.0;

// Grip point inside the wrist frame [m]: the wrist_yaw_link origin is at the
// wrist joint, while the palm is about 10 cm further along the local +x axis.
// This point plays two roles: the object is centred on the midpoint of the two
// grip points, and each weld constraint is anchored there, so the object hangs
// off the palms rather than off the wrists.
//
// The second role is what keeps the object from swinging: the two anchors and
// the centre of the box lie on the same axis, so the weight of the object has
// no lever arm about it. Anchoring at the wrist origins instead puts the box
// 10 cm off that axis and its own weight rocks it around the wrists.
//
// The offset is also passed to WalkingManager::setCarriedObjectLoad(), because
// a load held 10 cm in front of the wrist frame is a moment on the wrist joints
// and the inverse dynamics has to feed it forward.
static const Eigen::Vector3d kPalmOffsetInWrist(0.10, 0.0, 0.0);

// Closing angles of the (unactuated, spring-driven) fingers, for the LEFT hand;
// the right hand is mirrored. They are written into the spring reference of the
// finger joints, so the springs actively hold the grip instead of relaxing back
// to the open hand.
static constexpr double kFingerProximal = -0.21;  // index/middle, first phalanx
static constexpr double kFingerDistal   = -0.31;  // index/middle, second phalanx
static constexpr double kThumbProximal  =  1.00;  // thumb, first phalanx
static constexpr double kThumbDistal    =  0.90;  // thumb, second phalanx

// Direction of the push applied to the object, selected on the command line.
enum class PushMode { Forward, Lateral, Curve };
static PushMode push_mode = PushMode::Forward;

// Force applied to the object [N] and time window over which it acts [s].
// 6 N is the two-hand equivalent of the 3 N per wrist used by the push tests of
// main.cpp. The window starts well after the 2 s posture-regulation phase and
// the 2 s transient of the wrench observer.
static constexpr double kPushMagnitude  = 6.0;
// Lateral pushes use a smaller force: side-stepping is the tight balance
// direction. Note that --lateral is the marginal case of this experiment: with
// a 2 kg payload the robot follows the push for a few side-steps and then loses
// balance, whatever the magnitude (3 N gets the furthest). --forward and
// --curve instead run to the end of the experiment. If a full lateral run is
// needed, the knobs are a lighter object, a smaller force, and above all the
// 4 s single support of the cooperative planner (coop_T_ss_ms_ in
// WalkingManager.hpp), which is a long time to balance sideways on one foot
// with a load in the hands.
static constexpr double kPushMagnitudeLateral = 3.0;
static constexpr double kPushStartTime  = 8.0;
// Time taken by the force to ramp up to full magnitude, and to ramp back down
// at the end of the window [s].
static constexpr double kPushRampTime   = 1.5;
static constexpr double kPushEndTime    = 40.0;

// Wall-clock length of the experiment [s of simulated time].
static constexpr double kSimDuration = 60.0;

static constexpr std::string_view kObjectScenePath =
    "../../labrob_mujoco_environment/robot/g1/g1_mj_description/scene_object.xml";

// ─────────────────────────────────────────────────────────────────────────────

// ── Experiment duration bookkeeping ──────────────────────────────────────────
Clock::time_point experiment_start;
bool   experiment_started = false;
double last_sim_time      = 0.0;
double initial_sim_time   = 0.0;

alignas(EIGEN_MAX_ALIGN_BYTES) labrob::WalkingManager walking_manager;
labrob::Logger object_logger;

// ── Experiment duration report ───────────────────────────────────────────────
void printExperimentDuration() {
    if (!experiment_started) {
        std::cout << "Experiment duration: not started." << std::endl;
        return;
    }
    const double wall_s = std::chrono::duration<double>(Clock::now() - experiment_start).count();
    const double sim_s  = last_sim_time - initial_sim_time;

    const int    minutes = static_cast<int>(wall_s) / 60;
    const double seconds = wall_s - 60.0 * minutes;

    std::cout << std::fixed << std::setprecision(3)
              << "Experiment duration: " << wall_s << " s (wall clock";
    if (minutes > 0)
        std::cout << ", " << minutes << " min " << seconds << " s";
    std::cout << "), " << sim_s << " s (simulated), real-time factor "
              << (wall_s > 0.0 ? sim_s / wall_s : 0.0)
              << std::defaultfloat << std::endl;
}

// ── Signal handler ───────────────────────────────────────────────────────────
void signalHandler(int signum) {
    std::cerr << "Received signal " << signum << ", exiting..." << std::endl;
    printExperimentDuration();
    std::cout << "Do you want to save logs? [y/n]" << std::endl;
    std::string user_input;
    std::getline(std::cin, user_input);

    running = false;
    if (user_input == "y" || user_input == "Y" || user_input == "yes" ||
        user_input == "Yes" || user_input == "YES") {
        std::cout << "Saving logs..." << std::endl;
        walking_manager.saveLogs();
        // Object-specific channels go in their own folder, next to the standard
        // controller logs written by saveLogs().
        std::filesystem::create_directories("/tmp/object_logs");
        object_logger.save("/tmp/object_logs");
        std::cout << "Logs saved (controller logs + /tmp/object_logs)." << std::endl;
    } else {
        std::cout << "Logs not saved." << std::endl;
    }
    exit(signum);
}

// ── MuJoCo → RobotState ──────────────────────────────────────────────────────
labrob::RobotState robot_state_from_mujoco(mjModel* m, mjData* d) {
    labrob::RobotState rs;
    rs.position    = Eigen::Vector3d(d->qpos[0], d->qpos[1], d->qpos[2]);
    rs.orientation = Eigen::Quaterniond(d->qpos[3], d->qpos[4], d->qpos[5], d->qpos[6]);
    rs.linear_velocity  = rs.orientation.toRotationMatrix().transpose() *
                          Eigen::Vector3d(d->qvel[0], d->qvel[1], d->qvel[2]);
    rs.angular_velocity = Eigen::Vector3d(d->qvel[3], d->qvel[4], d->qvel[5]);

    for (int i = 1; i < m->njnt; ++i) {
        // The scene contains a second free joint (the carried object): skip any
        // joint that is not a 1-DoF articulation, otherwise a free joint would
        // be pushed into joint_state as if it were a scalar robot joint.
        if (m->jnt_type[i] != mjJNT_HINGE && m->jnt_type[i] != mjJNT_SLIDE) continue;
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

// ── Object setup helpers ─────────────────────────────────────────────────────

// Apply the size and the mass declared above to the object body of the model,
// so that the geometry of the experiment lives in this file and not in the XML.
// The inertia of a homogeneous box is recomputed accordingly, and mj_setConst()
// refreshes the derived constants (subtree masses, constraint inverse weights)
// that depend on it.
static void configureCarriedObject(mjModel* m, mjData* d) {
    const int obj_bid = mj_name2id(m, mjOBJ_BODY, "carried_object");
    const int obj_gid = mj_name2id(m, mjOBJ_GEOM, "carried_object_geom");
    if (obj_bid < 0 || obj_gid < 0) {
        std::cerr << "Object body/geom not found in the scene." << std::endl;
        exit(-1);
    }

    for (int i = 0; i < 3; ++i) m->geom_size[3 * obj_gid + i] = kObjectHalfSize[i];

    const double dx = 2.0 * kObjectHalfSize.x();
    const double dy = 2.0 * kObjectHalfSize.y();
    const double dz = 2.0 * kObjectHalfSize.z();
    m->body_mass[obj_bid] = kObjectMass;
    m->body_inertia[3 * obj_bid + 0] = kObjectMass * (dy * dy + dz * dz) / 12.0;
    m->body_inertia[3 * obj_bid + 1] = kObjectMass * (dx * dx + dz * dz) / 12.0;
    m->body_inertia[3 * obj_bid + 2] = kObjectMass * (dx * dx + dy * dy) / 12.0;

    // mj_setConst() also recomputes the visualisation statistics, which the
    // scene sets explicitly through <statistic>: save and restore them so the
    // camera framing does not change.
    const mjStatistic stat_backup = m->stat;
    mj_setConst(m, d);
    m->stat = stat_backup;
    mj_resetData(m, d);

    std::cout << "[OBJECT] size = " << dx << " x " << dy << " x " << dz
              << " m, mass = " << kObjectMass << " kg" << std::endl;
}

// Close the fingers on the object.
//
// The finger joints are unactuated: they are held by the springs of the "finger"
// default class in the model, which pull them towards their spring reference.
// Rewriting that reference with the grip angles is therefore what makes the
// hands hold the grip, and it keeps holding it for the whole run — setting only
// the initial qpos would let the fingers spring back open.
//
// Note that the fingers do not carry the object: the load path is the weld
// constraint anchored in the palm. The grip is what the grasp looks like, the
// weld is what the grasp does.
static void closeFingersOnObject(mjModel* m) {
    const std::map<std::string, double> grip = {
        {"left_hand_index_0_joint",   kFingerProximal},
        {"left_hand_index_1_joint",   kFingerDistal},
        {"left_hand_middle_0_joint",  kFingerProximal},
        {"left_hand_middle_1_joint",  kFingerDistal},
        {"left_hand_thumb_1_joint",   kThumbProximal},
        {"left_hand_thumb_2_joint",   kThumbDistal},
        // The right hand mirrors the left one: its finger joints turn about the
        // same local z axis but their travel range has the opposite sign.
        {"right_hand_index_0_joint",  -kFingerProximal},
        {"right_hand_index_1_joint",  -kFingerDistal},
        {"right_hand_middle_0_joint", -kFingerProximal},
        {"right_hand_middle_1_joint", -kFingerDistal},
        {"right_hand_thumb_1_joint",  -kThumbProximal},
        {"right_hand_thumb_2_joint",  -kThumbDistal},
    };

    for (const auto& [name, angle] : grip) {
        const int jid = mj_name2id(m, mjOBJ_JOINT, name.c_str());
        if (jid < 0) continue;
        // Clamp to the joint range: the grip angles are tuned by hand and an
        // out-of-range spring reference would just fight the joint limit.
        double a = angle;
        if (m->jnt_limited[jid]) {
            a = std::clamp(a, m->jnt_range[2 * jid], m->jnt_range[2 * jid + 1]);
        }
        m->qpos_spring[m->jnt_qposadr[jid]] = a;
    }
}

// Place the object between the two palms and re-anchor the two weld equality
// constraints on the configuration the experiment actually starts from.
//
// MuJoCo stores the relative pose of a weld in mjModel.eq_data, and the
// compiler fills it from the model reference configuration qpos0 — which is the
// zero posture, not the carrying posture set up here. Recomputing eq_data at
// runtime is what makes the grasp consistent with the initial posture, whatever
// that posture is.
//
// Layout of eq_data for a weld (see MuJoCo docs / engine_core_constraint.c):
//   [0..2]  the constrained point, expressed in body2 frame
//   [3..5]  the same point, expressed in body1 frame
//   [6..9]  orientation of body2 w.r.t. body1, as a quaternion
//   [10]    torque scale (left untouched)
// The constrained point is the grip point in the palm, not the origin of the
// wrist body: that is what makes the object hang off the hands (see
// kPalmOffsetInWrist for why it also stops the box from swinging).
static void placeObjectAndAnchorWelds(mjModel* m, mjData* d) {
    const int obj_bid  = mj_name2id(m, mjOBJ_BODY,  "carried_object");
    const int obj_jid  = mj_name2id(m, mjOBJ_JOINT, "carried_object_joint");
    const int lw_bid   = mj_name2id(m, mjOBJ_BODY,  "left_wrist_yaw_link");
    const int rw_bid   = mj_name2id(m, mjOBJ_BODY,  "right_wrist_yaw_link");

    // Forward kinematics on the carrying posture, to read where the hands are.
    mj_forward(m, d);

    // Grasp points = palm centres, i.e. the palm offset expressed in world.
    const mjtNum palm_local[3] = {kPalmOffsetInWrist.x(), kPalmOffsetInWrist.y(), kPalmOffsetInWrist.z()};
    mjtNum palm_l[3], palm_r[3], tmp[3];
    mju_mulMatVec3(tmp, d->xmat + 9 * lw_bid, palm_local);
    for (int i = 0; i < 3; ++i) palm_l[i] = d->xpos[3 * lw_bid + i] + tmp[i];
    mju_mulMatVec3(tmp, d->xmat + 9 * rw_bid, palm_local);
    for (int i = 0; i < 3; ++i) palm_r[i] = d->xpos[3 * rw_bid + i] + tmp[i];

    const double hand_distance = std::sqrt(
        (palm_l[0] - palm_r[0]) * (palm_l[0] - palm_r[0]) +
        (palm_l[1] - palm_r[1]) * (palm_l[1] - palm_r[1]) +
        (palm_l[2] - palm_r[2]) * (palm_l[2] - palm_r[2]));
    std::cout << "[OBJECT] distance between the palms in the carrying posture = "
              << hand_distance << " m (object width = " << 2.0 * kObjectHalfSize.y()
              << " m)" << std::endl;

    // Object pose: centred between the grip points, and oriented halfway between
    // the two hands. Aligning it with one hand instead would hand the whole
    // mismatch between the two wrists (a couple of degrees of yaw, since the
    // arms are mirrored rather than identical) to the other one, and the grip
    // would visibly bite deeper on that side.
    const int obj_qadr = m->jnt_qposadr[obj_jid];
    for (int i = 0; i < 3; ++i) d->qpos[obj_qadr + i] = 0.5 * (palm_l[i] + palm_r[i]);
    {
        const mjtNum* q_l = d->xquat + 4 * lw_bid;
        const mjtNum* q_r = d->xquat + 4 * rw_bid;
        // Take the two quaternions to the same hemisphere before averaging them,
        // otherwise q and -q (the same rotation) would cancel out.
        const mjtNum sign = (mju_dot(q_l, q_r, 4) < 0) ? -1.0 : 1.0;
        for (int i = 0; i < 4; ++i) d->qpos[obj_qadr + 3 + i] = q_l[i] + sign * q_r[i];
        mju_normalize4(d->qpos + obj_qadr + 3);
    }
    const int obj_dadr = m->jnt_dofadr[obj_jid];
    for (int i = 0; i < 6; ++i) d->qvel[obj_dadr + i] = 0.0;

    // Refresh the kinematics so that the object pose below is the new one.
    mj_forward(m, d);

    for (const char* eq_name : {"grasp_left", "grasp_right"}) {
        const int eq_id = mj_name2id(m, mjOBJ_EQUALITY, eq_name);
        if (eq_id < 0) {
            std::cerr << "Equality constraint " << eq_name << " not found." << std::endl;
            exit(-1);
        }
        // body1 is the object and body2 the hand, in this order, as declared in
        // the scene: the anchor below is expressed in the hand frame.
        const int b1 = m->eq_obj1id[eq_id];
        const int b2 = m->eq_obj2id[eq_id];
        if (b1 != obj_bid || (b2 != lw_bid && b2 != rw_bid)) {
            std::cerr << "Unexpected bodies on equality " << eq_name
                      << ": expected body1=carried_object, body2=<wrist>." << std::endl;
            exit(-1);
        }
        mjtNum* data = m->eq_data + mjNEQDATA * eq_id;

        // Constrained point on the hand: the grip point in the palm.
        mju_copy3(data + 0, palm_local);

        // The same physical point, expressed in the object frame.
        mjtNum p_grip_w[3], dp[3];
        mju_mulMatVec3(tmp, d->xmat + 9 * b2, palm_local);
        for (int i = 0; i < 3; ++i) p_grip_w[i] = d->xpos[3 * b2 + i] + tmp[i];
        mju_sub3(dp, p_grip_w, d->xpos + 3 * b1);
        mju_mulMatTVec3(data + 3, d->xmat + 9 * b1, dp);

        // Orientation of body2 w.r.t. body1: relpose = q1^-1 * q2.
        mjtNum q1_inv[4];
        mju_negQuat(q1_inv, d->xquat + 4 * b1);
        mju_mulQuat(data + 6, q1_inv, d->xquat + 4 * b2);
        mju_normalize4(data + 6);
    }

    std::cout << "[OBJECT] grasp welds anchored on the palms, on the initial "
                 "carrying posture." << std::endl;

    // Report how far each fingertip sits inside the side face of the box, so
    // that the grip can be checked without opening the viewer: a small positive
    // number means the finger closes on the face, a negative one means it never
    // reaches the object, a large positive one that it sinks through it.
    // Tip offsets are read off the model: index/middle phalanges extend along
    // their local +x, the thumb along its local -y (mirrored on the right hand).
    struct Tip { const char* body; mjtNum off[3]; };
    const Tip tips[] = {
        {"left_hand_index_1_link",   { 0.045,  0.0,   0.0}},
        {"left_hand_middle_1_link",  { 0.045,  0.0,   0.0}},
        {"left_hand_thumb_2_link",   { 0.0,   -0.045, 0.0}},
        {"right_hand_index_1_link",  { 0.045,  0.0,   0.0}},
        {"right_hand_middle_1_link", { 0.045,  0.0,   0.0}},
        {"right_hand_thumb_2_link",  { 0.0,    0.045, 0.0}},
    };
    for (const auto& tip : tips) {
        const int bid = mj_name2id(m, mjOBJ_BODY, tip.body);
        if (bid < 0) continue;
        mjtNum p_w[3], d_obj[3], p_obj[3];
        mju_mulMatVec3(tmp, d->xmat + 9 * bid, tip.off);
        for (int i = 0; i < 3; ++i) p_w[i] = d->xpos[3 * bid + i] + tmp[i];
        mju_sub3(d_obj, p_w, d->xpos + 3 * obj_bid);
        mju_mulMatTVec3(p_obj, d->xmat + 9 * obj_bid, d_obj);
        std::cout << "[GRIP] " << tip.body << ": "
                  << 1000.0 * (kObjectHalfSize.y() - std::abs(p_obj[1]))
                  << " mm inside the side face" << std::endl;
    }
}

// Point of the object the push is applied to, in the object frame [m].
// Zero means the centre of the box, i.e. a force shared equally by the two
// hands. For the curved walk the push is applied slightly off-centre, which is
// what a partner pulling on one side would do: the resulting yaw moment makes
// the two hands feel different forces, and the cooperative planner turns.
// The offset is a small fraction of the half-width on purpose. Pushing on the
// very end of a 32 cm box gives a ~1 N.m yaw moment, i.e. a differential of
// almost 3 N between hands 35 cm apart, which the robot cannot absorb; a fifth
// of that turns the walk into a wide arc that runs to the end of the experiment.
static Eigen::Vector3d pushOffsetInObject() {
    if (push_mode == PushMode::Curve) return Eigen::Vector3d(0.0, 0.18 * kObjectHalfSize.y(), 0.0);
    return Eigen::Vector3d::Zero();
}

// Force applied to the object in world frame [N], as a function of simulated
// time. This is the only excitation of the experiment: the robot has no other
// reason to move, everything else (hand admittance, footstep planning, walking)
// is a reaction to this force being felt through the grasp.
static Eigen::Vector3d objectPushForce(double t, const Eigen::Matrix3d& R_F_hac) {
    if (t < kPushStartTime || t >= kPushEndTime) return Eigen::Vector3d::Zero();

    // Smooth ramp in and out over kPushRampTime. A step force is an impulsive
    // excitation that the observer (which low-pass filters the wrist wrenches
    // with a ~0.3 s time constant) reports late and the admittance then chases;
    // a human partner leaning on the object does not push like that either.
    const double ramp_in  = std::min(1.0, (t - kPushStartTime) / kPushRampTime);
    const double ramp_out = std::min(1.0, (kPushEndTime - t) / kPushRampTime);
    const double s = 0.5 * (1.0 - std::cos(M_PI * std::min(ramp_in, ramp_out)));

    switch (push_mode) {
        case PushMode::Lateral:
            // Sideways pull: makes the robot side-step.
            return s * Eigen::Vector3d(0.0, kPushMagnitudeLateral, 0.0);
        case PushMode::Curve:
            // Force kept constant in the local frame F of the HAC (which follows
            // the support foot), so that as the robot turns the force turns with
            // it. Applied off-centre (see kPushOffsetInObject), which is what
            // actually bends the path: a force through the centre of the object
            // is felt identically by the two hands and only produces a straight
            // translation of the pair.
            return s * (R_F_hac * Eigen::Vector3d(kPushMagnitude, 0.0, -0.5 * kPushMagnitude));
        case PushMode::Forward:
        default:
            // Straight pull along +x.
            return s * Eigen::Vector3d(kPushMagnitude, 0.0, 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────

int main(const int argc, const char* argv[]) {

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-viz")        useViz = false;
        else if (a == "--stand")    reactiveStanding = true;  // stay in place, do not step
        else if (a == "--verbose")  verboseCoop = true;
        else if (a == "--forward")  push_mode = PushMode::Forward;
        else if (a == "--lateral")  push_mode = PushMode::Lateral;
        else if (a == "--curve")    push_mode = PushMode::Curve;
    }

    signal(SIGINT, signalHandler);

    // Load MJCF:
    mj_loadAllPluginLibraries("/usr/local/lib/mujoco", nullptr);
    const int kErrorLength = 1024;
    char loadError[kErrorLength] = "";
    mjModel* mj_model_ptr = mj_loadXML(kObjectScenePath.data(), nullptr, loadError, kErrorLength);
    if (!mj_model_ptr) {
        std::cerr << "mj_loadXML failed: " << loadError << std::endl;
        return -1;
    }
    mjData* mj_data_ptr = mj_makeData(mj_model_ptr);

    // Apply the size/mass of the object declared at the top of this file.
    configureCarriedObject(mj_model_ptr, mj_data_ptr);

    // Close the fingers on the object: this rewrites the spring reference of the
    // finger joints, so it has to come before the initial state is built below
    // (which starts every spring joint at its reference).
    closeFingersOnObject(mj_model_ptr);

    // MuJoCo initial state: same as main.cpp, except that the arms start in the
    // carrying posture (joint_initial_positions_object) instead of the default
    // one. The HAC rest positions and the WBC postural reference are both built
    // from the measured initial configuration inside WalkingManager::init(), so
    // changing this map is enough to move the whole "hands at rest" definition.
    for (int i = 0; i < mj_model_ptr->nq; ++i) mj_data_ptr->qpos[i] = 0.0;
    mj_data_ptr->qpos[2] = 0.728112;
    mj_data_ptr->qpos[3] = 1;
    for (int i = 0; i < mj_model_ptr->njnt; ++i) {
        const char* name = mj_id2name(mj_model_ptr, mjOBJ_JOINT, i);
        auto it = joint_initial_positions_object.find(name);
        if (it != joint_initial_positions_object.end())
            mj_data_ptr->qpos[mj_model_ptr->jnt_qposadr[i]] = it->second;
    }
    // Finger joints are unactuated and held by springs: start them at their
    // spring reference (same rationale as in main.cpp).
    for (int i = 0; i < mj_model_ptr->njnt; ++i) {
        if (mj_model_ptr->jnt_stiffness[i] > 0.0) {
            int adr = mj_model_ptr->jnt_qposadr[i];
            mj_data_ptr->qpos[adr] = mj_model_ptr->qpos_spring[adr];
        }
    }

    // Put the object in the hands and anchor the grasp on this configuration.
    placeObjectAndAnchorWelds(mj_model_ptr, mj_data_ptr);

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

    // Declare the static load held by each hand: half of the object weight,
    // pointing down, expressed as a force exerted ON the robot. This is the
    // rest force f_i_bar of the admittance (so that simply holding the object
    // is an equilibrium and does not make the hands drift), and it also enables
    // the compensation of the payload inside the WBC inverse dynamics. The grip
    // point is passed along because the load hangs from the palm: its weight is
    // also a moment on the wrist, and the wrist pitch cannot hold it on its own.
    const Eigen::Vector3d f_hand_bar(0.0, 0.0, -0.5 * kObjectMass * 9.81);
    walking_manager.setCarriedObjectLoad(f_hand_bar, f_hand_bar, kPalmOffsetInWrist);

    labrob::MujocoUI* mujoco_ui_ptr = useViz
        ? labrob::MujocoUI::getInstance(mj_model_ptr, mj_data_ptr)
        : nullptr;
    static constexpr int framerate = 60;

    experiment_start   = Clock::now();
    initial_sim_time   = mj_data_ptr->time;
    last_sim_time      = mj_data_ptr->time;
    experiment_started = true;

    const int obj_bid = mj_name2id(mj_model_ptr, mjOBJ_BODY, "carried_object");
    const int lw_bid  = mj_name2id(mj_model_ptr, mjOBJ_BODY, "left_wrist_yaw_link");
    const int rw_bid  = mj_name2id(mj_model_ptr, mjOBJ_BODY, "right_wrist_yaw_link");

    Eigen::Vector3d f_object = Eigen::Vector3d::Zero();

    std::cout << "Push mode: "
              << (push_mode == PushMode::Lateral ? "lateral"
                  : push_mode == PushMode::Curve ? "curve" : "forward")
              << " | "
              << (push_mode == PushMode::Lateral ? kPushMagnitudeLateral : kPushMagnitude)
              << " N applied to the object from t="
              << kPushStartTime << " s to t=" << kPushEndTime << " s" << std::endl;

    // ── Main loop ────────────────────────────────────────────────────────────
    while (running) {
        if (useViz && mujoco_ui_ptr->windowShouldClose()) break;

        mjtNum simstart = mj_data_ptr->time;
        while (mj_data_ptr->time - simstart < 1.0 / framerate) {

            // First tick: close the loops on the ground-truth MuJoCo state.
            if (!isWBCLoopClosed) {
                robot_state      = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);
                isWBCLoopClosed  = true;
                isMPCLoopClosed  = true;
                isObserverActive = true;
            }

            labrob::JointCommand joint_command;
            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)] = 0.0;
            }

            // ── Excitation: force applied to the OBJECT ──────────────────────
            // It is applied to the object body and reaches the two hands through
            // the grasp welds; the wrench observer then sees it at the wrists and
            // feeds it to the hand admittance controller, exactly as it would
            // with a human partner pushing the carried object.
            f_object = objectPushForce(mj_data_ptr->time, walking_manager.get_R_F_hac());
            mj_data_ptr->xfrc_applied[obj_bid * 6 + 0] = f_object.x();
            mj_data_ptr->xfrc_applied[obj_bid * 6 + 1] = f_object.y();
            mj_data_ptr->xfrc_applied[obj_bid * 6 + 2] = f_object.z();

            // xfrc_applied acts on the body origin, so a push applied somewhere
            // else on the object (pushOffsetInObject()) is reproduced by adding
            // the transport moment r x f, with r the offset rotated into world.
            {
                Eigen::Matrix3d R_obj;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        R_obj(r, c) = mj_data_ptr->xmat[9 * obj_bid + 3 * r + c];
                const Eigen::Vector3d m_object = (R_obj * pushOffsetInObject()).cross(f_object);
                mj_data_ptr->xfrc_applied[obj_bid * 6 + 3] = m_object.x();
                mj_data_ptr->xfrc_applied[obj_bid * 6 + 4] = m_object.y();
                mj_data_ptr->xfrc_applied[obj_bid * 6 + 5] = m_object.z();
            }

            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                std::string jname = mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid);
                measured_joint_velocity[i] = robot_state.joint_state.at(jname).vel;
            }

            walking_manager.update(robot_state, joint_command);

            mj_step1(mj_model_ptr, mj_data_ptr);
            for (int i = 0; i < mj_model_ptr->nu; ++i) {
                int jid = mj_model_ptr->actuator_trnid[i * 2];
                mj_data_ptr->ctrl[i] = joint_command[mj_id2name(mj_model_ptr, mjOBJ_JOINT, jid)];
            }
            mj_step2(mj_model_ptr, mj_data_ptr);

            robot_state = robot_state_from_mujoco(mj_model_ptr, mj_data_ptr);

            // ── Experiment-specific logs ─────────────────────────────────────
            // Ground truth of the excitation, pose of the object and forces the
            // observer reconstructs at the wrists (what the HAC actually reacts to).
            object_logger.log("object_applied_force", f_object);
            object_logger.log("object_position", Eigen::Vector3d(
                mj_data_ptr->xpos[3 * obj_bid + 0],
                mj_data_ptr->xpos[3 * obj_bid + 1],
                mj_data_ptr->xpos[3 * obj_bid + 2]));
            object_logger.log("estimated_force_lwrist_obj",
                              walking_manager.get_estimated_wrist_forces().head<3>());
            object_logger.log("estimated_force_rwrist_obj",
                              walking_manager.get_estimated_wrist_forces().tail<3>());

            last_sim_time = mj_data_ptr->time;

            if (mj_data_ptr->time > kSimDuration) {
                std::cout << "Reached " << kSimDuration
                          << " s of simulated time, stopping..." << std::endl;
                signalHandler(SIGINT);
            }
        }

        if (useViz) {
            const Eigen::Vector3d p_lhand(
                mj_data_ptr->xpos[3 * lw_bid + 0],
                mj_data_ptr->xpos[3 * lw_bid + 1],
                mj_data_ptr->xpos[3 * lw_bid + 2]);
            const Eigen::Vector3d p_rhand(
                mj_data_ptr->xpos[3 * rw_bid + 0],
                mj_data_ptr->xpos[3 * rw_bid + 1],
                mj_data_ptr->xpos[3 * rw_bid + 2]);
            const Eigen::Vector3d p_object(
                mj_data_ptr->xpos[3 * obj_bid + 0],
                mj_data_ptr->xpos[3 * obj_bid + 1],
                mj_data_ptr->xpos[3 * obj_bid + 2]);

            // Blue/red arrows: forces estimated at the hands. Green arrow: force
            // applied to the object, i.e. the input of the experiment.
            mujoco_ui_ptr->renderWithObjectForce(
                p_lhand, walking_manager.get_estimated_wrist_forces().head<3>(),
                p_rhand, walking_manager.get_estimated_wrist_forces().tail<3>(),
                p_object, f_object);
        }
    }

    printExperimentDuration();

    mj_deleteData(mj_data_ptr);
    mj_deleteModel(mj_model_ptr);
    return 0;
}
