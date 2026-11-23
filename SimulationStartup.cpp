#include "SimulationStartup.hpp"
#include "MujocoUI.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

namespace labrob {
namespace {

using Pose = std::vector<mjtNum>;

int modelId(const mjModel& m, mjtObj type, const std::string& name) {
  int id = mj_name2id(&m, type, name.c_str());
  if (id < 0) throw std::runtime_error("Missing G1 model element: " + name);
  return id;
}

void setJoint(const mjModel& m, Pose& pose, const std::string& name, double angle) {
  pose[m.jnt_qposadr[modelId(m, mjOBJ_JOINT, name + "_joint")]] = angle;
}

// Pitch and elbow flexion use the same sign on both arms. Roll and yaw are mirrored.
void setArms(const mjModel& m, Pose& pose, double pitch, double roll, double yaw, double elbow) {
  for (const std::string side : {"left_", "right_"}) {
    const double sign = side == "left_" ? 1.0 : -1.0;
    setJoint(m, pose, side + "shoulder_pitch", pitch);
    setJoint(m, pose, side + "shoulder_roll", sign * roll);
    setJoint(m, pose, side + "shoulder_yaw", sign * yaw);
    setJoint(m, pose, side + "elbow", elbow);
  }
}

void setFists(const mjModel& m, Pose& pose) {
  for (const std::string side : {"left_hand_", "right_hand_"}) {
    const double sign = side == "left_hand_" ? 1.0 : -1.0;
    setJoint(m, pose, side + "thumb_1", sign * 0.80);
    setJoint(m, pose, side + "thumb_2", sign * 0.65);
    for (const std::string finger : {"index", "middle"}) {
      setJoint(m, pose, side + finger + "_0", -sign * 0.55);
      setJoint(m, pose, side + finger + "_1", -sign * 1.40);
    }
  }
}

using Clock = std::chrono::steady_clock;
constexpr auto frame_period = std::chrono::milliseconds(16);

void renderFrame(MujocoUI& ui, Clock::time_point start) {
  ui.render();
  std::this_thread::sleep_until(start + frame_period);
}

bool waitForKey(MujocoUI& ui, int key, const char* message) {
  ui.promptStage(key, message);
  while (!ui.windowShouldClose() && !ui.takeStagePress()) {
    renderFrame(ui, Clock::now());
  }
  return !ui.windowShouldClose();
}

bool moveTo(const mjModel& m, mjData& d, MujocoUI& ui, const Pose& target, int frames) {
  const Pose start(d.qpos, d.qpos + m.nq);
  for (int frame = 1; frame <= frames; ++frame) {
    if (ui.windowShouldClose()) return false;
    const auto now = Clock::now();
    const double t = static_cast<double>(frame) / frames;
    const double blend = t * t * (3.0 - 2.0 * t);
    // The base stays fixed throughout the three pre-walk pose changes.
    for (int j = 7; j < m.nq; ++j) d.qpos[j] = start[j] + blend * (target[j] - start[j]);
    mj_forward(&m, &d);
    renderFrame(ui, now);
  }
  return true;
}

void enableStandingPd(mjModel& m, mjData& d, const std::vector<PdGains>& gains) {
  // Balance gravity with two vertical reactions whose resultant passes through the CoM.
  // Only joint motors receive the corresponding feedforward torques.
  const int left = modelId(m, mjOBJ_BODY, "left_ankle_roll_link");
  const int right = modelId(m, mjOBJ_BODY, "right_ankle_roll_link");
  const int pelvis = modelId(m, mjOBJ_BODY, "pelvis");
  const auto* com = d.subtree_com + 3 * pelvis;
  const double left_y = d.xpos[3 * left + 1], right_y = d.xpos[3 * right + 1];
  const double left_load = (com[1] - right_y) / (left_y - right_y);
  const double weight = -m.body_subtreemass[pelvis] * m.opt.gravity[2];
  std::vector<mjtNum> support(m.nv, 0.0);
  for (int foot : {left, right}) {
    const mjtNum point[] = {com[0], d.xpos[3 * foot + 1], 0.0};
    const mjtNum force[] = {0.0, 0.0, weight * (foot == left ? left_load : 1.0 - left_load)};
    mj_applyFT(&m, &d, force, nullptr, point, foot, support.data());
  }

  // Integrate motor damping implicitly, especially for the low-inertia finger joints.
  m.opt.integrator = mjINT_IMPLICITFAST;
  for (int i = 0; i < m.nu; ++i) {
    const int joint = m.actuator_trnid[2 * i];
    const int v = m.jnt_dofadr[joint];
    auto* bias = m.actuator_biasprm + mjNBIAS * i;
    m.actuator_biastype[i] = mjBIAS_AFFINE;
    // tau = gravity - support + Kp * (q_ref - q) - Kd * dq.
    bias[0] = d.qfrc_bias[v] - support[v] + gains[i].kp * d.qpos[m.jnt_qposadr[joint]];
    bias[1] = -gains[i].kp;
    bias[2] = -gains[i].kd;
  }
}

}  // namespace

bool prepareSimulation(mjModel& m, mjData& d, MujocoUI& ui, const std::vector<PdGains>& motor_gains) {
  Pose pose(d.qpos, d.qpos + m.nq);
  // Mirrored poses keep the palms beside the torso as the elbows move outward.
  setArms(m, pose, -0.0473, 0.20, 1.3954, 1.7231);
  mju_copy(d.qpos, pose.data(), m.nq);
  mj_forward(&m, &d);

  if (!waitForKey(ui, 1, "Ready. Press 1 to open both arms slightly to the sides.")) return false;
  setArms(m, pose, -0.0554, 0.42, 1.4655, 1.9414);
  if (!moveTo(m, d, ui, pose, 75)) return false;

  if (!waitForKey(ui, 2, "Stage 1 complete. Forearms are pointing down. Press 2 to close both hands.")) return false;
  setFists(m, pose);
  if (!moveTo(m, d, ui, pose, 75)) return false;

  if (!waitForKey(ui, 3, "Both fists are closed and clear of the legs. Press 3 to slowly raise the forearms.")) return false;
  setArms(m, pose, 0.0, 0.48, -0.010, -0.053);
  // A single eased lift takes at least six seconds at the startup frame rate.
  if (!moveTo(m, d, ui, pose, 375)) return false;

  if (!waitForKey(ui, 4, "Initialization complete. Both forearms are level. Press 4 for PD standing.")) return false;
  enableStandingPd(m, d, motor_gains);
  ui.promptStage(5, "Standing under PD control. Press 5 to switch to WBC standing.");
  while (!ui.windowShouldClose() && !ui.takeStagePress()) {
    const auto now = Clock::now();
    const double end = d.time + std::chrono::duration<double>(frame_period).count();
    do { mj_step(&m, &d); } while (d.time + 1e-9 < end);
    renderFrame(ui, now);
  }
  mj_forward(&m, &d);
  return !ui.windowShouldClose();
}

}  // namespace labrob
