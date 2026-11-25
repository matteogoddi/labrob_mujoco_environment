#include "SimulationObserver.hpp"

#include <hrp4_locomotion/WalkingManager.hpp>
#include <json/json.h>
#include <hrp4_locomotion/globals.h>
#include <chrono>
#include <sys/socket.h>
#include <unistd.h>
#include <cstddef>
#include <cstring>
#include <iostream>

namespace labrob {
namespace {
Json::Value xyz(const Eigen::Vector3d& v) {
  Json::Value value(Json::arrayValue);
  for (int i = 0; i < 3; ++i) value.append(v[i]);
  return value;
}

Json::Value pose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q) {
  Json::Value value = xyz(p);
  for (int i = 0; i < 4; ++i) value.append(q.coeffs()[i]);  // x, y, z, w
  return value;
}

Json::Value forceAt(const Eigen::Vector3d& point, const Eigen::Vector3d& force) {
  Json::Value value;
  value["point"] = xyz(point);
  value["force"] = xyz(force);
  return value;
}
void addMeasuredPose(Json::Value& frame, const mjModel& m, const mjData& d, const RobotState& state) {
  frame["pelvis"] = pose(state.position, state.orientation);
  for (int j = 1; j < m.njnt; ++j) {
    const char* name = mj_id2name(&m, mjOBJ_JOINT, j);
    frame["joints"][name] = state.joint_state[name].pos;
    frame["velocity"][name] = state.joint_state[name].vel;
    if (d.time > 0) frame["effort"][name] = d.qfrc_actuator[m.jnt_dofadr[j]];
  }
}

void sendFrame(int fd, Json::Value& frame, double time) {
  frame["time"] = time;
  frame["stamp_ns"] = Json::Int64(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  writer["precision"] = 12;
  const std::string bytes = Json::writeString(writer, frame);
  send(fd, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
}
}

Json::Value WalkingManager::visualizationSnapshot(bool hardware) const {
  Json::Value frame;
  const auto& plan = walking_data_.footstep_plan;
  frame["phase"] = to_string(walking_data_.getWalkingState());
  frame["support"] = "both";
  if (plan.front().getWalkingState() == WalkingState::SingleSupport) {
    frame["support"] = plan.front().getFeetPlacement().getSupportFoot() == Foot::LEFT ? "left" : "right";
  }

  auto& landings = frame["landings"] = Json::Value(Json::arrayValue);
  for (size_t i = 0; i + 1 < plan.size(); ++i) {
    if (plan[i].getWalkingState() != WalkingState::SingleSupport) continue;
    const bool left_lands = plan[i].getFeetPlacement().getSupportFoot() == Foot::RIGHT;
    const auto& next = plan[i + 1].getFeetPlacement();
    const auto& foot = left_lands ? next.getLeftFootConfiguration() : next.getRightFootConfiguration();
    Json::Value landing;
    landing["foot"] = left_lands ? "left" : "right";
    landing["pose"] = pose(foot.p, Eigen::Quaterniond(foot.R));
    landings.append(landing);
  }

  frame["desired"]["zmp"] = xyz(des_zmp_position_log_.back());
  frame["desired"]["com"] = xyz(des_com_position_log_.back());
  frame["desired"]["left_foot"] = xyz(p_lsole_des_log_.back());
  frame["desired"]["right_foot"] = xyz(p_rsole_des_log_.back());
  if (!hardware) {
    frame["feedback"]["com"] = xyz(sim_com_position_log_.back());
    frame["feedback"]["left_foot"] = xyz(p_lsole_sim_log_.back());
    frame["feedback"]["right_foot"] = xyz(p_rsole_sim_log_.back());
  } else if (isEKFactive && t_msec_ - controller_timestep_msec_ >= startTimeEKF) {
    frame["estimated_base"] = pose(fb_robot_state.position, fb_robot_state.orientation);
  }
  for (const auto id : {lsole_idx_, rsole_idx_, torso_idx_}) {
    const auto& tf = sim_robot_data.oMf[id];
    frame["frames"][robot_model.frames[id].name] = pose(tf.translation(), Eigen::Quaterniond(tf.rotation()));
  }
  frame["wbc"]["left"] = forceAt(sim_robot_data.oMf[lsole_idx_].translation(),
                                  whole_body_controller_ptr_->getLeftFootWrench().head<3>());
  frame["wbc"]["right"] = forceAt(sim_robot_data.oMf[rsole_idx_].translation(),
                                   whole_body_controller_ptr_->getRightFootWrench().head<3>());
  frame["estimated"]["left"] = forceAt(fb_robot_data.oMf[lsole_idx_].translation(), estimated_force.head<3>());
  frame["estimated"]["right"] = forceAt(fb_robot_data.oMf[rsole_idx_].translation(), estimated_force.tail<3>());
  // Replay this solve on the MPC grid, starting at the exact filtered input state.
  const auto& x = ismpc_ptr_->getInputSequenceX();
  const auto& y = ismpc_ptr_->getInputSequenceY();
  const auto& z = ismpc_ptr_->getInputSequenceZ();
  auto predicted = kf_LipState;
  frame["com"].append(xyz(predicted.com_pos_));
  frame["zmp"].append(xyz(predicted.zmp_pos_));
  for (Eigen::Index k = 0; k < x.size(); ++k) {
    predicted = discrete_lip_dynamics_ptr_mpc_->integrate(predicted, {x[k], y[k], z[k]});
    frame["com"].append(xyz(predicted.com_pos_));
    frame["zmp"].append(xyz(predicted.zmp_pos_));
  }
  return frame;
}

SimulationObserver::SimulationObserver(bool hardware)
    : socket_(socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0)) {
  const std::string name = (hardware ? "g1-wbc-hardware-" : "g1-wbc-rviz-") + std::to_string(getuid());
  receiver_.sun_family = AF_UNIX;
  std::memcpy(receiver_.sun_path + 1, name.data(), name.size());
  address_length_ = offsetof(sockaddr_un, sun_path) + 1 + name.size();
  if (socket_ < 0) std::perror("RViz observer socket");
}

SimulationObserver::~SimulationObserver() {
  if (socket_ >= 0) close(socket_);
}

bool SimulationObserver::sampleDue(double time) {
  if (socket_ < 0 || time + 1e-9 < next_sample_) return false;
  constexpr double period = 1.0 / 60.0;
  if (next_sample_ == 0 || time - next_sample_ >= period) next_sample_ = time;
  next_sample_ += period;  // Preserve the fractional deadline on the 2 ms control grid.
  return receiverAvailable();
}

bool SimulationObserver::receiverAvailable() const {
  // Check for a listener before building any JSON. Reconnect also handles receiver restarts.
  return socket_ >= 0 && connect(socket_, reinterpret_cast<const sockaddr*>(&receiver_), address_length_) == 0;
}

void SimulationObserver::sampleStartup(const mjModel& m, const mjData& d, const RobotState& state) {
  // Startup renders even when simulation time is frozen while waiting for a key.
  if (!receiverAvailable()) return;
  Json::Value frame;
  addMeasuredPose(frame, m, d, state);
  sendFrame(socket_, frame, d.time);
}

void SimulationObserver::sampleController(double time, const WalkingManager& manager, const RobotState& reference) {
  if (!sampleDue(time)) return;
  auto frame = manager.visualizationSnapshot(true);
  frame["pelvis"] = pose(reference.position, reference.orientation);
  frame["contacts"] = frame["external"] = Json::Value(Json::arrayValue);
  sendFrame(socket_, frame, time);
}

void SimulationObserver::sample(double time, const WalkingManager& manager, const RobotState& state,
                                const mjModel& m, const mjData& d) {
  if (!sampleDue(time)) return;
  auto frame = manager.visualizationSnapshot(false);
  addMeasuredPose(frame, m, d, state);
  for (int i = 0; i < m.nu; ++i) {
    const char* name = mj_id2name(&m, mjOBJ_ACTUATOR, i);
    if (std::string(name).find("_hand_") == std::string::npos) frame["torque_command"][name] = d.ctrl[i];
  }
  auto& contacts = frame["contacts"] = Json::Value(Json::arrayValue);
  Eigen::Vector3d support_force = Eigen::Vector3d::Zero();
  Eigen::Vector3d support_moment = Eigen::Vector3d::Zero();
  const int left_sole = mj_name2id(&m, mjOBJ_BODY, "left_ankle_roll_link");
  const int right_sole = mj_name2id(&m, mjOBJ_BODY, "right_ankle_roll_link");
  for (int i = 0; i < d.ncon; ++i) {
    const auto& c = d.contact[i];
    mjtNum local[6], world[3];
    mj_contactForce(&m, &d, i, local);
    mju_mulMatTVec(world, c.frame, local, 3, 3);
    // The solver reports the force on geom2. For world contact show the force on the robot.
    const bool reverse = m.geom_bodyid[c.geom2] == 0;
    auto contact = forceAt(Eigen::Map<const Eigen::Vector3d>(c.pos),
                           (reverse ? -1.0 : 1.0) * Eigen::Map<const Eigen::Vector3d>(world));
    contact["body"] = mj_id2name(&m, mjOBJ_BODY, m.geom_bodyid[reverse ? c.geom1 : c.geom2]);
    contacts.append(contact);
    const int other_body = m.geom_bodyid[reverse ? c.geom1 : c.geom2];
    if (m.geom_bodyid[reverse ? c.geom2 : c.geom1] == 0 &&
        (other_body == left_sole || other_body == right_sole)) {
      mjtNum torque[3];
      mju_mulMatTVec(torque, c.frame, local + 3, 3, 3);
      const double sign = reverse ? -1.0 : 1.0;
      const Eigen::Vector3d f = sign * Eigen::Map<const Eigen::Vector3d>(world);
      support_force += f;
      support_moment += Eigen::Map<const Eigen::Vector3d>(c.pos).cross(f)
                      + sign * Eigen::Map<const Eigen::Vector3d>(torque);
    }
  }
  if (support_force.z() > 1.0) {
    // Intersect the resultant support wrench with the same horizontal plane as the reference.
    const double z = frame["desired"]["zmp"][2].asDouble();
    frame["contact_zmp"] = xyz({(z * support_force.x() - support_moment.y()) / support_force.z(),
                                (z * support_force.y() + support_moment.x()) / support_force.z(), z});
  }
  auto& external = frame["external"] = Json::Value(Json::arrayValue);
  for (int b = 1; b < m.nbody; ++b) {
    Eigen::Map<const Eigen::Vector3d> force(d.xfrc_applied + 6 * b);
    if (force.squaredNorm() == 0) continue;
    auto applied = forceAt(Eigen::Map<const Eigen::Vector3d>(d.xipos + 3 * b), force);
    applied["body"] = mj_id2name(&m, mjOBJ_BODY, b);
    external.append(applied);
  }
  sendFrame(socket_, frame, time);
}
}
