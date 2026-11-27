// Suspended Stage 1 only. All operator input stays on the foreground terminal.
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

#include "G1Body.hpp"
#include "Stage1Pose.hpp"

namespace {
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
using namespace unitree::robot;
using State = unitree_hg::msg::dds_::LowState_;
using Cmd = unitree_hg::msg::dds_::LowCmd_;
using Angles = std::array<float, 8>;
constexpr std::array<int, 8> joint{15, 16, 17, 18, 22, 23, 24, 25};
// Mode 5 Rev 1.0 URDF limits, including the asymmetric shoulder rolls.
constexpr Angles lower{-3.0892f, -1.5882f, -2.618f, -1.0472f,
                       -3.0892f, -2.2515f, -2.618f, -1.0472f};
constexpr Angles upper{2.6704f, 2.2515f, 2.618f, 2.0944f, 2.6704f, 1.5882f, 2.618f, 2.0944f};
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }

template <class T>
uint32_t crc(const T& msg) {
  static_assert(sizeof(T) % 4 == 0);
  std::array<uint32_t, sizeof(T) / 4> words;
  std::memcpy(words.data(), &msg, sizeof(msg));
  return Crc32Core(words.data(), words.size() - 1);
}

class Terminal {
  termios saved{};

 public:
  Terminal() {
    if (!isatty(STDIN_FILENO) || tcgetpgrp(0) != getpgrp() || tcgetattr(0, &saved))
      throw std::runtime_error("Start step1 in a foreground interactive terminal.");
    termios raw = saved;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_iflag &= ~IXON;
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(0, TCSAFLUSH, &raw)) throw std::runtime_error("Cannot configure keyboard.");
  }
  ~Terminal() { tcsetattr(0, TCSANOW, &saved); }
  char key() const {
    if (interrupted || tcgetpgrp(0) != getpgrp()) return 'x';
    pollfd p{0, POLLIN, 0};
    int n = poll(&p, 1, 0);
    if (n < 0) return errno == EINTR && !interrupted ? 0 : 'x';
    if (!n) return 0;
    if (p.revents != POLLIN) return 'x';
    char keys[64];
    auto count = read(0, keys, sizeof(keys));
    if (count <= 0) return 'x';
    // A stop wins over all queued keys. Discard pasted/queued multi-key commands.
    for (int i = 0; i < count; ++i)
      if (keys[i] == 'x' || keys[i] == ' ' || keys[i] == 27 || keys[i] == 'q') return 'x';
    return count == 1 ? keys[0] : 0;
  }
};

struct Sample {
  Time received{};
  uint32_t tick{};
  bool valid{}, compatible{};
  Angles q{}, dq{};
  bool fresh(Time now) const { return valid && now - received < 100ms; }
};

class Feedback {
  std::mutex mutex;
  Sample last;

 public:
  void receive(const void* data) {
    const auto& state = *static_cast<const State*>(data);
    std::lock_guard<std::mutex> guard(mutex);
    if (crc(state) != state.crc()) {
      last.valid = false;
      return;
    }
    last.compatible = state.mode_machine() == 5 && state.mode_pr() == 0;
    if (!last.compatible) {
      last.valid = false;
      return;
    }
    if (last.received != Time{} && state.tick() == last.tick) return;
    Sample s;
    s.valid = s.compatible = true;
    s.tick = state.tick();
    s.received = Clock::now();
    for (size_t i = 0; i < joint.size(); ++i) {
      const auto& m = state.motor_state()[joint[i]];
      s.q[i] = m.q();
      s.dq[i] = m.dq();
      if (m.motorstate() || !std::isfinite(m.q()) || !std::isfinite(m.dq()) ||
          !std::isfinite(m.tau_est()) || m.q() < lower[i] || m.q() > upper[i] ||
          std::abs(m.dq()) > 1.0f || std::abs(m.tau_est()) > 15.f)
        s.valid = false;
    }
    last = s;
  }
  Sample get() {
    std::lock_guard<std::mutex> guard(mutex);
    return last;
  }
};

// RPCs have bounded timeouts and never run on the keyboard / command thread.
// Each async operation sends exactly one request; a late reply cannot start a move.
enum class Operation { check, release, restore };
struct Reply {
  int code;
  std::string name;
};
class Motion {
  b2::MotionSwitcherClient sdk;
  std::future<Reply> pending;
  Operation operation = Operation::check;
  Time requested{};

 public:
  Time verified{};
  std::string name;
  Motion() {
    sdk.SetTimeout(0.4f);
    sdk.Init();
  }
  bool busy() const { return pending.valid(); }
  bool fresh(Time now) const { return verified != Time{} && now - verified < 750ms; }
  void request(Operation op, std::string target = {}) {
    requested = Clock::now();
    operation = op;
    if (op != Operation::check) verified = {};
    pending = std::async(std::launch::async, [this, op, target] {
      Reply r{};
      std::string form;
      if (op == Operation::check)
        r.code = sdk.CheckMode(form, r.name);
      else if (op == Operation::release)
        r.code = sdk.ReleaseMode();
      else
        r.code = sdk.SelectMode(target);
      return r;
    });
  }
  void update(Time now) {
    if (busy() && pending.wait_for(0ms) == std::future_status::ready) {
      auto r = pending.get();
      if (r.code && operation != Operation::check)
        throw std::runtime_error("Mode change RPC failed (" + std::to_string(r.code) +
                                 "). Verify mode on the remote.");
      if (r.code) {
        if (verified != Time{}) std::cerr << "Motion service RPC failed (" << r.code << ").\n";
        verified = {};
      } else if (operation == Operation::check) {
        if (verified == Time{} || name != r.name)
          std::cout << "Motion service: " << (r.name.empty() ? "released" : r.name) << std::endl;
        verified = requested;
        name = r.name;
      }
      if (operation != Operation::check) request(Operation::check);
    }
    if (!busy() && now - requested >= 250ms) request(Operation::check);
  }
};

class Arms {
  ChannelPtr<Cmd> writer;
  Time start{}, last_write{}, settled_since{};
  Angles initial{}, target{};
  bool moving = false, completed = false;

 public:
  bool active() const { return moving; }
  void connect() {
    if (!writer) writer = ChannelFactory::Instance()->CreateSendChannel<Cmd>("rt/lowcmd");
  }
  bool soleWriter(dds::sub::DataReader<Cmd>& reader) const {
    if (!writer) return false;
    // The SDK's C++ matched_publications() throws UnsupportedError in Foxy.
    dds_instance_handle_t matched{}, own{};
    auto output = writer->GetWriter()->GetNative();
    return dds_get_instance_handle(output.delegate()->get_ddsc_entity(), &own) == DDS_RETCODE_OK &&
           dds_get_matched_publications(reader.delegate()->get_ddsc_entity(), &matched, 1) == 1 &&
           matched == own;
  }
  bool startMove(const Sample& s) {
    for (size_t i = 0; i < joint.size(); ++i) {
      const auto k = i % 4;
      const float sign = i >= 4 && (k == 1 || k == 2) ? -1.f : 1.f;
      target[i] = s.q[i] + sign * (labrob::open_arm_pose[k] - labrob::initial_arm_pose[k]);
      if (std::abs(s.dq[i]) > 0.15f || target[i] < lower[i] + 0.03f ||
          target[i] > upper[i] - 0.03f) {
        std::cerr << "Start rejected at " << body_joint_names[joint[i]]
                  << ": check rest pose and travel.\n";
        return false;
      }
    }
    initial = s.q;
    start = last_write = Clock::now();
    moving = true;
    std::cout << "Step 1 moving. Space / x stops. Targets in radians:";
    for (float v : target) std::cout << ' ' << v;
    std::cout << std::endl;
    return true;
  }
  void stop(bool owns) {
    if (moving && owns) {
      Cmd off{};
      off.mode_machine(5);
      off.mode_pr(0);
      off.crc(crc(off));
      for (int n = 0; n < 3; ++n) {
        if (!writer->Write(off, 0))
          std::cerr << "Disable packet delivery unconfirmed. Use physical stop.\n";
        std::this_thread::sleep_for(2ms);
      }
    }
    moving = false;
    writer.reset();
  }
  bool update(Time now, const Sample& s) {
    const float t = std::chrono::duration<float>(now - start).count();
    if (now - last_write > 50ms) return false;
    last_write = now;
    const float u = std::clamp(t / 4.f, 0.f, 1.f);
    const float blend = u * u * (3.f - 2.f * u);
    Cmd cmd{};
    cmd.mode_machine(5);
    cmd.mode_pr(0);
    bool settled = true;
    for (size_t i = 0; i < joint.size(); ++i) {
      float q = initial[i] + blend * (target[i] - initial[i]);
      if (std::abs(q - s.q[i]) > 0.20f) return false;
      settled &= std::abs(target[i] - s.q[i]) < 0.07f && std::abs(s.dq[i]) < 0.15f;
      auto& m = cmd.motor_cmd()[joint[i]];
      m.mode(1);
      m.q(q);
      m.kp(40.f * std::min(1.f, t / 0.5f));
      m.kd(1.f);  // SDK arm gains, zero feedforward
    }
    if (t >= 4.f && settled) {
      if (settled_since == Time{}) settled_since = now;
      if (!completed && now - settled_since >= 250ms) {
        completed = true;
        std::cout << "Step 1 measured at target. Holding; x stops, r requests handback."
                  << std::endl;
      }
    } else
      settled_since = {};
    if (!completed && t > 6.f) return false;
    cmd.crc(crc(cmd));
    return writer->Write(cmd, 0);
  }
};

enum class Phase { idle, confirm_develop, switching, ready, released, confirm_restore, restoring };
int run(const char* iface) {
  Terminal terminal;
  auto factory = ChannelFactory::Instance();
  factory->Init(0, iface);
  Feedback feedback;
  auto states = factory->CreateRecvChannel<State>(
      "rt/lowstate", [&](const void* p) { feedback.receive(p); }, 0);
  auto commands = factory->CreateRecvChannel<Cmd>("rt/lowcmd", [](const void*) {}, 0);
  auto command_reader = commands->GetReader()->GetNative();
  auto state_reader = states->GetReader()->GetNative();
  auto writers = [&] { return command_reader.subscription_matched_status().current_count(); };
  auto oneRobot = [&] { return state_reader.subscription_matched_status().current_count() == 1; };
  Motion mode;
  Arms arms;
  Phase phase = Phase::idle;
  Time deadline{};
  std::string previous, candidate;
  bool attempted = false;
  int result = 0;
  std::cout << "Mode 5 suspended arm test. Keep the hoist loaded and the physical stop in reach.\n"
               "d then y: confirm suspension and enter Develop. 1: open arms once.\n"
               "x / Space / Esc / Ctrl+C: stop and exit. r then y: release arms and restore prior "
               "service.\n";
  auto owns = [&] {
    return mode.fresh(Clock::now()) && mode.name.empty() && feedback.get().compatible &&
           arms.soleWriter(command_reader) && oneRobot();
  };
  try {
    for (;;) {
      const char key = terminal.key();
      if (key == 'x') break;
      const auto now = Clock::now();
      mode.update(now);
      const Sample s = feedback.get();
      const bool healthy = s.fresh(now) && oneRobot();
      if (phase == Phase::ready &&
          (!healthy || !mode.fresh(now) || !mode.name.empty() || (arms.active() && !owns()))) {
        result = 1;
        std::cerr << "Feedback or control ownership lost. Stopping.\n";
        break;
      }
      if ((phase == Phase::confirm_develop || phase == Phase::confirm_restore) && now > deadline) {
        phase = phase == Phase::confirm_restore ? Phase::released : Phase::idle;
        std::cout << "Confirmation expired. Repeat the requested key to try again." << std::endl;
      }
      if (phase == Phase::switching || phase == Phase::restoring) {
        if (mode.fresh(now) && !mode.busy() &&
            mode.name == (phase == Phase::switching ? "" : previous)) {
          if (phase == Phase::restoring) {
            std::cout << "Prior service restored.\n";
            break;
          }
          if (healthy && writers() == 0) {
            arms.connect();
            phase = Phase::ready;
            std::cout << "Develop verified. Arms idle. Press 1 when clear." << std::endl;
          }
        }
        if (phase != Phase::ready && now > deadline) {
          result = 1;
          std::cerr << "Switch unconfirmed; check remote and service state.\n";
          break;
        }
      } else if (key == 'd' && phase == Phase::idle && !attempted) {
        if (healthy && mode.fresh(now)) {
          candidate = mode.name;
          phase = Phase::confirm_develop;
          deadline = now + 10s;
          std::cout << "Robot fully suspended, zero torque, arms clear, operator at physical stop? "
                       "y confirms.\n";
        } else
          std::cout << "Need one fresh Mode 5 PR robot and a successful CheckMode.\n";
      } else if (key == 'y' && phase == Phase::confirm_develop) {
        if (healthy && mode.fresh(now) && mode.name == candidate && !mode.busy()) {
          previous = candidate;
          deadline = now + 3s;
          phase = Phase::switching;
          if (!previous.empty())
            mode.request(Operation::release);
          else
            mode.request(Operation::check);
        } else
          std::cout << "Confirmation unavailable; check feedback and press y again.\n";
      } else if (key == '1' && phase == Phase::ready && !attempted) {
        if (healthy && owns())
          attempted = arms.startMove(s);
        else
          std::cout << "Step 1 blocked: check fresh mode, state and other command publishers.\n";
      } else if (key == 'r' && (phase == Phase::ready || phase == Phase::released)) {
        arms.stop(owns());
        if (previous.empty()) {
          std::cout << "No previous service recorded. Use the physical remote.\n";
          break;
        }
        phase = Phase::confirm_restore;
        deadline = now + 10s;
        std::cout << "Arm command stream closed. Restore service '" << previous
                  << "' while suspended? y confirms.\n";
      } else if (key == 'y' && phase == Phase::confirm_restore) {
        if (healthy && mode.fresh(now) && mode.name.empty() && !mode.busy() && writers() == 0) {
          mode.request(Operation::restore, previous);
          phase = Phase::restoring;
          deadline = now + 3s;
        } else
          std::cout << "Restore blocked: check robot, service and lowcmd writers.\n";
      }
      if (arms.active() && !arms.update(Clock::now(), s)) {
        result = 1;
        std::cerr << "Command write, timing or arm tracking fault. Stopping.\n";
        break;
      }
      std::this_thread::sleep_for(5ms);
    }
    arms.stop(owns());
  } catch (...) {
    // Never compete with an unknown writer after a DDS failure.
    arms.stop(false);
    throw;
  }
  std::cout << "Test stopped. Verify mode on the remote. A previously sent switch request may "
               "still finish.\n"
               "Keep the hoist loaded and use the physical stop if needed.\n";
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: g1_step1 INTERFACE\n";
    return 2;
  }
  for (int sig : {SIGINT, SIGTERM, SIGHUP, SIGTSTP}) std::signal(sig, interrupt);
  try {
    return run(argv[1]);
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
