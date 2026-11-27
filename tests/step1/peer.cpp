// Loopback-only integration fixture, run inside docker --network none.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/server/server.hpp>

#include "G1Body.hpp"

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;
using namespace std::chrono_literals;

class Switcher : public Server {
  std::string mode = "ai";

 public:
  Switcher() : Server("motion_switcher") {}
  void Init() override {
    SetApiVersion("1.0.0.1");
    RegistHandler(1001, [this](const std::string&, std::string& r) {
      r = "{\"form\":\"0\",\"name\":\"" + mode + "\"}";
      return 0;
    });
    RegistHandler(1003, [this](const std::string&, std::string&) {
      std::cout << "RELEASE" << std::endl;
      if (std::filesystem::exists("/tmp/step1-delay")) std::this_thread::sleep_for(300ms);
      mode.clear();
      return 0;
    });
    RegistHandler(1002, [this](const std::string& p, std::string&) {
      if (p.find("\"ai\"") == std::string::npos) return 1;
      std::cout << "RESTORE" << std::endl;
      mode = "ai";
      return 0;
    });
  }
};

template <class T>
uint32_t crc(const T& msg) {
  std::array<uint32_t, sizeof(T) / 4> words;
  std::memcpy(words.data(), &msg, sizeof(msg));
  return Crc32Core(words.data(), words.size() - 1);
}

int main() {
  for (const auto& p : std::filesystem::directory_iterator("/sys/class/net"))
    if (p.path().filename() != "lo") return 2;
  ChannelFactory::Instance()->Init(0, "lo");
  Switcher server;
  server.Init();
  server.Start();
  LowState_ state{};
  state.mode_machine(5);
  for (int i = 0; i < 29; ++i) state.motor_state()[i].q(0.01f * i);
  std::mutex mutex;
  bool enabled = false;
  ChannelSubscriber<LowCmd_> sub("rt/lowcmd");
  sub.InitChannel(
      [&](const void* data) {
        const auto& c = *static_cast<const LowCmd_*>(data);
        std::lock_guard<std::mutex> guard(mutex);
        if (c.crc() != crc(c) || c.mode_pr() != 0 || c.mode_machine() != 5) std::abort();
        bool on = c.motor_cmd()[15].mode() == 1;
        for (int i = 0; i < 35; ++i) {
          const auto& m = c.motor_cmd()[i];
          bool arm = on && ((i >= 15 && i <= 18) || (i >= 22 && i <= 25));
          if (m.mode() != arm || m.tau() != 0 || m.dq() != 0) std::abort();
          if (!arm && (m.q() != 0 || m.kp() != 0 || m.kd() != 0)) std::abort();
          if (arm) {
            if (!std::isfinite(m.q()) || std::abs(m.q() - state.motor_state()[i].q()) > 0.025 ||
                m.kp() < 0 || m.kp() > 40 || m.kd() != 1)
              std::abort();
            state.motor_state()[i].q(m.q());
          }
        }
        if (!on && enabled) {
          std::cout << "DELTA";
          for (int i : {15, 16, 17, 18, 22, 23, 24, 25})
            std::cout << ' ' << state.motor_state()[i].q() - 0.01f * i;
          std::cout << std::endl;
        }
        if (on != enabled) std::cout << (on ? "ON" : "OFF") << std::endl;
        enabled = on;
      },
      0);
  ChannelPublisher<LowState_> pub("rt/lowstate");
  pub.InitChannel();
  ChannelPublisherPtr<LowCmd_> competitor;
  for (;;) {
    if (!competitor && std::filesystem::exists("/tmp/step1-compete")) {
      competitor = std::make_shared<ChannelPublisher<LowCmd_>>("rt/lowcmd");
      competitor->InitChannel();
    }
    {
      std::lock_guard<std::mutex> guard(mutex);
      state.mode_machine(std::filesystem::exists("/tmp/step1-wrong-mode") ? 13 : 5);
      state.tick()++;
      state.crc(crc(state));
      if (!std::filesystem::exists("/tmp/step1-drop")) pub.Write(state);
    }
    std::this_thread::sleep_for(4ms);
  }
}
