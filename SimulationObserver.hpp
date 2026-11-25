#pragma once

#include <mujoco/mujoco.h>
#include <sys/un.h>

namespace labrob {
class WalkingManager;
class RobotState;

// A best-effort, local datagram stream. The receiver may start or stop at any time.
class SimulationObserver {
 public:
  explicit SimulationObserver(bool hardware);
  ~SimulationObserver();
  SimulationObserver(const SimulationObserver&) = delete;
  SimulationObserver& operator=(const SimulationObserver&) = delete;
  void sampleStartup(const mjModel&, const mjData&, const RobotState&);
  void sample(double time, const WalkingManager&, const RobotState&, const mjModel&, const mjData&);

  void sampleController(double time, const WalkingManager&, const RobotState& reference);

 private:
  bool receiverAvailable() const;
  bool sampleDue(double time);
  int socket_;
  sockaddr_un receiver_{};
  unsigned int address_length_;
  double next_sample_ = 0;
};
}
