#include <hrp4_locomotion/DiscreteLIPDynamics.hpp>
#include <hrp4_locomotion/ISMPC.hpp>
#include <hrp4_locomotion/LIPSimulator.hpp>
#include <hrp4_locomotion/LIPState.hpp>
#include <hrp4_locomotion/WalkingData.hpp>

#include <iostream>

namespace labrob {

LIPSimulator::LIPSimulator(const LIPSimulatorOptions& options)
  : options_(options) {
}

void
LIPSimulator::run() {
  auto dynamics = labrob::DiscreteLIPDynamics(options_.eta, options_.timestep_msec);
  auto walking_data = labrob::WalkingData();
  walking_data.footstep_plan = options_.footstep_plan;
  auto lip_state = options_.initial_lip_state;
  auto ismpc = labrob::ISMPC(
      options_.mpc_prediction_horizon_msec,
      options_.mpc_timestep_msec,
      std::sqrt(9.81 / options_.mpc_com_target_height),
      options_.mpc_foot_constraint_square_width
  );
  for (int k = 0; k < options_.duration_msec / options_.timestep_msec; ++k) {
    int64_t t_msec = k * options_.timestep_msec;
    walking_data.updateWalkingState(t_msec);
    ismpc.solve(t_msec, walking_data, lip_state);
    lip_state = dynamics.integrate(lip_state, ismpc.getInput());
    std::cerr << t_msec << ": " << lip_state.com_pos_ << std::endl;
  }
}

} // end namespace labrob
