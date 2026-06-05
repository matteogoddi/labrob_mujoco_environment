#include <hrp4_locomotion/ComplianceReferenceGenerator.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Generator = labrob::ComplianceReferenceGenerator;
using Vector6d = Generator::Vector6d;
using Matrix6d = Generator::Matrix6d;

double objective(const Vector6d& x, const Matrix6d& H, const Vector6d& g) {
  return 0.5 * x.dot(H * x) + g.dot(x);
}

bool feasible(const Vector6d& x, const Vector6d& lbx, const Vector6d& ubx) {
  for (int i = 0; i < 6; ++i) {
    if (x(i) < lbx(i) - 1e-9 || x(i) > ubx(i) + 1e-9) {
      return false;
    }
  }
  return true;
}

Vector6d solveBoxQpByActiveSet(
    const Matrix6d& H,
    const Vector6d& g,
    const Vector6d& lbx,
    const Vector6d& ubx) {
  Vector6d best = Vector6d::Zero();
  double best_obj = std::numeric_limits<double>::infinity();

  const int combinations = 729;  // 3^6: free, lower, upper for each variable.
  for (int code = 0; code < combinations; ++code) {
    int tmp = code;
    std::vector<int> free_indices;
    Vector6d x = Vector6d::Zero();

    for (int i = 0; i < 6; ++i) {
      const int state = tmp % 3;
      tmp /= 3;

      if (state == 0) {
        free_indices.push_back(i);
      } else if (state == 1) {
        x(i) = lbx(i);
      } else {
        x(i) = ubx(i);
      }
    }

    const int nf = static_cast<int>(free_indices.size());
    if (nf > 0) {
      Eigen::MatrixXd Hff(nf, nf);
      Eigen::VectorXd rhs(nf);

      for (int r = 0; r < nf; ++r) {
        const int ir = free_indices[r];
        rhs(r) = -g(ir);

        for (int j = 0; j < 6; ++j) {
          bool is_free = false;
          for (const int free_index : free_indices) {
            if (j == free_index) {
              is_free = true;
              break;
            }
          }

          if (!is_free) {
            rhs(r) -= H(ir, j) * x(j);
          }
        }

        for (int c = 0; c < nf; ++c) {
          Hff(r, c) = H(ir, free_indices[c]);
        }
      }

      const Eigen::VectorXd xf = Hff.ldlt().solve(rhs);
      if (!xf.allFinite()) {
        continue;
      }

      for (int r = 0; r < nf; ++r) {
        x(free_indices[r]) = xf(r);
      }
    }

    if (!feasible(x, lbx, ubx)) {
      continue;
    }

    const double obj = objective(x, H, g);
    if (obj < best_obj) {
      best_obj = obj;
      best = x;
    }
  }

  return best;
}

Generator::Output runCase(
    const std::string& name,
    const Generator::Parameters& params,
    const Vector6d& delta_xc_left,
    const Vector6d& delta_xc_right,
    const Matrix6d& Jb_left,
    const Matrix6d& Jb_right,
    bool& passed) {
  Generator generator(params);

  Generator::Input input;
  input.dt = 0.001;
  input.use_manual_delta_xc = true;
  input.manual_delta_xc_left = delta_xc_left;
  input.manual_delta_xc_right = delta_xc_right;
  input.Jb_left = Jb_left;
  input.Jb_right = Jb_right;

  const Generator::Output output = generator.update(input);
  const Generator::DebugInfo& debug = generator.getDebugInfo();
  const Vector6d expected =
      solveBoxQpByActiveSet(debug.H, debug.g, debug.lbx, debug.ubx);

  const double solution_error = (output.delta_xb - expected).lpNorm<Eigen::Infinity>();
  const double objective_error =
      std::abs(objective(output.delta_xb, debug.H, debug.g) -
               objective(expected, debug.H, debug.g));

  const bool ok =
      output.qp_solved &&
      solution_error < 1e-7 &&
      objective_error < 1e-8 &&
      feasible(output.delta_xb, debug.lbx, debug.ubx);

  passed = passed && ok;

  std::cout << std::left << std::setw(26) << name
            << " qp_solved=" << output.qp_solved
            << " solution_inf_error=" << std::scientific << solution_error
            << " objective_error=" << objective_error
            << " status=" << debug.qp_status << '\n';

  if (!ok) {
    std::cout << "  qpOASES delta_xb: " << output.delta_xb.transpose() << '\n'
              << "  active-set ref : " << expected.transpose() << '\n'
              << "  lbx            : " << debug.lbx.transpose() << '\n'
              << "  ubx            : " << debug.ubx.transpose() << '\n';
  }

  return output;
}

}  // namespace

int main() {
  Generator::Parameters params;
  params.compliance_mode = Generator::ComplianceMode::TORSO_ONLY;
  params.filter_alpha = 0.0;
  params.bound_relaxation = 0.0;
  params.Kb = 0.2 * Matrix6d::Identity();
  params.Ka_left = Matrix6d::Identity();
  params.Ka_right = 2.0 * Matrix6d::Identity();
  params.S_left = Matrix6d::Identity();
  params.S_right = Matrix6d::Identity();
  params.W_smooth = Matrix6d::Zero();
  params.W_reg = 1e-8 * Matrix6d::Identity();
  params.delta_xb_min = Vector6d::Constant(-10.0);
  params.delta_xb_max = Vector6d::Constant(10.0);

  bool passed = true;

  Vector6d left = Vector6d::Zero();
  Vector6d right = Vector6d::Zero();
  left << 0.04, -0.02, 0.03, 0.01, -0.015, 0.02;
  right << -0.02, 0.05, 0.01, -0.02, 0.01, -0.03;
  runCase(
      "unconstrained identity J",
      params,
      left,
      right,
      Matrix6d::Identity(),
      Matrix6d::Identity(),
      passed);

  Generator::Parameters bounded = params;
  bounded.delta_xb_min << -0.015, -0.02, -0.01, -0.01, -0.008, -0.02;
  bounded.delta_xb_max << 0.015, 0.025, 0.02, 0.012, 0.008, 0.018;
  runCase(
      "active bounds",
      bounded,
      left,
      right,
      Matrix6d::Identity(),
      Matrix6d::Identity(),
      passed);

  Matrix6d J_left = Matrix6d::Identity();
  Matrix6d J_right = Matrix6d::Identity();
  J_left(0, 3) = 0.12;
  J_left(1, 5) = -0.08;
  J_right(2, 4) = 0.10;
  J_right(5, 0) = -0.06;

  Generator::Parameters coupled = bounded;
  coupled.Kb.diagonal() << 0.1, 0.3, 0.2, 0.4, 0.5, 0.25;
  runCase(
      "coupled J and bounds",
      coupled,
      left,
      right,
      J_left,
      J_right,
      passed);

  if (!passed) {
    std::cerr << "Torso compliance qpOASES validation failed.\n";
    return 1;
  }

  std::cout << "Torso compliance qpOASES validation passed.\n";
  return 0;
}
