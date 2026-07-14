#include <hrp4_locomotion/ComplianceReferenceGenerator.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Generator = labrob::ComplianceReferenceGenerator;
using Vector3d = Generator::Vector3d;
using Vector6d = Generator::Vector6d;
using Matrix3d = Generator::Matrix3d;
using Matrix6d = Generator::Matrix6d;
using Matrix6x3d = Generator::Matrix6x3d;

constexpr double kNumericalRegularization = 1e-10;

struct IndependentQp {
  Matrix3d H = Matrix3d::Zero();
  Vector3d g = Vector3d::Zero();
  Vector3d lbx = Vector3d::Zero();
  Vector3d ubx = Vector3d::Zero();
};

struct CaseResult {
  Generator::Output output;
  Vector3d expected = Vector3d::Zero();
  double solution_error = 0.0;
  double decomposition_error = 0.0;
  double selected_residual_norm = 0.0;
  bool passed = false;
};

double objective(const Vector3d& x, const Matrix3d& H, const Vector3d& g) {
  return 0.5 * x.dot(H * x) + g.dot(x);
}

bool feasible(const Vector3d& x, const Vector3d& lbx, const Vector3d& ubx) {
  for (int i = 0; i < 3; ++i) {
    if (x(i) < lbx(i) - 1e-9 || x(i) > ubx(i) + 1e-9) {
      return false;
    }
  }
  return true;
}

Vector3d solveBoxQpByActiveSet(
    const Matrix3d& H,
    const Vector3d& g,
    const Vector3d& lbx,
    const Vector3d& ubx) {
  Vector3d best = Vector3d::Zero();
  double best_objective = std::numeric_limits<double>::infinity();

  const int combinations = 27;  // 3^3: free, lower, upper per variable.
  for (int code = 0; code < combinations; ++code) {
    int remaining = code;
    std::vector<int> free_indices;
    Vector3d x = Vector3d::Zero();

    for (int i = 0; i < 3; ++i) {
      const int state = remaining % 3;
      remaining /= 3;
      if (state == 0) {
        free_indices.push_back(i);
      } else if (state == 1) {
        x(i) = lbx(i);
      } else {
        x(i) = ubx(i);
      }
    }

    const int number_free = static_cast<int>(free_indices.size());
    if (number_free > 0) {
      Eigen::MatrixXd H_free(number_free, number_free);
      Eigen::VectorXd rhs(number_free);

      for (int row = 0; row < number_free; ++row) {
        const int original_row = free_indices[row];
        rhs(row) = -g(original_row);

        for (int column = 0; column < 3; ++column) {
          const bool is_free = std::find(
              free_indices.begin(), free_indices.end(), column) !=
              free_indices.end();
          if (!is_free) {
            rhs(row) -= H(original_row, column) * x(column);
          }
        }

        for (int column = 0; column < number_free; ++column) {
          H_free(row, column) =
              H(original_row, free_indices[column]);
        }
      }

      const Eigen::VectorXd solution = H_free.ldlt().solve(rhs);
      if (!solution.allFinite()) {
        continue;
      }
      for (int row = 0; row < number_free; ++row) {
        x(free_indices[row]) = solution(row);
      }
    }

    if (!feasible(x, lbx, ubx)) {
      continue;
    }

    const double candidate_objective = objective(x, H, g);
    if (candidate_objective < best_objective) {
      best_objective = candidate_objective;
      best = x;
    }
  }

  return best;
}

IndependentQp buildIndependentQp(
    const Generator::Parameters& params,
    const Vector6d& delta_xc_left,
    const Vector6d& delta_xc_right,
    const Matrix6x3d& Ab_left,
    const Matrix6x3d& Ab_right,
    const Vector3d& previous_qp_solution = Vector3d::Zero()) {
  IndependentQp qp;

  const Matrix6x3d E_left = params.S_allocation_left * Ab_left;
  const Matrix6x3d E_right = params.S_allocation_right * Ab_right;
  const Vector6d y_left = params.S_allocation_left * delta_xc_left;
  const Vector6d y_right = params.S_allocation_right * delta_xc_right;
  const Matrix6d W_left = 0.5 * (params.W_left + params.W_left.transpose());
  const Matrix6d W_right = 0.5 * (params.W_right + params.W_right.transpose());
  const Matrix3d Kb = 0.5 * (
      params.Kb.bottomRightCorner<3, 3>() +
      params.Kb.bottomRightCorner<3, 3>().transpose());
  const Matrix3d W_reg = 0.5 * (
      params.W_reg.bottomRightCorner<3, 3>() +
      params.W_reg.bottomRightCorner<3, 3>().transpose());
  const Matrix3d W_smooth = 0.5 * (
      params.W_smooth.bottomRightCorner<3, 3>() +
      params.W_smooth.bottomRightCorner<3, 3>().transpose());

  qp.H = Kb + E_left.transpose() * W_left * E_left +
      E_right.transpose() * W_right * E_right + W_smooth + W_reg +
      kNumericalRegularization * Matrix3d::Identity();
  qp.H = 0.5 * (qp.H + qp.H.transpose());
  qp.g = -params.rho_left * E_left.transpose() * W_left * y_left
      -params.rho_right * E_right.transpose() * W_right * y_right
      -W_smooth * previous_qp_solution;

  qp.lbx = params.delta_xb_min.tail<3>();
  qp.ubx = params.delta_xb_max.tail<3>();
  const double relaxation = std::isfinite(params.bound_relaxation)
      ? std::max(params.bound_relaxation, 0.0)
      : 0.0;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(qp.lbx(i))) {
      qp.lbx(i) = -1e9;
    }
    if (!std::isfinite(qp.ubx(i))) {
      qp.ubx(i) = 1e9;
    }
    if (qp.lbx(i) > qp.ubx(i)) {
      const double midpoint = 0.5 * (qp.lbx(i) + qp.ubx(i));
      qp.lbx(i) = midpoint;
      qp.ubx(i) = midpoint;
    }
    qp.lbx(i) -= relaxation;
    qp.ubx(i) += relaxation;
  }

  return qp;
}

double selectedResidualNorm(
    const Generator::Parameters& params,
    const Generator::Output& output) {
  const Vector6d left =
      params.S_allocation_left * output.delta_x_left_arm;
  const Vector6d right =
      params.S_allocation_right * output.delta_x_right_arm;
  return std::sqrt(left.squaredNorm() + right.squaredNorm());
}

CaseResult runCase(
    const std::string& name,
    const Generator::Parameters& params,
    const Vector6d& delta_xc_left,
    const Vector6d& delta_xc_right,
    const Matrix6x3d& Ab_left,
    const Matrix6x3d& Ab_right) {
  Generator generator(params);
  generator.reset();

  Generator::Input input;
  input.dt = 0.001;
  input.use_manual_delta_xc = true;
  input.manual_delta_xc_left = delta_xc_left;
  input.manual_delta_xc_right = delta_xc_right;
  input.use_explicit_Ab = true;
  input.Ab_left = Ab_left;
  input.Ab_right = Ab_right;

  CaseResult result;
  result.output = generator.update(input);
  const IndependentQp qp = buildIndependentQp(
      params, delta_xc_left, delta_xc_right, Ab_left, Ab_right);
  result.expected = solveBoxQpByActiveSet(qp.H, qp.g, qp.lbx, qp.ubx);

  const Vector3d actual = result.output.delta_xb.tail<3>();
  result.solution_error =
      (actual - result.expected).lpNorm<Eigen::Infinity>();
  const double left_decomposition_error =
      (result.output.delta_xc_left_filtered -
       result.output.delta_x_left_torso -
       result.output.delta_x_left_arm).lpNorm<Eigen::Infinity>();
  const double right_decomposition_error =
      (result.output.delta_xc_right_filtered -
       result.output.delta_x_right_torso -
       result.output.delta_x_right_arm).lpNorm<Eigen::Infinity>();
  result.decomposition_error =
      std::max(left_decomposition_error, right_decomposition_error);
  result.selected_residual_norm = selectedResidualNorm(params, result.output);

  const double reference_error = std::max(
      (result.output.x_left_ref_local -
       result.output.delta_x_left_arm).lpNorm<Eigen::Infinity>(),
      (result.output.x_right_ref_local -
       result.output.delta_x_right_arm).lpNorm<Eigen::Infinity>());
  const double first_derivative_norm = std::max({
      result.output.delta_dxb.lpNorm<Eigen::Infinity>(),
      result.output.delta_ddxb.lpNorm<Eigen::Infinity>(),
      result.output.delta_dx_left_arm.lpNorm<Eigen::Infinity>(),
      result.output.delta_dx_right_arm.lpNorm<Eigen::Infinity>(),
      result.output.delta_ddx_left_arm.lpNorm<Eigen::Infinity>(),
      result.output.delta_ddx_right_arm.lpNorm<Eigen::Infinity>()});

  result.passed = result.output.qp_solved && result.output.valid &&
      result.solution_error < 1e-7 &&
      result.decomposition_error < 1e-12 &&
      reference_error < 1e-12 &&
      first_derivative_norm < 1e-12 &&
      result.output.delta_xb.head<3>().norm() < 1e-14 &&
      feasible(actual, qp.lbx, qp.ubx);

  std::cout << std::left << std::setw(25) << name
            << " solved=" << result.output.qp_solved
            << " solution_err=" << std::scientific << result.solution_error
            << " decomposition_err=" << result.decomposition_error
            << " selected_residual=" << result.selected_residual_norm
            << '\n';
  if (!result.passed) {
    std::cout << "  actual xb   : " << actual.transpose() << '\n'
              << "  expected xb : " << result.expected.transpose() << '\n'
              << "  bounds      : [" << qp.lbx.transpose() << "] ["
              << qp.ubx.transpose() << "]\n";
  }

  return result;
}

Generator::Parameters endpointParameters(double rho) {
  Generator::Parameters params;
  params.compliance_mode = Generator::ComplianceMode::HAND_AND_TORSO;
  params.rho_left = rho;
  params.rho_right = rho;
  params.filter_alpha = 0.0;
  params.bound_relaxation = 0.0;
  params.S_allocation_left = Matrix6d::Identity();
  params.S_allocation_right = Matrix6d::Identity();
  params.W_left = Matrix6d::Identity();
  params.W_right = Matrix6d::Identity();
  params.Kb.setZero();
  params.Kb.bottomRightCorner<3, 3>() =
      1e-6 * Matrix3d::Identity();
  params.W_smooth.setZero();
  params.W_reg.setZero();
  params.W_reg.bottomRightCorner<3, 3>() =
      1e-10 * Matrix3d::Identity();
  params.delta_xb_min.tail<3>() = Vector3d::Constant(-0.2);
  params.delta_xb_max.tail<3>() = Vector3d::Constant(0.2);
  params.delta_xc_left_limit = Vector6d::Constant(1e9);
  params.delta_xc_right_limit = Vector6d::Constant(1e9);
  return params;
}

bool runEndpoint(double rho, bool verbose = true) {
  const Generator::Parameters params = endpointParameters(rho);
  const Matrix6x3d Ab_left = Generator::makeTorsoRotationMap(
      Eigen::Vector3d(0.10, 0.28, -0.12));
  const Matrix6x3d Ab_right = Generator::makeTorsoRotationMap(
      Eigen::Vector3d(0.10, -0.28, -0.12));
  Vector3d target;
  target << 0.04, -0.03, 0.05;
  const Vector6d delta_xc_left = Ab_left * target;
  const Vector6d delta_xc_right = Ab_right * target;

  const std::string label = rho == 0.0
      ? "rho=0 hand-only"
      : "rho=1 torso-dominant";
  const CaseResult result = runCase(
      label, params, delta_xc_left, delta_xc_right, Ab_left, Ab_right);

  const double torso_error = rho == 0.0
      ? result.output.delta_xb.tail<3>().lpNorm<Eigen::Infinity>()
      : (result.output.delta_xb.tail<3>() - target)
            .lpNorm<Eigen::Infinity>();
  const double arm_identity_error = std::max(
      (result.output.delta_x_left_arm - delta_xc_left)
          .lpNorm<Eigen::Infinity>(),
      (result.output.delta_x_right_arm - delta_xc_right)
          .lpNorm<Eigen::Infinity>());

  bool endpoint_passed = result.passed;
  if (rho == 0.0) {
    endpoint_passed = endpoint_passed && torso_error < 1e-10 &&
        arm_identity_error < 1e-10;
  } else {
    endpoint_passed = endpoint_passed && torso_error < 1e-6 &&
        result.output.delta_xb.tail<3>().norm() > 1e-3 &&
        result.selected_residual_norm < 1e-6;
  }

  if (verbose) {
    std::cout << "  torso_rpy       : "
              << result.output.delta_xb.tail<3>().transpose() << '\n'
              << "  torso_error_inf : " << torso_error << '\n'
              << "  arm_residual_L2 : "
              << result.selected_residual_norm << '\n'
              << "  interpretation  : "
              << (rho == 0.0
                      ? "torso=0 and the arms retain the full admittance offset"
                      : "torso reproduces the compatible target; general "
                        "rho=1 cases may retain arm residual")
              << '\n';
  }

  return endpoint_passed;
}

bool runHistoryAndLegacyRegressionCases() {
  const Matrix6x3d Ab_left = Generator::makeTorsoRotationMap(
      Eigen::Vector3d(0.13, 0.29, -0.10));
  const Matrix6x3d Ab_right = Generator::makeTorsoRotationMap(
      Eigen::Vector3d(0.11, -0.26, -0.12));

  Generator::Parameters legacy_params = endpointParameters(1.0);
  legacy_params.rho_left = 0.35;
  legacy_params.rho_right = 0.75;
  Vector6d legacy_left;
  Vector6d legacy_right;
  legacy_left << 0.02, -0.01, 0.03, 0.015, -0.02, 0.01;
  legacy_right << -0.01, 0.025, -0.02, -0.005, 0.01, 0.03;

  Generator explicit_generator(legacy_params);
  Generator::Input explicit_input;
  explicit_input.dt = 0.001;
  explicit_input.use_manual_delta_xc = true;
  explicit_input.manual_delta_xc_left = legacy_left;
  explicit_input.manual_delta_xc_right = legacy_right;
  explicit_input.use_explicit_Ab = true;
  explicit_input.Ab_left = Ab_left;
  explicit_input.Ab_right = Ab_right;
  const Generator::Output explicit_output =
      explicit_generator.update(explicit_input);

  Generator legacy_generator(legacy_params);
  Generator::Input legacy_input = explicit_input;
  legacy_input.use_explicit_Ab = false;
  legacy_input.Jb_left.leftCols<3>().setConstant(100.0);
  legacy_input.Jb_right.leftCols<3>().setConstant(-100.0);
  legacy_input.Jb_left.rightCols<3>() = Ab_left;
  legacy_input.Jb_right.rightCols<3>() = Ab_right;
  const Generator::Output legacy_output = legacy_generator.update(legacy_input);
  const double legacy_error = std::max({
      (legacy_output.delta_xb - explicit_output.delta_xb)
          .lpNorm<Eigen::Infinity>(),
      (legacy_output.delta_x_left_arm - explicit_output.delta_x_left_arm)
          .lpNorm<Eigen::Infinity>(),
      (legacy_output.delta_x_right_arm - explicit_output.delta_x_right_arm)
          .lpNorm<Eigen::Infinity>()});
  const bool legacy_ok = explicit_output.valid && legacy_output.valid &&
      legacy_error < 1e-10;
  std::cout << std::left << std::setw(25) << "legacy Jb fallback"
            << " equivalence_err=" << std::scientific << legacy_error << '\n';

  Generator::Parameters history_params = endpointParameters(1.0);
  history_params.filter_alpha = 0.5;
  history_params.Kb.bottomRightCorner<3, 3>() =
      0.2 * Matrix3d::Identity();
  history_params.W_smooth.bottomRightCorner<3, 3>() =
      0.4 * Matrix3d::Identity();
  Generator history_generator(history_params);
  Generator::Input history_input;
  history_input.dt = 0.01;
  history_input.use_manual_delta_xc = true;
  history_input.use_explicit_Ab = true;
  history_input.Ab_left = Ab_left;
  history_input.Ab_right = Ab_right;

  Vector3d target_first;
  Vector3d target_second;
  target_first << 0.05, -0.035, 0.04;
  target_second << -0.02, 0.03, -0.01;
  const Vector6d raw_left_first = Ab_left * target_first;
  const Vector6d raw_right_first = Ab_right * target_first;
  const Vector6d raw_left_second = Ab_left * target_second;
  const Vector6d raw_right_second = Ab_right * target_second;

  history_input.manual_delta_xc_left = raw_left_first;
  history_input.manual_delta_xc_right = raw_right_first;
  const Generator::Output first = history_generator.update(history_input);
  const Vector6d filtered_left_first = 0.5 * raw_left_first;
  const Vector6d filtered_right_first = 0.5 * raw_right_first;
  const IndependentQp first_qp = buildIndependentQp(
      history_params,
      filtered_left_first,
      filtered_right_first,
      Ab_left,
      Ab_right);
  const Vector3d expected_raw_first = solveBoxQpByActiveSet(
      first_qp.H, first_qp.g, first_qp.lbx, first_qp.ubx);
  const Vector3d expected_final_first = 0.5 * expected_raw_first;

  history_input.manual_delta_xc_left = raw_left_second;
  history_input.manual_delta_xc_right = raw_right_second;
  const Generator::Output second = history_generator.update(history_input);
  const Vector6d filtered_left_second =
      0.5 * filtered_left_first + 0.5 * raw_left_second;
  const Vector6d filtered_right_second =
      0.5 * filtered_right_first + 0.5 * raw_right_second;
  const IndependentQp second_qp = buildIndependentQp(
      history_params,
      filtered_left_second,
      filtered_right_second,
      Ab_left,
      Ab_right,
      expected_raw_first);
  const Vector3d expected_raw_second = solveBoxQpByActiveSet(
      second_qp.H, second_qp.g, second_qp.lbx, second_qp.ubx);
  const Vector3d expected_final_second =
      0.5 * expected_final_first + 0.5 * expected_raw_second;
  const Vector3d expected_velocity =
      (expected_final_second - expected_final_first) / history_input.dt;

  const double history_error = std::max({
      (first.delta_xb.tail<3>() - expected_raw_first)
          .lpNorm<Eigen::Infinity>(),
      (first.delta_xb_final.tail<3>() - expected_final_first)
          .lpNorm<Eigen::Infinity>(),
      (second.delta_xb.tail<3>() - expected_raw_second)
          .lpNorm<Eigen::Infinity>(),
      (second.delta_xb_final.tail<3>() - expected_final_second)
          .lpNorm<Eigen::Infinity>(),
      (second.delta_dxb.tail<3>() - expected_velocity)
          .lpNorm<Eigen::Infinity>()});
  const bool first_derivative_zero =
      first.delta_dxb.lpNorm<Eigen::Infinity>() < 1e-12 &&
      first.delta_ddxb.lpNorm<Eigen::Infinity>() < 1e-12;

  history_generator.reset();
  const Generator::Output after_reset = history_generator.update(history_input);
  const Vector6d filtered_left_reset = 0.5 * raw_left_second;
  const Vector6d filtered_right_reset = 0.5 * raw_right_second;
  const IndependentQp reset_qp = buildIndependentQp(
      history_params,
      filtered_left_reset,
      filtered_right_reset,
      Ab_left,
      Ab_right);
  const Vector3d expected_raw_reset = solveBoxQpByActiveSet(
      reset_qp.H, reset_qp.g, reset_qp.lbx, reset_qp.ubx);
  const double reset_error =
      (after_reset.delta_xb.tail<3>() - expected_raw_reset)
          .lpNorm<Eigen::Infinity>();
  const bool history_ok = first.valid && second.valid && after_reset.valid &&
      first_derivative_zero && history_error < 1e-7 && reset_error < 1e-7;
  std::cout << std::left << std::setw(25) << "raw-QP smooth history"
            << " recursion_err=" << history_error
            << " reset_err=" << reset_error << '\n';

  return legacy_ok && history_ok;
}

bool runSafetyRegressionCases() {
  const Matrix6x3d Ab_left = Generator::makeTorsoRotationMap(
      Eigen::Vector3d(0.10, 0.25, -0.10));
  const Matrix6x3d Ab_right = Generator::makeTorsoRotationMap(
      Eigen::Vector3d(0.10, -0.25, -0.10));
  Vector3d target;
  target << 0.15, 0.15, 0.15;

  Generator::Parameters bounded_params = endpointParameters(1.0);
  bounded_params.filter_alpha = 0.99;
  bounded_params.delta_xb_min.tail<3>() = Vector3d::Constant(0.10);
  bounded_params.delta_xb_max.tail<3>() = Vector3d::Constant(0.20);
  Generator bounded_generator(bounded_params);
  Generator::Input bounded_input;
  bounded_input.dt = 0.001;
  bounded_input.use_manual_delta_xc = true;
  bounded_input.manual_delta_xc_left = Ab_left * target;
  bounded_input.manual_delta_xc_right = Ab_right * target;
  bounded_input.use_explicit_Ab = true;
  bounded_input.Ab_left = Ab_left;
  bounded_input.Ab_right = Ab_right;
  const Generator::Output bounded_output =
      bounded_generator.update(bounded_input);
  const bool filtered_bounds_ok = bounded_output.valid &&
      (bounded_output.delta_xb_final.tail<3>().array() >= 0.10).all() &&
      (bounded_output.delta_xb_final.tail<3>().array() <= 0.20).all();
  std::cout << std::left << std::setw(25) << "filtered torso bounds"
            << " final=" << bounded_output.delta_xb_final.tail<3>().transpose()
            << '\n';

  Generator::Parameters invalid_params = endpointParameters(1.0);
  invalid_params.Kb(3, 3) = std::numeric_limits<double>::quiet_NaN();
  Generator invalid_generator(invalid_params);
  Generator::Input invalid_input = bounded_input;
  invalid_input.dt = 0.001;
  const Generator::Output invalid_output = invalid_generator.update(invalid_input);
  const bool invalid_qp_ok = !invalid_output.qp_solved &&
      !invalid_output.valid &&
      std::string(invalid_generator.getDebugInfo().qpStatus()).find(
          "non-finite") != std::string::npos;
  std::cout << std::left << std::setw(25) << "non-finite QP reject"
            << " solved=" << invalid_output.qp_solved
            << " status=" << invalid_generator.getDebugInfo().qpStatus()
            << '\n';

  Generator hold_generator(endpointParameters(1.0));
  Generator::Input hold_input = bounded_input;
  hold_input.dt = 0.001;
  const Generator::Output before_failure = hold_generator.update(hold_input);
  Generator::Parameters failing_params = endpointParameters(1.0);
  failing_params.Kb(3, 3) = std::numeric_limits<double>::quiet_NaN();
  hold_generator.setParameters(failing_params);
  const Generator::Output during_failure = hold_generator.update(hold_input);
  const double failure_hold_error =
      (during_failure.delta_xb_final - before_failure.delta_xb_final)
          .lpNorm<Eigen::Infinity>();
  const bool failure_hold_ok = before_failure.valid &&
      !during_failure.valid && !during_failure.qp_solved &&
      failure_hold_error < 1e-12 &&
      during_failure.delta_dxb.lpNorm<Eigen::Infinity>() < 1e-12;
  std::cout << std::left << std::setw(25) << "QP failure hold"
            << " hold_err=" << failure_hold_error << '\n';

  Generator::Parameters hand_mode = endpointParameters(0.0);
  hand_mode.compliance_mode = Generator::ComplianceMode::HAND_ONLY;
  hold_generator.setParameters(hand_mode);
  const Generator::Output hand_only = hold_generator.update(hold_input);
  hold_generator.setParameters(endpointParameters(1.0));
  const Generator::Output torso_reenabled = hold_generator.update(hold_input);
  const bool mode_derivative_ok = hand_only.valid && torso_reenabled.valid &&
      hand_only.delta_dxb.lpNorm<Eigen::Infinity>() < 1e-12 &&
      torso_reenabled.delta_dxb.lpNorm<Eigen::Infinity>() < 1e-12;
  std::cout << std::left << std::setw(25) << "mode derivative reset"
            << " hand_dxb=" << hand_only.delta_dxb.norm()
            << " reenable_dxb=" << torso_reenabled.delta_dxb.norm() << '\n';

  Generator::Parameters dt_params = endpointParameters(1.0);
  Generator dt_generator(dt_params);
  Generator::Input dt_input = bounded_input;
  dt_input.dt = std::numeric_limits<double>::quiet_NaN();
  const Generator::Output invalid_dt = dt_generator.update(dt_input);
  dt_input.dt = 0.001;
  const Generator::Output recovered = dt_generator.update(dt_input);
  Generator fresh_generator(dt_params);
  const Generator::Output fresh = fresh_generator.update(dt_input);
  const double recovery_error = std::max(
      (recovered.delta_xb - fresh.delta_xb).lpNorm<Eigen::Infinity>(),
      (recovered.delta_x_left_arm - fresh.delta_x_left_arm)
          .lpNorm<Eigen::Infinity>());
  const bool invalid_dt_ok = !invalid_dt.valid && recovered.valid &&
      fresh.valid && recovery_error < 1e-10;
  std::cout << std::left << std::setw(25) << "invalid dt isolation"
            << " recovery_err=" << recovery_error << '\n';

  return filtered_bounds_ok && invalid_qp_ok && failure_hold_ok &&
      mode_derivative_ok && invalid_dt_ok;
}

bool runGeneralRegressionCases() {
  Generator::Parameters params = endpointParameters(1.0);
  params.rho_left = 0.25;
  params.rho_right = 0.80;
  params.Kb.bottomRightCorner<3, 3>().diagonal() << 0.2, 0.4, 0.3;
  params.W_left.diagonal() << 2.0, 1.0, 3.0, 0.5, 1.5, 2.5;
  params.W_right.diagonal() << 1.0, 3.0, 2.0, 2.0, 0.75, 1.25;
  params.S_allocation_left.setZero();
  params.S_allocation_right.setZero();
  params.S_allocation_left.diagonal() << 1.0, 0.0, 1.0, 1.0, 0.0, 1.0;
  params.S_allocation_right.diagonal() << 0.0, 1.0, 1.0, 0.0, 1.0, 1.0;
  params.delta_xb_min.tail<3>() << -0.012, -0.020, -0.010;
  params.delta_xb_max.tail<3>() << 0.015, 0.018, 0.012;

  const Matrix6x3d Ab_left = Generator::makeTorsoRotationMap(
      Eigen::Vector3d(0.15, 0.30, -0.10));
  const Matrix6x3d Ab_right = Generator::makeTorsoRotationMap(
      Eigen::Vector3d(0.12, -0.27, -0.11));
  Vector6d left;
  Vector6d right;
  left << 0.04, -0.02, 0.03, 0.01, -0.015, 0.02;
  right << -0.02, 0.05, 0.01, -0.02, 0.01, -0.03;

  const CaseResult bounded = runCase(
      "mixed-rho active bounds", params, left, right, Ab_left, Ab_right);

  Generator::Parameters clamped = params;
  clamped.rho_left = -0.4;
  clamped.rho_right = 1.7;
  Generator clamped_generator(clamped);
  const bool clamp_ok =
      clamped_generator.getParameters().rho_left == 0.0 &&
      clamped_generator.getParameters().rho_right == 1.0;
  std::cout << std::left << std::setw(25) << "rho clamping"
            << " left=" << clamped_generator.getParameters().rho_left
            << " right=" << clamped_generator.getParameters().rho_right
            << '\n';

  // Regression for the former TORSO_ONLY bug: it must still run admittance
  // so rho=1 receives a non-zero hand displacement without manual injection.
  Generator::Parameters torso_mode = endpointParameters(1.0);
  torso_mode.compliance_mode = Generator::ComplianceMode::TORSO_ONLY;
  torso_mode.use_admittance_dynamics = false;
  torso_mode.Ka_left = Matrix6d::Identity();
  torso_mode.Ka_right = Matrix6d::Identity();
  Generator torso_generator(torso_mode);
  Generator::Input torso_input;
  torso_input.dt = 0.001;
  torso_input.use_explicit_Ab = true;
  torso_input.Ab_left = Ab_left;
  torso_input.Ab_right = Ab_right;
  Vector3d target;
  target << 0.01, -0.015, 0.02;
  torso_input.wrench_left = Ab_left * target;
  torso_input.wrench_right = Ab_right * target;
  const Generator::Output torso_output = torso_generator.update(torso_input);
  const bool torso_mode_ok = torso_output.qp_solved &&
      torso_output.delta_xc_left.norm() > 1e-4 &&
      torso_output.delta_xb.tail<3>().norm() > 1e-4;
  std::cout << std::left << std::setw(25) << "TORSO_ONLY admittance"
            << " dxc_norm=" << torso_output.delta_xc_left.norm()
            << " xb_norm=" << torso_output.delta_xb.tail<3>().norm()
            << '\n';

  return bounded.passed && clamp_ok && torso_mode_ok;
}

void printUsage(const char* executable) {
  std::cout << "Usage:\n"
            << "  " << executable << "             # run full CRG regression\n"
            << "  " << executable << " --rho 0     # hand-only endpoint\n"
            << "  " << executable << " --rho 1     # torso-dominant endpoint\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    printUsage(argv[0]);
    return 0;
  }

  if (argc == 3 && std::string(argv[1]) == "--rho") {
    char* end = nullptr;
    const double rho = std::strtod(argv[2], &end);
    if (end == argv[2] || *end != '\0' || (rho != 0.0 && rho != 1.0)) {
      std::cerr << "--rho endpoint validation accepts exactly 0 or 1.\n";
      return 2;
    }

    const bool passed = runEndpoint(rho);
    std::cout << (passed ? "CRG endpoint validation passed.\n"
                         : "CRG endpoint validation failed.\n");
    return passed ? 0 : 1;
  }

  if (argc != 1) {
    printUsage(argv[0]);
    return 2;
  }

  bool passed = true;
  passed = runEndpoint(0.0) && passed;
  passed = runEndpoint(1.0) && passed;
  passed = runHistoryAndLegacyRegressionCases() && passed;
  passed = runSafetyRegressionCases() && passed;
  passed = runGeneralRegressionCases() && passed;

  std::cout << (passed ? "CRG rho/QP validation passed.\n"
                       : "CRG rho/QP validation failed.\n");
  return passed ? 0 : 1;
}
