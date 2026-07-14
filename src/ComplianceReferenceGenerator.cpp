#include <hrp4_locomotion/ComplianceReferenceGenerator.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <Eigen/Geometry>

namespace {

using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix6x3d = Eigen::Matrix<double, 6, 3>;

bool isFiniteVector(const Vector6d& x) {
  for (int i = 0; i < 6; ++i) {
    if (!std::isfinite(x(i))) {
      return false;
    }
  }
  return true;
}

Eigen::Matrix3d rpyToRotation(double roll, double pitch, double yaw) {
  const Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());

  return (Rz * Ry * Rx).toRotationMatrix();
}

Vector6d makePose6dFromRotation(
    const Eigen::Vector3d& p,
    const Eigen::Matrix3d& R) {
  Vector6d x = Vector6d::Zero();

  const Eigen::Vector3d ypr = R.eulerAngles(2, 1, 0);

  x.head<3>() = p;
  x(3) = ypr(2);  // roll
  x(4) = ypr(1);  // pitch
  x(5) = ypr(0);  // yaw

  return x;
}

double clampUnitInterval(double value) {
  if (!std::isfinite(value)) {
    return 0.0;
  }
  return std::min(std::max(value, 0.0), 1.0);
}

Matrix3d skewSymmetric(const Vector3d& v) {
  Matrix3d skew;
  skew << 0.0, -v.z(), v.y(),
          v.z(), 0.0, -v.x(),
          -v.y(), v.x(), 0.0;
  return skew;
}

double objectiveValue(
    const Vector3d& x,
    const Matrix3d& H,
    const Vector3d& g) {
  return 0.5 * x.dot(H * x) + g.dot(x);
}

template <std::size_t N>
void setDebugStatus(
    std::array<char, N>& destination,
    const std::string& status) {
  destination.fill('\0');
  const std::size_t count = std::min(status.size(), N - 1);
  std::copy_n(status.data(), count, destination.data());
}

}  // anonymous namespace


namespace labrob {

ComplianceReferenceGenerator::ComplianceReferenceGenerator()
    : params_(Parameters()) {
  buildSolver();
  reset();
}

ComplianceReferenceGenerator::ComplianceReferenceGenerator(
    const Parameters& params)
    : params_(params) {
  params_.rho_left = clampUnitInterval(params_.rho_left);
  params_.rho_right = clampUnitInterval(params_.rho_right);
  buildSolver();
  reset();
}

void ComplianceReferenceGenerator::reset() {
  debug_ = DebugInfo();

  delta_xc_left_prev_.setZero();
  delta_xc_right_prev_.setZero();
  delta_xb_qp_prev_.setZero();

  delta_dxc_left_prev_.setZero();
  delta_dxc_right_prev_.setZero();

  delta_xc_left_filtered_prev_.setZero();
  delta_xc_right_filtered_prev_.setZero();
  delta_xb_filtered_prev_.setZero();

  delta_dxb_prev_.setZero();
  delta_x_left_arm_prev_.setZero();
  delta_x_right_arm_prev_.setZero();
  delta_dx_left_arm_prev_.setZero();
  delta_dx_right_arm_prev_.setZero();

  torso_output_history_initialized_ = false;
  arm_output_history_initialized_ = false;
}

void ComplianceReferenceGenerator::setParameters(
    const Parameters& params) {
  params_ = params;
  params_.rho_left = clampUnitInterval(params_.rho_left);
  params_.rho_right = clampUnitInterval(params_.rho_right);
  buildSolver();
}

const ComplianceReferenceGenerator::Parameters&
ComplianceReferenceGenerator::getParameters() const {
  return params_;
}

const ComplianceReferenceGenerator::DebugInfo&
ComplianceReferenceGenerator::getDebugInfo() const {
  return debug_;
}

ComplianceReferenceGenerator::Matrix6x3d
ComplianceReferenceGenerator::makeTorsoRotationMap(
    const Eigen::Vector3d& r_b_hand) {
  Matrix6x3d Ab = Matrix6x3d::Zero();
  Ab.topRows<3>() = -skewSymmetric(r_b_hand);
  Ab.bottomRows<3>() = Matrix3d::Identity();
  return Ab;
}

void ComplianceReferenceGenerator::buildSolver() {
  const int nx = 3;
  const int na = 0;

  casadi::SpDict qp;
  qp["h"] = casadi::Sparsity::dense(nx, nx);
  qp["a"] = casadi::Sparsity::dense(na, nx);

  casadi::Dict opts;
  opts["print_time"] = false;
  opts["error_on_fail"] = false;

  // qpOASES plugin option.
  // If your CasADi version complains about this option, remove this line.
  opts["printLevel"] = "none";

  try {
    qp_solver_ = casadi::conic(
        "torso_compliance_qp",
        "qpoases",
        qp,
        opts);

    solver_built_ = true;
    setDebugStatus(
        debug_.qp_status,
        "CasADi qpOASES solver built successfully.");
  } catch (const std::exception& e) {
    solver_built_ = false;
    setDebugStatus(
        debug_.qp_status,
        std::string("Failed to build CasADi qpOASES solver: ") + e.what());
  }
}

ComplianceReferenceGenerator::Output
ComplianceReferenceGenerator::update(const Input& input) {
  Output output;
  debug_ = DebugInfo();

  const bool compliance_enabled =
      params_.compliance_mode != ComplianceMode::NONE;

  const bool torso_enabled =
      params_.compliance_mode == ComplianceMode::TORSO_ONLY ||
      params_.compliance_mode == ComplianceMode::HAND_AND_TORSO;

  const bool hand_reference_enabled =
      params_.compliance_mode == ComplianceMode::HAND_ONLY ||
      params_.compliance_mode == ComplianceMode::HAND_AND_TORSO;

  const bool input_dt_valid = std::isfinite(input.dt) && input.dt > 0.0;
  const double dt = input_dt_valid ? std::max(input.dt, 1e-6) : 1e-6;
  const Matrix6x3d Ab_left = input.use_explicit_Ab
      ? input.Ab_left
      : Matrix6x3d(input.Jb_left.rightCols<3>());
  const Matrix6x3d Ab_right = input.use_explicit_Ab
      ? input.Ab_right
      : Matrix6x3d(input.Jb_right.rightCols<3>());

  debug_.Ab_left = Ab_left;
  debug_.Ab_right = Ab_right;
  debug_.rho_left = params_.rho_left;
  debug_.rho_right = params_.rho_right;

  if (!input_dt_valid) {
    debug_.qp_solved = false;
    setDebugStatus(
        debug_.qp_status,
        "ComplianceReferenceGenerator input dt must be finite and positive.");
    output.x_left_ref_local = input.x_left_nominal_local;
    output.x_right_ref_local = input.x_right_nominal_local;
    output.x_torso_ref = input.x_torso_nominal;
    const Eigen::Isometry3d T_W_F = getSelectedReferenceTransform(input);
    output.x_left_ref_world =
        transformPoseToWorld(T_W_F, output.x_left_ref_local);
    output.x_right_ref_world =
        transformPoseToWorld(T_W_F, output.x_right_ref_local);
    return output;
  }

  // --------------------------------------------------------------------------
  // 1. Compute hand compliance.
  // --------------------------------------------------------------------------

  if (params_.compliance_mode == ComplianceMode::NONE) {
    output.delta_xc_left.setZero();
    output.delta_xc_right.setZero();
    output.delta_dxc_left.setZero();
    output.delta_dxc_right.setZero();
    output.delta_ddxc_left.setZero();
    output.delta_ddxc_right.setZero();
  } else if (input.use_manual_delta_xc) {
    output.delta_xc_left = input.manual_delta_xc_left;
    output.delta_xc_right = input.manual_delta_xc_right;

    output.delta_dxc_left =
        (output.delta_xc_left - delta_xc_left_prev_) / dt;
    output.delta_dxc_right =
        (output.delta_xc_right - delta_xc_right_prev_) / dt;

    output.delta_ddxc_left =
        (output.delta_dxc_left - delta_dxc_left_prev_) / dt;
    output.delta_ddxc_right =
        (output.delta_dxc_right - delta_dxc_right_prev_) / dt;

    delta_xc_left_prev_ = output.delta_xc_left;
    delta_xc_right_prev_ = output.delta_xc_right;
    delta_dxc_left_prev_ = output.delta_dxc_left;
    delta_dxc_right_prev_ = output.delta_dxc_right;
  } else if (compliance_enabled) {
    if (params_.use_admittance_dynamics) {
      output.delta_xc_left = computeArmCompliance(
          input.wrench_left,
          input.wrench_left_ref,
          params_.Ma_left,
          params_.Da_left,
          params_.Ka_left,
          params_.S_left,
          dt,
          delta_xc_left_prev_,
          delta_dxc_left_prev_,
          output.delta_ddxc_left);

      output.delta_xc_right = computeArmCompliance(
          input.wrench_right,
          input.wrench_right_ref,
          params_.Ma_right,
          params_.Da_right,
          params_.Ka_right,
          params_.S_right,
          dt,
          delta_xc_right_prev_,
          delta_dxc_right_prev_,
          output.delta_ddxc_right);

      output.delta_dxc_left = delta_dxc_left_prev_;
      output.delta_dxc_right = delta_dxc_right_prev_;
    } else {
      output.delta_xc_left = computeQuasiStaticCompliance(
          input.wrench_left,
          input.wrench_left_ref,
          params_.Ka_left,
          params_.S_left);

      output.delta_xc_right = computeQuasiStaticCompliance(
          input.wrench_right,
          input.wrench_right_ref,
          params_.Ka_right,
          params_.S_right);

      output.delta_dxc_left =
          (output.delta_xc_left - delta_xc_left_prev_) / dt;
      output.delta_dxc_right =
          (output.delta_xc_right - delta_xc_right_prev_) / dt;

      output.delta_ddxc_left =
          (output.delta_dxc_left - delta_dxc_left_prev_) / dt;
      output.delta_ddxc_right =
          (output.delta_dxc_right - delta_dxc_right_prev_) / dt;

      delta_xc_left_prev_ = output.delta_xc_left;
      delta_xc_right_prev_ = output.delta_xc_right;
      delta_dxc_left_prev_ = output.delta_dxc_left;
      delta_dxc_right_prev_ = output.delta_dxc_right;
    }
  } else {
    output.delta_xc_left.setZero();
    output.delta_xc_right.setZero();
    output.delta_dxc_left.setZero();
    output.delta_dxc_right.setZero();
    output.delta_ddxc_left.setZero();
    output.delta_ddxc_right.setZero();
  }

  // Apply arm displacement limits.
  const Vector6d delta_xc_left_before_limit = output.delta_xc_left;
  const Vector6d delta_xc_right_before_limit = output.delta_xc_right;

  output.delta_xc_left =
      applyVectorLimit(output.delta_xc_left, params_.delta_xc_left_limit);
  output.delta_xc_right =
      applyVectorLimit(output.delta_xc_right, params_.delta_xc_right_limit);

  // Anti-windup for admittance velocity when saturation happens.
  for (int i = 0; i < 6; ++i) {
    if (std::abs(output.delta_xc_left(i) - delta_xc_left_before_limit(i)) >
        1e-12) {
      delta_dxc_left_prev_(i) = 0.0;
      output.delta_dxc_left(i) = 0.0;
    }

    if (std::abs(output.delta_xc_right(i) - delta_xc_right_before_limit(i)) >
        1e-12) {
      delta_dxc_right_prev_(i) = 0.0;
      output.delta_dxc_right(i) = 0.0;
    }
  }

  delta_xc_left_prev_ = output.delta_xc_left;
  delta_xc_right_prev_ = output.delta_xc_right;

  if (compliance_enabled) {
    output.delta_xc_left_filtered = firstOrderLowpass(
        output.delta_xc_left,
        delta_xc_left_filtered_prev_,
        params_.filter_alpha);

    output.delta_xc_right_filtered = firstOrderLowpass(
        output.delta_xc_right,
        delta_xc_right_filtered_prev_,
        params_.filter_alpha);
  } else {
    // NONE is an immediate disable, not a slow decay through the filter.
    output.delta_xc_left_filtered.setZero();
    output.delta_xc_right_filtered.setZero();
  }

  delta_xc_left_filtered_prev_ = output.delta_xc_left_filtered;
  delta_xc_right_filtered_prev_ = output.delta_xc_right_filtered;

  // --------------------------------------------------------------------------
  // 2. Solve torso compliance QP.
  // --------------------------------------------------------------------------

  const Vector6d delta_xb_final_previous = delta_xb_filtered_prev_;

  if (torso_enabled) {
    output.delta_xb.tail<3>() = solveTorsoComplianceQP(
        output.delta_xc_left_filtered,
        output.delta_xc_right_filtered,
        Ab_left,
        Ab_right);

    output.qp_solved = debug_.qp_solved;

    if (output.qp_solved) {
      // Eq. (10) uses the previous raw QP optimum, not the filtered output.
      delta_xb_qp_prev_ = output.delta_xb.tail<3>();

      output.delta_xb_filtered = firstOrderLowpass(
          output.delta_xb,
          delta_xb_filtered_prev_,
          params_.filter_alpha);

      output.delta_xb_filtered.head<3>().setZero();
      output.delta_xb_filtered.tail<3>() = applyTorsoOrientationBounds(
          output.delta_xb_filtered.tail<3>());
      output.delta_xb_final = output.delta_xb_filtered;
    } else {
      // A failed QP is not a new command. Hold the last valid torso output and
      // leave all output histories unchanged for a consistent recovery.
      output.delta_xb_filtered = delta_xb_filtered_prev_;
      output.delta_xb_final = delta_xb_filtered_prev_;
    }
  } else {
    output.delta_xb.setZero();
    output.delta_xb_filtered.setZero();
    output.delta_xb_final.setZero();
    output.qp_solved = true;
    debug_.qp_solved = true;
    setDebugStatus(debug_.qp_status, "torso_allocation_disabled");
    delta_xb_qp_prev_.setZero();
  }

  // --------------------------------------------------------------------------
  // 3. Compute residual arm compliance after torso allocation.
  // --------------------------------------------------------------------------

  output.delta_x_left_torso = Ab_left * output.delta_xb_final.tail<3>();
  output.delta_x_right_torso = Ab_right * output.delta_xb_final.tail<3>();

  if (compliance_enabled) {
    output.delta_x_left_arm =
        output.delta_xc_left_filtered - output.delta_x_left_torso;

    output.delta_x_right_arm =
        output.delta_xc_right_filtered - output.delta_x_right_torso;
  } else {
    output.delta_x_left_arm.setZero();
    output.delta_x_right_arm.setZero();
  }

  // Eqs. (20)--(21) use derivatives of enabled, valid allocated outputs. The
  // first update after reset/mode enable is initialized without an impulse.
  if (output.qp_solved) {
    if (torso_enabled) {
      if (torso_output_history_initialized_) {
        output.delta_dxb =
            (output.delta_xb_final - delta_xb_final_previous) / dt;
        output.delta_ddxb =
            (output.delta_dxb - delta_dxb_prev_) / dt;
      }

      delta_xb_filtered_prev_ = output.delta_xb_final;
      delta_dxb_prev_ = output.delta_dxb;
      torso_output_history_initialized_ = true;
    } else {
      delta_xb_filtered_prev_.setZero();
      delta_dxb_prev_.setZero();
      torso_output_history_initialized_ = false;
    }

    if (hand_reference_enabled) {
      if (arm_output_history_initialized_) {
        output.delta_dx_left_arm =
            (output.delta_x_left_arm - delta_x_left_arm_prev_) / dt;
        output.delta_dx_right_arm =
            (output.delta_x_right_arm - delta_x_right_arm_prev_) / dt;

        output.delta_ddx_left_arm =
            (output.delta_dx_left_arm - delta_dx_left_arm_prev_) / dt;
        output.delta_ddx_right_arm =
            (output.delta_dx_right_arm - delta_dx_right_arm_prev_) / dt;
      }

      delta_x_left_arm_prev_ = output.delta_x_left_arm;
      delta_x_right_arm_prev_ = output.delta_x_right_arm;
      delta_dx_left_arm_prev_ = output.delta_dx_left_arm;
      delta_dx_right_arm_prev_ = output.delta_dx_right_arm;
      arm_output_history_initialized_ = true;
    } else {
      delta_x_left_arm_prev_.setZero();
      delta_x_right_arm_prev_.setZero();
      delta_dx_left_arm_prev_.setZero();
      delta_dx_right_arm_prev_.setZero();
      arm_output_history_initialized_ = false;
    }
  }

  // --------------------------------------------------------------------------
  // 4. Build final references.
  // --------------------------------------------------------------------------
  //
  // Per Eqs. (13)--(14), an enabled hand reference receives the residual arm
  // displacement. TORSO_ONLY keeps the final hand references nominal.

  if (!hand_reference_enabled) {
    output.x_left_ref_local = input.x_left_nominal_local;
    output.x_right_ref_local = input.x_right_nominal_local;
  } else {
    output.x_left_ref_local =
        input.x_left_nominal_local + output.delta_x_left_arm;

    output.x_right_ref_local =
        input.x_right_nominal_local + output.delta_x_right_arm;
  }

  output.x_torso_ref =
      input.x_torso_nominal + output.delta_xb_final;

  const Eigen::Isometry3d T_W_F =
      getSelectedReferenceTransform(input);

  output.x_left_ref_world =
      transformPoseToWorld(T_W_F, output.x_left_ref_local);

  output.x_right_ref_world =
      transformPoseToWorld(T_W_F, output.x_right_ref_local);

  output.valid =
      isFiniteVector(output.delta_xc_left) &&
      isFiniteVector(output.delta_xc_right) &&
      isFiniteVector(output.delta_dxc_left) &&
      isFiniteVector(output.delta_dxc_right) &&
      isFiniteVector(output.delta_ddxc_left) &&
      isFiniteVector(output.delta_ddxc_right) &&
      isFiniteVector(output.delta_xb_final) &&
      isFiniteVector(output.delta_x_left_arm) &&
      isFiniteVector(output.delta_x_right_arm) &&
      isFiniteVector(output.delta_dxb) &&
      isFiniteVector(output.delta_ddxb) &&
      isFiniteVector(output.delta_dx_left_arm) &&
      isFiniteVector(output.delta_dx_right_arm) &&
      isFiniteVector(output.delta_ddx_left_arm) &&
      isFiniteVector(output.delta_ddx_right_arm) &&
      isFiniteVector(output.x_left_ref_world) &&
      isFiniteVector(output.x_right_ref_world) &&
      isFiniteVector(output.x_torso_ref) &&
      input_dt_valid &&
      output.qp_solved;

  return output;
}

ComplianceReferenceGenerator::Vector6d
ComplianceReferenceGenerator::computeArmCompliance(
    const Vector6d& wrench,
    const Vector6d& wrench_ref,
    const Matrix6d& Ma,
    const Matrix6d& Da,
    const Matrix6d& Ka,
    const Matrix6d& S,
    double dt,
    Vector6d& delta_x,
    Vector6d& delta_dx,
    Vector6d& delta_ddx) const {
  const Vector6d wrench_error = S * (wrench - wrench_ref);

  Matrix6d Ma_reg = Ma;
  Ma_reg += 1e-10 * Matrix6d::Identity();

  // M * ddx + D * dx + K * x = F
  const Vector6d rhs =
      wrench_error - Da * delta_dx - Ka * delta_x;

  delta_ddx = Ma_reg.ldlt().solve(rhs);

  if (!isFiniteVector(delta_ddx)) {
    delta_ddx.setZero();
  }

  // Semi-implicit Euler integration.
  delta_dx += delta_ddx * dt;
  delta_x += delta_dx * dt;

  if (!isFiniteVector(delta_dx)) {
    delta_dx.setZero();
  }

  if (!isFiniteVector(delta_x)) {
    delta_x.setZero();
  }

  return delta_x;
}

ComplianceReferenceGenerator::Vector6d
ComplianceReferenceGenerator::computeQuasiStaticCompliance(
    const Vector6d& wrench,
    const Vector6d& wrench_ref,
    const Matrix6d& Ka,
    const Matrix6d& S) const {
  const Vector6d wrench_error = S * (wrench - wrench_ref);

  Matrix6d Ka_reg = Ka;
  Ka_reg += 1e-10 * Matrix6d::Identity();

  Vector6d delta_x = Ka_reg.ldlt().solve(wrench_error);

  if (!isFiniteVector(delta_x)) {
    delta_x.setZero();
  }

  return delta_x;
}

ComplianceReferenceGenerator::Vector6d
ComplianceReferenceGenerator::applyVectorLimit(
    const Vector6d& x,
    const Vector6d& limit) const {
  Vector6d y = x;

  for (int i = 0; i < 6; ++i) {
    const double lim = std::abs(limit(i));

    if (!std::isfinite(lim)) {
      continue;
    }

    y(i) = std::min(std::max(y(i), -lim), lim);
  }

  return y;
}

ComplianceReferenceGenerator::Vector3d
ComplianceReferenceGenerator::applyTorsoOrientationBounds(
    const Vector3d& x) const {
  Vector3d bounded = x;

  for (int i = 0; i < 3; ++i) {
    double lower = params_.delta_xb_min(i + 3);
    double upper = params_.delta_xb_max(i + 3);

    if (!std::isfinite(lower)) {
      lower = -1e9;
    }
    if (!std::isfinite(upper)) {
      upper = 1e9;
    }
    if (lower > upper) {
      const double midpoint = 0.5 * (lower + upper);
      lower = midpoint;
      upper = midpoint;
    }

    bounded(i) = std::min(std::max(bounded(i), lower), upper);
  }

  return bounded;
}

ComplianceReferenceGenerator::Vector6d
ComplianceReferenceGenerator::firstOrderLowpass(
    const Vector6d& x,
    const Vector6d& x_prev,
    double alpha) const {
  const double a = std::min(std::max(alpha, 0.0), 1.0);
  return a * x_prev + (1.0 - a) * x;
}

void ComplianceReferenceGenerator::buildTorsoComplianceQP(
    const Vector6d& delta_xc_left,
    const Vector6d& delta_xc_right,
    const Matrix6x3d& Ab_left,
    const Matrix6x3d& Ab_right,
    Matrix3d& H,
    Vector3d& g,
    Vector3d& lbx,
    Vector3d& ubx) const {
  // Eq. (12): E_i = S_i A_b,i and y_i = S_i delta_xc_i.
  const Matrix6x3d E_left = params_.S_allocation_left * Ab_left;
  const Matrix6x3d E_right = params_.S_allocation_right * Ab_right;
  const Vector6d y_left = params_.S_allocation_left * delta_xc_left;
  const Vector6d y_right = params_.S_allocation_right * delta_xc_right;

  const Matrix6d W_left =
      0.5 * (params_.W_left + params_.W_left.transpose());
  const Matrix6d W_right =
      0.5 * (params_.W_right + params_.W_right.transpose());
  const Matrix3d Kb = 0.5 * (
      params_.Kb.bottomRightCorner<3, 3>() +
      params_.Kb.bottomRightCorner<3, 3>().transpose());
  const Matrix3d W_smooth = 0.5 * (
      params_.W_smooth.bottomRightCorner<3, 3>() +
      params_.W_smooth.bottomRightCorner<3, 3>().transpose());
  const Matrix3d W_reg = 0.5 * (
      params_.W_reg.bottomRightCorner<3, 3>() +
      params_.W_reg.bottomRightCorner<3, 3>().transpose());

  H = Kb +
      E_left.transpose() * W_left * E_left +
      E_right.transpose() * W_right * E_right +
      W_smooth +
      W_reg;

  g =
      -params_.rho_left * E_left.transpose() * W_left * y_left
      -params_.rho_right * E_right.transpose() * W_right * y_right
      -W_smooth * delta_xb_qp_prev_;

  H = 0.5 * (H + H.transpose());

  // Numerical regularization for qpOASES.
  H += 1e-10 * Matrix3d::Identity();

  lbx = params_.delta_xb_min.tail<3>();
  ubx = params_.delta_xb_max.tail<3>();

  const double bound_relaxation = std::isfinite(params_.bound_relaxation)
      ? std::max(params_.bound_relaxation, 0.0)
      : 0.0;

  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(lbx(i))) {
      lbx(i) = -1e9;
    }

    if (!std::isfinite(ubx(i))) {
      ubx(i) = 1e9;
    }

    if (lbx(i) > ubx(i)) {
      const double mid = 0.5 * (lbx(i) + ubx(i));
      lbx(i) = mid;
      ubx(i) = mid;
    }

    lbx(i) -= bound_relaxation;
    ubx(i) += bound_relaxation;
  }
}

ComplianceReferenceGenerator::Vector3d
ComplianceReferenceGenerator::solveTorsoComplianceQP(
    const Vector6d& delta_xc_left,
    const Vector6d& delta_xc_right,
    const Matrix6x3d& Ab_left,
    const Matrix6x3d& Ab_right) {
  Matrix3d H;
  Vector3d g;
  Vector3d lbx;
  Vector3d ubx;

  buildTorsoComplianceQP(
      delta_xc_left,
      delta_xc_right,
      Ab_left,
      Ab_right,
      H,
      g,
      lbx,
      ubx);

  debug_.H = H;
  debug_.g = g;
  debug_.lbx = lbx;
  debug_.ubx = ubx;

  Vector3d delta_xb = Vector3d::Zero();

  if (!H.allFinite() || !g.allFinite() ||
      !lbx.allFinite() || !ubx.allFinite()) {
    debug_.qp_solved = false;
    setDebugStatus(
        debug_.qp_status,
        "Torso compliance QP contains non-finite coefficients.");
    debug_.objective_value = std::numeric_limits<double>::quiet_NaN();
    return delta_xb;
  }

  if (!solver_built_) {
    buildSolver();
  }

  if (!solver_built_) {
    debug_.qp_solved = false;
    setDebugStatus(
        debug_.qp_status,
        "CasADi qpOASES solver is not built. Returning zero delta_xb.");
    debug_.objective_value = computeObjective(delta_xb, H, g);
    return delta_xb;
  }

  const auto t0 = std::chrono::steady_clock::now();

  try {
    casadi::DMDict arg;

    arg["h"] = eigenToDM(Eigen::MatrixXd(H));
    arg["g"] = eigenToDM(Eigen::VectorXd(g));

    // No general linear constraints.
    arg["a"] = casadi::DM::zeros(0, 3);
    arg["lba"] = casadi::DM::zeros(0, 1);
    arg["uba"] = casadi::DM::zeros(0, 1);

    // Box constraints.
    arg["lbx"] = eigenToDM(Eigen::VectorXd(lbx));
    arg["ubx"] = eigenToDM(Eigen::VectorXd(ubx));

    casadi::DMDict res = qp_solver_(arg);

    const casadi::Dict stats = qp_solver_.stats();
    const auto success = stats.find("success");
    if (success != stats.end() && !success->second.to_bool()) {
      std::string return_status = "unknown_status";
      const auto status = stats.find("return_status");
      if (status != stats.end()) {
        return_status = status->second.to_string();
      }
      throw std::runtime_error(
          "CasADi qpOASES reported failure: " + return_status);
    }

    if (res.find("x") == res.end()) {
      throw std::runtime_error("CasADi result does not contain key 'x'.");
    }

    Eigen::VectorXd sol = dmToEigen(res.at("x"));

    if (sol.size() < 3) {
      throw std::runtime_error("CasADi solution size is smaller than 3.");
    }

    delta_xb = sol.head<3>();

    bool within_bounds = true;
    for (int i = 0; i < 3; ++i) {
      if (delta_xb(i) < lbx(i) - params_.bound_tolerance ||
          delta_xb(i) > ubx(i) + params_.bound_tolerance) {
        within_bounds = false;
        break;
      }
    }

    if (!delta_xb.allFinite()) {
      debug_.qp_solved = false;
      setDebugStatus(
          debug_.qp_status,
          "CasADi qpOASES returned non-finite solution.");
      delta_xb.setZero();
    } else if (!within_bounds) {
      debug_.qp_solved = false;
      setDebugStatus(
          debug_.qp_status,
          "CasADi qpOASES solution violates bounds. Returning saturated solution.");

      for (int i = 0; i < 3; ++i) {
        delta_xb(i) = std::min(std::max(delta_xb(i), lbx(i)), ubx(i));
      }
    } else {
      debug_.qp_solved = true;
      setDebugStatus(debug_.qp_status, "solved_by_casadi_qpoases");
    }

  } catch (const std::exception& e) {
    debug_.qp_solved = false;
    setDebugStatus(
        debug_.qp_status,
        std::string("CasADi qpOASES exception: ") + e.what());
    delta_xb.setZero();
  }

  const auto t1 = std::chrono::steady_clock::now();
  debug_.qp_solve_time_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  debug_.objective_value = computeObjective(delta_xb, H, g);

  return delta_xb;
}

double ComplianceReferenceGenerator::computeObjective(
    const Vector3d& delta_xb,
    const Matrix3d& H,
    const Vector3d& g) const {
  return objectiveValue(delta_xb, H, g);
}

Eigen::Isometry3d
ComplianceReferenceGenerator::getSelectedReferenceTransform(
    const Input& input) const {
  switch (params_.hand_reference_frame) {
    case HandReferenceFrame::TORSO:
      return input.T_W_B_ref;

    case HandReferenceFrame::GAIT_CENTER:
      return input.T_W_G;

    default:
      return input.T_W_B_ref;
  }
}

ComplianceReferenceGenerator::Vector6d
ComplianceReferenceGenerator::transformPoseToWorld(
    const Eigen::Isometry3d& T_W_F,
    const Vector6d& x_F) const {
  const Eigen::Vector3d p_F = x_F.head<3>();

  const double roll = x_F(3);
  const double pitch = x_F(4);
  const double yaw = x_F(5);

  const Eigen::Matrix3d R_F_local =
      rpyToRotation(roll, pitch, yaw);

  const Eigen::Vector3d p_W =
      T_W_F.translation() + T_W_F.linear() * p_F;

  const Eigen::Matrix3d R_W =
      T_W_F.linear() * R_F_local;

  return makePose6dFromRotation(p_W, R_W);
}

casadi::DM ComplianceReferenceGenerator::eigenToDM(
    const Eigen::MatrixXd& M) {
  casadi::DM dm = casadi::DM::zeros(M.rows(), M.cols());

  for (int r = 0; r < M.rows(); ++r) {
    for (int c = 0; c < M.cols(); ++c) {
      dm(r, c) = M(r, c);
    }
  }

  return dm;
}

casadi::DM ComplianceReferenceGenerator::eigenToDM(
    const Eigen::VectorXd& v) {
  casadi::DM dm = casadi::DM::zeros(v.rows(), 1);

  for (int r = 0; r < v.rows(); ++r) {
    dm(r, 0) = v(r);
  }

  return dm;
}

Eigen::VectorXd ComplianceReferenceGenerator::dmToEigen(
    const casadi::DM& dm) {
  const int rows = static_cast<int>(dm.size1());
  const int cols = static_cast<int>(dm.size2());

  Eigen::VectorXd v(rows * cols);

  int k = 0;
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      v(k++) = static_cast<double>(dm(r, c));
    }
  }

  return v;
}

}  // namespace labrob
