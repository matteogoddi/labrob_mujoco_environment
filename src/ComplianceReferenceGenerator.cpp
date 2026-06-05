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

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

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

double objectiveValue(
    const Vector6d& x,
    const Matrix6d& H,
    const Vector6d& g) {
  return 0.5 * x.dot(H * x) + g.dot(x);
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
  buildSolver();
  reset();
}

void ComplianceReferenceGenerator::reset() {
  debug_ = DebugInfo();

  delta_xc_left_prev_.setZero();
  delta_xc_right_prev_.setZero();
  delta_xb_prev_.setZero();

  delta_dxc_left_prev_.setZero();
  delta_dxc_right_prev_.setZero();

  delta_xc_left_filtered_prev_.setZero();
  delta_xc_right_filtered_prev_.setZero();
  delta_xb_filtered_prev_.setZero();

  initialized_ = true;
}

void ComplianceReferenceGenerator::setParameters(
    const Parameters& params) {
  params_ = params;
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

void ComplianceReferenceGenerator::buildSolver() {
  const int nx = 6;
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
    debug_.qp_status = "CasADi qpOASES solver built successfully.";
  } catch (const std::exception& e) {
    solver_built_ = false;
    debug_.qp_status =
        std::string("Failed to build CasADi qpOASES solver: ") + e.what();
  }
}

ComplianceReferenceGenerator::Output
ComplianceReferenceGenerator::update(const Input& input) {
  Output output;
  debug_ = DebugInfo();

  const bool hand_enabled =
      params_.compliance_mode == ComplianceMode::HAND_ONLY ||
      params_.compliance_mode == ComplianceMode::HAND_AND_TORSO;

  const bool torso_enabled =
      params_.compliance_mode == ComplianceMode::TORSO_ONLY ||
      params_.compliance_mode == ComplianceMode::HAND_AND_TORSO;

  const double dt = std::max(input.dt, 1e-6);

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
  } else if (hand_enabled) {
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

  output.delta_xc_left_filtered = firstOrderLowpass(
      output.delta_xc_left,
      delta_xc_left_filtered_prev_,
      params_.filter_alpha);

  output.delta_xc_right_filtered = firstOrderLowpass(
      output.delta_xc_right,
      delta_xc_right_filtered_prev_,
      params_.filter_alpha);

  delta_xc_left_filtered_prev_ = output.delta_xc_left_filtered;
  delta_xc_right_filtered_prev_ = output.delta_xc_right_filtered;

  // --------------------------------------------------------------------------
  // 2. Solve torso compliance QP.
  // --------------------------------------------------------------------------

  if (torso_enabled) {
    output.delta_xb = solveTorsoComplianceQP(
        output.delta_xc_left_filtered,
        output.delta_xc_right_filtered,
        input.Jb_left,
        input.Jb_right,
        dt);

    output.qp_solved = debug_.qp_solved;

    output.delta_xb_filtered = firstOrderLowpass(
        output.delta_xb,
        delta_xb_filtered_prev_,
        params_.filter_alpha);

    delta_xb_filtered_prev_ = output.delta_xb_filtered;
    output.delta_xb_final = output.delta_xb_filtered;
  } else {
    output.delta_xb.setZero();
    output.delta_xb_filtered.setZero();
    output.delta_xb_final.setZero();
    output.qp_solved = true;
  }

  delta_xb_prev_ = output.delta_xb_final;

  // --------------------------------------------------------------------------
  // 3. Compute residual arm compliance after torso allocation.
  // --------------------------------------------------------------------------

  output.delta_x_left_arm =
      output.delta_xc_left_filtered - input.Jb_left * output.delta_xb_final;

  output.delta_x_right_arm =
      output.delta_xc_right_filtered - input.Jb_right * output.delta_xb_final;

  if (params_.compliance_mode == ComplianceMode::TORSO_ONLY) {
    output.delta_x_left_arm.setZero();
    output.delta_x_right_arm.setZero();
  }

  // --------------------------------------------------------------------------
  // 4. Build final references.
  // --------------------------------------------------------------------------
  //
  // Important:
  // Here the absolute hand reference uses total hand compliance delta_xc.
  // The residual delta_x_arm is kept for debug or torso-relative arm task usage.

  if (params_.compliance_mode == ComplianceMode::NONE ||
      params_.compliance_mode == ComplianceMode::TORSO_ONLY) {
    output.x_left_ref_local = input.x_left_nominal_local;
    output.x_right_ref_local = input.x_right_nominal_local;
  } else {
    output.x_left_ref_local =
        input.x_left_nominal_local + output.delta_xc_left_filtered;

    output.x_right_ref_local =
        input.x_right_nominal_local + output.delta_xc_right_filtered;
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
      isFiniteVector(output.delta_xb_final) &&
      isFiniteVector(output.x_left_ref_world) &&
      isFiniteVector(output.x_right_ref_world) &&
      isFiniteVector(output.x_torso_ref) &&
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
    const Matrix6d& Jb_left,
    const Matrix6d& Jb_right,
    double /*dt*/,
    Matrix6d& H,
    Vector6d& g,
    Vector6d& lbx,
    Vector6d& ubx) const {
  const Matrix6d Ka_left_eff =
      params_.S_left.transpose() * params_.Ka_left * params_.S_left;

  const Matrix6d Ka_right_eff =
      params_.S_right.transpose() * params_.Ka_right * params_.S_right;

  H =
      params_.Kb +
      Jb_left.transpose() * Ka_left_eff * Jb_left +
      Jb_right.transpose() * Ka_right_eff * Jb_right +
      params_.W_smooth +
      params_.W_reg;

  g =
      -Jb_left.transpose() * Ka_left_eff * delta_xc_left
      -Jb_right.transpose() * Ka_right_eff * delta_xc_right
      -params_.W_smooth * delta_xb_prev_;

  H = 0.5 * (H + H.transpose());

  // Numerical regularization for qpOASES.
  H += 1e-10 * Matrix6d::Identity();

  lbx = params_.delta_xb_min;
  ubx = params_.delta_xb_max;

  for (int i = 0; i < 6; ++i) {
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

    lbx(i) -= params_.bound_relaxation;
    ubx(i) += params_.bound_relaxation;
  }
}

ComplianceReferenceGenerator::Vector6d
ComplianceReferenceGenerator::solveTorsoComplianceQP(
    const Vector6d& delta_xc_left,
    const Vector6d& delta_xc_right,
    const Matrix6d& Jb_left,
    const Matrix6d& Jb_right,
    double dt) {
  Matrix6d H;
  Vector6d g;
  Vector6d lbx;
  Vector6d ubx;

  buildTorsoComplianceQP(
      delta_xc_left,
      delta_xc_right,
      Jb_left,
      Jb_right,
      dt,
      H,
      g,
      lbx,
      ubx);

  debug_.H = H;
  debug_.g = g;
  debug_.lbx = lbx;
  debug_.ubx = ubx;

  Vector6d delta_xb = Vector6d::Zero();

  if (!solver_built_) {
    buildSolver();
  }

  if (!solver_built_) {
    debug_.qp_solved = false;
    debug_.qp_status =
        "CasADi qpOASES solver is not built. Returning zero delta_xb.";
    debug_.objective_value = computeObjective(delta_xb, H, g);
    return delta_xb;
  }

  const auto t0 = std::chrono::steady_clock::now();

  try {
    casadi::DMDict arg;

    arg["h"] = eigenToDM(Eigen::MatrixXd(H));
    arg["g"] = eigenToDM(Eigen::VectorXd(g));

    // No general linear constraints.
    arg["a"] = casadi::DM::zeros(0, 6);
    arg["lba"] = casadi::DM::zeros(0, 1);
    arg["uba"] = casadi::DM::zeros(0, 1);

    // Box constraints.
    arg["lbx"] = eigenToDM(Eigen::VectorXd(lbx));
    arg["ubx"] = eigenToDM(Eigen::VectorXd(ubx));

    casadi::DMDict res = qp_solver_(arg); // ！！！！solve QP ！！！

    if (res.find("x") == res.end()) {
      throw std::runtime_error("CasADi result does not contain key 'x'.");
    }

    Eigen::VectorXd sol = dmToEigen(res.at("x")); //final QP solution

    if (sol.size() < 6) {
      throw std::runtime_error("CasADi solution size is smaller than 6.");
    }

    delta_xb = sol.head<6>(); //transfer solution(DM) to Eigen vector

    bool within_bounds = true;
    for (int i = 0; i < 6; ++i) {
      if (delta_xb(i) < lbx(i) - params_.bound_tolerance ||
          delta_xb(i) > ubx(i) + params_.bound_tolerance) {
        within_bounds = false;
        break;
      }
    }

    if (!isFiniteVector(delta_xb)) {
      debug_.qp_solved = false;
      debug_.qp_status = "CasADi qpOASES returned non-finite solution.";
      delta_xb.setZero();
    } else if (!within_bounds) {
      debug_.qp_solved = false;
      debug_.qp_status =
          "CasADi qpOASES solution violates bounds. Returning saturated solution.";

      for (int i = 0; i < 6; ++i) {
        delta_xb(i) = std::min(std::max(delta_xb(i), lbx(i)), ubx(i));
      }
    } else {
      debug_.qp_solved = true;
      debug_.qp_status = "solved_by_casadi_qpoases";
    }

  } catch (const std::exception& e) {
    debug_.qp_solved = false;
    debug_.qp_status =
        std::string("CasADi qpOASES exception: ") + e.what();
    delta_xb.setZero();
  }

  const auto t1 = std::chrono::steady_clock::now();
  debug_.qp_solve_time_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  debug_.objective_value = computeObjective(delta_xb, H, g); //print info for debug

  return delta_xb;
}

double ComplianceReferenceGenerator::computeObjective(
    const Vector6d& delta_xb,
    const Matrix6d& H,
    const Vector6d& g) const {
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