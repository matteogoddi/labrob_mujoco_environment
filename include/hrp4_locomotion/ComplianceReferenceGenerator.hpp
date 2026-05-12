#ifndef LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_
#define LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_

#include <string>

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace labrob {

class ComplianceReferenceGenerator {
public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;

  enum class ComplianceMode {
    NONE = 0,
    HAND_ONLY,
    TORSO_ONLY,
    HAND_AND_TORSO
  };

  enum class HandReferenceFrame {
    TORSO = 0,
    GAIT_CENTER
  };

  struct Parameters {
    // Validation switches
    ComplianceMode compliance_mode = ComplianceMode::HAND_AND_TORSO;
    HandReferenceFrame hand_reference_frame = HandReferenceFrame::TORSO;

    // If true, use M-D-K admittance dynamics.
    // If false, use quasi-static mapping: delta_x = K^{-1} F.
    bool use_admittance_dynamics = true;

    // Arm admittance parameters:
    // M_a * delta_ddx + D_a * delta_dx + K_a * delta_x = S * (wrench - wrench_ref)
    Matrix6d Ma_left = Matrix6d::Identity();
    Matrix6d Da_left = Matrix6d::Identity();
    Matrix6d Ka_left = Matrix6d::Identity();

    Matrix6d Ma_right = Matrix6d::Identity();
    Matrix6d Da_right = Matrix6d::Identity();
    Matrix6d Ka_right = Matrix6d::Identity();

    // Selection matrices for enabled compliance DOFs.
    // Example: enable only translation by setting diag = [1,1,1,0,0,0].
    Matrix6d S_left = Matrix6d::Identity();
    Matrix6d S_right = Matrix6d::Identity();

    // Torso compliance stiffness.
    // Larger Kb means torso motion is more expensive, so hand compliance dominates.
    Matrix6d Kb = Matrix6d::Identity();

    // Additional QP regularization / smoothing.
    Matrix6d W_smooth = Matrix6d::Zero();
    Matrix6d W_reg = 1e-8 * Matrix6d::Identity();

    // Torso displacement bounds.
    Vector6d delta_xb_min = Vector6d::Constant(-1e9);
    Vector6d delta_xb_max = Vector6d::Constant(1e9);

    // Arm compliant displacement bounds.
    // This is symmetric: |delta_xc| <= delta_xc_limit.
    Vector6d delta_xc_left_limit = Vector6d::Constant(1e9);
    Vector6d delta_xc_right_limit = Vector6d::Constant(1e9);

    // Low-pass filter alpha in [0,1].
    // Higher alpha means smoother but slower response.
    double filter_alpha = 0.99;

    // CasADi / qpOASES settings.
    int print_level = 0;
    double bound_tolerance = 1e-8;
    double bound_relaxation = 1e-8;
  };

  struct Input {
    double dt = 0.001;

    // Estimated external wrenches at left and right wrists.
    Vector6d wrench_left = Vector6d::Zero();
    Vector6d wrench_right = Vector6d::Zero();

    // Reference / nominal wrenches.
    // For object carrying, Fz may include nominal object weight share.
    Vector6d wrench_left_ref = Vector6d::Zero();
    Vector6d wrench_right_ref = Vector6d::Zero();

    // Torso-to-hand Jacobians used in the torso allocation QP:
    // delta_x_hand_due_to_torso = Jb * delta_xb.
    Matrix6d Jb_left = Matrix6d::Zero();
    Matrix6d Jb_right = Matrix6d::Zero();

    // Nominal hand references expressed in the selected local frame:
    // if frame == TORSO, these are expressed in torso frame;
    // if frame == GAIT_CENTER, these are expressed in gait-center frame.
    Vector6d x_left_nominal_local = Vector6d::Zero();
    Vector6d x_right_nominal_local = Vector6d::Zero();

    // Nominal torso reference in world or WBC task coordinates.
    Vector6d x_torso_nominal = Vector6d::Zero();

    // Transform of reference torso frame and gait-center frame in world.
    // Use T_W_B_ref rather than actual torso pose to avoid feedback-induced hand reference shaking.
    Eigen::Isometry3d T_W_B_ref = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d T_W_G = Eigen::Isometry3d::Identity();

    // Optional manually injected hand compliance displacement.
    // Useful for TORSO_ONLY validation.
    bool use_manual_delta_xc = false;
    Vector6d manual_delta_xc_left = Vector6d::Zero();
    Vector6d manual_delta_xc_right = Vector6d::Zero();
  };

  struct Output {
    // Raw hand compliance displacement from admittance or stiffness mapping.
    Vector6d delta_xc_left = Vector6d::Zero();
    Vector6d delta_xc_right = Vector6d::Zero();

    // Hand compliance velocity / acceleration.
    Vector6d delta_dxc_left = Vector6d::Zero();
    Vector6d delta_dxc_right = Vector6d::Zero();
    Vector6d delta_ddxc_left = Vector6d::Zero();
    Vector6d delta_ddxc_right = Vector6d::Zero();

    // Torso compliance displacement from QP.
    Vector6d delta_xb = Vector6d::Zero();

    // Filtered values.
    Vector6d delta_xc_left_filtered = Vector6d::Zero();
    Vector6d delta_xc_right_filtered = Vector6d::Zero();
    Vector6d delta_xb_filtered = Vector6d::Zero();

    // Arm-only residual compliance after torso allocation:
    // delta_x_arm = delta_xc - Jb * delta_xb.
    Vector6d delta_x_left_arm = Vector6d::Zero();
    Vector6d delta_x_right_arm = Vector6d::Zero();

    // Final torso correction sent to WBC.
    Vector6d delta_xb_final = Vector6d::Zero();

    // Final references for WBC.
    // x_*_ref_local are expressed in the selected local frame.
    // x_*_ref_world are transformed into world frame.
    Vector6d x_left_ref_local = Vector6d::Zero();
    Vector6d x_right_ref_local = Vector6d::Zero();
    Vector6d x_left_ref_world = Vector6d::Zero();
    Vector6d x_right_ref_world = Vector6d::Zero();
    Vector6d x_torso_ref = Vector6d::Zero();

    bool valid = false;
    bool qp_solved = false;
  };

  struct DebugInfo {
    Matrix6d H = Matrix6d::Zero();
    Vector6d g = Vector6d::Zero();
    Vector6d lbx = Vector6d::Zero();
    Vector6d ubx = Vector6d::Zero();

    double objective_value = 0.0;
    double qp_solve_time_ms = 0.0;

    bool qp_solved = false;
    std::string qp_status;
  };

public:
  ComplianceReferenceGenerator();
  explicit ComplianceReferenceGenerator(const Parameters& params);

  void reset();

  void setParameters(const Parameters& params);
  const Parameters& getParameters() const;

  Output update(const Input& input);

  const DebugInfo& getDebugInfo() const;

private:
  void buildSolver();

  Vector6d computeArmCompliance(
      const Vector6d& wrench,
      const Vector6d& wrench_ref,
      const Matrix6d& Ma,
      const Matrix6d& Da,
      const Matrix6d& Ka,
      const Matrix6d& S,
      double dt,
      Vector6d& delta_x,
      Vector6d& delta_dx,
      Vector6d& delta_ddx) const;

  Vector6d computeQuasiStaticCompliance(
      const Vector6d& wrench,
      const Vector6d& wrench_ref,
      const Matrix6d& Ka,
      const Matrix6d& S) const;

  Vector6d applyVectorLimit(
      const Vector6d& x,
      const Vector6d& limit) const;

  Vector6d firstOrderLowpass(
      const Vector6d& x,
      const Vector6d& x_prev,
      double alpha) const;

  void buildTorsoComplianceQP(
      const Vector6d& delta_xc_left,
      const Vector6d& delta_xc_right,
      const Matrix6d& Jb_left,
      const Matrix6d& Jb_right,
      double dt,
      Matrix6d& H,
      Vector6d& g,
      Vector6d& lbx,
      Vector6d& ubx) const;

  Vector6d solveTorsoComplianceQP(
      const Vector6d& delta_xc_left,
      const Vector6d& delta_xc_right,
      const Matrix6d& Jb_left,
      const Matrix6d& Jb_right,
      double dt);

  double computeObjective(
      const Vector6d& delta_xb,
      const Matrix6d& H,
      const Vector6d& g) const;

  Eigen::Isometry3d getSelectedReferenceTransform(
      const Input& input) const;

  Vector6d transformPoseToWorld(
      const Eigen::Isometry3d& T_W_F,
      const Vector6d& x_F) const;

  static casadi::DM eigenToDM(const Eigen::MatrixXd& M);
  static casadi::DM eigenToDM(const Eigen::VectorXd& v);
  static Eigen::VectorXd dmToEigen(const casadi::DM& dm);

private:
  Parameters params_;
  DebugInfo debug_;

  casadi::Function qp_solver_;

  bool solver_built_ = false;
  bool initialized_ = false;

  Vector6d delta_xc_left_prev_ = Vector6d::Zero();
  Vector6d delta_xc_right_prev_ = Vector6d::Zero();
  Vector6d delta_xb_prev_ = Vector6d::Zero();

  Vector6d delta_dxc_left_prev_ = Vector6d::Zero();
  Vector6d delta_dxc_right_prev_ = Vector6d::Zero();

  Vector6d delta_xc_left_filtered_prev_ = Vector6d::Zero();
  Vector6d delta_xc_right_filtered_prev_ = Vector6d::Zero();
  Vector6d delta_xb_filtered_prev_ = Vector6d::Zero();
};

}  // namespace labrob

#endif  // LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_