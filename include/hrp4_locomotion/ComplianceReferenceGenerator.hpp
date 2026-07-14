#ifndef LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_
#define LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_

#include <array>

#include <Eigen/Dense>
#include <casadi/casadi.hpp>

namespace labrob {

class ComplianceReferenceGenerator {
public:
  using Vector3d = Eigen::Matrix<double, 3, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Matrix3d = Eigen::Matrix<double, 3, 3>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Matrix6x3d = Eigen::Matrix<double, 6, 3>;

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
    // Module switches kept for source and output compatibility. TORSO_ONLY
    // still computes admittance internally to drive the torso QP, but leaves
    // the final hand references nominal. The paper's rho endpoint experiments
    // should use HAND_AND_TORSO and set rho_left/right.
    ComplianceMode compliance_mode = ComplianceMode::HAND_ONLY;
    HandReferenceFrame hand_reference_frame = HandReferenceFrame::TORSO;

    // Per-hand arm--torso allocation coefficients from Eq. (12), clamped to
    // [0, 1]. rho=0 is hand-only after reset; rho=1 is torso-dominant (not
    // necessarily strictly torso-only because stiffness, bounds and
    // incompatible dual-hand targets can leave a non-zero arm residual).
    double rho_left = 1.0;
    double rho_right = 1.0;

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

    // Optional selection matrices for enabled admittance input DOFs. These
    // preserve the previous CRG API; the paper's allocation selectors are
    // S_allocation_left/right below.
    // Example: enable only translation by setting diag = [1,1,1,0,0,0].
    Matrix6d S_left = Matrix6d::Identity();
    Matrix6d S_right = Matrix6d::Identity();

    // Eq. (12) task selectors S_i and allocation weights W_i.
    Matrix6d S_allocation_left = Matrix6d::Identity();
    Matrix6d S_allocation_right = Matrix6d::Identity();
    Matrix6d W_left = Matrix6d::Identity();
    Matrix6d W_right = Matrix6d::Identity();

    // Torso orientation stiffness. The 6D storage is retained for source
    // compatibility, but only bottomRight<3,3>() (roll, pitch, yaw) is used.
    // Larger Kb means torso motion is more expensive, so arm compliance dominates.
    Matrix6d Kb = Matrix6d::Identity();

    // Additional QP regularization / smoothing. Only the angular 3x3 blocks
    // are used. W_smooth is applied to the previous raw QP optimum.
    Matrix6d W_smooth = Matrix6d::Zero();
    Matrix6d W_reg = 1e-8 * Matrix6d::Identity();

    // Torso orientation bounds. Only tail<3>() is used; torso translation is
    // never an optimization variable in the paper-aligned CRG.
    Vector6d delta_xb_min = Vector6d::Constant(-1e9);
    Vector6d delta_xb_max = Vector6d::Constant(1e9);

    // Arm compliant displacement bounds.
    // This is symmetric: |delta_xc| <= delta_xc_limit.
    Vector6d delta_xc_left_limit = Vector6d::Constant(1e9);
    Vector6d delta_xc_right_limit = Vector6d::Constant(1e9);

    // Optional implementation-level low-pass filter alpha in [0,1]. Higher
    // alpha means smoother but slower response. Use 0 for the exact unfiltered
    // signals in Eqs. (12)--(14); W_smooth provides the paper's QP smoothing.
    double filter_alpha = 0.99;

    // CasADi / qpOASES settings.
    int print_level = 0;
    double bound_tolerance = 1e-8;
    double bound_relaxation = 1e-8;
  };

  struct Input {
    double dt = 0.001;

    // Frame contract: wrench deviations, compliant offsets and A_b,i/J_b,i
    // must all be expressed in the same task frame as x_*_nominal_local
    // selected by hand_reference_frame. CRG does not transform wrenches/maps.

    // Estimated external wrenches at left and right wrists.
    Vector6d wrench_left = Vector6d::Zero();
    Vector6d wrench_right = Vector6d::Zero();

    // Reference / nominal wrenches.
    // For object carrying, Fz may include nominal object weight share.
    Vector6d wrench_left_ref = Vector6d::Zero();
    Vector6d wrench_right_ref = Vector6d::Zero();

    // Preferred Eq. (11) torso-orientation-to-hand maps A_b,i (6x3). Set
    // use_explicit_Ab=true when these are supplied by the caller.
    bool use_explicit_Ab = false;
    Matrix6x3d Ab_left = Matrix6x3d::Zero();
    Matrix6x3d Ab_right = Matrix6x3d::Zero();

    // Legacy 6x6 torso-to-hand Jacobians retained for main.cpp compatibility.
    // When use_explicit_Ab=false, rightCols<3>() is used as A_b,i, assuming
    // the spatial ordering [linear; angular].
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

    // Torso compliance displacement from the 3D orientation QP. The first
    // three components are always zero; tail<3>() is [roll,pitch,yaw].
    Vector6d delta_xb = Vector6d::Zero();

    // Filtered values.
    Vector6d delta_xc_left_filtered = Vector6d::Zero();
    Vector6d delta_xc_right_filtered = Vector6d::Zero();
    Vector6d delta_xb_filtered = Vector6d::Zero();

    // Hand-level motion contributed by the final torso orientation.
    Vector6d delta_x_left_torso = Vector6d::Zero();
    Vector6d delta_x_right_torso = Vector6d::Zero();

    // Arm-only residual compliance after torso allocation:
    // delta_x_arm = delta_xc_filtered - A_b * delta_xb_final.
    Vector6d delta_x_left_arm = Vector6d::Zero();
    Vector6d delta_x_right_arm = Vector6d::Zero();

    // Final torso correction sent to WBC.
    Vector6d delta_xb_final = Vector6d::Zero();

    // Discrete derivatives of the final torso and residual-arm outputs used
    // by Eqs. (20)--(21). They are zero on the first update after reset().
    Vector6d delta_dxb = Vector6d::Zero();
    Vector6d delta_ddxb = Vector6d::Zero();
    Vector6d delta_dx_left_arm = Vector6d::Zero();
    Vector6d delta_dx_right_arm = Vector6d::Zero();
    Vector6d delta_ddx_left_arm = Vector6d::Zero();
    Vector6d delta_ddx_right_arm = Vector6d::Zero();

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
    Matrix3d H = Matrix3d::Zero();
    Vector3d g = Vector3d::Zero();
    Vector3d lbx = Vector3d::Zero();
    Vector3d ubx = Vector3d::Zero();
    Matrix6x3d Ab_left = Matrix6x3d::Zero();
    Matrix6x3d Ab_right = Matrix6x3d::Zero();
    double rho_left = 0.0;
    double rho_right = 0.0;

    double objective_value = 0.0;
    double qp_solve_time_ms = 0.0;

    bool qp_solved = false;
    // Fixed storage keeps DebugInfo ABI-stable when CasADi requires the legacy
    // libstdc++ string ABI but callers use the default ABI.
    std::array<char, 256> qp_status{};

    const char* qpStatus() const { return qp_status.data(); }
  };

public:
  ComplianceReferenceGenerator();
  explicit ComplianceReferenceGenerator(const Parameters& params);

  void reset();

  void setParameters(const Parameters& params);
  const Parameters& getParameters() const;

  Output update(const Input& input);

  const DebugInfo& getDebugInfo() const;

  // Eq. (11): maps a small torso orientation offset to the corresponding
  // hand translation/orientation offset for lever arm r_b_hand.
  static Matrix6x3d makeTorsoRotationMap(
      const Eigen::Vector3d& r_b_hand);

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

  Vector3d applyTorsoOrientationBounds(
      const Vector3d& x) const;

  Vector6d firstOrderLowpass(
      const Vector6d& x,
      const Vector6d& x_prev,
      double alpha) const;

  void buildTorsoComplianceQP(
      const Vector6d& delta_xc_left,
      const Vector6d& delta_xc_right,
      const Matrix6x3d& Ab_left,
      const Matrix6x3d& Ab_right,
      Matrix3d& H,
      Vector3d& g,
      Vector3d& lbx,
      Vector3d& ubx) const;

  Vector3d solveTorsoComplianceQP(
      const Vector6d& delta_xc_left,
      const Vector6d& delta_xc_right,
      const Matrix6x3d& Ab_left,
      const Matrix6x3d& Ab_right);

  double computeObjective(
      const Vector3d& delta_xb,
      const Matrix3d& H,
      const Vector3d& g) const;

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
  bool torso_output_history_initialized_ = false;
  bool arm_output_history_initialized_ = false;

  Vector6d delta_xc_left_prev_ = Vector6d::Zero();
  Vector6d delta_xc_right_prev_ = Vector6d::Zero();
  Vector3d delta_xb_qp_prev_ = Vector3d::Zero();

  Vector6d delta_dxc_left_prev_ = Vector6d::Zero();
  Vector6d delta_dxc_right_prev_ = Vector6d::Zero();

  Vector6d delta_xc_left_filtered_prev_ = Vector6d::Zero();
  Vector6d delta_xc_right_filtered_prev_ = Vector6d::Zero();
  Vector6d delta_xb_filtered_prev_ = Vector6d::Zero();

  Vector6d delta_dxb_prev_ = Vector6d::Zero();
  Vector6d delta_x_left_arm_prev_ = Vector6d::Zero();
  Vector6d delta_x_right_arm_prev_ = Vector6d::Zero();
  Vector6d delta_dx_left_arm_prev_ = Vector6d::Zero();
  Vector6d delta_dx_right_arm_prev_ = Vector6d::Zero();
};

}  // namespace labrob

#endif  // LABROB_COMPLIANCE_REFERENCE_GENERATOR_HPP_
