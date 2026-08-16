#include <HandAdmittanceController.hpp>
#include <iostream>

namespace labrob {

HandAdmittanceController::HandAdmittanceController(
    double dt,
    const Eigen::Vector3d& mass_diag,
    const Eigen::Vector3d& damping_diag,
    const Eigen::Vector3d& stiffness_diag,
    const Eigen::Vector3d& r_l_bar,
    const Eigen::Vector3d& r_r_bar,
    const Eigen::Vector3d& f_l_bar_W,
    const Eigen::Vector3d& f_r_bar_W,
    const double eh_threshold,
    const int below_threshold_limit,
    const int below_threshold_counter
)
    : dt_(dt),
      M_inv_(mass_diag.array().inverse()),
      C_(damping_diag.array()),
      K_(stiffness_diag.array()),
      r_l_bar_(r_l_bar),
      r_r_bar_(r_r_bar),
      f_l_bar_W_(f_l_bar_W),
      f_r_bar_W_(f_r_bar_W),
      eh_threshold_(eh_threshold),
      below_threshold_limit_(below_threshold_limit),
      below_threshold_counter_(below_threshold_counter),
      // Initialize state at the rest position (zero velocity)
      r_l_F_(r_l_bar),
      v_l_F_(Eigen::Vector3d::Zero()),
      r_r_F_(r_r_bar),
      v_r_F_(Eigen::Vector3d::Zero()),
      // Initialize cached frame to identity / zero origin
      p_F_(Eigen::Vector3d::Zero()),
      R_F_(Eigen::Matrix3d::Identity())
{}

// ---------------------------------------------------------------------------

Eigen::Vector3d HandAdmittanceController::world2local(
    const Eigen::Vector3d& v_W,
    const Eigen::Matrix3d& R_F
) {
    return R_F.transpose() * v_W;
}


void HandAdmittanceController::integrate(
    const Eigen::Vector3d& f_l_W,
    const Eigen::Vector3d& f_r_W,
    const Eigen::Vector3d& p_F,
    const Eigen::Matrix3d& R_F,
    const Eigen::Vector3d& v_F
) {
    // Map force errors from world frame to F frame:
    //   fe_i = R_F^T * (f_i - f_i_bar)
    const Eigen::Vector3d fe_l = world2local(f_l_W - f_l_bar_W_, R_F);
    const Eigen::Vector3d fe_r = world2local(f_r_W - f_r_bar_W_, R_F);

    // Compute accelerations in F:
    //   a_i = M^{-1} * [ fe_i  -  C * v_i  -  K * (r_i - r_i_bar) ]
    //
    // Note: .array() enables element-wise multiplication with Array3d;
    //       .matrix() converts the result back to Vector3d.
    const Eigen::Vector3d a_l = (M_inv_ * (
        fe_l.array() - C_ * v_l_F_.array() - K_ * (r_l_F_ - r_l_bar_).array()
    )).matrix();

    const Eigen::Vector3d a_r = (M_inv_ * (
        fe_r.array() - C_ * v_r_F_.array() - K_ * (r_r_F_ - r_r_bar_).array()
    )).matrix();

    // Semi-implicit (symplectic) Euler integration:
    //   velocity updated first, then position uses the NEW velocity.
    //   More stable than explicit Euler for spring-damper systems.
    v_l_F_ += dt_ * a_l;
    r_l_F_ += dt_ * v_l_F_;

    v_r_F_ += dt_ * a_r;
    r_r_F_ += dt_ * v_r_F_;

    // Cache the current frame for getters and onSupportSwitch
    p_F_ = p_F;
    R_F_ = R_F;
    v_F_ = v_F;
}

// ---------------------------------------------------------------------------

void HandAdmittanceController::onSupportSwitch(
    const Eigen::Vector3d& p_F_new,
    const Eigen::Matrix3d& R_F_new
) {
    // 1. Convert current hands positions from old F to world frame to keep continuity in the world space
    const Eigen::Vector3d r_l_W = p_F_ + R_F_ * r_l_F_;
    const Eigen::Vector3d r_r_W = p_F_ + R_F_ * r_r_F_;

    // 2. Express hands positions in new F frame:
    r_l_F_ = world2local(r_l_W - p_F_new, R_F_new);
    r_r_F_ = world2local(r_r_W - p_F_new, R_F_new);

    // 3. Rotate velocity: only the orientation of F changes (no translational
    //    velocity jump), so we simply apply the relative rotation.
    const Eigen::Matrix3d R_rel = R_F_new.transpose() * R_F_;       // old F --> world --> new F
    v_l_F_ = R_rel * v_l_F_;
    v_r_F_ = R_rel * v_r_F_;

    // Re-map rest positions to the new local frame F
    r_l_bar_ = R_rel * r_l_bar_;
    r_r_bar_ = R_rel * r_r_bar_;

    // Re-symmetrize: a z-rotation does not commute with the L/R mirror reflection,
    // so R_rel*r_bar_ can pick up a spurious x-offset between r_l_bar_/r_r_bar_
    // even though the physical (world-frame) rest posture is unchanged by this
    // bookkeeping transform. Left uncorrected, this reintroduces a delta_theta
    // bias at every support switch.
    const double x_mid  = 0.5 * (r_l_bar_.x() + r_r_bar_.x());
    const double y_half = 0.5 * (r_l_bar_.y() - r_r_bar_.y());
    r_l_bar_.x() = x_mid;   r_r_bar_.x() = x_mid;
    r_l_bar_.y() = y_half;  r_r_bar_.y() = -y_half;

    // 4. Update cached frame
    p_F_ = p_F_new;
    R_F_ = R_F_new;
}

// ---------------------------------------------------------------------------

void HandAdmittanceController::reset(
    const Eigen::Vector3d& r_l_W,
    const Eigen::Vector3d& r_r_W,
    const Eigen::Vector3d& p_F,
    const Eigen::Matrix3d& R_F
) {
    r_l_F_ = world2local(r_l_W - p_F, R_F);
    r_r_F_ = world2local(r_r_W - p_F, R_F);



    // Added to correctly reset rest positions after posture regulation
    r_l_bar_ = r_l_F_;   // re-anchor rest position to current pose
    r_r_bar_ = r_r_F_;



    
    v_l_F_ = Eigen::Vector3d::Zero();
    v_r_F_ = Eigen::Vector3d::Zero();
    p_F_   = p_F;
    R_F_   = R_F;
}

// ---------------------------------------------------------------------------
// Getters
// ---------------------------------------------------------------------------

// Return left hand position reference in world frame
Eigen::Vector3d HandAdmittanceController::getLeftHandRef() const {
    return p_F_ + R_F_ * r_l_F_;
}

// Return right hand position reference in world frame
Eigen::Vector3d HandAdmittanceController::getRightHandRef() const {
    return p_F_ + R_F_ * r_r_F_;
}

// Return left hand velocity reference in world frame
Eigen::Vector3d HandAdmittanceController::getLeftHandVelRef() const {
    return v_F_ + R_F_ * v_l_F_;
}

// Return right hand velocity reference in world frame
Eigen::Vector3d HandAdmittanceController::getRightHandVelRef() const {
    return v_F_ + R_F_ * v_r_F_;
}

// Return hand position error in local frame F
Eigen::Vector2d HandAdmittanceController::getEh() const {
    // e_h = 0.5 * [(r_l_bar - r_l^F)^xy + (r_r_bar - r_r^F)^xy]
    return 0.5 * (
        (r_l_bar_ - r_l_F_).head<2>() +
        (r_r_bar_ - r_r_F_).head<2>()
    );
}

// Return hand position error rate in local frame F
Eigen::Vector2d HandAdmittanceController::getEhDot() const {
    // e_h_dot = -0.5 * (v_l^F_xy + v_r^F_xy)
    return -0.5 * (v_l_F_.head<2>() + v_r_F_.head<2>());
}

// Check if Average Hand Error is above threshold
bool HandAdmittanceController::isEhAboveThreshold() const {
    return getEh().norm() > eh_threshold_;
}

// Stop request: Average Hand Error below threshold for long enough
bool HandAdmittanceController::stoppingRequested() {

    if (!isEhAboveThreshold()) {
        below_threshold_counter_ += 1; // increment counter
        std::cout << "Counter state: " << below_threshold_counter_ << std::endl;
    } else {
        below_threshold_counter_ = 0;
        return false;}

    if (below_threshold_counter_ < below_threshold_limit_) {
        return false;
    } else {
        below_threshold_counter_ = 0; // reset counter
        return true;
    }

}

// Return left hand position in F frame
Eigen::Vector2d HandAdmittanceController::getLeftHandPos() const {
    return r_l_F_.head<2>();
}
// Return right hand position in F frame
Eigen::Vector2d HandAdmittanceController::getRightHandPos() const {
    return r_r_F_.head<2>();
}

} // end namespace labrob