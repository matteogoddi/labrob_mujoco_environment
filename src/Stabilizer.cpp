#include <Stabilizer.hpp>
#include <algorithm>
#include <cmath>

namespace labrob {

    // Constructor
    Stabilizer::Stabilizer(
        double Kp,
        double Kd,
        double Ki,
        double integral_bound_x,
        double integral_bound_y,
        double integral_bound_z,
        double mb_half_x,
        double mb_half_y,
        const labrob::DiscreteLIPDynamics& lip_model)
        : lip_model_(lip_model)
    {
        params_.Kp_ = Kp;
        params_.Kd_ = Kd;
        params_.Ki_ = Ki;
        params_.integral_bound_x_ = integral_bound_x;
        params_.integral_bound_y_ = integral_bound_y;
        params_.integral_bound_z_ = integral_bound_z;
        params_.mb_x_ = mb_half_x;
        params_.mb_y_ = mb_half_y;
        init();
    }


    // Initialization
    void Stabilizer::init() {
        reset_integral();
        box_center_.setZero();
        box_rotation_.setIdentity();
        saturated_ = false;
    }


    // ### CORRECTION ###
    Eigen::Vector3d Stabilizer::compute_correction(
        const LIPState& current_state,
        const Eigen::Vector3d& current_zmp,
        const LIPState& target_state,
        const Eigen::Vector3d& target_zmp,
        double dt)
    {
        // Current eta from the model (frozen for this control step)
        const double eta = lip_model_.getEta();

        // Compute current DCM
        Eigen::Vector3d current_dcm = current_state.com_pos_ + 1/eta * current_state.com_vel_;

        // Compute target DCM
        Eigen::Vector3d desired_dcm = target_state.com_pos_ + 1/eta * target_state.com_vel_;

        // Compute DCM error
        Eigen::Vector3d dcm_error = desired_dcm - current_dcm;

        // Compute ZMP error
        Eigen::Vector3d zmp_error = target_zmp - current_zmp;

        // Update integral term
        update_integral(dcm_error, dt);

        // Compute correction
        Eigen::Vector3d correction = -(1 + params_.Kp_/eta + params_.Kd_) * dcm_error
        + params_.Kd_* zmp_error
        - (params_.Ki_ / eta) * integral_;
        return correction;
    }

    // ###  INTEGRAL ###
    // Update integral term
    void Stabilizer::update_integral(const Eigen::Vector3d& error, double dt) {

        // Keep the previous value so that the step can be undone (hold_integral)
        integral_prev_ = integral_;

        // Cumulate and saturate (anti-windup)
        integral_.x() = std::clamp(
            integral_.x() + error.x() * dt,
            -params_.integral_bound_x_, +params_.integral_bound_x_
        );

        integral_.y() = std::clamp(
            integral_.y() + error.y() * dt,
            -params_.integral_bound_y_, +params_.integral_bound_y_
        );

        integral_.z() = std::clamp(
            integral_.z() + error.z() * dt,
            -params_.integral_bound_z_, +params_.integral_bound_z_
        );

    }

    // Reset integral term
    void Stabilizer::reset_integral() {
        integral_.setZero();
        integral_prev_.setZero();
    }

    // Conditional-integration anti-windup: undo the last update_integral() step.
    // To be called when the corrected ZMP had to be saturated onto the moving box,
    // so the integral does not keep growing while its effect is clipped away.
    void Stabilizer::hold_integral() {
        integral_ = integral_prev_;
    }


    // ### SATURATION ###
    // Update the moving box state (center and rotation box frame -> world)
    void Stabilizer::update_mb_state(const Eigen::Vector3d& box_center, double box_yaw) {
        box_center_ = box_center.head<2>();
        const double c = std::cos(box_yaw);
        const double s = std::sin(box_yaw);
        box_rotation_ << c, -s,
                         s,  c;
    }

    // Saturation: Euclidean projection of the ZMP onto the moving box.
    // In the box frame the box is the axis-aligned rectangle [-mb_x, mb_x] x [-mb_y, mb_y],
    // so the closest admissible point is the component-wise clamp:
    //  - only |x_b| > mb_x  -> orthogonal projection onto the front/back side
    //  - only |y_b| > mb_y  -> orthogonal projection onto the left/right side
    //  - both               -> closest vertex
    //  - inside             -> unchanged
    Eigen::Vector3d Stabilizer::saturate(const Eigen::Vector3d& zmp_world, const Eigen::Vector3d& box_center, double box_yaw) {

        update_mb_state(box_center, box_yaw);

        // ZMP in the box frame
        const Eigen::Vector2d zmp_box = box_rotation_.transpose() * (zmp_world.head<2>() - box_center_);

        // Closest point of the box (box frame)
        const Eigen::Vector2d zmp_box_sat(
            std::clamp(zmp_box.x(), -params_.mb_x_, params_.mb_x_),
            std::clamp(zmp_box.y(), -params_.mb_y_, params_.mb_y_));

        saturated_ = (zmp_box_sat != zmp_box);

        // Back to world frame (z unchanged)
        Eigen::Vector3d saturated_zmp = zmp_world;
        saturated_zmp.head<2>() = box_center_ + box_rotation_ * zmp_box_sat;
        return saturated_zmp;
    }

} // namespace labrob
