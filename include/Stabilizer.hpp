#ifndef LABROB_STABILIZER_HPP_
#define LABROB_STABILIZER_HPP_

#include <Eigen/Dense>
#include <LIPState.hpp>
#include <DiscreteLIPDynamics.hpp>

namespace labrob {

    // DCM-based stabilizer: computes a correction to be added to the
    // (feedforward) ZMP produced by the open-loop IS-MPC.
    //
    //   correction = -(1 + Kp/eta + Kd) * e_dcm + Kd * e_zmp - (Ki/eta) * int(e_dcm)
    //   e_dcm = dcm_des - dcm,   e_zmp = zmp_des - zmp_meas
    //
    // eta is read from the LIP model at every call (it changes with PLIP).
    //
    // The corrected ZMP can be saturated onto the moving box (the same
    // admissible region used by the IS-MPC ZMP constraint): a rectangle
    // centered in c with yaw theta and half-extents (mb_x, mb_y) along the
    // box-frame axes (x sagittal, y lateral).
    class Stabilizer {

        public:

            // Constructor
            Stabilizer(
                double Kp,
                double Kd,
                double Ki,
                double integral_bound_x,   ///< [m*s]
                double integral_bound_y,   ///< [m*s]
                double integral_bound_z,   ///< [m*s]
                double mb_half_x,          ///< [m] moving-box half-length (sagittal)
                double mb_half_y,          ///< [m] moving-box half-width  (lateral)
                const labrob::DiscreteLIPDynamics& lip_model);

            // Initialization
            void init();

            // Compute correction
            Eigen::Vector3d compute_correction(
                const LIPState& current_state,
                const Eigen::Vector3d& current_zmp,
                const LIPState& target_state,
                const Eigen::Vector3d& target_zmp,
                double dt);

            // Integral term
            void update_integral(const Eigen::Vector3d& error, double dt);
            void reset_integral();
            void hold_integral();   // undo last integral step (anti-windup on ZMP saturation)
            const Eigen::Vector3d& get_integral() const { return integral_; }

            // Saturate final ZMP to preserve admissibility (moving box).
            // Returns the closest point of the box to zmp_world (Euclidean
            // projection, computed in the box frame); z is left unchanged.
            Eigen::Vector3d saturate(const Eigen::Vector3d& zmp_world,
                                     const Eigen::Vector3d& box_center,
                                     double box_yaw);
            bool is_saturated() const { return saturated_; }

        private:

            void update_mb_state(const Eigen::Vector3d& box_center, double box_yaw);

            struct Params {
                double Kp_;
                double Kd_;
                double Ki_;

                double integral_bound_x_;  ///< [m*s]
                double integral_bound_y_;  ///< [m*s]
                double integral_bound_z_;  ///< [m*s]

                double mb_x_;              ///< [m] moving-box half-length
                double mb_y_;              ///< [m] moving-box half-width
            } params_;


            const labrob::DiscreteLIPDynamics& lip_model_;
            Eigen::Vector3d integral_ = Eigen::Vector3d::Zero();
            Eigen::Vector3d integral_prev_ = Eigen::Vector3d::Zero(); // value before last update

            Eigen::Vector2d box_center_;   // center of the moving box (world, xy)
            Eigen::Matrix2d box_rotation_; // R(yaw): box frame -> world
            bool saturated_ = false;       // true if the last saturate() clipped the ZMP

    };

} // namespace labrob



#endif // LABROB_STABILIZER_HPP_
