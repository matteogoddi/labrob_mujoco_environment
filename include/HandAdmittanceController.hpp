#ifndef HAND_ADMITTANCE_CONTROLLER_HPP
#define HAND_ADMITTANCE_CONTROLLER_HPP

#include <Eigen/Core>

namespace labrob {

/**
 * Hand Admittance Controller
 *
 * Implements the mass-spring-damper admittance model from Humanoids24:
 *
 *   M r_i'' + C r_i' + K (r_i - r_i_bar) = R_F^T (f_i - f_i_bar)
 *
 * where r_i is the position of hand i expressed in the local frame F.
 * The state (r_i, r_i') is integrated with semi-implicit Euler at each tick.
 *
 * Outputs:
 *   - getLeftHandRef() / getRightHandRef()     -> reference positions (world) for WBC
 *   - getLeftHandVelRef() / getRightHandVelRef() -> reference velocities (world) for WBC
 *   - getEh() / getEhDot()                     -> average hand error for footstep planner
 */
class HandAdmittanceController {
public:
    /**
     * @param dt              Control timestep [s]
     * @param mass_diag       Diagonal of mass matrix M [kg]
     * @param damping_diag    Diagonal of damping matrix C [N·s/m]
     * @param stiffness_diag  Diagonal of stiffness matrix K [N/m]
     * @param r_l_bar         Rest position of left hand in F frame [m]
     * @param r_r_bar         Rest position of right hand in F frame [m]
     * @param f_l_bar_W       Rest force on left hand in WORLD frame [N]
     *                        (typically (0,0, fz_bar/2) = quarter of object weight)
     * @param f_r_bar_W       Rest force on right hand in WORLD frame [N]
     */
    HandAdmittanceController(
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
    );

    
    /**
    Map a free vector in the world frame into the local frame F
    *
    * @param v_W Vector expressed in the world frame
    * @param R_F Orientation of F w.r.t. world (columns = F axes in world)
    * @return v_F Vector expressed in the local frame F
    */
    static Eigen::Vector3d world2local(
        const Eigen::Vector3d& v_W,
        const Eigen::Matrix3d& R_F
    );

    /**
     * Integrate one control step.
     * Forces are expressed in world frame; the frame F pose is passed so that
     * the force error can be mapped to F and for use by the output getters.
     *
     * @param f_l_W  Measured force on left hand (world frame) [N]
     * @param f_r_W  Measured force on right hand (world frame) [N]
     * @param p_F    Origin of F expressed in world [m]
     * @param R_F    Orientation of F w.r.t. world (columns = F axes in world)
     */
    void integrate(
        const Eigen::Vector3d& f_l_W,
        const Eigen::Vector3d& f_r_W,
        const Eigen::Vector3d& p_F,
        const Eigen::Matrix3d& R_F
    );

    /**
     * Call when the support foot changes (phase transition).
     * Re-expresses the internal state from the old F frame to the new one,
     * preserving continuity of the world-frame reference position.
     *
     * @param p_F_new  New origin of F in world
     * @param R_F_new  New orientation of F
     */
    void onSupportSwitch(
        const Eigen::Vector3d& p_F_new,
        const Eigen::Matrix3d& R_F_new
    );

    
    /**
     * Hard reset of the admittance state.
     * Use at standing->walking transition or on first activation.
     * Sets r_i^F from measured hand positions; velocities set to zero.
     *
     * @param r_l_W  Measured left hand position (world frame)
     * @param r_r_W  Measured right hand position (world frame)
     * @param p_F    Current frame F origin (world)
     * @param R_F    Current frame F orientation
     */
    void reset(
        const Eigen::Vector3d& r_l_W,
        const Eigen::Vector3d& r_r_W,
        const Eigen::Vector3d& p_F,
        const Eigen::Matrix3d& R_F
    );

    // --- Outputs for WBC (world frame) ---

    Eigen::Vector3d getLeftHandRef()     const;  ///< r_l*   (world)
    Eigen::Vector3d getLeftHandVelRef()  const;  ///< r_l'*  (world)
    Eigen::Vector3d getRightHandRef()    const;  ///< r_r*   (world)
    Eigen::Vector3d getRightHandVelRef() const;  ///< r_r'*  (world)

    // --- Outputs for footstep planner (F frame, xy only) ---

    /**
     * Average horizontal hand error (in F frame):
     *   e_h = 0.5 * [(r_l_bar - r_l)^xy + (r_r_bar - r_r)^xy]
     */
    Eigen::Vector2d getEh()    const;

    /**
     * Derivative of e_h:
     *   e_h_dot = -0.5 * (r_l'^xy + r_r'^xy)
     */
    Eigen::Vector2d getEhDot() const;

    bool isEhAboveThreshold() const;
    bool stoppingRequested();

    /**
     * Get current hands x-y position in F frame
     */
    Eigen::Vector2d getLeftHandPos() const;
    Eigen::Vector2d getRightHandPos() const;

private:
    double dt_;

    // Mass/damping/stiffness stored as element-wise arrays for efficiency
    Eigen::Array3d M_inv_;   ///< element-wise inverse of mass diagonal
    Eigen::Array3d C_;       ///< damping diagonal
    Eigen::Array3d K_;       ///< stiffness diagonal

    Eigen::Vector3d r_l_bar_;   ///< rest position left hand in F frame
    Eigen::Vector3d r_r_bar_;   ///< rest position right hand in F frame
    Eigen::Vector3d f_l_bar_W_; ///< rest force left hand in world frame
    Eigen::Vector3d f_r_bar_W_; ///< rest force right hand in world frame

    // Internal state in F frame
    Eigen::Vector3d r_l_F_;  ///< position left hand in F
    Eigen::Vector3d v_l_F_;  ///< velocity left hand in F
    Eigen::Vector3d r_r_F_;  ///< position right hand in F
    Eigen::Vector3d v_r_F_;  ///< velocity right hand in F

    // Cached frame from last integrate() call (needed by getters and onSupportSwitch)
    Eigen::Vector3d p_F_;
    Eigen::Matrix3d R_F_;

    double eh_threshold_;
    int below_threshold_counter_;
    int below_threshold_limit_;
};

} // end namespace labrob

#endif // HAND_ADMITTANCE_CONTROLLER_HPP