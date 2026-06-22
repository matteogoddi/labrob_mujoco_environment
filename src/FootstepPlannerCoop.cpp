#include <FootstepPlannerCoop.hpp>
#include <utils.hpp>   // labrob::Rz, labrob::Rz_planar

#include <algorithm>  // std::clamp (C++17)
#include <cmath>
#include <vector>

namespace labrob {

// ============================================================
//  Costruttore
// ============================================================

FootstepPlannerCoop::FootstepPlannerCoop(const Params& params)
    : params_(params),
      support_foot_(Foot::RIGHT),
      T_lsole_planned_(SE3()),
      T_rsole_planned_(SE3()),
      integral_eh_(Eigen::Vector2d::Zero()),
      delta_theta_prev_(0.0)
{
    num_variables_        = 2 * params_.F;
    num_ineq_constraints_ = 2 * params_.F;

    H_         = Eigen::MatrixXd::Zero(num_variables_, num_variables_);
    f_         = Eigen::VectorXd::Zero(num_variables_);
    A_kin_     = Eigen::MatrixXd::Zero(num_ineq_constraints_, num_variables_);
    b_kin_min_ = Eigen::VectorXd::Zero(num_ineq_constraints_);
    b_kin_max_ = Eigen::VectorXd::Zero(num_ineq_constraints_);
    A_eq_      = Eigen::MatrixXd::Zero(0, num_variables_);
    b_eq_      = Eigen::VectorXd::Zero(0);

    buildCostMatrix();

    qp_solver_ptr_ = std::make_unique<labrob::QpSolver>(
        num_variables_,
        0,
        num_ineq_constraints_
    );
}

// ============================================================
//  init
// ============================================================

void FootstepPlannerCoop::init(
    const labrob::SE3& T_lsole,
    const labrob::SE3& T_rsole,
    labrob::Foot support_foot)
{
    T_lsole_planned_ = T_lsole;
    T_rsole_planned_ = T_rsole;
    support_foot_    = support_foot;
    resetIntegral();
}

// ============================================================
//  resetIntegral
// ============================================================

void FootstepPlannerCoop::resetIntegral()
{
    integral_eh_.setZero();
}

// ============================================================
//  buildCostMatrix  —  H = D^T D, costante
// ============================================================

void FootstepPlannerCoop::buildCostMatrix()
{
    // Costo: 1/2 * x^T H x + f^T x,  con H = D^T D
    //
    // D è la matrice delle differenze prime a blocchi 2x2.
    // Struttura di H (blocchi 2x2):
    //   diagonale j < F-1 :  2*I_2
    //   diagonale j = F-1 :    I_2
    //   off-diagonale     :   -I_2

    H_.setZero();

    for (int j = 0; j < params_.F; ++j) {
        const int idx = 2 * j;

        const Eigen::Matrix2d diag_block = (j < params_.F - 1)
            ? Eigen::Matrix2d(2.0 * Eigen::Matrix2d::Identity())
            : Eigen::Matrix2d(Eigen::Matrix2d::Identity());
        H_.block<2,2>(idx, idx) = diag_block;

        if (j > 0) {
            const int prev = 2 * (j - 1);
            H_.block<2,2>(prev, idx) = -Eigen::Matrix2d::Identity();
            H_.block<2,2>(idx, prev) = -Eigen::Matrix2d::Identity();
        }
    }
}


// ============================================================
//  updateErrorIntegral
// ============================================================

Eigen::Vector2d FootstepPlannerCoop::updateErrorIntegral(
    const Eigen::Vector2d& e_h,
    double dt)
{
    // Update integral with forward Euler integration + anti-windup via clamping
    integral_eh_.x() = std::clamp(
        integral_eh_.x() + e_h.x() * dt,
        -params_.integral_cap_x, +params_.integral_cap_x
    );

    integral_eh_.y() = std::clamp(
        integral_eh_.y() + e_h.y() * dt,
        -params_.integral_cap_y, +params_.integral_cap_y
    );

    return integral_eh_;

}


// ============================================================
//  computeDeltaP
// ============================================================

Eigen::Vector2d FootstepPlannerCoop::computeDeltaP(
    const Eigen::Vector2d& e_h,
    const Eigen::Vector2d& e_h_dot,
    const Eigen::Vector2d& integral_eh
    )
{
    
    return Eigen::Vector2d(
        -(params_.kp_x * e_h.x() + params_.kd_x * e_h_dot.x() + params_.ki_x * integral_eh.x()),
        -(params_.kp_y * e_h.y() + params_.kd_y * e_h_dot.y() + params_.ki_y * integral_eh.y())
    );
}

// ============================================================
//  computeDeltaTheta
// ============================================================

double FootstepPlannerCoop::computeDeltaTheta(
    const Eigen::Vector2d& r_l_F_xy,
    const Eigen::Vector2d& r_r_F_xy)
{
    const Eigen::Vector2d diff = r_r_F_xy - r_l_F_xy;

    double delta_theta;

    if (diff.norm() < params_.transportation_axis_deadband) {
        // Caso degenere: proiezioni coincidenti.
        // L'asse di trasporto è la retta dall'origine al punto medio.
        const Eigen::Vector2d midpoint = 0.5 * (r_l_F_xy + r_r_F_xy);

        if (midpoint.norm() < params_.transportation_axis_deadband) {
            // Mani all'origine di F: nessuna informazione angolare.
            // Congela l'ultimo valore valido.
            delta_theta = delta_theta_prev_;
        } else {
            delta_theta = std::atan2(midpoint.y(), midpoint.x());
        }
    } else {
        // Caso regolare: asse di trasporto perpendicolare al segmento r_l -- r_r.
        // t = (-diff.y, diff.x)  →  Delta_theta = atan2(t.y, t.x)
        const Eigen::Vector2d t(-diff.y(), diff.x());
        delta_theta = std::atan2(t.y(), t.x());
    }
    delta_theta = std::clamp(delta_theta, -M_PI / 6.0, M_PI / 6.0);  // ±30°
    delta_theta_prev_ = delta_theta;
    return delta_theta;
}

// ============================================================
//  updateQPMatrices
// ============================================================

void FootstepPlannerCoop::updateQPMatrices(
    const Eigen::Vector2d& delta_p,
    double delta_theta,
    const Eigen::Matrix3d& R_F)
{
    // Rotazioni 2x2:
    //   R_1     = R_F_2x2          (primo passo, allineato al frame corrente)
    //   R_j     = R_F_2x2 * R_dth  (passi j>=2, tutti uguali)
    const Eigen::Matrix2d R_F_2x2  = R_F.topLeftCorner<2,2>();
    const Eigen::Matrix2d R_dth    = labrob::Rz_planar<double>(delta_theta);
    const Eigen::Matrix2d R_j_rest = R_F_2x2 * R_dth;

    // p0: posa misurata del piede di supporto (già ancorata in computeNextSteps)
    const Eigen::Vector2d p0 = (support_foot_ == Foot::LEFT)
        ? T_lsole_planned_.translation().head<2>()
        : T_rsole_planned_.translation().head<2>();

    // Precalcolo n_j = R_j * (delta_p + (0, s_j*ell))
    // s_j = +1 se oscillante a sinistra, -1 se a destra
    // Alternanza (j 0-indexed):
    //   j pari   → swing opposto al supporto corrente
    //   j dispari → swing uguale al supporto corrente
    std::vector<Eigen::Vector2d> n(params_.F);

    for (int j = 0; j < params_.F; ++j) {
        const bool swing_is_left = (j % 2 == 0)
            ? (support_foot_ == Foot::RIGHT)
            : (support_foot_ == Foot::LEFT);
        const double s = swing_is_left ? +1.0 : -1.0;

        const Eigen::Matrix2d& Rj = (j == 0) ? R_F_2x2 : R_j_rest;
        n[j] = Rj * (delta_p + Eigen::Vector2d(0.0, s * params_.ell));
    }

    // -----------------------------------------------------------
    //  Gradiente lineare  f = D^T * (v + n_vec)
    //
    //  w_j = (v + n_vec)_j:
    //    w_0 = -p0 - n[0]
    //    w_j = -n[j]    j > 0
    //
    //  f_j = w_j - w_{j+1}   per j < F-1
    //  f_j = w_j              per j = F-1
    // -----------------------------------------------------------

    f_.setZero();

    for (int j = 0; j < params_.F; ++j) {
        const Eigen::Vector2d w_j = (j == 0)
            ? Eigen::Vector2d(-p0 - n[0])
            : Eigen::Vector2d(-n[j]);

        if (j < params_.F - 1) {
            f_.segment<2>(2 * j) = w_j - (-n[j + 1]);
        } else {
            f_.segment<2>(2 * j) = w_j;
        }
    }

    // -----------------------------------------------------------
    //  Vincoli cinematici:  d_min^j <= R_j^T (p_f^j - p_f^{j-1}) <= d_max^j
    //
    //  A_kin_ riga j:  +R_j^T sul blocco p_f^j
    //                  -R_j^T sul blocco p_f^{j-1}  (j > 0)
    //
    //  Per j=0: p_f^0 = p0 costante → assorbita nei bound:
    //    b_kin_{min,max}[0] = d_{min,max}^0 + R_1^T * p0
    //
    //  Bound nominali:
    //    d_min^j = (-da_x/2,  -da_y/2 + s_j*ell)
    //    d_max^j = (+da_x/2,  +da_y/2 + s_j*ell)
    // -----------------------------------------------------------

    A_kin_.setZero();

    for (int j = 0; j < params_.F; ++j) {
        const int row = 2 * j;
        const int col = 2 * j;

        const Eigen::Matrix2d& Rj  = (j == 0) ? R_F_2x2 : R_j_rest;
        const Eigen::Matrix2d  RjT = Rj.transpose();

        A_kin_.block<2,2>(row, col) = RjT;

        if (j > 0) {
            A_kin_.block<2,2>(row, col - 2) = -RjT;
        }

        const bool swing_is_left = (j % 2 == 0)
            ? (support_foot_ == Foot::RIGHT)
            : (support_foot_ == Foot::LEFT);
        const double s = swing_is_left ? +1.0 : -1.0;

        const Eigen::Vector2d d_min_j(-params_.da_x / 2.0,
                                      -params_.da_y / 2.0 + s * params_.ell);
        const Eigen::Vector2d d_max_j(+params_.da_x / 2.0,
                                      +params_.da_y / 2.0 + s * params_.ell);

        if (j == 0) {
            const Eigen::Vector2d correction = RjT * p0;
            b_kin_min_.segment<2>(row) = d_min_j + correction;
            b_kin_max_.segment<2>(row) = d_max_j + correction;
        } else {
            b_kin_min_.segment<2>(row) = d_min_j;
            b_kin_max_.segment<2>(row) = d_max_j;
        }
    }
}

// ============================================================
//  buildPlan
// ============================================================

std::vector<labrob::FootstepPlanElement> FootstepPlannerCoop::buildPlan(
    const Eigen::VectorXd& qp_solution,
    double delta_theta,
    double lsole_z,
    double rsole_z)
{
    std::vector<labrob::FootstepPlanElement> plan;
    plan.reserve(2 * params_.F);

    // Orientazione comune a tutti i passi:
    //   passo 1 → yaw supporto corrente + delta_theta
    //   passi 2..F → stessa orientazione del primo  (paper, Section V)
    const SE3& current_support_pose = (support_foot_ == Foot::LEFT)
        ? T_lsole_planned_ : T_rsole_planned_;
    const double current_yaw = std::atan2(
        current_support_pose.rotation()(1, 0),
        current_support_pose.rotation()(0, 0));
    const Eigen::Matrix3d R_step = labrob::Rz<double>(current_yaw + delta_theta);

    // Durate DS e SS [ms]
    const int64_t T_ds = static_cast<int64_t>(
        params_.T_step_ms * params_.double_support_ratio);
    const int64_t T_ss = static_cast<int64_t>(
        params_.T_step_ms * (1.0 - params_.double_support_ratio));

    for (int j = 0; j < params_.F; ++j) {

        // Il piede oscillante è sempre l'opposto del supporto corrente.
        // support_foot_ viene aggiornato alla fine di ogni iterazione.
        const Foot swing_foot  = (support_foot_ == Foot::RIGHT)
            ? Foot::LEFT : Foot::RIGHT;
        const Foot support_foot = support_foot_;

        // Salva posizioni PRIMA dell'aggiornamento
        const SE3 T_lsole_start = T_lsole_planned_;
        const SE3 T_rsole_start = T_rsole_planned_;

        // Posa target del piede oscillante (posizione dal QP, z=0)
        const Eigen::Vector2d pos_2d = qp_solution.segment<2>(2 * j);
        const double target_z = (swing_foot == Foot::LEFT) ? lsole_z : rsole_z;
        const SE3 new_swing_pose(
            R_step,
            Eigen::Vector3d(pos_2d.x(), pos_2d.y(), target_z)
        );

        // DS: pose dei piedi PRIMA dello swing
        plan.push_back(FootstepPlanElement(
            DoubleSupportConfiguration(
                T_lsole_planned_,
                T_rsole_planned_,
                support_foot),
            0.0,
            T_ds,
            WalkingState::DoubleSupport
        ));

        // Aggiorna la posa pianificata del piede oscillante
        if (swing_foot == Foot::LEFT) {
            T_lsole_planned_ = new_swing_pose;
        } else {
            T_rsole_planned_ = new_swing_pose;
        }

        // SS: pose dei piedi DOPO lo swing (oscillante alla posizione target)
        plan.push_back(FootstepPlanElement(
            DoubleSupportConfiguration(
                T_lsole_start,
                T_rsole_start,
                support_foot),
            params_.step_height,
            T_ss,
            WalkingState::SingleSupport
        ));

        // Il piede oscillante diventa il nuovo supporto
        support_foot_ = swing_foot;
    }

    return plan;
}



// ============================================================
//  computeNextSteps
// ============================================================

labrob::FootstepPlannerCoop::QP2DResult
FootstepPlannerCoop::computeNextSteps(
    const labrob::SE3& T_lsole,
    const labrob::SE3& T_rsole,
    labrob::Foot current_support_foot,
    const Eigen::Vector2d& e_h,
    const Eigen::Vector2d& e_h_dot,
    const Eigen::Vector2d& r_l_F_xy,
    const Eigen::Vector2d& r_r_F_xy,
    const Eigen::Matrix3d& R_F,
    double dt)
{
    support_foot_ = current_support_foot;

    // Ancora entrambi i piedi alla misura corrente
    T_lsole_planned_ = T_lsole;
    T_rsole_planned_ = T_rsole;
    
    /*if (support_foot_ == Foot::LEFT) {
        T_lsole_planned_ = T_lsole;
    } else {
        T_rsole_planned_ = T_rsole;
    }*/

    const Eigen::Vector2d delta_p     = computeDeltaP(e_h, e_h_dot, integral_eh_);
    const double          delta_theta = computeDeltaTheta(r_l_F_xy, r_r_F_xy);

    last_delta_p_     = delta_p;
    delta_theta_prev_ = delta_theta;

    updateQPMatrices(delta_p, delta_theta, R_F);
    qp_solver_ptr_->solve(H_, f_, A_eq_, b_eq_, A_kin_, b_kin_min_, b_kin_max_);

    // Target z is always the z of the floor
    const double floor_z = 0.007;
    /*(current_support_foot == Foot::LEFT)
    ? T_lsole.translation().z()
    : T_rsole.translation().z();*/


    QP2DResult result;
    result.F                 = params_.F;
    result.qp_solution       = qp_solver_ptr_->get_solution();
    result.delta_theta       = delta_theta;
    result.delta_p           = delta_p;
    result.support_foot_start = support_foot_;
    result.lsole_z           = floor_z;
    result.rsole_z           = floor_z;
    return result;
}

// ============================================================
//  buildDequeElements
// ============================================================

std::vector<labrob::FootstepPlanElement>
FootstepPlannerCoop::buildDequeElements(const QP2DResult& result)
{
    return buildPlan(result.qp_solution, result.delta_theta,
                     result.lsole_z, result.rsole_z);
}

} // end namespace labrob