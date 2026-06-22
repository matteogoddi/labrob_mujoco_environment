#ifndef LABROB_FOOTSTEP_PLANNER_COOP_HPP
#define LABROB_FOOTSTEP_PLANNER_COOP_HPP

#include <vector>
#include <memory>

#include <Eigen/Core>

#include <FootstepPlanElement.hpp>
#include <HandAdmittanceController.hpp>
#include <SE3.hpp>
#include <WalkingState.hpp>

#include <QpSolver.hpp>

namespace labrob {

/**
 * FootstepPlannerCoop
 *
 * Online footstep planner per trasporto cooperativo (ruolo follower).
 *
 * Ad ogni transizione di fase, calcola le posizioni dei prossimi F passi
 * risolvendo un QP. La funzione costo penalizza la deviazione dal
 * posizionamento nominale indotto da (Delta_p, Delta_theta), dove:
 *
 *   Delta_p     = PID su e_h  (errore medio mani, frame F, xy)
 *   Delta_theta = orientazione dell'asse di trasporto nel frame F
 *
 * I vincoli cinematici garantiscono che ogni passo sia raggiungibile.
 * I vincoli di obstacle avoidance (non lineari) non sono implementati
 * in questa versione (solo simulazione senza ostacoli).
 *
 * Output: vettore di FootstepPlanElement (coppie DS+SS) da
 * push_back nel deque di WalkingData.
 */
class FootstepPlannerCoop {
 public:

  /**
   * Parametri del planner.
   * Corrispondono alle quantità indicate in Section V di Humanoids24.
   */
  struct Params {
    int    F = 10;                      ///< numero di passi nell'orizzonte del piano
    double T_step_ms = 600.0;          ///< durata nominale di un passo (DS+SS) [ms]
    double double_support_ratio = 0.5; ///< frazione di T_step dedicata alla fase DS
    double step_height = 0.06;         ///< altezza massima del piede oscillante [m]
    double ell = 0.1;                  ///< semidistanza laterale nominale tra i piedi [m]

    // Guadagni PID per Delta_p
    double kp_x = 1.0;   double kp_y = 1.2;
    double kd_x = 0.1;   double kd_y = 0.15;
    double ki_x = 0.1;   double ki_y = 0.05;

    // Saturazione dell'integrale (anti-windup)
    double integral_cap_x = 0.5;  ///< [m]
    double integral_cap_y = 0.5;  ///< [m]

    // Dimensioni regione ammissibile cinematica [m]
    double da_x = 0.4;
    double da_y = 0.1;

    // Dead-band su ||r_l^xy - r_r^xy|| sotto cui congelare Delta_theta [m]
    double transportation_axis_deadband = 0.01;
  };

  explicit FootstepPlannerCoop(const Params& params);

  // Struct containing 2D positions of the F footsteps found by the QP solver
  struct QP2DResult {
    int F;                             ///< numero di passi
    Eigen::VectorXd qp_solution;      ///< soluzione raw [x1,y1, x2,y2, ..., xF,yF]
    double delta_theta;               ///< angolo asse trasporto nel frame F [rad]
    Eigen::Vector2d delta_p;          ///< spostamento nominale PID [m, frame F]
    labrob::Foot support_foot_start;  ///< piede di supporto al passo 0
    double lsole_z;                   ///< z misurata piede sx al momento della chiamata
    double rsole_z;                   ///< z misurata piede dx al momento della chiamata

    /**
    * Restituisce la posizione 2D del piede oscillante al passo j (0-indexed).
    */
    Eigen::Vector2d getFootPos2D(int j) const {
        return qp_solution.segment<2>(2*j);
    }

    /**
    * Restituisce il piede oscillante al passo j.
    * Il supporto alterna a partire da support_foot_start.
    */
    labrob::Foot getSwingFoot(int j) const {
        labrob::Foot sf = support_foot_start;
        for (int i = 0; i < j; ++i)
            sf = (sf == labrob::Foot::LEFT) ? labrob::Foot::RIGHT : labrob::Foot::LEFT;
        return (sf == labrob::Foot::LEFT) ? labrob::Foot::RIGHT : labrob::Foot::LEFT;
    }
  };

  /**
   * Inizializza il planner con le pose iniziali dei piedi e il piede
   * di supporto corrente. Da chiamare in WalkingManager::init().
   */
  void init(
      const labrob::SE3& T_lsole,
      const labrob::SE3& T_rsole,
      labrob::Foot support_foot
  );

  /**
  * Update integral of the error at every tick
  */
  Eigen::Vector2d updateErrorIntegral(
    const Eigen::Vector2d& e_h,
    double dt
  );

  /**
   * Chiamata da WalkingManager a ogni pop_front di WalkingData.
   *
   * Ancora il piede di supporto alla posa misurata, calcola Delta_p e
   * Delta_theta, risolve il QP e restituisce 2*F nuovi FootstepPlanElement
   * (coppie DS+SS).
   *
   * Il piede di supporto viene ancorato alla posa misurata a ogni chiamata,
   * garantendo robustezza rispetto agli errori di esecuzione del passo.
   *
   * @param T_lsole    Posa misurata piede sinistro (da FK / MuJoCo)
   * @param T_rsole    Posa misurata piede destro (da FK / MuJoCo)
   * @param e_h        Errore medio mani [frame F, xy] da HAC::getEh()
   * @param e_h_dot    Derivata di e_h [frame F, xy] da HAC::getEhDot()
   * @param r_l_F_xy   Proiezione a terra mano sinistra nel frame F
   * @param r_r_F_xy   Proiezione a terra mano destra nel frame F
   * @param R_F        Orientazione del frame F rispetto al world frame
   * @param dt         Timestep del controllore [s], per aggiornare l'integrale
   * @return           Vettore di 2*F FootstepPlanElement da push_back nel deque
   */
    QP2DResult computeNextSteps(
      const labrob::SE3& T_lsole,
      const labrob::SE3& T_rsole,
      labrob::Foot support_foot,
      const Eigen::Vector2d& e_h,
      const Eigen::Vector2d& e_h_dot,
      const Eigen::Vector2d& r_l_F_xy,
      const Eigen::Vector2d& r_r_F_xy,
      const Eigen::Matrix3d& R_F,
      double dt
    );

    /**
    * Converte QP2DResult in FootstepPlanElement per inserimento nella deque.
    * Modifica lo stato interno (T_lsole_planned_, T_rsole_planned_, support_foot_).
    */
    std::vector<labrob::FootstepPlanElement> buildDequeElements(
        const QP2DResult& result
    );

  /**
   * Reset dell'integrale del PID.
   * Da chiamare alla transizione Standing -> Starting.
   */
  void resetIntegral();

  Eigen::Vector2d getLastDeltaP() const { return last_delta_p_; }
  double getLastDeltaTheta() const { return delta_theta_prev_; }

  

 private:

  Params params_;

  // --- Stato interno ---

  labrob::Foot    support_foot_;       ///< piede di supporto corrente
  labrob::SE3     T_lsole_planned_;    ///< ultima posa pianificata piede sx
  labrob::SE3     T_rsole_planned_;    ///< ultima posa pianificata piede dx

  Eigen::Vector2d integral_eh_;        ///< integrale cumulativo di e_h [frame F]
  double          delta_theta_prev_;   ///< ultimo Delta_theta valido (per dead-band)

  Eigen::Vector2d last_delta_p_ = Eigen::Vector2d::Zero();       ///< ultimo delta_p calcolato (per debug)

  // --- QP ---
  // Variabili: x = [x_f^1, y_f^1, ..., x_f^F, y_f^F]  (dim 2F)
  // Vincoli:   2F di disuguaglianza cinematica (2 righe per passo)
  //            0  di uguaglianza

  int num_variables_;        ///< = 2*F
  int num_ineq_constraints_; ///< = 2*F

  Eigen::MatrixXd H_;          ///< hessiana del costo (2F x 2F), costante
  Eigen::VectorXd f_;          ///< gradiente lineare del costo (2F), aggiornato a ogni chiamata
  Eigen::MatrixXd A_kin_;      ///< matrice vincoli cinematici (2F x 2F), aggiornata a ogni chiamata
  Eigen::VectorXd b_kin_min_;  ///< bound inferiore vincoli (2F)
  Eigen::VectorXd b_kin_max_;  ///< bound superiore vincoli (2F)
  Eigen::MatrixXd A_eq_;       ///< matrice vuota (0 x 2F), richiesta dall'interfaccia solver
  Eigen::VectorXd b_eq_;       ///< vettore vuoto (0), richiesto dall'interfaccia solver

  std::shared_ptr<labrob::QpSolver> qp_solver_ptr_;

  // --- Metodi privati ---

  /**
   * Calcola Delta_p dal PID su e_h e aggiorna l'integrale cumulativo.
   */
  Eigen::Vector2d computeDeltaP(
      const Eigen::Vector2d& e_h,
      const Eigen::Vector2d& e_h_dot,
      const Eigen::Vector2d& integral_eh
  );

  /**
   * Calcola Delta_theta come orientazione dell'asse di trasporto nel frame F.
   * Gestisce il caso degenere (proiezioni coincidenti) e la dead-band.
   */
  double computeDeltaTheta(
      const Eigen::Vector2d& r_l_F_xy,
      const Eigen::Vector2d& r_r_F_xy
  );

  /**
   * Costruisce H_ = D^T D (chiamata una sola volta nel costruttore).
   */
  void buildCostMatrix();

  /**
   * Aggiorna f_, A_kin_, b_kin_min_, b_kin_max_ con i valori correnti.
   * p0 viene letto da T_lsole_planned_ o T_rsole_planned_ a seconda
   * di support_foot_ (già ancorato alla misura da computeNextSteps).
   */
  void updateQPMatrices(
      const Eigen::Vector2d& delta_p,
      double delta_theta,
      const Eigen::Matrix3d& R_F
  );

  /**
   * Converte la soluzione del QP (2F valori) in una sequenza di
   * FootstepPlanElement (DS+SS alternati). Aggiorna lo stato interno
   * (T_lsole_planned_, T_rsole_planned_, support_foot_).
   */
  std::vector<labrob::FootstepPlanElement> buildPlan(
      const Eigen::VectorXd& qp_solution,
      double delta_theta,
      double lsole_z,
      double rsole_z
  );

}; // end class FootstepPlannerCoop

} // end namespace labrob

#endif // LABROB_FOOTSTEP_PLANNER_COOP_HPP